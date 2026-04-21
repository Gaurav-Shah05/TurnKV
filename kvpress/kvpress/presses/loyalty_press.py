# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.turn_aware_base import TurnAwareMixin
from kvpress.utils import get_prerope_query_states


@dataclass(kw_only=True)
class LoyaltyPress(TurnAwareMixin):
    """
    Policy C (ADR 001 §3-C) -- per-KV-position loyalty counter.

    Rewards past-turn tokens that the model keeps attending to when later
    turns are processed. The harness (via ``TurnAwareGlobalPress``'s
    forward hook, ADR 002 §2) calls ``update_loyalty`` on every attention
    layer during both the **prefill** and **decode** of each current turn.
    For every current-turn query row, we identify the top ``top_p`` (25%
    by default) of its attention distribution over **past-turn** keys and
    increment those positions' counters. Tokens inside the current turn's
    own span are ineligible for loyalty accrual on this call; they only
    become eligible starting from turn ``k+1`` (ADR 001 §3-C).

    Attention is recomputed internally from ``hidden_states`` and ``keys``
    following the ``SnapKVPress.compute_window_attention`` pattern
    (``snapkv_press.py:42``): pre-RoPE query states via
    ``get_prerope_query_states``, RoPE via ``kwargs["position_embeddings"]``,
    GQA broadcast via ``repeat_kv``, then ``softmax(QKᵀ/√d)`` with no
    causal mask (past keys all precede current queries by construction).
    This path never requests ``output_attentions=True``, keeping the press
    flash-attention-2 compatible (ADR 002 §2).

    Per-layer contributions sum: every attention layer's hook call votes
    independently, so a position attended to in many layers accumulates
    proportionally. ADR 001 §3-C phrases eligibility as "top 25% of a
    query's distribution"; we interpret "a query's distribution" as the
    **head-mean** per-query distribution (reduce over heads, then top-k
    per query row). This stays faithful to the per-query granularity
    while matching SnapKV's head reduction (``snapkv_press.py:95-100``)
    so a single head's outlier attention can't dominate the vote.

    The class inherits ``TurnAwareMixin``; it does not evict, install
    forward hooks, or call the base scorer. It only maintains state and
    produces a shape-``(kv_len,)`` normalised weight tensor consumed by
    ``TurnAwareGlobalPress``. The mixin's advertised ``loyalty: dict[int,
    int]`` field is the canonical store.

    Parameters
    ----------
    top_p : float, default=0.25
        Fraction of past keys per query that count as "attended to". Must
        satisfy ``0 < top_p <= 1``. ADR 001 §3-C fixes this at 25%.

    Notes
    -----
    Performance: the loyalty dict is Python-native. For LongMemEval (~105K
    past_end × 490 turns × 32 layers) a tensor-backed store would be
    faster; Week-1 scope is correctness, Week-2 may profile and optimise.
    Global compression: ADR 001 §0 rewrites cache positions on eviction;
    a post-compression remap callback is a Week-2 concern -- ``loyalty``
    keys are absolute positions at accumulation time and must be shifted
    alongside keys/values by whatever index tensor the composer applied.
    """

    top_p: float = 0.25

    # Start-of-current-turn cache offset, set in on_turn_start. Everything
    # at cache positions < _current_turn_start_kv is "past-turn" and
    # eligible for loyalty accumulation on the current forward pass.
    _current_turn_start_kv: int = field(default=0, init=False)

    def __post_init__(self):
        assert 0 < self.top_p <= 1, f"top_p must be in (0, 1], got {self.top_p}"

    def on_turn_start(self, turn_idx: int, role: str, start_kv: int) -> None:
        super().on_turn_start(turn_idx, role, start_kv)
        self._current_turn_start_kv = start_kv

    def reset_turn_state(self) -> None:
        super().reset_turn_state()
        self._current_turn_start_kv = 0

    def update_loyalty(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict,
        query_turn_idx: int,
    ) -> None:
        """
        Increment loyalty counters for past-turn positions in the top-p
        of each current-turn query's recomputed attention distribution.
        See class docstring for the recipe and the ADR 001 §3-C semantics.
        """
        k_len = keys.shape[2]
        q_len = hidden_states.shape[1]
        if q_len == 0:
            return

        # query_turn_idx is advertised in ADR 001 §4 for the caller to
        # declare which turn the query rows belong to; the current-turn
        # ineligibility is enforced via past_end, which must match. A
        # mismatch means the composer's bookkeeping and ours have
        # diverged -- fail loudly rather than silently undercount.
        assert query_turn_idx == self.current_turn, (
            f"query_turn_idx={query_turn_idx} disagrees with current_turn={self.current_turn}; "
            "on_turn_start was not called for the turn the composer thinks is active."
        )

        past_end = min(self._current_turn_start_kv, k_len)
        if past_end <= 0:
            return

        # past_end + q_len must not exceed k_len: current-turn queries sit
        # at cache positions [k_len - q_len, k_len), and past_end must be
        # <= k_len - q_len so the past/current split does not overlap.
        # Violations mean _current_turn_start_kv is stale (on_turn_start
        # was skipped or fired with a smaller start_kv than the harness
        # actually appended) and we would mis-count current-turn tokens
        # as past-turn.
        assert past_end + q_len <= k_len, (
            f"past_end ({past_end}) + q_len ({q_len}) > k_len ({k_len}); stale _current_turn_start_kv."
        )

        bsz, num_kv_heads, _, head_dim = keys.shape
        num_heads = module.config.num_attention_heads
        num_kv_groups = num_heads // num_kv_heads

        # Broadcast KV heads across GQA groups (matches snapkv_press.py:61)
        past_keys = repeat_kv(keys[:, :, :past_end, :], num_kv_groups)

        # Pre-RoPE queries then apply RoPE using the current forward pass's
        # position embeddings (matches snapkv_press.py:54-58 but over all
        # queries rather than a trailing window).
        q = get_prerope_query_states(module, hidden_states)

        cos, sin = kwargs["position_embeddings"]
        # Contract: cos/sin are either aligned to the layer's hidden_states
        # (shape (bsz, q_len, head_dim), same q_len) or to the full cache
        # (shape (bsz, k_len, head_dim)). In both cases the trailing q_len
        # rows correspond to the current queries' absolute positions, so
        # the slice below is correct. Assert the lower bound explicitly
        # so a surprising caller (e.g. decode + buffered hidden_states)
        # fails fast instead of silently RoPE-ing wrong positions.
        assert cos.shape[1] >= q_len and sin.shape[1] >= q_len, (
            f"position_embeddings too short: cos.shape={tuple(cos.shape)}, q_len={q_len}"
        )
        cos = cos[:, -q_len:]
        sin = sin[:, -q_len:]
        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))

        # Scaled dot-product attention against past-only keys. No causal
        # mask needed: past keys strictly precede current queries.
        scores = torch.matmul(q, past_keys.transpose(2, 3)) / math.sqrt(head_dim)
        attn = torch.softmax(scores, dim=-1, dtype=torch.float32)
        attn = attn.mean(dim=1)  # head-mean: (bsz, q_len, past_end)

        n_top = max(1, int(past_end * self.top_p))
        top_idx = attn.topk(n_top, dim=-1).indices  # (bsz, q_len, n_top)

        # Vote per position: bincount over the flattened top-k indices
        # yields the number of (head-mean query row) votes for each past
        # position. Iterate only non-zero entries into the dict.
        for b in range(bsz):
            counts = torch.bincount(top_idx[b].flatten(), minlength=past_end)
            nonzero = torch.nonzero(counts, as_tuple=False).flatten()
            if nonzero.numel() == 0:
                continue
            positions = nonzero.cpu().tolist()
            votes = counts[nonzero].cpu().tolist()
            for pos, v in zip(positions, votes):
                self.loyalty[pos] = self.loyalty.get(pos, 0) + v

    def compute_weights(
        self,
        kv_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return ``loyalty(t) / max_loyalty`` per ADR 001 §3 (weight term for
        policy C). Returned shape ``(kv_len,)``; positions with no
        accumulated loyalty get 0.0.
        """
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.zeros(kv_len, device=device, dtype=dtype)

        if not self.loyalty:
            return weights

        max_loyalty = max(self.loyalty.values())
        if max_loyalty <= 0:
            return weights

        for pos, count in self.loyalty.items():
            if 0 <= pos < kv_len and count > 0:
                weights[pos] = count / max_loyalty

        return weights
