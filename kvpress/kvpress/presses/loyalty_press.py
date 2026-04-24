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

    Storage: the per-position counter is a ``torch.int32`` tensor kept on
    the same device as the KV cache. ``update_loyalty`` performs a single
    GPU ``scatter_add_`` per hook call -- no Python loop over top-k
    positions, no GPU->CPU sync in the hot path. This replaces Week-1's
    ``dict[int, int]`` implementation and removes the ~2x slowdown that
    pinned the Week-1 deferred-perf item.

    The mixin's legacy ``loyalty: dict[int, int]`` field is preserved for
    type-compatibility but is NEVER written to by this class; use
    :meth:`loyalty_as_dict` when you need a dict view for debugging or
    tests. Other policies that inspect ``self.loyalty`` (there are none
    currently) will see the empty inherited dict, which is the correct
    "no loyalty data stored here" signal.

    Parameters
    ----------
    top_p : float, default=0.25
        Fraction of past keys per query that count as "attended to". Must
        satisfy ``0 < top_p <= 1``. ADR 001 §3-C fixes this at 25%.
    update_every : int, default=5
        Decode-step subsampling factor. ``update_loyalty`` runs every
        Nth call PER LAYER during decode (``q_len == 1``); prefill
        (``q_len > 1``) always runs since it is high-signal-per-call.
        ``update_every=1`` disables subsampling. Higher N reduces the
        attention-recompute overhead inside ``update_loyalty`` linearly
        in 1/N at the cost of higher variance in counts (rankings
        preserved in expectation since sampling is uniform). Empirically
        N=5 brings turnkv_snapkv close to plain snapkv per-iter
        wall-clock without changing relative loyalty rankings on
        ConvCodeWorld scale.

    Notes
    -----
    Global compression: ADR 001 §0 rewrites cache positions on eviction;
    the tensor counter stays indexed by absolute position at accumulation
    time. The composer's :meth:`run_global_compression` must call
    :meth:`remap_loyalty_after_compression` with the gathered index
    tensor so surviving counts move to their post-compression positions.
    Not yet implemented at the Week-1 level; relevant once compression
    fires mid-turn in real runs.
    """

    top_p: float = 0.25
    update_every: int = 5

    # Start-of-current-turn cache offset, set in on_turn_start. Everything
    # at cache positions < _current_turn_start_kv is "past-turn" and
    # eligible for loyalty accumulation on the current forward pass.
    _current_turn_start_kv: int = field(default=0, init=False)

    # int32 counter, shape (capacity,), on the KV-cache device. Created
    # lazily on first update_loyalty call. Grown on demand if the cache
    # outruns current capacity (rare after the first few turns).
    _loyalty_counts: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    # Per-layer counter of decode-step calls. Used to subsample at rate
    # 1/update_every during decode (q_len==1). Reset between sessions.
    _decode_step_counter: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert 0 < self.top_p <= 1, f"top_p must be in (0, 1], got {self.top_p}"
        assert self.update_every >= 1, f"update_every must be >= 1, got {self.update_every}"

    def on_turn_start(self, turn_idx: int, role: str, start_kv: int) -> None:
        super().on_turn_start(turn_idx, role, start_kv)
        self._current_turn_start_kv = start_kv

    def reset_turn_state(self) -> None:
        super().reset_turn_state()
        self._current_turn_start_kv = 0
        self._loyalty_counts = None
        self._decode_step_counter = {}

    def _ensure_counts_capacity(self, min_size: int, device: torch.device) -> None:
        """Allocate / grow ``_loyalty_counts`` to fit at least ``min_size``
        positions, preserving accumulated counts when the tensor grows.
        Capacity rounds up to a power of two with a 128-token floor, so
        routine cache growth over a conversation does not trigger a
        realloc on every turn boundary."""
        if self._loyalty_counts is not None and self._loyalty_counts.shape[0] >= min_size:
            # Make sure we're on the right device (KV cache may move).
            if self._loyalty_counts.device != device:
                self._loyalty_counts = self._loyalty_counts.to(device)
            return

        new_cap = max(128, 1 << max(0, (min_size - 1).bit_length()))  # next pow2 >= min_size, min 128
        new_tensor = torch.zeros(new_cap, dtype=torch.int32, device=device)
        if self._loyalty_counts is not None:
            old = self._loyalty_counts.to(device)
            new_tensor[: old.shape[0]] = old
        self._loyalty_counts = new_tensor

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

        # Decode-step subsampling: only every Nth call per layer during
        # decode (q_len == 1). Prefill is always processed -- it's the
        # high-signal-per-call case (multi-token forward in one shot)
        # and skipping it would bias the loyalty estimator toward
        # decode-only attention patterns.
        if q_len == 1 and self.update_every > 1:
            layer_idx = getattr(module, "layer_idx", id(module))
            step = self._decode_step_counter.get(layer_idx, 0)
            self._decode_step_counter[layer_idx] = step + 1
            if step % self.update_every != 0:
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

        # Ensure counter is on the same device as the cache and sized for
        # at least past_end positions. Grow (preserving) on first hit.
        self._ensure_counts_capacity(past_end, device=keys.device)

        # Scatter-add 1 for every (batch, query, top-k) tuple. The counter
        # is per-position (not per-batch) since the conversation state is
        # shared across batch rows in this harness. If batched multi-turn
        # is ever added, promote _loyalty_counts to (bsz, cap) and
        # scatter_add_ along dim=-1 per batch row.
        flat = top_idx.reshape(-1).to(torch.int64)
        if flat.numel() > 0:
            self._loyalty_counts.scatter_add_(
                0, flat, torch.ones_like(flat, dtype=self._loyalty_counts.dtype)
            )

    def compute_weights(
        self,
        kv_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return ``loyalty(t) / max_loyalty`` per ADR 001 §3 (weight term for
        policy C). Returned shape ``(kv_len,)``; positions with no
        accumulated loyalty get 0.0. The counter tensor is truncated or
        zero-padded to ``kv_len`` before normalisation.
        """
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.zeros(kv_len, device=device, dtype=dtype)

        if self._loyalty_counts is None:
            return weights

        counts = self._loyalty_counts.to(device=device, dtype=torch.float32)
        if counts.shape[0] < kv_len:
            padded = torch.zeros(kv_len, device=device, dtype=torch.float32)
            padded[: counts.shape[0]] = counts
            counts = padded
        else:
            counts = counts[:kv_len]

        max_loyalty = counts.max()
        if max_loyalty.item() <= 0:
            return weights
        return (counts / max_loyalty).to(dtype=dtype)

    def loyalty_as_dict(self) -> dict[int, int]:
        """Materialise the counter tensor as a ``{position: count}`` dict,
        skipping zeros. For debugging and tests only -- the hot path uses
        the tensor directly; this call incurs a GPU->CPU sync and a Python
        loop over non-zero entries."""
        if self._loyalty_counts is None:
            return {}
        counts = self._loyalty_counts
        nonzero = torch.nonzero(counts, as_tuple=False).flatten()
        if nonzero.numel() == 0:
            return {}
        nz_cpu = nonzero.cpu().tolist()
        values = counts[nonzero].cpu().tolist()
        return {int(p): int(v) for p, v in zip(nz_cpu, values)}
