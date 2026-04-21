# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn

VALID_ROLES = ("context", "user", "assistant", "feedback")


@dataclass
class TurnBoundary:
    """
    A half-open span ``[start_kv, end_kv)`` in the KV cache covering one
    turn's tokens. The harness (``multi_turn_evaluate.py``) constructs these
    and hands them to turn-aware presses through ``TurnAwareMixin`` callbacks
    (ADR 001 §4). Tokens are addressed by their position inside the KV cache,
    so boundaries remain valid after global compression only if the cache has
    not been rewritten since they were recorded.

    Parameters
    ----------
    turn_idx : int
        0 = static context; 1..N = conversational turns.
    start_kv : int
        Inclusive KV cache position where this span begins.
    end_kv : int
        Exclusive KV cache position where this span ends.
    role : str
        One of ``{"context", "user", "assistant", "feedback"}``.
    """

    turn_idx: int
    start_kv: int
    end_kv: int
    role: str

    def __post_init__(self):
        assert self.end_kv >= self.start_kv, (
            f"TurnBoundary end_kv ({self.end_kv}) must be >= start_kv ({self.start_kv})"
        )
        assert self.start_kv >= 0, f"start_kv must be non-negative, got {self.start_kv}"
        assert self.role in VALID_ROLES, f"Unknown role: {self.role!r}; expected one of {VALID_ROLES}"
        assert self.turn_idx >= 0, f"turn_idx must be non-negative, got {self.turn_idx}"

    def __len__(self) -> int:
        return self.end_kv - self.start_kv


@dataclass
class TurnAwareMixin:
    """
    Mixin providing turn-aware state and default callbacks for Week-1 policies.

    Subclasses override ``compute_weights`` to produce per-KV-position weights
    that the composer (``TurnAwareGlobalPress``, forthcoming) multiplies into
    the base scorer at global-compression time (ADR 001 §3). The mixin itself
    does not evict, does not install hooks, and does not carry tensors
    across instances -- all state is plain Python. Harness calls the
    callbacks in the order fixed by ADR 001 §4.

    Parameters
    ----------
    turn_boundaries : list[TurnBoundary]
        Ordered list of turn spans populated by ``on_turn_end``. turn_idx 0
        is the static context (if any); indices 1..N are conversational turns.
        The harness may emit multiple spans sharing the same ``turn_idx``
        (one for each role within the turn -- e.g. a user span then an
        assistant span per ADR 001 §4). Policies that need per-turn
        aggregates must group-by ``turn_idx``; they must not assume
        ``turn_boundaries[k].turn_idx == k``.
    loyalty : dict[int, int]
        Per-KV-position loyalty count; only populated by ``LoyaltyPress``.
        Keyed by absolute KV position at the time of accumulation. If global
        compression shifts positions, ``LoyaltyPress`` is responsible for
        remapping the dict along the same indices applied to keys/values.
    current_turn : int
        Index of the turn currently being prefilled or generated, set by
        ``on_turn_start`` and read by policy A's decay term and by policy C's
        self-skip guard.
    """

    turn_boundaries: list[TurnBoundary] = field(default_factory=list)
    loyalty: dict[int, int] = field(default_factory=dict)
    current_turn: int = 0

    def on_turn_start(self, turn_idx: int, role: str, start_kv: int) -> None:
        """Harness calls this immediately before a turn's prefill begins."""
        self.current_turn = turn_idx

    def on_turn_end(self, turn_idx: int, role: str, start_kv: int, end_kv: int) -> None:
        """Harness calls this after a turn's tokens are settled in the cache."""
        self.turn_boundaries.append(
            TurnBoundary(turn_idx=turn_idx, start_kv=start_kv, end_kv=end_kv, role=role)
        )

    def update_loyalty(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        kwargs: dict,
        query_turn_idx: int,
    ) -> None:
        """
        Default no-op. ``LoyaltyPress`` overrides this; other policies ignore it.

        Refines the abstract ``update_loyalty(query_attentions, query_turn_idx)``
        shown in ADR 001 §4: to stay flash-attention-2-compatible, we never
        request ``output_attentions=True``. Subclasses recompute attention
        internally from ``hidden_states`` and ``keys`` (with RoPE via
        ``module.rotary_emb``) following the ``SnapKVPress.compute_window_attention``
        pattern (``snapkv_press.py:42``), and apply the top-25%-per-query
        loyalty update in ADR 001 §3-C.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer for this update (provides
            ``module.head_dim``, ``module.config``, ``module.rotary_emb``).
        hidden_states : torch.Tensor
            Layer input, shape ``(batch, q_len, hidden_dim)``.
        keys : torch.Tensor
            Full cache keys post-RoPE, shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        kwargs : dict
            Attention-layer kwargs; ``position_embeddings`` (``cos``, ``sin``)
            live here.
        query_turn_idx : int
            Turn index of the query tokens. Positions inside this turn's
            span(s) are ineligible for loyalty accumulation (ADR 001 §3-C).
        """

    def compute_weights(
        self,
        kv_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Default identity weights. Subclasses override to inject policy signal.

        Contract:

        - ``kv_len`` must equal ``keys.shape[2]`` at the composer's call site.
          Returned shape ``(kv_len,)`` broadcasts against the base scorer's
          ``(batch, num_kv_heads, kv_len)`` output by right-alignment, which is
          identical across heads (policies A/B/C are position-based).
        - ``device`` and ``dtype`` default to CPU / float32. Callers operating
          on CUDA/bf16 base scores *must* pass ``device=keys.device,
          dtype=torch.float32`` (weights are kept at fp32 so ``α·w`` does not
          underflow; the composer casts to the scorer dtype before multiplying).
          The CPU/fp32 default exists only for direct unit-test construction.
        - Returned entries are non-negative by convention; per-policy
          normalisation and blending with αs is the composer's responsibility,
          not the mixin's.
        - When all αs are 0, the composer must short-circuit to the raw base
          scorer to preserve bit-identical ``topk`` evictions (ADR 002 §7
          ``test_all_alphas_zero_equivalent_to_base``); this method's ones
          output is not sufficient on its own for that guarantee.

        Returns
        -------
        torch.Tensor
            Shape ``(kv_len,)`` on the chosen device/dtype.
        """
        return torch.ones(
            kv_len,
            device=device if device is not None else torch.device("cpu"),
            dtype=dtype if dtype is not None else torch.float32,
        )

    def reset_turn_state(self) -> None:
        """Clear per-conversation state. Harness calls between sessions/probes."""
        self.turn_boundaries = []
        self.loyalty = {}
        self.current_turn = 0
