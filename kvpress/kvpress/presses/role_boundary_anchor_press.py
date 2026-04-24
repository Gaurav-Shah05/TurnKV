# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass
from typing import Optional

import torch

from kvpress.presses.turn_aware_base import TurnAwareMixin


@dataclass(kw_only=True)
class RoleBoundaryAnchorPress(TurnAwareMixin):
    """
    Policy B (ADR 001 §3-B) -- role-boundary anchor.

    Protects tokens at each turn's edges so the intent-bearing head of a
    user utterance and the content-bearing tail of every turn survive
    aggressive global compression. For each span::

        w = max(min_anchor_tokens, floor(beta * |span|))

    and the reserved positions are:

    - ``role == "user"``: first ``w`` and last ``w`` positions (intent head +
      full-request tail).
    - ``role == "assistant"``: last ``w`` positions only (openings tend to
      be boilerplate).
    - ``role == "context"`` (turn_idx 0): skipped -- context is the KEEP
      bucket per ADR 001 §0 and must not be in the compression surface.
    - ``role == "feedback"`` is accepted by ``TurnBoundary`` for forward
      compatibility but is never emitted by the current ConvCodeWorld
      driver (feedback text lives inside the next iteration's user prompt
      rather than in its own span), so this policy does not special-case
      it. Re-introduce an assistant-like branch if a future loader begins
      emitting role="feedback".

    This is a per-span operation, not per-turn: because the harness emits a
    separate boundary for each role within a turn, a user+assistant turn
    gets two independent anchor decisions. ``|span|`` in the formula is the
    span length, not the aggregate turn length -- the aggregate would
    conflate heads of opposite meanings (user first-w protects intent;
    assistant first-w protects boilerplate).

    For very short spans where ``2 * w > |span|``, the "first w" and "last w"
    windows overlap and their union is the full span. In that degenerate
    case all span positions get weight 1.0 for user turns; for assistant
    only the last ``w`` are set (even if that covers the whole span). Tests
    should use spans of length ``>= 2 * min_anchor_tokens`` to unambiguously
    exercise "first w == 0" behaviour.

    This class inherits ``TurnAwareMixin`` only; it does not evict, register
    hooks, or call the base scorer. It produces a shape-``(kv_len,)`` binary
    weight tensor consumed by ``TurnAwareGlobalPress``.

    Parameters
    ----------
    beta : float, default=0.15
        Length-proportional coefficient ``β`` from ADR 001 §3-B.
    min_anchor_tokens : int, default=3
        Hard minimum ``w`` -- single-token spans still get ``min(3, |span|)``
        positions reserved so very short turns are not silently discarded.
    """

    beta: float = 0.15
    min_anchor_tokens: int = 3

    def __post_init__(self):
        assert 0 <= self.beta <= 1, f"beta must be in [0, 1], got {self.beta}"
        assert self.min_anchor_tokens >= 0, f"min_anchor_tokens must be non-negative, got {self.min_anchor_tokens}"

    def compute_weights(
        self,
        kv_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return a shape-``(kv_len,)`` binary tensor with 1.0 at anchor positions.
        """
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.zeros(kv_len, device=device, dtype=dtype)

        if not self.turn_boundaries or kv_len == 0:
            return weights

        for b in self.turn_boundaries:
            if b.turn_idx == 0 or b.role == "context":
                continue

            length = len(b)
            if length == 0:
                continue

            # math.floor matches turn_floor_press's rounding (deterministic,
            # conservative); final min(..., length) clamps w to span length so
            # min_anchor_tokens on a shorter span doesn't over-reserve.
            w = min(max(self.min_anchor_tokens, math.floor(self.beta * length)), length)
            if w <= 0:
                continue

            # Clip the span to the current cache extent; positions outside
            # [0, kv_len) may exist if global compression has rewritten the
            # cache since the boundary was recorded.
            span_start = max(b.start_kv, 0)
            span_end = min(b.end_kv, kv_len)
            if span_end <= span_start:
                continue

            if b.role == "user":
                head_end = min(span_start + w, span_end)
                tail_start = max(span_end - w, span_start)
                weights[span_start:head_end] = 1.0
                weights[tail_start:span_end] = 1.0
            elif b.role == "assistant":
                tail_start = max(span_end - w, span_start)
                weights[tail_start:span_end] = 1.0
            # role=="feedback" is never emitted by live_loop.py's static-replay
            # or live-loop driver -- feedback text is embedded in the next
            # iteration's user prompt rather than allocated its own span --
            # so there is no branch here. If a future loader starts emitting
            # role="feedback", re-introduce assistant-like tail handling.

        return weights
