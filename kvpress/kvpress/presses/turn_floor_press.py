# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch

from kvpress.presses.turn_aware_base import TurnAwareMixin


@dataclass(kw_only=True)
class TurnFloorPress(TurnAwareMixin):
    """
    Policy A (ADR 001 §3-A) -- per-turn reserved floor.

    Allocates a minimum-survival budget for each past turn so early turns do
    not get fully evicted under aggressive global compression. The floor size
    grows with the turn's share of total chat length and shrinks with an
    exponential decay ``gamma`` for older turns::

        floor_k = max(min_floor_tokens,
                      alpha_floor_len * global_budget * |T_k| / Σ_j |T_j|)
                  * gamma ** max(0, current_turn - k)

    The computed ``floor_k`` is clamped to ``|T_k|`` (can't reserve more
    tokens than exist in the turn) and then rounded to an integer. Turns
    shorter than ``exempt_shorter_than`` (single-sentence acknowledgements
    like "ok", "thanks") receive no allocation.

    ADR 001 §3 expresses the weight as a binary indicator
    ``1[t ∈ reserved floor of its turn]``. Because ``compute_weights`` runs
    before the base scorer, we deterministically reserve the **last**
    ``floor_k`` positions of each turn (concatenating a turn's user+assistant
    spans in cache order). A consequence: for a turn with both user and
    assistant spans, the reserved positions are drawn from the end of the
    assistant span first and only fall into the user span when
    ``floor_k > |assistant_span|``. Policy B (role-boundary anchor)
    independently covers user-turn heads, so under the composer's additive
    weighting ``base × (1 + Σ αᵢ · wᵢ)`` the union of A and B protects both
    intent (user head, via B) and summary (turn tail, via A) without
    double-eviction risk.

    Context spans (``turn_idx == 0`` or ``role == "context"``) are treated as
    KEEP bucket (ADR 001 §0) and excluded from both ``|T_k|`` and
    ``Σ_j |T_j|`` so they do not dilute the per-turn floors. Under very
    old turns (many decay steps), ``floor_k`` may round down to 0 and the
    turn receives no allocation that call -- this matches the ADR's
    geometric decay intent; "never vanish" in §3-A is an ordering property,
    not a hard integer invariant.

    This class inherits ``TurnAwareMixin`` only; it does not evict, does
    not register forward hooks, and does not call the base scorer. It only
    produces a shape-``(kv_len,)`` weight tensor for the composer.

    Parameters
    ----------
    global_budget : int (keyword-only, required)
        Total KV budget ``G`` for the benchmark (ADR 001 §2). Required so
        callers cannot accidentally construct a press that silently ignores
        the length term.
    alpha_floor_len : float, default=0.3
        Length-proportional coefficient from ADR 001 §3-A.
    gamma : float, default=0.9
        Per-turn exponential decay. Must satisfy ``0 < gamma <= 1``.
    min_floor_tokens : int, default=5
        Hard minimum ``c`` from ADR 001 §3-A. Applied before decay.
    exempt_shorter_than : int, default=10
        Turns with total length below this threshold receive no floor
        (ADR 001 §3-A: "single-sentence acknowledgements are exempt").
    """

    global_budget: int
    alpha_floor_len: float = 0.3
    gamma: float = 0.9
    min_floor_tokens: int = 5
    exempt_shorter_than: int = 10

    def __post_init__(self):
        assert self.global_budget >= 0, f"global_budget must be non-negative, got {self.global_budget}"
        assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"
        assert 0 <= self.alpha_floor_len, f"alpha_floor_len must be non-negative, got {self.alpha_floor_len}"
        assert self.min_floor_tokens >= 0, f"min_floor_tokens must be non-negative, got {self.min_floor_tokens}"
        assert self.exempt_shorter_than >= 0, f"exempt_shorter_than must be non-negative, got {self.exempt_shorter_than}"

    @staticmethod
    def _is_context(boundary) -> bool:
        # ADR 001 §0: static context is KEEP bucket, never evicted by the composer,
        # and therefore excluded from per-turn aggregates.
        return boundary.turn_idx == 0 or boundary.role == "context"

    def compute_weights(
        self,
        kv_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Return a shape-``(kv_len,)`` binary tensor with 1.0 at reserved-floor
        positions and 0.0 elsewhere. See class docstring for the selection rule.
        """
        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32
        weights = torch.zeros(kv_len, device=device, dtype=dtype)

        if not self.turn_boundaries or kv_len == 0:
            return weights

        # Aggregate spans by turn_idx: a single "turn" may emit multiple
        # boundaries (user + assistant), so we group by turn_idx per ADR 001 §4.
        # Context (turn_idx=0 / role="context") is KEEP bucket and excluded.
        turn_length: dict[int, int] = defaultdict(int)
        turn_spans: dict[int, list] = defaultdict(list)
        for b in self.turn_boundaries:
            if self._is_context(b):
                continue
            turn_length[b.turn_idx] += len(b)
            turn_spans[b.turn_idx].append(b)

        total_length = sum(turn_length.values())
        if total_length == 0:
            return weights

        for k, length_k in turn_length.items():
            if length_k < self.exempt_shorter_than:
                continue

            decay = self.gamma ** max(0, self.current_turn - k)
            scaled = self.alpha_floor_len * self.global_budget * length_k / total_length
            floor_k_raw = max(self.min_floor_tokens, scaled) * decay
            # math.floor -- deterministic across Python versions, conservative:
            # total reserved tokens never exceed alpha_floor_len * global_budget
            # in the un-clamped regime. Python's round() uses banker's rounding.
            floor_k = min(math.floor(floor_k_raw), length_k)
            if floor_k <= 0:
                continue

            # Concatenate positions of this turn in cache order (start_kv ascending)
            positions: list[int] = []
            for b in sorted(turn_spans[k], key=lambda x: x.start_kv):
                positions.extend(range(b.start_kv, b.end_kv))

            # Reserve the last floor_k positions; clamp to kv_len for safety
            # (kv_len may be smaller than the recorded spans after global
            # compression has rewritten the cache).
            reserved = [p for p in positions[-floor_k:] if 0 <= p < kv_len]
            if reserved:
                weights[torch.tensor(reserved, device=device, dtype=torch.long)] = 1.0

        return weights
