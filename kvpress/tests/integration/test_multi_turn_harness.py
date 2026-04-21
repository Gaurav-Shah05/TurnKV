# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Integration test for the Week-1 press primitives (ADR 002 §2 row for
``test_multi_turn_harness.py``). Replays a 10-turn fake session against the
``unit_test_model`` fixture and asserts:

- callbacks fire in order (``on_turn_start`` then ``on_turn_end`` per turn),
- cache length stays within ``global_budget`` at every step,
- global compression fires at least once over the 10-turn run,
- ``run_global_compression`` always drives the cache to exactly the target,
- model predictions (logits) remain non-empty across the whole trajectory.
"""


import torch
from transformers import DynamicCache

from kvpress import SnapKVPress
from kvpress.presses.loyalty_press import LoyaltyPress
from kvpress.presses.role_boundary_anchor_press import RoleBoundaryAnchorPress
from kvpress.presses.turn_aware_global_press import TurnAwareGlobalPress
from kvpress.presses.turn_floor_press import TurnFloorPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_10_turn_fake_session(unit_test_model):  # noqa: F811
    """Simulates the multi-turn harness manually until ``multi_turn_evaluate.py``
    lands (Week 2). Exercises the full Week-1 stack: per-policy callbacks,
    forward_hook buffering + loyalty accrual, and ``run_global_compression``.
    """
    device = unit_test_model.device
    torch.manual_seed(0)

    budget = 200  # 10 turns * ~30 tokens = ~300 -> compression fires at least once

    floor = TurnFloorPress(global_budget=budget)
    anchor = RoleBoundaryAnchorPress()
    loyalty = LoyaltyPress()
    press = TurnAwareGlobalPress(
        # window_size=16 is well under the 30-token prefill so SnapKV's
        # window assertion holds at global-compression time.
        base_press=SnapKVPress(compression_ratio=0.0, window_size=16),
        global_budget=budget,
        policies={"floor": floor, "anchor": anchor, "loyalty": loyalty},
        alphas={"floor": 1.0, "anchor": 1.0, "loyalty": 1.0},
    )

    call_log: list[tuple] = []
    cache = DynamicCache()
    compression_count = 0
    n_turns = 10

    with press(unit_test_model), torch.no_grad():
        for turn_idx in range(1, n_turns + 1):
            # Harness-side cursor comes from the cache's current length so
            # it stays correct across the compression step below (which
            # shrinks the cache to ``budget`` and re-bases future spans).
            start_kv = cache.get_seq_length()
            press.on_turn_start(turn_idx, "user", start_kv)
            call_log.append(("start", turn_idx, start_kv))

            prefill_len = 30
            ids = torch.randint(0, 1024, (1, prefill_len), device=device)
            out = unit_test_model(ids, past_key_values=cache)
            # Predictions non-empty per the ADR 002 integration-test contract.
            assert out.logits.shape[1] == prefill_len, (
                f"turn {turn_idx}: logits must cover all prefill tokens, got {out.logits.shape}"
            )

            end_kv = cache.get_seq_length()
            press.on_turn_end(turn_idx, "user", start_kv, end_kv)
            call_log.append(("end", turn_idx, start_kv, end_kv))

            if cache.get_seq_length() > budget:
                press.run_global_compression(unit_test_model, cache, target=budget)
                compression_count += 1
                call_log.append(("compress", turn_idx, cache.get_seq_length()))
                assert cache.get_seq_length() == budget, (
                    f"turn {turn_idx}: run_global_compression did not hit target exactly "
                    f"(got {cache.get_seq_length()}, want {budget})"
                )

            assert cache.get_seq_length() <= budget, (
                f"turn {turn_idx}: cache ({cache.get_seq_length()}) exceeded budget ({budget})"
            )

    # Callback-ordering invariant: each turn's ``start`` must precede its
    # ``end`` in linear call-log order (a zipped check alone would miss a
    # reordering bug that swapped a start/end pair within the same index).
    starts = [e for e in call_log if e[0] == "start"]
    ends = [e for e in call_log if e[0] == "end"]
    assert len(starts) == n_turns and len(ends) == n_turns
    for turn_idx in range(1, n_turns + 1):
        s_pos = next(i for i, e in enumerate(call_log) if e[0] == "start" and e[1] == turn_idx)
        e_pos = next(i for i, e in enumerate(call_log) if e[0] == "end" and e[1] == turn_idx)
        assert s_pos < e_pos, f"turn {turn_idx}: start at {s_pos} must precede end at {e_pos}"

    # Budget arithmetic: 10 turns * 30 tokens = 300, budget = 200.
    # Running cache length after each turn (pre-compression):
    #   t1..t6: 30, 60, 90, 120, 150, 180 -- all <= 200, no compress.
    #   t7: 210 > 200 -> compress to 200.
    #   t8..t10: 230 > 200 each time -> compress.
    # Therefore exactly 4 compressions fire.
    assert compression_count == 4, (
        f"expected exactly 4 compressions (turns 7-10), got {compression_count}"
    )

    # After the context manager exits, reset() runs (composer __call__).
    # Verify state is cleared so repeated usage doesn't leak.
    assert press._last_hidden_states == {} and press._last_kwargs == {}
    assert floor.turn_boundaries == [] and floor.current_turn == 0
    assert anchor.turn_boundaries == [] and anchor.current_turn == 0
    assert loyalty.turn_boundaries == [] and loyalty.current_turn == 0
    assert loyalty._current_turn_start_kv == 0 and loyalty.loyalty == {}
