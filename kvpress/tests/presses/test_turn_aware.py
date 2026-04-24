# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from transformers import DynamicCache

from kvpress import SnapKVPress
from kvpress.presses.loyalty_press import LoyaltyPress
from kvpress.presses.role_boundary_anchor_press import RoleBoundaryAnchorPress
from kvpress.presses.turn_aware_base import VALID_ROLES, TurnAwareMixin, TurnBoundary
from kvpress.presses.turn_aware_global_press import TurnAwareGlobalPress
from kvpress.presses.turn_floor_press import TurnFloorPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_turn_boundary_and_mixin_callbacks():
    """Covers turn_aware_base.py: TurnBoundary invariants and the TurnAwareMixin
    callback contract (on_turn_start sets current_turn, on_turn_end appends a span,
    the same turn_idx may be emitted twice for user+assistant per ADR 001 §4).
    """
    # TurnBoundary construction and invariants
    tb = TurnBoundary(turn_idx=1, start_kv=0, end_kv=10, role="user")
    assert len(tb) == 10

    with pytest.raises(AssertionError):
        TurnBoundary(turn_idx=1, start_kv=10, end_kv=5, role="user")  # end < start
    with pytest.raises(AssertionError):
        TurnBoundary(turn_idx=1, start_kv=-1, end_kv=5, role="user")  # negative start
    with pytest.raises(AssertionError):
        TurnBoundary(turn_idx=1, start_kv=0, end_kv=5, role="system")  # bad role
    with pytest.raises(AssertionError):
        TurnBoundary(turn_idx=-1, start_kv=0, end_kv=5, role="user")  # negative turn_idx

    # VALID_ROLES advertises the full allow-list; downstream loaders depend on it
    assert set(VALID_ROLES) == {"context", "user", "assistant", "feedback"}

    # Mixin callbacks: fresh instance has empty state, callbacks populate it
    mixin = TurnAwareMixin()
    assert mixin.turn_boundaries == [] and mixin.loyalty == {} and mixin.current_turn == 0

    # Harness emits two spans for one turn (user then assistant)
    mixin.on_turn_start(turn_idx=1, role="user", start_kv=0)
    assert mixin.current_turn == 1
    mixin.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=10)
    mixin.on_turn_end(turn_idx=1, role="assistant", start_kv=10, end_kv=25)

    assert len(mixin.turn_boundaries) == 2
    assert mixin.turn_boundaries[0].turn_idx == mixin.turn_boundaries[1].turn_idx == 1
    assert mixin.turn_boundaries[0].role == "user"
    assert mixin.turn_boundaries[1].role == "assistant"
    assert mixin.turn_boundaries[0].end_kv == 10
    assert mixin.turn_boundaries[1].end_kv == 25

    # Default compute_weights returns ones on CPU/fp32 (the composer overrides)
    weights = mixin.compute_weights(kv_len=25)
    assert weights.shape == (25,)
    assert weights.dtype == torch.float32
    assert torch.all(weights == 1.0)

    # reset_turn_state clears all bookkeeping so the press can be reused across probes
    mixin.reset_turn_state()
    assert mixin.turn_boundaries == [] and mixin.loyalty == {} and mixin.current_turn == 0

    # default_factory gives each instance its own list/dict (classic dataclass gotcha)
    a, b = TurnAwareMixin(), TurnAwareMixin()
    a.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=3)
    assert b.turn_boundaries == [], "mixin instances must not share default state"


def test_turn_floor_exempts_short_turns():
    """Covers turn_floor_press.py (ADR 002 §7): a 5-token turn receives zero floor
    allocation because it falls under ``exempt_shorter_than``. A 20-token turn
    receives a non-zero allocation.
    """
    # Short turn (5 tokens) below the default 10-token exemption threshold
    short = TurnFloorPress(global_budget=1000)
    short.on_turn_start(turn_idx=1, role="user", start_kv=0)
    short.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=5)
    short.current_turn = 2  # we are past that turn; decay would be 0.9^1

    weights_short = short.compute_weights(kv_len=5)
    assert weights_short.shape == (5,)
    assert torch.all(weights_short == 0.0), "short turn must receive no floor allocation"

    # Control: a 20-token turn (above threshold) gets a non-empty allocation
    regular = TurnFloorPress(global_budget=1000)
    regular.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=20)
    regular.current_turn = 2
    weights_regular = regular.compute_weights(kv_len=20)
    assert weights_regular.sum().item() > 0, "non-exempt turn must have non-zero floor"

    # Context span (turn_idx=0 / role="context") is excluded from both numerator
    # and denominator of the per-turn share, matching ADR 001 §0 KEEP bucket
    ctx_mixed = TurnFloorPress(global_budget=100, alpha_floor_len=0.3, gamma=1.0, min_floor_tokens=0)
    ctx_mixed.on_turn_end(turn_idx=0, role="context", start_kv=0, end_kv=1000)
    ctx_mixed.on_turn_end(turn_idx=1, role="user", start_kv=1000, end_kv=1050)
    ctx_mixed.current_turn = 2
    weights_ctx = ctx_mixed.compute_weights(kv_len=1050)
    assert torch.all(weights_ctx[:1000] == 0.0), "context span must not receive floor"


def test_role_boundary_assistant_last_only():
    """Covers role_boundary_anchor_press.py (ADR 002 §7): an assistant span has
    ones at ``[end-w, end)`` and zeros at ``[start, start+w)``. Requires
    ``|span| >= 2 * min_anchor_tokens`` so the head and tail windows don't overlap.
    """
    # Span of length 20: w = max(3, floor(0.15*20)) = max(3, 3) = 3. 2*w <= 20 ⇒ no overlap.
    press = RoleBoundaryAnchorPress()
    press.on_turn_start(turn_idx=1, role="assistant", start_kv=0)
    press.on_turn_end(turn_idx=1, role="assistant", start_kv=0, end_kv=20)

    weights = press.compute_weights(kv_len=20)
    assert weights.shape == (20,)
    assert torch.all(weights[17:20] == 1.0), "assistant tail anchor [end-w, end) must be 1.0"
    assert torch.all(weights[0:3] == 0.0), "assistant must NOT set head anchor [start, start+w)"
    assert torch.all(weights[3:17] == 0.0), "assistant interior must be 0"

    # Symmetry check: a same-sized USER span has BOTH head and tail set
    user_press = RoleBoundaryAnchorPress()
    user_press.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=20)
    w_user = user_press.compute_weights(kv_len=20)
    assert torch.all(w_user[0:3] == 1.0), "user head anchor must be 1.0"
    assert torch.all(w_user[17:20] == 1.0), "user tail anchor must be 1.0"
    assert torch.all(w_user[3:17] == 0.0), "user interior must be 0"

    # role="feedback" is accepted by TurnBoundary (VALID_ROLES) but the
    # current ConvCodeWorld driver embeds feedback in the next iteration's
    # user prompt rather than emitting a dedicated span, so this policy
    # no longer special-cases it. A feedback span therefore receives no
    # anchors. If a future loader begins emitting role="feedback" spans,
    # restore an assistant-like last-w branch and update this assertion.
    fb_press = RoleBoundaryAnchorPress()
    fb_press.on_turn_end(turn_idx=1, role="feedback", start_kv=0, end_kv=20)
    w_fb = fb_press.compute_weights(kv_len=20)
    assert torch.all(w_fb == 0.0), "unhandled role must receive no anchors"

    # Context is skipped (KEEP bucket per ADR 001 §0)
    ctx_press = RoleBoundaryAnchorPress()
    ctx_press.on_turn_end(turn_idx=0, role="context", start_kv=0, end_kv=20)
    w_ctx = ctx_press.compute_weights(kv_len=20)
    assert torch.all(w_ctx == 0.0), "context span must not receive anchors"


# ---------------------------------------------------------------------------
# The three remaining ADR 002 §7 tests. These exercise TurnAwareGlobalPress
# end-to-end against the unit_test_model fixture and are the Week-1 gate
# for the press primitives landing on main.
# ---------------------------------------------------------------------------


def _capture_layer0_forward(model, ids):
    """Run a forward pass and capture layer-0 hidden_states + kwargs so
    tests can feed ``ScorerPress.score`` the exact inputs it would see
    during the real forward_hook path (matches the composer's buffer).
    """
    layer0 = model.model.layers[0].self_attn
    layer0.rotary_emb = model.model.rotary_emb

    capture: dict = {}

    def hook(_module, _input, kwargs, output):
        capture["hidden_states"] = kwargs["hidden_states"].detach().clone()
        capture["kwargs"] = dict(kwargs)
        return output

    handle = layer0.register_forward_hook(hook, with_kwargs=True)
    cache = DynamicCache()
    with torch.no_grad():
        model(ids, past_key_values=cache)
    handle.remove()
    return layer0, cache, capture


def test_all_alphas_zero_equivalent_to_base(unit_test_model):  # noqa: F811
    """ADR 002 §7 critical regression guard: TurnAwareGlobalPress with
    every alpha = 0 must produce bit-identical evictions to its stock
    base press. If this fails the whole ablation matrix is meaningless.
    """
    device = unit_test_model.device
    torch.manual_seed(42)
    # 256 tokens >> SnapKV's window=16 so the base scorer has enough context
    ids = torch.randint(0, 1024, (1, 256), device=device)

    layer0, cache, capture = _capture_layer0_forward(unit_test_model, ids)
    keys = cache.layers[0].keys
    values = cache.layers[0].values

    stock = SnapKVPress(compression_ratio=0.5, window_size=16)
    k_stock, v_stock = stock.compress(
        layer0, capture["hidden_states"], keys.clone(), values.clone(), None, capture["kwargs"]
    )

    wrapped = TurnAwareGlobalPress(
        base_press=SnapKVPress(compression_ratio=0.5, window_size=16),
        global_budget=128,
        policies={
            "floor": TurnFloorPress(global_budget=128),
            "anchor": RoleBoundaryAnchorPress(),
            "loyalty": LoyaltyPress(),
        },
        alphas={"floor": 0.0, "anchor": 0.0, "loyalty": 0.0},
    )
    k_wrapped, v_wrapped = wrapped.compress(
        layer0, capture["hidden_states"], keys.clone(), values.clone(), None, capture["kwargs"]
    )

    assert k_stock.shape == k_wrapped.shape
    assert torch.equal(k_stock, k_wrapped), "all-αs-zero short-circuit must be bit-identical keys"
    assert torch.equal(v_stock, v_wrapped), "all-αs-zero short-circuit must be bit-identical values"


def test_loyalty_updates_during_prefill(unit_test_model):  # noqa: F811
    """ADR 002 §7: replay a fake multi-turn session and assert loyalty > 0
    on past-turn positions after turn 2's prefill. Turn 1 (no past) must
    not produce any loyalty. All populated positions must lie inside
    turn 1's span per the current-turn-self-skip rule (ADR 001 §3-C).
    """
    device = unit_test_model.device
    torch.manual_seed(0)

    loyalty = LoyaltyPress()
    press = TurnAwareGlobalPress(
        base_press=SnapKVPress(compression_ratio=0.0, window_size=16),
        global_budget=4096,
        policies={"loyalty": loyalty},
        alphas={"loyalty": 1.0},  # non-zero so forward_hook drives update_loyalty
    )

    cache = DynamicCache()
    n_layers = len(unit_test_model.model.layers)
    with press(unit_test_model), torch.no_grad():
        # Turn 1: 30 tokens, no past context -> loyalty stays empty
        press.on_turn_start(turn_idx=1, role="user", start_kv=0)
        ids1 = torch.randint(0, 1024, (1, 30), device=device)
        unit_test_model(ids1, past_key_values=cache)
        turn1_end = cache.get_seq_length()
        press.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=turn1_end)
        assert loyalty.loyalty == {}, "turn 1 has no past -> no loyalty accrual"

        # Turn 2: 20 tokens. All current-turn queries attend to turn 1's keys
        # (indices [0, turn1_end)). Top-25% positions should get loyalty += 1.
        press.on_turn_start(turn_idx=2, role="user", start_kv=turn1_end)
        ids2 = torch.randint(0, 1024, (1, 20), device=device)
        unit_test_model(ids2, past_key_values=cache)
        press.on_turn_end(turn_idx=2, role="user", start_kv=turn1_end, end_kv=cache.get_seq_length())

        # Loyalty assertions must run INSIDE the context manager; the
        # composer's ``__call__`` runs ``reset()`` on exit (by design --
        # see turn_aware_global_press.py) which would clear loyalty.
        assert len(loyalty.loyalty) > 0, "turn 2 prefill must have populated loyalty counters"
        # update_loyalty fires once per layer per forward pass so a position
        # voted for in every layer gets count >= n_layers; require >= n_layers
        # on the maximum to prove all layers' hooks fired (not just one).
        for pos, count in loyalty.loyalty.items():
            assert 0 <= pos < turn1_end, f"loyalty key {pos} outside past-turn span [0, {turn1_end})"
            assert count > 0, f"loyalty count at {pos} must be positive, got {count}"
        assert max(loyalty.loyalty.values()) >= n_layers, (
            f"max loyalty ({max(loyalty.loyalty.values())}) < n_layers ({n_layers}); "
            "suggests forward_hook fired on only a subset of layers"
        )

    # After the context manager exits, reset() runs and clears all state.
    assert loyalty.loyalty == {}, "reset() on __call__ exit must clear loyalty"


def test_budget_hit(unit_test_model):  # noqa: F811
    """ADR 002 §7: ``run_global_compression(target=N)`` always exits with
    ``cache.get_seq_length() == N``. Also spot-check the binary-search
    helper on adversarial ``(k_len, target)`` pairs where naive
    ``1 - target/k_len`` would truncate by one.
    """
    device = unit_test_model.device
    torch.manual_seed(0)

    press = TurnAwareGlobalPress(
        base_press=SnapKVPress(compression_ratio=0.0, window_size=16),
        global_budget=100,
        policies={},
        alphas={},
    )

    cache = DynamicCache()
    with press(unit_test_model), torch.no_grad():
        press.on_turn_start(turn_idx=1, role="user", start_kv=0)
        ids = torch.randint(0, 1024, (1, 256), device=device)
        unit_test_model(ids, past_key_values=cache)
        press.on_turn_end(turn_idx=1, role="user", start_kv=0, end_kv=256)
        assert cache.get_seq_length() == 256
        press.run_global_compression(unit_test_model, cache, target=100)
        assert cache.get_seq_length() == 100, "target=100 must yield exactly 100 KV positions"

    # Static adversarial coverage of the binary search itself
    for k_len, target in [(104999, 4096), (131071, 8192), (100, 17), (7, 3), (65536, 32768)]:
        ratio = TurnAwareGlobalPress._ratio_for_exact_target(k_len, target)
        n_kept = int(k_len * (1 - ratio))
        assert n_kept == target, f"binary search off by {target - n_kept} at k_len={k_len} target={target}"
