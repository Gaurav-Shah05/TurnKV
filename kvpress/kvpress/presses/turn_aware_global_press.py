# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.turn_aware_base import TurnAwareMixin
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class TurnAwareGlobalPress(BasePress):
    """
    Weighted global-compression composer (ADR 001 §3, ADR 002 §2 row 5).

    Combines a ``ScorerPress`` base and a set of ``TurnAwareMixin`` policies::

        final_score(t) = base_scorer(t) · (1 + Σ_i alpha_i · w_i(t))

    With every ``alpha_i == 0`` (e.g. ``baseline_*`` registry entries) the
    composite short-circuits to ``base_press.compress(...)`` unchanged,
    preserving bit-identical ``topk`` evictions -- the regression guard
    ``test_all_alphas_zero_equivalent_to_base`` (ADR 002 §7) enforces.

    Compression cadence (ADR 001 §1): local/decode is handled by wrapping
    the base press in ``DecodingPress`` separately; this composer is the
    global/turn-boundary path, invoked via :meth:`run_global_compression`
    after :meth:`on_turn_end`. :meth:`forward_hook` does not evict; it
    only buffers per-layer ``hidden_states`` + ``kwargs`` for the next
    global compression and drives every policy's ``update_loyalty`` so
    ``LoyaltyPress`` accumulates during prefill and decode (ADR 001 §3-C).
    Turn callbacks dispatch to every policy so the harness talks only to
    the composer (ADR 001 §4).

    Week-1 scope limits: ``base_press`` must be a plain ``ScorerPress``
    (the per-head masking surface of ``AdaKVPress`` needs a Week-2
    extension), and the buffered hidden_states feeding
    ``run_global_compression`` must be long enough for the base scorer
    (e.g. ``SnapKVPress.window_size + 1``). Week-1 tests trigger global
    compression after prefill to satisfy that; a rolling buffer
    (analogous to ``DecodingPress.hidden_states_buffer``) is a Week-2
    refinement.

    Parameters
    ----------
    base_press : ScorerPress (keyword-only, required)
        Scorer press whose ``score`` drives the base ranking.
    global_budget : int (keyword-only, required)
        Default target cache size, in KV positions. Can be overridden
        per-call via ``run_global_compression(..., target=N)``.
    policies : dict[str, TurnAwareMixin], default empty
        Named policies (typical keys: ``floor``, ``anchor``, ``loyalty``).
    alphas : dict[str, float], default empty
        Per-policy blend coefficients; alphas without a matching policy
        warn at construction and are ignored.
    """

    base_press: ScorerPress
    global_budget: int
    policies: dict[str, TurnAwareMixin] = field(default_factory=dict)
    alphas: dict[str, float] = field(default_factory=dict)

    _last_hidden_states: dict = field(default_factory=dict, init=False, repr=False)
    _last_kwargs: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.base_press, ScorerPress), (
            f"TurnAwareGlobalPress requires a ScorerPress base; got {type(self.base_press).__name__} "
            "(AdaKVPress support is Week-2)."
        )
        assert self.global_budget > 0, f"global_budget must be positive, got {self.global_budget}"
        for name, policy in self.policies.items():
            assert isinstance(policy, TurnAwareMixin), (
                f"policy '{name}' must inherit TurnAwareMixin; got {type(policy).__name__}"
            )
        for name in self.alphas:
            if name not in self.policies:
                logger.warning("alpha entry %r has no matching policy; its coefficient is ignored.", name)

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        self.base_press.post_init_from_model(model)

    # ADR 001 §4 turn-boundary callbacks: dispatch to every policy.

    def on_turn_start(self, turn_idx: int, role: str, start_kv: int) -> None:
        for policy in self.policies.values():
            policy.on_turn_start(turn_idx, role, start_kv)

    def on_turn_end(self, turn_idx: int, role: str, start_kv: int, end_kv: int) -> None:
        for policy in self.policies.values():
            policy.on_turn_end(turn_idx, role, start_kv, end_kv)

    def forward_hook(self, module: nn.Module, input: list, kwargs: dict, output: list):
        """Buffer per-layer state, drive ``update_loyalty`` on all policies,
        never write to the cache. Eviction only in :meth:`run_global_compression`.
        """
        hidden_states = kwargs.get("hidden_states")
        cache = kwargs.get("past_key_values")
        if hidden_states is None or cache is None:
            return output

        layer_idx = module.layer_idx
        # detach().clone() matches DecodingPress (decoding_press.py:133) so a
        # later forward pass cannot rewrite our buffered tensor in place.
        self._last_hidden_states[layer_idx] = hidden_states.detach().clone()
        # Shallow-copy the kwargs dict; inner tensors (position_embeddings)
        # are refreshed per forward pass, and at compression time we slice
        # cos/sin by q_len -- fine under Week-1 prefill-only compression.
        self._last_kwargs[layer_idx] = dict(kwargs)

        # All-alphas-zero: no policy weight contributes, so skip loyalty
        # accumulation -- matches baseline_* behaviour from ADR 002 §2.
        if self._all_alphas_zero():
            return output

        try:
            keys, _ = extract_keys_and_values(cache, layer_idx)
        except (IndexError, AttributeError, TypeError):
            return output

        for policy in self.policies.values():
            policy.update_loyalty(module, hidden_states, keys, kwargs, policy.current_turn)

        return output

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-layer compression via weighted composite or the all-αs=0
        short-circuit. Uses ``self.base_press.compression_ratio``;
        :meth:`run_global_compression` sets it transiently to hit an exact
        target size (binary-searched in :meth:`_ratio_for_exact_target`).
        """
        ratio = self.base_press.compression_ratio
        if ratio == 0:
            return keys, values

        # All-αs=0 short-circuit = bit-equivalence regression guard (ADR 002 §7)
        if self._all_alphas_zero():
            return self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)

        base_scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        # base_scores: (bsz, num_kv_heads, k_len). Upcast to fp32 for the
        # weighted multiply so small α·w terms (e.g. 1 + 1e-3) do not
        # underflow bf16's ~4e-3 resolution near 1.0; topk is
        # rank-invariant under monotone transforms so picking indices in
        # fp32 is safe. weight shape (k_len,) broadcasts over (bsz, kv_heads, ·).
        k_len = keys.shape[2]
        weight = self._combined_weight(k_len, device=base_scores.device, dtype=torch.float32)
        final_scores = base_scores.float() * weight

        n_kept = int(k_len * (1 - ratio))
        indices = final_scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values

    def _combined_weight(self, kv_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return ``1 + Σ_i alpha_i · w_i`` at shape ``(kv_len,)``."""
        weight = torch.ones(kv_len, device=device, dtype=dtype)
        for name, policy in self.policies.items():
            alpha = float(self.alphas.get(name, 0.0))
            if alpha == 0.0:
                continue
            w_i = policy.compute_weights(kv_len, device=device, dtype=dtype)
            weight = weight + alpha * w_i
        return weight

    def _all_alphas_zero(self) -> bool:
        return all(abs(float(self.alphas.get(name, 0.0))) == 0.0 for name in self.policies)

    def run_global_compression(
        self,
        model: PreTrainedModel,
        cache: Cache,
        target: Optional[int] = None,
    ) -> None:
        """Evict cache down to ``target`` (default ``self.global_budget``)
        across every layer. No-op if it already fits. Harness-invoked per
        ADR 001 §4 (``run_global_compression(target=global_budget)``).

        Uses a binary-searched ratio (``_ratio_for_exact_target``) so the
        ``int(k_len * (1 - r))`` inside ``ScorerPress.compress`` hits the
        target exactly -- a plain ``1 - target/k_len`` misses by one at
        realistic LongMemEval shapes and would break ``test_budget_hit``
        (ADR 002 §7).
        """
        effective_target = self.global_budget if target is None else target
        assert effective_target >= 0, f"target must be non-negative, got {effective_target}"
        current_len = cache.get_seq_length()
        if current_len <= effective_target:
            return

        target_ratio = self._ratio_for_exact_target(current_len, effective_target)
        orig_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_ratio
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                module = layer.self_attn
                layer_idx = getattr(module, "layer_idx", None)
                if layer_idx is None:
                    continue
                hidden_states = self._last_hidden_states.get(layer_idx)
                kwargs = self._last_kwargs.get(layer_idx)
                if hidden_states is None or kwargs is None:
                    # Expected for layers skipped during __call__ (e.g.
                    # Gemma3 sliding-window); nothing to compress.
                    continue
                keys, values = extract_keys_and_values(cache, layer_idx)
                new_keys, new_values = self.compress(module, hidden_states, keys, values, None, kwargs)
                self._write_layer(cache, layer_idx, new_keys, new_values)
        finally:
            self.base_press.compression_ratio = orig_ratio

    @staticmethod
    def _ratio_for_exact_target(k_len: int, target: int) -> float:
        """Binary-search ``r`` such that ``int(k_len * (1 - r)) == target``.
        Mirrors ``DecodingPress._find_target_compression_ratio``.
        """
        if k_len <= target:
            return 0.0
        ratio = 1.0 - (target / k_len)
        low, high = 0.0, 1.0
        for _ in range(30):
            n_kept = int(k_len * (1 - ratio))
            if n_kept == target:
                return ratio
            if n_kept > target:
                low = ratio
                ratio = (ratio + high) / 2
            else:
                high = ratio
                ratio = (low + ratio) / 2
        logger.warning("Binary search failed: k_len=%d target=%d got=%d", k_len, target, n_kept)
        return ratio

    @staticmethod
    def _write_layer(cache: Cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Cache writeback mirroring ``BasePress.forward_hook``."""
        cache_layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.values = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache_layer.cumulative_length = keys.shape[2]
        else:
            cache_layer.keys = keys
            cache_layer.values = values

    def reset(self) -> None:
        """Clear per-layer buffers and every policy's turn state."""
        self._last_hidden_states.clear()
        self._last_kwargs.clear()
        for policy in self.policies.values():
            policy.reset_turn_state()

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """Like ``BasePress.__call__`` but ``reset()`` on exit so repeated
        use doesn't leak buffered state between sessions (mirrors
        ``DecodingPress.__call__``).
        """
        try:
            with super().__call__(model):
                yield
        finally:
            self.reset()
