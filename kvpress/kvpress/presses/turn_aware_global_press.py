# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
import math
from collections import defaultdict
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
    Per-turn-budget global-compression composer (ADR 001 §3, amended).

    Allocates a per-turn compression budget using the formula::

        base_floor = (global_budget - |context|) / num_conversational_turns
        budget_i   = max( alpha_anchor  * |anchor positions in turn i|
                        + alpha_loyalty * |loyalty-scored positions in turn i|,
                          alpha_floor   * base_floor * exp(-gamma * (T - i)) )

    clamped to turn length. Within each turn, anchor positions are
    mandatory keeps; the remaining ``budget_i - anchor_count`` slots are
    filled by the top-scoring positions under::

        score_adj(t) = base_scorer(t) * (1 + alpha_loyalty * loyalty_norm(t))

    Context spans (``turn_idx == 0`` or ``role == "context"``) are the
    KEEP bucket per ADR 001 §0 and are preserved in full regardless of
    budget.

    **Short-circuit** to stock base-press compression (bit-identical
    ``topk`` evictions) in two cases so ``baseline_*`` registry entries
    and the ``test_all_alphas_zero_equivalent_to_base`` regression guard
    keep working:

    1. All alphas are 0 (no policy contribution).
    2. No ``turn_boundaries`` recorded on any policy yet (e.g. the first
       compression before any ``on_turn_end`` has fired).

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
    (e.g. ``SnapKVPress.window_size + 1``).

    Parameters
    ----------
    base_press : ScorerPress (keyword-only, required)
        Scorer press whose ``score`` drives the in-turn ranking.
    global_budget : int (keyword-only, required)
        Default target cache size, in KV positions. Drives the
        ``base_floor = (global_budget - |context|) / num_turns`` term.
    policies : dict[str, TurnAwareMixin], default empty
        Named policies. The composer recognises the keys ``floor``,
        ``anchor``, and ``loyalty``; unknown keys are accepted for
        forward compatibility but their weights are ignored here.
    alphas : dict[str, float], default empty
        Per-policy coefficients: ``alpha_floor`` scales the floor term,
        ``alpha_anchor`` and ``alpha_loyalty`` gate their respective
        contributions to ``budget_i`` and the scoring weight for
        non-anchor fills.
    per_turn_gamma : float, default=0.1
        Exponential-decay coefficient in ``exp(-gamma * (T - i))``. With
        ``gamma=0.1`` the decay is roughly ``0.9047^(T - i)``, matching
        the ADR 001 §3-A geometric ``gamma=0.9`` within ~1%.
    """

    base_press: ScorerPress
    global_budget: int
    policies: dict[str, TurnAwareMixin] = field(default_factory=dict)
    alphas: dict[str, float] = field(default_factory=dict)
    per_turn_gamma: float = 0.1

    _last_hidden_states: dict = field(default_factory=dict, init=False, repr=False)
    _last_kwargs: dict = field(default_factory=dict, init=False, repr=False)
    _suspend_hooks: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        assert isinstance(self.base_press, ScorerPress), (
            f"TurnAwareGlobalPress requires a ScorerPress base; got {type(self.base_press).__name__} "
            "(AdaKVPress support is Week-2)."
        )
        assert self.global_budget > 0, f"global_budget must be positive, got {self.global_budget}"
        assert self.per_turn_gamma >= 0.0, f"per_turn_gamma must be non-negative, got {self.per_turn_gamma}"
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
        if self._suspend_hooks:
            return output

        hidden_states = kwargs.get("hidden_states")
        cache = kwargs.get("past_key_values")
        if hidden_states is None or cache is None:
            return output

        layer_idx = module.layer_idx
        # detach().clone() matches DecodingPress (decoding_press.py:133) so a
        # later forward pass cannot rewrite our buffered tensor in place.
        # Keep the latest multi-token prefill/query state. Single-token decode
        # forwards are still useful for loyalty updates below, but they are too
        # short for scorer presses such as SnapKV that need an observation
        # window during turn-boundary compression.
        if hidden_states.shape[1] > 1 or layer_idx not in self._last_hidden_states:
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
        """Single-layer per-turn-budget compression. Short-circuits to
        the stock base press when either (a) all alphas are zero, or
        (b) no turn boundaries are recorded yet -- the latter handles
        the initial pre-first-turn compress cleanly and preserves the
        ``test_all_alphas_zero_equivalent_to_base`` regression guard.
        """
        new_keys, new_values, _ = self._compress_impl(
            module, hidden_states, keys, values, attentions, kwargs
        )
        return new_keys, new_values

    def _compress_impl(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: Optional[torch.Tensor],
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Like :meth:`compress` but also returns the ``all_kept`` index
        tensor (shape ``(bsz, nkv_heads, n_kept)``, dtype ``int64``) so
        callers such as :meth:`run_global_compression` can remap policy
        state after eviction.  Returns ``None`` for ``all_kept`` whenever
        a short-circuit path is taken (no eviction actually occurred).
        """
        ratio = self.base_press.compression_ratio
        if ratio == 0:
            return keys, values, None

        # Short-circuit 1: all alphas 0 -> baseline behaviour. Keeps
        # baseline_snapkv-style registry entries bit-identical to stock
        # SnapKVPress.
        if self._all_alphas_zero():
            new_k, new_v = self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
            return new_k, new_v, None

        # Short-circuit 2: no turn structure to leverage yet. Degrade to
        # plain base compression. Policies share turn_boundaries via the
        # composer's on_turn_end dispatch, so checking any one is fine.
        first_policy = next(iter(self.policies.values()), None)
        if first_policy is None or not first_policy.turn_boundaries:
            new_k, new_v = self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
            return new_k, new_v, None

        k_len = keys.shape[2]
        bsz, num_kv_heads, _, head_dim = keys.shape

        turn_budgets, context_spans = self._compute_turn_budgets(k_len)

        if not turn_budgets and not context_spans:
            new_k, new_v = self.base_press.compress(module, hidden_states, keys, values, attentions, kwargs)
            return new_k, new_v, None

        # If our total kept already fits (or exceeds) the cache, there is
        # nothing to evict. Happens when budgets + context >= k_len.
        total_kept = sum(ce - cs for cs, ce in context_spans) + sum(b for _, _, b in turn_budgets.values())
        if total_kept >= k_len:
            return keys, values, None

        # Base scoring is per-layer; upcast to fp32 so small alpha*loyalty
        # bumps do not underflow bf16 and so topk ties resolve deterministically.
        base_scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs).float()

        # Loyalty weighting applied to base_scores BEFORE per-turn topk so the
        # retained positions within each turn prefer high-loyalty tokens.
        alpha_loyalty = float(self.alphas.get("loyalty", 0.0))
        if alpha_loyalty > 0.0 and "loyalty" in self.policies:
            loyalty_w = self.policies["loyalty"].compute_weights(
                k_len, device=base_scores.device, dtype=torch.float32
            )
            base_scores = base_scores * (1.0 + alpha_loyalty * loyalty_w)

        # Anchor mask: positions marked by RoleBoundaryAnchorPress. Mandatory
        # keeps inside their turn, implemented by +inf scoring before topk so
        # they sort to the top regardless of base score.
        anchor_mask_gpu: Optional[torch.Tensor] = None
        alpha_anchor = float(self.alphas.get("anchor", 0.0))
        if alpha_anchor > 0.0 and "anchor" in self.policies:
            anchor_w = self.policies["anchor"].compute_weights(
                k_len, device=base_scores.device, dtype=torch.float32
            )
            anchor_mask_gpu = anchor_w > 0

        kept_chunks: list[torch.Tensor] = []
        dev = base_scores.device

        # Context spans: always kept in full (KEEP bucket).
        for cs, ce in context_spans:
            if ce <= cs:
                continue
            idx = torch.arange(cs, ce, device=dev, dtype=torch.long)
            kept_chunks.append(idx.expand(bsz, num_kv_heads, -1))

        # Per turn: budget is deterministic (scalar per turn). Within the
        # budget, each (batch, kv_head) picks its own top-k, so head-wise
        # attention asymmetries surface naturally -- same pattern as
        # ScorerPress.compress.
        for turn_idx in sorted(turn_budgets):
            ts, te, budget = turn_budgets[turn_idx]
            if ts >= te or budget <= 0:
                continue
            turn_len = te - ts
            budget = min(budget, turn_len)

            turn_scores = base_scores[..., ts:te]  # (bsz, nkv, turn_len)

            if anchor_mask_gpu is not None:
                turn_anchor = anchor_mask_gpu[ts:te]
                # Boost anchor positions to +inf so topk selects them first.
                # If anchor_count > budget, topk returns the k=budget anchor
                # positions ranked by original score (since torch's topk is
                # stable under equal +inf entries -- falls through to the
                # pre-infinity values).
                boosted = turn_scores.clone()
                boosted[..., turn_anchor] = float("inf")
                top_local = boosted.topk(budget, dim=-1).indices
            else:
                top_local = turn_scores.topk(budget, dim=-1).indices

            kept_chunks.append(top_local + ts)

        if not kept_chunks:
            return keys, values, None

        # Concatenate and sort for positional locality (RoPE-friendly, and
        # downstream attention kernels that assume monotonic positions).
        all_kept = torch.cat(kept_chunks, dim=-1)
        all_kept, _ = all_kept.sort(dim=-1)

        indices_expanded = all_kept.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        new_keys = keys.gather(2, indices_expanded).contiguous()
        new_values = values.gather(2, indices_expanded).contiguous()
        return new_keys, new_values, all_kept

    def _compute_turn_budgets(
        self, k_len: int
    ) -> tuple[dict[int, tuple[int, int, int]], list[tuple[int, int]]]:
        """Compute per-turn budgets and context spans for the cache state.

        Returns
        -------
        (turn_budgets, context_spans)
            ``turn_budgets`` maps ``turn_idx -> (span_start, span_end, budget)``
            for each conversational turn; ``context_spans`` is the list of
            ``(start, end)`` ranges belonging to the KEEP bucket.

        The budget math follows the formula in the class docstring. All
        counts are computed on shared CPU copies of the anchor/loyalty
        weight masks -- one transfer per call, not per policy per turn.
        """
        first_policy = next(iter(self.policies.values()), None)
        if first_policy is None:
            return {}, []

        # Group boundaries by turn_idx -- policies share the list via the
        # composer's on_turn_end dispatch.
        turn_spans: dict[int, list] = defaultdict(list)
        for b in first_policy.turn_boundaries:
            turn_spans[b.turn_idx].append(b)

        # Context: turn_idx == 0 OR role == "context" per ADR 001 §0.
        context_spans: list[tuple[int, int]] = []
        for turn_idx, spans in turn_spans.items():
            for b in spans:
                if b.turn_idx == 0 or b.role == "context":
                    cs = max(0, min(b.start_kv, k_len))
                    ce = max(cs, min(b.end_kv, k_len))
                    if ce > cs:
                        context_spans.append((cs, ce))

        # Conversational turn ids = everything non-context.
        conv_turn_ids = sorted(
            tid for tid, spans in turn_spans.items()
            if tid != 0 and any(b.role != "context" for b in spans)
        )
        if not conv_turn_ids:
            return {}, context_spans

        num_turns = len(conv_turn_ids)
        context_len = sum(ce - cs for cs, ce in context_spans)
        available_budget = max(0, self.global_budget - context_len)
        base_floor = available_budget / max(1, num_turns)
        current_turn = max(conv_turn_ids)

        alpha_floor = float(self.alphas.get("floor", 0.0))
        alpha_anchor = float(self.alphas.get("anchor", 0.0))
        alpha_loyalty = float(self.alphas.get("loyalty", 0.0))

        # Per-turn anchor/loyalty counts. One CPU-side pass to avoid the
        # host<->device ping-pong the old dict-based implementation did.
        anchor_counts: dict[int, int] = {}
        loyalty_counts: dict[int, int] = {}
        if alpha_anchor > 0.0 and "anchor" in self.policies:
            anchor_cpu = (self.policies["anchor"].compute_weights(k_len) > 0).cpu()
            for tid in conv_turn_ids:
                total = 0
                for b in turn_spans[tid]:
                    if b.role == "context":
                        continue
                    s = max(0, min(b.start_kv, k_len))
                    e = max(s, min(b.end_kv, k_len))
                    if e > s:
                        total += int(anchor_cpu[s:e].sum().item())
                anchor_counts[tid] = total
        if alpha_loyalty > 0.0 and "loyalty" in self.policies:
            loyalty_cpu = (self.policies["loyalty"].compute_weights(k_len) > 0).cpu()
            for tid in conv_turn_ids:
                total = 0
                for b in turn_spans[tid]:
                    if b.role == "context":
                        continue
                    s = max(0, min(b.start_kv, k_len))
                    e = max(s, min(b.end_kv, k_len))
                    if e > s:
                        total += int(loyalty_cpu[s:e].sum().item())
                loyalty_counts[tid] = total

        turn_budgets: dict[int, tuple[int, int, int]] = {}
        for tid in conv_turn_ids:
            spans = [b for b in turn_spans[tid] if b.role != "context"]
            if not spans:
                continue
            raw_start = min(b.start_kv for b in spans)
            raw_end = max(b.end_kv for b in spans)
            span_start = max(0, min(raw_start, k_len))
            span_end = max(span_start, min(raw_end, k_len))
            turn_len = span_end - span_start
            if turn_len <= 0:
                continue

            # Retained-by-aux. anchor_counts/loyalty_counts already
            # respect alpha==0 via the gating above (they stay empty).
            # We use sum (not union) because in practice anchor and
            # loyalty rarely mark the same position and a slight
            # over-budget is harmless (less compression, same correctness).
            retained = anchor_counts.get(tid, 0) + loyalty_counts.get(tid, 0)

            decay = math.exp(-self.per_turn_gamma * max(0, current_turn - tid))
            floor_i = alpha_floor * base_floor * decay

            budget = max(retained, int(round(floor_i)))
            budget = min(budget, turn_len)
            turn_budgets[tid] = (span_start, span_end, int(budget))

        # Global-budget cap: if per-turn budgets + context exceed
        # ``global_budget``, scale conversational budgets down
        # proportionally so ``total_kept <= global_budget``. Rare under
        # the formula's natural decay (sum_i exp(-gamma*(T-i)) tops out
        # around 6-7 for 10 turns) but can happen when anchor+loyalty
        # retained positions dominate a specific turn.
        total_conv = sum(b for _, _, b in turn_budgets.values())
        max_conv = max(0, self.global_budget - context_len)
        if total_conv > max_conv and total_conv > 0:
            scale = max_conv / total_conv
            for tid in list(turn_budgets):
                ts, te, b = turn_budgets[tid]
                new_b = max(0, int(b * scale))  # allow 0 so the cap is strict
                new_b = min(new_b, te - ts)
                turn_budgets[tid] = (ts, te, new_b)

        return turn_budgets, context_spans

    def _remap_all_state(self, kept: torch.Tensor) -> None:
        """Remap every policy's turn boundaries and loyalty counts to the
        new (post-compression) cache positions described by ``kept``.

        Parameters
        ----------
        kept : torch.Tensor
            1-D CPU int64 tensor of length ``n_kept`` where
            ``kept[new_pos] = old_pos``, sorted ascending.  Produced by
            taking batch 0 / head 0 of the ``all_kept`` tensor returned by
            :meth:`_compress_impl` for a representative layer.

        Both remaps are computed for ALL policies before any policy's state
        is mutated, so a mid-loop exception cannot leave some policies
        remapped and others not.
        """
        # Compute new boundaries for every policy first (pure, no mutation).
        remapped: list[tuple] = []
        for policy in self.policies.values():
            new_boundaries = policy._compute_remapped_boundaries(kept)
            new_loyalty = (
                policy._compute_remapped_loyalty(kept)
                if hasattr(policy, "_compute_remapped_loyalty")
                else None
            )
            remapped.append((policy, new_boundaries, new_loyalty))
        # Apply all updates atomically.
        for policy, new_boundaries, new_loyalty in remapped:
            policy.turn_boundaries = new_boundaries
            if new_loyalty is not None:
                policy._loyalty_counts = new_loyalty

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

        Under per-turn budgeting the ``target`` replaces ``global_budget``
        for this call, so each turn's ``base_floor`` is recomputed
        against the tighter target. A sentinel non-zero
        ``compression_ratio`` is set transiently so :meth:`compress`'s
        early ``ratio == 0`` short-circuit does not trip; the actual
        keep-count per turn is budget-driven, not ratio-driven.
        """
        effective_target = self.global_budget if target is None else target
        assert effective_target >= 0, f"target must be non-negative, got {effective_target}"
        current_len = cache.get_seq_length()
        if current_len <= effective_target:
            return

        orig_ratio = self.base_press.compression_ratio
        orig_budget = self.global_budget
        # A positive sentinel keeps compress() past its ratio==0 short-circuit.
        # The actual keep-count comes from _compute_turn_budgets(target).
        self.base_press.compression_ratio = max(orig_ratio, 0.5)
        self.global_budget = effective_target
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            # canonical_kept: kept indices from the first successfully compressed
            # layer (batch 0, head 0), used after the loop to remap policy state.
            # All layers share the same turn-budget structure so one layer
            # provides an accurate positional skeleton for boundary remapping.
            canonical_kept: Optional[torch.Tensor] = None
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
                new_keys, new_values, layer_kept = self._compress_impl(
                    module, hidden_states, keys, values, None, kwargs
                )
                self._write_layer(cache, layer_idx, new_keys, new_values)
                if canonical_kept is None and layer_kept is not None:
                    canonical_kept = layer_kept[0, 0].cpu()
            cache_len_after = cache.get_seq_length()
            logger.info(
                "Applied global compression: cache=%s->%s target=%s budget=%s",
                current_len,
                cache_len_after,
                effective_target,
                orig_budget,
            )
            # Remap all policy state so turn_boundaries and loyalty counts
            # reference new (post-eviction) cache positions rather than the
            # stale old ones.
            if canonical_kept is not None:
                self._remap_all_state(canonical_kept)
        finally:
            self.base_press.compression_ratio = orig_ratio
            self.global_budget = orig_budget

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
    def suspend_hooks(self) -> Generator:
        """Temporarily ignore forwards from sidecar model calls.

        Live-loop benchmarks use the same model to simulate verbal feedback
        outside the measured conversation cache. Those forwards must not
        refresh scorer buffers or loyalty state.
        """
        old = self._suspend_hooks
        self._suspend_hooks = True
        try:
            yield
        finally:
            self._suspend_hooks = old

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
