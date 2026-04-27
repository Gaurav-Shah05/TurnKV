# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Decode-only KV compression: prune **assistant-generated** tokens only.

Prefix KV (long context + prior turns + current question) is left unchanged.
Compression uses a wrapped :class:`ScorerPress` (e.g. SnapKV) on the answer suffix.
"""

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import extract_keys_and_values

logger = logging.getLogger(__name__)


@dataclass
class AnswerSuffixDecodingPress(BasePress):
    """
    Apply a scorer press only to the **answer** segment of the KV cache (decode tokens).

    Parameters
    ----------
    base_press
        Scorer press used to score/prune answer tokens (e.g. SnapKV, Knorm).
    answer_start_seq_len
        Cache sequence length **before** assistant decoding starts (after question prefill).
    compression_interval
        Every this many **single-token decode** forwards, run a compression *attempt*.
    target_size
        Compression runs only if the answer suffix length is **greater** than this; prune toward
        at most this many tokens (same semantics as :class:`DecodingPress`).
    hidden_states_buffer_size
        Recent decode hidden states to retain for scorer queries (cf. :class:`DecodingPress`).
    """

    base_press: ScorerPress
    answer_start_seq_len: int
    compression_interval: int = 16
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    _hidden_buffer: dict[int, list[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    _layer_decode_steps: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    # Last attention kwargs per layer (decode steps only) — needed for SnapKV finalize.
    _last_layer_kwargs: dict[int, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert isinstance(self.base_press, ScorerPress), "AnswerSuffixDecodingPress requires ScorerPress"
        assert self.compression_interval > 0
        assert self.target_size > 0
        assert self.answer_start_seq_len >= 0
        assert self.hidden_states_buffer_size >= 0

    def post_init_from_model(self, model: PreTrainedModel) -> None:
        self.base_press.post_init_from_model(model)

    def _find_target_compression_ratio(self, q_len: int, target_tokens: int) -> float:
        """Same goal as DecodingPress: map current length to ratio that yields ~target_tokens kept."""
        if q_len <= target_tokens:
            return 0.0
        ratio = 1.0 - (target_tokens / q_len)
        low, high = 0.0, 1.0
        for _ in range(20):
            n_kept = int(q_len * (1 - ratio))
            if n_kept == target_tokens:
                break
            if n_kept > target_tokens:
                low = ratio
                ratio = (ratio + high) / 2
            else:
                high = ratio
                ratio = (low + ratio) / 2
        return ratio

    def _compress_answer_suffix(
        self,
        module: nn.Module,
        hidden_states_cat: torch.Tensor,
        keys_suf: torch.Tensor,
        values_suf: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_len = keys_suf.shape[2]
        if k_len <= self.target_size:
            return keys_suf, values_suf

        # SnapKV needs enough query positions for its observation window.
        min_queries = getattr(self.base_press, "window_size", 64) + 1
        if isinstance(self.base_press, ScorerPress) and hidden_states_cat.shape[1] < min_queries:
            logger.debug(
                "Skipping suffix compress: hidden length %s < min_queries %s",
                hidden_states_cat.shape[1],
                min_queries,
            )
            return keys_suf, values_suf

        ratio = self._find_target_compression_ratio(k_len, self.target_size)
        prev = self.base_press.compression_ratio
        self.base_press.compression_ratio = ratio
        try:
            out = self.base_press.compress(module, hidden_states_cat, keys_suf, values_suf, attentions, kwargs)
        finally:
            self.base_press.compression_ratio = prev
        return out

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list) -> list:
        hidden_states = kwargs["hidden_states"]
        cache = kwargs["past_key_values"]
        q_len = hidden_states.shape[1]
        layer_idx = module.layer_idx

        # Multi-token forwards (long context, question prefill): do not touch cache here.
        if q_len != 1:
            return output

        # Single-token decode: update answer suffix buffers and maybe compress.
        self._hidden_buffer[layer_idx].append(hidden_states.detach().clone())
        self._layer_decode_steps[layer_idx] += 1
        self._last_layer_kwargs[layer_idx] = kwargs

        keys, values = extract_keys_and_values(cache, module.layer_idx)
        total_len = keys.shape[2]
        suffix_len = total_len - self.answer_start_seq_len
        if suffix_len <= 0:
            return output

        attentions = output[1] if len(output) > 1 and output[1] is not None else None

        should_try = self._layer_decode_steps[layer_idx] >= self.compression_interval
        if should_try:
            self._layer_decode_steps[layer_idx] = 0
            if suffix_len > self.target_size:
                self._apply_layer_suffix_compress(
                    module, cache, layer_idx, keys, values, attentions, kwargs
                )

        if self.hidden_states_buffer_size > 0:
            self._hidden_buffer[layer_idx] = self._hidden_buffer[layer_idx][-self.hidden_states_buffer_size :]

        return output

    def _apply_layer_suffix_compress(
        self,
        module: nn.Module,
        cache: Any,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> None:
        start = self.answer_start_seq_len
        cache_len_before = keys.shape[2]
        keys_pre = keys[:, :, :start, :]
        vals_pre = values[:, :, :start, :]
        keys_suf = keys[:, :, start:, :]
        vals_suf = values[:, :, start:, :]
        suffix_len_before = keys_suf.shape[2]

        buf = self._hidden_buffer[layer_idx]
        if not buf:
            return
        hidden_cat = torch.cat(buf, dim=1)

        new_k_suf, new_v_suf = self._compress_answer_suffix(
            module, hidden_cat, keys_suf, vals_suf, attentions, kwargs
        )
        new_keys = torch.cat([keys_pre, new_k_suf], dim=2)
        new_vals = torch.cat([vals_pre, new_v_suf], dim=2)
        cache_len_after = new_keys.shape[2]
        suffix_len_after = new_k_suf.shape[2]

        cache_layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(new_keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(new_vals, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)  # type: ignore[index]
            cache_layer.values = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)  # type: ignore[index]
            cache_layer.cumulative_length = new_keys.shape[2]
        else:
            cache_layer.keys = new_keys
            cache_layer.values = new_vals

        if cache_len_after != cache_len_before or suffix_len_after != suffix_len_before:
            logger.info(
                "Applied local suffix compression: layer=%s cache=%s->%s answer_suffix=%s->%s target=%s",
                layer_idx,
                cache_len_before,
                cache_len_after,
                suffix_len_before,
                suffix_len_after,
                self.target_size,
            )

        self._hidden_buffer[layer_idx] = []
        if self.hidden_states_buffer_size > 0:
            # Keep a short tail for the next scorer window (same idea as DecodingPress).
            self._hidden_buffer[layer_idx] = buf[-self.hidden_states_buffer_size :]

    def finalize_if_needed(self, model: PreTrainedModel, cache: Any) -> None:
        """
        After greedy decoding, compress the answer suffix once more if it still exceeds ``target_size``.

        Must be called while hooks are still registered (inside ``with press(model):``).
        """
        for layer_idx, layer in enumerate(language_module_layers(model)):
            module = layer.self_attn
            keys, values = extract_keys_and_values(cache, layer_idx)
            suffix_len = keys.shape[2] - self.answer_start_seq_len
            if suffix_len <= self.target_size:
                continue
            kwargs = self._last_layer_kwargs.get(layer_idx)
            if kwargs is None:
                logger.warning("Missing last decode kwargs for layer %s; skipping finalize compress", layer_idx)
                continue
            self._apply_layer_suffix_compress(module, cache, layer_idx, keys, values, None, kwargs)

    def reset(self) -> None:
        self._hidden_buffer = defaultdict(list)
        self._layer_decode_steps = defaultdict(int)
        self._last_layer_kwargs = {}

    @contextmanager
    def __call__(self, model: PreTrainedModel):
        try:
            with super().__call__(model):
                yield
        finally:
            self.reset()


def language_module_layers(model: PreTrainedModel) -> Any:
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    return language_model.layers
