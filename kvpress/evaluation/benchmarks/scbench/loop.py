# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SCDQ (shared-context, different-query) multi-turn inference with KV cache carry-over.

``kv_compression`` modes:

- ``context_prefill``: compress only the initial long-context prefill (original SCBench integration).
- ``decode_only``: do **not** compress long context or question prefills; compress **assistant decode**
  KV only (periodic every ``decode_compression_interval`` steps when above ``decode_token_limit``,
  plus a final pass), using :class:`~kvpress.presses.answer_suffix_decoding_press.AnswerSuffixDecodingPress`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Literal

import torch
from transformers import DynamicCache, PreTrainedModel, PreTrainedTokenizer

from kvpress.presses.answer_suffix_decoding_press import AnswerSuffixDecodingPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

KvCompressionMode = Literal["context_prefill", "decode_only"]


def infer_device(model: PreTrainedModel) -> torch.device:
    """Resolve device for tensors; ``device_map='auto'`` models may not expose ``model.device`` reliably."""
    try:
        d = getattr(model, "device", None)
        if d is not None and getattr(d, "type", None) != "meta":
            return d
    except Exception:
        pass
    return next(model.parameters()).device


def truncate_prompt_tokens(tokenizer: PreTrainedTokenizer, text: str, max_tokens: int | None) -> list[int]:
    """Middle-truncate a prompt to at most ``max_tokens`` (MInference-style)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens is None or max_tokens < 0 or len(ids) <= max_tokens:
        return ids
    half = max_tokens // 2
    return ids[:half] + ids[-half:]


def truncate_first_prompt(
    tokenizer: PreTrainedTokenizer, first: str | list[int], max_tokens: int | None
) -> list[int]:
    """Middle-truncate the first SCDQ prompt (string or pre-tokenized ids)."""
    if isinstance(first, str):
        return truncate_prompt_tokens(tokenizer, first, max_tokens)
    return _maybe_truncate_token_list(list(first), max_tokens)


def _maybe_truncate_token_list(ids: list[int], max_tokens: int | None) -> list[int]:
    if max_tokens is None or max_tokens < 0 or len(ids) <= max_tokens:
        return ids
    half = max_tokens // 2
    return ids[:half] + ids[-half:]


def _first_prompt_token_cap(model: PreTrainedModel, max_context_tokens: int | None) -> int:
    """
    Upper bound for turn-0 prefill length (must be a finite int — ``None`` never means unlimited).

    Long SCDQ contexts can be 100k+ tokens; without a cap, ``max_context_tokens is None`` would pass
    the full sequence to ``model.model`` and blow memory / exceed ``max_position_embeddings``.
    """
    lim = getattr(model.config, "max_position_embeddings", 8192)
    if not isinstance(lim, int) or lim > 1_000_000:
        lim = 8192
    safe = max(lim - 256, 64)
    if max_context_tokens is None:
        return min(safe, 8192)
    return max(64, min(int(max_context_tokens), safe))


def _resolve_max_new_tokens(
    max_new_tokens_cfg: int | dict[str, int],
    prompts_meta: dict[str, Any],
    answer_turn_index: int,
) -> int:
    """Resolve generation budget for the ``answer_turn_index``-th answer (0-based)."""
    if isinstance(max_new_tokens_cfg, dict):
        if "task" in prompts_meta:
            task_name = prompts_meta["task"][answer_turn_index]
            return int(max_new_tokens_cfg[task_name])
        return int(next(iter(max_new_tokens_cfg.values())))
    return int(max_new_tokens_cfg)


def _as_decode_scorer(press: BasePress) -> ScorerPress:
    if isinstance(press, ScorerPress):
        return press
    raise TypeError(
        f"decode_only mode requires a ScorerPress (e.g. snapkv, knorm), got {type(press).__name__}. "
        "AdaKVPress is not supported for answer-suffix tensor pruning."
    )


def prefill_long_context(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    cache: DynamicCache,
    press: BasePress | None,
) -> None:
    """Run a single prefill over the long-context chunk (SCDQ turn 0)."""
    device = infer_device(model)
    input_ids = input_ids.to(device)
    perform_prefill = press is not None and not isinstance(press, DecodingPress)
    ctx = press(model) if perform_prefill else contextlib.nullcontext()
    with ctx:
        model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)


def generate_assistant_greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cache: DynamicCache,
    prompt: str | list[int],
    max_new_tokens: int,
    *,
    kv_compression: KvCompressionMode = "context_prefill",
    decode_scorer: ScorerPress | None = None,
    decode_compression_interval: int = 16,
    decode_token_limit: int = 2048,
) -> str:
    """Append ``prompt`` tokens to ``cache``, then greedy-decode an assistant continuation."""
    device = infer_device(model)
    if isinstance(prompt, str):
        new_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    else:
        new_token_ids = list(prompt)
    question_ids = torch.tensor([new_token_ids], device=device, dtype=torch.long)

    context_length = cache.get_seq_length()
    position_ids = torch.arange(
        context_length, context_length + question_ids.shape[1], device=device
    ).unsqueeze(0)

    outputs = model(
        input_ids=question_ids,
        past_key_values=cache,
        position_ids=position_ids,
        use_cache=True,
        num_logits_to_keep=1,
    )

    position_ids = position_ids[:, -1:] + 1
    generated_ids = [outputs.logits[0, -1].argmax()]

    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    eos_ids = [e for e in eos_ids if e is not None]

    if kv_compression == "decode_only" and decode_scorer is not None:
        answer_start = cache.get_seq_length()
        decode_wrap = AnswerSuffixDecodingPress(
            base_press=decode_scorer,
            answer_start_seq_len=answer_start,
            compression_interval=decode_compression_interval,
            target_size=decode_token_limit,
        )
        with decode_wrap(model):
            for i in range(max_new_tokens - 1):
                out = model(
                    input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                    past_key_values=cache,
                    position_ids=position_ids + i,
                    use_cache=True,
                    num_logits_to_keep=1,
                )
                new_id = out.logits[0, -1].argmax()
                generated_ids.append(new_id)
                if new_id.item() in eos_ids:
                    break
            decode_wrap.finalize_if_needed(model, cache)
    else:
        for i in range(max_new_tokens - 1):
            out = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
                use_cache=True,
                num_logits_to_keep=1,
            )
            new_id = out.logits[0, -1].argmax()
            generated_ids.append(new_id)
            if new_id.item() in eos_ids:
                break

    return str(tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True))


def run_scdq_example(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts_data: dict[str, Any],
    press: BasePress | None,
    max_new_tokens_cfg: int | dict[str, int],
    max_context_tokens: int | None,
    *,
    kv_compression: KvCompressionMode = "decode_only",
    decode_compression_interval: int = 16,
    decode_token_limit: int = 2048,
) -> list[str]:
    """
    Run SCDQ inference for one benchmark example.

    Parameters
    ----------
    prompts_data
        Output of :func:`create_scdq_prompt` (``prompts`` list, optional per-turn ``task``).
    press
        Optional kvpress. For ``context_prefill``, applied on turn 0 only (``KVzipPress`` supported).
        For ``decode_only``, must be ``None`` or a :class:`~kvpress.presses.scorer_press.ScorerPress`
        used only during assistant decode (AdaKVPress is not supported).
    kv_compression
        ``context_prefill`` or ``decode_only`` (see module docstring).
    decode_compression_interval
        Decode steps between compression attempts (``decode_only``).
    decode_token_limit
        Answer KV must exceed this length for a compression attempt to actually prune (``decode_only``).
    max_context_tokens
        Hard cap for turn-0 prefill (middle-truncated). If ``None``, a safe cap is derived from
        ``model.config.max_position_embeddings`` (never unlimited).
    """
    prompts: list[str | list[int]] = prompts_data["prompts"]
    cache: DynamicCache = DynamicCache()

    cap = _first_prompt_token_cap(model, max_context_tokens)
    if len(prompts) > 0:
        n0 = len(tokenizer.encode(prompts[0], add_special_tokens=False)) if isinstance(prompts[0], str) else len(prompts[0])
        if n0 > cap:
            logger.warning(
                "Truncating first SCDQ prompt from %s to %s tokens (model context / max_context_tokens).",
                n0,
                cap,
            )

    first = prompts[0]
    if isinstance(first, str):
        ids = truncate_prompt_tokens(tokenizer, first, cap)
    else:
        ids = _maybe_truncate_token_list(list(first), cap)
    device = infer_device(model)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    if isinstance(press, KVzipPress):
        if kv_compression == "decode_only":
            raise ValueError("decode_only mode does not support KVzipPress on the long-context path.")
        with press(model):
            model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    else:
        prefill_press = None if kv_compression == "decode_only" else press
        prefill_long_context(model, input_ids, cache, prefill_press)

    decode_scorer: ScorerPress | None = None
    if kv_compression == "decode_only" and press is not None:
        decode_scorer = _as_decode_scorer(press)

    answers: list[str] = []
    for turn_i, prompt in enumerate(prompts[1:]):
        mtoks = _resolve_max_new_tokens(max_new_tokens_cfg, prompts_data, turn_i)
        ans = generate_assistant_greedy(
            model,
            tokenizer,
            cache,
            prompt,
            mtoks,
            kv_compression=kv_compression,
            decode_scorer=decode_scorer,
            decode_compression_interval=decode_compression_interval,
            decode_token_limit=decode_token_limit,
        )
        answers.append(ans)

    return answers
