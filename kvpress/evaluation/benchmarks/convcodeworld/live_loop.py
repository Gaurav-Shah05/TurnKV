# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvCodeWorld live-loop benchmark runner with KV cache carry-over."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, FineGrainedFP8Config

_ROOT_EVAL = Path(__file__).resolve().parents[2]
_KV_ROOT = _ROOT_EVAL.parent
for _path in (str(_ROOT_EVAL), str(_KV_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluate_registry import PRESS_REGISTRY, SCORER_REGISTRY  # noqa: E402
from kvpress import (  # noqa: E402
    ComposedPress,
    DMSPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    LoyaltyPress,
    RoleBoundaryAnchorPress,
    ScorerPress,
    ThinKPress,
    TurnFloorPress,
    TurnAwareGlobalPress,
)
from kvpress.presses.answer_suffix_decoding_press import AnswerSuffixDecodingPress  # noqa: E402
from kvpress.presses.base_press import BasePress  # noqa: E402

from benchmarks.convcodeworld.executor import (  # noqa: E402
    PASSED_ALL_TEST_RUNS,
    build_feedback,
    extract_code,
    run_candidate,
    task_get,
    trim_feedback,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_STOP_SEQUENCES = ("\n```", "\n### Feedback", "\n### Iteration")


@dataclass
class ConvCodeWorldLiveConfig:
    model: str = DEFAULT_MODEL
    press_name: str = "snapkv"
    compression_ratio: float = 0.5
    key_channel_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    snapkv_window_size: Optional[int] = None
    snapkv_kernel_size: Optional[int] = None
    streaming_llm_n_sink: Optional[int] = None
    expected_attention_n_future_positions: Optional[int] = None
    expected_attention_n_sink: Optional[int] = None
    expected_attention_use_covariance: Optional[bool] = None
    expected_attention_use_vnorm: Optional[bool] = None
    expected_attention_epsilon: Optional[float] = None
    alpha_floor: Optional[float] = None
    alpha_anchor: Optional[float] = None
    alpha_loyalty: Optional[float] = None
    anchor_beta: Optional[float] = None
    floor_gamma: Optional[float] = None
    loyalty_top_p: Optional[float] = None
    alpha_floor_len: Optional[float] = None
    min_floor_tokens: Optional[int] = None
    feedback_config: str = "CF_EF_UNIT_SNF"
    auto_feedback_options: bool = True
    include_compilation_feedback: bool = True
    include_execution_feedback: bool = True
    include_verbal_feedback: bool = True
    user_expertise: str = "novice"
    max_turns: int = 10
    max_new_tokens: int = 1024
    verbal_feedback_max_new_tokens: int = 256
    num_eval_examples: int = 1
    fraction: float = 1.0
    task_ids: Optional[str] = None
    shuffle: bool = False
    seed: int = 42
    global_budget: int = 4500
    local_budget: int = 4096
    decode_compression_interval: int = 128
    decode_hidden_states_buffer_size: int = 256
    dataset_name: str = "bigcode/bigcodebench"
    bigcodebench_split: str = "v0.1.0_hf"
    restrict_to_convcodeworld_tasks: bool = True
    output_dir: str = "./results_convcodeworld_live"
    log_level: str = "INFO"
    fp8: bool = False
    model_kwargs: Optional[dict[str, Any]] = None
    max_input_tokens: Optional[int] = None
    early_stop_on_pass: bool = True
    executor_timeout_s: int = 30
    executor_memory_mb: int = 1024
    network_isolation: str = "auto"
    cot: bool = True


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT_EVAL.parent, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def infer_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        d = getattr(model, "device", None)
        if d is not None and getattr(d, "type", None) != "meta":
            return d
    except Exception:
        pass
    return next(model.parameters()).device


def _target_from_ratio(global_budget: int, compression_ratio: float) -> int:
    keep_rate = max(0.0, min(1.0, 1.0 - float(compression_ratio)))
    return max(1, int(round(global_budget * keep_rate)))


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _apply_feedback_config(cfg: ConvCodeWorldLiveConfig) -> None:
    if not cfg.auto_feedback_options:
        return
    name = cfg.feedback_config
    cfg.include_compilation_feedback = name.startswith("CF")
    cfg.include_execution_feedback = "_EF_" in name or name.endswith("_EF")
    cfg.include_verbal_feedback = name.endswith("_SNF") or name.endswith("_SEF") or name == "CF_SEF"
    if name.endswith("_SEF") or name == "CF_SEF":
        cfg.user_expertise = "expert"
    elif name.endswith("_SNF"):
        cfg.user_expertise = "novice"


def _iter_press_components(press: BasePress | None) -> Iterable[BasePress]:
    seen: set[int] = set()
    stack: list[Any] = [press]
    while stack:
        component = stack.pop()
        if component is None or id(component) in seen:
            continue
        seen.add(id(component))
        yield component
        for attr in ("base_press", "press"):
            child = getattr(component, attr, None)
            if child is not None:
                stack.append(child)
        children = getattr(component, "presses", None)
        if isinstance(children, (list, tuple)):
            stack.extend(children)


def _has_dataclass_field(obj: Any, name: str) -> bool:
    if not is_dataclass(obj):
        return False
    return any(field.name == name for field in fields(obj))


def _set_press_field(press: BasePress, field_name: str, value: Any, *, class_names: set[str]) -> None:
    if value is None:
        return
    for component in _iter_press_components(press):
        if component.__class__.__name__ not in class_names:
            continue
        if _has_dataclass_field(component, field_name):
            setattr(component, field_name, value)


def _apply_press_hyperparameters(press: BasePress, cfg: ConvCodeWorldLiveConfig) -> None:
    _set_press_field(
        press,
        "window_size",
        cfg.snapkv_window_size,
        class_names={"SnapKVPress", "PyramidKVPress"},
    )
    _set_press_field(
        press,
        "kernel_size",
        cfg.snapkv_kernel_size,
        class_names={"SnapKVPress", "PyramidKVPress"},
    )
    _set_press_field(
        press,
        "n_sink",
        cfg.streaming_llm_n_sink,
        class_names={"StreamingLLMPress"},
    )
    _set_press_field(
        press,
        "n_future_positions",
        cfg.expected_attention_n_future_positions,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(
        press,
        "n_sink",
        cfg.expected_attention_n_sink,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(
        press,
        "use_covariance",
        cfg.expected_attention_use_covariance,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(
        press,
        "use_vnorm",
        cfg.expected_attention_use_vnorm,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(
        press,
        "epsilon",
        cfg.expected_attention_epsilon,
        class_names={"ExpectedAttentionPress"},
    )


def _has_turn_aware_overrides(cfg: ConvCodeWorldLiveConfig) -> bool:
    return any(
        getattr(cfg, name) is not None
        for name in (
            "alpha_floor",
            "alpha_anchor",
            "alpha_loyalty",
            "anchor_beta",
            "floor_gamma",
            "loyalty_top_p",
            "alpha_floor_len",
            "min_floor_tokens",
        )
    )


def _policy_requested(cfg: ConvCodeWorldLiveConfig, name: str) -> bool:
    names = {
        "floor": ("alpha_floor", "floor_gamma", "alpha_floor_len", "min_floor_tokens"),
        "anchor": ("alpha_anchor", "anchor_beta"),
        "loyalty": ("alpha_loyalty", "loyalty_top_p"),
    }[name]
    return any(getattr(cfg, field_name) is not None for field_name in names)


def _validate_turn_aware_overrides(cfg: ConvCodeWorldLiveConfig) -> None:
    if cfg.anchor_beta is not None and not 0 <= cfg.anchor_beta <= 1:
        raise ValueError(f"anchor_beta must be in [0, 1], got {cfg.anchor_beta}")
    if cfg.floor_gamma is not None and not 0 < cfg.floor_gamma <= 1:
        raise ValueError(f"floor_gamma must be in (0, 1], got {cfg.floor_gamma}")
    if cfg.loyalty_top_p is not None and not 0 < cfg.loyalty_top_p <= 1:
        raise ValueError(f"loyalty_top_p must be in (0, 1], got {cfg.loyalty_top_p}")
    if cfg.alpha_floor_len is not None and cfg.alpha_floor_len < 0:
        raise ValueError(f"alpha_floor_len must be non-negative, got {cfg.alpha_floor_len}")
    if cfg.min_floor_tokens is not None and cfg.min_floor_tokens < 0:
        raise ValueError(f"min_floor_tokens must be non-negative, got {cfg.min_floor_tokens}")


def _configure_turn_aware_press(
    press: TurnAwareGlobalPress,
    cfg: ConvCodeWorldLiveConfig,
    *,
    create_missing: bool = False,
) -> None:
    _validate_turn_aware_overrides(cfg)
    press.global_budget = cfg.global_budget

    if create_missing and _policy_requested(cfg, "floor") and "floor" not in press.policies:
        press.policies["floor"] = TurnFloorPress(global_budget=cfg.global_budget)
        press.alphas.setdefault("floor", 0.0)
    if create_missing and _policy_requested(cfg, "anchor") and "anchor" not in press.policies:
        press.policies["anchor"] = RoleBoundaryAnchorPress()
        press.alphas.setdefault("anchor", 0.0)
    if create_missing and _policy_requested(cfg, "loyalty") and "loyalty" not in press.policies:
        press.policies["loyalty"] = LoyaltyPress()
        press.alphas.setdefault("loyalty", 0.0)

    if cfg.alpha_floor is not None:
        press.alphas["floor"] = cfg.alpha_floor
    if cfg.alpha_anchor is not None:
        press.alphas["anchor"] = cfg.alpha_anchor
    if cfg.alpha_loyalty is not None:
        press.alphas["loyalty"] = cfg.alpha_loyalty

    floor = press.policies.get("floor")
    if isinstance(floor, TurnFloorPress):
        floor.global_budget = cfg.global_budget
        if cfg.floor_gamma is not None:
            floor.gamma = cfg.floor_gamma
        if cfg.alpha_floor_len is not None:
            floor.alpha_floor_len = cfg.alpha_floor_len
        if cfg.min_floor_tokens is not None:
            floor.min_floor_tokens = cfg.min_floor_tokens

    anchor = press.policies.get("anchor")
    if isinstance(anchor, RoleBoundaryAnchorPress) and cfg.anchor_beta is not None:
        anchor.beta = cfg.anchor_beta

    loyalty = press.policies.get("loyalty")
    if isinstance(loyalty, LoyaltyPress) and cfg.loyalty_top_p is not None:
        loyalty.top_p = cfg.loyalty_top_p


def _setup_press(cfg: ConvCodeWorldLiveConfig) -> BasePress | None:
    if cfg.press_name == "no_press":
        return None
    if cfg.press_name not in PRESS_REGISTRY:
        raise ValueError(f"Unknown press_name '{cfg.press_name}'. See PRESS_REGISTRY in evaluate_registry.py.")

    if cfg.press_name == "expected_attention":
        press = ExpectedAttentionPress(epsilon=1e-2)
    else:
        press = copy.deepcopy(PRESS_REGISTRY[cfg.press_name])
    if press is None:
        return None

    if isinstance(press, DuoAttentionPress):
        press.head_compression_ratio = cfg.compression_ratio
    elif isinstance(press, DMSPress):
        if cfg.threshold is None:
            raise ValueError("threshold must be set for DMSPress")
        press.threshold = cfg.threshold
    elif isinstance(press, TurnAwareGlobalPress):
        if hasattr(press.base_press, "compression_ratio"):
            press.base_press.compression_ratio = cfg.compression_ratio
    elif isinstance(press, ComposedPress):
        for child in press.presses:
            if isinstance(child, ThinKPress):
                if cfg.key_channel_compression_ratio is None:
                    raise ValueError("key_channel_compression_ratio must be set for ThinKPress")
                child.key_channel_compression_ratio = cfg.key_channel_compression_ratio
            elif hasattr(child, "compression_ratio"):
                child.compression_ratio = cfg.compression_ratio
    elif isinstance(press, ThinKPress):
        if cfg.key_channel_compression_ratio is None:
            raise ValueError("key_channel_compression_ratio must be set for ThinKPress")
        press.key_channel_compression_ratio = cfg.key_channel_compression_ratio
    elif hasattr(press, "compression_ratio"):
        press.compression_ratio = cfg.compression_ratio

    _apply_press_hyperparameters(press, cfg)
    return press


def _as_global_press(press: BasePress | None, cfg: ConvCodeWorldLiveConfig) -> TurnAwareGlobalPress | None:
    if press is None:
        return None
    if isinstance(press, TurnAwareGlobalPress):
        _configure_turn_aware_press(press, cfg, create_missing=True)
        return press
    if isinstance(press, ScorerPress):
        global_press = TurnAwareGlobalPress(base_press=press, global_budget=cfg.global_budget, policies={}, alphas={})
        if _has_turn_aware_overrides(cfg):
            _configure_turn_aware_press(global_press, cfg, create_missing=True)
        return global_press
    raise TypeError(
        f"ConvCodeWorld live-loop global compression requires a ScorerPress-compatible press, "
        f"got {type(press).__name__}. Use snapkv, streaming_llm, expected_attention, knorm, or no_press."
    )


def _decode_scorer(press: TurnAwareGlobalPress | None) -> ScorerPress | None:
    if press is None:
        return None
    return press.base_press


def _load_dataset_split(name: str, split: str):
    try:
        return load_dataset(name, split=split)
    except Exception:
        return load_dataset(name)[split]


def _convcodeworld_task_ids(feedback_config: str) -> set[str]:
    ds = load_dataset("ConvCodeWorld/convcodebench", split="train")
    row = ds[0]
    cfg = row.get(feedback_config)
    if not cfg or "ITER=1" not in cfg:
        return set()
    return set(str(x) for x in cfg["ITER=1"]["task_id"])


def _parse_task_ids(value: str | None) -> set[str] | None:
    if value is None or value.strip() == "":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def _slug_value(value: Any) -> str:
    return str(value).replace("/", "--").replace(".", "p").replace("-", "m")


def _load_tasks(cfg: ConvCodeWorldLiveConfig) -> list[dict[str, Any]]:
    ds = _load_dataset_split(cfg.dataset_name, cfg.bigcodebench_split)
    rows = [dict(row) for row in ds]

    requested = _parse_task_ids(cfg.task_ids)
    if requested:
        rows = [row for row in rows if str(row.get("task_id")) in requested]
    elif cfg.restrict_to_convcodeworld_tasks:
        try:
            allowed = _convcodeworld_task_ids(cfg.feedback_config)
            if allowed:
                rows = [row for row in rows if str(row.get("task_id")) in allowed]
        except Exception as exc:
            logger.warning("Could not load ConvCodeWorld task ids; using all BigCodeBench tasks: %s", exc)

    if cfg.fraction < 1.0:
        rows = random.Random(cfg.seed).sample(rows, max(1, int(len(rows) * cfg.fraction)))
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(rows)
    if cfg.num_eval_examples > 0:
        rows = rows[: cfg.num_eval_examples]
    return rows


def _results_dir(cfg: ConvCodeWorldLiveConfig) -> Path:
    parts = [
        "live",
        cfg.feedback_config,
        cfg.model.replace("/", "--"),
        cfg.press_name,
        f"{cfg.compression_ratio:.2f}",
    ]
    turn_bits = []
    for attr in (
        "alpha_floor",
        "alpha_anchor",
        "alpha_loyalty",
        "anchor_beta",
        "floor_gamma",
        "loyalty_top_p",
        "alpha_floor_len",
        "min_floor_tokens",
    ):
        value = getattr(cfg, attr)
        if value is not None:
            turn_bits.append(f"{attr}-{_slug_value(value)}")
    if turn_bits:
        parts.append("turnaware_" + "_".join(turn_bits))
    if cfg.fraction < 1.0:
        parts.append(f"frac{_slug_value(cfg.fraction)}")
    parts.append(f"n{cfg.num_eval_examples}")
    name = "__".join(parts)
    out = Path(cfg.output_dir) / name
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        return out
    i = 1
    while (out / str(i)).exists():
        i += 1
    out = out / str(i)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _language_model(model: AutoModelForCausalLM):
    return model.model.language_model if hasattr(model.model, "language_model") else model.model


def _encode(tokenizer, text: str, max_tokens: int | None = None) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_tokens is None or len(ids) <= max_tokens:
        return ids
    half = max_tokens // 2
    return ids[:half] + ids[-half:]


def _model_forward(model, **kwargs):
    try:
        return model(**kwargs, num_logits_to_keep=1)
    except TypeError:
        return model(**kwargs)


@torch.inference_mode()
def _prefill_text(model, tokenizer, cache: DynamicCache, text: str, max_tokens: int | None = None) -> None:
    ids = _encode(tokenizer, text, max_tokens=max_tokens)
    if not ids:
        return
    device = infer_device(model)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    context_len = cache.get_seq_length()
    position_ids = torch.arange(context_len, context_len + input_ids.shape[1], device=device).unsqueeze(0)
    _language_model(model)(
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=position_ids,
        use_cache=True,
    )


@torch.inference_mode()
def _generate_after_prompt(
    model,
    tokenizer,
    cache: DynamicCache,
    prompt: str,
    *,
    max_new_tokens: int,
    decode_press: AnswerSuffixDecodingPress | None,
    stop_sequences: Iterable[str] = DEFAULT_STOP_SEQUENCES,
) -> str:
    device = infer_device(model)
    prompt_ids = _encode(tokenizer, prompt)
    if not prompt_ids:
        return ""
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    context_len = cache.get_seq_length()
    position_ids = torch.arange(context_len, context_len + input_ids.shape[1], device=device).unsqueeze(0)
    outputs = _model_forward(
        model,
        input_ids=input_ids,
        past_key_values=cache,
        position_ids=position_ids,
        use_cache=True,
    )

    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    eos_ids = {int(x) for x in eos_ids if x is not None}

    generated: list[torch.Tensor] = []
    next_token = outputs.logits[0, -1].argmax()

    ctx = decode_press(model) if decode_press is not None else torch.inference_mode()
    with ctx:
        for _ in range(max_new_tokens):
            generated.append(next_token)
            token_id = int(next_token.item())
            token_tensor = next_token.reshape(1, 1)
            pos = torch.tensor([[cache.get_seq_length()]], dtype=torch.long, device=device)
            outputs = _model_forward(
                model,
                input_ids=token_tensor,
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
            )
            text = str(tokenizer.decode(torch.stack(generated), skip_special_tokens=True))
            if token_id in eos_ids or any(seq in text for seq in stop_sequences):
                break
            next_token = outputs.logits[0, -1].argmax()
        if decode_press is not None:
            decode_press.finalize_if_needed(model, cache)

    return str(tokenizer.decode(torch.stack(generated), skip_special_tokens=True)) if generated else ""


def _initial_context(task: dict[str, Any], cfg: ConvCodeWorldLiveConfig, tokenizer) -> str:
    prompt = (
        task_get(task, "instruct_prompt")
        or task_get(task, "complete_prompt")
        or task_get(task, "prompt")
        or task_get(task, "code_prompt")
        or ""
    )
    cot = "\nThink through the feedback before writing code.\n" if cfg.cot else ""
    bos = tokenizer.bos_token or ""
    return (
        f"{bos}You are solving a Python programming task over multiple refinement turns.\n"
        "Return complete Python code when asked for a revision. Do not include explanations outside the code block.\n"
        f"{cot}\n### Task\n{prompt}\n"
    )


def _revision_prompt(iteration: int, feedback: str) -> str:
    if iteration == 1:
        body = "No previous candidate exists. Write the initial solution."
    else:
        body = feedback or "No new feedback was provided. Keep the solution correct and complete."
    return (
        f"\n### Iteration {iteration}\n"
        f"{body}\n\n"
        "Return only the revised complete Python code.\n"
        "```python\n"
    )


def _clip(text: str, max_chars: int = 6000) -> str:
    return trim_feedback(str(text or ""), max_chars=max_chars)


def _simulator_prompt(task: dict[str, Any], code: str, result, cfg: ConvCodeWorldLiveConfig) -> str:
    task_prompt = (
        task_get(task, "instruct_prompt")
        or task_get(task, "complete_prompt")
        or task_get(task, "prompt")
        or task_get(task, "code_prompt")
        or ""
    )
    expertise = cfg.user_expertise.lower()
    if expertise == "expert":
        reference = str(task_get(task, "code_prompt", "") or "") + str(task_get(task, "canonical_solution", "") or "")
        reference_section = (
            "\n### Private Reference Solution\n"
            f"{_clip(reference)}\n"
            "Use this only to identify the issue. Do not quote or reveal the reference solution.\n"
        )
        audience = "an expert developer"
    else:
        reference_section = ""
        audience = "a novice user"

    return (
        "You are a ConvCodeWorld feedback simulator.\n"
        f"Act as {audience} reviewing the previous Python solution.\n"
        "Return concise verbal feedback for the next coding turn.\n"
        "Do not write revised code. Do not include chain-of-thought. Do not mention hidden tests.\n"
        "\n### Task\n"
        f"{_clip(task_prompt)}\n"
        "\n### Previous Code\n"
        f"```python\n{_clip(code)}\n```\n"
        "\n### Compilation Feedback\n"
        f"{_clip(result.compilation_feedback)}\n"
        "\n### Execution Feedback\n"
        f"{_clip(result.execution_feedback)}\n"
        f"{reference_section}"
        "\n### Verbal Feedback\n"
    )


def _clean_simulator_feedback(text: str) -> str:
    text = str(text or "").strip()
    for marker in ("###", "```", "\n\n---"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text or "Revise the code using the compilation and execution feedback above."


def _simulate_verbal_feedback(model, tokenizer, task: dict[str, Any], code: str, result, cfg: ConvCodeWorldLiveConfig):
    if result.passed and cfg.early_stop_on_pass:
        return "The previous code passed the available tests. Stop the live loop."
    prompt = _simulator_prompt(task, code, result, cfg)
    raw = _generate_after_prompt(
        model,
        tokenizer,
        DynamicCache(),
        prompt,
        max_new_tokens=cfg.verbal_feedback_max_new_tokens,
        decode_press=None,
        stop_sequences=("\n###", "```", "\n\n---"),
    )
    return _clean_simulator_feedback(raw)


def _maybe_global_compress(model, cache: DynamicCache, press: TurnAwareGlobalPress | None, target: int) -> None:
    if press is not None and cache.get_seq_length() > press.global_budget:
        try:
            with torch.no_grad():
                press.run_global_compression(model, cache, target=target)
        except AssertionError as exc:
            logger.warning("Skipping global compression because the scorer had insufficient query context: %s", exc)


def _append_after_pass_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: ConvCodeWorldLiveConfig,
    task: dict[str, Any],
    from_iteration: int,
    code: str,
    cache_len: int,
) -> None:
    for iteration in range(from_iteration, cfg.max_turns + 1):
        rows.append(
            {
                "session_id": f"{cfg.feedback_config}/{task_get(task, 'task_id')}",
                "task_id": task_get(task, "task_id"),
                "feedback_config": cfg.feedback_config,
                "iteration": iteration,
                "predicted_answer": code,
                "generated_code": code,
                "passed": True,
                "status": "skipped_after_pass",
                "compilation_feedback": "",
                "execution_feedback": PASSED_ALL_TEST_RUNS,
                "verbal_feedback": "The previous code passed the available tests. Stop the live loop.",
                "feedback": "Skipped because an earlier live-loop iteration passed.",
                "cache_len_before_global": cache_len,
                "cache_len_after_global": cache_len,
                "skipped_after_pass": True,
                "metric_excluded": True,
            }
        )


class ConvCodeWorldLiveRunner:
    def __init__(self, cfg: ConvCodeWorldLiveConfig):
        self.cfg = cfg
        _apply_feedback_config(self.cfg)
        _setup_logging(self.cfg.log_level)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.output_dir = _results_dir(self.cfg)
        self.global_target = _target_from_ratio(self.cfg.global_budget, self.cfg.compression_ratio)

    def setup_model(self):
        model_kwargs = dict(self.cfg.model_kwargs or {})
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        if self.cfg.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
        try:
            import flash_attn  # noqa: F401

            model_kwargs.setdefault("attn_implementation", "flash_attention_2")
        except ImportError:
            pass
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model,
            trust_remote_code=True,
            device_map="auto",
            **model_kwargs,
        )
        model.eval()
        return model, tokenizer

    def run(self) -> None:
        tasks = _load_tasks(self.cfg)
        model, tokenizer = self.setup_model()
        base_press = _setup_press(self.cfg)
        global_press = _as_global_press(base_press, self.cfg)
        scorer = _decode_scorer(global_press)

        rows: list[dict[str, Any]] = []
        for task in tqdm(tasks, desc="ConvCodeWorld live"):
            rows.extend(self._run_task(model, tokenizer, task, global_press, scorer))
            torch.cuda.empty_cache()

        df = pd.DataFrame(rows)
        predictions_jsonl = self.output_dir / "predictions.jsonl"
        predictions_csv = self.output_dir / "predictions.csv"
        metrics_path = self.output_dir / "metrics.json"
        config_path = self.output_dir / "config.yaml"

        df.to_json(predictions_jsonl, orient="records", lines=True)
        df.to_csv(predictions_csv, index=False)
        metrics = SCORER_REGISTRY["convcodeworld"](df)
        metrics["git_revision"] = _git_revision()
        metrics["config"] = asdict(self.cfg)
        metrics["global_target"] = self.global_target
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        config_path.write_text(yaml.dump(asdict(self.cfg), sort_keys=False), encoding="utf-8")
        logger.info("Wrote %s, %s, and %s", predictions_jsonl, predictions_csv, metrics_path)
        logger.info("Metrics: %s", json.dumps(metrics, indent=2))

    def _run_task(
        self,
        model,
        tokenizer,
        task: dict[str, Any],
        global_press: TurnAwareGlobalPress | None,
        scorer: ScorerPress | None,
    ) -> list[dict[str, Any]]:
        cache = DynamicCache()
        rows: list[dict[str, Any]] = []
        session_id = f"{self.cfg.feedback_config}/{task_get(task, 'task_id')}"
        initial_context = _initial_context(task, self.cfg, tokenizer)
        feedback = ""

        context_manager = global_press(model) if global_press is not None else torch.inference_mode()
        with context_manager:
            if global_press is not None:
                global_press.on_turn_start(0, "context", cache.get_seq_length())
            context_start = cache.get_seq_length()
            _prefill_text(model, tokenizer, cache, initial_context, max_tokens=self.cfg.max_input_tokens)
            if global_press is not None:
                global_press.on_turn_end(0, "context", context_start, cache.get_seq_length())

            for iteration in range(1, self.cfg.max_turns + 1):
                prompt = _revision_prompt(iteration, feedback)
                if global_press is not None:
                    global_press.on_turn_start(iteration, "user", cache.get_seq_length())
                user_start = cache.get_seq_length()

                answer_start = cache.get_seq_length() + len(_encode(tokenizer, prompt))
                decode_press = None
                if scorer is not None:
                    decode_press = AnswerSuffixDecodingPress(
                        base_press=scorer,
                        answer_start_seq_len=answer_start,
                        compression_interval=self.cfg.decode_compression_interval,
                        target_size=self.cfg.local_budget,
                        hidden_states_buffer_size=self.cfg.decode_hidden_states_buffer_size,
                    )
                generated = _generate_after_prompt(
                    model,
                    tokenizer,
                    cache,
                    prompt,
                    max_new_tokens=self.cfg.max_new_tokens,
                    decode_press=decode_press,
                )
                user_end = answer_start
                if global_press is not None:
                    global_press.on_turn_end(iteration, "user", user_start, user_end)
                    global_press.on_turn_end(iteration, "assistant", user_end, cache.get_seq_length())

                code = extract_code(generated)
                result = run_candidate(
                    task,
                    code,
                    timeout_s=self.cfg.executor_timeout_s,
                    memory_mb=self.cfg.executor_memory_mb,
                    network_isolation=self.cfg.network_isolation,
                    work_dir="/tmp",
                )
                cache_before = cache.get_seq_length()
                _maybe_global_compress(model, cache, global_press, self.global_target)
                cache_after = cache.get_seq_length()
                verbal_feedback = None
                if self.cfg.include_verbal_feedback:
                    if global_press is None:
                        verbal_feedback = _simulate_verbal_feedback(model, tokenizer, task, code, result, self.cfg)
                    else:
                        with global_press.suspend_hooks():
                            verbal_feedback = _simulate_verbal_feedback(model, tokenizer, task, code, result, self.cfg)
                feedback = build_feedback(
                    result,
                    include_compilation=self.cfg.include_compilation_feedback,
                    include_execution=self.cfg.include_execution_feedback,
                    include_verbal=False,
                )
                if verbal_feedback:
                    feedback = (feedback + "\n\n" if feedback else "") + "Verbal feedback:\n" + verbal_feedback
                row = {
                    "session_id": session_id,
                    "task_id": task_get(task, "task_id"),
                    "feedback_config": self.cfg.feedback_config,
                    "iteration": iteration,
                    "predicted_answer": code,
                    "generated_code": code,
                    "raw_generation": generated,
                    "passed": result.passed,
                    "status": result.status,
                    "compilation_feedback": result.compilation_feedback,
                    "execution_feedback": result.execution_feedback,
                    "verbal_feedback": verbal_feedback,
                    "feedback": feedback,
                    "cache_len_before_global": cache_before,
                    "cache_len_after_global": cache_after,
                    "skipped_after_pass": False,
                    "metric_excluded": False,
                }
                rows.append(row)

                if result.passed and self.cfg.early_stop_on_pass:
                    _append_after_pass_rows(
                        rows,
                        cfg=self.cfg,
                        task=task,
                        from_iteration=iteration + 1,
                        code=code,
                        cache_len=cache_after,
                    )
                    break

        return rows


def run(config: ConvCodeWorldLiveConfig | None = None, config_file: Optional[str] = None, **cli_overrides: Any) -> None:
    args = asdict(ConvCodeWorldLiveConfig())
    if config is not None:
        args.update(asdict(config))
    if config_file:
        p = Path(config_file)
        if p.exists():
            args.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    args.update({k: v for k, v in cli_overrides.items() if v is not None})
    env_model = os.environ.get("KV_PRESS_CONVCODEWORLD_MODEL", "").strip()
    if env_model:
        args["model"] = env_model
    cfg_kwargs = {k: v for k, v in args.items() if k in ConvCodeWorldLiveConfig.__dataclass_fields__}
    cfg = ConvCodeWorldLiveConfig(**cfg_kwargs)
    ConvCodeWorldLiveRunner(cfg).run()


def _cli_entrypoint(config_file: Optional[str] = None, **kwargs: Any) -> None:
    run(config_file=config_file, **kwargs)


if __name__ == "__main__":
    Fire(_cli_entrypoint)
