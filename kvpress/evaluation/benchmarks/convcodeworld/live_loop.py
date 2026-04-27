# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvCodeWorld live-loop benchmark runner with KV cache carry-over."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import requests
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, DynamicCache, FineGrainedFP8Config

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
from kvpress.attention_patch import flashdecode_used_layers, reset_flashdecode_tracking  # noqa: E402
from kvpress.presses.answer_suffix_decoding_press import AnswerSuffixDecodingPress  # noqa: E402
from kvpress.presses.base_press import BasePress  # noqa: E402

from benchmarks.convcodeworld.executor import (  # noqa: E402
    PASSED_ALL_TEST_RUNS,
    build_feedback,
    extract_code,
    normalize_candidate_code,
    normalize_tokenizer_artifacts,
    run_candidate,
    task_get,
    trim_feedback,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_FEEDBACK_MODEL = "google/gemma-4-26B-A4B-it"
VLLM_TRITON_ATTN_IMPLEMENTATION = "vllm_triton"
DEFAULT_FEEDBACK_ATTN_IMPLEMENTATION = VLLM_TRITON_ATTN_IMPLEMENTATION
# Stop sequences are last-resort guards. The primary stop signal is the
# tokenizer's EOS/EOT token, which chat-template prompting causes the model
# to emit naturally at the end of an assistant turn. These literal patterns
# match observed DeepSeek-R1-Distill-Llama-8B degenerate outputs where the
# model (a) closes a code fence but keeps talking, (b) hallucinates future
# iterations in the format our old raw-text prompts used, or (c) fakes its
# own feedback transcript. Adding them costs nothing because they only fire
# when EOS/EOT did not, and they short-circuit runs that would otherwise
# burn tokens on garbage.
DEFAULT_STOP_SEQUENCES = (
    "\n```\n",
    "```\n\n",
    "\n### Iteration",
    "\n###Iteration",
    "\nExecutionFeedback:",
    "\nUserFeedback:",
    "\nCompilationFeedback:",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
    "<|eot_id|>",
)


@dataclass
class ConvCodeWorldLiveConfig:
    benchmark_mode: str = "live"
    model: str = DEFAULT_MODEL
    feedback_model: Optional[str] = DEFAULT_FEEDBACK_MODEL
    attn_implementation: Optional[str] = "flash_attention_3"
    feedback_attn_implementation: Optional[str] = DEFAULT_FEEDBACK_ATTN_IMPLEMENTATION
    feedback_vllm_port: int = 8001
    feedback_vllm_cuda_visible_devices: Optional[str] = None
    feedback_vllm_max_model_len: int = 32768
    feedback_vllm_gpu_memory_utilization: float = 0.75
    feedback_vllm_start_timeout_s: int = 1800
    full_kv_cache: bool = False
    require_flashdecode: bool = False
    error_on_kv_cache_vram_exhaustion: bool = False
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
    loyalty_update_every: Optional[int] = None
    alpha_floor_len: Optional[float] = None
    min_floor_tokens: Optional[int] = None
    feedback_config: str = "CF_EF_UNIT_SNF"
    auto_feedback_options: bool = True
    include_compilation_feedback: bool = True
    include_execution_feedback: bool = True
    include_verbal_feedback: bool = True
    user_expertise: str = "novice"
    max_turns: int = 10
    # 512 accommodates typical BigCodeBench solutions (<=~400 tokens of code)
    # without leaving a runaway budget for R1-Distill's reasoning to hallucinate
    # fake future iterations after it finishes the real code. Bump back to
    # 1024 if a task genuinely needs more tokens.
    max_new_tokens: int = 512
    code_generation_until_eos: bool = False
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
    # CoT defaults to False: DeepSeek-R1-Distill-Llama-8B (the default model)
    # has reasoning trained into its chat template's <think>...</think> block
    # already; adding an external "think through the feedback" clause to the
    # prompt causes double-reasoning that consumes max_new_tokens with
    # meta-commentary and yields broken/truncated code (see commit writeup).
    # ADR 002 §1 also specifies CoT off-by-default on every benchmark; this
    # resolves the contradiction with ADR 001 §4.
    cot: bool = False


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
        for parameter in model.parameters():
            if parameter.device.type == "cuda":
                return parameter.device
    except Exception:
        pass
    try:
        d = getattr(model, "device", None)
        if d is not None and getattr(d, "type", None) != "meta":
            return d
    except Exception:
        pass
    return next(model.parameters()).device


def _cache_tensor_devices(cache: DynamicCache) -> set[torch.device]:
    devices: set[torch.device] = set()
    for layer in getattr(cache, "layers", []):
        for attr in ("keys", "values", "_quantized_keys", "_quantized_values"):
            tensor = getattr(layer, attr, None)
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                devices.add(tensor.device)
    return devices


def _assert_cache_on_device(cache: DynamicCache, expected_device: torch.device, label: str) -> None:
    expected = torch.device(expected_device)
    devices = _cache_tensor_devices(cache)
    bad_devices = sorted(
        str(device)
        for device in devices
        if device.type != expected.type
        or (expected.type == "cuda" and device.index not in (None, expected.index))
    )
    if bad_devices:
        raise RuntimeError(
            f"{label} moved off {expected}; observed cache tensor devices: {bad_devices}."
        )


def _target_from_ratio(global_budget: int, compression_ratio: float) -> int:
    keep_rate = max(0.0, min(1.0, 1.0 - float(compression_ratio)))
    return max(1, int(round(global_budget * keep_rate)))


def _is_flash_attention_3(implementation: Optional[str]) -> bool:
    return str(implementation or "").endswith("flash_attention_3")


def _is_vllm_triton_attention(implementation: Optional[str]) -> bool:
    normalized = str(implementation or "").strip().lower().replace("-", "_")
    return normalized in {
        VLLM_TRITON_ATTN_IMPLEMENTATION,
        "vllm_triton_attn",
        "vllm_triton_attention",
        "triton_attn",
    }


def _model_uses_flash_attention_3(model) -> bool:
    return _is_flash_attention_3(getattr(model.config, "_attn_implementation", None))


def _text_config(model) -> Any:
    config = model.config
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config(decoder=True)
        except TypeError:
            return get_text_config()
    return getattr(config, "text_config", config)


def _format_num_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if unit == units[-1] or value < 1024.0:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _estimate_kv_cache_bytes(model, seq_len: int) -> int:
    cfg = _text_config(model)
    num_attention_heads = int(getattr(cfg, "num_attention_heads"))
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    num_key_value_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = int(getattr(cfg, "hidden_size") // num_attention_heads)
    dtype = getattr(model, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        dtype = next(model.parameters()).dtype
    bytes_per_element = torch.empty((), dtype=dtype).element_size()
    return 2 * num_layers * num_key_value_heads * int(head_dim) * int(seq_len) * bytes_per_element


def _assert_kv_cache_fits_available_vram(model, cache: DynamicCache, extra_tokens: int, *, label: str) -> None:
    device = infer_device(model)
    if device.type != "cuda":
        return
    projected_seq_len = cache.get_seq_length() + max(0, int(extra_tokens))
    estimated_bytes = _estimate_kv_cache_bytes(model, projected_seq_len)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if estimated_bytes <= free_bytes:
        return
    raise RuntimeError(
        f"{label} would grow the code-generation KV cache to about "
        f"{_format_num_bytes(estimated_bytes)} at seq_len={projected_seq_len}, "
        f"but only {_format_num_bytes(free_bytes)} of free VRAM remains on {device} "
        f"(total {_format_num_bytes(total_bytes)}). Refusing to continue because "
        "error_on_kv_cache_vram_exhaustion=True and full_kv_cache keeps the cache resident."
    )


def _normalize_benchmark_mode(value: str) -> str:
    mode = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "live": "live",
        "live_loop": "live",
        "liveloop": "live",
        "static": "static",
        "static_replay": "static",
    }
    if mode not in aliases:
        raise ValueError(
            f"benchmark_mode must be one of live, live_loop, static, or static_replay; got {value!r}"
        )
    return aliases[mode]


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
            "loyalty_update_every",
            "alpha_floor_len",
            "min_floor_tokens",
        )
    )


def _policy_requested(cfg: ConvCodeWorldLiveConfig, name: str) -> bool:
    names = {
        "floor": ("alpha_floor", "floor_gamma", "alpha_floor_len", "min_floor_tokens"),
        "anchor": ("alpha_anchor", "anchor_beta"),
        "loyalty": ("alpha_loyalty", "loyalty_top_p", "loyalty_update_every"),
    }[name]
    return any(getattr(cfg, field_name) is not None for field_name in names)


def _validate_turn_aware_overrides(cfg: ConvCodeWorldLiveConfig) -> None:
    if cfg.anchor_beta is not None and not 0 <= cfg.anchor_beta <= 1:
        raise ValueError(f"anchor_beta must be in [0, 1], got {cfg.anchor_beta}")
    if cfg.floor_gamma is not None and not 0 < cfg.floor_gamma <= 1:
        raise ValueError(f"floor_gamma must be in (0, 1], got {cfg.floor_gamma}")
    if cfg.loyalty_top_p is not None and not 0 < cfg.loyalty_top_p <= 1:
        raise ValueError(f"loyalty_top_p must be in (0, 1], got {cfg.loyalty_top_p}")
    if cfg.loyalty_update_every is not None and cfg.loyalty_update_every < 1:
        raise ValueError(f"loyalty_update_every must be >= 1, got {cfg.loyalty_update_every}")
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
    if isinstance(loyalty, LoyaltyPress):
        if cfg.loyalty_top_p is not None:
            loyalty.top_p = cfg.loyalty_top_p
        if cfg.loyalty_update_every is not None:
            loyalty.update_every = cfg.loyalty_update_every


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


def _label_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"pass", "passed", "true", "1", "yes"}


def _load_reference_trajectories(feedback_config: str) -> dict[str, list[dict[str, Any]]]:
    ds = load_dataset("ConvCodeWorld/convcodebench", split="train")
    row = ds[0]
    cfg = row.get(feedback_config)
    if not cfg or "ITER=1" not in cfg:
        raise ValueError(f"Feedback config {feedback_config!r} not found in ConvCodeWorld/convcodebench")

    first_iter = cfg["ITER=1"]
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for task_idx, task_id in enumerate(first_iter["task_id"]):
        turns: list[dict[str, Any]] = []
        for iteration in range(1, 11):
            it = cfg.get(f"ITER={iteration}")
            if not it or task_idx >= len(it.get("previous_code", [])):
                continue
            turns.append(
                {
                    "iteration": iteration,
                    "task_id": str(task_id),
                    "previous_code": it["previous_code"][task_idx],
                    "compilation_feedback": it["compilation_feedback"][task_idx],
                    "execution_feedback": it["execution_feedback"][task_idx],
                    "verbal_feedback": it["verbal_feedback"][task_idx],
                    "label": _label_to_bool(it["label"][task_idx]),
                }
            )
        trajectories[str(task_id)] = turns
    return trajectories


def _parse_task_ids(value: str | None) -> set[str] | None:
    """
    Accept either a comma-separated list of task IDs, or '@<path>' to load a
    JSON list of task IDs (e.g. the splits we keep under benchmarks/convcodeworld/splits/).

    Returns None when the input is empty, signalling 'no explicit override'.
    """
    if value is None or value.strip() == "":
        return None
    text = value.strip()
    if text.startswith("@"):
        path = Path(text[1:]).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"task_ids @<path> points to missing file: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list) or not all(isinstance(x, (str, int)) for x in data):
            raise ValueError(
                f"task_ids @<path> file {path} must be a JSON list of strings/ints, got {type(data).__name__}"
            )
        return {str(x).strip() for x in data if str(x).strip()}
    return {x.strip() for x in text.split(",") if x.strip()}


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
        _normalize_benchmark_mode(cfg.benchmark_mode),
        cfg.feedback_config,
        cfg.model.replace("/", "--"),
        f"fb-{(cfg.feedback_model or cfg.model).replace('/', '--')}",
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
        "loyalty_update_every",
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


class VllmTritonFeedbackClient:
    """Small OpenAI-compatible client for a local vLLM feedback server."""

    def __init__(self, model_name: str, cfg: ConvCodeWorldLiveConfig, log_dir: Path):
        self.model_name = model_name
        self.port = int(cfg.feedback_vllm_port)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_dir / "vllm_feedback_server.log"
        self._log_handle = None
        self.process: subprocess.Popen | None = None
        self._start(cfg)

    def _command(self, cfg: ConvCodeWorldLiveConfig) -> list[str]:
        vllm_cli = Path(sys.executable).with_name("vllm")
        common_args = [
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(cfg.feedback_vllm_max_model_len),
            "--gpu-memory-utilization",
            str(cfg.feedback_vllm_gpu_memory_utilization),
            "--served-model-name",
            self.model_name,
        ]
        if vllm_cli.is_file():
            return [str(vllm_cli), "serve", self.model_name, *common_args]
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            *common_args,
        ]

    def _start(self, cfg: ConvCodeWorldLiveConfig) -> None:
        env = os.environ.copy()
        env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        # Keep vLLM on its Triton unified-attention path, which covers prompt
        # processing and one-token decode in one backend.
        env.setdefault("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "0")
        # Gemma4 MoE can trigger DeepGEMM warmup in vLLM nightlies. The Modal
        # image does not install deep_gemm, so keep MoE on the Triton backend.
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
        if cfg.feedback_vllm_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cfg.feedback_vllm_cuda_visible_devices

        cmd = self._command(cfg)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        logger.info(
            "Starting vLLM Triton feedback server: %s (log: %s)",
            shlex.join(cmd),
            self.log_path,
        )
        try:
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self._wait_until_ready(cfg.feedback_vllm_start_timeout_s)
        except Exception:
            self.close()
            raise

    def _wait_until_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + max(1, int(timeout_s))
        last_error = ""
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                log_tail = self._log_tail()
                raise RuntimeError(
                    "vLLM feedback server exited before becoming ready; "
                    f"see {self.log_path}.\n{log_tail}"
                )
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code < 500:
                    logger.info("vLLM Triton feedback server is ready on %s", self.base_url)
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2)
        raise TimeoutError(
            "Timed out waiting for vLLM feedback server to become ready at "
            f"{self.base_url}; last error: {last_error}; see {self.log_path}.\n"
            f"{self._log_tail()}"
        )

    def _log_tail(self, max_chars: int = 4000) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"Could not read vLLM feedback server log: {exc}"
        if not text:
            return "vLLM feedback server log is empty."
        return "vLLM feedback server log tail:\n" + text[-max_chars:]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None,
        stop_sequences: Iterable[str],
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_new_tokens if max_new_tokens is not None else 256,
            "temperature": 0.0,
        }
        stops = [seq for seq in stop_sequences if seq]
        if stops:
            payload["stop"] = stops
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=max(60, int(payload["max_tokens"]) * 2),
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0].get("text") or "")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None


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


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return normalize_tokenizer_artifacts(tokenizer.decode(token_ids, skip_special_tokens=True))


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
    max_new_tokens: Optional[int],
    decode_press: AnswerSuffixDecodingPress | None,
    stop_sequences: Iterable[str] = DEFAULT_STOP_SEQUENCES,
    prompt_ids: Optional[list[int]] = None,
    require_flashdecode: bool = False,
) -> str:
    if require_flashdecode and not _model_uses_flash_attention_3(model):
        raise RuntimeError(
            "require_flashdecode=True but the model is not configured with attn_implementation='flash_attention_3'."
        )
    device = infer_device(model)
    prompt_ids = list(prompt_ids) if prompt_ids is not None else _encode(tokenizer, prompt)
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

    generated_ids: list[int] = []
    next_token = outputs.logits[0, -1].argmax()

    ctx = decode_press(model) if decode_press is not None else torch.inference_mode()
    if require_flashdecode:
        reset_flashdecode_tracking(model)
    with ctx:
        generated_count = 0
        while max_new_tokens is None or generated_count < max_new_tokens:
            token_id = int(next_token.item())
            generated_ids.append(token_id)
            generated_count += 1
            token_tensor = next_token.reshape(1, 1)
            pos = torch.tensor([[cache.get_seq_length()]], dtype=torch.long, device=device)
            outputs = _model_forward(
                model,
                input_ids=token_tensor,
                past_key_values=cache,
                position_ids=pos,
                use_cache=True,
            )
            tail_text = _decode_token_ids(tokenizer, generated_ids[-32:])
            if token_id in eos_ids or any(seq in tail_text for seq in stop_sequences):
                break
            next_token = outputs.logits[0, -1].argmax()
        if decode_press is not None:
            decode_press.finalize_if_needed(model, cache)

    if require_flashdecode and generated_ids:
        used_layers = flashdecode_used_layers(model)
        if not used_layers:
            raise RuntimeError(
                "Flashdecode was required but no attention layer used flash_attn_with_kvcache during decode."
            )
        logger.debug("Verified flashdecode on decode layers: %s", used_layers)

    return _decode_token_ids(tokenizer, generated_ids)


# Llama-3.1 chat-template tokens. DeepSeek-R1-Distill-Llama-8B inherits them
# verbatim; so does meta-llama/Meta-Llama-3.1-8B-Instruct. ``_assert_llama3_tokenizer``
# guards at setup time so a surprise tokenizer (Qwen3, DeepSeek-V3 original)
# fails loudly rather than silently producing garbage.
_LLAMA3_USER_OPEN = "<|start_header_id|>user<|end_header_id|>\n\n"
_LLAMA3_ASSISTANT_OPEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"
_LLAMA3_SYSTEM_OPEN = "<|start_header_id|>system<|end_header_id|>\n\n"
_LLAMA3_TURN_END = "<|eot_id|>"
_LLAMA3_BOS_FALLBACK = "<|begin_of_text|>"


def _assert_llama3_tokenizer(tokenizer, model_name: str) -> None:
    """Verify the tokenizer recognises the Llama-3.1 chat-template markers
    used by _initial_context / _revision_prompt / _reference_answer_text.
    Raises RuntimeError if any required special token is missing, pointing
    the caller at the two supported model families.
    """
    vocab = set(tokenizer.get_vocab().keys())
    required = {_LLAMA3_TURN_END, "<|start_header_id|>", "<|end_header_id|>"}
    missing = sorted(required - vocab)
    if missing:
        raise RuntimeError(
            f"tokenizer for {model_name!r} is missing Llama-3.1 chat markers {missing}. "
            "live_loop.py's prompt construction currently hardcodes Llama-3.1-style tokens; "
            "supported model families are Llama-3.1-Instruct and "
            "DeepSeek-R1-Distill-Llama. Use one of those, or generalise "
            "_revision_prompt to use tokenizer.apply_chat_template."
        )


def _initial_context(task: dict[str, Any], cfg: ConvCodeWorldLiveConfig, tokenizer) -> str:
    """Return the chat-template-formatted opening: BOS + system + first user
    turn (closed with EOT). The assistant opener is emitted by
    ``_revision_prompt(1, "")`` so every iteration -- including the first --
    uniformly prefixes its generation with an assistant-turn header, and
    the generate loop always has non-empty prompt tokens to drive.
    """
    prompt = (
        task_get(task, "instruct_prompt")
        or task_get(task, "complete_prompt")
        or task_get(task, "prompt")
        or task_get(task, "code_prompt")
        or ""
    )
    system_body = (
        "You are a Python coding assistant solving a programming task over multiple "
        "refinement turns. Each assistant reply MUST be a single ```python ... ``` "
        "fenced block containing the complete solution function. Do not include "
        "explanations, pseudocode, or commentary outside the code fence."
    )
    if cfg.cot:
        # Opt-in CoT. Off by default -- see ConvCodeWorldLiveConfig.cot.
        system_body += " Think through the feedback briefly before writing the code."
    user_body = f"Task:\n{prompt}"
    bos = tokenizer.bos_token or _LLAMA3_BOS_FALLBACK
    return (
        f"{bos}"
        f"{_LLAMA3_SYSTEM_OPEN}{system_body}{_LLAMA3_TURN_END}"
        f"{_LLAMA3_USER_OPEN}{user_body}{_LLAMA3_TURN_END}"
    )


def _revision_prompt(iteration: int, feedback: str) -> str:
    """Return the continuation text to prefill at the start of iter ``iteration``.

    iter 1: the initial_context ended at ``<|eot_id|>`` closing the first
    user turn. This returns just the assistant opener so the model starts
    producing code immediately.

    iter k>1: closes the previous assistant turn (``<|eot_id|>``), opens a
    new user turn carrying the feedback body, closes it, and opens the new
    assistant turn. Leading ``<|eot_id|>`` is harmless even if the prior
    turn already ended with one (the model treats consecutive EOTs as
    redundant turn separators, not content).
    """
    if iteration == 1:
        return _LLAMA3_ASSISTANT_OPEN
    body = feedback or "No new feedback was provided. Keep the solution correct and complete."
    return (
        f"{_LLAMA3_TURN_END}"
        f"{_LLAMA3_USER_OPEN}{body}{_LLAMA3_TURN_END}"
        f"{_LLAMA3_ASSISTANT_OPEN}"
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
        "Return 2-4 complete sentences of actionable verbal feedback for the next coding turn.\n"
        "Be specific: mention the failing behavior, what the tests expected instead, and the concrete area to change.\n"
        "Avoid vague feedback such as 'fix the error' or 'check the title'. Do not write revised code, bullets, "
        "chain-of-thought, or hidden-test claims.\n"
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
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if "<|channel>thought" in text and "<channel|>" in text:
        text = text.split("<channel|>", 1)[1].strip()
    for marker in ("###", "```", "\n\n---"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text or "Revise the code using the compilation and execution feedback above."


def _is_degenerate_feedback(text: str) -> bool:
    words = str(text or "").split()
    if len(words) < 24:
        return False
    for width in (3, 4, 5):
        grams = [" ".join(words[i : i + width]).lower() for i in range(len(words) - width + 1)]
        if not grams:
            continue
        top_count = max(grams.count(gram) for gram in set(grams))
        if top_count * width >= len(words) * 0.45:
            return True
    return False


def _last_error_line(text: str) -> str:
    for line in reversed(str(text or "").splitlines()):
        stripped = line.strip()
        if stripped and not stripped.startswith(("~", "^", "-", "=")):
            return stripped
    return ""


def _failed_test_summary(text: str, *, max_cases: int = 3) -> str:
    cases: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if " ... FAIL" in line or " ... ERROR" in line:
            case = line.split(" ... ", 1)[0].strip()
            if case and case not in cases:
                cases.append(case)
        if len(cases) >= max_cases:
            break
    if not cases:
        return ""
    suffix = "" if len(cases) == 1 else "s"
    return f"failing test case{suffix}: " + ", ".join(cases)


def _fallback_verbal_feedback(result) -> str:
    if result.passed:
        return "The previous code passed the available tests. Stop the live loop."
    if result.status == "compile_error":
        summary = _last_error_line(result.compilation_feedback)
        if summary:
            return (
                f"The solution does not compile; the compiler reports: {summary}. "
                "Revise the implementation so all required names are defined/imported, keep the required function "
                "signature unchanged, and then re-run the tests."
            )
        return (
            "The solution does not compile. Fix the syntax/import/name issue, preserve the required function "
            "signature, and make sure the function can be imported by the test harness."
        )
    if result.status == "timeout":
        return (
            "The candidate timed out before the tests completed. Reduce unnecessary work or non-terminating loops, "
            "handle edge cases directly, and keep the implementation bounded for the tested input sizes."
        )
    summary = _last_error_line(result.execution_feedback)
    cases = _failed_test_summary(result.execution_feedback)
    if summary:
        case_text = f" The relevant {cases} show where to focus." if cases else ""
        return (
            f"The implementation still fails the available tests: {summary}.{case_text} "
            "Compare the actual value in the traceback with the expected value, adjust the function logic for that "
            "behavior, and preserve the required function signature."
        )
    return (
        "The implementation still fails the available tests. Use the failing unittest output to identify the expected "
        "behavior, update only the function logic needed for that behavior, and preserve the required signature."
    )


def _format_feedback_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def _simulate_verbal_feedback(model, tokenizer, task: dict[str, Any], code: str, result, cfg: ConvCodeWorldLiveConfig):
    if result.passed and cfg.early_stop_on_pass:
        return "The previous code passed the available tests. Stop the live loop."
    prompt = _format_feedback_prompt(tokenizer, _simulator_prompt(task, code, result, cfg))
    stop_sequences = ("\n###", "```", "\n\n---")
    if isinstance(model, VllmTritonFeedbackClient):
        raw = model.generate(
            prompt,
            max_new_tokens=cfg.verbal_feedback_max_new_tokens,
            stop_sequences=stop_sequences,
        )
    else:
        raw = _generate_after_prompt(
            model,
            tokenizer,
            DynamicCache(),
            prompt,
            max_new_tokens=cfg.verbal_feedback_max_new_tokens,
            decode_press=None,
            stop_sequences=stop_sequences,
            require_flashdecode=cfg.require_flashdecode,
        )
    feedback = _clean_simulator_feedback(raw)
    if _is_degenerate_feedback(feedback):
        logger.warning("Feedback model output was degenerate; using deterministic feedback fallback.")
        return _fallback_verbal_feedback(result)
    if not result.passed and len(feedback.split()) < 30:
        logger.warning("Feedback model output was too short; using deterministic feedback fallback.")
        return _fallback_verbal_feedback(result)
    return feedback


def _reference_feedback(turn: dict[str, Any], cfg: ConvCodeWorldLiveConfig) -> str:
    sections: list[str] = []
    if cfg.include_compilation_feedback:
        compilation = str(turn.get("compilation_feedback") or "").strip()
        if compilation:
            sections.append("Compilation feedback:\n" + trim_feedback(compilation))
    if cfg.include_execution_feedback:
        execution = str(turn.get("execution_feedback") or "").strip()
        if execution:
            sections.append("Execution feedback:\n" + trim_feedback(execution))
    if cfg.include_verbal_feedback:
        verbal = str(turn.get("verbal_feedback") or "").strip()
        if verbal:
            sections.append("Verbal feedback:\n" + trim_feedback(verbal))
    return "\n\n".join(sections).strip()


def _reference_answer_text(code: str) -> str:
    """Assistant-turn body for Mode 2 teacher-forced prefill. Wraps the
    reference code in a ```python fence matching the model's expected
    output format. The turn-end marker (``<|eot_id|>``) is NOT appended
    here -- the next iteration's ``_revision_prompt`` prepends one
    unconditionally, so adding it here would double-emit EOT.
    """
    clean = extract_code(code).strip()
    return f"```python\n{clean}\n```"


def _truncate_cache(cache: DynamicCache, seq_len: int) -> None:
    for cache_layer in cache.layers:
        if hasattr(cache_layer, "keys") and cache_layer.keys is not None and cache_layer.keys.numel() > 0:
            cache_layer.keys = cache_layer.keys[:, :, :seq_len, :]
        if hasattr(cache_layer, "values") and cache_layer.values is not None and cache_layer.values.numel() > 0:
            cache_layer.values = cache_layer.values[:, :, :seq_len, :]
        if hasattr(cache_layer, "cumulative_length"):
            cache_layer.cumulative_length = min(int(getattr(cache_layer, "cumulative_length")), seq_len)


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
                "benchmark_mode": cfg.benchmark_mode,
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
        self.cfg.benchmark_mode = _normalize_benchmark_mode(self.cfg.benchmark_mode)
        _apply_feedback_config(self.cfg)
        if self.cfg.full_kv_cache:
            if self.cfg.benchmark_mode != "live":
                raise ValueError("full_kv_cache is only supported in benchmark_mode='live'.")
            if self.cfg.press_name != "no_press":
                raise ValueError(
                    "full_kv_cache=True requires press_name='no_press' so the code-generation KV cache is never evicted."
                )
            self.cfg.compression_ratio = 0.0
        if self.cfg.require_flashdecode:
            if not _is_flash_attention_3(self.cfg.attn_implementation):
                raise ValueError(
                    "require_flashdecode=True requires attn_implementation='flash_attention_3' for the code model."
                )
            effective_feedback_attn = self.cfg.feedback_attn_implementation or self.cfg.attn_implementation
            if (
                self.cfg.include_verbal_feedback
                and not _is_vllm_triton_attention(effective_feedback_attn)
                and not _is_flash_attention_3(effective_feedback_attn)
            ):
                raise ValueError(
                    "require_flashdecode=True requires the feedback model to use either "
                    "feedback_attn_implementation='flash_attention_3' or "
                    "feedback_attn_implementation='vllm_triton'."
                )
        _setup_logging(self.cfg.log_level)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.output_dir = _results_dir(self.cfg)
        self.global_target = _target_from_ratio(self.cfg.global_budget, self.cfg.compression_ratio)

    def _code_generation_limit(self) -> int | None:
        return None if self.cfg.code_generation_until_eos else self.cfg.max_new_tokens

    def _load_tokenizer(self, model_name: str):
        try:
            return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            return getattr(processor, "tokenizer", processor)

    def _load_model(self, model_name: str, attn_implementation: Optional[str]):
        model_kwargs = dict(self.cfg.model_kwargs or {})
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        if attn_implementation:
            model_kwargs.setdefault("attn_implementation", attn_implementation)
        if self.cfg.fp8:
            model_kwargs["quantization_config"] = FineGrainedFP8Config()
        if model_kwargs.get("attn_implementation") == "flash_attention_3":
            try:
                import flash_attn_interface  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "attn_implementation='flash_attention_3' requires FlashAttention-3. "
                    "Install the Dao-AILab flash-attention hopper package in the runtime image."
                ) from exc
        logger.info(
            "Loading %s with attn_implementation=%s",
            model_name,
            model_kwargs.get("attn_implementation"),
        )
        device_map = model_kwargs.pop("device_map", None)
        if device_map is None:
            device_map = {"": 0} if torch.cuda.is_available() else "auto"
        tokenizer = self._load_tokenizer(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
            **model_kwargs,
        )
        model.eval()
        if model_kwargs.get("attn_implementation") == "flash_attention_3":
            non_cuda_devices = sorted(
                {
                    str(parameter.device)
                    for parameter in model.parameters()
                    if parameter.device.type != "cuda"
                }
            )
            if non_cuda_devices:
                raise RuntimeError(
                    "FlashAttention-3 requires all model parameters on CUDA, "
                    f"but {model_name} has parameters on {non_cuda_devices}. "
                    "Use a larger GPU or an attention implementation that supports CPU/offload."
                )
        return model, tokenizer

    def setup_models(self):
        model, tokenizer = self._load_model(self.cfg.model, self.cfg.attn_implementation)
        # Prompt construction (_initial_context / _revision_prompt /
        # _reference_answer_text) hardcodes Llama-3.1 chat-template markers.
        # Verify up-front that the loaded tokenizer understands them; a
        # surprise tokenizer (e.g. Qwen3, Mistral, DeepSeek-V3 original) would
        # produce garbage prefills and 0% Pass@1 without this guard.
        _assert_llama3_tokenizer(tokenizer, self.cfg.model)
        if self.cfg.benchmark_mode == "static":
            return model, tokenizer, None, None
        if not self.cfg.include_verbal_feedback:
            return model, tokenizer, model, tokenizer
        feedback_model_name = self.cfg.feedback_model or self.cfg.model
        feedback_attn_implementation = (
            self.cfg.feedback_attn_implementation or self.cfg.attn_implementation
        )
        if _is_vllm_triton_attention(feedback_attn_implementation):
            logger.info("Loading feedback tokenizer for %s", feedback_model_name)
            feedback_tokenizer = self._load_tokenizer(feedback_model_name)
            logger.info("Loaded feedback tokenizer for %s", feedback_model_name)
            feedback_model = VllmTritonFeedbackClient(
                feedback_model_name,
                self.cfg,
                self.output_dir,
            )
            return model, tokenizer, feedback_model, feedback_tokenizer
        if feedback_model_name == self.cfg.model:
            return model, tokenizer, model, tokenizer
        feedback_model, feedback_tokenizer = self._load_model(
            feedback_model_name,
            feedback_attn_implementation,
        )
        return model, tokenizer, feedback_model, feedback_tokenizer

    def run(self) -> None:
        tasks = _load_tasks(self.cfg)
        feedback_model = None
        try:
            model, tokenizer, feedback_model, feedback_tokenizer = self.setup_models()
            base_press = _setup_press(self.cfg)
            global_press = _as_global_press(base_press, self.cfg)
            scorer = _decode_scorer(global_press)
            reference_trajectories = (
                _load_reference_trajectories(self.cfg.feedback_config)
                if self.cfg.benchmark_mode == "static"
                else None
            )

            rows: list[dict[str, Any]] = []
            for task_idx, task in enumerate(tqdm(tasks, desc=f"ConvCodeWorld {self.cfg.benchmark_mode}"), start=1):
                task_start = time.perf_counter()
                task_id = task_get(task, "task_id")
                logger.info("Starting task %s/%s: %s", task_idx, len(tasks), task_id)
                if self.cfg.benchmark_mode == "static":
                    task_rows = self._run_task_static(
                        model,
                        tokenizer,
                        task,
                        reference_trajectories or {},
                        global_press,
                        scorer,
                    )
                else:
                    task_rows = self._run_task(
                        model,
                        tokenizer,
                        feedback_model,
                        feedback_tokenizer,
                        task,
                        global_press,
                        scorer,
                    )
                rows.extend(task_rows)
                logger.info(
                    "Finished task %s/%s: %s in %.1fs",
                    task_idx,
                    len(tasks),
                    task_id,
                    time.perf_counter() - task_start,
                )
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
        finally:
            if isinstance(feedback_model, VllmTritonFeedbackClient):
                feedback_model.close()

    def _run_task(
        self,
        model,
        tokenizer,
        feedback_model,
        feedback_tokenizer,
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
            initial_context_ids = _encode(tokenizer, initial_context, max_tokens=self.cfg.max_input_tokens)
            if self.cfg.error_on_kv_cache_vram_exhaustion:
                _assert_kv_cache_fits_available_vram(
                    model,
                    cache,
                    len(initial_context_ids),
                    label=f"task {task_get(task, 'task_id')} initial context",
                )
            _prefill_text(model, tokenizer, cache, initial_context, max_tokens=self.cfg.max_input_tokens)
            if global_press is not None:
                global_press.on_turn_end(0, "context", context_start, cache.get_seq_length())

            for iteration in range(1, self.cfg.max_turns + 1):
                iter_start = time.perf_counter()
                logger.info("Starting task %s iteration %s", task_get(task, "task_id"), iteration)
                prompt = _revision_prompt(iteration, feedback)
                prompt_ids = _encode(tokenizer, prompt)
                code_generation_limit = self._code_generation_limit()
                if self.cfg.error_on_kv_cache_vram_exhaustion and code_generation_limit is not None:
                    _assert_kv_cache_fits_available_vram(
                        model,
                        cache,
                        len(prompt_ids) + code_generation_limit,
                        label=f"task {task_get(task, 'task_id')} iteration {iteration}",
                    )
                if global_press is not None:
                    global_press.on_turn_start(iteration, "user", cache.get_seq_length())
                user_start = cache.get_seq_length()

                answer_start = cache.get_seq_length() + len(prompt_ids)
                decode_press = None
                if (
                    not self.cfg.full_kv_cache
                    and scorer is not None
                    and code_generation_limit is not None
                    and code_generation_limit > self.cfg.local_budget
                ):
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
                    max_new_tokens=code_generation_limit,
                    decode_press=decode_press,
                    prompt_ids=prompt_ids,
                    require_flashdecode=self.cfg.require_flashdecode,
                )
                user_end = answer_start
                if global_press is not None:
                    global_press.on_turn_end(iteration, "user", user_start, user_end)
                    global_press.on_turn_end(iteration, "assistant", user_end, cache.get_seq_length())

                code = normalize_candidate_code(task, extract_code(generated))
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
                    generation_device = infer_device(model)
                    _assert_cache_on_device(
                        cache,
                        generation_device,
                        "code generation cache before feedback",
                    )
                    feedback_args = (
                        feedback_model,
                        feedback_tokenizer,
                        task,
                        code,
                        result,
                        self.cfg,
                    )
                    if feedback_model is model and global_press is not None:
                        with global_press.suspend_hooks():
                            verbal_feedback = _simulate_verbal_feedback(*feedback_args)
                    else:
                        verbal_feedback = _simulate_verbal_feedback(*feedback_args)
                    _assert_cache_on_device(
                        cache,
                        generation_device,
                        "code generation cache after feedback",
                    )
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
                    "benchmark_mode": self.cfg.benchmark_mode,
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
                logger.info(
                    "Finished task %s iteration %s in %.1fs: status=%s passed=%s cache=%s->%s",
                    task_get(task, "task_id"),
                    iteration,
                    time.perf_counter() - iter_start,
                    result.status,
                    result.passed,
                    cache_before,
                    cache_after,
                )
                if not result.passed:
                    logger.debug(
                        "Task %s iteration %s execution details:\n"
                        "  generated (first 300): %s\n"
                        "  code (first 300): %s\n"
                        "  compilation: %s\n"
                        "  execution: %s\n"
                        "  stdout: %s\n"
                        "  stderr: %s",
                        task_get(task, "task_id"),
                        iteration,
                        repr(generated[:300]),
                        repr(code[:300]),
                        result.compilation_feedback[:500],
                        result.execution_feedback[:500],
                        result.stdout[:500],
                        result.stderr[:500],
                    )

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

    def _run_task_static(
        self,
        model,
        tokenizer,
        task: dict[str, Any],
        reference_trajectories: dict[str, list[dict[str, Any]]],
        global_press: TurnAwareGlobalPress | None,
        scorer: ScorerPress | None,
    ) -> list[dict[str, Any]]:
        cache = DynamicCache()
        rows: list[dict[str, Any]] = []
        task_id = str(task_get(task, "task_id"))
        session_id = f"{self.cfg.feedback_config}/{task_id}"
        reference_turns = reference_trajectories.get(task_id)
        if not reference_turns:
            logger.warning("No ConvCodeWorld reference trajectory for %s; skipping", task_id)
            return rows

        context_manager = global_press(model) if global_press is not None else torch.inference_mode()
        with context_manager:
            if global_press is not None:
                global_press.on_turn_start(0, "context", cache.get_seq_length())
            context_start = cache.get_seq_length()
            _prefill_text(
                model,
                tokenizer,
                cache,
                _initial_context(task, self.cfg, tokenizer),
                max_tokens=self.cfg.max_input_tokens,
            )
            if global_press is not None:
                global_press.on_turn_end(0, "context", context_start, cache.get_seq_length())

            feedback = ""
            for iteration, reference_turn in enumerate(reference_turns[: self.cfg.max_turns], start=1):
                iter_start = time.perf_counter()
                logger.info("Starting static task %s iteration %s", task_id, iteration)
                prompt = _revision_prompt(iteration, feedback)
                prompt_ids = _encode(tokenizer, prompt)
                code_generation_limit = self._code_generation_limit()
                if global_press is not None:
                    global_press.on_turn_start(iteration, "user", cache.get_seq_length())
                user_start = cache.get_seq_length()

                answer_start = cache.get_seq_length() + len(prompt_ids)
                decode_press = None
                if (
                    not self.cfg.full_kv_cache
                    and scorer is not None
                    and code_generation_limit is not None
                    and code_generation_limit > self.cfg.local_budget
                ):
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
                    max_new_tokens=code_generation_limit,
                    decode_press=decode_press,
                    prompt_ids=prompt_ids,
                    require_flashdecode=self.cfg.require_flashdecode,
                )
                user_end = answer_start
                if global_press is not None:
                    global_press.on_turn_end(iteration, "user", user_start, user_end)

                code = extract_code(generated)
                result = run_candidate(
                    task,
                    code,
                    timeout_s=self.cfg.executor_timeout_s,
                    memory_mb=self.cfg.executor_memory_mb,
                    network_isolation=self.cfg.network_isolation,
                    work_dir="/tmp",
                )

                _truncate_cache(cache, user_end)
                if global_press is not None:
                    global_press.on_turn_start(iteration, "assistant", cache.get_seq_length())
                assistant_start = cache.get_seq_length()
                _prefill_text(
                    model,
                    tokenizer,
                    cache,
                    _reference_answer_text(str(reference_turn.get("previous_code") or "")),
                )
                if global_press is not None:
                    global_press.on_turn_end(iteration, "assistant", assistant_start, cache.get_seq_length())

                feedback = _reference_feedback(reference_turn, self.cfg)
                cache_before = cache.get_seq_length()
                _maybe_global_compress(model, cache, global_press, self.global_target)
                cache_after = cache.get_seq_length()
                rows.append(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "feedback_config": self.cfg.feedback_config,
                        "benchmark_mode": self.cfg.benchmark_mode,
                        "iteration": iteration,
                        "predicted_answer": code,
                        "generated_code": code,
                        "raw_generation": generated,
                        "passed": result.passed,
                        "reference_label": reference_turn.get("label"),
                        "reference_code": reference_turn.get("previous_code"),
                        "status": result.status,
                        "compilation_feedback": result.compilation_feedback,
                        "execution_feedback": result.execution_feedback,
                        "reference_compilation_feedback": reference_turn.get("compilation_feedback"),
                        "reference_execution_feedback": reference_turn.get("execution_feedback"),
                        "reference_verbal_feedback": reference_turn.get("verbal_feedback"),
                        "feedback": feedback,
                        "cache_len_before_global": cache_before,
                        "cache_len_after_global": cache_after,
                        "skipped_after_pass": False,
                        "metric_excluded": False,
                    }
                )
                logger.info(
                    "Finished static task %s iteration %s in %.1fs: status=%s passed=%s cache=%s->%s",
                    task_id,
                    iteration,
                    time.perf_counter() - iter_start,
                    result.status,
                    result.passed,
                    cache_before,
                    cache_after,
                )

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
    env_feedback_model = os.environ.get("KV_PRESS_CONVCODEWORLD_FEEDBACK_MODEL", "").strip()
    if env_feedback_model:
        args["feedback_model"] = env_feedback_model
    env_attn_implementation = os.environ.get("KV_PRESS_CONVCODEWORLD_ATTN_IMPLEMENTATION", "").strip()
    if env_attn_implementation:
        args["attn_implementation"] = env_attn_implementation
    env_feedback_attn_implementation = os.environ.get(
        "KV_PRESS_CONVCODEWORLD_FEEDBACK_ATTN_IMPLEMENTATION",
        "",
    ).strip()
    if env_feedback_attn_implementation:
        args["feedback_attn_implementation"] = env_feedback_attn_implementation
    env_feedback_vllm_cuda_visible_devices = os.environ.get(
        "KV_PRESS_CONVCODEWORLD_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES",
        "",
    ).strip()
    if env_feedback_vllm_cuda_visible_devices:
        args["feedback_vllm_cuda_visible_devices"] = env_feedback_vllm_cuda_visible_devices
    cfg_kwargs = {k: v for k, v in args.items() if k in ConvCodeWorldLiveConfig.__dataclass_fields__}
    cfg = ConvCodeWorldLiveConfig(**cfg_kwargs)
    ConvCodeWorldLiveRunner(cfg).run()


def _cli_entrypoint(config_file: Optional[str] = None, **kwargs: Any) -> None:
    run(config_file=config_file, **kwargs)


if __name__ == "__main__":
    Fire(_cli_entrypoint)
