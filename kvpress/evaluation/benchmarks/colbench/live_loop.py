# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ColBench (Backend) live-loop runner with KV cache carry-over.

ColBench (Meta FAIR, SWEET-RL, March 2025) is a multi-turn coding benchmark
where the agent solves a Python task by chatting with a *simulated human*
(here: ``google/gemma-4-26B-A4B-it`` over a local vLLM Triton server) that has
access to the reference solution and the hidden tests. At each turn the agent
either:

  1. asks the simulator a clarifying question (natural language only), or
  2. submits a final candidate as a fenced ``\`\`\`python`` block.

The loop runs until the agent submits or ``max_turns`` is reached. After the
submission we run the candidate against the hidden tests and record pass/fail.

The harness is structurally close to ``convcodeworld/live_loop.py`` (cache
carry-over, press setup, vLLM Triton feedback sidecar, FA3 flashdecode
checks, KV-VRAM guards) but drops the five ConvCodeWorld feedback-config
branches (ColBench has only one feedback distribution) and the static-replay
mode (ColBench ships no reference dialogues).
"""

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

from evaluate_registry import PRESS_REGISTRY  # noqa: E402
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

from benchmarks.colbench.calculate_metrics import calculate_metrics as colbench_scorer  # noqa: E402
from benchmarks.colbench.executor import (  # noqa: E402
    PASSED_ALL_TEST_RUNS,
    build_feedback_after_submit,
    build_simulator_prompt,
    detect_submission,
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

# Stop sequences for the agent's generation. Same intent as convcodeworld:
# the primary stop signal is the chat-template EOT token, but these guards
# catch DeepSeek-R1-Distill's degenerate continuations (closes a code fence
# but keeps talking, hallucinates a fake user reply, etc.).
DEFAULT_STOP_SEQUENCES = (
    "\n```\n\n",
    "\n### User",
    "\n###User",
    "\nHuman:",
    "\nUser:",
    "<｜User｜>",
    "<｜end▁of▁sentence｜>",
    "<|eot_id|>",
)


@dataclass
class ColBenchLiveConfig:
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
    # ColBench-specific knobs.
    colbench_split: str = "backend"
    dataset_name: str = "facebook/collaborative_agent_bench"
    dataset_subset: str = "backend"
    hf_split: str = "train"
    max_turns: int = 10
    max_questions_before_submit: int = 9
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
    output_dir: str = "./results_colbench_live"
    log_level: str = "INFO"
    fp8: bool = False
    model_kwargs: Optional[dict[str, Any]] = None
    max_input_tokens: Optional[int] = None
    early_stop_on_pass: bool = True
    executor_timeout_s: int = 30
    executor_memory_mb: int = 1024
    network_isolation: str = "auto"
    # CoT defaults to False per ADR 002 §1 (DeepSeek-R1-Distill already has
    # reasoning baked into its chat template; an external "think first"
    # clause causes double-reasoning that burns max_new_tokens on
    # meta-commentary). Same default as convcodeworld.
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


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


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


def _apply_press_hyperparameters(press: BasePress, cfg: ColBenchLiveConfig) -> None:
    _set_press_field(press, "window_size", cfg.snapkv_window_size, class_names={"SnapKVPress", "PyramidKVPress"})
    _set_press_field(press, "kernel_size", cfg.snapkv_kernel_size, class_names={"SnapKVPress", "PyramidKVPress"})
    _set_press_field(press, "n_sink", cfg.streaming_llm_n_sink, class_names={"StreamingLLMPress"})
    _set_press_field(
        press,
        "n_future_positions",
        cfg.expected_attention_n_future_positions,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(press, "n_sink", cfg.expected_attention_n_sink, class_names={"ExpectedAttentionPress"})
    _set_press_field(
        press,
        "use_covariance",
        cfg.expected_attention_use_covariance,
        class_names={"ExpectedAttentionPress"},
    )
    _set_press_field(press, "use_vnorm", cfg.expected_attention_use_vnorm, class_names={"ExpectedAttentionPress"})
    _set_press_field(press, "epsilon", cfg.expected_attention_epsilon, class_names={"ExpectedAttentionPress"})


def _has_turn_aware_overrides(cfg: ColBenchLiveConfig) -> bool:
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


def _policy_requested(cfg: ColBenchLiveConfig, name: str) -> bool:
    names = {
        "floor": ("alpha_floor", "floor_gamma", "alpha_floor_len", "min_floor_tokens"),
        "anchor": ("alpha_anchor", "anchor_beta"),
        "loyalty": ("alpha_loyalty", "loyalty_top_p", "loyalty_update_every"),
    }[name]
    return any(getattr(cfg, field_name) is not None for field_name in names)


def _validate_turn_aware_overrides(cfg: ColBenchLiveConfig) -> None:
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
    cfg: ColBenchLiveConfig,
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


def _setup_press(cfg: ColBenchLiveConfig) -> BasePress | None:
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


def _as_global_press(press: BasePress | None, cfg: ColBenchLiveConfig) -> TurnAwareGlobalPress | None:
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
        f"ColBench live-loop global compression requires a ScorerPress-compatible press, "
        f"got {type(press).__name__}. Use snapkv, streaming_llm, expected_attention, knorm, or no_press."
    )


def _decode_scorer(press: TurnAwareGlobalPress | None) -> ScorerPress | None:
    if press is None:
        return None
    return press.base_press


def _load_dataset_split(name: str, subset: str, hf_split: str):
    if subset:
        try:
            return load_dataset(name, subset, split=hf_split)
        except (ValueError, FileNotFoundError):
            pass
    try:
        return load_dataset(name, split=hf_split)
    except Exception:
        return load_dataset(name)[hf_split]


def _resolve_task_id(row: dict[str, Any], idx: int) -> str:
    for key in ("task_id", "id", "uid", "name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"colbench/backend/{idx}"


def _parse_task_ids(value: str | None) -> set[str] | None:
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


def _load_tasks(cfg: ColBenchLiveConfig) -> list[dict[str, Any]]:
    ds = _load_dataset_split(cfg.dataset_name, cfg.dataset_subset, cfg.hf_split)
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(ds):
        row = dict(raw)
        row.setdefault("task_id", _resolve_task_id(row, idx))
        rows.append(row)

    requested = _parse_task_ids(cfg.task_ids)
    if requested:
        rows = [row for row in rows if str(row.get("task_id")) in requested]

    if cfg.fraction < 1.0:
        rows = random.Random(cfg.seed).sample(rows, max(1, int(len(rows) * cfg.fraction)))
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(rows)
    if cfg.num_eval_examples > 0:
        rows = rows[: cfg.num_eval_examples]
    return rows


def _results_dir(cfg: ColBenchLiveConfig) -> Path:
    parts = [
        "live",
        cfg.colbench_split,
        cfg.model.replace("/", "--"),
        f"fb-{(cfg.feedback_model or cfg.model).replace('/', '--')}",
        cfg.press_name,
        f"{cfg.compression_ratio:.2f}",
    ]
    turn_bits = []
    turn_attr_labels = {
        "alpha_floor": "af",
        "alpha_anchor": "aa",
        "alpha_loyalty": "al",
        "anchor_beta": "ab",
        "floor_gamma": "fg",
        "loyalty_top_p": "ltp",
        "loyalty_update_every": "lue",
        "alpha_floor_len": "afl",
        "min_floor_tokens": "mft",
    }
    for attr, label in turn_attr_labels.items():
        value = getattr(cfg, attr)
        if value is not None:
            turn_bits.append(f"{label}-{_slug_value(value)}")
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
    """OpenAI-compatible client for a local vLLM Triton feedback server.

    Identical pattern to ``convcodeworld/live_loop.py:VllmTritonFeedbackClient``;
    duplicated rather than imported because the convcodeworld harness has
    benchmark-specific imports we'd otherwise pull in.
    """

    def __init__(self, model_name: str, cfg: ColBenchLiveConfig, log_dir: Path):
        self.model_name = model_name
        self.port = int(cfg.feedback_vllm_port)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.log_path = log_dir / "vllm_feedback_server.log"
        self._log_handle = None
        self.process: subprocess.Popen | None = None
        self._start(cfg)

    def _command(self, cfg: ColBenchLiveConfig) -> list[str]:
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

    def _start(self, cfg: ColBenchLiveConfig) -> None:
        env = os.environ.copy()
        env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        env.setdefault("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "0")
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


# Llama-3.1 chat-template tokens. Same family/markers as convcodeworld.
_LLAMA3_USER_OPEN = "<|start_header_id|>user<|end_header_id|>\n\n"
_LLAMA3_ASSISTANT_OPEN = "<|start_header_id|>assistant<|end_header_id|>\n\n"
_LLAMA3_SYSTEM_OPEN = "<|start_header_id|>system<|end_header_id|>\n\n"
_LLAMA3_TURN_END = "<|eot_id|>"
_LLAMA3_BOS_FALLBACK = "<|begin_of_text|>"


def _assert_llama3_tokenizer(tokenizer, model_name: str) -> None:
    vocab = set(tokenizer.get_vocab().keys())
    required = {_LLAMA3_TURN_END, "<|start_header_id|>", "<|end_header_id|>"}
    missing = sorted(required - vocab)
    if missing:
        raise RuntimeError(
            f"tokenizer for {model_name!r} is missing Llama-3.1 chat markers {missing}. "
            "live_loop.py's prompt construction currently hardcodes Llama-3.1-style tokens; "
            "supported model families are Llama-3.1-Instruct and "
            "DeepSeek-R1-Distill-Llama. Use one of those, or generalise "
            "the prompt builders to use tokenizer.apply_chat_template."
        )


def _agent_system_body(cfg: ColBenchLiveConfig) -> str:
    body = (
        "You are a Python coding agent collaborating with a non-coder human user "
        "to solve a programming task. The user knows the goal but not how to code; "
        "they have access to a private reference solution and hidden tests but "
        "will not show them to you. You have a budget of "
        f"{cfg.max_turns} turns total to either ask clarifying questions or submit code.\n\n"
        "On each turn you must produce EXACTLY ONE of:\n"
        "  (A) a clarifying question in plain English (no code blocks), or\n"
        "  (B) a final solution wrapped in a single ```python ... ``` fenced block.\n\n"
        "Submission rules:\n"
        "- Submit only when you are confident; once you submit, the loop ends.\n"
        "- Submissions MUST be one ```python ... ``` block containing the complete "
        "solution function. Do not include explanations, pseudocode, or commentary "
        "outside the code fence.\n"
        "- Do not include test cases, example calls, or print statements in the submission.\n\n"
        "Question rules:\n"
        "- Ask one specific, answerable clarifying question per turn.\n"
        "- Do not propose code in a question turn (no fences, no inline backticks "
        "wrapping multiline code). The user will not run code you propose.\n"
        "- The user will not reveal the reference solution; questions that ask for "
        "it will be redirected."
    )
    if cfg.cot:
        body += " Think briefly before answering."
    return body


def _initial_context(task: dict[str, Any], cfg: ColBenchLiveConfig, tokenizer) -> str:
    description = (
        task_get(task, "description")
        or task_get(task, "instruction")
        or task_get(task, "instruct_prompt")
        or task_get(task, "prompt")
        or ""
    )
    code_prompt = str(task_get(task, "code_prompt") or task_get(task, "starter_code") or "")
    entry_point = str(task_get(task, "entry_point") or "")

    user_lines = [f"Task description:\n{description}"]
    if entry_point:
        user_lines.append(f"\nThe required entry-point function is `{entry_point}`.")
    if code_prompt:
        user_lines.append(
            "\nA starter stub is provided below. Your final submission must be "
            "compatible with this signature.\n```python\n"
            f"{code_prompt}\n```"
        )
    user_body = "\n".join(user_lines)
    bos = tokenizer.bos_token or _LLAMA3_BOS_FALLBACK
    return (
        f"{bos}"
        f"{_LLAMA3_SYSTEM_OPEN}{_agent_system_body(cfg)}{_LLAMA3_TURN_END}"
        f"{_LLAMA3_USER_OPEN}{user_body}{_LLAMA3_TURN_END}"
    )


def _agent_turn_prompt(iteration: int, user_reply: str) -> str:
    """Continuation text appended at the start of agent turn ``iteration``.

    iter 1: the initial_context closed the first user turn with EOT, so we
    just open the assistant turn.

    iter k>1: close the prior assistant turn (idempotent leading EOT), open
    a new user turn with the simulator's reply, close it, and open the new
    assistant turn.
    """
    if iteration == 1:
        return _LLAMA3_ASSISTANT_OPEN
    body = user_reply or "(The user did not reply. Continue with your best plan.)"
    return (
        f"{_LLAMA3_TURN_END}"
        f"{_LLAMA3_USER_OPEN}{body}{_LLAMA3_TURN_END}"
        f"{_LLAMA3_ASSISTANT_OPEN}"
    )


def _format_simulator_prompt_for_chat(tokenizer, prompt: str) -> str:
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


def _clean_simulator_reply(text: str) -> str:
    text = str(text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    for marker in ("###", "```", "\n\n---", "<|"):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text or "I'm not sure - please make a best-effort attempt and I'll react."


def _simulate_user_reply(
    feedback_model,
    feedback_tokenizer,
    task: dict[str, Any],
    agent_question: str,
    dialogue_so_far: str,
    cfg: ColBenchLiveConfig,
) -> str:
    prompt = _format_simulator_prompt_for_chat(
        feedback_tokenizer,
        build_simulator_prompt(task, agent_question, dialogue_so_far),
    )
    stop_sequences = ("\n###", "```", "\n\n---")
    if isinstance(feedback_model, VllmTritonFeedbackClient):
        raw = feedback_model.generate(
            prompt,
            max_new_tokens=cfg.verbal_feedback_max_new_tokens,
            stop_sequences=stop_sequences,
        )
    else:
        raw = _generate_after_prompt(
            feedback_model,
            feedback_tokenizer,
            DynamicCache(),
            prompt,
            max_new_tokens=cfg.verbal_feedback_max_new_tokens,
            decode_press=None,
            stop_sequences=stop_sequences,
            require_flashdecode=cfg.require_flashdecode,
        )
    return _clean_simulator_reply(raw)


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
    cfg: ColBenchLiveConfig,
    task: dict[str, Any],
    from_iteration: int,
    code: str,
    cache_len: int,
) -> None:
    """Pad with skipped rows after a successful submission, mirroring the
    convcodeworld behavior so per-iteration aggregation is consistent.
    """
    for iteration in range(from_iteration, cfg.max_turns + 1):
        rows.append(
            {
                "session_id": str(task_get(task, "task_id")),
                "task_id": task_get(task, "task_id"),
                "colbench_split": cfg.colbench_split,
                "iteration": iteration,
                "is_question": False,
                "is_submission": False,
                "agent_message": "",
                "user_reply": "",
                "predicted_answer": code,
                "generated_code": code,
                "passed": True,
                "status": "skipped_after_pass",
                "compilation_feedback": "",
                "execution_feedback": PASSED_ALL_TEST_RUNS,
                "feedback": "Skipped because an earlier live-loop iteration submitted passing code.",
                "cache_len_before_global": cache_len,
                "cache_len_after_global": cache_len,
                "skipped_after_pass": True,
                "metric_excluded": True,
            }
        )


class ColBenchLiveRunner:
    def __init__(self, cfg: ColBenchLiveConfig):
        self.cfg = cfg
        if self.cfg.full_kv_cache:
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
                not _is_vllm_triton_attention(effective_feedback_attn)
                and not _is_flash_attention_3(effective_feedback_attn)
            ):
                raise ValueError(
                    "require_flashdecode=True requires the feedback model to use either "
                    "feedback_attn_implementation='flash_attention_3' or "
                    "feedback_attn_implementation='vllm_triton'."
                )
        if self.cfg.max_questions_before_submit >= self.cfg.max_turns:
            # Reserve at least one turn for a submission; if the agent issues
            # only questions up to max_questions_before_submit, the remaining
            # turn(s) are forced-submit.
            self.cfg.max_questions_before_submit = max(1, self.cfg.max_turns - 1)
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
                {str(parameter.device) for parameter in model.parameters() if parameter.device.type != "cuda"}
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
        _assert_llama3_tokenizer(tokenizer, self.cfg.model)
        feedback_model_name = self.cfg.feedback_model or self.cfg.model
        feedback_attn_implementation = self.cfg.feedback_attn_implementation or self.cfg.attn_implementation
        if _is_vllm_triton_attention(feedback_attn_implementation):
            logger.info("Loading feedback tokenizer for %s", feedback_model_name)
            feedback_tokenizer = self._load_tokenizer(feedback_model_name)
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

            rows: list[dict[str, Any]] = []
            for task_idx, task in enumerate(tqdm(tasks, desc=f"ColBench {self.cfg.colbench_split}"), start=1):
                task_start = time.perf_counter()
                task_id = task_get(task, "task_id")
                logger.info("Starting task %s/%s: %s", task_idx, len(tasks), task_id)
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
            metrics = colbench_scorer(df)
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
        task_id = str(task_get(task, "task_id"))
        session_id = task_id
        initial_context = _initial_context(task, self.cfg, tokenizer)
        user_reply = ""
        dialogue_log: list[str] = []
        questions_so_far = 0

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
                    label=f"task {task_id} initial context",
                )
            _prefill_text(model, tokenizer, cache, initial_context, max_tokens=self.cfg.max_input_tokens)
            if global_press is not None:
                global_press.on_turn_end(0, "context", context_start, cache.get_seq_length())

            for iteration in range(1, self.cfg.max_turns + 1):
                iter_start = time.perf_counter()
                logger.info("Starting task %s iteration %s", task_id, iteration)
                # On the final allowed turn, force a submission to give the
                # agent a chance to convert any remaining understanding into
                # a tested attempt rather than burning the budget on questions.
                turns_left = self.cfg.max_turns - iteration
                must_submit_now = (
                    questions_so_far >= self.cfg.max_questions_before_submit or turns_left == 0
                )
                if must_submit_now:
                    forcing_note = (
                        "\n\nThe budget for clarifying questions has been used. "
                        "On this turn you MUST submit your best solution as a "
                        "single ```python ... ``` block."
                    )
                    user_reply_with_force = (user_reply or "Please submit your best solution now.") + forcing_note
                else:
                    user_reply_with_force = user_reply

                prompt = _agent_turn_prompt(iteration, user_reply_with_force)
                prompt_ids = _encode(tokenizer, prompt)
                code_generation_limit = self._code_generation_limit()
                if self.cfg.error_on_kv_cache_vram_exhaustion and code_generation_limit is not None:
                    _assert_kv_cache_fits_available_vram(
                        model,
                        cache,
                        len(prompt_ids) + code_generation_limit,
                        label=f"task {task_id} iteration {iteration}",
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

                # Classify the agent's reply: submission vs. question.
                submitted_body = detect_submission(generated)
                if submitted_body is not None:
                    code = normalize_candidate_code(task, submitted_body)
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
                    feedback = build_feedback_after_submit(result)
                    rows.append(
                        {
                            "session_id": session_id,
                            "task_id": task_id,
                            "colbench_split": self.cfg.colbench_split,
                            "iteration": iteration,
                            "is_question": False,
                            "is_submission": True,
                            "agent_message": generated,
                            "user_reply": user_reply,
                            "predicted_answer": code,
                            "generated_code": code,
                            "raw_generation": generated,
                            "passed": result.passed,
                            "status": result.status if not result.passed else "pass",
                            "compilation_feedback": result.compilation_feedback,
                            "execution_feedback": result.execution_feedback,
                            "feedback": feedback,
                            "cache_len_before_global": cache_before,
                            "cache_len_after_global": cache_after,
                            "skipped_after_pass": False,
                            "metric_excluded": False,
                        }
                    )
                    logger.info(
                        "Task %s iteration %s SUBMITTED in %.1fs: status=%s passed=%s cache=%s->%s",
                        task_id,
                        iteration,
                        time.perf_counter() - iter_start,
                        result.status,
                        result.passed,
                        cache_before,
                        cache_after,
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
                    # ColBench convention: terminate after the first submission
                    # regardless of pass/fail. Refining post-submission would
                    # change the question-budget metric we report.
                    break

                # Otherwise treat the reply as a clarifying question and ask
                # the simulator to respond.
                agent_question = generated.strip()
                questions_so_far += 1
                dialogue_log.append(f"Agent: {agent_question}")
                generation_device = infer_device(model)
                _assert_cache_on_device(cache, generation_device, "code generation cache before simulator")
                feedback_args = (
                    feedback_model,
                    feedback_tokenizer,
                    task,
                    agent_question,
                    "\n".join(dialogue_log[-12:]),
                    self.cfg,
                )
                if feedback_model is model and global_press is not None:
                    with global_press.suspend_hooks():
                        sim_reply = _simulate_user_reply(*feedback_args)
                else:
                    sim_reply = _simulate_user_reply(*feedback_args)
                _assert_cache_on_device(cache, generation_device, "code generation cache after simulator")
                dialogue_log.append(f"User: {sim_reply}")
                user_reply = sim_reply
                cache_before = cache.get_seq_length()
                _maybe_global_compress(model, cache, global_press, self.global_target)
                cache_after = cache.get_seq_length()
                rows.append(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "colbench_split": self.cfg.colbench_split,
                        "iteration": iteration,
                        "is_question": True,
                        "is_submission": False,
                        "agent_message": agent_question,
                        "user_reply": sim_reply,
                        "predicted_answer": "",
                        "generated_code": "",
                        "raw_generation": generated,
                        "passed": False,
                        "status": "pending",
                        "compilation_feedback": "",
                        "execution_feedback": "",
                        "feedback": sim_reply,
                        "cache_len_before_global": cache_before,
                        "cache_len_after_global": cache_after,
                        "skipped_after_pass": False,
                        "metric_excluded": False,
                    }
                )
                logger.info(
                    "Task %s iteration %s QUESTION in %.1fs: cache=%s->%s",
                    task_id,
                    iteration,
                    time.perf_counter() - iter_start,
                    cache_before,
                    cache_after,
                )

            else:
                # Hit max_turns without submitting. Record a synthetic
                # submitted_no_pass row so the session shows up in the
                # session-level aggregates.
                rows.append(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "colbench_split": self.cfg.colbench_split,
                        "iteration": self.cfg.max_turns,
                        "is_question": False,
                        "is_submission": False,
                        "agent_message": "",
                        "user_reply": user_reply,
                        "predicted_answer": "",
                        "generated_code": "",
                        "passed": False,
                        "status": "submitted_no_pass",
                        "compilation_feedback": "",
                        "execution_feedback": "Agent ran out of turns without submitting.",
                        "feedback": "",
                        "cache_len_before_global": cache.get_seq_length(),
                        "cache_len_after_global": cache.get_seq_length(),
                        "skipped_after_pass": False,
                        "metric_excluded": False,
                    }
                )

        return rows


def run(config: ColBenchLiveConfig | None = None, config_file: Optional[str] = None, **cli_overrides: Any) -> None:
    args = asdict(ColBenchLiveConfig())
    if config is not None:
        args.update(asdict(config))
    if config_file:
        p = Path(config_file)
        if p.exists():
            args.update(yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    args.update({k: v for k, v in cli_overrides.items() if v is not None})
    env_model = os.environ.get("KV_PRESS_COLBENCH_MODEL", "").strip()
    if env_model:
        args["model"] = env_model
    env_feedback_model = os.environ.get("KV_PRESS_COLBENCH_FEEDBACK_MODEL", "").strip()
    if env_feedback_model:
        args["feedback_model"] = env_feedback_model
    env_attn_implementation = os.environ.get("KV_PRESS_COLBENCH_ATTN_IMPLEMENTATION", "").strip()
    if env_attn_implementation:
        args["attn_implementation"] = env_attn_implementation
    env_feedback_attn_implementation = os.environ.get(
        "KV_PRESS_COLBENCH_FEEDBACK_ATTN_IMPLEMENTATION",
        "",
    ).strip()
    if env_feedback_attn_implementation:
        args["feedback_attn_implementation"] = env_feedback_attn_implementation
    env_feedback_vllm_cuda_visible_devices = os.environ.get(
        "KV_PRESS_COLBENCH_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES",
        "",
    ).strip()
    if env_feedback_vllm_cuda_visible_devices:
        args["feedback_vllm_cuda_visible_devices"] = env_feedback_vllm_cuda_visible_devices
    cfg_kwargs = {k: v for k, v in args.items() if k in ColBenchLiveConfig.__dataclass_fields__}
    cfg = ColBenchLiveConfig(**cfg_kwargs)
    ColBenchLiveRunner(cfg).run()


def _cli_entrypoint(config_file: Optional[str] = None, **kwargs: Any) -> None:
    run(config_file=config_file, **kwargs)


if __name__ == "__main__":
    Fire(_cli_entrypoint)
