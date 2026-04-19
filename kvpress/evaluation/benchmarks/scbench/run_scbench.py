# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for SCBench SCDQ evaluation with kvpress (run from the ``evaluation/`` directory or via ``python -m``)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import torch
import yaml
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import FineGrainedFP8Config, pipeline

_ROOT_EVAL = Path(__file__).resolve().parents[2]
if str(_ROOT_EVAL) not in sys.path:
    sys.path.insert(0, str(_ROOT_EVAL))

from evaluate_registry import PRESS_REGISTRY  # noqa: E402
from kvpress import ComposedPress, DMSPress, DuoAttentionPress, ThinKPress  # noqa: E402

from benchmarks.scbench.calculate_metrics import calculate_metrics  # noqa: E402
from benchmarks.scbench.scdq_prompts import (  # noqa: E402
    DATA_NAME_TO_MAX_NEW_TOKENS,
    create_scdq_prompt,
    get_ground_truth,
)
from benchmarks.scbench.loop import KvCompressionMode, run_scdq_example, truncate_first_prompt  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_example_id(eg_dict: dict, row_fallback: Any) -> int:
    v = eg_dict.get("id", row_fallback)
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(row_fallback)
        except Exception:
            return 0


@dataclass
class ScbenchConfig:
    dataset_config: str = "scbench_kv"
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    press_name: str = "snapkv"
    compression_ratio: float = 0.5
    key_channel_compression_ratio: Optional[float] = None
    threshold: Optional[float] = None
    max_seq_length: int = 160_000
    max_turns: int = 0
    num_eval_examples: int = -1
    fraction: float = 1.0
    seed: int = 42
    output_dir: str = "./results_scbench"
    use_chat_template: bool = True
    log_level: str = "INFO"
    fp8: bool = False
    model_kwargs: Optional[dict[str, Any]] = None
    #: ``context_prefill`` = compress long context once; ``decode_only`` = no prefill compression, compress decode KV only.
    kv_compression: KvCompressionMode = "decode_only"
    decode_compression_interval: int = 16
    decode_token_limit: int = 2048


def _git_revision() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT_EVAL.parent, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _setup_press(config: ScbenchConfig):
    press_name = config.press_name
    compression_ratio = config.compression_ratio
    key_channel_compression_ratio = config.key_channel_compression_ratio

    press = PRESS_REGISTRY[press_name]

    if isinstance(press, DuoAttentionPress):
        press.head_compression_ratio = compression_ratio
    elif isinstance(press, DMSPress):
        assert config.threshold is not None, "threshold must be set for DMSPress"
        press.threshold = config.threshold
    elif isinstance(press, ComposedPress):
        for ps in press.presses:
            if isinstance(ps, ThinKPress):
                assert key_channel_compression_ratio is not None
                ps.key_channel_compression_ratio = key_channel_compression_ratio
            elif hasattr(ps, "compression_ratio"):
                ps.compression_ratio = compression_ratio
    elif isinstance(press, ThinKPress):
        assert key_channel_compression_ratio is not None
        press.key_channel_compression_ratio = key_channel_compression_ratio
    else:
        if hasattr(press, "compression_ratio"):
            press.compression_ratio = compression_ratio

    return press


def run(config: ScbenchConfig | None = None, config_file: Optional[str] = None, **cli_overrides: Any) -> None:
    """Run SCBench SCDQ evaluation."""
    args_dict = asdict(ScbenchConfig())
    if config_file:
        p = Path(config_file)
        if p.exists():
            args_dict.update(yaml.safe_load(p.read_text()) or {})
    args_dict.update({k: v for k, v in cli_overrides.items() if v is not None})
    # Fire parses ``model=org/name`` incorrectly (splits on ``/``). Modal and scripts set this env instead.
    _env_model = os.environ.get("KV_PRESS_SCBENCH_MODEL", "").strip()
    if _env_model:
        args_dict["model"] = _env_model
    cfg = ScbenchConfig(**{k: v for k, v in args_dict.items() if k in ScbenchConfig.__dataclass_fields__})

    if cfg.press_name not in PRESS_REGISTRY:
        raise ValueError(f"Unknown press_name '{cfg.press_name}'. See PRESS_REGISTRY in evaluate_registry.py.")

    logging.basicConfig(level=getattr(logging, cfg.log_level.upper(), logging.INFO))

    torch.manual_seed(cfg.seed)

    data_name = cfg.dataset_config
    max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

    examples = load_dataset("microsoft/SCBench", data_name, split="test")
    df_examples = examples.to_pandas()
    if cfg.fraction < 1.0:
        df_examples = df_examples.sample(frac=cfg.fraction, random_state=cfg.seed)

    max_turn_size = len(df_examples.iloc[0]["multi_turns"])
    if cfg.max_turns > 0 and cfg.max_turns < max_turn_size:
        df_examples["multi_turns"] = df_examples["multi_turns"].apply(lambda m: m[: cfg.max_turns])

    if cfg.num_eval_examples > 0:
        df_examples = df_examples.head(cfg.num_eval_examples)

    press = None if cfg.press_name == "no_press" else _setup_press(cfg)

    model_kwargs = dict(cfg.model_kwargs or {})
    if cfg.fp8:
        model_kwargs["quantization_config"] = FineGrainedFP8Config()
    try:
        import flash_attn  # noqa: F401

        model_kwargs.setdefault("attn_implementation", "flash_attention_2")
    except ImportError:
        pass

    pipe = pipeline(
        "kv-press-text-generation",
        model=cfg.model,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
        device_map="auto",
    )
    model = pipe.model
    tokenizer = pipe.tokenizer
    model.eval()

    out_dir = (
        Path(cfg.output_dir)
        / f"{data_name}__{cfg.model.replace('/', '--')}__{cfg.press_name}__{cfg.compression_ratio:.2f}__kv-{cfg.kv_compression}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"

    rows: list[dict[str, Any]] = []
    max_turns_now = len(df_examples.iloc[0]["multi_turns"])

    if isinstance(max_new_tokens, dict):
        max_budget = sum(max_new_tokens.values()) * max_turns_now // 2
    else:
        max_budget = max_new_tokens * max_turns_now
    max_input_length = cfg.max_seq_length - max_budget - 1000
    # Negative values disable truncation in ``truncate_prompt_tokens`` (treated as unlimited).
    max_input_length = max(256, max_input_length)
    # Reserve headroom for multi-turn decode inside the model context window.
    ctx_cap = getattr(model.config, "max_position_embeddings", None)
    if isinstance(ctx_cap, int) and ctx_cap < 1_000_000:
        max_input_length = min(max_input_length, max(ctx_cap // 2, 256))

    for row_num, (_df_idx, eg) in tqdm(
        enumerate(df_examples.iterrows()), total=len(df_examples), desc="SCBench"
    ):
        eg_dict = eg.to_dict()
        encoded = create_scdq_prompt(
            eg_dict,
            data_name=data_name,
            tok=tokenizer,
            use_chat_template=cfg.use_chat_template,
            use_vllm=False,
        )
        encoded["prompts"] = [
            truncate_first_prompt(tokenizer, encoded["prompts"][0], max_input_length)
        ] + list(encoded["prompts"][1:])

        answers = run_scdq_example(
            model,
            tokenizer,
            encoded,
            press,
            max_new_tokens,
            max_context_tokens=max_input_length,
            kv_compression=cfg.kv_compression,
            decode_compression_interval=cfg.decode_compression_interval,
            decode_token_limit=cfg.decode_token_limit,
        )
        gts = get_ground_truth(eg_dict, data_name)
        for turn_idx, (ans, gt) in enumerate(zip(answers, gts)):
            case: dict[str, Any] = {
                "id": _safe_example_id(eg_dict, row_num),
                "turn_idx": turn_idx,
                "prediction": ans,
                "ground_truth": gt,
            }
            if data_name in ("scbench_summary_with_needles", "scbench_repoqa_and_kv", "scbench_kv_compressible"):
                case["task"] = eg_dict["multi_turns"][turn_idx]["task"]
            rows.append(case)

    pd.DataFrame(rows).to_json(preds_path, orient="records", lines=True)
    df_pred = pd.read_json(preds_path, lines=True)
    metrics = calculate_metrics(df_pred, data_name, cfg.model.split("/")[-1])
    metrics["git_revision"] = _git_revision()
    metrics["kvpress_version"] = _try_kvpress_version()
    metrics["config"] = asdict(cfg)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote %s and %s", preds_path, metrics_path)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


def _try_kvpress_version() -> str:
    try:
        from importlib.metadata import version

        return version("kvpress")
    except Exception:
        return "unknown"


def _cli_entrypoint(config_file: Optional[str] = None, **kwargs: Any) -> None:
    """CLI wrapper so Fire resolves plain function kwargs reliably."""
    run(config_file=config_file, **kwargs)


if __name__ == "__main__":
    Fire(_cli_entrypoint)
