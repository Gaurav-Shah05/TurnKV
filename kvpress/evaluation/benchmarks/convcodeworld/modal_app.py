# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modal runner for ConvCodeWorld live-loop runs.

Example:

    modal run evaluation/benchmarks/convcodeworld/modal_app.py::main \\
        --press-names snapkv,streaming_llm,expected_attention \\
        --num-eval-examples 10
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import modal


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").is_file():
            return d
    fallback = Path("/root/kvpress")
    if (fallback / "pyproject.toml").is_file() or fallback.is_dir():
        return fallback
    raise RuntimeError(f"Could not locate kvpress repo root from {here}")


REPO_ROOT = _find_repo_root()


def _container_eval_python() -> str:
    venv_py = Path("/root/kvpress/.venv/bin/python")
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "util-linux")
    .pip_install("uv")
    .workdir("/root/kvpress")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/kvpress",
        copy=True,
        ignore=[
            ".git",
            ".venv",
            "**/__pycache__",
            "*.pyc",
            ".pytest_cache",
            "evaluation/results*",
        ],
    )
    .run_commands(
        "cd /root/kvpress && uv sync --extra eval",
    )
)

hf_cache = modal.Volume.from_name("kvpress-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("kvpress-convcodeworld-results", create_if_missing=True)

app = modal.App("kvpress-convcodeworld-live", image=image)


def _read_hf_token_from_env_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for key in ("HF_TOKEN=", "HUGGING_FACE_HUB_TOKEN="):
            if line.startswith(key):
                val = line[len(key) :].strip().strip('"').strip("'")
                return val or None
    return None


def _huggingface_secret() -> list[modal.Secret]:
    modal_hf = os.environ.get("MODAL_HF_SECRET_NAME")
    if modal_hf:
        return [modal.Secret.from_name(modal_hf)]

    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.is_file():
        token = _read_hf_token_from_env_file(dotenv_path)
        if token:
            return [modal.Secret.from_dict({"HF_TOKEN": token})]

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        raw = token_path.read_text(encoding="utf-8").strip()
        if raw.startswith("hf_"):
            return [modal.Secret.from_dict({"HF_TOKEN": raw})]

    if os.environ.get("HF_TOKEN"):
        return [modal.Secret.from_local_environ(["HF_TOKEN"])]
    if os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return [modal.Secret.from_local_environ(["HUGGING_FACE_HUB_TOKEN"])]
    return []


def _append_optional_flag(cmd: list[str], name: str, value: object | None) -> None:
    if value is not None:
        cmd.append(f"--{name}={value}")


@app.function(
    gpu="A100",
    timeout=86400,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/kvpress/evaluation/results_convcodeworld_live_modal": results_volume,
    },
    secrets=_huggingface_secret(),
)
def run_convcodeworld_live(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    press_name: str = "snapkv",
    compression_ratio: float = 0.5,
    key_channel_compression_ratio: Optional[float] = None,
    threshold: Optional[float] = None,
    snapkv_window_size: Optional[int] = None,
    snapkv_kernel_size: Optional[int] = None,
    streaming_llm_n_sink: Optional[int] = None,
    expected_attention_n_future_positions: Optional[int] = None,
    expected_attention_n_sink: Optional[int] = None,
    expected_attention_use_covariance: Optional[bool] = None,
    expected_attention_use_vnorm: Optional[bool] = None,
    expected_attention_epsilon: Optional[float] = None,
    alpha_floor: Optional[float] = None,
    alpha_anchor: Optional[float] = None,
    alpha_loyalty: Optional[float] = None,
    anchor_beta: Optional[float] = None,
    floor_gamma: Optional[float] = None,
    loyalty_top_p: Optional[float] = None,
    alpha_floor_len: Optional[float] = None,
    min_floor_tokens: Optional[int] = None,
    feedback_config: str = "CF_EF_UNIT_SNF",
    num_eval_examples: int = 1,
    fraction: float = 1.0,
    max_turns: int = 10,
    max_new_tokens: int = 1024,
    verbal_feedback_max_new_tokens: int = 256,
    global_budget: int = 4500,
    local_budget: int = 4096,
    decode_compression_interval: int = 128,
    decode_hidden_states_buffer_size: int = 256,
    early_stop_on_pass: bool = True,
    network_isolation: str = "auto",
    cot: bool = True,
) -> str:
    env = os.environ.copy()
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TRANSFORMERS_CACHE"] = env["HF_HOME"]
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    hf_token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    needs_hf_token = model.lower().startswith("meta-llama/")
    if needs_hf_token and not hf_token:
        raise RuntimeError(
            "No Hugging Face token is available inside the Modal worker for gated Llama weights. "
            "Run `huggingface-cli login`, set HF_TOKEN locally, add HF_TOKEN to .env, or set "
            "MODAL_HF_SECRET_NAME to a Modal secret before `modal run`."
        )
    print(f"Hugging Face token available in Modal worker: {bool(hf_token)}", flush=True)
    env["KV_PRESS_CONVCODEWORLD_MODEL"] = model

    cmd = [
        _container_eval_python(),
        "/root/kvpress/evaluation/benchmarks/convcodeworld/live_loop.py",
        f"--press_name={press_name}",
        f"--compression_ratio={compression_ratio}",
        f"--feedback_config={feedback_config}",
        f"--num_eval_examples={num_eval_examples}",
        f"--fraction={fraction}",
        f"--max_turns={max_turns}",
        f"--max_new_tokens={max_new_tokens}",
        f"--verbal_feedback_max_new_tokens={verbal_feedback_max_new_tokens}",
        f"--global_budget={global_budget}",
        f"--local_budget={local_budget}",
        f"--decode_compression_interval={decode_compression_interval}",
        f"--decode_hidden_states_buffer_size={decode_hidden_states_buffer_size}",
        f"--early_stop_on_pass={early_stop_on_pass}",
        f"--network_isolation={network_isolation}",
        f"--cot={cot}",
        "--output_dir=/root/kvpress/evaluation/results_convcodeworld_live_modal",
    ]
    _append_optional_flag(cmd, "key_channel_compression_ratio", key_channel_compression_ratio)
    _append_optional_flag(cmd, "threshold", threshold)
    _append_optional_flag(cmd, "snapkv_window_size", snapkv_window_size)
    _append_optional_flag(cmd, "snapkv_kernel_size", snapkv_kernel_size)
    _append_optional_flag(cmd, "streaming_llm_n_sink", streaming_llm_n_sink)
    _append_optional_flag(cmd, "expected_attention_n_future_positions", expected_attention_n_future_positions)
    _append_optional_flag(cmd, "expected_attention_n_sink", expected_attention_n_sink)
    _append_optional_flag(cmd, "expected_attention_use_covariance", expected_attention_use_covariance)
    _append_optional_flag(cmd, "expected_attention_use_vnorm", expected_attention_use_vnorm)
    _append_optional_flag(cmd, "expected_attention_epsilon", expected_attention_epsilon)
    _append_optional_flag(cmd, "alpha_floor", alpha_floor)
    _append_optional_flag(cmd, "alpha_anchor", alpha_anchor)
    _append_optional_flag(cmd, "alpha_loyalty", alpha_loyalty)
    _append_optional_flag(cmd, "anchor_beta", anchor_beta)
    _append_optional_flag(cmd, "floor_gamma", floor_gamma)
    _append_optional_flag(cmd, "loyalty_top_p", loyalty_top_p)
    _append_optional_flag(cmd, "alpha_floor_len", alpha_floor_len)
    _append_optional_flag(cmd, "min_floor_tokens", min_floor_tokens)
    subprocess.check_call(cmd, env=env, cwd="/root/kvpress/evaluation")
    hf_cache.commit()
    results_volume.commit()
    return f"ok: {press_name}"


@app.local_entrypoint()
def main(
    model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    press_names: str = "snapkv,streaming_llm,expected_attention",
    compression_ratio: float = 0.5,
    key_channel_compression_ratio: Optional[float] = None,
    threshold: Optional[float] = None,
    snapkv_window_size: Optional[int] = None,
    snapkv_kernel_size: Optional[int] = None,
    streaming_llm_n_sink: Optional[int] = None,
    expected_attention_n_future_positions: Optional[int] = None,
    expected_attention_n_sink: Optional[int] = None,
    expected_attention_use_covariance: Optional[bool] = None,
    expected_attention_use_vnorm: Optional[bool] = None,
    expected_attention_epsilon: Optional[float] = None,
    alpha_floor: Optional[float] = None,
    alpha_anchor: Optional[float] = None,
    alpha_loyalty: Optional[float] = None,
    anchor_beta: Optional[float] = None,
    floor_gamma: Optional[float] = None,
    loyalty_top_p: Optional[float] = None,
    alpha_floor_len: Optional[float] = None,
    min_floor_tokens: Optional[int] = None,
    feedback_config: str = "CF_EF_UNIT_SNF",
    num_eval_examples: int = 1,
    fraction: float = 1.0,
    max_turns: int = 10,
    max_new_tokens: int = 1024,
    verbal_feedback_max_new_tokens: int = 256,
    global_budget: int = 4500,
    local_budget: int = 4096,
    decode_compression_interval: int = 128,
    decode_hidden_states_buffer_size: int = 256,
    early_stop_on_pass: bool = True,
    network_isolation: str = "auto",
    cot: bool = True,
) -> None:
    for press_name in [p.strip() for p in press_names.split(",") if p.strip()]:
        run_convcodeworld_live.remote(
            model=model,
            press_name=press_name,
            compression_ratio=compression_ratio,
            key_channel_compression_ratio=key_channel_compression_ratio,
            threshold=threshold,
            snapkv_window_size=snapkv_window_size,
            snapkv_kernel_size=snapkv_kernel_size,
            streaming_llm_n_sink=streaming_llm_n_sink,
            expected_attention_n_future_positions=expected_attention_n_future_positions,
            expected_attention_n_sink=expected_attention_n_sink,
            expected_attention_use_covariance=expected_attention_use_covariance,
            expected_attention_use_vnorm=expected_attention_use_vnorm,
            expected_attention_epsilon=expected_attention_epsilon,
            alpha_floor=alpha_floor,
            alpha_anchor=alpha_anchor,
            alpha_loyalty=alpha_loyalty,
            anchor_beta=anchor_beta,
            floor_gamma=floor_gamma,
            loyalty_top_p=loyalty_top_p,
            alpha_floor_len=alpha_floor_len,
            min_floor_tokens=min_floor_tokens,
            feedback_config=feedback_config,
            num_eval_examples=num_eval_examples,
            fraction=fraction,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            verbal_feedback_max_new_tokens=verbal_feedback_max_new_tokens,
            global_budget=global_budget,
            local_budget=local_budget,
            decode_compression_interval=decode_compression_interval,
            decode_hidden_states_buffer_size=decode_hidden_states_buffer_size,
            early_stop_on_pass=early_stop_on_pass,
            network_isolation=network_isolation,
            cot=cot,
        )
