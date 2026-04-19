# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modal runner for SCBench. Deploy secrets and volumes once, then invoke remotely.

Example::

    modal run evaluation/benchmarks/scbench/modal_app.py::run_scbench --dataset-config scbench_kv --press-name snapkv

Hugging Face auth for gated models (e.g. Llama 3): set ``HF_TOKEN`` or
``HUGGING_FACE_HUB_TOKEN`` in your **local** environment before ``modal run``; it is
forwarded into the container via :meth:`modal.Secret.from_local_environ`.
Alternatively, replace the secrets list with ``modal.Secret.from_name("your-secret")``
after creating that secret in the Modal dashboard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


def _find_repo_root() -> Path:
    """
    Resolve the kvpress repo root without assuming a fixed ``parents[N]`` depth.

    ``Path(__file__).parents[3]`` breaks on Modal when ``__file__`` is mounted at a
    shallower path (IndexError). Walking upward until ``pyproject.toml`` matches
    both local checkouts and ``/root/kvpress`` in the container.
    """
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").is_file():
            return d
    # Container layout from ``add_local_dir(..., remote_path="/root/kvpress")``
    fallback = Path("/root/kvpress")
    if (fallback / "pyproject.toml").is_file() or fallback.is_dir():
        return fallback
    raise RuntimeError(f"Could not locate repo root from {here}")


REPO_ROOT = _find_repo_root()


def _container_eval_python() -> str:
    """``uv sync`` installs deps into ``/root/kvpress/.venv``; Modal's ``sys.executable`` is the bare image Python."""
    venv_py = Path("/root/kvpress/.venv/bin/python")
    if venv_py.is_file():
        return str(venv_py)
    return sys.executable


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl")
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
            "evaluation/results_scbench",
        ],
    )
    .run_commands(
        "cd /root/kvpress && uv sync --extra eval",
    )
)

hf_cache = modal.Volume.from_name("kvpress-hf-cache", create_if_missing=True)

app = modal.App("kvpress-scbench", image=image)


def _read_hf_token_from_env_file(path: Path) -> str | None:
    """Parse ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` from a simple ``.env`` (no ``python-dotenv`` dep)."""
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
    """
    Resolve HF credentials for gated models (Llama, etc.).

    Order: repo ``.env`` → Hugging Face CLI token file → ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` in the shell.

    ``modal login`` only authenticates Modal; it does **not** download gated Hugging Face models.
    Optional: create ``modal secret create hf-secret HF_TOKEN=hf_...`` and add
    ``return [modal.Secret.from_name("hf-secret")]`` here if you prefer named secrets.
    """
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
    modal_hf = os.environ.get("MODAL_HF_SECRET_NAME")
    if modal_hf:
        return [modal.Secret.from_name(modal_hf)]
    raise RuntimeError(
        "No Hugging Face token found. Modal login is not enough for Llama weights. Do one of: "
        "(1) `huggingface-cli login` on this machine, "
        "(2) add `HF_TOKEN=hf_...` to a `.env` file in the repo root, "
        "(3) `modal secret create hf-secret HF_TOKEN=hf_...` then "
        "`$env:MODAL_HF_SECRET_NAME='hf-secret'` (PowerShell) before `modal run`, "
        "or set `HF_TOKEN` in the shell before `modal run`."
    )


@app.function(
    gpu="A100",
    timeout=86400,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=_huggingface_secret(),
)
def run_scbench(
    dataset_config: str = "scbench_kv",
    # Default to an open model so Modal smoke runs work without Llama gating; override for Llama 3.
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    press_name: str = "snapkv",
    compression_ratio: float = 0.5,
    num_eval_examples: int = 1,
    # Keep moderate so ``truncate_first_prompt`` caps long SCBench JSON (full 160k blows memory on one prefill).
    max_seq_length: int = 8192,
    kv_compression: str = "decode_only",
    decode_compression_interval: int = 16,
    decode_token_limit: int = 2048,
) -> str:
    """Run :mod:`run_scbench` on a Modal GPU worker."""
    env = os.environ.copy()
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TRANSFORMERS_CACHE"] = env["HF_HOME"]
    # Modal injects secrets into the worker env; ensure the child process inherits them explicitly.
    for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if _k in os.environ and os.environ[_k]:
            env[_k] = os.environ[_k]
    # Fire breaks on ``model=org/name``; run_scbench reads ``KV_PRESS_SCBENCH_MODEL``.
    env["KV_PRESS_SCBENCH_MODEL"] = model
    cmd = [
        _container_eval_python(),
        "/root/kvpress/evaluation/benchmarks/scbench/run_scbench.py",
        # Fire v0.6 requires --key=value (double-dash) to populate **kwargs;
        # bare `key=value` positional args are rejected with exit code 2.
        f"--dataset_config={dataset_config}",
        f"--press_name={press_name}",
        f"--compression_ratio={compression_ratio}",
        f"--num_eval_examples={num_eval_examples}",
        f"--max_seq_length={max_seq_length}",
        f"--kv_compression={kv_compression}",
        f"--decode_compression_interval={decode_compression_interval}",
        f"--decode_token_limit={decode_token_limit}",
        f"--output_dir=/root/kvpress/evaluation/results_scbench_modal",
    ]
    subprocess.check_call(cmd, env=env, cwd="/root/kvpress/evaluation")
    hf_cache.commit()
    return "ok"


@app.local_entrypoint()
def main(
    dataset_config: str = "scbench_kv",
    press_name: str = "snapkv",
    compression_ratio: float = 0.5,
    num_eval_examples: int = 1,
) -> None:
    run_scbench.remote(
        dataset_config=dataset_config,
        press_name=press_name,
        compression_ratio=compression_ratio,
        num_eval_examples=num_eval_examples,
    )
