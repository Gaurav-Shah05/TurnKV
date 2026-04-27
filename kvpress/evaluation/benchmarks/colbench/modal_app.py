# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modal runner for ColBench (Backend) live-loop runs.

Example:

    modal run evaluation/benchmarks/colbench/modal_app.py::main \\
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

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEFAULT_FEEDBACK_MODEL = "google/gemma-4-26B-A4B-it"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_3"
DEFAULT_FEEDBACK_ATTN_IMPLEMENTATION = "vllm_triton"
MODAL_GPU_SPEC = os.environ.get("KV_PRESS_COLBENCH_MODAL_GPU", "H200")
DEFAULT_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES = os.environ.get(
    "KV_PRESS_COLBENCH_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES",
    "1" if ":2" in MODAL_GPU_SPEC else "",
) or None

# Image constants are kept identical to convcodeworld so the FA3 build layer
# is reusable across both benchmarks - changing any of these values in just
# one folder forces an unnecessary rebuild for the other.
CUDA_BASE_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
MODAL_TORCH_VERSION = "2.8.0"
FLASH_ATTN3_REF = "v2.8.3"
TRANSFORMERS_GIT_REF = "bc4b330451d0e3e33f4ac63593ed9f245227712e"
TRANSFORMERS_SOURCE = (
    f"git+https://github.com/huggingface/transformers.git@{TRANSFORMERS_GIT_REF}"
)
FLASH_ATTN3_BUILD_ENV = (
    "CC=gcc CXX=g++ CUDAHOSTCXX=g++ "
    "FLASH_ATTENTION_DISABLE_BACKWARD=TRUE "
    "FLASH_ATTENTION_DISABLE_SM80=TRUE "
    "FLASH_ATTENTION_DISABLE_SPLIT=TRUE "
    "FLASH_ATTENTION_DISABLE_FP16=TRUE "
    "FLASH_ATTENTION_DISABLE_FP8=TRUE "
    "MAX_JOBS=8 NVCC_THREADS=2"
)
MODAL_EVAL_REQUIREMENTS = (
    "numpy>=2.0.0,<3",
    "datasets>=2.21.0",
    "pandas>=2.2.2,<3",
    "accelerate>=1.0.0,<2",
    "requests>=2.32.3,<3",
    "beautifulsoup4>=4.12.3,<5",
    "faker>=25.0.0,<40",
    "natsort>=8.4.0,<9",
    "openpyxl>=3.1.5,<4",
    "scikit-learn>=1.5.0,<2",
    "seaborn>=0.13.2,<0.14",
    "statsmodels>=0.14.2,<0.15",
    "xlsxwriter>=3.2.0,<4",
    "cachetools>=5.5.2,<6",
    "fire>=0.6.0,<0.7",
    "rouge>=1.0.1,<2",
    "nltk>=3.9.1,<4",
    "tqdm>=4.66.4,<5",
    "scipy>=1.13.1,<2",
    "bert-score>=0.3.13,<0.4",
    "jieba>=0.42.1",
    "fuzzywuzzy>=0.18.0",
    "pyyaml>=6.0.1,<7",
    "sentencepiece>=0.2.0,<0.3",
    "protobuf>=5.27.2,<6",
    "einops>=0.8.0,<1",
)
VLLM_INSTALL_COMMAND = (
    "uv pip install --python /root/kvpress/.venv/bin/python --upgrade --pre vllm "
    "--extra-index-url https://wheels.vllm.ai/nightly/cu129 "
    "--extra-index-url https://download.pytorch.org/whl/cu129 "
    "--index-strategy unsafe-best-match"
)


def _shell_requirements(requirements: tuple[str, ...]) -> str:
    return " ".join(f"'{requirement}'" for requirement in requirements)


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


base_image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.11")
    .apt_install("git", "curl", "util-linux", "build-essential", "ninja-build", "r-base-core")
    .pip_install("uv")
    .workdir("/root/kvpress")
    .run_commands("uv venv /root/kvpress/.venv")
    .run_commands(
        f"uv pip install --python /root/kvpress/.venv/bin/python torch=={MODAL_TORCH_VERSION} "
        "wheel setuptools setuptools_scm packaging ninja einops"
    )
    .run_commands(VLLM_INSTALL_COMMAND)
    .run_commands(
        "git clone --depth=1 --branch "
        f"{FLASH_ATTN3_REF} https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attention && "
        "cd /tmp/flash-attention/hopper && "
        f"{FLASH_ATTN3_BUILD_ENV} "
        "/root/kvpress/.venv/bin/python setup.py bdist_wheel -d /opt/fa3-wheelhouse && "
        "rm -rf /tmp/flash-attention",
    )
    .run_commands(
        "uv pip install --python /root/kvpress/.venv/bin/python "
        + _shell_requirements(MODAL_EVAL_REQUIREMENTS),
        f"uv pip install --python /root/kvpress/.venv/bin/python --upgrade '{TRANSFORMERS_SOURCE}'",
        "/root/kvpress/.venv/bin/python -c "
        "'import transformers; print(f\"Transformers {transformers.__version__} installed\")'",
        "uv pip install --python /root/kvpress/.venv/bin/python "
        "--no-index --find-links=/opt/fa3-wheelhouse --no-deps --prerelease=allow flash_attn_3",
        "/root/kvpress/.venv/bin/python -c "
        "'import flash_attn_3, flash_attn_interface; print(\"FlashAttention-3 available\")'",
    )
)

image = (
    base_image
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
        "uv pip install --python /root/kvpress/.venv/bin/python --no-deps -e /root/kvpress",
        "/root/kvpress/.venv/bin/python -c "
        "'import flash_attn_3, flash_attn_interface; print(\"FlashAttention-3 available\")'",
    )
)

hf_cache = modal.Volume.from_name("kvpress-hf-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("kvpress-colbench-results", create_if_missing=True)

app = modal.App("kvpress-colbench-live", image=image)


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


def _translate_task_ids(value: Optional[str]) -> Optional[str]:
    """Rewrite a host ``@<path>`` task-id reference to its in-container form.

    Modal mounts ``REPO_ROOT`` at ``/root/kvpress`` inside the worker, so a
    local path like ``./evaluation/benchmarks/colbench/splits/shards/foo.json``
    becomes ``/root/kvpress/evaluation/benchmarks/colbench/splits/shards/foo.json``.
    """
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    if not text.startswith("@"):
        return text
    local_path = Path(text[1:]).expanduser().resolve()
    if not local_path.is_file():
        raise FileNotFoundError(f"task_ids @<path> points to missing file: {local_path}")
    try:
        rel = local_path.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"task_ids @<path> {local_path} is not inside REPO_ROOT={REPO_ROOT}; "
            "Modal only mounts REPO_ROOT into /root/kvpress, so the file "
            "would not exist inside the container."
        ) from exc
    container_path = "/root/kvpress/" + rel.as_posix()
    return f"@{container_path}"


def _needs_hf_token(model_name: str | None) -> bool:
    if not model_name:
        return False
    normalized = model_name.lower()
    return normalized.startswith(("meta-llama/", "google/gemma", "google/txgemma"))


def _is_vllm_triton_attention(implementation: Optional[str]) -> bool:
    normalized = str(implementation or "").strip().lower().replace("-", "_")
    return normalized in {
        "vllm_triton",
        "vllm_triton_attn",
        "vllm_triton_attention",
        "triton_attn",
    }


@app.function(
    gpu=MODAL_GPU_SPEC,
    timeout=86400,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/kvpress/evaluation/results_colbench_live_modal": results_volume,
    },
    secrets=_huggingface_secret(),
)
def run_colbench_live(
    model: str = DEFAULT_MODEL,
    feedback_model: Optional[str] = DEFAULT_FEEDBACK_MODEL,
    attn_implementation: Optional[str] = DEFAULT_ATTN_IMPLEMENTATION,
    feedback_attn_implementation: Optional[str] = DEFAULT_FEEDBACK_ATTN_IMPLEMENTATION,
    feedback_vllm_port: int = 8001,
    feedback_vllm_cuda_visible_devices: Optional[str] = DEFAULT_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES,
    feedback_vllm_max_model_len: int = 32768,
    feedback_vllm_gpu_memory_utilization: float = 0.75,
    feedback_vllm_start_timeout_s: int = 1800,
    full_kv_cache: bool = False,
    require_flashdecode: bool = False,
    error_on_kv_cache_vram_exhaustion: bool = False,
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
    loyalty_update_every: Optional[int] = None,
    alpha_floor_len: Optional[float] = None,
    min_floor_tokens: Optional[int] = None,
    colbench_split: str = "backend",
    dataset_name: str = "facebook/collaborative_agent_bench",
    dataset_subset: str = "backend",
    hf_split: str = "train",
    num_eval_examples: int = 1,
    fraction: float = 1.0,
    task_ids: Optional[str] = None,
    max_turns: int = 10,
    max_questions_before_submit: int = 9,
    max_new_tokens: int = 1024,
    code_generation_until_eos: bool = False,
    verbal_feedback_max_new_tokens: int = 256,
    global_budget: int = 4500,
    local_budget: int = 4096,
    decode_compression_interval: int = 128,
    decode_hidden_states_buffer_size: int = 256,
    early_stop_on_pass: bool = True,
    network_isolation: str = "auto",
    cot: bool = False,
    log_level: str = "INFO",
    output_subdir: Optional[str] = None,
) -> str:
    env = os.environ.copy()
    env["HF_HOME"] = "/root/.cache/huggingface"
    env["TRANSFORMERS_CACHE"] = env["HF_HOME"]
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    hf_token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or env.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    needs_hf_token = _needs_hf_token(model) or _needs_hf_token(feedback_model)
    if needs_hf_token and not hf_token:
        raise RuntimeError(
            "No Hugging Face token is available inside the Modal worker for gated model weights. "
            "Run `huggingface-cli login`, set HF_TOKEN locally, add HF_TOKEN to .env, or set "
            "MODAL_HF_SECRET_NAME to a Modal secret before `modal run`."
        )
    print(f"Hugging Face token available in Modal worker: {bool(hf_token)}", flush=True)
    env["KV_PRESS_COLBENCH_MODEL"] = model
    if feedback_model:
        env["KV_PRESS_COLBENCH_FEEDBACK_MODEL"] = feedback_model
    if attn_implementation:
        env["KV_PRESS_COLBENCH_ATTN_IMPLEMENTATION"] = attn_implementation
    if feedback_attn_implementation:
        env["KV_PRESS_COLBENCH_FEEDBACK_ATTN_IMPLEMENTATION"] = feedback_attn_implementation
    if _is_vllm_triton_attention(feedback_attn_implementation):
        env["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"
        env.setdefault("VLLM_V1_USE_PREFILL_DECODE_ATTENTION", "0")
        env.setdefault("VLLM_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_MOE_USE_DEEP_GEMM", "0")
        env.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
        if feedback_vllm_cuda_visible_devices:
            env["KV_PRESS_COLBENCH_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES"] = (
                feedback_vllm_cuda_visible_devices
            )

    base_output_dir = "/root/kvpress/evaluation/results_colbench_live_modal"
    if output_subdir:
        cleaned = output_subdir.strip("/")
        if not cleaned or any(ch in cleaned for ch in ("..", "\\")) or any(
            not (ch.isalnum() or ch in "._-/") for ch in cleaned
        ):
            raise ValueError(f"output_subdir {output_subdir!r} is invalid; allowed: [A-Za-z0-9._-/].")
        output_dir = f"{base_output_dir}/{cleaned}"
    else:
        output_dir = base_output_dir

    cmd = [
        _container_eval_python(),
        "/root/kvpress/evaluation/benchmarks/colbench/live_loop.py",
        f"--model={model}",
        f"--press_name={press_name}",
        f"--compression_ratio={compression_ratio}",
        f"--colbench_split={colbench_split}",
        f"--dataset_name={dataset_name}",
        f"--dataset_subset={dataset_subset}",
        f"--hf_split={hf_split}",
        f"--num_eval_examples={num_eval_examples}",
        f"--fraction={fraction}",
        f"--max_turns={max_turns}",
        f"--max_questions_before_submit={max_questions_before_submit}",
        f"--max_new_tokens={max_new_tokens}",
        f"--code_generation_until_eos={code_generation_until_eos}",
        f"--verbal_feedback_max_new_tokens={verbal_feedback_max_new_tokens}",
        f"--global_budget={global_budget}",
        f"--local_budget={local_budget}",
        f"--decode_compression_interval={decode_compression_interval}",
        f"--decode_hidden_states_buffer_size={decode_hidden_states_buffer_size}",
        f"--early_stop_on_pass={early_stop_on_pass}",
        f"--network_isolation={network_isolation}",
        f"--cot={cot}",
        f"--log_level={log_level}",
        f"--output_dir={output_dir}",
    ]
    _append_optional_flag(cmd, "feedback_model", feedback_model)
    _append_optional_flag(cmd, "attn_implementation", attn_implementation)
    _append_optional_flag(cmd, "feedback_attn_implementation", feedback_attn_implementation)
    _append_optional_flag(cmd, "feedback_vllm_port", feedback_vllm_port)
    _append_optional_flag(cmd, "feedback_vllm_cuda_visible_devices", feedback_vllm_cuda_visible_devices)
    _append_optional_flag(cmd, "feedback_vllm_max_model_len", feedback_vllm_max_model_len)
    _append_optional_flag(cmd, "feedback_vllm_gpu_memory_utilization", feedback_vllm_gpu_memory_utilization)
    _append_optional_flag(cmd, "feedback_vllm_start_timeout_s", feedback_vllm_start_timeout_s)
    if full_kv_cache:
        cmd.append("--full_kv_cache=True")
    if require_flashdecode:
        cmd.append("--require_flashdecode=True")
    if error_on_kv_cache_vram_exhaustion:
        cmd.append("--error_on_kv_cache_vram_exhaustion=True")
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
    _append_optional_flag(cmd, "loyalty_update_every", loyalty_update_every)
    _append_optional_flag(cmd, "alpha_floor_len", alpha_floor_len)
    _append_optional_flag(cmd, "min_floor_tokens", min_floor_tokens)
    _append_optional_flag(cmd, "task_ids", task_ids)
    try:
        subprocess.check_call(cmd, env=env, cwd="/root/kvpress/evaluation")
    finally:
        hf_cache.commit()
        results_volume.commit()
    return f"ok: colbench:{press_name}"


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    feedback_model: Optional[str] = DEFAULT_FEEDBACK_MODEL,
    attn_implementation: Optional[str] = DEFAULT_ATTN_IMPLEMENTATION,
    feedback_attn_implementation: Optional[str] = DEFAULT_FEEDBACK_ATTN_IMPLEMENTATION,
    feedback_vllm_port: int = 8001,
    feedback_vllm_cuda_visible_devices: Optional[str] = DEFAULT_FEEDBACK_VLLM_CUDA_VISIBLE_DEVICES,
    feedback_vllm_max_model_len: int = 32768,
    feedback_vllm_gpu_memory_utilization: float = 0.75,
    feedback_vllm_start_timeout_s: int = 1800,
    full_kv_cache: bool = False,
    require_flashdecode: bool = False,
    error_on_kv_cache_vram_exhaustion: bool = False,
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
    loyalty_update_every: Optional[int] = None,
    alpha_floor_len: Optional[float] = None,
    min_floor_tokens: Optional[int] = None,
    colbench_split: str = "backend",
    dataset_name: str = "facebook/collaborative_agent_bench",
    dataset_subset: str = "backend",
    hf_split: str = "train",
    num_eval_examples: int = 1,
    fraction: float = 1.0,
    task_ids: Optional[str] = None,
    max_turns: int = 10,
    max_questions_before_submit: int = 9,
    max_new_tokens: int = 1024,
    code_generation_until_eos: bool = False,
    verbal_feedback_max_new_tokens: int = 256,
    global_budget: int = 4500,
    local_budget: int = 4096,
    decode_compression_interval: int = 128,
    decode_hidden_states_buffer_size: int = 256,
    early_stop_on_pass: bool = True,
    network_isolation: str = "auto",
    cot: bool = False,
    log_level: str = "INFO",
    output_subdir: Optional[str] = None,
    detach_remote: bool = False,
) -> None:
    task_ids = _translate_task_ids(task_ids)
    for press_name in [p.strip() for p in press_names.split(",") if p.strip()]:
        kwargs = dict(
            model=model,
            feedback_model=feedback_model,
            attn_implementation=attn_implementation,
            feedback_attn_implementation=feedback_attn_implementation,
            feedback_vllm_port=feedback_vllm_port,
            feedback_vllm_cuda_visible_devices=feedback_vllm_cuda_visible_devices,
            feedback_vllm_max_model_len=feedback_vllm_max_model_len,
            feedback_vllm_gpu_memory_utilization=feedback_vllm_gpu_memory_utilization,
            feedback_vllm_start_timeout_s=feedback_vllm_start_timeout_s,
            full_kv_cache=full_kv_cache,
            require_flashdecode=require_flashdecode,
            error_on_kv_cache_vram_exhaustion=error_on_kv_cache_vram_exhaustion,
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
            loyalty_update_every=loyalty_update_every,
            alpha_floor_len=alpha_floor_len,
            min_floor_tokens=min_floor_tokens,
            colbench_split=colbench_split,
            dataset_name=dataset_name,
            dataset_subset=dataset_subset,
            hf_split=hf_split,
            num_eval_examples=num_eval_examples,
            fraction=fraction,
            task_ids=task_ids,
            max_turns=max_turns,
            max_questions_before_submit=max_questions_before_submit,
            max_new_tokens=max_new_tokens,
            code_generation_until_eos=code_generation_until_eos,
            verbal_feedback_max_new_tokens=verbal_feedback_max_new_tokens,
            global_budget=global_budget,
            local_budget=local_budget,
            decode_compression_interval=decode_compression_interval,
            decode_hidden_states_buffer_size=decode_hidden_states_buffer_size,
            early_stop_on_pass=early_stop_on_pass,
            network_isolation=network_isolation,
            cot=cot,
            log_level=log_level,
            output_subdir=output_subdir,
        )
        if detach_remote:
            call = run_colbench_live.spawn(**kwargs)
            print(
                f"Spawned ColBench run for {press_name}: "
                f"{call.object_id} {call.get_dashboard_url()}",
                flush=True,
            )
        else:
            run_colbench_live.remote(**kwargs)
