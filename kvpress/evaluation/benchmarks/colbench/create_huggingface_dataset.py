# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialise the ColBench Backend split into the flat schema this folder uses.

Upstream: ``facebook/collaborative_agent_bench`` (Meta FAIR, SWEET-RL,
March 2025). Unlike ConvCodeWorld there is no nested per-iteration unrolling
to do - each row is a single self-contained Backend task. We flatten the
dataset to a stable column set that ``live_loop.py`` consumes directly:

    task_id            : str    stable per-task identifier
    description        : str    natural-language goal (the prompt the agent sees)
    reference_solution : str    Python solution (private; given to the simulator)
    private_tests      : str    Python ``unittest`` source (private)
    entry_point        : str    function name the tests import
    code_prompt        : str    optional stub with imports + signature
    metadata           : dict   any remaining upstream fields, kept for provenance

Field-name resilience: the upstream HF schema is not 100% stable across
release snapshots. We accept several aliases (``task``/``problem``/``description``
for the prompt, ``solution``/``canonical_solution`` for the reference, etc.)
and fall through gracefully when fields are missing.

Usage:
    # Local save under ./colbench_backend_flat/
    python create_huggingface_dataset.py

    # Push to HF Hub
    HF_REPO_ID=your-org/colbench-backend-kvpress python create_huggingface_dataset.py --push
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from fire import Fire


DEFAULT_DATASET = "facebook/collaborative_agent_bench"
DEFAULT_SUBSET = "backend"


def _first_present(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
        elif isinstance(value, (list, tuple)):
            joined = "\n\n".join(str(item) for item in value if item)
            if joined.strip():
                return joined
        else:
            return str(value)
    return ""


def _flatten_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    task_id = _first_present(row, "task_id", "id", "uid", "name") or f"colbench/backend/{idx}"
    description = _first_present(
        row,
        "description",
        "instruction",
        "problem",
        "task",
        "instruct_prompt",
        "prompt",
    )
    reference = _first_present(
        row,
        "reference_solution",
        "canonical_solution",
        "solution",
        "code",
    )
    private_tests = _first_present(
        row,
        "private_tests",
        "hidden_tests",
        "tests",
        "test",
    )
    entry_point = _first_present(row, "entry_point", "function_name", "func_name")
    code_prompt = _first_present(row, "code_prompt", "starter_code", "stub")

    consumed = {
        "task_id", "id", "uid", "name",
        "description", "instruction", "problem", "task", "instruct_prompt", "prompt",
        "reference_solution", "canonical_solution", "solution", "code",
        "private_tests", "hidden_tests", "tests", "test",
        "entry_point", "function_name", "func_name",
        "code_prompt", "starter_code", "stub",
    }
    metadata = {k: v for k, v in row.items() if k not in consumed}

    return {
        "task_id": str(task_id),
        "description": description,
        "reference_solution": reference,
        "private_tests": private_tests,
        "entry_point": entry_point,
        "code_prompt": code_prompt,
        "metadata": json.dumps(metadata, default=str) if metadata else "",
    }


def main(
    output_dir: str = "./colbench_backend_flat",
    dataset: str = DEFAULT_DATASET,
    subset: str = DEFAULT_SUBSET,
    split: str = "train",
    push: bool = False,
    repo_id: str | None = None,
):
    """Load ColBench Backend, flatten to the per-task schema, save locally / push.

    Parameters
    ----------
    output_dir : str
        Local directory to write the flattened dataset to.
    dataset : str
        HF dataset id. Defaults to ``facebook/collaborative_agent_bench``.
    subset : str
        Configuration name (e.g. ``backend``). Pass ``""`` to skip.
    split : str
        HF split name. Most release snapshots use ``train``; some use ``test``.
    push : bool
        If True, also push to HF Hub under ``repo_id`` (or ``$HF_REPO_ID``).
    repo_id : str | None
        HF Hub repo id for the push.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_id = repo_id or os.environ.get("HF_REPO_ID")

    load_kwargs: dict[str, Any] = {"split": split}
    if subset:
        try:
            src = load_dataset(dataset, subset, **load_kwargs)
        except (ValueError, FileNotFoundError):
            src = load_dataset(dataset, **load_kwargs)
    else:
        src = load_dataset(dataset, **load_kwargs)

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(src):
        flat = _flatten_row(dict(row), idx)
        if not flat["description"]:
            print(f"[warn] skipping {flat['task_id']}: empty description")
            continue
        if not flat["private_tests"] and not flat["reference_solution"]:
            print(f"[warn] skipping {flat['task_id']}: empty tests AND empty reference")
            continue
        rows.append(flat)

    ds = Dataset.from_list(rows)
    print(f"Flattened {len(ds)} ColBench {subset!r} tasks (from {dataset!r}/{split!r}).")
    save_path = out / (subset or "default")
    ds.save_to_disk(str(save_path))
    print(f"Saved to {save_path}.")

    if push:
        if not repo_id:
            raise ValueError("push=True but no repo_id or HF_REPO_ID provided")
        ds.push_to_hub(repo_id, config_name=subset or "default", split="train")
        print(f"Pushed to {repo_id}.")


if __name__ == "__main__":
    Fire(main)
