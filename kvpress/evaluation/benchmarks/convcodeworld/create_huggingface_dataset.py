# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reshape ConvCodeWorld/convcodebench into the flat per-turn schema kvpress's multi-turn
harness expects.

Upstream schema (one HF row, five dict-typed columns, one per feedback configuration):

    {
      "CF_EF_UNIT_SNF": {
        "ITER=1": {"task_id": [..N tasks..], "previous_code": [...],
                   "compilation_feedback": [...], "execution_feedback": [...],
                   "verbal_feedback": [...], "label": [...]},
        "ITER=2": {...},
        ...
        "ITER=10": {...},
      },
      "CF_EF_FULL_SNF": {...},
      "CF_SEF": {...},
      "CF_EF_UNIT_SEF": {...},
      "CF_EF_FULL_SEF": {...},
    }

Flat schema this script emits (one row per (feedback_config, task, iteration) triple):

    task_id           : str   e.g. "BigCodeBench/42"
    feedback_config   : str   one of the five column names above
    iteration         : int   1..10
    context           : str   reconstructed conversation history up to (but not
                              including) this iteration
    answer_prefix     : str   e.g. "Here's the revised code:\n"
    question          : str   the feedback the user/environment gave at this iteration
    reference_code    : str   the ground-truth code at this iteration
                              (the upstream dataset's `previous_code`)
    label             : bool  pass/fail at this iteration
    max_new_tokens    : int   generation budget for the code output
    task              : str   "convcodeworld"
    session_id        : str   f"{feedback_config}/{task_id}" — unique per trajectory

Usage:
    # Local save under ./convcodeworld_flat/
    python create_huggingface_dataset.py

    # Push to HF Hub
    HF_REPO_ID=your-org/convcodeworld-kvpress python create_huggingface_dataset.py --push

NOTE: the reconstructed `context` is the concatenation of prior iterations' code +
feedback, which is what the multi-turn harness needs to feed through the press. It is
NOT the BigCodeBench problem statement — for that you need to fetch the matching task
from the BigCodeBench dataset separately (TODO: extend this script to prepend it).
"""

import os
from pathlib import Path

from datasets import Dataset, load_dataset
from fire import Fire


FEEDBACK_CONFIGS = [
    "CF_EF_UNIT_SNF",
    "CF_EF_FULL_SNF",
    "CF_SEF",
    "CF_EF_UNIT_SEF",
    "CF_EF_FULL_SEF",
]

ANSWER_PREFIX = "Here is the revised code:\n"
MAX_NEW_TOKENS = 1024


def _reconstruct_context(config_dict: dict, task_idx: int, up_to_iter: int) -> str:
    """
    Walk iterations 1..up_to_iter-1 and concatenate code + all feedback fields
    in the order they'd appear in a real refinement conversation.
    """
    turns = []
    for i in range(1, up_to_iter):
        it_key = f"ITER={i}"
        if it_key not in config_dict:
            continue
        it = config_dict[it_key]
        code = it["previous_code"][task_idx] if task_idx < len(it["previous_code"]) else ""
        compilation = it["compilation_feedback"][task_idx] if task_idx < len(it["compilation_feedback"]) else ""
        execution = it["execution_feedback"][task_idx] if task_idx < len(it["execution_feedback"]) else ""
        verbal = it["verbal_feedback"][task_idx] if task_idx < len(it["verbal_feedback"]) else ""
        parts = [f"# Iteration {i} — code\n{code}"]
        for name, content in [
            ("compilation_feedback", compilation),
            ("execution_feedback", execution),
            ("verbal_feedback", verbal),
        ]:
            if content and str(content).strip():
                parts.append(f"# {name}\n{content}")
        turns.append("\n".join(parts))
    return "\n\n---\n\n".join(turns)


def _feedback_text(it: dict, task_idx: int) -> str:
    """Concatenate this iteration's feedback fields into one 'question' string."""
    compilation = it["compilation_feedback"][task_idx] if task_idx < len(it["compilation_feedback"]) else ""
    execution = it["execution_feedback"][task_idx] if task_idx < len(it["execution_feedback"]) else ""
    verbal = it["verbal_feedback"][task_idx] if task_idx < len(it["verbal_feedback"]) else ""
    lines = []
    for name, content in [("compilation", compilation), ("execution", execution), ("verbal", verbal)]:
        if content and str(content).strip():
            lines.append(f"[{name}] {content}")
    return "\n".join(lines) or "(no feedback — code passes.)"


def main(output_dir: str = "./convcodeworld_flat", push: bool = False, repo_id: str | None = None):
    """
    Parameters
    ----------
    output_dir : str
        Local directory to write the flattened per-config dataset to.
    push : bool
        If True, also push each feedback-config slice to the HF Hub under `repo_id`.
    repo_id : str | None
        HF Hub repo id for the push. Defaults to env var HF_REPO_ID.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_id = repo_id or os.environ.get("HF_REPO_ID")

    src = load_dataset("ConvCodeWorld/convcodebench", split="train")
    row = src[0]

    for cfg_name in FEEDBACK_CONFIGS:
        if cfg_name not in row:
            print(f"[{cfg_name}] missing from dataset — skipping")
            continue
        cfg = row[cfg_name]
        first_iter = cfg.get("ITER=1", None)
        if first_iter is None:
            print(f"[{cfg_name}] missing ITER=1 — skipping")
            continue
        num_tasks = len(first_iter["task_id"])
        print(f"[{cfg_name}] flattening {num_tasks} tasks across 10 iterations...")

        rows = []
        for task_idx in range(num_tasks):
            task_id = first_iter["task_id"][task_idx]
            for iter_num in range(1, 11):
                it_key = f"ITER={iter_num}"
                if it_key not in cfg:
                    continue
                it = cfg[it_key]
                if task_idx >= len(it["previous_code"]):
                    continue
                rows.append(
                    {
                        "task_id": task_id,
                        "feedback_config": cfg_name,
                        "iteration": iter_num,
                        "context": _reconstruct_context(cfg, task_idx, iter_num),
                        "answer_prefix": ANSWER_PREFIX,
                        "question": _feedback_text(it, task_idx),
                        "reference_code": it["previous_code"][task_idx],
                        "label": bool(it["label"][task_idx]),
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "task": "convcodeworld",
                        "session_id": f"{cfg_name}/{task_id}",
                    }
                )

        ds = Dataset.from_list(rows)
        print(f"[{cfg_name}] {len(ds)} rows (10 iterations × {num_tasks} tasks, with holes dropped)")

        ds.save_to_disk(str(out / cfg_name))

        if push:
            if not repo_id:
                raise ValueError("push=True but no repo_id or HF_REPO_ID provided")
            ds.push_to_hub(repo_id, config_name=cfg_name, split="test")
            print(f"[{cfg_name}] pushed to {repo_id}")

    print(f"Done. Flattened per-config datasets under {out}.")


if __name__ == "__main__":
    Fire(main)
