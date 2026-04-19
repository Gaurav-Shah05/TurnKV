# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reshape microsoft/SCBench into the flat schema kvpress's evaluate.py expects.

Upstream schema (one row = one session):
    {id, context, multi_turns: [{input, answer, options?}, ...]}

kvpress-eval schema (one row = one question):
    {context, question, answer_prefix, answers, max_new_tokens, task, all_classes,
     session_id, turn_index, options}

This script loads each of the 12 SCBench subsets, flattens `multi_turns` into one row
per (session, turn) pair, attaches a task-appropriate `answer_prefix` and
`max_new_tokens`, and either saves to disk (default) or pushes to HF Hub.

Usage:
    # Local save to ./scbench_flat/
    python create_huggingface_dataset.py

    # Push to HF Hub (requires `huggingface-cli login`)
    HF_REPO_ID=your-org/SCBench-kvpress python create_huggingface_dataset.py --push

NOTE: The flattened representation treats turns as independent queries (multi-request
mode). It does NOT carry the running conversation state that true multi-turn evaluation
requires — that's the job of the (TODO) multi-turn harness.
"""

import os
from pathlib import Path

from datasets import Dataset, load_dataset
from fire import Fire


# See SCBench paper §3 for task categories. These prompts are permissive defaults;
# refine against microsoft/MInference/scbench/args.py before reporting final numbers.
ANSWER_PREFIX = {
    "scbench_kv": "The value is:",
    "scbench_prefix_suffix": "The matching string is:",
    "scbench_vt": "The final value is:",
    "scbench_repoqa": "The function is:",
    "scbench_qa_eng": "Answer:",
    "scbench_qa_chn": "回答：",
    "scbench_choice_eng": "The correct option is:",
    "scbench_many_shot": "Answer:",
    "scbench_mf": "The answer is:",
    "scbench_summary": "Summary:",
    "scbench_summary_with_needles": "Answer:",
    "scbench_repoqa_and_kv": "Answer:",
}

MAX_NEW_TOKENS = {
    "scbench_kv": 64,
    "scbench_prefix_suffix": 64,
    "scbench_vt": 32,
    "scbench_repoqa": 256,
    "scbench_qa_eng": 128,
    "scbench_qa_chn": 128,
    "scbench_choice_eng": 32,
    "scbench_many_shot": 64,
    "scbench_mf": 32,
    "scbench_summary": 512,
    "scbench_summary_with_needles": 512,
    "scbench_repoqa_and_kv": 256,
}

SUBSETS = list(ANSWER_PREFIX.keys())


def _flatten_subset(subset: str) -> Dataset:
    """Load one SCBench subset from HF and flatten (session, turn) → one row."""
    ds = load_dataset("microsoft/SCBench", subset, split="test")
    rows = []
    for session in ds:
        session_id = session["id"]
        context = session["context"]
        for turn_idx, turn in enumerate(session["multi_turns"]):
            rows.append(
                {
                    "context": context,
                    "question": turn["input"],
                    "answers": [turn["answer"]],
                    "answer_prefix": ANSWER_PREFIX[subset],
                    "max_new_tokens": MAX_NEW_TOKENS[subset],
                    "task": subset,
                    "all_classes": None,
                    "session_id": session_id,
                    "turn_index": turn_idx,
                    "options": turn.get("options"),
                }
            )
    return Dataset.from_list(rows)


def main(output_dir: str = "./scbench_flat", push: bool = False, repo_id: str | None = None):
    """
    Parameters
    ----------
    output_dir : str
        Local directory for the flattened per-subset datasets (one config per subset).
    push : bool
        Push to HF Hub instead of (or in addition to) saving locally.
    repo_id : str | None
        HF repo id for the push. Defaults to env var HF_REPO_ID.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    repo_id = repo_id or os.environ.get("HF_REPO_ID")

    for subset in SUBSETS:
        print(f"[{subset}] flattening...")
        ds = _flatten_subset(subset)
        print(f"[{subset}] {len(ds)} rows (from {len(set(ds['session_id']))} sessions)")

        ds.save_to_disk(str(out / subset))

        if push:
            if not repo_id:
                raise ValueError("push=True but no repo_id or HF_REPO_ID provided")
            ds.push_to_hub(repo_id, config_name=subset, split="test")
            print(f"[{subset}] pushed to {repo_id}")

    print(f"Done. Flattened subsets saved under {out}.")


if __name__ == "__main__":
    Fire(main)
