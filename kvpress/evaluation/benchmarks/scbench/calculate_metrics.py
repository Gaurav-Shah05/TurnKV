# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scoring for SCBench (https://arxiv.org/abs/2412.10319).

NOTE: These metric functions are a first-pass approximation of the scoring used in
microsoft/MInference/scbench/compute_scores.py. Before reporting numbers comparable
to the SCBench paper, port the upstream implementations verbatim — especially for
`scbench_repoqa`, `scbench_summary_with_needles`, and `scbench_repoqa_and_kv` which
use composite scores.
"""

import re
import string
from collections import Counter

import numpy as np
from rouge import Rouge


# ---------- Normalization helpers ----------

def _normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


# ---------- Metric primitives ----------

def _exact_match(prediction: str, ground_truth: str, **_) -> float:
    return float(_normalize_answer(prediction) == _normalize_answer(ground_truth))


def _substring_match(prediction: str, ground_truth: str, **_) -> float:
    """1.0 iff the (normalized) ground truth appears anywhere in the (normalized) prediction."""
    return float(_normalize_answer(ground_truth) in _normalize_answer(prediction))


def _f1(prediction: str, ground_truth: str, **_) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, ground_truth: str, **_) -> float:
    if not prediction.strip() or not ground_truth.strip():
        return 0.0
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]
    except Exception:
        return 0.0


def _choice_match(prediction: str, ground_truth: str, options=None, **_) -> float:
    """For multiple-choice: pick the option with the highest substring overlap with prediction."""
    if not options:
        return _exact_match(prediction, ground_truth)
    pred_norm = _normalize_answer(prediction)
    gold_norm = _normalize_answer(ground_truth)
    best_opt, best_score = None, -1
    for opt in options:
        score = sum(1 for tok in _normalize_answer(opt).split() if tok in pred_norm)
        if score > best_score:
            best_score, best_opt = score, opt
    return float(_normalize_answer(best_opt or "") == gold_norm)


# ---------- Task → metric mapping ----------
# Categories per SCBench §3.1; approximate scorers, see file docstring.

TASK_TO_METRIC = {
    # String retrieval (exact / substring)
    "scbench_kv": _substring_match,
    "scbench_prefix_suffix": _substring_match,
    "scbench_vt": _exact_match,
    # Semantic retrieval (F1 / ROUGE)
    "scbench_repoqa": _rouge_l,  # TODO: upstream uses function-level match; port verbatim
    "scbench_qa_eng": _f1,
    "scbench_qa_chn": _f1,  # TODO: use jieba tokenization for Chinese (see longbench/qa_f1_zh_score)
    "scbench_choice_eng": _choice_match,
    # Global processing
    "scbench_many_shot": _exact_match,
    "scbench_mf": _exact_match,
    "scbench_summary": _rouge_l,
    # Multi-tasking (composite — TODO: port upstream composite scorers)
    "scbench_summary_with_needles": _rouge_l,
    "scbench_repoqa_and_kv": _rouge_l,
}


# ---------- Public API ----------

def _score_row(task: str, prediction: str, answers, options=None) -> float:
    metric = TASK_TO_METRIC.get(task)
    if metric is None:
        raise ValueError(f"Unknown SCBench task: {task}")
    # `answers` is a list of acceptable references (at minimum a single-item list).
    if isinstance(answers, str):
        answers = [answers]
    return max(metric(prediction, gt, options=options) for gt in answers)


def calculate_metrics(df) -> dict:
    """
    Compute per-task (and optionally per-turn) accuracy on a SCBench results frame.

    Expected columns:
        - `predicted_answer`: model output (string)
        - `answers`: list[str] of acceptable references
        - `task`: one of the 12 `scbench_*` task names
        - `options` (optional): list[str] for multiple-choice tasks
        - `turn_index` (optional): int — if present, emit per-turn breakdowns
    """
    results: dict = {}

    # Per-task aggregate
    for task, group in df.groupby("task"):
        scores = [
            _score_row(
                task,
                row["predicted_answer"] or "",
                row["answers"],
                options=row.get("options") if "options" in row else None,
            )
            for _, row in group.iterrows()
        ]
        results[task] = round(100 * float(np.mean(scores)), 2)

        # Per-turn breakdown (only if turn_index is provided, SCBench-style reporting)
        if "turn_index" in group.columns:
            per_turn = {}
            for turn_idx, turn_group in group.groupby("turn_index"):
                turn_scores = [
                    _score_row(
                        task,
                        row["predicted_answer"] or "",
                        row["answers"],
                        options=row.get("options") if "options" in row else None,
                    )
                    for _, row in turn_group.iterrows()
                ]
                per_turn[f"turn_{int(turn_idx)}"] = round(100 * float(np.mean(turn_scores)), 2)
            results[f"{task}__per_turn"] = per_turn

    # Overall (averaged over all rows, not over tasks — matches SCBench overall avg)
    if len(df) > 0:
        overall_scores = [
            _score_row(
                row["task"],
                row["predicted_answer"] or "",
                row["answers"],
                options=row.get("options") if "options" in row else None,
            )
            for _, row in df.iterrows()
        ]
        results["overall"] = round(100 * float(np.mean(overall_scores)), 2)

    return results
