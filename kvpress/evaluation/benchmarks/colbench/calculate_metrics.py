# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoring for ColBench (Backend split).

ColBench (Meta FAIR, SWEET-RL, March 2025) is a multi-turn coding benchmark
where each task is a single problem the agent solves over up to ``max_turns``
turns of conversation with a simulated human (a feedback LLM with reference-
solution access). The agent emits either a clarifying *question* (no code) or
a *submission* (fenced ``\`\`\`python`` block). Final pass/fail is decided when
the agent submits and we run the hidden tests.

Expected columns on the input DataFrame (produced by ``live_loop.py``):

    task_id           : str
    iteration         : int    1..max_turns - turn index within this task
    is_question       : bool   True if the agent's reply was a question
    is_submission     : bool   True if the agent's reply was a code submission
    predicted_answer  : str    last submitted code (carried forward across rows)
    passed            : bool   final pass/fail at submission (NaN before submit)
    status            : str    pass / runtime_error / compile_error / timeout /
                               submitted_no_pass / pending / skipped_after_pass
    session_id        : str    one per task

Headline metrics:
    overall                       pooled pass rate over sessions (one row/session)
    per_iteration                 pass rate observed at each iteration index
    mrr                           mean reciprocal rank of first passing iteration
    recall                        sessions where any submission passed
    final_pass_rate               sessions whose final submitted code passed
    mean_questions_before_submit  ColBench efficiency signal: avg # clarifying
                                  questions across passing sessions
    status_counts                 frequency of every status value
"""

from collections import defaultdict

import numpy as np


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"pass", "passed", "true", "1", "yes"}


def calculate_metrics(df) -> dict:
    """Aggregate pass rate + efficiency on ColBench Backend trajectories.

    The DataFrame may carry one row per turn (live-loop trajectory) or one
    summary row per task. Both cases are handled by computing per-session
    aggregates from rows that are not ``metric_excluded`` / ``skipped_after_pass``.
    """
    if len(df) == 0:
        return {"overall": 0.0}

    raw_df = df
    if "metric_excluded" in df.columns:
        df = df[~df["metric_excluded"].fillna(False)]
    elif "status" in df.columns:
        df = df[df["status"] != "skipped_after_pass"]
    if len(df) == 0:
        results = {"overall": 0.0}
        if "status" in raw_df.columns:
            results["status_counts"] = {
                str(k): int(v) for k, v in raw_df["status"].value_counts(dropna=False).sort_index().items()
            }
        return results

    label_col = "passed" if "passed" in df.columns else "label"
    labels = [_coerce_bool(x) for x in df[label_col].tolist()]
    results: dict = {"overall": round(100 * float(np.mean(labels)), 2)}

    if "iteration" in df.columns:
        by_iter: dict = defaultdict(list)
        for it, lbl in zip(df["iteration"].tolist(), labels):
            by_iter[int(it)].append(lbl)
        results["per_iteration"] = {
            f"iter_{it}": round(100 * float(np.mean(v)), 2) for it, v in sorted(by_iter.items())
        }

    if "session_id" in df.columns and "iteration" in df.columns:
        mrr_values: list[float] = []
        recall_values: list[bool] = []
        final_values: list[bool] = []
        questions_before_submit: list[int] = []
        for _, group in df.sort_values("iteration").groupby("session_id"):
            group_labels = [_coerce_bool(x) for x in group[label_col].tolist()]
            recall_values.append(any(group_labels))
            first_pass_rank = next((i + 1 for i, passed in enumerate(group_labels) if passed), None)
            mrr_values.append(0.0 if first_pass_rank is None else 1.0 / first_pass_rank)
            final_values.append(group_labels[-1])
            if "is_question" in group.columns:
                # Count clarifying questions issued before the first submission
                # within this session. If the agent never submitted, fall back
                # to the total question count.
                questions = 0
                for _, row in group.iterrows():
                    if _coerce_bool(row.get("is_submission", False)):
                        break
                    if _coerce_bool(row.get("is_question", False)):
                        questions += 1
                questions_before_submit.append(questions)
        if mrr_values:
            results["mrr"] = round(100 * float(np.mean(mrr_values)), 2)
            results["recall"] = round(100 * float(np.mean(recall_values)), 2)
            results["final_pass_rate"] = round(100 * float(np.mean(final_values)), 2)
        if questions_before_submit:
            results["mean_questions_before_submit"] = round(float(np.mean(questions_before_submit)), 2)

    if "status" in raw_df.columns:
        results["status_counts"] = {
            str(k): int(v) for k, v in raw_df["status"].value_counts(dropna=False).sort_index().items()
        }

    return results
