# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scoring for ConvCodeWorld / ConvCodeBench (https://arxiv.org/abs/2502.19852).

Primary signal comes directly from the dataset's `label` field — a per-turn pass/fail
boolean inherited from BigCodeBench's unit tests. No LLM judge, no execution needed at
evaluation time: the reference trajectory was already judged when the dataset was built.

Expected columns on the input DataFrame (produced by create_huggingface_dataset.py +
the multi-turn harness — see README for the pipeline):

    task_id           : str    e.g. "BigCodeBench/42"
    feedback_config   : str    one of CF_EF_UNIT_SNF / CF_EF_FULL_SNF / CF_SEF /
                               CF_EF_UNIT_SEF / CF_EF_FULL_SEF
    iteration         : int    1..10 — turn number within the trajectory
    predicted_answer  : str    model's code output after compressing + attending to
                               the trajectory up to this iteration
    reference_code    : str    the dataset's ground-truth code at this iteration
    label             : bool   does predicted_answer pass the unit tests (True/False)?
                               For the MVP we trust the dataset's label directly; a
                               later pass can re-execute the generated code for a
                               true per-prediction label.
"""

from collections import defaultdict

import numpy as np

try:
    from fuzzywuzzy import fuzz  # optional; also used by the longbench scorer
except Exception:
    fuzz = None


def _code_similarity(prediction: str, reference: str) -> float:
    """Fuzz-ratio similarity in [0, 1]. Returns 0 if fuzzywuzzy is unavailable."""
    if fuzz is None or not prediction or not reference:
        return 0.0
    return fuzz.ratio(prediction, reference) / 100.0


def calculate_metrics(df) -> dict:
    """
    Aggregate pass rate on ConvCodeWorld trajectories.

    Reports:
      - overall pass rate (pooled over all tasks / iterations / configs)
      - pass rate per feedback configuration
      - pass rate per iteration (1..10) — lets us plot decay over turn count
      - mean code similarity as a secondary signal for partial credit on fails
    """
    if len(df) == 0:
        return {"overall": 0.0}

    labels = [bool(x) for x in df["label"].tolist()]
    results: dict = {"overall": round(100 * float(np.mean(labels)), 2)}

    # Per feedback configuration
    by_config: dict = defaultdict(list)
    for cfg, lbl in zip(df["feedback_config"].tolist(), labels):
        by_config[cfg].append(lbl)
    results["per_feedback_config"] = {
        cfg: round(100 * float(np.mean(v)), 2) for cfg, v in sorted(by_config.items())
    }

    # Per iteration
    if "iteration" in df.columns:
        by_iter: dict = defaultdict(list)
        for it, lbl in zip(df["iteration"].tolist(), labels):
            by_iter[int(it)].append(lbl)
        results["per_iteration"] = {
            f"iter_{it}": round(100 * float(np.mean(v)), 2) for it, v in sorted(by_iter.items())
        }

    # Code similarity (optional secondary signal)
    if "predicted_answer" in df.columns and "reference_code" in df.columns:
        sims = [
            _code_similarity(str(p or ""), str(r or ""))
            for p, r in zip(df["predicted_answer"].tolist(), df["reference_code"].tolist())
        ]
        if sims:
            results["mean_code_similarity"] = round(100 * float(np.mean(sims)), 2)

    return results
