# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1] / "evaluation"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.scbench.calculate_metrics import calculate_metrics, get_score_one  # noqa: E402


def test_get_score_kv_retrieval():
    assert get_score_one('x "needle"', "needle", "kv_retrieval") == 1.0
    assert get_score_one("no", "needle", "kv_retrieval") == 0.0


def test_calculate_metrics_smoke():
    df = pd.DataFrame(
        [
            {"prediction": "The value is 42", "ground_truth": "42", "task": "scbench_kv"},
        ]
    )
    out = calculate_metrics(df, "scbench_kv")
    assert out["num_rows"] == 1
    assert "mean_score" in out
