# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build a deterministic 80/20 split of ConvCodeWorld CF_EF_UNIT_SNF task IDs.

Per ADR 001 and the smoke-test plan agreed on 2026-04-24:
- Universe: all task_ids in row[0]['CF_EF_UNIT_SNF']['ITER=1']['task_id'] (1140 tasks).
- Tuning split: random.Random(42).sample(sorted_ids, k=0.2*N) -> 228 tasks.
- Hold-out split: complement (912 tasks).
- Sorted (lexicographic) before sampling so the result is independent of dataset row order.

Outputs:
  splits/tune_20pct_seed42.json        # 228 task_ids, JSON list of strings.
  splits/holdout_80pct_seed42.json     # 912 task_ids, JSON list of strings.
  splits/split_manifest.json           # provenance: seed, sizes, sha256 of each list.

Run from the convcodeworld benchmark directory:
    python scripts/build_split.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET = "ConvCodeWorld/convcodebench"
DEFAULT_FEEDBACK = "CF_EF_UNIT_SNF"
DEFAULT_SEED = 42
DEFAULT_TUNE_FRAC = 0.20


def _sha256_of_list(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_split(
    feedback_config: str,
    seed: int,
    tune_fraction: float,
    out_dir: Path,
) -> None:
    ds = load_dataset(DATASET, split="train")
    row = ds[0]
    cfg = row.get(feedback_config)
    if not cfg or "ITER=1" not in cfg:
        raise SystemExit(f"feedback_config {feedback_config!r} not found in {DATASET}")

    universe = sorted(str(x) for x in cfg["ITER=1"]["task_id"])
    n_total = len(universe)
    n_tune = max(1, int(round(n_total * tune_fraction)))
    rng = random.Random(seed)
    tune_set = set(rng.sample(universe, k=n_tune))
    tune = sorted(tune_set)
    holdout = sorted(t for t in universe if t not in tune_set)

    assert len(tune) + len(holdout) == n_total
    assert set(tune).isdisjoint(holdout)

    out_dir.mkdir(parents=True, exist_ok=True)
    tune_path = out_dir / f"tune_{int(round(tune_fraction * 100))}pct_seed{seed}.json"
    hold_path = out_dir / f"holdout_{int(round((1 - tune_fraction) * 100))}pct_seed{seed}.json"
    manifest_path = out_dir / "split_manifest.json"

    tune_path.write_text(json.dumps(tune, indent=2) + "\n")
    hold_path.write_text(json.dumps(holdout, indent=2) + "\n")

    manifest = {
        "dataset": DATASET,
        "feedback_config": feedback_config,
        "seed": seed,
        "tune_fraction": tune_fraction,
        "n_total": n_total,
        "n_tune": len(tune),
        "n_holdout": len(holdout),
        "tune_path": tune_path.name,
        "holdout_path": hold_path.name,
        "tune_sha256": _sha256_of_list(tune),
        "holdout_sha256": _sha256_of_list(holdout),
        "stratification": "none (random sampling, sorted-lexicographic universe)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-config", default=DEFAULT_FEEDBACK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tune-fraction", type=float, default=DEFAULT_TUNE_FRAC)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "splits",
    )
    args = parser.parse_args()
    build_split(args.feedback_config, args.seed, args.tune_fraction, args.out_dir)


if __name__ == "__main__":
    main()
