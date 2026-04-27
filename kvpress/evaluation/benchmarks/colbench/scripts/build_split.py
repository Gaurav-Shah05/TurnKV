# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build a deterministic 80/20 split of ColBench Backend task IDs.

Mirrors ``convcodeworld/scripts/build_split.py``: sort the task-id universe
lexicographically, sample ``tune_fraction`` of it with ``random.Random(seed)``,
and write tune + holdout JSON lists alongside a ``split_manifest.json`` with
sha256 provenance.

Outputs (defaults):
  splits/tune_20pct_seed42.json
  splits/holdout_80pct_seed42.json
  splits/split_manifest.json

Run from the colbench benchmark directory:
    python scripts/build_split.py

Or explicitly choose dataset / subset / split:
    python scripts/build_split.py \\
        --dataset facebook/collaborative_agent_bench \\
        --subset backend \\
        --hf-split train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET = "facebook/collaborative_agent_bench"
DEFAULT_SUBSET = "backend"
DEFAULT_HF_SPLIT = "train"
DEFAULT_SEED = 42
DEFAULT_TUNE_FRAC = 0.20


def _sha256_of_list(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _resolve_task_id(row: dict, idx: int) -> str:
    for key in ("task_id", "id", "uid", "name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"colbench/backend/{idx}"


def build_split(
    dataset: str,
    subset: str,
    hf_split: str,
    seed: int,
    tune_fraction: float,
    out_dir: Path,
) -> None:
    if subset:
        try:
            ds = load_dataset(dataset, subset, split=hf_split)
        except (ValueError, FileNotFoundError):
            ds = load_dataset(dataset, split=hf_split)
    else:
        ds = load_dataset(dataset, split=hf_split)

    universe = sorted({_resolve_task_id(dict(row), idx) for idx, row in enumerate(ds)})
    n_total = len(universe)
    if n_total == 0:
        raise SystemExit(f"empty universe loaded from {dataset!r}/{subset!r}/{hf_split!r}")
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
        "dataset": dataset,
        "subset": subset,
        "hf_split": hf_split,
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
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--hf-split", default=DEFAULT_HF_SPLIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tune-fraction", type=float, default=DEFAULT_TUNE_FRAC)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "splits",
    )
    args = parser.parse_args()
    build_split(args.dataset, args.subset, args.hf_split, args.seed, args.tune_fraction, args.out_dir)


if __name__ == "__main__":
    main()
