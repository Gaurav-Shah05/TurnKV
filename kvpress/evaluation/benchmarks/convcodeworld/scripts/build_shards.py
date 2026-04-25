# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shard a JSON list of task IDs into N evenly-sized shard files.

Used by the smoke run scripts to dispatch one Modal container per shard so the
228-task tune split can be evaluated across multiple GPUs in parallel.

Sharding is deterministic: tasks are interleaved by index modulo num_shards
(rather than contiguous chunks) so each shard sees a representative spread of
BigCodeBench task IDs and per-task latency is roughly balanced.

Outputs (default num_shards=10):
  splits/shards/<input_stem>_shard_0_of_10.json
  ...
  splits/shards/<input_stem>_shard_9_of_10.json
  splits/shards/<input_stem>_shards_manifest.json   # provenance.

Run from the convcodeworld benchmark directory:
    python scripts/build_shards.py --input splits/tune_20pct_seed42.json --num-shards 10
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256_of_list(items: list[str]) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def build_shards(input_path: Path, num_shards: int, out_dir: Path) -> None:
    if num_shards < 1:
        raise SystemExit(f"--num-shards must be >= 1, got {num_shards}")
    with input_path.open("r", encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list) or not all(isinstance(x, (str, int)) for x in ids):
        raise SystemExit(f"input {input_path} must be a JSON list of task IDs")
    ids = [str(x) for x in ids]
    if not ids:
        raise SystemExit(f"input {input_path} is empty; nothing to shard")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    shard_lists: list[list[str]] = [[] for _ in range(num_shards)]
    for idx, tid in enumerate(ids):
        shard_lists[idx % num_shards].append(tid)

    shard_records = []
    for shard_idx, shard_ids in enumerate(shard_lists):
        shard_path = out_dir / f"{stem}_shard_{shard_idx}_of_{num_shards}.json"
        shard_path.write_text(json.dumps(shard_ids, indent=2) + "\n")
        shard_records.append(
            {
                "shard": shard_idx,
                "size": len(shard_ids),
                "path": shard_path.name,
                "sha256": _sha256_of_list(shard_ids),
            }
        )

    manifest = {
        "input": str(input_path),
        "num_shards": num_shards,
        "n_total": len(ids),
        "stride_strategy": "interleaved (idx % num_shards)",
        "shards": shard_records,
    }
    (out_dir / f"{stem}_shards_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <input parent>/shards/",
    )
    args = parser.parse_args()
    out_dir = args.out_dir or (args.input.resolve().parent / "shards")
    build_shards(args.input.resolve(), args.num_shards, out_dir)


if __name__ == "__main__":
    main()
