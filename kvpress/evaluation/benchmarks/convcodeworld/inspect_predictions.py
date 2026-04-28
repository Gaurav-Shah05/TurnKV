# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Inspect a predictions.jsonl file from a ConvCodeWorld run to surface
environment / executor issues.

Usage (run from kvpress/):
    python evaluation/benchmarks/convcodeworld/inspect_predictions.py \\
        --input ./predictions_sample_shard0.jsonl

Or point at the volume path after downloading:
    modal volume get kvpress-convcodeworld-results \\
        sample_50pct_no_press_fullkv_shard0/predictions.jsonl \\
        ./predictions_sample_shard0.jsonl

The script prints:
  - Overall pass/fail counts and per-turn breakdown
  - Grouped error patterns (top N most-common error strings)
  - Full raw exec_output for the first few failures of each error class
  - A list of tasks to re-run after fixes
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def _error_text(row: dict) -> str:
    """Return the most informative error text from a predictions row."""
    # live_loop.py fields (preferred):
    #   execution_feedback — test runner output (most useful)
    #   compilation_feedback — compiler/import errors
    #   status — short status string
    for key in ("execution_feedback", "compilation_feedback", "status"):
        v = row.get(key)
        if v and str(v).strip():
            return str(v).strip()
    # Fallback for older schemas
    for key in ("exec_output", "failed_reason", "error", "stderr"):
        v = row.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return ""


def _extract_error_class(text: str) -> str:
    """Return the first recognisable Python exception type, or a short slug."""
    if not text:
        return "<empty>"
    # Catch standard Python exception names
    m = re.search(
        r"\b(ModuleNotFoundError|ImportError|AttributeError|TypeError|ValueError"
        r"|NameError|FileNotFoundError|PermissionError|OSError|RuntimeError"
        r"|TimeoutError|RecursionError|MemoryError|SyntaxError|IndentationError"
        r"|AssertionError|NotImplementedError|KeyError|IndexError|StopIteration"
        r"|UnicodeDecodeError|UnicodeEncodeError)\b",
        text,
    )
    if m:
        return m.group(1)
    if "Timeout" in text or "timed out" in text.lower():
        return "TimeoutError"
    if "segfault" in text.lower() or "segmentation fault" in text.lower():
        return "SegfaultError"
    if "CUDA" in text or "cuda" in text:
        return "CUDAError"
    if "OOM" in text or "out of memory" in text.lower():
        return "OOMError"
    # Trim to first 60 chars as a catch-all label
    return text.strip().splitlines()[0][:60]


def _load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: JSON parse error — {exc}")
    return rows


def inspect(input_path: Path, top_n: int = 15, show_raw: int = 2) -> None:
    print(f"\n{'='*70}")
    print(f"  ConvCodeWorld predictions inspector")
    print(f"  File : {input_path}")
    print(f"{'='*70}\n")

    rows = _load_predictions(input_path)
    if not rows:
        print("No rows found — file may be empty or not yet written.")
        return

    print(f"Total rows (task×turn) : {len(rows)}")

    # Group by task
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        tid = row.get("task_id") or row.get("id") or "unknown"
        by_task[tid].append(row)

    n_tasks = len(by_task)
    print(f"Unique tasks           : {n_tasks}\n")

    # Per-iteration pass rate (live_loop.py uses "iteration" not "turn_idx")
    iters: Counter[int] = Counter()
    iter_pass: Counter[int] = Counter()
    for row in rows:
        it = row.get("iteration", row.get("turn_idx", row.get("turn", -1)))
        passed = bool(row.get("passed") or row.get("pass"))
        if row.get("metric_excluded"):
            continue
        iters[it] += 1
        if passed:
            iter_pass[it] += 1

    print("Per-iteration pass rate (non-excluded rows):")
    for t in sorted(iters):
        n = iters[t]
        p = iter_pass[t]
        pct = 100 * p / n if n else 0
        print(f"  Iter {t:2d}: {p:4d}/{n:4d} passed  ({pct:5.1f}%)")

    # Final pass (last non-excluded iteration per task)
    final_pass = 0
    for tid, task_rows in by_task.items():
        active = [r for r in task_rows if not r.get("metric_excluded")]
        if active:
            last = max(
                active,
                key=lambda r: r.get("iteration", r.get("turn_idx", r.get("turn", 0))),
            )
            if last.get("passed") or last.get("pass"):
                final_pass += 1
    print(f"\nFinal pass (last active iteration): {final_pass}/{n_tasks}  ({100*final_pass/n_tasks:.1f}%)")

    # Status distribution
    status_counts: Counter[str] = Counter()
    for row in rows:
        if not row.get("metric_excluded"):
            status_counts[str(row.get("status", "<none>"))] += 1
    if status_counts:
        print("\nStatus distribution (non-excluded rows):")
        for s, c in status_counts.most_common():
            print(f"  {s:<40} {c:>6}")

    # Collect failures
    fail_rows = [
        r for r in rows
        if not (r.get("passed") or r.get("pass"))
        and not r.get("metric_excluded")
    ]
    print(f"\nFailed rows (non-excluded)   : {len(fail_rows)}")

    if not fail_rows:
        print("\nNo failures found — no environment issues to report.")
        return

    # Extract error text from various fields
    error_classes: Counter[str] = Counter()
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in fail_rows:
        text = _error_text(row)
        cls = _extract_error_class(text)
        error_classes[cls] += 1
        by_class[cls].append(row)

    print(f"\nTop-{top_n} error classes across failed rows:")
    print(f"  {'Error class':<35}  {'Count':>6}  {'% of fails':>10}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*10}")
    for cls, cnt in error_classes.most_common(top_n):
        pct = 100 * cnt / len(fail_rows)
        print(f"  {cls:<35}  {cnt:>6}  {pct:>9.1f}%")

    # ModuleNotFoundError detail — most actionable for env fixes
    if "ModuleNotFoundError" in by_class or "ImportError" in by_class:
        missing: Counter[str] = Counter()
        for cls in ("ModuleNotFoundError", "ImportError"):
            for row in by_class.get(cls, []):
                text = str(
                    row.get("exec_output")
                    or row.get("failed_reason")
                    or ""
                )
                for m in re.findall(
                    r"No module named ['\"]([^'\"]+)['\"]", text
                ):
                    missing[m.split(".")[0]] += 1
        if missing:
            print("\nMissing modules (add to MODAL_EVAL_REQUIREMENTS in modal_app.py):")
            for mod, cnt in missing.most_common():
                print(f"  pip install {mod:<35}  (seen in {cnt} rows)")

    # Show raw execution/compilation feedback for first few failures of each top class
    print(f"\nExecution feedback samples (first {show_raw} per class):")
    for cls, _ in error_classes.most_common(min(5, top_n)):
        print(f"\n  --- {cls} ---")
        for row in by_class[cls][:show_raw]:
            tid = row.get("task_id") or row.get("id") or "?"
            it = row.get("iteration", row.get("turn_idx", row.get("turn", "?")))
            text = _error_text(row)
            snippet = (text or "<no error text>").strip()[:800]
            print(f"  task={tid}  iteration={it}  status={row.get('status','?')}")
            print(f"  {snippet}")
            print()

    # Tasks that never passed — candidates for re-run after env fix
    never_passed = [
        tid for tid, task_rows in by_task.items()
        if not any(
            (r.get("passed") or r.get("pass"))
            for r in task_rows
            if not r.get("metric_excluded")
        )
    ]
    print(f"\nTasks that never passed: {len(never_passed)}/{n_tasks}")
    if never_passed:
        rerun_path = input_path.parent / "tasks_to_rerun.json"
        rerun_path.write_text(json.dumps(sorted(never_passed), indent=2) + "\n")
        print(f"  Written to: {rerun_path}")
        print("  To re-run just these tasks after env fixes:")
        print(f'    --task-ids "@{rerun_path}"')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Show top-N error classes (default: 15)",
    )
    parser.add_argument(
        "--show-raw",
        type=int,
        default=2,
        help="Number of raw exec_output samples per error class (default: 2)",
    )
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"File not found: {args.input}")
    inspect(args.input, top_n=args.top_n, show_raw=args.show_raw)


if __name__ == "__main__":
    main()
