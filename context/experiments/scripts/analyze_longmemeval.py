"""
Regenerate LongMemEval_S per-turn and per-session token CSVs.

Prereq: run the extraction from longmemeval-data-cleaned.tar.gz per
`context/experiments/longmemeval_cleanup_status.md`, so that
`context/datasets/longmemeval/longmemeval_s_cleaned.json` exists locally.

Usage (from repo root):
    python context/experiments/scripts/analyze_longmemeval.py

Outputs (overwritten):
    context/experiments/longmemeval_s_per_turn_tokens.csv
    context/experiments/longmemeval_s_per_session_aggregated.csv

Tokenizer: unsloth/Llama-3.1-8B-Instruct (byte-identical to Meta Llama-3.1,
no HF auth required). Adjust TOKENIZER_ID below if switching models.
"""

import csv
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer


TOKENIZER_ID = "unsloth/Llama-3.1-8B-Instruct"
SRC = Path("context/datasets/longmemeval/longmemeval_s_cleaned.json")
PER_TURN = Path("context/experiments/longmemeval_s_per_turn_tokens.csv")
PER_SESSION = Path("context/experiments/longmemeval_s_per_session_aggregated.csv")


def to_str(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def main():
    t0 = time.time()
    print(f"Loading {SRC}...")
    with SRC.open(encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {len(data)} probes loaded in {time.time() - t0:.1f}s")

    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    cache: dict = {}

    def ntok(s):
        key = to_str(s)
        if key not in cache:
            cache[key] = len(tok(key, add_special_tokens=False)["input_ids"])
        return cache[key]

    PER_TURN.parent.mkdir(parents=True, exist_ok=True)

    with PER_TURN.open("w", encoding="utf-8", newline="") as tf, PER_SESSION.open(
        "w", encoding="utf-8", newline=""
    ) as af:
        tw = csv.writer(tf)
        aw = csv.writer(af)
        tw.writerow(
            [
                "question_id",
                "session_id",
                "turn_no",
                "role",
                "content_tokens",
                "has_answer",
                "is_evidence_session",
            ]
        )
        aw.writerow(
            [
                "question_id",
                "question_type",
                "num_sessions",
                "num_turns",
                "question_tokens",
                "answer_tokens",
                "total_haystack_tokens",
                "user_avg",
                "user_min",
                "user_max",
                "assistant_avg",
                "assistant_min",
                "assistant_max",
                "evidence_session_count",
                "evidence_turn_count",
            ]
        )

        t_start = time.time()
        for qi, entry in enumerate(data):
            qid = entry["question_id"]
            qtype = entry.get("question_type", "")
            q_tok = ntok(entry.get("question", ""))
            a_tok = ntok(entry.get("answer", ""))
            ans_ids = set(entry.get("answer_session_ids", []) or [])
            sess_ids = entry.get("haystack_session_ids", []) or []
            sessions = entry.get("haystack_sessions", []) or []

            user_lens: list[int] = []
            assistant_lens: list[int] = []
            total_hay = 0
            num_turns = 0
            ev_turns = 0
            for sid, turns in zip(sess_ids, sessions):
                is_ev = sid in ans_ids
                for ti, turn in enumerate(turns or []):
                    role = turn.get("role", "") if isinstance(turn, dict) else ""
                    content = turn.get("content", "") if isinstance(turn, dict) else turn
                    n = ntok(content)
                    ha = bool(turn.get("has_answer", False)) if isinstance(turn, dict) else False
                    tw.writerow([qid, sid, ti, role, n, int(ha), int(is_ev)])
                    total_hay += n
                    num_turns += 1
                    if ha:
                        ev_turns += 1
                    if role == "user":
                        user_lens.append(n)
                    elif role == "assistant":
                        assistant_lens.append(n)

            def agg(xs):
                if not xs:
                    return (0, 0, 0)
                return (round(statistics.mean(xs), 1), min(xs), max(xs))

            ua, umn, umx = agg(user_lens)
            aa, amn, amx = agg(assistant_lens)
            aw.writerow(
                [
                    qid,
                    qtype,
                    len(sessions),
                    num_turns,
                    q_tok,
                    a_tok,
                    total_hay,
                    ua,
                    umn,
                    umx,
                    aa,
                    amn,
                    amx,
                    len(ans_ids),
                    ev_turns,
                ]
            )
            if (qi + 1) % 100 == 0:
                print(f"  {qi + 1}/{len(data)} probes, elapsed {time.time() - t_start:.0f}s, cache {len(cache)}")

    print(f"Done in {time.time() - t_start:.0f}s")
    print(f"  {PER_TURN}: {PER_TURN.stat().st_size:,} bytes")
    print(f"  {PER_SESSION}: {PER_SESSION.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
