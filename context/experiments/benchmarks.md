# Benchmarks

## Mental model

For every benchmark, one problem consists of:

- A **long context** at the start (may be empty).
- **N turns**, each with a **Q** (user / environment asks something) and an **A** (model answers).

Our KV-cache compression work treats this uniformly — the press compresses the cache of everything seen so far at turn boundaries.

| Benchmark | N turns | Long context at start? | What Q is | What A is | Ground-truth signal |
|-----------|--------:|:----------------------:|-----------|-----------|---------------------|
| **SCBench** | 2–5 | **yes** — a document | a query about the doc | expected answer | exact-match / F1 depending on task |
| **ConvCodeWorld** | exactly 10 | short task prompt (~500 tok) | feedback (compile + execute + verbal, combined) | revised code | per-turn pass/fail unit tests |
| **TopiOCQA** | 6–25 | none; each turn has its own retrieved Wikipedia passage | user question | user answer | exact-match / F1 |
| **LongMemEval_S** | **1** final question (with ~490 prior turns *inside* the context) | **yes** — a chat history of ~47 prior sessions (~103K tokens) | the final probe | expected answer | exact-match / LLM-judge |

**One line per dataset in plain English:**

- **SCBench** — "document + N questions over the document." Turns are independent.
- **ConvCodeWorld** — "task prompt + 10 rounds of (feedback → revised code)." Turns build on each other; after the code passes, subsequent iterations are no-ops.
- **TopiOCQA** — "no pre-provided doc; 12-ish turns of Q + A + Wikipedia passage, with topic shifts mid-conversation."
- **LongMemEval_S** — "here's a long chat history, now answer ONE question about something the user mentioned in it." The multi-turn structure is inside the chat history, not at the probing level.

## Where to look at actual data

Two CSVs per benchmark. Content for scanning rows, stats for aggregate analysis.

**Per-turn content CSVs** (one row per turn; content truncated to 400–600 chars per cell):

| File | Rows | One row = | Columns |
|------|-----:|-----------|---------|
| `scbench.csv` | 5,143 | one turn | `subset, session_id, turn_index, question, answer, options, context_start, context_end, context_total_chars` |
| `convcodeworld.csv` | 57,000 | one iteration | `feedback_config, task_id, iteration, label, previous_code, compilation_feedback, execution_feedback, verbal_feedback` |
| `topiocqa.csv` | 47,964 | one turn | `split, Conversation_no, Turn_no, Topic, Topic_section, Question, Answer, Gold_passage_title, Gold_passage_text` |

**Per-session aggregated stats CSVs** (one row per problem; Llama-3.1 tokenizer counts):

| File | Rows | One row = | Columns |
|------|-----:|-----------|---------|
| `scbench_stats.csv` | 922 | one session | `subset, session_id, context_tokens, turn_count, q_avg, q_min, q_max, a_avg, a_min, a_max, total_session_tokens` |
| `convcodeworld_stats.csv` | 5,700 | one (feedback_config, task) pair | `feedback_config, task_id, turn_count, q_avg, q_min, q_max, a_avg, a_min, a_max, total_turn_tokens, final_label` |
| `topiocqa_stats.csv` | 3,714 | one conversation | `split, Conversation_no, turn_count, unique_topics, q_avg, q_min, q_max, a_avg, a_min, a_max, passage_avg, passage_min, passage_max, total_session_tokens` |

**Combined (both content + stats in one file)**:

| File | Rows | One row = | Columns |
|------|-----:|-----------|---------|
| `longmemeval_s.csv` | 500 | one probe | `question_id, question_type, question, answer, evidence_session_ids, evidence_turn_excerpt, num_sessions, num_turns, context_tokens, user_avg_tokens, assistant_avg_tokens, evidence_turn_count` |

For full untruncated content, load the native HF dataset (or, for LongMemEval, the extracted JSON under `context/datasets/longmemeval/` per `longmemeval_cleanup_status.md`).

## Per-dataset notes

### SCBench (appendix benchmark)
12 task subsets (`scbench_kv`, `scbench_qa_eng`, etc.). **Turns within a session are independent queries**, not a conversation — which is why we demoted it to the appendix and pivoted to the three below. Kept for cross-query KV retention numbers.

### ConvCodeWorld
5 feedback configurations (`CF_EF_UNIT_SNF`, `CF_EF_FULL_SNF`, `CF_SEF`, `CF_EF_UNIT_SEF`, `CF_EF_FULL_SEF`) × 1,140 BigCodeBench problems × 10 iterations = 57,000 flattened rows. About 40% of rows have empty `compilation/execution/verbal_feedback` — those are "no-op" turns that happen after a solution already passed (same code re-fed). Harness skips them at evaluation time. **CoT enabled for this benchmark** — code generation benefits from the model reasoning through the feedback before emitting code.

### TopiOCQA
Includes both train (3,509 conversations) and validation (205 conversations) splits. Mean 3.85 unique topics per conversation — that's the topic-shift signal we validate Cross-Turn Accumulation / Loyalty against. Short trajectories (median 1.7K tokens total) limit practical compression to 1/2 and 1/4; skip 1/16 and 1/32 for this benchmark.

### LongMemEval_S
500 probes. Each probe's history is ~47 sessions (~490 user/assistant turns) padded to ~103K tokens. "Filler sessions" are the 40-ish sessions unrelated to the probe question; the 1 or 2 "evidence sessions" contain the actual answer buried in real conversation content. The row's `evidence_turn_excerpt` column shows you the first user/assistant turn that's tagged with the evidence.

## Gotchas summary (for the harness)

- **SCBench**: `scbench_vt` stores the long context under `input` not `context`; `scbench_mf` context is a `list[int]` not a string; `scbench_mf` per-turn answer is also a list. Harness must coerce.
- **ConvCodeWorld**: 40% no-op trailing turns (post-pass). Skip or early-stop.
- **TopiOCQA**: no static context — each turn brings its own retrieved passage; the cache accumulates passages + Q + A across turns.
- **LongMemEval_S**: context is a chat history, not a document. Uses real user/assistant turn structure, which is what gives our turn-aware policies something to bite on.

## Regenerating

These CSVs are produced by scripts under `scripts/`. Full generation uses the extracted LongMemEval data (`context/datasets/longmemeval/`, gitignored — see `longmemeval_cleanup_status.md`) plus auto-downloaded HF datasets. Tokenizer: `unsloth/Llama-3.1-8B-Instruct` (ungated mirror of Meta Llama-3.1, no auth needed).
