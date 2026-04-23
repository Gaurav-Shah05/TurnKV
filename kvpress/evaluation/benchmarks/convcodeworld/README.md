# ConvCodeWorld / ConvCodeBench

[ConvCodeWorld](https://arxiv.org/abs/2502.19852) (Han et al., ICLR 2025) is a conversational code-generation benchmark built on top of BigCodeBench's 1,140 problems. It extends each problem into a **10-turn iterative refinement trajectory** under each of five feedback-type combinations, making it well-suited to study KV-cache eviction in multi-turn code workflows.

- Paper: https://arxiv.org/abs/2502.19852
- Dataset: https://huggingface.co/datasets/ConvCodeWorld/convcodebench
- Code / reference implementation: https://github.com/stovecat/convcodeworld

## Why it's a good fit for TurnKV

- **Real multi-turn structure**: every trajectory has 10 iterations. Each iteration's input depends on prior iterations' code + feedback.
- **Verifiable ground truth**: per-iteration `label` (pass/fail) inherited from BigCodeBench's unit tests.
- **Rich ablation axes**: five feedback-type combinations let us measure which kind of turn content our presses preserve well (compilation errors vs. execution traces vs. verbal critique).
- **Brutal compression regimes are meaningful**: median trajectory is ~4,500 tokens (measured on a 50-task sample with Llama-3.1 tokenizer). At 1/32 compression that's ~140 tokens kept — aggressive enough that press quality dominates the result.

## Dataset schema

The HF dataset has **one row** with **five dict-typed columns**, one per feedback-type combination:

| Column | Meaning |
|--------|---------|
| `CF_EF_UNIT_SNF` | Compilation + Execution-on-UNIT-tests + Simulated-Non-expert-Feedback |
| `CF_EF_FULL_SNF` | Compilation + Execution-on-FULL-tests + Simulated-Non-expert-Feedback |
| `CF_SEF` | Compilation + Simulated-Expert-Feedback (no execution) |
| `CF_EF_UNIT_SEF` | Compilation + Execution-on-UNIT + Simulated-Expert-Feedback |
| `CF_EF_FULL_SEF` | Compilation + Execution-on-FULL + Simulated-Expert-Feedback |

Each column is a dict:

```python
column = {
    "ITER=1": {
        "task_id":              [...1140 BigCodeBench task ids...],
        "previous_code":        [...code state at this turn, per task...],
        "compilation_feedback": [...],
        "execution_feedback":   [...],
        "verbal_feedback":      [...],
        "label":                [...pass/fail per task at this turn...],
    },
    "ITER=2": {...same structure, one iteration later...},
    ...
    "ITER=10": {...},
}
```

So one **trajectory** is what you get by walking a fixed task across `ITER=1..10` for one feedback config. Total: 1,140 tasks × 5 configs = **5,700 unique 10-turn trajectories**.

## Token-count distribution (measured)

Reconstructed from the 5 fields above, Llama-3.1 tokenizer, `add_special_tokens=False`, 50-task sample from `CF_EF_UNIT_SNF`:

| Stat | Tokens |
|------|-------:|
| Min  | 1,603 |
| Median | **4,513** |
| Mean | 6,013 |
| Max  | 21,820 |

All trajectories have exactly 10 turns.

## Evaluation protocol

### Primary metric — pass rate at final turn

Use the `label` field from `ITER=10` to score whether the final refined code passed BigCodeBench's unit tests. Report overall pass rate, broken out by:
- Compression ratio: {1, 1/2, 1/4, 1/8, 1/16, 1/32}
- Feedback configuration (5 columns)
- Press method (no_press, snapkv, streaming_llm, observed_attention, kvzip, expected_attention, our three turn-aware presses)

### Secondary metric — per-iteration pass rate

Because every task has labels at every iteration, we can plot pass rate over turns 1..10 for each press. This reveals whether a press degrades gracefully as the trajectory grows or collapses at some turn-count threshold.

### Tertiary metric — code similarity

For partial credit on failing cases, use `fuzzywuzzy.fuzz.ratio` (or CodeBLEU if a teammate implements it) between the final generated code and the ground-truth passing reference at the task's earliest-passing iteration.

## Usage

### Prepare the flattened dataset

```bash
# Loads ConvCodeWorld/convcodebench, emits one row per (feedback_config, task, iteration)
# with the reconstructed trajectory text in the 'context' field.
python kvpress/evaluation/benchmarks/convcodeworld/create_huggingface_dataset.py
```

### Run (once the multi-turn harness is implemented)

```bash
cd kvpress/evaluation
python evaluate.py \
    --dataset convcodeworld \
    --data_dir CF_EF_UNIT_SNF \
    --press_name snapkv \
    --compression_ratio 0.875 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

`--data_dir` selects one of the five feedback configurations. `--compression_ratio 0.875` corresponds to the 1/8 keep-rate (7/8 evicted).

### Static vs. live-loop runner

The ConvCodeWorld execution runner supports `benchmark_mode`:

- `static`: static replay. Prior reference code and feedback are teacher-forced from `ConvCodeWorld/convcodebench`; the generated code at each turn is still executed against BigCodeBench tests. All turns run because later context is fixed by the reference trajectory.
- `live`: live loop. Generated code is executed, new feedback is built from that generated code, and the result feeds the next turn.

The default is `live` for backward compatibility. For the local Fire CLI, use
`--benchmark_mode=static` or `--benchmark_mode=live`:

```bash
cd kvpress/evaluation
python benchmarks/convcodeworld/live_loop.py \
    --benchmark_mode=live \
    --press_name=snapkv \
    --compression_ratio=0.5 \
    --model=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --feedback_model=google/gemma-4-E2B-it \
    --feedback_config=CF_EF_UNIT_SNF \
    --num_eval_examples=10
```

For Modal:

```bash
cd kvpress
modal run evaluation/benchmarks/convcodeworld/modal_app.py::main \
    --benchmark-mode live \
    --press-names snapkv,streaming_llm,expected_attention \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --feedback-model google/gemma-4-E2B-it \
    --attn-implementation flash_attention_3 \
    --feedback-attn-implementation flash_attention_3 \
    --compression-ratio 0.5 \
    --snapkv-window-size 64 \
    --snapkv-kernel-size 5 \
    --streaming-llm-n-sink 4 \
    --expected-attention-n-future-positions 512 \
    --expected-attention-n-sink 4 \
    --expected-attention-epsilon 0.01 \
    --local-budget 4096 \
    --num-eval-examples 10
```

Use `--benchmark-mode static` on the same Modal command to run static replay
instead. In static mode, `--feedback-model` and
`--verbal-feedback-max-new-tokens` are ignored because verbal feedback comes
from the reference trajectory rather than a simulator model.

Static replay and live loop answer different questions: static replay isolates
the compression effect under a fixed conversation, while live loop measures the
fully interactive cascade where later turns depend on each generated solution.
Live-loop Modal runs default to exact H100 GPUs with `flash_attention_3` for DeepSeek-R1-Distill-Llama-8B code generation, `sdpa` for Gemma 4 E2B verbal feedback, `--cot=True`, early-stop once generated code passes the available tests, and expose the base-press knobs through the top-level Modal command.
The Modal image builds and caches a reduced BF16-only FlashAttention-3 wheel before copying repo source so ordinary code changes do not rebuild the H100 kernels.
It also pins Transformers to an upstream GitHub commit with Gemma4 support for the feedback model.
Turn-aware runs can use registry names such as `turnkv_snapkv` or pass
`--alpha-floor`, `--alpha-anchor`, and `--alpha-loyalty` with a base press.
See `MODAL_SETUP.md` for the full Modal setup runbook.

## TODOs before headline-quality numbers

1. **Confirm the dataset license**. The HF card does not list one as of 2026-04-19. Either ping the authors or fall back to running their public GitHub pipeline locally.
2. **Implement the multi-turn harness** (`kvpress/evaluation/multi_turn_evaluate.py`). Without it, the 10-turn conversational structure collapses to independent per-turn predictions, which defeats the point of measuring cross-turn KV preservation.
3. **Add CodeBLEU or fuzz.ratio scorer** in `calculate_metrics.py` for partial-credit analysis on failing cases.
4. **Filter short trajectories** at the most aggressive compression ratios (e.g., exclude trajectories <500 tokens when evaluating at 1/32) — 50 tokens kept isn't enough for coherent generation.
