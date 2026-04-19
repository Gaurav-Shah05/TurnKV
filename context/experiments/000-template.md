# 000: Experiment title

- **Date**: YYYY-MM-DD
- **Owner**: Name
- **Branch / commit**: `<sha>`
- **Related decision(s)**: `decisions/NNN-<slug>.md`

## Question

What are we trying to find out? State it as a falsifiable claim if possible.

## Setup

- **Model**: e.g. meta-llama/Meta-Llama-3.1-8B-Instruct
- **Benchmark**: e.g. `scbench_repoqa_and_kv`
- **Press**: e.g. `turn_floor + snapkv`, compression_ratio=0.5
- **Baselines compared**: e.g. `snapkv`, `streaming_llm`, `no_press`
- **Hardware**: e.g. 1× H100 80GB
- **Results dir**: `./results/<config-dir>/`

## Results

Paste or link the table. If the finding is visual, include the plot.

## Takeaway

One or two sentences. What do we now believe that we didn't before? Does this change any decision? If yes, open a new ADR or supersede an existing one.
