# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from benchmarks.convcodeworld.calculate_metrics import calculate_metrics as convcodeworld_scorer
from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.scbench.calculate_metrics import calculate_metrics as scbench_scorer
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    AdaKVPress,
    BlockPress,
    CAMPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVComposePress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LoyaltyPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    RoleBoundaryAnchorPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    TurnAwareGlobalPress,
    TurnFloorPress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    # SCBench: raw schema is {id, context, multi_turns:[...]}. Run
    # `evaluation/benchmarks/scbench/create_huggingface_dataset.py` to produce the
    # flattened per-turn schema expected by evaluate.py; until published, point this
    # entry at a local HF dataset dir (`load_dataset("./scbench_flat/<subset>")`) or
    # your own HF Hub repo id. DEMOTED to appendix benchmark (see
    # documentation/findings.md) — turns are independent queries over shared context,
    # not truly conversational.
    "scbench": "microsoft/SCBench",
    # ConvCodeWorld / ConvCodeBench: 1,140 BigCodeBench tasks x 5 feedback configs,
    # each with 10-turn refinement trajectories. Run
    # `evaluation/benchmarks/convcodeworld/create_huggingface_dataset.py` to produce
    # the flattened per-(config,task,iteration) schema; evaluation uses per-turn
    # pass/fail labels from the dataset itself.
    "convcodeworld": "ConvCodeWorld/convcodebench",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "math500": "alessiodevoto/math500",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "scbench": scbench_scorer,
    "convcodeworld": convcodeworld_scorer,
    "aime25": aime25_scorer,
    "math500": math500_scorer,
}


PRESS_REGISTRY = {
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "cur": CURPress(),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "fastkvzip": FastKVzipPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvcompose": KVComposePress(),
    "kvcompose_unstructured": KVComposePress(structured=False),
    "kvzip": KVzipPress(),
    "kvzip_plus": KVzipPress(kvzip_plus_normalization=True),
    "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
    "kvzap_mlp_head": KVzapPress(model_type="mlp"),
    "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
    "lagkv": LagKVPress(),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "no_press": None,
    "cam_streaming_llm": CAMPress(base_press=StreamingLLMPress()),
    "cam_knorm": CAMPress(base_press=KnormPress()),
    "cam_adakv_snapkv": CAMPress(base_press=AdaKVPress(SnapKVPress())),
    "cam_tova": CAMPress(base_press=TOVAPress()),
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
}


def _turnkv_composite(base_press, budget: int, alpha: float) -> TurnAwareGlobalPress:
    """Factory for turnkv_*/baseline_* registry entries (ADR 002 §2 row for
    ``evaluate_registry.py``). ``alpha=1.0`` -> turn-aware variant; ``alpha=0.0``
    -> bit-equivalent-to-base short-circuit. ``budget`` is a placeholder for
    the CLI entry point; the multi-turn harness (Week 2) will override it
    per benchmark (ADR 001 §2 ``global_budget`` table).
    """
    return TurnAwareGlobalPress(
        base_press=base_press,
        global_budget=budget,
        policies={
            "floor": TurnFloorPress(global_budget=budget),
            "anchor": RoleBoundaryAnchorPress(),
            "loyalty": LoyaltyPress(),
        },
        alphas={"floor": alpha, "anchor": alpha, "loyalty": alpha},
    )


# TurnKV Week-1 registry entries (ADR 002 §2). Each entry constructs a
# fresh base press so ``compression_ratio`` mutation on one variant cannot
# leak into another (standard registry-wide convention). ADR 002 §2 lists
# ``turnkv_adakv_snapkv`` in the Week-1 slate; the composer's
# ``ScorerPress``-only ``__post_init__`` guard (see
# ``turn_aware_global_press.py``) defers AdaKV to Week 2, so only the three
# plain-ScorerPress bases ship here.
_TURNKV_BUDGET_PLACEHOLDER = 4096
_TURNKV_BASE_FACTORIES = {
    "snapkv": SnapKVPress,
    "streaming_llm": StreamingLLMPress,
    "expected_attention": ExpectedAttentionPress,
    # TODO(Week 2): "adakv_snapkv": lambda: AdaKVPress(SnapKVPress()) once the
    # composer supports per-head masking.
}
for _name, _factory in _TURNKV_BASE_FACTORIES.items():
    PRESS_REGISTRY[f"turnkv_{_name}"] = _turnkv_composite(
        _factory(), _TURNKV_BUDGET_PLACEHOLDER, alpha=1.0
    )
    PRESS_REGISTRY[f"baseline_{_name}"] = _turnkv_composite(
        _factory(), _TURNKV_BUDGET_PLACEHOLDER, alpha=0.0
    )
del _name, _factory  # avoid leaking loop vars into the module namespace
