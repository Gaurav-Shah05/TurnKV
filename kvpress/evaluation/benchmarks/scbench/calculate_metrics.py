# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Scoring helpers adapted from Microsoft MInference scbench/compute_scores.py (MIT License).

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

import pandas as pd
from tqdm import tqdm

Multiturnbench_to_Infinitebench = {
    "scbench_choice_eng": "longbook_choice_eng",
    "scbench_qa_eng": "longdialogue_qa_eng",
    "scbench_qa_chn": "longbook_qa_chn",
    "scbench_kv": "kv_retrieval",
    "scbench_kv_hard": "kv_retrieval",
    "scbench_hashhop": "kv_retrieval",
    "scbench_prefix_suffix": "kv_retrieval",
    "scbench_mf": "math_find",
    "scbench_passkey": "passkey",
}


def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s: str) -> str:
    def white_space_fix(text: str) -> str:
        return "".join(text.split())

    def remove_punc(text: str) -> str:
        cn_punctuation = (
            "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        )
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def string_match_all(pred: str, ref: list | str, model_name: str = "") -> float:
    if not isinstance(ref, list):
        ref = [ref]
    score = sum([1.0 if r.lower() in pred.lower() else 0.0 for r in ref]) / len(ref)
    return round(score, 2)


def f1_score(prediction: list, ground_truth: list) -> tuple[float, float, float]:
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def qa_f1_score(pred: str, ground_truths: list) -> float:
    f1 = 0.0
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(pred)
        normalized_ground_truth = normalize_answer(ground_truth)
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        scores = f1_score(prediction_tokens, ground_truth_tokens)
        this_f1, _, _ = scores
        f1 = max(f1, this_f1)
    return f1


def qa_f1_score_zh(pred: str, ground_truths: list[str]) -> float:
    f1 = 0.0
    for ground_truth in ground_truths:
        norm_pred = normalize_zh_answer(pred)
        norm_label = normalize_zh_answer(ground_truth)
        pred_tokens = list(norm_pred)
        label_tokens = list(norm_label)
        scores = f1_score(pred_tokens, label_tokens)
        this_f1, _, _ = scores
        f1 = max(f1, this_f1)
    return f1


def first_int_match(prediction: str) -> str:
    pred_list = re.split("[^0-9]", prediction)
    for item in pred_list:
        if item != "":
            return item
    return ""


def get_score_one_kv_retrieval(pred: str, label: str | list, model_name: str = "") -> float:
    if isinstance(label, list):
        label = label[0]
    return 1.0 if label in pred else 0.0


def get_score_one_passkey(pred: str, label: str | list, model_name: str = "") -> float:
    if isinstance(label, list):
        label = label[0]
    return 1.0 if label == first_int_match(pred) else 0.0


def get_score_one_number_string(pred: str, label: str | list, model_name: str = "") -> float:
    if isinstance(label, list):
        label = label[0]
    return 1.0 if label == first_int_match(pred) else 0.0


def get_score_one_math_find(pred: str, label: str | list | int | float, model_name: str = "") -> float:
    if isinstance(label, list):
        label = label[0]
    if isinstance(label, int):
        first_num = re.search(r"\d+\.\d+|\d+", pred)
        if first_num is None:
            return 0.0
        return 1.0 if int(float(first_num.group(0).strip())) == label else 0.0
    if isinstance(label, float):
        first_float = re.search(r"\d+\.\d+|\d+", pred)
        if first_float is None:
            return 0.0
        return 1.0 if float(first_float.group(0).strip()) == label else 0.0
    raise TypeError(f"Expected int or float label, got {type(label)}")


def get_score_one_longdialogue_qa_eng(pred: str, label: list | str, model_name: str = "") -> float:
    pred = pred.strip().upper()
    if not isinstance(label, list):
        label = [label]
    for item in label:
        if item.upper() in pred:
            return 1.0
    return 0.0


def get_score_one_longbook_choice_eng(pred: str, label: list | str, model_name: str = "") -> float:
    pred = pred.strip()
    if pred == "":
        return 0.0
    if not isinstance(label, list):
        label = [label]
    if pred[0] in "ABCD":
        return 1.0 if pred[0] in label else 0.0
    if pred in label:
        return 1.0
    for c in ["\n", '"', "'", ".", ",", "?", "!", "{", "}"]:
        pred = pred.replace(c, " ")
    while "  " in pred:
        pred = pred.replace("  ", " ")
    ans_prefixes = ["answer is:", "answer:", "answer is", "option is"]
    for prefix in ans_prefixes:
        idx = pred.find(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            continue
        after_prefix = pred[idx + len(prefix) + 1 :]
        for s in label:
            if after_prefix.startswith(s):
                return 1.0
    for word in pred.split():
        if word in "ABCD":
            return 1.0 if word in label else 0.0
    return 0.0


def get_score_one_longbook_qa_eng(pred: str, label: list, model_name: str = "") -> float:
    return qa_f1_score(pred, label)


def get_score_one_longbook_qa_chn(pred: str, label: list, model_name: str = "") -> float:
    return qa_f1_score_zh(pred, label)


def get_score_one_repoqa_proxy(pred: str, label: Any, model_name: str = "") -> float:
    """Lightweight substitute for RepoQA tree-sitter scoring (substring check)."""
    if isinstance(label, list):
        label = label[0]
    needle = str(label).strip()
    return 1.0 if needle and needle in pred else 0.0


def get_score_one_longbook_sum_eng(pred: str, label: str, model_name: str = "") -> float:
    if not pred.strip() or not str(label).strip():
        return 0.0
    try:
        from rouge import Rouge

        scores = Rouge().get_scores(pred, label)
        return float(scores[0]["rouge-l"]["f"])
    except ImportError:
        return 1.0 if pred.strip() == str(label).strip() else 0.0


def get_score_one(pred: str, label: Any, task_name: str, model_name: str = "") -> float:
    NAME_TO_SCORE_GETTER = {
        "kv_retrieval": get_score_one_kv_retrieval,
        "kv_retrieval_prefix": get_score_one_kv_retrieval,
        "kv_retrieval_both": get_score_one_kv_retrieval,
        "passkey": get_score_one_passkey,
        "number_string": get_score_one_number_string,
        "longdialogue_qa_eng": get_score_one_longdialogue_qa_eng,
        "longbook_qa_eng": get_score_one_longbook_qa_eng,
        "longbook_sum_eng": get_score_one_longbook_sum_eng,
        "longbook_choice_eng": get_score_one_longbook_choice_eng,
        "longbook_qa_chn": get_score_one_longbook_qa_chn,
        "math_find": get_score_one_math_find,
        "scbench_summary": get_score_one_longbook_sum_eng,
        "scbench_vt": string_match_all,
        "scbench_many_shot": get_score_one_longdialogue_qa_eng,
        "scbench_kv_compressible": get_score_one_kv_retrieval,
        "scbench_repoqa": get_score_one_repoqa_proxy,
    }
    if task_name not in NAME_TO_SCORE_GETTER:
        raise KeyError(f"Unknown SCBench metric task: {task_name}")
    score = NAME_TO_SCORE_GETTER[task_name](pred, label, model_name)
    return float(score)


def calculate_metrics(df: pd.DataFrame, data_name: str, model_name: str = "") -> dict[str, Any]:
    """
    Aggregate scores for SCBench rows (one row per turn).

    Expected columns: ``prediction`` (or ``pred``), ``ground_truth`` (or ``label``), optional ``task``.
    """
    pred_key = "prediction" if "prediction" in df.columns else "pred"
    label_key = "ground_truth" if "ground_truth" in df.columns else "label"

    if data_name in Multiturnbench_to_Infinitebench:
        task_name = Multiturnbench_to_Infinitebench[data_name]
    else:
        task_name = data_name

    scores: list[float] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="scoring"):
        pred = str(row[pred_key])
        label = row[label_key]
        if "task" in df.columns and pd.notna(row.get("task")):
            tname = str(row["task"])
            if tname in Multiturnbench_to_Infinitebench:
                tname = Multiturnbench_to_Infinitebench[tname]
        else:
            tname = task_name
        scores.append(get_score_one(pred, label, tname, model_name))

    mean_score = sum(scores) / len(scores) if scores else 0.0
    return {
        "data_name": data_name,
        "task_metric": task_name,
        "mean_score": mean_score,
        "num_rows": len(scores),
    }
