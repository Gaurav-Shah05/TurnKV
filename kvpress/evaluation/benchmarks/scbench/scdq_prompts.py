# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# SCDQ prompt construction adapted from Microsoft MInference (MIT License):
# https://github.com/microsoft/MInference/blob/main/scbench/eval_utils.py

from __future__ import annotations

import re
DATA_NAME_TO_MAX_NEW_TOKENS = {
    "scbench_choice_eng": 40,
    "scbench_qa_eng": 40,
    "scbench_qa_chn": 40,
    "scbench_kv": 150,
    "scbench_kv_hard": 150,
    "scbench_mf": 5,
    "scbench_hashhop": 150,
    "scbench_prefix_suffix": 150,
    "scbench_kv_compressible": 150,
    "scbench_passkey": 15,
    "scbench_repoqa": 1024,
    "scbench_summary": 200,
    "scbench_vt": 30,
    "scbench_many_shot": 10,
    "scbench_summary_with_needles": {"scbench_summary": 800, "scbench_passkey": 15},
    "scbench_repoqa_and_kv": {"scbench_repoqa": 1024, "scbench_kv": 80},
}

multiturn_templates_scdq = {
    "scbench_passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}",  # noqa
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}",  # noqa
    "scbench_kv_hard": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}",  # noqa
    "scbench_kv_compressible": "Extract the value corresponding to the specified key in the following passage.\n\n{context}",  # noqa
    "scbench_choice_eng": (
        "Read the book and answer the question.\n\n{context}",
        "Question: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",
    ),
    "scbench_qa_eng": (
        "Read the book and answer the question. Be very concise in your answer.\n\n{context}",
        "Question: {question}\nAnswer:",
    ),
    "scbench_qa_chn": ("阅读以下书籍然后回答问题。\n\n{context}", "问题：{question}\n答案："),
    "scbench_mf": "{prefix}\n\n{context}",
    "scbench_repoqa": "Based on the function description and code context, please retrieve and repeat the exact described function from the code context in a code block wrapped by ```:\n\n{context}",
    "scbench_summary": "{context}",
    "scbench_vt": "{context}",
    "scbench_many_shot": "{context}",
    "scbench_summary_with_needles": "{context}",
    "scbench_repoqa_and_kv": "{context}",
    "scbench_hashhop": "{context}",
    "scbench_prefix_suffix": "{context}",
}

multiturn_follow_up_templates_in_chat_tempate = {
    "scbench_passkey": "{input}",  # noqa
    "scbench_kv": "{input}",  # noqa
    "scbench_kv_hard": "{input}",  # noqa
    "scbench_kv_compressible": "{input}",  # noqa
    "scbench_choice_eng": "Question: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe the correct answer is",  # noqa
    "scbench_qa_eng": "Question: {question}\nAnswer:",  # noqa
    "scbench_qa_chn": "问题：{question}\n答案：",  # noqa
    "scbench_mf": "{prefix}\n\n{input}",
    "scbench_repoqa": "{input}",
    "scbench_summary": "{input}",
    "scbench_vt": "{input}",
    "scbench_many_shot": "{input}",
    "scbench_summary_with_needles": "{input}",
    "scbench_repoqa_and_kv": "{input}",
    "scbench_hashhop": "{input}",
    "scbench_prefix_suffix": "{input}",
}

def create_scdq_prompt(
    eg: dict, data_name: str, tok, use_chat_template, use_vllm=False
):
    template = multiturn_templates_scdq[data_name]
    query_template = multiturn_follow_up_templates_in_chat_tempate[data_name]

    special_delimiter = "[SEPSEPSEP]"

    if data_name == "scbench_choice_eng":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
                OPTION_A=turn["options"][0],
                OPTION_B=turn["options"][1],
                OPTION_C=turn["options"][2],
                OPTION_D=turn["options"][3],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        prompts = [context_prompt] + query_prompts

        return {
            "prompts": prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
            "options": eg["multi_turns"][0]["options"],
        }

    elif data_name == "scbench_qa_eng":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_qa_chn":
        context = eg["context"]
        context_prompt = template[0].format(context=context)
        query_prompts = [
            template[1].format(
                question=turn["input"],
            )
            for turn in eg["multi_turns"]
        ]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name == "scbench_mf":
        context = eg["context"]
        context_prompt = template.format(
            prefix=eg["multi_turns"][0]["input"],
            context=context,
        )

        query_prompts = []
        for i in range(len(eg["multi_turns"])):
            target = re.findall(r"The .+ is", eg["multi_turns"][i]["input"])[0].lower()[
                :-3
            ]
            prefix = f"What is {target}?"
            query_prompts.append(
                query_template.format(
                    prefix=prefix,
                    input=eg["multi_turns"][i]["input"],
                )
            )

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        return {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

    elif data_name in [
        "scbench_repoqa",
        "scbench_summary",
        "scbench_passkey",
        "scbench_kv",
        "scbench_vt",
        "scbench_many_shot",
        "scbench_summary_with_needles",
        "scbench_repoqa_and_kv",
        "scbench_kv_hard",
        "scbench_hashhop",
        "scbench_prefix_suffix",
        "scbench_kv_compressible",
    ]:
        context = eg["context"] if "context" in eg else eg["input"]
        context_prompt = template.format(context=context)
        query_prompts = [turn["input"] for turn in eg["multi_turns"]]

        if use_chat_template:
            context_prompt = tok.apply_chat_template(
                [{"role": "user", "content": context_prompt + special_delimiter}],
                add_generation_prompt=True,
                tokenize=False,
            )
            context_prompt = context_prompt.split(special_delimiter)[0]

            query_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "system", "content": ""},
                        {"role": "user", "content": special_delimiter + query_prompt},
                    ],
                    add_generation_prompt=True,
                    tokenize=False,
                ).split(special_delimiter)[1]
                for query_prompt in query_prompts
            ]

        output = {
            "prompts": [context_prompt] + query_prompts,
            "ground_truth": [gt["answer"] for gt in eg["multi_turns"]],
        }

        if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
            output["task"] = [gt["task"] for gt in eg["multi_turns"]]

        return output


def get_ground_truth(eg: dict, data_name: str):
    """Ground-truth labels formatted like MInference ``get_ground_truth``."""
    gts = []
    options_letters = "ABCD"
    for turn in eg["multi_turns"]:
        if data_name == "scbench_choice_eng":
            ans_ = turn["answer"]
            options = turn["options"]
            gts.append([ans_, options_letters[options.index(ans_)]])
        elif data_name == "scbench_qa_eng":
            gts.append([turn["answer"]])
        else:
            gts.append(turn["answer"])
    return gts
