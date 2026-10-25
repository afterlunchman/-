"""LLM 评审：rubric 打分与 JSON 解析。"""

from __future__ import annotations

import json
import re
from typing import Any

from src.generation.client import LLMClient

JUDGE_SYSTEM = """你是严格、客观的评测员。根据问题、金标准答案和待评答案，按 1—5 分打分：
- correctness：事实是否正确，数值、期限、条号是否准确；
- citation_support：答案中的引用是否确实支持对应陈述；
- completeness：是否覆盖金标准中的关键要点。
只输出 JSON，不要输出其他文字：
{"correctness": 1-5, "citation_support": 1-5, "completeness": 1-5, "rationale": "简短理由"}"""


def build_judge_prompt(
    question: str,
    gold_answer: str,
    model_answer: str,
    contexts_text: str = "",
) -> str:
    parts = [
        f"问题：{question}",
        f"金标准答案：{gold_answer}",
        f"待评答案：{model_answer}",
    ]
    if contexts_text:
        parts.append(f"检索上下文：\n{contexts_text}")
    return "\n\n".join(parts)


def parse_judge_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError(f"评审输出不是 JSON：{text[:200]}")
    return json.loads(match.group(0))


def judge_answer(
    client: LLMClient,
    question: str,
    gold_answer: str,
    model_answer: str,
    contexts_text: str = "",
) -> dict[str, Any]:
    response = client.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": build_judge_prompt(
                    question, gold_answer, model_answer, contexts_text
                ),
            },
        ],
        temperature=0.0,
    )
    result = parse_judge_json(response["content"])
    result["usage"] = response["usage"]
    return result
