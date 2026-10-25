"""生成 80 题评测集草稿：70 可答 + 10 不可答。

注意：这是 AI 起草版本，字段中标记 annotator="ai-draft"，正式使用前需人工复核。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from src.config import load_config
from src.generation.client import LLMClient
from src.ingest.chunker import Chunk
from src.retrieval.base import load_chunks

TYPES = ["定义/禁止", "数值/期限", "流程/职责", "边界/否定"]

GEN_PROMPT = """你是课程项目的测试集编写助手。下面给出若干条学校制度条文，请为每条写一个学生、辅导员或教职工在校园场景中可能问的问题，并给出仅依据该条文的简短答案。

要求：
0. 问题必须发生在校园场景；不要问企业、铁路、矿山、航空等与校园无关的主体；若条文主体与校园无关，跳过该条；
1. 问题必须能由对应条文直接回答，不得引入条文以外的知识；
2. 答案 1—3 句，保留关键数值、期限、条件；
3. 每条条文只出 1 题；题型从 ["定义/禁止","数值/期限","流程/职责","边界/否定"] 中选一个；
4. 只输出 JSON 数组，每项 {"index": 序号, "question": "...", "answer": "...", "type": "..."}；index 必须与方括号中的编号完全一致（从 0 开始）；不要输出 Markdown 代码块或其他文字。

条文：
{items}"""

UNANSWERABLE = [
    ("补考成绩怎么算？", "学籍考试"),
    ("奖学金评定需要满足哪些条件？", "奖惩评优"),
    ("学费最晚什么时候缴纳？", "收费"),
    ("如何申请转专业？", "学籍"),
    ("保研资格是怎么确定的？", "学籍"),
    ("图书馆每天几点开门？", "校园服务"),
    ("校园网密码怎么修改？", "校园服务"),
    ("校医院周末上班吗？", "校园服务"),
    ("宿舍热水供应时间是几点？", "校园服务"),
    ("食堂饭菜价格在哪里查询？", "校园服务"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成评测集草稿")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-answerable", type=int, default=70)
    parser.add_argument("--num-unanswerable", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    chunks_path = Path(cfg["paths"]["index_dir"]) / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    candidates = select_candidates(chunks, args.seed)
    print(f"候选条文：{len(candidates)} 条")

    client = LLMClient(cfg)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < args.num_answerable and cursor < len(candidates):
        batch = candidates[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        items = _generate_batch(client, batch)
        for item in items:
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(batch):
                continue
            try:
                selected.append(_to_record(batch[index], item))
            except ValueError:
                continue
        print(f"进度：{len(selected)}/{args.num_answerable}（已尝试 {cursor} 条）")

    selected = selected[: args.num_answerable]
    records = [_with_id(item, f"q{index:03d}") for index, item in enumerate(selected, 1)]
    for index, (question, topic) in enumerate(
        UNANSWERABLE[: args.num_unanswerable], start=1
    ):
        records.append(
            {
                "question_id": f"u{index:03d}",
                "question": question,
                "type": "不可答",
                "answerable": False,
                "gold_doc_ids": [],
                "gold_clauses": [],
                "gold_answer": "现有语料未覆盖该主题，应拒答并建议咨询相关部门。",
                "evidence_span": "",
                "source_url": "",
                "chunk_id": "",
                "difficulty": "easy",
                "topic": topic,
                "annotator": "ai-draft",
                "reviewer": "",
            }
        )

    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg["paths"]["eval_set"])
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"已写入 {out_path}：{len(records)} 题")
    print(f"可答 {len(records) - args.num_unanswerable} 题，不可答 {args.num_unanswerable} 题")


def select_candidates(chunks: list[Chunk], seed: int) -> list[Chunk]:
    pool = [
        chunk
        for chunk in chunks
        if chunk.chunk_kind == "clause"
        and chunk.clause_no
        and 80 <= len(chunk.text) <= 800
        and chunk.confidence >= 0.9
    ]
    by_group: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in pool:
        by_group[chunk.section.split("/")[0].strip() or "未分类"].append(chunk)

    rng = random.Random(seed)
    for group_chunks in by_group.values():
        rng.shuffle(group_chunks)

    ordered: list[Chunk] = []
    names = sorted(by_group, key=lambda name: len(by_group[name]), reverse=True)
    while any(by_group[name] for name in names):
        for name in names:
            if by_group[name]:
                ordered.append(by_group[name].pop())

    # 校级制度优先，国家/地方法规作为补充。
    school = [chunk for chunk in ordered if "东北大学" in chunk.title]
    other = [chunk for chunk in ordered if "东北大学" not in chunk.title]
    return school + other


def _generate_batch(client: LLMClient, batch: list[Chunk]) -> list[dict]:
    items = "\n\n".join(
        f"[{index}] 制度：{chunk.title}；条号：{chunk.clause_no}；原文：{chunk.text[:600]}"
        for index, chunk in enumerate(batch)
    )
    for _ in range(2):
        response = client.chat(
            [
                {"role": "system", "content": "你是严谨的测试集编写助手，只输出 JSON。"},
                {"role": "user", "content": GEN_PROMPT.replace("{items}", items)},
            ],
            temperature=0.2,
        )
        try:
            return _parse_array(response["content"])
        except (ValueError, json.JSONDecodeError):
            continue
    return []


def _parse_array(text: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\[.*\]", cleaned, re.S)
    if not match:
        raise ValueError("no JSON array")
    data = json.loads(match.group(0))
    if not isinstance(data, list):
        raise ValueError("not a list")
    return [item for item in data if isinstance(item, dict)]


def _to_record(chunk: Chunk, item: dict) -> dict:
    question = str(item.get("question", "")).strip()
    answer = str(item.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError("empty question or answer")
    question_type = item.get("type", "定义/禁止")
    if question_type not in TYPES:
        question_type = _classify_question(question)
    return {
        "question": question,
        "type": question_type,
        "answerable": True,
        "gold_doc_ids": [chunk.doc_id],
        "gold_clauses": [chunk.clause_no],
        "gold_answer": answer,
        "evidence_span": chunk.text[:300],
        "source_url": chunk.source_url,
        "chunk_id": chunk.chunk_id,
        "difficulty": "medium",
        "topic": chunk.section,
        "annotator": "ai-draft",
        "reviewer": "",
    }


def _classify_question(question: str) -> str:
    if re.search(r"多久|多长时间|几天|期限|何时|多少|收费|标准|时间", question):
        return "数值/期限"
    if re.search(r"是否|能否|可以|允许|禁止|不得|严禁", question):
        return "边界/否定"
    if re.search(r"如何|怎么|流程|步骤|向谁|报告|申请|办理", question):
        return "流程/职责"
    return "定义/禁止"


def _with_id(record: dict, question_id: str) -> dict:
    result = {"question_id": question_id}
    result.update(record)
    return result


if __name__ == "__main__":
    main()
