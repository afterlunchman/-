"""答案生成提示词。"""

from __future__ import annotations

from src.ingest.chunker import Chunk

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """你是校园安全制度问答助手。只能依据给定的制度条文回答，不得使用条文以外的知识。

回答要求：
1. 先给结论，再给依据，语言简洁。
2. 每个事实性陈述后用 [制度名 第X条] 标注引用，制度名与条号必须与证据一致。
3. 条文不足以回答时，第一句话必须是「现有制度未找到明确依据」，简要说明缺什么，并建议咨询的部门；不要引用与问题无关的条款来凑引用。
4. 不得编造条号、数值、期限或处罚。
5. 如果依据来自扫描件识别（证据中标注了低置信度），提醒用户核对原始附件。
6. 不要输出与问题无关的制度条款。"""


def build_user_prompt(question: str, contexts: list[Chunk]) -> str:
    blocks: list[str] = []
    for index, chunk in enumerate(contexts, start=1):
        meta = [f"制度：{chunk.title}", f"条号：{chunk.clause_no or '无'}"]
        if chunk.page is not None:
            meta.append(f"页码：第 {chunk.page} 页")
        meta.append(f"置信度：{chunk.confidence:.2f}")
        blocks.append(
            f"【证据 {index}】" + "；".join(meta) + "\n"
            f"原文：{chunk.text}\n"
            f"来源：{chunk.source_url}"
        )

    evidence = "\n\n".join(blocks) if blocks else "（无检索结果）"
    return f"以下是检索到的制度条文：\n\n{evidence}\n\n用户问题：{question}\n\n请按要求作答。"
