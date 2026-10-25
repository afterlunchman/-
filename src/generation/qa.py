"""RAG 问答编排：检索 -> 生成 -> 引用校验。"""

from __future__ import annotations

from typing import Any

from src.generation.citation import check_citations
from src.generation.client import LLMClient
from src.generation.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from src.ingest.chunker import Chunk
from src.retrieval.base import Retriever

_REFUSAL_MARKERS = (
    "未找到明确依据",
    "没有明确依据",
    "未找到相关",
    "没有相关规定",
    "未涵盖",
    "无法回答",
    "无法确定",
    "未作规定",
    "未规定",
)


def answer_question(
    question: str,
    retriever: Retriever,
    llm: LLMClient,
    cfg: dict[str, Any],
    top_k: int | None = None,
) -> dict[str, Any]:
    retrieval_cfg = cfg.get("retrieval", {})
    generation_cfg = cfg.get("generation", {})
    k = int(top_k or retrieval_cfg.get("top_k", 10))

    retrieved = retriever.search(question, top_k=k)
    if not retrieved:
        return _refused("未检索到相关制度条文。", [])

    max_chunks = int(generation_cfg.get("max_context_chunks", 5))
    contexts: list[Chunk] = [item.chunk for item in retrieved[:max_chunks]]

    best_score = max(item.score for item in retrieved)
    min_score = float(retrieval_cfg.get("min_score", 0.0))
    if best_score < min_score:
        return _refused("检索置信度不足，未找到可靠依据。", contexts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, contexts)},
    ]
    response = llm.chat(messages)
    citation_check = check_citations(response["content"], contexts)
    citation_required = bool(generation_cfg.get("require_citation", True))
    citation_ok = (not citation_required) or (
        citation_check.total > 0 and citation_check.invalid == 0
    )
    refused_by_text = _detect_refusal(response["content"])

    return {
        "answer": response["content"],
        "refused": refused_by_text,
        "refusal_detected": refused_by_text,
        "citation_ok": True if refused_by_text else citation_ok,
        "citation_required": citation_required,
        "citations": citation_check.to_dict(),
        "contexts": [chunk.to_dict() for chunk in contexts],
        "retrieved": [_retrieved_dict(item) for item in retrieved],
        "usage": response["usage"],
        "model": response["model"],
        "prompt_version": PROMPT_VERSION,
    }


def _retrieved_dict(item) -> dict[str, Any]:
    chunk = item.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "score": item.score,
        "retriever": item.retriever,
        "title": chunk.title,
        "clause_no": chunk.clause_no,
        "page": chunk.page,
        "confidence": chunk.confidence,
        "source_url": chunk.source_url,
        "text": chunk.text[:200],
    }


def _detect_refusal(text: str) -> bool:
    head = text[:120]
    return any(marker in head for marker in _REFUSAL_MARKERS)


def _refused(reason: str, contexts: list[Chunk]) -> dict[str, Any]:
    return {
        "answer": reason,
        "refused": True,
        "refusal_detected": True,
        "citation_ok": True,
        "citation_required": True,
        "citations": {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "has_citation": False,
            "details": [],
        },
        "contexts": [chunk.to_dict() for chunk in contexts],
        "retrieved": [],
        "usage": {},
        "model": None,
        "prompt_version": PROMPT_VERSION,
    }
