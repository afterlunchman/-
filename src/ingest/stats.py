"""语料与切块统计。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .chunker import Chunk
from .loader import Document


def collect_stats(
    docs: list[Document],
    chunks: list[Chunk],
    errors: list[str] | None = None,
) -> dict[str, Any]:
    docs_with_clause = {c.doc_id for c in chunks if c.chunk_kind == "clause"}
    chars = [len(doc.text) for doc in docs]
    chunk_chars = [len(c.text) for c in chunks]
    confidences = [round(c.confidence, 2) for c in chunks]

    return {
        "documents": {
            "total": len(docs),
            "with_clause": len(docs_with_clause),
            "without_clause": len(docs) - len(docs_with_clause),
            "characters_total": sum(chars),
            "characters_avg": round(sum(chars) / len(chars), 1) if chars else 0,
            "characters_min": min(chars) if chars else 0,
            "characters_max": max(chars) if chars else 0,
            "by_group": dict(Counter(d.group for d in docs).most_common()),
            "by_text_source": dict(Counter(d.text_source for d in docs).most_common()),
            "attachments_total": sum(len(d.attachments) for d in docs),
        },
        "chunks": {
            "total": len(chunks),
            "clause": sum(1 for c in chunks if c.chunk_kind == "clause"),
            "paragraph": sum(1 for c in chunks if c.chunk_kind == "paragraph"),
            "characters_avg": round(sum(chunk_chars) / len(chunk_chars), 1) if chunk_chars else 0,
            "characters_max": max(chunk_chars) if chunk_chars else 0,
            "by_confidence": dict(Counter(confidences).most_common()),
        },
        "sections": dict(Counter(d.section for d in docs).most_common(15)),
        "errors": errors or [],
    }


def print_summary(stats: dict[str, Any]) -> None:
    d = stats["documents"]
    c = stats["chunks"]
    print(f"文档：{d['total']} 份（缺失/失败 {len(stats['errors'])} 份）")
    print(
        f"正文：{d['characters_total']:,} 字，"
        f"平均 {d['characters_avg']} 字/份，最长 {d['characters_max']:,} 字"
    )
    print(
        f"切块：{c['total']:,} 个（条文 {c['clause']:,} / 段落 {c['paragraph']:,}），"
        f"平均 {c['characters_avg']} 字，最长 {c['characters_max']:,} 字"
    )
    print(
        f"条文覆盖：{d['with_clause']} 份有条号，"
        f"{d['without_clause']} 份回退段落切分"
    )
    print(f"附件：{d['attachments_total']} 个")
    print(f"正文来源：{d['by_text_source']}")
    print(f"置信度分布：{c['by_confidence']}")
    if stats["errors"]:
        print("错误：")
        for error in stats["errors"][:10]:
            print(f"  - {error}")
