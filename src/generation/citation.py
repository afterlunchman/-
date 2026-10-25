"""引用校验：条号存在性与原文对齐（一级，零 API 成本）。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.ingest.chunker import Chunk
from src.ingest.numerals import clause_key

CITATION_RE = re.compile(
    r"\[([^\[\]]{1,80}?)\s+(第\s*[一二三四五六七八九十百千零〇两\d]+\s*条)\]"
)


@dataclass
class CitationDetail:
    raw: str
    title_hint: str
    clause_no: str
    matched_chunk_id: str | None
    status: str  # ok | clause_not_found | title_mismatch


@dataclass
class CitationCheck:
    total: int
    valid: int
    invalid: int
    has_citation: bool
    details: list[CitationDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_citations(answer: str, contexts: list[Chunk]) -> CitationCheck:
    details: list[CitationDetail] = []
    for match in CITATION_RE.finditer(answer):
        title_hint, clause_raw = match.group(1).strip(), match.group(2)
        clause_no = re.sub(r"\s+", "", clause_raw)
        matched = _match_chunk(title_hint, clause_no, contexts)
        if matched is None:
            status = "clause_not_found"
        elif not _title_matches(title_hint, matched.title):
            status = "title_mismatch"
        else:
            status = "ok"
        details.append(
            CitationDetail(
                raw=match.group(0),
                title_hint=title_hint,
                clause_no=clause_no,
                matched_chunk_id=matched.chunk_id if matched else None,
                status=status,
            )
        )

    valid = sum(1 for item in details if item.status == "ok")
    return CitationCheck(
        total=len(details),
        valid=valid,
        invalid=len(details) - valid,
        has_citation=bool(details),
        details=details,
    )


def _match_chunk(
    title_hint: str, clause_no: str, contexts: list[Chunk]
) -> Chunk | None:
    key = clause_key(clause_no)
    candidates = [c for c in contexts if clause_key(c.clause_no) == key]
    for chunk in candidates:
        if _title_matches(title_hint, chunk.title):
            return chunk
    return candidates[0] if candidates else None


def _title_matches(title_hint: str, title: str) -> bool:
    left = re.sub(r"\s+", "", title_hint)
    right = re.sub(r"\s+", "", title)
    return left in right or right in left
