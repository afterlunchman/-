"""条文级切块、段落回退与父级扩展。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .loader import Document
from .numerals import clause_to_int

CLAUSE_RE = re.compile(r"第\s*([一二三四五六七八九十百千零〇两\d]+)\s*条")
PAGE_RE = re.compile(r"【第\s*(\d+)\s*页】")

_REFERENCE_NEXT = ("规定", "之规定", "第", "款", "项", "、", "，", "。", "；", "）", ")")
_REFERENCE_PREV = ("《", "》", "根据", "依据", "按照", "参照", "依照", "见", "本法", "本办法", "该办法")

CONFIDENCE = {
    "web_page": 1.0,
    "pdf_text_layer": 0.9,
    "docx_text": 0.9,
    "ocr_vision": 0.75,
    "ocr_rapidocr_local": 0.6,
    "ocr_vision+rapidocr": 0.7,
}
DEFAULT_CONFIDENCE = 0.7


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    clause_no: str | None
    chunk_kind: str
    order: int
    page: int | None
    text: str
    text_source: str
    confidence: float
    source_url: str
    attachment_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def chunk_document(
    doc: Document, max_chars: int = 600, overlap: int = 100
) -> list[Chunk]:
    body = doc.text
    if not body.strip():
        return []

    matches = _select_clause_matches(body)
    if matches:
        return _clause_chunks(doc, body, matches, max_chars, overlap)
    return _paragraph_chunks(doc, body, max_chars, overlap)


def _select_clause_matches(body: str) -> list[re.Match[str]]:
    """选出条号连续递增的一串条文，过滤正文中的交叉引用。"""
    candidates: list[tuple[re.Match[str], int]] = []
    for match in CLAUSE_RE.finditer(body):
        number = clause_to_int(match.group(1))
        if number is not None:
            candidates.append((match, number))
    if not candidates:
        return []

    best_seq: list[int] = []
    best_score = -1
    for i in range(len(candidates)):
        seq = [i]
        current = candidates[i][1]
        j = i + 1
        while j < len(candidates):
            if candidates[j][1] == current + 1:
                seq.append(j)
                current += 1
            j += 1
        score = len(seq) * 2 - sum(
            _reference_penalty(body, candidates[k][0]) for k in seq
        )
        if score > best_score or (score == best_score and len(seq) > len(best_seq)):
            best_score = score
            best_seq = seq

    if len(best_seq) < 2:
        return []
    return [candidates[k][0] for k in best_seq]


def _reference_penalty(body: str, match: re.Match[str]) -> int:
    after = body[match.end() : match.end() + 4]
    before = body[max(0, match.start() - 4) : match.start()]
    penalty = 0
    if any(after.startswith(token) for token in _REFERENCE_NEXT):
        penalty += 1
    if any(before.endswith(token) for token in _REFERENCE_PREV):
        penalty += 2
    return penalty


def _clause_chunks(
    doc: Document,
    body: str,
    matches: list[re.Match[str]],
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    max_clause_chars = max(max_chars * 2, 1200)
    overlap = max(0, min(overlap, max_clause_chars // 4))
    step = max_clause_chars - overlap

    chunks: list[Chunk] = []
    order = 0
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        raw = body[start:end].strip()
        text = PAGE_RE.sub("", raw).strip()
        if len(text) < 5:
            continue
        clause_no = _normalize_clause(match.group(0))

        if len(text) <= max_clause_chars:
            chunks.append(
                _make_chunk(
                    doc=doc,
                    order=order,
                    clause_no=clause_no,
                    chunk_kind="clause",
                    page=_page_at(body, start),
                    text=text,
                )
            )
            order += 1
            continue

        for offset in range(0, len(raw), step):
            piece = PAGE_RE.sub("", raw[offset : offset + max_clause_chars]).strip()
            if len(piece) < 20 and offset > 0:
                break
            chunks.append(
                _make_chunk(
                    doc=doc,
                    order=order,
                    clause_no=clause_no,
                    chunk_kind="clause",
                    page=_page_at(body, start + offset),
                    text=piece,
                )
            )
            order += 1
    return chunks


def _paragraph_chunks(
    doc: Document, body: str, max_chars: int, overlap: int
) -> list[Chunk]:
    text = body.strip()
    max_chars = max(max_chars, 100)
    overlap = max(0, min(overlap, max_chars // 2))
    step = max_chars - overlap

    chunks: list[Chunk] = []
    order = 0
    for start in range(0, len(text), step):
        piece = text[start : start + max_chars]
        cleaned = PAGE_RE.sub("", piece).strip()
        if len(cleaned) < 20:
            if start > 0:
                break
            continue
        chunks.append(
            _make_chunk(
                doc=doc,
                order=order,
                clause_no=None,
                chunk_kind="paragraph",
                page=_page_at(body, start),
                text=cleaned,
            )
        )
        order += 1
    return chunks


def _make_chunk(
    doc: Document,
    order: int,
    clause_no: str | None,
    chunk_kind: str,
    page: int | None,
    text: str,
) -> Chunk:
    attachment_url = None
    for item in doc.attachments:
        url = item.get("url") if isinstance(item, dict) else None
        if url:
            attachment_url = str(url)
            break
    return Chunk(
        chunk_id=f"{doc.doc_id}-{order:03d}",
        doc_id=doc.doc_id,
        title=doc.title,
        section=doc.section,
        clause_no=clause_no,
        chunk_kind=chunk_kind,
        order=order,
        page=page,
        text=text,
        text_source=doc.text_source,
        confidence=CONFIDENCE.get(doc.text_source, DEFAULT_CONFIDENCE),
        source_url=doc.source_url,
        attachment_url=attachment_url,
    )


def _normalize_clause(raw: str) -> str:
    return re.sub(r"\s+", "", raw)


def _page_at(body: str, position: int) -> int | None:
    page = None
    for match in PAGE_RE.finditer(body, 0, position):
        page = int(match.group(1))
    return page


def expand_parent(
    chunks: list[Chunk], index: int, window: int = 1
) -> list[Chunk]:
    """返回命中块及其前后 window 条，限定在同文档内。"""
    if not chunks:
        return []
    target = chunks[index]
    result: list[Chunk] = []
    step = -window
    while step <= window:
        pos = index + step
        if 0 <= pos < len(chunks) and chunks[pos].doc_id == target.doc_id:
            result.append(chunks[pos])
        step += 1
    return result
