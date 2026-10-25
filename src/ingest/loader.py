"""读取 index.jsonl 与对应 Markdown 正文。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)


@dataclass
class Document:
    doc_id: str
    title: str
    section: str
    group: str
    source_url: str
    published_at: str
    text_source: str
    md_path: Path
    text: str
    attachments: list[dict[str, Any]] = field(default_factory=list)
    frontmatter: dict[str, Any] = field(default_factory=dict)


def read_index(corpus_dir: Path) -> list[dict[str, Any]]:
    index_path = corpus_dir / "index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"未找到语料索引：{index_path}")
    rows: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_documents(
    corpus_dir: Path, limit: int | None = None
) -> tuple[list[Document], list[str]]:
    """返回 (文档列表, 错误列表)。单条文档失败不阻断整体加载。"""
    docs: list[Document] = []
    errors: list[str] = []
    rows = read_index(corpus_dir)
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        doc_id = str(row.get("doc_id", ""))
        md_rel = row.get("md_path") or ""
        md_path = corpus_dir / md_rel
        if not md_rel or not md_path.exists():
            errors.append(f"{doc_id}: 缺少 Markdown 文件 {md_rel!r}")
            continue
        try:
            raw = md_path.read_text(encoding="utf-8")
            meta, body = split_frontmatter(raw)
            text = clean_body(body)
        except Exception as exc:  # noqa: BLE001 - 汇总到错误列表
            errors.append(f"{doc_id}: 读取失败 {exc}")
            continue

        attachments = _normalize_attachments(meta.get("attachments") or row.get("attachments"))
        docs.append(
            Document(
                doc_id=doc_id,
                title=str(meta.get("title") or row.get("title") or md_path.stem),
                section=str(meta.get("section") or row.get("category") or ""),
                group=str(meta.get("group") or row.get("group") or ""),
                source_url=str(meta.get("source_url") or row.get("source_url") or ""),
                published_at=str(meta.get("published_at") or row.get("published_at") or ""),
                text_source=str(meta.get("text_source") or row.get("text_source") or ""),
                md_path=md_path,
                text=text,
                attachments=attachments,
                frontmatter=meta,
            )
        )
    return docs, errors


def split_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    return meta, raw[match.end() :]


def clean_body(body: str) -> str:
    """去掉标题、溯源块与附件段，只保留正文。"""
    kept: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("## 附件"):
            break
        if not stripped:
            kept.append("")
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("> **溯源**") or stripped.startswith("> **正文来源**"):
            continue
        if stripped == "---":
            continue
        kept.append(line.rstrip())
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_attachments(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item if isinstance(item, dict) else {"path": str(item)} for item in value]
    if isinstance(value, str) and value:
        return [{"path": value}]
    return []
