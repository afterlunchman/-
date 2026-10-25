"""检索器公共接口与 chunk 读取工具。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.ingest.chunker import Chunk


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    retriever: str


class Retriever(Protocol):
    name: str

    def search(self, query: str, top_k: int = 10) -> list[Retrieved]:
        ...


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks
