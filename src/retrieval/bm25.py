"""BM25 关键词检索基线（E1 / B1）。"""

from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingest.chunker import Chunk

from .base import Retrieved
from .tokenizer import tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.tokenized = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(self.tokenized, k1=k1, b=b)

    def search(self, query: str, top_k: int = 10) -> list[Retrieved]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = np.asarray(self.index.get_scores(tokens), dtype=float)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Retrieved(chunk=self.chunks[int(i)], score=float(scores[i]), retriever=self.name)
            for i in order
            if scores[i] > 0
        ]
