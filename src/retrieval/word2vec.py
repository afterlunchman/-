"""词向量平均检索基线（E1 / B2，课程第 5 讲）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gensim.models import Word2Vec

from src.ingest.chunker import Chunk

from .base import Retrieved
from .tokenizer import tokenize


class Word2VecRetriever:
    name = "word2vec_avg"

    def __init__(
        self,
        chunks: list[Chunk],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 20,
        seed: int = 42,
        model_path: str | Path | None = None,
    ):
        self.chunks = chunks
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.model_path = Path(model_path) if model_path else None
        self.tokenized = [tokenize(chunk.text) for chunk in chunks]

        if self.model_path and self.model_path.exists():
            self.model = Word2Vec.load(str(self.model_path))
        else:
            self.model = Word2Vec(
                sentences=self.tokenized,
                vector_size=vector_size,
                window=window,
                min_count=min_count,
                workers=1,
                sg=1,
                epochs=epochs,
                seed=seed,
            )
            if self.model_path:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.model_path))

        self.oov_rate = self._compute_oov_rate()
        self.vectors = self._build_matrix()

    def search(self, query: str, top_k: int = 10) -> list[Retrieved]:
        query_vector = self._sentence_vector(tokenize(query))
        if not np.any(query_vector):
            return []
        scores = self.vectors @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Retrieved(chunk=self.chunks[int(i)], score=float(scores[i]), retriever=self.name)
            for i in order
            if scores[i] > 0
        ]

    def _sentence_vector(self, tokens: list[str]) -> np.ndarray:
        vectors = [self.model.wv[token] for token in tokens if token in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size, dtype=float)
        vector = np.mean(vectors, axis=0)
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def _build_matrix(self) -> np.ndarray:
        matrix = np.vstack([self._sentence_vector(tokens) for tokens in self.tokenized])
        return matrix.astype(np.float32)

    def _compute_oov_rate(self) -> float:
        total = sum(len(tokens) for tokens in self.tokenized)
        if not total:
            return 0.0
        known = sum(
            1 for tokens in self.tokenized for token in tokens if token in self.model.wv
        )
        return 1.0 - known / total
