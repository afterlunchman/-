"""重排器（E6 / B4）。"""

from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from .base import Retrieved


class Reranker:
    name = "rerank"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        top_n: int = 5,
    ):
        self.model_name = model_name
        self.device = device
        self.top_n = top_n
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.text) for item in candidates]
        scores = np.asarray(
            self.model.predict(pairs, batch_size=16, show_progress_bar=False),
            dtype=float,
        )
        order = np.argsort(scores)[::-1][: self.top_n]
        return [
            Retrieved(
                chunk=candidates[int(index)].chunk,
                score=float(scores[index]),
                retriever=self.name,
            )
            for index in order
        ]
