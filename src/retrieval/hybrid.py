"""多路检索融合（RRF，B5）。"""

from __future__ import annotations

from .base import Retriever, Retrieved


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = 60,
        reranker=None,
    ):
        self.retrievers = retrievers
        self.rrf_k = rrf_k
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> list[Retrieved]:
        candidate_k = max(top_k * 3, 20)
        scores: dict[str, float] = {}
        chunks = {}

        for retriever in self.retrievers:
            results = retriever.search(query, top_k=candidate_k)
            for rank, item in enumerate(results, start=1):
                chunk_id = item.chunk.chunk_id
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )
                chunks[chunk_id] = item.chunk

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        candidates = [
            Retrieved(chunk=chunks[chunk_id], score=score, retriever=self.name)
            for chunk_id, score in ranked
        ]

        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates)

        return candidates[:top_k]
