"""检索器工厂：CLI、评测与 Web 共用。"""

from __future__ import annotations

from pathlib import Path

from src.config import load_config
from src.ingest.chunker import Chunk
from src.retrieval.base import Retriever
from src.retrieval.bm25 import BM25Retriever


def build_retriever(
    name: str,
    chunks: list[Chunk],
    cfg: dict,
    rerank: bool = False,
    retrain: bool = False,
    rebuild: bool = False,
) -> Retriever:
    retrieval = cfg.get("retrieval", {})
    index_dir = Path(cfg["paths"]["index_dir"])

    if name == "bm25":
        return BM25Retriever(
            chunks,
            k1=float(retrieval.get("bm25_k1", 1.5)),
            b=float(retrieval.get("bm25_b", 0.75)),
        )

    if name == "word2vec":
        from src.retrieval.word2vec import Word2VecRetriever

        model_path = index_dir / "word2vec.model"
        if retrain and model_path.exists():
            model_path.unlink()
        return Word2VecRetriever(chunks, model_path=model_path)

    if name == "dense":
        from src.retrieval.dense import DenseRetriever

        embedding = cfg.get("embedding", {})
        return DenseRetriever(
            chunks,
            model_name=embedding.get("model", "BAAI/bge-small-zh-v1.5"),
            persist_dir=index_dir / "vectors",
            device=embedding.get("device", "cpu"),
            batch_size=int(embedding.get("batch_size", 16)),
            normalize=bool(embedding.get("normalize", True)),
            rebuild=rebuild,
        )

    if name == "hybrid":
        from src.retrieval.dense import DenseRetriever
        from src.retrieval.hybrid import HybridRetriever

        embedding = cfg.get("embedding", {})
        bm25 = BM25Retriever(
            chunks,
            k1=float(retrieval.get("bm25_k1", 1.5)),
            b=float(retrieval.get("bm25_b", 0.75)),
        )
        dense = DenseRetriever(
            chunks,
            model_name=embedding.get("model", "BAAI/bge-small-zh-v1.5"),
            persist_dir=index_dir / "vectors",
            device=embedding.get("device", "cpu"),
            batch_size=int(embedding.get("batch_size", 16)),
            normalize=bool(embedding.get("normalize", True)),
            rebuild=rebuild,
        )
        return HybridRetriever(
            [bm25, dense],
            rrf_k=int(retrieval.get("rrf_k", 60)),
            reranker=build_reranker(cfg) if rerank else None,
        )

    raise ValueError(f"未知检索器：{name}")


def build_reranker(cfg: dict):
    from src.retrieval.rerank import Reranker

    rerank_cfg = cfg.get("rerank", {})
    return Reranker(
        model_name=rerank_cfg.get("model", "BAAI/bge-reranker-base"),
        device=rerank_cfg.get("device", "cpu"),
        top_n=int(rerank_cfg.get("top_n", 5)),
    )
