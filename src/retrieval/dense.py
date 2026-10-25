"""BGE 密集检索（E1 / B3，本地 numpy 向量索引）。

Chroma 1.5.9 在本机 Windows 上二次打开 HNSW 索引时报错，因此改用
numpy 持久化：embeddings.npy + chunk_ids.json + meta.json。
索引规模约 4,976 × 512，单查询余弦相似度为毫秒级。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingest.chunker import Chunk

from .base import Retrieved


class DenseRetriever:
    name = "dense_bge"

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = "BAAI/bge-small-zh-v1.5",
        persist_dir: str | Path = "data/index/vectors",
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        rebuild: bool = False,
    ):
        self.chunks = chunks
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.persist_dir / "embeddings.npy"
        self.ids_path = self.persist_dir / "chunk_ids.json"
        self.meta_path = self.persist_dir / "meta.json"

        self.model = SentenceTransformer(model_name, device=device)
        if rebuild or not self._load():
            self._build()

    def _load(self) -> bool:
        if not (
            self.embeddings_path.exists()
            and self.ids_path.exists()
            and self.meta_path.exists()
        ):
            return False
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") != self.model_name:
            return False
        if int(meta.get("count", -1)) != len(self.chunks):
            return False
        self.embeddings = np.load(self.embeddings_path)
        self.chunk_ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        return len(self.chunk_ids) == len(self.embeddings)

    def _build(self) -> None:
        embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        np.save(self.embeddings_path, self.embeddings)
        self.ids_path.write_text(
            json.dumps(self.chunk_ids, ensure_ascii=False), encoding="utf-8"
        )
        self.meta_path.write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "count": len(self.chunk_ids),
                    "dim": int(self.embeddings.shape[1]),
                    "normalize": self.normalize,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def search(self, query: str, top_k: int = 10) -> list[Retrieved]:
        if len(self.embeddings) == 0:
            return []
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )[0].astype(np.float32)
        scores = self.embeddings @ query_vector
        order = np.argsort(scores)[::-1][:top_k]
        results: list[Retrieved] = []
        for index in order:
            chunk = self.by_id.get(self.chunk_ids[int(index)])
            if chunk is not None:
                results.append(
                    Retrieved(
                        chunk=chunk,
                        score=float(scores[index]),
                        retriever=self.name,
                    )
                )
        return results
