"""第 5 讲证据：Word2Vec 的 OOV 率与否定句表示。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.config import load_config
from src.retrieval.base import load_chunks
from src.retrieval.tokenizer import tokenize
from src.retrieval.word2vec import Word2VecRetriever


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    index_dir = Path(cfg["paths"]["index_dir"])
    chunks = load_chunks(index_dir / "chunks.jsonl")
    retriever = Word2VecRetriever(chunks, model_path=index_dir / "word2vec.model")

    positive = "宿舍可以留宿外来人员吗"
    negative = "宿舍不可以留宿外来人员吗"
    v_pos = retriever._sentence_vector(tokenize(positive))
    v_neg = retriever._sentence_vector(tokenize(negative))

    payload = {
        "oov_rate": round(retriever.oov_rate, 4),
        "negation_cosine": round(float(v_pos @ v_neg), 4),
        "queries": {
            positive: [f"{item.chunk.title} {item.chunk.clause_no}" for item in retriever.search(positive, 5)],
            negative: [f"{item.chunk.title} {item.chunk.clause_no}" for item in retriever.search(negative, 5)],
        },
    }
    out_path = Path(cfg["paths"]["results_dir"]) / "word2vec_analysis.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"已写入：{out_path}")


if __name__ == "__main__":
    main()
