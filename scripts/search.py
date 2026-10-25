"""命令行检索入口。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.config import load_config
from src.retrieval.base import load_chunks
from src.retrieval.factory import build_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在制度语料上做检索")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.yaml")
    parser.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "word2vec", "dense", "hybrid"],
        help="检索器",
    )
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数")
    parser.add_argument("--rerank", action="store_true", help="hybrid 后接重排")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--retrain", action="store_true", help="重新训练 Word2Vec")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    parser.add_argument("--limit-chunks", type=int, default=None, help="只加载前 N 个块")
    return parser.parse_args()


def print_results(query: str, results) -> None:
    print(f"查询：{query}")
    if not results:
        print("（无结果）")
        return
    for rank, item in enumerate(results, start=1):
        chunk = item.chunk
        meta = [chunk.title, chunk.clause_no or "无条号"]
        if chunk.page is not None:
            meta.append(f"第 {chunk.page} 页")
        meta.append(f"置信度 {chunk.confidence:.2f}")
        print(f"[{rank}] {item.score:.4f}  " + " | ".join(meta))
        text = chunk.text.replace("\n", " ")
        print(f"    {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"    来源：{chunk.source_url}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    chunks_path = Path(cfg["paths"]["index_dir"]) / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"未找到 {chunks_path}，请先运行：python -m scripts.build_index")
        return

    chunks = load_chunks(chunks_path)
    if args.limit_chunks:
        chunks = chunks[: args.limit_chunks]

    retriever = build_retriever(
        args.retriever,
        chunks,
        cfg,
        rerank=args.rerank,
        retrain=args.retrain,
        rebuild=args.rebuild,
    )
    top_k = args.top_k or int(cfg.get("retrieval", {}).get("top_k", 10))
    results = retriever.search(args.query, top_k=top_k)

    if args.json:
        payload = [
            {
                "score": item.score,
                "retriever": item.retriever,
                "chunk": asdict(item.chunk),
            }
            for item in results
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_results(args.query, results)


if __name__ == "__main__":
    main()
