"""统计语料规模、条文覆盖与切块结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import load_config
from src.ingest.chunker import chunk_document
from src.ingest.loader import load_documents
from src.ingest.stats import collect_stats, print_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查语料完整性并统计切块结果")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 份，便于调试")
    parser.add_argument("--out", default=None, help="统计结果输出路径")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    corpus_dir = Path(cfg["paths"]["corpus_dir"])
    retrieval = cfg.get("retrieval", {})
    max_chars = int(retrieval.get("chunk_size", 600))
    overlap = int(retrieval.get("chunk_overlap", 100))

    docs, errors = load_documents(corpus_dir, limit=args.limit)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, max_chars=max_chars, overlap=overlap))

    stats = collect_stats(docs, chunks, errors)
    stats["config"] = {
        "corpus_dir": str(corpus_dir),
        "chunk_size": max_chars,
        "chunk_overlap": overlap,
        "limit": args.limit,
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg["paths"]["results_dir"]) / "corpus_stats.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(stats)
    print(f"结果已写入：{out_path}")


if __name__ == "__main__":
    main()
