"""导出切块结果，供 BM25 / 向量索引使用。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.ingest.chunker import chunk_document
from src.ingest.loader import load_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建统一 chunk 文件")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 份")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    corpus_dir = Path(cfg["paths"]["corpus_dir"])
    index_dir = Path(cfg["paths"]["index_dir"])
    retrieval = cfg.get("retrieval", {})
    max_chars = int(retrieval.get("chunk_size", 600))
    overlap = int(retrieval.get("chunk_overlap", 100))

    docs, errors = load_documents(corpus_dir, limit=args.limit)
    chunks_path = index_dir / "chunks.jsonl"
    index_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with chunks_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            for chunk in chunk_document(doc, max_chars=max_chars, overlap=overlap):
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
                count += 1

    manifest = {
        "documents": len(docs),
        "chunks": count,
        "errors": errors,
        "chunk_size": max_chars,
        "chunk_overlap": overlap,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"文档：{len(docs)} 份，切块：{count:,} 个")
    if errors:
        print(f"读取失败：{len(errors)} 份")
    print(f"已写入：{chunks_path}")


if __name__ == "__main__":
    main()
