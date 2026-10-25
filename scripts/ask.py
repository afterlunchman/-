"""端到端问答入口：检索 -> LLM -> 引用校验。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from src.config import load_config
from src.generation.client import LLMClient
from src.generation.qa import answer_question
from src.retrieval.base import load_chunks
from src.retrieval.factory import build_retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校园安全制度问答")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.yaml")
    parser.add_argument("--question", required=True, help="用户问题")
    parser.add_argument(
        "--retriever",
        default="hybrid",
        choices=["bm25", "word2vec", "dense", "hybrid"],
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="重建向量索引")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    parser.add_argument("--show-context", action="store_true", help="打印检索到的原文")
    return parser.parse_args()


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
    retriever = build_retriever(
        args.retriever,
        chunks,
        cfg,
        rerank=not args.no_rerank,
        rebuild=args.rebuild,
    )
    llm = LLMClient(cfg)

    start = time.perf_counter()
    try:
        result = answer_question(
            args.question, retriever, llm, cfg, top_k=args.top_k
        )
    except Exception as exc:  # noqa: BLE001 - CLI 需要给出可读错误
        print(f"调用失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    result["latency_s"] = round(time.perf_counter() - start, 3)
    result["question"] = args.question
    result["retriever"] = args.retriever

    _write_log(cfg, result)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"问题：{args.question}")
    print(f"检索器：{args.retriever} | 延迟：{result['latency_s']:.2f}s")
    print("\n答案：")
    print(result["answer"])

    citations = result["citations"]
    print(
        f"\n引用校验：{citations['valid']}/{citations['total']} 条有效"
        f" | 通过：{result['citation_ok']}"
    )
    for detail in citations["details"]:
        print(
            f"  - {detail['raw']} -> {detail['status']}"
            + (f" ({detail['matched_chunk_id']})" if detail["matched_chunk_id"] else "")
        )

    if result["retrieved"]:
        print("\n检索 Top-k：")
        for rank, item in enumerate(result["retrieved"], start=1):
            print(f"  [{rank}] {item['score']:.4f} {item['chunk_id']}")

    usage = result.get("usage") or {}
    if usage:
        print(
            f"\nToken：prompt {usage.get('prompt_tokens', 0)}"
            f" / completion {usage.get('completion_tokens', 0)}"
        )

    if args.show_context:
        print("\n检索原文：")
        for index, chunk in enumerate(result["contexts"], start=1):
            print(f"--- 证据 {index}: {chunk['title']} {chunk['clause_no'] or ''} ---")
            print(chunk["text"])


def _write_log(cfg: dict, result: dict) -> None:
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "question": result["question"],
        "retriever": result["retriever"],
        "answer": result["answer"],
        "refused": result["refused"],
        "citation_ok": result["citation_ok"],
        "citations": result["citations"],
        "retrieved": result["retrieved"],
        "usage": result.get("usage", {}),
        "model": result.get("model"),
        "latency_s": result["latency_s"],
        "prompt_version": result.get("prompt_version"),
    }
    with (log_dir / "qa.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
