"""检索层评测：Recall@k / MRR@10。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.evaluation.metrics import mean, mrr_at_k, recall_at_k
from src.retrieval.base import load_chunks
from src.retrieval.factory import build_retriever

DEFAULT_RETRIEVERS = "bm25,word2vec,dense,hybrid,hybrid_rerank"
KS = (1, 3, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索层评测")
    parser.add_argument("--config", default=None)
    parser.add_argument("--retrievers", default=DEFAULT_RETRIEVERS)
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 道可答题")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    chunks_path = Path(cfg["paths"]["index_dir"]) / "chunks.jsonl"
    chunks = load_chunks(chunks_path)
    questions = _load_questions(Path(cfg["paths"]["eval_set"]), args.limit)
    print(f"可答题：{len(questions)} 道；chunk：{len(chunks):,} 个")

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for name in args.retrievers.split(","):
        name = name.strip()
        if not name:
            continue
        print(f"\n=== {name} ===")
        rerank = name == "hybrid_rerank"
        base_name = "hybrid" if rerank else name
        retriever = build_retriever(base_name, chunks, cfg, rerank=rerank)
        records, latencies = _evaluate(retriever, questions, args.top_k)
        metrics = _aggregate(records, latencies)
        summary[name] = metrics
        _write_jsonl(results_dir / f"retrieval_{name}.jsonl", records)
        print(_format_metrics(name, metrics))
        del retriever
        gc.collect()

    payload = {
        "eval_set": cfg["paths"]["eval_set"],
        "answerable": len(questions),
        "top_k": args.top_k,
        "results": summary,
    }
    (results_dir / "retrieval_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n汇总已写入：{results_dir / 'retrieval_metrics.json'}")


def _load_questions(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("answerable") and row.get("chunk_id"):
                rows.append(row)
    return rows[:limit] if limit else rows


def _evaluate(retriever, questions: list[dict], top_k: int):
    records = []
    latencies = []
    for index, row in enumerate(questions, start=1):
        start = time.perf_counter()
        results = retriever.search(row["question"], top_k=top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        retrieved_ids = [item.chunk.chunk_id for item in results]
        gold = {row["chunk_id"]}
        record = {
            "question_id": row["question_id"],
            "question": row["question"],
            "gold_chunk_id": row["chunk_id"],
            "gold_clause": (row.get("gold_clauses") or [""])[0],
            "retrieved": retrieved_ids,
            "scores": [round(item.score, 6) for item in results],
        }
        for k in KS:
            record[f"hit@{k}"] = recall_at_k(retrieved_ids, gold, k)
        record["rr@10"] = mrr_at_k(retrieved_ids, gold, 10)
        records.append(record)
        if index % 10 == 0 or index == len(questions):
            print(f"  {index}/{len(questions)}")
    return records, latencies


def _aggregate(records: list[dict], latencies: list[float]) -> dict:
    metrics: dict[str, float | None] = {}
    for k in KS:
        metrics[f"recall@{k}"] = mean(record[f"hit@{k}"] for record in records)
    metrics["mrr@10"] = mean(record["rr@10"] for record in records)
    metrics["avg_latency_ms"] = mean(latencies)
    metrics["queries"] = len(records)
    return metrics


def _format_metrics(name: str, metrics: dict) -> str:
    parts = [
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    ]
    return f"{name}: " + "  ".join(parts)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
