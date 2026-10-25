"""答案层评测：B0 闭卷、B1 BM25、B3 dense、B5 hybrid+rerank。"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from src.config import load_config
from src.evaluation.judge import judge_answer
from src.evaluation.metrics import mean
from src.generation.client import LLMClient
from src.generation.qa import _detect_refusal, answer_question
from src.retrieval.base import load_chunks
from src.retrieval.factory import build_retriever

PIPELINES = {
    "b0": None,
    "b1": "bm25",
    "b3": "dense",
    "b5": "hybrid_rerank",
}

CLOSED_BOOK_SYSTEM = """你是校园安全制度问答助手。只凭你已有的知识回答，不要检索。
如果不知道或没有把握，明确说「不知道」或「现有知识未找到明确依据」，不要编造条号、数值或处罚。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="答案层评测")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pipelines", default="b0,b1,b3,b5")
    parser.add_argument("--answerable", type=int, default=30, help="可答题抽样数")
    parser.add_argument("--unanswerable", type=int, default=10, help="不可答题数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-judge", action="store_true", help="只跑生成，不调用评审")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--metrics-out", default=None, help="指标输出路径")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cfg = load_config(args.config)
    questions = _sample_questions(
        Path(cfg["paths"]["eval_set"]), args.answerable, args.unanswerable, args.seed
    )
    print(
        f"评测题：{len(questions)} 道（可答 {sum(q['answerable'] for q in questions)}，"
        f"不可答 {sum(not q['answerable'] for q in questions)}）"
    )

    chunks = load_chunks(Path(cfg["paths"]["index_dir"]) / "chunks.jsonl")
    llm = LLMClient(
        cfg,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    judge = None if args.no_judge else _build_judge(cfg, args)

    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for pipeline in [name.strip() for name in args.pipelines.split(",") if name.strip()]:
        if pipeline not in PIPELINES:
            print(f"跳过未知 pipeline：{pipeline}")
            continue
        print(f"\n=== {pipeline} ===")
        retriever = None
        if PIPELINES[pipeline]:
            retriever = build_retriever(
                "hybrid" if PIPELINES[pipeline] == "hybrid_rerank" else PIPELINES[pipeline],
                chunks,
                cfg,
                rerank=PIPELINES[pipeline] == "hybrid_rerank",
            )
        out_path = results_dir / f"answers_{pipeline}.jsonl"
        records = _run_pipeline(
            pipeline, retriever, questions, llm, judge, cfg, out_path
        )
        summary[pipeline] = _aggregate(records)
        print(_format_metrics(pipeline, summary[pipeline]))
        del retriever
        gc.collect()

    payload = {"questions": len(questions), "results": summary}
    metrics_path = (
        Path(args.metrics_out)
        if args.metrics_out
        else results_dir / "answer_metrics.json"
    )
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n汇总已写入：{metrics_path}")


def _build_judge(cfg: dict, args: argparse.Namespace) -> LLMClient:
    judge_cfg = copy.deepcopy(cfg)
    judge_model = cfg.get("evaluation", {}).get("judge_model") or cfg["llm"]["model"]
    judge_cfg["llm"]["model"] = judge_model
    return LLMClient(
        judge_cfg,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )


def _sample_questions(
    path: Path, n_answerable: int, n_unanswerable: int, seed: int
) -> list[dict]:
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["answerable"]:
            by_type[row["type"]].append(row)
    rng = random.Random(seed)
    for items in by_type.values():
        rng.shuffle(items)

    sampled: list[dict] = []
    types = list(by_type)
    while len(sampled) < n_answerable and any(by_type.values()):
        for key in types:
            if by_type[key] and len(sampled) < n_answerable:
                sampled.append(by_type[key].pop())

    unanswerable = [row for row in rows if not row["answerable"]][:n_unanswerable]
    return sampled + unanswerable


def _run_pipeline(
    pipeline: str,
    retriever,
    questions: list[dict],
    llm: LLMClient,
    judge: LLMClient | None,
    cfg: dict,
    out_path: Path,
) -> list[dict]:
    done = _load_done(out_path)
    records = list(done.values())
    with out_path.open("a", encoding="utf-8") as fh:
        for index, row in enumerate(questions, start=1):
            if row["question_id"] in done:
                continue
            start = time.perf_counter()
            try:
                if pipeline == "b0":
                    result = _answer_closed_book(row["question"], llm)
                else:
                    result = answer_question(row["question"], retriever, llm, cfg)
                error = None
            except Exception as exc:  # noqa: BLE001 - 记录失败继续跑
                result = {"answer": "", "refused": False, "citation_ok": False,
                          "citations": {}, "retrieved": [], "contexts": [],
                          "usage": {}, "model": None}
                error = f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - start

            verdict = None
            if judge is not None and result["answer"]:
                try:
                    verdict = judge_answer(
                        judge,
                        row["question"],
                        row["gold_answer"],
                        result["answer"],
                        _contexts_text(result.get("contexts", [])),
                    )
                except Exception as exc:  # noqa: BLE001
                    verdict = {"error": f"{type(exc).__name__}: {exc}"}

            retrieved_ids = [item["chunk_id"] for item in result.get("retrieved", [])]
            record = {
                "question_id": row["question_id"],
                "pipeline": pipeline,
                "question": row["question"],
                "answerable": row["answerable"],
                "gold_chunk_id": row.get("chunk_id", ""),
                "gold_answer": row["gold_answer"],
                "answer": result["answer"],
                "refused": bool(result["refused"]),
                "citation_ok": bool(result["citation_ok"]),
                "citations": result.get("citations", {}),
                "retrieved_ids": retrieved_ids,
                "hit@5": (
                    row.get("chunk_id") in retrieved_ids[:5]
                    if row["answerable"]
                    else None
                ),
                "latency_s": round(latency, 3),
                "usage": result.get("usage", {}),
                "model": result.get("model"),
                "judge": verdict,
                "error": error,
            }
            records.append(record)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  {index}/{len(questions)} {row['question_id']} {latency:.1f}s")
    return records


def _answer_closed_book(question: str, llm: LLMClient) -> dict:
    response = llm.chat(
        [
            {"role": "system", "content": CLOSED_BOOK_SYSTEM},
            {"role": "user", "content": question},
        ]
    )
    return {
        "answer": response["content"],
        "refused": _detect_refusal(response["content"]),
        "citation_ok": False,
        "citations": {"total": 0, "valid": 0, "invalid": 0, "has_citation": False, "details": []},
        "retrieved": [],
        "contexts": [],
        "usage": response["usage"],
        "model": response["model"],
    }


def _contexts_text(contexts: list[dict]) -> str:
    blocks = []
    for chunk in contexts[:3]:
        blocks.append(
            f"{chunk.get('title', '')} {chunk.get('clause_no') or ''}：{chunk.get('text', '')[:300]}"
        )
    return "\n\n".join(blocks)


def _load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if line:
            record = json.loads(line)
            done[record["question_id"]] = record
    return done


def _aggregate(records: list[dict]) -> dict:
    errors = sum(1 for r in records if r.get("error"))
    valid = [r for r in records if not r.get("error")]
    answerable = [r for r in valid if r["answerable"]]
    unanswerable = [r for r in valid if not r["answerable"]]
    judged = [r for r in answerable if r.get("judge") and "correctness" in r["judge"]]

    return {
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "correct_rate": mean(
            1.0 if r["judge"]["correctness"] >= 4 else 0.0 for r in judged
        ),
        "citation_support_rate": mean(
            1.0 if r["judge"]["citation_support"] >= 4 else 0.0
            for r in judged
            if not r["refused"]
        ),
        "citation_rule_pass_rate": mean(
            1.0 if r["citation_ok"] else 0.0
            for r in answerable
            if not r["refused"]
        ),
        "over_refusal_rate": mean(1.0 if r["refused"] else 0.0 for r in answerable),
        "refusal_rate_unanswerable": mean(
            1.0 if r["refused"] else 0.0 for r in unanswerable
        ),
        "hit@5": mean(
            1.0 if r["hit@5"] else 0.0 for r in answerable if r["hit@5"] is not None
        ),
        "avg_latency_s": mean(r["latency_s"] for r in records),
        "errors": errors,
    }


def _format_metrics(name: str, metrics: dict) -> str:
    parts = [
        f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
        for key, value in metrics.items()
    ]
    return f"{name}: " + "  ".join(parts)


if __name__ == "__main__":
    main()
