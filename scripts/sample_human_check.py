"""从答案评测结果中抽取 24 题，生成人工抽检表。"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from src.config import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    records = _load_records(results_dir)
    if not records:
        print("没有找到 answers_*.jsonl，请先运行答案评测。")
        return

    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    rng = random.Random(42)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)
    sample = answerable[:20] + unanswerable[:4]

    out_path = results_dir / "human_check.csv"
    columns = [
        "question_id",
        "pipeline",
        "question",
        "gold_answer",
        "model_answer",
        "judge_correctness",
        "judge_citation_support",
        "judge_completeness",
        "human_correctness_1_5",
        "human_citation_support_1_5",
        "human_completeness_1_5",
        "comment",
    ]
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in sample:
            judge = record.get("judge") or {}
            writer.writerow(
                {
                    "question_id": record["question_id"],
                    "pipeline": record["pipeline"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "model_answer": record["answer"],
                    "judge_correctness": judge.get("correctness", ""),
                    "judge_citation_support": judge.get("citation_support", ""),
                    "judge_completeness": judge.get("completeness", ""),
                    "human_correctness_1_5": "",
                    "human_citation_support_1_5": "",
                    "human_completeness_1_5": "",
                    "comment": "",
                }
            )
    print(f"已写入 {out_path}：{len(sample)} 题（20 可答 + 4 不可答）")
    print("请 4 名组员各自评分后，计算与 judge 的一致率或 Kappa。")


def _load_records(results_dir: Path) -> list[dict]:
    fallback: list[dict] = []
    for name in ("b5", "b1", "b3", "b0"):
        path = results_dir / f"answers_{name}.jsonl"
        if not path.exists():
            continue
        records = []
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("answer") and not record.get("error"):
                    records.append(record)
        if not records:
            continue
        answerable = sum(1 for r in records if r["answerable"])
        unanswerable = sum(1 for r in records if not r["answerable"])
        if answerable >= 20 and unanswerable >= 4:
            return records
        fallback = fallback or records
    return fallback


if __name__ == "__main__":
    main()
