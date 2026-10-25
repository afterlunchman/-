"""汇总评测结果，生成报告用的 Markdown 表格与错误案例。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.config import load_config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    lines: list[str] = ["# 结果汇总", ""]

    corpus = _load_json(results_dir / "corpus_stats.json")
    if corpus:
        d = corpus["documents"]
        c = corpus["chunks"]
        lines += [
            "## 语料",
            "",
            f"- 文档：{d['total']} 份，正文 {d['characters_total']:,} 字",
            f"- 切块：{c['total']:,} 个（条文 {c['clause']:,} / 段落 {c['paragraph']:,}）",
            f"- 条文覆盖：{d['with_clause']} 份有条号，{d['without_clause']} 份回退段落",
            "",
        ]

    eval_path = Path(cfg["paths"]["eval_set"])
    if eval_path.exists():
        rows = [json.loads(line) for line in eval_path.open("r", encoding="utf-8") if line.strip()]
        answerable = sum(1 for row in rows if row["answerable"])
        lines += [
            "## 评测集",
            "",
            f"- 题目：{len(rows)} 道（可答 {answerable} / 不可答 {len(rows) - answerable}）",
            "- 来源：`data/eval/questions.jsonl`，`annotator=ai-draft`，待人工复核",
            "",
        ]

    retrieval = _load_json(results_dir / "retrieval_metrics.json")
    if retrieval:
        lines += ["## 检索层（70 道可答题）", ""]
        lines.append("| 检索器 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 延迟 ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name, metrics in retrieval.get("results", {}).items():
            lines.append(
                f"| {name} | {_fmt(metrics.get('recall@1'))} | {_fmt(metrics.get('recall@3'))} | "
                f"{_fmt(metrics.get('recall@5'))} | {_fmt(metrics.get('recall@10'))} | "
                f"{_fmt(metrics.get('mrr@10'))} | {_fmt(metrics.get('avg_latency_ms'))} |"
            )
        lines.append("")

    word2vec = _load_json(results_dir / "word2vec_analysis.json")
    if word2vec:
        lines += [
            "## Word2Vec 分析",
            "",
            f"- OOV 率：{word2vec.get('oov_rate')}",
            f"- 「可以留宿」与「不可以留宿」句向量余弦：{word2vec.get('negation_cosine')}",
            "- 两个查询的 Top-5 结果相同，说明词向量平均无法区分否定。",
            "",
        ]

    answer = _load_json(results_dir / "answer_metrics.json")
    if answer:
        lines += ["## 答案层", ""]
        lines.append(
            "| 方案 | 题数 | 正确率 | 引用支持率 | 规则引用通过率 | 误拒率 | 不可答拒答率 | Hit@5 | 延迟 s |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name, metrics in answer.get("results", {}).items():
            lines.append(
                f"| {name} | {metrics.get('answerable', 0)}+{metrics.get('unanswerable', 0)} | "
                f"{_fmt(metrics.get('correct_rate'))} | {_fmt(metrics.get('citation_support_rate'))} | "
                f"{_fmt(metrics.get('citation_rule_pass_rate'))} | {_fmt(metrics.get('over_refusal_rate'))} | "
                f"{_fmt(metrics.get('refusal_rate_unanswerable'))} | {_fmt(metrics.get('hit@5'))} | "
                f"{_fmt(metrics.get('avg_latency_s'))} |"
            )
        lines.append("")

    records = _load_answers(results_dir)
    if records:
        lines += _error_section(records)
        lines += _cost_section(records)

    out_path = results_dir / "summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入：{out_path}")
    print("\n".join(lines[:40]))


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_answers(results_dir: Path) -> list[dict]:
    records = []
    for path in sorted(results_dir.glob("answers_*.jsonl")):
        for line in path.open("r", encoding="utf-8"):
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _error_section(records: list[dict]) -> list[str]:
    lines = ["## 错误案例", ""]
    groups = {
        "可答题被误拒": [
            r for r in records if not r.get("error") and r["answerable"] and r["refused"]
        ],
        "不可答题未拒答": [
            r
            for r in records
            if not r.get("error") and not r["answerable"] and not r["refused"]
        ],
        "引用校验失败": [
            r
            for r in records
            if not r.get("error")
            and r["pipeline"] != "b0"
            and r["answerable"]
            and not r["refused"]
            and not r["citation_ok"]
        ],
        "检索 Top-5 未命中": [
            r
            for r in records
            if not r.get("error")
            and r["pipeline"] != "b0"
            and r["answerable"]
            and r.get("hit@5") is False
        ],
    }
    for title, items in groups.items():
        lines += [f"### {title}（{len(items)}）", ""]
        for record in items[:5]:
            answer = record["answer"].replace("\n", " ")[:120]
            lines.append(
                f"- `{record['pipeline']}` `{record['question_id']}` {record['question'][:60]}"
                f" → {answer}"
            )
        if not items:
            lines.append("- 无")
        lines.append("")
    return lines


def _cost_section(records: list[dict]) -> list[str]:
    lines = ["## 成本与延迟", ""]
    by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_pipeline[record["pipeline"]].append(record)

    total_prompt = 0
    total_completion = 0
    for name, items in sorted(by_pipeline.items()):
        prompt = sum(int((r.get("usage") or {}).get("prompt_tokens", 0)) for r in items)
        completion = sum(int((r.get("usage") or {}).get("completion_tokens", 0)) for r in items)
        total_prompt += prompt
        total_completion += completion
        latency = sum(r.get("latency_s", 0) for r in items) / max(len(items), 1)
        lines.append(
            f"- `{name}`：{len(items)} 次调用，prompt {prompt:,} tokens，"
            f"completion {completion:,} tokens，平均延迟 {latency:.2f}s"
        )
    lines.append(
        f"- 合计：prompt {total_prompt:,} tokens，completion {total_completion:,} tokens。"
        "网关未提供单价，最终费用按实际账单填写；报告中不得编造单价。"
    )
    lines.append("")
    return lines


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
