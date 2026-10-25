"""生成报告用图表：检索 Recall@k 与答案层指标。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    _plot_retrieval(results_dir)
    _plot_answers(results_dir)


def _plot_retrieval(results_dir: Path) -> None:
    path = results_dir / "retrieval_metrics.json"
    if not path.exists():
        print(f"跳过检索图：{path} 不存在")
        return
    data = json.loads(path.read_text(encoding="utf-8"))["results"]
    names = list(data)
    ks = [1, 3, 5, 10]
    x = np.arange(len(ks))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, name in enumerate(names):
        values = [data[name].get(f"recall@{k}", 0.0) or 0.0 for k in ks]
        offset = (index - (len(names) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Recall@{k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("检索层 Recall@k（70 道可答题）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = results_dir / "retrieval_recall.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已写入：{out}")


def _plot_answers(results_dir: Path) -> None:
    path = results_dir / "answer_metrics.json"
    if not path.exists():
        print(f"跳过答案图：{path} 不存在")
        return
    data = json.loads(path.read_text(encoding="utf-8"))["results"]
    pipelines = [name for name in ("b0", "b1", "b3", "b5") if name in data]
    labels = {"b0": "B0 闭卷", "b1": "B1 BM25", "b3": "B3 BGE", "b5": "B5 hybrid+rerank"}

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(pipelines))
    width = 0.26

    metrics = [
        ("correct_rate", "正确率"),
        ("over_refusal_rate", "误拒率"),
        ("refusal_rate_unanswerable", "不可答拒答率"),
    ]
    for index, (key, label) in enumerate(metrics):
        values = [data[name].get(key, 0.0) or 0.0 for name in pipelines]
        bars = left.bar(x + (index - 1) * width, values, width, label=label)
        left.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    left.set_xticks(x)
    left.set_xticklabels([labels[name] for name in pipelines], rotation=15)
    left.set_ylim(0, 1.05)
    left.set_ylabel("比例")
    left.set_title("答案层：正确率 / 误拒 / 拒答")
    left.grid(axis="y", alpha=0.3)
    left.legend(fontsize=8)

    rag = [name for name in pipelines if name != "b0"]
    rx = np.arange(len(rag))
    support = [data[name].get("citation_support_rate", 0.0) or 0.0 for name in rag]
    rule = [data[name].get("citation_rule_pass_rate", 0.0) or 0.0 for name in rag]
    bars1 = right.bar(rx - 0.18, support, 0.35, label="LLM 引用支持率")
    bars2 = right.bar(rx + 0.18, rule, 0.35, label="规则引用通过率")
    right.bar_label(bars1, fmt="%.2f", fontsize=7, padding=2)
    right.bar_label(bars2, fmt="%.2f", fontsize=7, padding=2)
    right.set_xticks(rx)
    right.set_xticklabels([labels[name] for name in rag], rotation=15)
    right.set_ylim(0, 1.05)
    right.set_ylabel("比例")
    right.set_title("引用质量（B0 无引用，不参与）")
    right.grid(axis="y", alpha=0.3)
    right.legend(fontsize=8)

    fig.tight_layout()
    out = results_dir / "answer_metrics.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"已写入：{out}")


if __name__ == "__main__":
    main()
