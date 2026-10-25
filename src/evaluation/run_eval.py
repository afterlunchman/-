"""批量评测入口（骨架）。

下一步：接入 B0—B5 各条流水线，逐题运行并汇总指标。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查并统计评测集")
    parser.add_argument("--config", default=None, help="配置文件路径，默认 config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    path = Path(cfg["paths"]["eval_set"])
    if not path.exists():
        print(f"评测集尚未创建：{path}")
        print("下一步：按 需求分析.md 第九节构造 80 题（70 可答 + 10 不可答）。")
        return

    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    answerable = Counter(bool(row.get("answerable")) for row in rows)
    types = Counter(row.get("type", "unknown") for row in rows)
    print(f"评测集：{path}")
    print(f"题目：{len(rows)} 道")
    print(f"可答：{answerable.get(True, 0)} 道；不可答：{answerable.get(False, 0)} 道")
    print(f"题型：{dict(types)}")
    print("批量推理尚未接入：下一步实现 B0—B5 与指标汇总。")


if __name__ == "__main__":
    main()
