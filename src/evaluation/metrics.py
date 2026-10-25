"""检索与答案评测指标。"""

from __future__ import annotations

from collections.abc import Iterable


def recall_at_k(
    retrieved_ids: list[str], gold_ids: Iterable[str], k: int
) -> float | None:
    gold = set(gold_ids)
    if not gold:
        return None
    return 1.0 if gold & set(retrieved_ids[:k]) else 0.0


def mrr_at_k(
    retrieved_ids: list[str], gold_ids: Iterable[str], k: int
) -> float | None:
    gold = set(gold_ids)
    if not gold:
        return None
    for rank, item in enumerate(retrieved_ids[:k], start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)
