"""条号数字解析与归一化。"""

from __future__ import annotations

import re

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_CLAUSE_NUM_RE = re.compile(r"第\s*([一二三四五六七八九十百千零〇两\d]+)\s*条")


def clause_to_int(raw: str) -> int | None:
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    total = 0
    section = 0
    number = 0
    for char in raw:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            section += (number or 1) * _CN_UNITS[char]
            number = 0
        else:
            return None
    return section + number


def clause_key(raw: str | None) -> str | int:
    """把「第三十条」「第30条」归一到同一个 key。"""
    if not raw:
        return ""
    match = _CLAUSE_NUM_RE.search(raw)
    numeral = match.group(1) if match else raw
    number = clause_to_int(numeral)
    if number is not None:
        return number
    return re.sub(r"\s+", "", raw)
