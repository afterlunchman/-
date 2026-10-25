"""中文分词与停用词。"""

from __future__ import annotations

import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

import jieba

_TOKEN_RE = re.compile(r"[一-鿿A-Za-z0-9]")

STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "在", "是", "为", "对", "按", "本", "该",
    "其", "有", "中", "等", "应", "由", "向", "从", "并", "被", "将", "所",
    "一个", "以及", "进行", "相关", "有关", "规定", "要求",
}


def tokenize(text: str) -> list[str]:
    words = (word.strip().lower() for word in jieba.lcut(text))
    return [
        word
        for word in words
        if word and word not in STOPWORDS and _TOKEN_RE.search(word)
    ]
