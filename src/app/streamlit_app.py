"""Streamlit 演示界面。"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.generation.client import LLMClient
from src.generation.qa import answer_question
from src.retrieval.base import load_chunks
from src.retrieval.factory import build_retriever

st.set_page_config(
    page_title="校园安全制度问答助手",
    page_icon=None,
    layout="wide",
)

EXAMPLES = [
    "宿舍可以使用大功率电器吗？",
    "实验室发生事故后多久内要报告？",
    "可以让外校同学留宿吗？",
    "补考成绩怎么算？",
]


@st.cache_resource(show_spinner=False)
def load_resources(retriever_name: str, rerank: bool):
    cfg = load_config()
    chunks_path = Path(cfg["paths"]["index_dir"]) / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"未找到 {chunks_path}，请先运行 python -m scripts.build_index"
        )
    chunks = load_chunks(chunks_path)
    retriever = build_retriever(retriever_name, chunks, cfg, rerank=rerank)
    llm = LLMClient(cfg)
    return cfg, retriever, llm


with st.sidebar:
    st.header("检索设置")
    retriever_name = st.selectbox(
        "检索器",
        options=["hybrid", "bm25", "dense", "word2vec"],
        index=0,
        help="hybrid = BM25 + BGE 向量检索，RRF 融合",
    )
    rerank = False
    if retriever_name == "hybrid":
        rerank = st.checkbox("启用重排", value=False)
        if rerank:
            st.caption("CPU 上首次加载与推理较慢，演示时建议关闭。")
    top_k = st.slider("Top-k", min_value=3, max_value=20, value=10)
    show_debug = st.checkbox("显示检索详情", value=True)
    st.divider()
    st.caption(
        "语料：东北大学安全管理委员会 155 份制度。"
        "答案仅依据检索到的条文，证据不足时会拒答。"
    )

st.title("校园安全制度问答助手")
st.caption("方向 A · 领域 RAG 问答｜检索 + 强制引用 + 引用校验")

if "question" not in st.session_state:
    st.session_state["question"] = ""

cols = st.columns(len(EXAMPLES))
for index, (col, example) in enumerate(zip(cols, EXAMPLES)):
    if col.button(example, key=f"example_{index}", use_container_width=True):
        st.session_state["question"] = example
        st.rerun()

st.text_input(
    "请输入你的问题",
    key="question",
    placeholder="例如：宿舍里可以使用大功率电器吗？",
)
submitted = st.button("提问", type="primary")


def _render_result(result: dict, show_debug: bool) -> None:
    if result["refused"]:
        st.warning("现有制度未找到明确依据，建议咨询相关管理部门。")
    elif not result["citation_ok"]:
        st.warning("答案未通过引用校验，请核对原文后使用。")
    else:
        st.success("已找到制度依据。")

    st.markdown(result["answer"])

    citations = result.get("citations", {})
    details = citations.get("details", [])
    if details and result["refused"]:
        st.caption("拒答时检索到的低置信条款未作为依据展示。")
    if details and not result["refused"]:
        st.subheader("引用与原文")
        context_map = {chunk["chunk_id"]: chunk for chunk in result["contexts"]}
        for detail in details:
            chunk = context_map.get(detail.get("matched_chunk_id") or "")
            with st.container(border=True):
                left, right = st.columns([4, 1])
                with left:
                    st.markdown(f"**{detail['raw']}**")
                    status = {
                        "ok": "条号与制度名匹配",
                        "clause_not_found": "条号不在本次检索上下文",
                        "title_mismatch": "条号存在但制度名不匹配",
                    }.get(detail["status"], detail["status"])
                    st.caption(status)
                    if chunk:
                        st.caption(
                            f"{chunk['title']} · {chunk['clause_no'] or '无条号'}"
                            f" · 置信度 {chunk['confidence']:.2f}"
                        )
                        if chunk["confidence"] < 0.9:
                            st.warning("该条来自扫描件识别，建议核对 PDF 原件。")
                with right:
                    if chunk and chunk.get("source_url"):
                        st.link_button("查看原文", chunk["source_url"])
                    if chunk and chunk.get("attachment_url"):
                        st.link_button("PDF 附件", chunk["attachment_url"])

    usage = result.get("usage") or {}
    st.caption(
        f"延迟 {result['latency_s']:.2f}s · "
        f"prompt {usage.get('prompt_tokens', 0)} / "
        f"completion {usage.get('completion_tokens', 0)} tokens · "
        f"模型 {result.get('model') or 'unknown'}"
    )

    if show_debug and result.get("retrieved"):
        with st.expander("检索 Top-k 与分数", expanded=True):
            rows = [
                {
                    "排名": rank,
                    "分数": round(item["score"], 4),
                    "制度": item["title"],
                    "条号": item["clause_no"] or "无",
                    "页码": item["page"] if item["page"] is not None else "-",
                    "置信度": item["confidence"],
                    "chunk_id": item["chunk_id"],
                }
                for rank, item in enumerate(result["retrieved"], start=1)
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


if submitted:
    question = st.session_state["question"].strip()
    if not question:
        st.warning("请输入问题。")
    else:
        try:
            with st.spinner("加载检索器..."):
                cfg, retriever, llm = load_resources(retriever_name, rerank)
            with st.spinner("检索并生成答案..."):
                start = time.perf_counter()
                result = answer_question(question, retriever, llm, cfg, top_k=top_k)
                result["latency_s"] = round(time.perf_counter() - start, 3)
        except Exception as exc:  # noqa: BLE001 - 界面需要可读错误
            st.error(f"调用失败：{exc}")
        else:
            _render_result(result, show_debug)
