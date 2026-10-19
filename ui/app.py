"""Streamlit research workspace for the SC-RAG FastAPI service."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import streamlit as st


API_URL = os.getenv("SC_RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")


def post_json(path: str, payload: dict[str, object]) -> dict[str, object]:
    request = urllib.request.Request(
        f"{API_URL}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"无法连接 FastAPI ({error.reason})") from error


def render_citations(claims: list[dict[str, object]]) -> None:
    for index, claim in enumerate(claims, start=1):
        st.markdown(f"**Claim {index}**：{claim['text']}")
        for citation in claim.get("citations", []):
            label = f"{citation['alias']} · PDF p.{citation['page']} · {citation['span_id']}"
            with st.expander(label):
                st.caption(f"chunk: {citation['chunk_id']}")
                st.write(citation["excerpt"])


st.set_page_config(page_title="SC-RAG Research Workspace", page_icon="🔬", layout="wide")
st.title("🔬 SC-RAG Research Workspace")
st.caption(f"FastAPI: {API_URL}")

answer_tab, review_tab = st.tabs(["Grounded Q&A", "Literature Review"])

with answer_tab:
    st.subheader("Evidence-grounded stochastic-computing assistant")
    question = st.text_area("问题", placeholder="例如：比较 [Sobol] 和 [P2LSG] 的并行化方法", height=100)
    mode = st.selectbox("回答模式", ["auto", "fundamentals", "paper_lookup", "comparison"])
    if st.button("检索并回答", type="primary", disabled=not question.strip(), key="answer"):
        with st.spinner("正在检索证据并生成回答…"):
            try:
                result = post_json("/answer", {"question": question, "mode": mode})
            except RuntimeError as error:
                st.error(str(error))
            else:
                status = result["status"]
                (st.success if status in {"answered", "partial"} else st.warning)(
                    f"状态：{status} · 置信度：{result['confidence']:.2f}"
                )
                st.markdown(result["answer"])
                st.write(f"证据论文：{', '.join(result['evidence_aliases']) or '无'}")
                if result.get("missing_information"):
                    st.info("未覆盖信息：" + "；".join(result["missing_information"]))
                if result.get("claims"):
                    st.subheader("可审计 Claims")
                    render_citations(result["claims"])

with review_tab:
    st.subheader("Two-pass literature review")
    topic = st.text_input("综述主题", placeholder="低离散位流生成器的精度、硬件成本与并行化")
    aliases_text = st.text_input("论文别名（空格分隔）", placeholder="Sobol PSobol P2LSG FSM-LDSG")
    chunks_per_paper = st.slider("每篇论文证据块数", min_value=1, max_value=8, value=5)
    if st.button("生成文献综述", type="primary", disabled=not topic.strip() or not aliases_text.strip(), key="review"):
        with st.spinner("逐篇抽取事实，再进行跨论文综合…"):
            try:
                result = post_json(
                    "/literature-review",
                    {"topic": topic, "aliases": aliases_text.split(), "chunks_per_paper": chunks_per_paper},
                )
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success(f"已完成 {len(result['papers'])} 篇论文的抽取")
                for row in result["comparison"]:
                    with st.expander(row["dimension"]):
                        for cell in row["cells"]:
                            st.markdown(f"**{cell['alias']}**：{cell['summary']}")
                if result["ideas"]:
                    st.subheader("研究想法")
                    for idea in result["ideas"]:
                        st.markdown(f"**假设**：{idea['hypothesis']}")
                        st.caption(f"实验：{idea['proposed_experiment']} · 证伪条件：{idea['falsifier']}")
