"""
Executor 节点 —— 逐步执行子任务，包含 DeepResearch 探索链
适配三路检索接口（Dense + BM25 + Graph），整合结构化图谱上下文
"""

from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils.retriever import retrieve_documents


EXECUTOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的执行器。\n"
        "你需要根据给定的子任务和检索到的相关文档来回答问题。\n"
        "请基于证据进行推理，如果文档不足以回答，请明确说明。\n\n"
        "当前上下文（短期记忆）：\n{short_term_memory}\n\n"
        "检索到的文档片段：\n{retrieved_docs}\n\n"
        "知识图谱路径（如有）：\n{graph_paths}\n",
    ),
    ("human", "子任务: {step}\n\n请给出回答，并标注你引用了哪些文档。"),
])

MAX_DEEP_RESEARCH_ROUNDS = 3


def _format_graph_paths(docs: list[dict]) -> str:
    """将图谱文档格式化为可读的路径摘要"""
    graph_docs = [d for d in docs if d.get("graph_path")]
    if not graph_docs:
        return "无"
    return "\n".join(
        f"- {d['graph_path']} (来源: {d['source_file']})"
        for d in graph_docs[:8]
    )


def _deep_research(
    step: str,
    initial_docs: list[dict],
    llm: ChatOllama,
    use_graph: bool = False,
) -> dict:
    """
    DeepResearch 探索链：
    1. 第一轮用原始 step 检索
    2. 根据结果判断是否需要追加检索（证据不充分时生成新 query）
    3. 最多 MAX_DEEP_RESEARCH_ROUNDS 轮，合并所有证据
    图谱文档与普通文档分别传入 Prompt，使 LLM 能感知结构化关系信息。
    """
    all_docs = list(initial_docs)
    evidence_chain = []
    response = None

    for round_idx in range(MAX_DEEP_RESEARCH_ROUNDS):
        # 分离图谱文档与普通文档
        text_docs = [d for d in all_docs if not d.get("graph_path")]
        graph_docs = [d for d in all_docs if d.get("graph_path")]

        docs_text = "\n---\n".join(
            f"[{d['source_file']}] {d['content']}" for d in text_docs
        ) or "无相关文档"

        graph_paths_text = _format_graph_paths(graph_docs)

        result = EXECUTOR_PROMPT.format_messages(
            short_term_memory="",
            retrieved_docs=docs_text,
            graph_paths=graph_paths_text,
            step=step,
        )
        response = llm.invoke(result)

        evidence_chain.append({
            "round": round_idx + 1,
            "answer_snippet": response.content[:200],
            "num_docs": len(text_docs),
            "num_graph_paths": len(graph_docs),
        })

        if "证据不足" not in response.content and "无法确定" not in response.content:
            break

        followup_docs = retrieve_documents(
            f"补充: {step} - {response.content[:100]}",
            top_k=3,
            use_graph=use_graph,
        )
        all_docs.extend(followup_docs)

    return {
        "result": response.content if response else "",
        "sources": all_docs,
        "evidence_chain": evidence_chain,
    }


def executor_node(state: dict) -> dict:
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    plan = state["plan"]
    idx = state["current_step"]
    step = plan[idx]
    use_graph = state.get("use_graph", False)

    docs = retrieve_documents(step, top_k=5, use_graph=use_graph)
    research = _deep_research(step, docs, llm, use_graph=use_graph)

    step_result = {
        "step": step,
        "result": research["result"],
        "evidence_chain": research["evidence_chain"],
    }

    new_sources = [
        {
            "content": d["content"],
            "source_file": d["source_file"],
            "page": d.get("page"),
            "relevance_score": d.get("relevance_score", 0.0),
            "graph_path": d.get("graph_path"),
        }
        for d in research["sources"]
    ]

    # 单独收集本步骤新增的图谱上下文
    new_graph_context = [d for d in new_sources if d.get("graph_path")]

    return {
        "current_step": idx + 1,
        "steps_results": state["steps_results"] + [step_result],
        "sources": state["sources"] + new_sources,
        "short_term_memory": [f"步骤 {idx + 1} 完成: {step}"],
        "graph_context": state.get("graph_context", []) + new_graph_context,
    }
