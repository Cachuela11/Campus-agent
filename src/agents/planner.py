"""
Planner 节点 —— 接收用户问题，拆解为可执行的子步骤
新增：识别复杂关系类问题，设置 use_graph 标志以调度图检索任务
"""

from __future__ import annotations

import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.utils.memory import load_long_term_memory


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的规划器。\n"
        "你的任务是将用户的问题拆解为 2-5 个可独立检索并回答的子步骤。\n"
        "每个子步骤应该是一个具体的信息检索或推理任务。\n\n"
        "历史知识（长期记忆）：\n{long_term_memory}\n\n"
        "请以 JSON 格式输出，包含两个字段：\n"
        '- "steps": 步骤列表（字符串数组，2-5 个元素）\n'
        '- "use_graph": 布尔值，当问题涉及实体间关系、依赖、影响、'
        "比较、关联、前置条件等复杂关系时为 true，否则为 false\n\n"
        '示例: {{"steps": ["步骤1", "步骤2"], "use_graph": false}}\n',
    ),
    ("human", "{query}"),
])

# 关系型问题关键词模式（LLM 判断的补充保障）
_RELATIONAL_PATTERN = re.compile(
    r"关系|联系|影响|依赖|区别|比较|对比|如何.*导致|什么.*导致|"
    r"前提|条件|前置|才能|需要.*才|怎么.*影响|哪些.*相关|"
    r"和.*有什么|与.*区别|跟.*联系|之间|相互|关联",
    re.IGNORECASE,
)


def _is_relational_query(query: str) -> bool:
    """关键词快速检测是否为关系类问题"""
    return bool(_RELATIONAL_PATTERN.search(query))


def planner_node(state: dict) -> dict:
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    long_mem = load_long_term_memory()
    long_mem_str = json.dumps(long_mem, ensure_ascii=False) if long_mem else "无"

    chain = PLANNER_PROMPT | llm
    result = chain.invoke({
        "query": state["query"],
        "long_term_memory": long_mem_str,
    })

    # 关键词快速检测（兜底，防止 LLM 漏判）
    keyword_use_graph = _is_relational_query(state["query"])

    try:
        parsed = json.loads(result.content)
        if isinstance(parsed, dict):
            plan = parsed.get("steps", [])
            use_graph = bool(parsed.get("use_graph", keyword_use_graph))
        elif isinstance(parsed, list):
            plan = parsed
            use_graph = keyword_use_graph
        else:
            plan = [result.content]
            use_graph = keyword_use_graph
    except json.JSONDecodeError:
        plan = [result.content]
        use_graph = keyword_use_graph

    if not isinstance(plan, list) or not plan:
        plan = [state["query"]]

    # use_graph 取 LLM 判断与关键词检测的并集
    use_graph = use_graph or keyword_use_graph

    if use_graph:
        print("[Planner] 检测到关系类问题，已启用图检索路径")

    return {
        "plan": plan,
        "current_step": 0,
        "steps_results": [],
        "sources": [],
        "short_term_memory": [f"用户问题: {state['query']}"],
        "needs_revision": False,
        "use_graph": use_graph,
        "graph_context": [],
        "kg_entities": [],
    }
