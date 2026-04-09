"""
Reflector 节点 —— 审查执行结果的完整性与准确性
"""

from __future__ import annotations

import json
import os

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


REFLECTOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个校园智能问答助手的审查器。\n"
        "你需要审查执行器完成的所有子步骤结果，判断：\n"
        "1. 是否所有子步骤都得到了充分回答\n"
        "2. 回答之间是否存在矛盾\n"
        "3. 证据链是否完整可靠\n\n"
        "请以 JSON 格式输出：\n"
        '{{"needs_revision": true/false, "reason": "原因说明", '
        '"revision_hints": ["需要修正的要点"]}}\n',
    ),
    (
        "human",
        "用户问题: {query}\n\n"
        "执行计划: {plan}\n\n"
        "各步骤结果:\n{steps_results}\n",
    ),
])


def reflector_node(state: dict) -> dict:
    """
    Reflector 节点逻辑：
    1. 汇总所有步骤结果
    2. LLM 判断是否需要修正
    3. 如需修正，重置 current_step 回退
    """
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    steps_text = "\n\n".join(
        f"步骤 {i+1}: {r['step']}\n结果: {r['result']}"
        for i, r in enumerate(state["steps_results"])
    )

    chain = REFLECTOR_PROMPT | llm
    result = chain.invoke({
        "query": state["query"],
        "plan": json.dumps(state["plan"], ensure_ascii=False),
        "steps_results": steps_text,
    })

    try:
        review = json.loads(result.content)
        needs_revision = review.get("needs_revision", False)
        reason = review.get("reason", "")
    except json.JSONDecodeError:
        needs_revision = False
        reason = ""

    update: dict = {
        "needs_revision": needs_revision,
        "short_term_memory": [f"Reflector 审查: {'需要修正 - ' + reason if needs_revision else '通过'}"],
    }

    if needs_revision:
        # 回退到第一步重新执行，同时清除图谱上下文避免旧数据污染
        update["current_step"] = 0
        update["steps_results"] = []
        update["graph_context"] = []

    return update
