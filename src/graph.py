"""
LangGraph 状态机核心 —— Plan-Execute-Report 工作流

拓扑：
    START -> Planner -> Executor (循环执行子步骤 / DeepResearch)
                                    -> Reflector
                                        ├─ needs_revision=True  -> Executor
                                        └─ needs_revision=False -> Reporter -> END
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.agents.planner import planner_node
from src.agents.executor import executor_node
from src.agents.reflector import reflector_node
from src.agents.reporter import reporter_node


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """LangGraph 全局状态"""

    # 用户原始问题
    query: str

    # Planner 输出的子步骤列表
    plan: list[str]

    # 当前正在执行的步骤索引
    current_step: int

    # 每步执行结果 [{step, result, evidence_chain}, ...]
    steps_results: list[dict]

    # 证据链 [{content, source_file, page, relevance_score, graph_path?}, ...]
    sources: list[dict]

    # 短期记忆：当前会话上下文摘要
    short_term_memory: Annotated[list[str], add_messages]

    # 长期记忆：跨会话持久化知识
    long_term_memory: dict

    # Reporter 生成的最终回答
    response: str

    # Reflector 是否要求回退重做
    needs_revision: bool

    # ── GraphRAG 扩展字段 ──────────────────────────────────

    # Planner 判断是否需要图检索（关系类问题时为 True）
    use_graph: bool

    # Executor 收集的图路径上下文 [{content, graph_path, source_file, ...}, ...]
    graph_context: list[dict]

    # Planner/Executor 识别出的关键实体名称列表（供后续节点参考）
    kg_entities: list[str]


# ---------------------------------------------------------------------------
# 条件路由
# ---------------------------------------------------------------------------

def should_continue_execution(state: AgentState) -> str:
    """判断 Executor 是否还有未完成的步骤"""
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "reflector"


def should_revise(state: AgentState) -> str:
    """Reflector 判断是否需要修正"""
    if state.get("needs_revision", False):
        return "executor"
    return "reporter"


# ---------------------------------------------------------------------------
# 构建图
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """构建并返回编译后的 LangGraph 工作流"""

    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reflector", reflector_node)
    workflow.add_node("reporter", reporter_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")

    workflow.add_conditional_edges(
        "executor",
        should_continue_execution,
        {"executor": "executor", "reflector": "reflector"},
    )

    workflow.add_conditional_edges(
        "reflector",
        should_revise,
        {"executor": "executor", "reporter": "reporter"},
    )

    workflow.add_edge("reporter", END)

    return workflow.compile()
