"""
校园智能问答助手 —— 入口文件
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def main() -> None:
    from src.graph import build_graph
    from src.utils.retriever import index_documents

    force_reindex = "--reindex" in sys.argv

    # 1. 建立 / 更新向量索引
    data_dir = os.getenv("DATA_DIR", "./data")
    print("=" * 50)
    print("校园智能问答助手 (Plan-Execute-Report)")
    print("=" * 50)

    count = index_documents(data_dir, force=force_reindex)
    if count == 0:
        print("提示: data/ 目录为空，请先放入 PDF 或 Markdown 文件。\n")

    # 2. 编译 LangGraph 工作流
    graph = build_graph()
    print("工作流已编译完成。\n")

    # 3. 交互式问答循环
    while True:
        try:
            query = input("请输入问题 (输入 q 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not query or query.lower() == "q":
            print("再见！")
            break

        print(f"\n正在处理: {query}\n")

        # 执行工作流
        initial_state = {
            "query": query,
            "plan": [],
            "current_step": 0,
            "steps_results": [],
            "sources": [],
            "short_term_memory": [],
            "long_term_memory": {},
            "response": "",
            "needs_revision": False,
            # GraphRAG 扩展字段
            "use_graph": False,
            "graph_context": [],
            "kg_entities": [],
        }

        try:
            result = graph.invoke(initial_state)
            print("\n" + "=" * 50)
            print("回答:")
            print("=" * 50)
            print(result["response"])
            print("\n" + "-" * 50)
            print(f"引用来源: {len(result['sources'])} 个文档片段")
            graph_hits = len([s for s in result["sources"] if s.get("graph_path")])
            if graph_hits:
                print(f"  其中图谱路径: {graph_hits} 条")
            print(f"执行步骤: {len(result['steps_results'])} 步")
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"\n处理出错: {e}\n")


if __name__ == "__main__":
    main()
