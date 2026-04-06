# Multi-agent-campus-Q-A-assistant

基于 LangGraph 多智能体架构的校园智能问答助手，采用 **Plan-Execute-Reflect-Report** 工作流，支持混合检索与深度研究能力。

## 架构

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌───────────┐     ┌──────────┐
│ Planner │────▶│ Executor │────▶│ Reflector │────▶│ Reporter │
│ 问题拆解 │     │ 检索+推理 │◀─┐  │  质量审查  │     │ 汇总输出 │
└─────────┘     └──────────┘  │  └───────────┘     └──────────┘
                    │  ▲      │       │
                    ▼  │      │       │ needs_revision
                 循环执行      └───────┘
                 子步骤
```

- **Planner** — 将用户问题拆解为 2-5 个可独立检索的子步骤
- **Executor** — 对每个子步骤执行混合检索 + DeepResearch（最多 3 轮迭代补充检索）
- **Reflector** — 审查完整性与一致性，必要时回退重做
- **Reporter** — 汇总生成结构化 Markdown 回答，写入长期记忆

## 检索流程

```
Query → Dense Search (ChromaDB cosine) ──┐
                                         ├─ RRF 融合 → Cross-Encoder 重排序 → Top-K
Query → BM25 Search (jieba 分词) ────────┘
```

| 组件 | 模型/工具 |
|------|----------|
| LLM | Ollama + Qwen2.5:7b（本地） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-base |
| 向量数据库 | ChromaDB (HNSW cosine) |
| 稀疏检索 | BM25Okapi + jieba |
| 工作流引擎 | LangGraph |

## 快速开始

### 前置要求

- Python 3.10+
- [Ollama](https://ollama.com) 已安装并运行

```bash
# 拉取模型
ollama pull qwen2.5:7b
```

### 安装

```bash
git clone https://github.com/<your-username>/campus-agent.git
cd campus-agent
pip install -r requirements.txt
```

### 配置

复制并编辑 `.env`：

```env
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
RERANKER_MODEL=BAAI/bge-reranker-base
```

### 运行

```bash
# 将文档放入 data/ 目录（支持 PDF、Markdown）
python main.py

# 强制重建索引
python main.py --reindex
```

## 项目结构

```
campus-agent/
├── main.py                  # 入口：建索引 → 编译图 → 交互循环
├── src/
│   ├── graph.py             # LangGraph 状态机与条件路由
│   ├── agents/
│   │   ├── planner.py       # 问题拆解
│   │   ├── executor.py      # 子步骤执行 + DeepResearch
│   │   ├── reflector.py     # 质量审查与回退
│   │   └── reporter.py      # 汇总输出 + 长期记忆
│   └── utils/
│       ├── retriever.py     # Dense + BM25 + RRF + Reranker
│       ├── memory.py        # 长期/短期记忆管理
│       └── loader.py        # PDF/Markdown 解析与分块
├── data/                    # 原始文档目录
├── index/                   # ChromaDB + BM25 持久化索引
└── requirements.txt
```

