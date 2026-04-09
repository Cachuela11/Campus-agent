
# Campus Agent

基于 LangGraph 的校园智能问答系统，采用**硬编码拓扑工作流**架构（Harness Loop），在 Plan-Execute-Reflect-Report 固定拓扑中集成 **GraphRAG** 混合检索能力。

## 架构

### Harness Loop 工作流拓扑

本系统是一个**硬编码拓扑工作流智能体**，而非多智能体系统。所有节点的调度逻辑、执行顺序与条件路由均在编译期静态确定，由 LangGraph Harness 驱动循环执行。各节点是工作流中具有特定职责的处理单元，而非自主决策的独立 Agent。

```
                        ┌─────────────────────────────────────┐
                        │          Harness Loop               │
                        │                                     │
  User Query            │  ┌─────────┐     ┌──────────┐       │
──────────────────────▶ │  │ Planner │────▶│ Executor │◀─┐    │
                        │  │ 拆解问题 │     │ 检索+推理  │  │    │
                        │  │ 识别图检索│     │ DeepRs   │  │    │
                        │  └─────────┘     └──────────┘  │    │
                        │                       │        │    │
                        │              子步骤未完成时循环    │    │
                        │                       │        │    │
                        │                       ▼        │    │
                        │               ┌───────────┐    │    │
                        │               │ Reflector │────┘    │
                        │               │  质量审查  │ needs_   │
                        │               └───────────┘ revision │
                        │                       │             │
                        │                       ▼             │
                        │               ┌──────────┐          │
                        │               │ Reporter │          │
                        │               │ 汇总输出  │           │
                        │               └──────────┘          │
                        └─────────────────────────────────────┘
                                         │
                                         ▼
                                    Final Answer
```

**固定拓扑路由规则：**

| 当前节点 | 条件 | 下一节点 |
|---------|------|---------|
| Planner | 无条件 | Executor |
| Executor | `current_step < len(plan)` | Executor（继续循环） |
| Executor | `current_step == len(plan)` | Reflector |
| Reflector | `needs_revision == True` | Executor（回退重做） |
| Reflector | `needs_revision == False` | Reporter |
| Reporter | 无条件 | END |

### 节点职责

- **Planner** — 将用户问题拆解为 2-5 个子步骤；通过关键词检测与 LLM 判断识别关系类问题，设置 `use_graph` 标志以调度图检索路径
- **Executor** — 逐步执行子步骤；调用三路混合检索，执行 DeepResearch（最多 3 轮迭代补充检索）；将图谱路径结构化信息注入 Prompt
- **Reflector** — 审查所有步骤结果的完整性与一致性；判定失败时将 `current_step` 回退至 0，触发 Harness 重新循环
- **Reporter** — 汇总各步骤结果，生成结构化 Markdown 回答，写入长期记忆

---

## GraphRAG 检索架构

在传统 Dense + BM25 双路检索基础上引入知识图谱（KG）召回，形成三路混合检索。

```
                    ┌─ Dense Search ─────────────────────────────┐
                    │  ChromaDB HNSW cosine (bge-small-zh-v1.5)  │
                    │                                            │
Query ──────────────┼─ BM25 Search ───────────────────────────── ┼──▶ RRF 融合 ──▶ Cross-Encoder ──▶ Top-K
                    │  BM25Okapi + jieba 分词                    │       (三路)      bge-reranker-base
                    │                                            │
                    └─ Graph Search (use_graph=True 时启用) ─────┘
                       NetworkX DiGraph
                       实体子串匹配 → 1-2 跳邻居展开 → 边路径文档化
```

### 知识图谱构建流程

```
PDF/Markdown
     │
     ▼
文档切块 (chunk_size=512)
     │
     ▼  （仅首次建索引或 --reindex 时执行）
LLM 三元组抽取
"实体1 | 关系 | 实体2"
     │
     ▼
NetworkX DiGraph
每条边携带: relation / source_file / page / chunk_content
     │
     ▼
index/kg_graph.pkl  （持久化）
```

### 图检索触发条件

Planner 检测到以下模式时自动启用图路径召回：

- 关系类关键词：`关系 / 联系 / 影响 / 依赖 / 区别 / 比较 / 关联 / 之间 / 相互`
- 条件类关键词：`前提 / 条件 / 前置 / 才能 / 需要…才`
- LLM 语义判断（`use_graph` 字段，与关键词检测取并集）

### 组件总览

| 组件 | 实现 |
|------|------|
| LLM | Ollama + Qwen2.5:7b（本地） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Reranker | BAAI/bge-reranker-base（Cross-Encoder） |
| 向量数据库 | ChromaDB（HNSW cosine） |
| 稀疏检索 | BM25Okapi + jieba |
| 知识图谱 | NetworkX DiGraph + LLM 三元组抽取 |
| 工作流引擎 | LangGraph（硬编码拓扑） |
| 记忆系统 | 短期（会话内 messages）+ 长期（JSON 持久化） |

---
