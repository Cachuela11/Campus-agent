"""
文件加载与语义化切分 + 知识图谱构建 —— 支持 PDF / Markdown
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 默认切分参数（可通过 .env 覆盖）
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


def _get_splitter() -> RecursiveCharacterTextSplitter:
    chunk_size = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )


def load_pdf(file_path: str | Path) -> list[dict]:
    """加载 PDF 文件，返回切分后的文档片段列表"""
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    splitter = _get_splitter()
    chunks = splitter.split_documents(pages)
    return [
        {
            "content": chunk.page_content,
            "source_file": os.path.basename(str(file_path)),
            "page": chunk.metadata.get("page", 0),
        }
        for chunk in chunks
    ]


def load_markdown(file_path: str | Path) -> list[dict]:
    """加载 Markdown 文件，返回切分后的文档片段列表"""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    splitter = _get_splitter()
    chunks = splitter.split_text(text)
    return [
        {
            "content": chunk,
            "source_file": path.name,
            "page": None,
        }
        for chunk in chunks
    ]


def load_directory(dir_path: str | Path) -> list[dict]:
    """遍历目录，加载所有 PDF 和 Markdown 文件"""
    dir_path = Path(dir_path)
    all_docs: list[dict] = []

    for file in dir_path.rglob("*"):
        if file.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(file))
        elif file.suffix.lower() in (".md", ".markdown"):
            all_docs.extend(load_markdown(file))

    print(f"[Loader] 共加载 {len(all_docs)} 个文档片段")
    return all_docs


# ── 知识图谱构建 ───────────────────────────────────────────

# 批量 Prompt：一次处理 N 个文本块，减少 LLM 调用次数
_BATCH_TRIPLE_PROMPT = """\
以下是 {n} 个文本块，请分别为每个块提取实体关系三元组。

{chunks_section}

要求：
- 严格按"[块 X]"标题逐块输出，不得合并或跳过任何块
- 每块最多5个三元组，格式：实体1 | 关系 | 实体2
- 实体为名词短语（专业、课程、人名、组织、概念等），关系为动词短语
- 仅提取文本中明确存在的关系，不要推断
- 某块无明确实体关系时，该块下只写"无"

按以下格式输出：
[块 1]
三元组或"无"
[块 2]
三元组或"无"
"""

# 单块 Prompt：仅用于批量调用失败时的降级回退
_SINGLE_TRIPLE_PROMPT = """\
从以下文本中提取实体关系三元组，每行输出一个三元组，格式为：
实体1 | 关系 | 实体2

要求：
- 仅提取文本中明确存在的关系，不要推断
- 实体应是名词或名词短语（人名、组织、专业、课程、概念等）
- 关系应是动词或动词短语
- 每行严格按照"实体1 | 关系 | 实体2"格式输出
- 最多提取5个最重要的三元组
- 如果文本中没有明确的实体关系，输出"无"

文本：
{text}

三元组（每行一个）："""

# [块 X] 标题的正则，兼容中文空格和数字变体
_BLOCK_HEADER_RE = re.compile(r"\[块\s*(\d+)\]")


def _parse_batch_output(
    raw: str, n: int
) -> list[list[tuple[str, str, str]]]:
    """
    解析批量三元组输出，按 [块 X] 标题切割后逐行提取三元组。
    返回长度为 n 的列表，每个元素是对应块的三元组列表（可为空）。
    """
    results: list[list[tuple[str, str, str]]] = [[] for _ in range(n)]

    # split 会将捕获组插入结果：['前缀', '1', '内容1', '2', '内容2', ...]
    parts = _BLOCK_HEADER_RE.split(raw)

    i = 1  # 跳过 split 前的前缀
    while i + 1 < len(parts):
        try:
            idx = int(parts[i]) - 1  # 转为 0-indexed
            content = parts[i + 1]
            if 0 <= idx < n:
                triples: list[tuple[str, str, str]] = []
                for line in content.strip().splitlines():
                    line = line.strip()
                    if not line or line == "无" or "|" not in line:
                        continue
                    fields = [f.strip() for f in line.split("|")]
                    if len(fields) == 3 and all(fields):
                        triples.append((fields[0], fields[1], fields[2]))
                results[idx] = triples
        except (ValueError, IndexError):
            pass
        i += 2

    return results


def _extract_triples_from_chunk(
    content: str, llm
) -> list[tuple[str, str, str]]:
    """单块降级接口：仅在批量调用整体失败时使用"""
    prompt = _SINGLE_TRIPLE_PROMPT.format(text=content[:800])
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"[Loader] 单块三元组提取失败: {e}")
        return []

    triples = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line == "无" or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _extract_triples_batch(
    docs_batch: list[dict], llm
) -> list[list[tuple[str, str, str]]]:
    """
    批量调用 LLM，一次处理 kg_batch_size 个文本块。
    若 LLM 输出解析失败（全空），自动降级为逐块单独调用。
    """
    n = len(docs_batch)
    chunks_section = "\n\n".join(
        f"[块 {i + 1}]\n{doc['content'][:600]}"
        for i, doc in enumerate(docs_batch)
    )
    prompt = _BATCH_TRIPLE_PROMPT.format(n=n, chunks_section=chunks_section)

    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        results = _parse_batch_output(raw, n)
        # 只要解析到至少一个块的结果，就认为批量调用成功
        if any(results):
            return results
        print("[Loader] 批量输出解析为空，降级为逐块模式")
    except Exception as e:
        print(f"[Loader] 批量三元组提取失败，降级为逐块模式: {e}")

    # 降级：逐块单独调用
    return [_extract_triples_from_chunk(doc["content"], llm) for doc in docs_batch]


def build_knowledge_graph(
    docs: list[dict],
    kg_path: str | None = None,
    kg_batch_size: int = 5,
) -> "networkx.DiGraph":
    """
    从文档片段中提取三元组并构建有向知识图谱，持久化至 kg_path。
    每条边携带 source_file / page / chunk_content 属性。

    kg_batch_size=5：每次 LLM 调用处理 5 个块，比逐块调用快 ~5x。
    选择 5 的依据：qwen2.5:7b 在 ~2100 tokens 输入下结构化输出最稳定；
    超过 8 块时模型易出现块乱序或漏块，收益递减。
    """
    import networkx as nx
    from langchain_ollama import ChatOllama

    kg_path = kg_path or os.getenv("KG_INDEX_PATH", "./index/kg_graph.pkl")

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0,
    )

    G = nx.DiGraph()
    total_triples = 0
    n_batches = (len(docs) + kg_batch_size - 1) // kg_batch_size

    print(
        f"[Loader] 开始知识图谱构建: {len(docs)} 个片段，"
        f"批大小 {kg_batch_size}，共 {n_batches} 批..."
    )

    for batch_idx in range(n_batches):
        batch = docs[batch_idx * kg_batch_size : (batch_idx + 1) * kg_batch_size]
        batch_triples = _extract_triples_batch(batch, llm)

        for doc, triples in zip(batch, batch_triples):
            for head, rel, tail in triples:
                if G.has_edge(head, tail):
                    existing = G[head][tail].get("relation", "")
                    if rel not in existing:
                        G[head][tail]["relation"] = f"{existing}; {rel}"
                else:
                    G.add_edge(
                        head,
                        tail,
                        relation=rel,
                        source_file=doc["source_file"],
                        page=doc.get("page", 0),
                        chunk_content=doc["content"][:300],
                    )
                total_triples += 1

        processed = min((batch_idx + 1) * kg_batch_size, len(docs))
        if (batch_idx + 1) % 10 == 0 or processed == len(docs):
            print(
                f"[Loader] 已处理 {processed}/{len(docs)} 片段 "
                f"({batch_idx + 1}/{n_batches} 批)，三元组数: {total_triples}"
            )

    Path(kg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(kg_path, "wb") as f:
        pickle.dump(G, f)

    print(
        f"[Loader] 知识图谱构建完成: {G.number_of_nodes()} 节点, "
        f"{G.number_of_edges()} 边，已保存至 {kg_path}"
    )
    return G


def load_knowledge_graph(kg_path: str | None = None) -> "networkx.DiGraph | None":
    """从磁盘加载知识图谱，不存在则返回 None"""
    kg_path = kg_path or os.getenv("KG_INDEX_PATH", "./index/kg_graph.pkl")
    if not Path(kg_path).exists():
        return None
    try:
        with open(kg_path, "rb") as f:
            G = pickle.load(f)
        print(
            f"[Loader] 知识图谱已加载: {G.number_of_nodes()} 节点, "
            f"{G.number_of_edges()} 边"
        )
        return G
    except Exception as e:
        print(f"[Loader] 知识图谱加载失败: {e}")
        return None
