"""
混合检索与重排序 —— Dense (ChromaDB) + BM25 + Graph + RRF + Cross-Encoder
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import chromadb

import jieba
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.utils.loader import load_directory


# ── 模块级单例 ──────────────────────────────────────────────

_chroma_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None
_embeddings: HuggingFaceEmbeddings | None = None

_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None

_reranker: CrossEncoder | None = None

_kg_graph = None  # networkx.DiGraph 单例


# ── ChromaDB / Embedding 初始化 ────────────────��───────────

def _get_collection() -> chromadb.Collection:
    """获取或初始化 ChromaDB 集合（单例）"""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./index")
    _chroma_client = chromadb.PersistentClient(path=persist_dir)
    _collection = _chroma_client.get_or_create_collection(
        name="campus_docs",
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def _get_embeddings() -> HuggingFaceEmbeddings:
    """���取 HuggingFace 本地 Embedding 模型（单例，首次调用自动下载）"""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    _embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[Retriever] Embedding 模型已加载: {model_name}")
    return _embeddings


# ── Cross-Encoder Reranker ─────────────────────────────────

def _get_reranker() -> CrossEncoder:
    """获取 Cross-Encoder 重排序模型（单例，首次调用自动下载）"""
    global _reranker
    if _reranker is not None:
        return _reranker

    model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    _reranker = CrossEncoder(model_name)
    print(f"[Retriever] Reranker 模型已加载: {model_name}")
    return _reranker


# ── BM25 工具函数 ──────────────────────────────────────────

def _tokenize_chinese(text: str) -> list[str]:
    """使用 jieba 对中文文本分词，用于 BM25 索引与查询"""
    return list(jieba.cut(text))


def _build_bm25_index(texts: list[str], corpus_meta: list[dict]) -> BM25Okapi:
    """
    构建 BM25 索引并持久化到磁盘。
    """
    global _bm25_index, _bm25_corpus

    tokenized = [_tokenize_chinese(t) for t in texts]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_corpus = corpus_meta

    bm25_path = os.getenv("BM25_INDEX_PATH", "./index/bm25_index.pkl")
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
    with open(bm25_path, "wb") as f:
        pickle.dump({"bm25": _bm25_index, "corpus": _bm25_corpus}, f)

    print(f"[Retriever] BM25 索引已构建: {len(texts)} 个片段")
    return _bm25_index


def _load_bm25_index() -> BM25Okapi | None:
    """加载 BM25 索引。优先级: 内存单例 > 磁盘 pickle > 从 ChromaDB 重建。"""
    global _bm25_index, _bm25_corpus

    if _bm25_index is not None:
        return _bm25_index

    bm25_path = os.getenv("BM25_INDEX_PATH", "./index/bm25_index.pkl")

    if Path(bm25_path).exists():
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            _bm25_index = data["bm25"]
            _bm25_corpus = data["corpus"]
            print(f"[Retriever] BM25 索引已从磁盘加载: {len(_bm25_corpus)} 个片段")
            return _bm25_index
        except Exception as e:
            print(f"[Retriever] BM25 pickle 加载失败，将从 ChromaDB 重建: {e}")

    collection = _get_collection()
    if collection.count() == 0:
        return None

    all_data = collection.get(include=["documents", "metadatas"])
    texts = all_data["documents"]
    corpus_meta = [
        {
            "content": texts[i],
            "source_file": all_data["metadatas"][i].get("source_file", "unknown"),
            "page": all_data["metadatas"][i].get("page"),
        }
        for i in range(len(texts))
    ]

    return _build_bm25_index(texts, corpus_meta)


# ── 知识图谱检索 ──────────────────────────────────────────

def _get_knowledge_graph():
    """获取知识图谱单例（延迟加载）"""
    global _kg_graph
    if _kg_graph is not None:
        return _kg_graph
    from src.utils.loader import load_knowledge_graph
    _kg_graph = load_knowledge_graph()
    return _kg_graph


def _graph_search(query: str, top_k: int) -> list[dict]:
    """
    知识图谱路径检索：
    1. 在图节点中找与 query 匹配的实体（子串 + jieba 分词匹配）
    2. 展开 1-2 跳出入边，收集路径
    3. 将边数据转换为统一文档格式返回
    """
    G = _get_knowledge_graph()
    if G is None or G.number_of_nodes() == 0:
        return []

    query_lower = query.lower()
    query_tokens = {t for t in _tokenize_chinese(query) if len(t) > 1}

    # 步骤1: 实体匹配（子串优先，分词补充）
    matched_nodes: list[str] = []
    seen_nodes: set[str] = set()

    for node in G.nodes():
        node_str = str(node).lower()
        if query_lower in node_str or node_str in query_lower:
            if node not in seen_nodes:
                matched_nodes.append(node)
                seen_nodes.add(node)

    if not matched_nodes:
        for node in G.nodes():
            node_str = str(node).lower()
            if any(tok in node_str for tok in query_tokens):
                if node not in seen_nodes:
                    matched_nodes.append(node)
                    seen_nodes.add(node)

    if not matched_nodes:
        return []

    # 步骤2: 展开邻居（最多取5个匹配节点，各扩展1-2跳）
    collected_edges: list[tuple] = []
    seen_edges: set[tuple] = set()

    for node in matched_nodes[:5]:
        for u, v, data in list(G.out_edges(node, data=True)) + list(G.in_edges(node, data=True)):
            key = (u, v, data.get("relation", ""))
            if key not in seen_edges:
                seen_edges.add(key)
                collected_edges.append((u, v, data))

        # 2跳：沿出边方向扩展
        for _, neighbor in G.out_edges(node):
            for u, v, data in G.out_edges(neighbor, data=True):
                key = (u, v, data.get("relation", ""))
                if key not in seen_edges:
                    seen_edges.add(key)
                    collected_edges.append((u, v, data))

    # 步骤3: 转换为文档格式（与 Dense/BM25 输出结构一致）
    results = []
    for head, tail, data in collected_edges[: top_k * 2]:
        relation = data.get("relation", "相关")
        source_file = data.get("source_file", "unknown")
        page = data.get("page", 0)
        chunk_content = data.get("chunk_content", "")
        content = f"[图谱] {head} --{relation}--> {tail}\n上下文: {chunk_content}"
        results.append({
            "content": content,
            "source_file": source_file,
            "page": page,
            "relevance_score": 1.0,  # reranker 将覆盖此分数
            "graph_path": f"{head} --{relation}--> {tail}",
        })

    return results[:top_k]


# ── 双/三路检索 ────────────────────────────────────────────

def _dense_search(query: str, top_k: int) -> list[dict]:
    """稠密向量检索（ChromaDB cosine）"""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    embeddings = _get_embeddings()
    query_vector = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "content": results["documents"][0][i],
            "source_file": results["metadatas"][0][i].get("source_file", "unknown"),
            "page": results["metadatas"][0][i].get("page"),
            "relevance_score": 1 - results["distances"][0][i],
        })
    return docs


def _bm25_search(query: str, top_k: int) -> list[dict]:
    """BM25 稀疏检索（适合缩写/关键词精确匹配）"""
    bm25 = _load_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = _tokenize_chinese(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        doc = _bm25_corpus[idx]
        results.append({
            "content": doc["content"],
            "source_file": doc["source_file"],
            "page": doc["page"],
            "relevance_score": float(scores[idx]),
        })
    return results


# ── RRF 融合 ──────────────────────────────────────────────

def reciprocal_rank_fusion(
    results_lists: list[list[dict]], k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion: 合并多路检索结果（兼容 2 路或 3 路）。
    RRF(d) = Σ 1 / (k + rank(d))，以文档 content 去重。
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for results in results_lists:
        for rank, doc in enumerate(results):
            key = doc["content"]
            if key not in doc_map:
                doc_map[key] = doc
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k + rank + 1)

    sorted_keys = sorted(fused_scores, key=lambda c: fused_scores[c], reverse=True)

    return [
        {**doc_map[key], "relevance_score": fused_scores[key]}
        for key in sorted_keys
    ]


# ── Cross-Encoder 重排序 ──────────────────────────────────

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    使用 Cross-Encoder 对三路召回候选文档重排序，返回 top_k 个最相关文档。
    graph_path 字段在排序后保留，不参与 pair 构造。
    """
    if not candidates:
        return []

    reranker = _get_reranker()
    pairs = [(query, doc["content"]) for doc in candidates]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(candidates, scores), key=lambda x: x[1], reverse=True
    )

    return [
        {**doc, "relevance_score": float(score)}
        for doc, score in scored_docs[:top_k]
    ]


# ── 公开接口 ──────────────────────────────────────────────

def get_index_count() -> int:
    """返回当前索引中的文档片段数量"""
    return _get_collection().count()


def index_documents(data_dir: str = "./data", force: bool = False) -> int:
    """
    从 data_dir 加载文档并建立向量索引 + BM25 索引 + 知识图谱索引。
    如果索引已存在且 force=False，则跳过。返回索引中的文档片段数量。
    """
    collection = _get_collection()
    existing = collection.count()

    if existing > 0 and not force:
        print(f"[Retriever] 索引已存在 ({existing} 个片段)，跳过重建。使用 --reindex 强制重建。")
        return existing

    docs = load_directory(data_dir)
    if not docs:
        print("[Retriever] data 目录为空，跳过索引")
        return 0

    if force and existing > 0:
        _chroma_client.delete_collection("campus_docs")
        globals()["_collection"] = None
        collection = _get_collection()
        print("[Retriever] 已清空旧索引")

    embeddings = _get_embeddings()

    texts = [d["content"] for d in docs]
    metadatas = [
        {"source_file": d["source_file"], "page": d.get("page") or 0}
        for d in docs
    ]
    ids = [f"doc_{i}" for i in range(len(docs))]

    vectors = embeddings.embed_documents(texts)

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"[Retriever] 已索引 {len(docs)} 个文档片段")

    corpus_meta = [
        {"content": d["content"], "source_file": d["source_file"], "page": d.get("page")}
        for d in docs
    ]
    _build_bm25_index(texts, corpus_meta)

    # 构建知识图谱索引（仅首次或强制重建时执行）
    kg_path = os.getenv("KG_INDEX_PATH", "./index/kg_graph.pkl")
    if not Path(kg_path).exists() or force:
        from src.utils.loader import build_knowledge_graph
        build_knowledge_graph(docs, kg_path=kg_path)
        global _kg_graph
        _kg_graph = None  # 清除旧单例，下次检索时重新加载
    else:
        print("[Retriever] 知识图谱索引已存在，跳过重建。")

    return len(docs)


def retrieve_documents(
    query: str, top_k: int = 5, use_graph: bool = False
) -> list[dict]:
    """
    混合检索: Dense + BM25 [+ Graph] -> RRF 融合 -> Cross-Encoder 重排序。
    返回 top_k 个最相关文档片段。

    返回格式: [{"content", "source_file", "page", "relevance_score", "graph_path"?}, ...]
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    rrf_k = int(os.getenv("RRF_K", "60"))
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "20"))

    dense_results = _dense_search(query, top_k=reranker_top_k)
    bm25_results = _bm25_search(query, top_k=reranker_top_k)

    results_lists: list[list[dict]] = [dense_results, bm25_results]

    if use_graph:
        graph_results = _graph_search(query, top_k=reranker_top_k)
        if graph_results:
            results_lists.append(graph_results)
            print(f"[Retriever] 图路径召回: {len(graph_results)} 条")

    fused = reciprocal_rank_fusion(results_lists, k=rrf_k)
    candidates = fused[:reranker_top_k]
    reranked = _rerank(query, candidates, top_k=top_k)

    return reranked
