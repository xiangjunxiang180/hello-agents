import os
import uuid
from typing import List, Dict, Any, Optional

from hello_agents.memory.rag.document import (
    convert_to_markdown, chunk_document
)


class RAGPipeline:
    """
    RAG管道：五层七步架构
    
    数据流：
    文档 → MarkItDown → Markdown → 分块 → 向量化 → Qdrant存储
                                                        ↓
    用户问题 → (MQE扩展) → (HyDE假设文档) → 向量检索 → 上下文构建 → LLM回答
    """

    def __init__(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default"
    ):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "")
        self.qdrant_api_key = (
            qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
        )
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace

        self._init_storage()

    def _init_storage(self):
        """初始化向量存储"""
        from hello_agents.memory.storage.qdrant_store import (
            QdrantVectorStore
        )
        from hello_agents.memory.embedding import get_dimension

        self.vector_store = QdrantVectorStore(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            vector_size=get_dimension(384)
        )
        print(
            f"[RAG] 初始化完成: namespace={self.rag_namespace}, "
            f"collection={self.collection_name}"
        )

    # ─────────────────────────────────────────
    # 数据准备：加载文档
    # ─────────────────────────────────────────

    def add_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        document_id: str = None
    ) -> Dict[str, Any]:
        """
        加载文档到知识库
        
        完整流程：
        MarkItDown转换 → 智能分块 → 批量向量化 → 存入Qdrant
        """
        doc_id = document_id or str(uuid.uuid4())

        # Step1: 转换为Markdown
        print(f"[RAG] 开始处理文档: {file_path}")
        markdown_text = convert_to_markdown(file_path)
        if not markdown_text:
            return {"success": False, "error": "文档转换失败"}

        # Step2: 智能分块
        chunks = chunk_document(
            markdown_text,
            chunk_tokens=chunk_size,
            overlap_tokens=chunk_overlap
        )
        if not chunks:
            return {"success": False, "error": "分块结果为空"}

        # Step3: 向量化并存储
        self._index_chunks(chunks, doc_id, file_path)

        return {
            "success": True,
            "document_id": doc_id,
            "chunks": len(chunks),
            "file": os.path.basename(file_path)
        }

    def add_text(
        self,
        text: str,
        document_id: str = None,
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """直接添加文本到知识库"""
        doc_id = document_id or str(uuid.uuid4())
        chunks = chunk_document(text)
        self._index_chunks(
            chunks, doc_id, source="text_input",
            extra_meta=metadata or {}
        )
        return {
            "success": True,
            "document_id": doc_id,
            "chunks": len(chunks)
        }

    def _index_chunks(
        self,
        chunks: List[Dict],
        doc_id: str,
        source: str = "",
        extra_meta: Dict = None,
        batch_size: int = 32
    ):
        """批量向量化并存入Qdrant"""
        from hello_agents.memory.embedding import get_text_embedder

        embedder = get_text_embedder()
        extra_meta = extra_meta or {}

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c["content"] for c in batch]

            try:
                vecs = embedder.encode(texts)
                # 统一为二维列表
                if vecs and not isinstance(vecs[0], list):
                    vecs = [
                        v.tolist() if hasattr(v, "tolist") else list(v)
                        for v in vecs
                    ]
            except Exception as e:
                print(f"[RAG] 向量化失败: {e}")
                continue

            ids = [c["id"] for c in batch]
            metas = [
                {
                    "memory_id": c["id"],
                    "content": c["content"],
                    "heading_path": c.get("heading_path", ""),
                    "document_id": doc_id,
                    "source": source,
                    "rag_namespace": self.rag_namespace,
                    "memory_type": "rag_chunk",
                    "is_rag_data": True,
                    "data_source": "rag_pipeline",
                    **extra_meta
                }
                for c in batch
            ]

            self.vector_store.add_vectors(
                vectors=vecs, metadata=metas, ids=ids
            )
            print(
                f"[RAG] 已索引 "
                f"{min(i + batch_size, len(chunks))}/{len(chunks)} 块"
            )

    # ─────────────────────────────────────────
    # 检索：基础 + 高级策略
    # ─────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_mqe: bool = False,
        enable_hyde: bool = False
    ) -> List[Dict]:
        """
        向量检索
        
        enable_mqe:  启用多查询扩展，提升召回率
        enable_hyde: 启用假设文档嵌入，改善检索精度
        """
        from hello_agents.memory.embedding import embed_query

        where = {
            "memory_type": "rag_chunk",
            "rag_namespace": self.rag_namespace
        }

        # 构建扩展查询列表
        queries = [query]

        if enable_mqe:
            expanded = self._mqe(query, n=2)
            queries.extend(expanded)
            print(f"[RAG] MQE扩展查询: {expanded}")

        if enable_hyde:
            hyde_doc = self._hyde(query)
            if hyde_doc:
                queries.append(hyde_doc)
                print(f"[RAG] HyDE假设文档已生成")

        # 对每个查询检索，合并去重取最高分
        agg: Dict[str, Dict] = {}
        for q in queries:
            qv = embed_query(q)
            hits = self.vector_store.search_similar(
                query_vector=qv,
                limit=limit * 3,
                score_threshold=min_score,
                where=where
            )
            for hit in hits:
                mid = hit["metadata"].get("memory_id", hit["id"])
                if (mid not in agg or
                        hit["score"] > agg[mid]["score"]):
                    agg[mid] = hit

        merged = sorted(
            agg.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        return merged[:limit]

    def ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = True,
        enable_mqe: bool = True,
        enable_hyde: bool = True
    ) -> str:
        """
        智能问答：检索 + LLM增强生成
        
        流程：
        1. 检索相关文档块
        2. 构建上下文
        3. 调用LLM基于上下文生成回答
        """
        # 检索
        hits = self.search(
            query=question,
            limit=limit,
            enable_mqe=enable_mqe and enable_advanced_search,
            enable_hyde=enable_hyde and enable_advanced_search
        )

        if not hits:
            return "❌ 知识库中没有找到相关内容，请先加载文档。"

        # 构建上下文
        context_parts = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            content = meta.get("content", "")
            heading = meta.get("heading_path", "")
            score = hit["score"]
            part = f"[片段{i}]"
            if heading:
                part += f"（来自：{heading}）"
            part += f"\n{content}"
            context_parts.append(part)

        context = "\n\n---\n\n".join(context_parts)

        # 调用LLM生成回答
        try:
            from hello_agents.core.llm import chat
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是知识库问答助手。请严格根据以下参考资料回答问题，"
                        "如果资料中没有相关信息，请明确说明。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"参考资料：\n{context}\n\n"
                        f"问题：{question}"
                    )
                }
            ]
            return chat(prompt)
        except Exception as e:
            # LLM不可用时直接返回检索结果

            return (
                f"检索到 {len(hits)} 条相关内容：\n\n"
                + "\n\n".join(
                    h["metadata"].get("content", "")[:200]
                    for h in hits
                )
            )

    # ─────────────────────────────────────────
    # 高级检索策略：MQE 和 HyDE
    # ─────────────────────────────────────────

    def _mqe(self, query: str, n: int = 2) -> List[str]:
        """
        多查询扩展（Multi-Query Expansion）
        
        核心思想：同一问题可以有多种表述，
        不同表述能匹配到不同相关文档，提升召回率30%-50%。
        用LLM生成n个语义等价但表述不同的查询。
        """
        try:
            from hello_agents.core.llm import chat
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "你是检索查询扩展助手。"
                        "生成语义等价或互补的多样化查询。"
                        "使用中文，简短，避免标点。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"原始查询：{query}\n"
                        f"请给出{n}个不同表述的查询，每行一个。"
                    )
                }
            ]
            text = chat(prompt)

            lines = [
                ln.strip("- \t")
                for ln in (text or "").splitlines()
            ]
            return [ln for ln in lines if ln][:n] or [query]
        except Exception:
            return [query]

    def _hyde(self, query: str) -> Optional[str]:
        """
        假设文档嵌入（Hypothetical Document Embeddings）
        
        核心思想：用问题的答案去找文档，比用问题本身更准确。
        原因：问题（疑问句）和答案（陈述句）在向量空间中分布不同，
        HyDE先生成假设答案，缩小这个语义鸿沟。
        """
        try:
            from hello_agents.core.llm import chat
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "根据用户问题，写一段可能的答案性段落，"
                        "用于向量检索（不要分析过程）。"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{query}\n"
                        "请直接写一段中等长度、客观、包含关键术语的段落。"
                    )
                }
            ]
            return chat(prompt)
        
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计"""
        count = self.vector_store.count()
        return {
            "namespace": self.rag_namespace,
            "collection": self.collection_name,
            "total_chunks": count
        }
