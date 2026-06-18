import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from hello_agents.memory.base import BaseMemory, MemoryConfig, MemoryItem


class EpisodicMemory(BaseMemory):
    """
    情景记忆
    
    特点：
    - SQLite + Qdrant 混合存储
    - 支持按时间序列和会话检索
    - 结构化过滤 + 语义向量检索
    
    对应人类认知：自传体记忆，记录"什么时候发生了什么事"
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._init_storage()

    def _init_storage(self):
        """初始化SQLite和Qdrant存储"""
        from hello_agents.memory.storage.document_store import (
            SQLiteDocumentStore
        )
        from hello_agents.memory.storage.qdrant_store import (
            QdrantVectorStore
        )
        from hello_agents.memory.embedding import get_dimension

        self.doc_store = SQLiteDocumentStore(self.config.database_path)
        self.vector_store = QdrantVectorStore(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.qdrant_collection,
            vector_size=get_dimension(384)
        )

    def add(self, memory_item: MemoryItem) -> str:
        """
        添加情景记忆
        1. 生成文本向量
        2. 存入Qdrant（向量检索用）
        3. 存入SQLite（结构化查询用）
        """
        from hello_agents.memory.embedding import (
            get_text_embedder, embed_query
        )

        # 向量化
        embedding = embed_query(memory_item.content)

        # 构建元数据
        metadata = {
            "memory_id": memory_item.id,
            "content": memory_item.content,
            "memory_type": "episodic",
            "importance": memory_item.importance,
            "timestamp": memory_item.timestamp.isoformat(),
            **memory_item.metadata
        }

        # 存入Qdrant
        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[memory_item.id]
        )

        # 存入SQLite
        self.doc_store.save(
            memory_id=memory_item.id,
            content=memory_item.content,
            memory_type="episodic",
            importance=memory_item.importance,
            timestamp=memory_item.timestamp,
            metadata=memory_item.metadata
        )

        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[MemoryItem]:
        """
        混合检索：向量语义 + 时间近因性 + 重要性
        
        评分公式：
        (向量相似度×0.8 + 时间近因性×0.2) × (0.8 + 重要性×0.4)
        """
        from hello_agents.memory.embedding import embed_query

        query_vec = embed_query(query)
        hits = self.vector_store.search_similar(
            query_vector=query_vec,
            limit=limit * 3,    # 多取一些，评分后再筛选
            where={"memory_type": "episodic"}
        )

        scored = []
        for hit in hits:
            meta = hit["metadata"]
            vec_score = float(hit["score"])
            recency = self._recency_score(
                meta.get("timestamp", "")
            )
            importance = float(meta.get("importance", 0.5))

            # 书中评分公式
            base = vec_score * 0.8 + recency * 0.2
            weight = 0.8 + importance * 0.4
            final_score = base * weight

            memory_item = MemoryItem(
                id=meta.get("memory_id", hit["id"]),
                content=meta.get("content", ""),
                memory_type="episodic",
                importance=importance,
                timestamp=self._parse_time(
                    meta.get("timestamp", "")
                ),
                metadata={
                    k: v for k, v in meta.items()
                    if k not in (
                        "memory_id", "content",
                        "memory_type", "importance", "timestamp"
                    )
                }
            )
            scored.append((final_score, memory_item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def clear(self):
        """清空情景记忆（SQLite中的episodic类型记录）"""
        rows = self.doc_store.query(memory_type="episodic", limit=9999)
        for row in rows:
            self.doc_store.delete(row["id"])

    # ── 内部方法 ──────────────────────────────

    def _recency_score(self, timestamp_str: str) -> float:
        """时间近因性：越近的记忆得分越高，指数衰减"""
        try:
            t = datetime.fromisoformat(timestamp_str)
            age_hours = (
                datetime.now() - t
            ).total_seconds() / 3600
            return max(0.1, math.exp(-0.1 * age_hours / 24))
        except Exception:
            return 0.5

    def _parse_time(self, timestamp_str: str) -> datetime:
        try:
            return datetime.fromisoformat(timestamp_str)
        except Exception:
            return datetime.now()
