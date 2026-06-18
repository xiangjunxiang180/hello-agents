import math
from datetime import datetime
from typing import List

from hello_agents.memory.base import BaseMemory, MemoryConfig, MemoryItem


class PerceptualMemory(BaseMemory):
    """
    感知记忆
    
    特点：
    - 支持多模态数据（文本、图像、音频）
    - 按模态分离存储，避免维度不匹配
    - 当前实现以文本模态为主
    
    对应人类认知：感觉记忆，存储感知通道接收到的信息
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._init_storage()

    def _init_storage(self):
        from hello_agents.memory.storage.qdrant_store import (
            QdrantVectorStore
        )
        from hello_agents.memory.storage.document_store import (
            SQLiteDocumentStore
        )
        from hello_agents.memory.embedding import get_dimension

        self.vector_store = QdrantVectorStore(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.qdrant_collection,
            vector_size=get_dimension(384)
        )
        self.doc_store = SQLiteDocumentStore(self.config.database_path)

    def add(self, memory_item: MemoryItem) -> str:
        """添加感知记忆，存储文本描述和模态标注"""
        from hello_agents.memory.embedding import embed_query

        embedding = embed_query(memory_item.content)
        modality = memory_item.metadata.get("modality", "text")

        metadata = {
            "memory_id": memory_item.id,
            "content": memory_item.content,
            "memory_type": "perceptual",
            "modality": modality,
            "importance": memory_item.importance,
            "timestamp": memory_item.timestamp.isoformat(),
            **memory_item.metadata
        }

        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[memory_item.id]
        )
        self.doc_store.save(
            memory_id=memory_item.id,
            content=memory_item.content,
            memory_type="perceptual",
            importance=memory_item.importance,
            timestamp=memory_item.timestamp,
            metadata=memory_item.metadata
        )

        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[MemoryItem]:
        """
        检索感知记忆
        
        评分公式（与情景记忆相同）：
        (向量相似度×0.8 + 时间近因性×0.2) × (0.8 + 重要性×0.4)
        """
        from hello_agents.memory.embedding import embed_query

        target_modality = kwargs.get("target_modality")
        query_vec = embed_query(query)

        where = {"memory_type": "perceptual"}
        if target_modality:
            where["modality"] = target_modality

        hits = self.vector_store.search_similar(
            query_vector=query_vec,
            limit=limit * 3,
            where=where
        )

        scored = []
        for hit in hits:
            meta = hit["metadata"]
            vec_score = float(hit["score"])
            recency = self._recency_score(
                meta.get("timestamp", "")
            )
            importance = float(meta.get("importance", 0.5))

            base = vec_score * 0.8 + recency * 0.2
            weight = 0.8 + importance * 0.4
            final = base * weight

            memory_item = MemoryItem(
                id=meta.get("memory_id", hit["id"]),
                content=meta.get("content", ""),
                memory_type="perceptual",
                importance=importance,
                timestamp=self._parse_time(
                    meta.get("timestamp", "")
                ),
                metadata=meta
            )
            scored.append((final, memory_item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def clear(self):
        rows = self.doc_store.query(
            memory_type="perceptual", limit=9999
        )
        for row in rows:
            self.doc_store.delete(row["id"])

    def _recency_score(self, timestamp_str: str) -> float:
        try:
            t = datetime.fromisoformat(timestamp_str)
            age_hours = (
                datetime.now() - t
            ).total_seconds() / 3600
            return max(0.1, math.exp(-0.1 * age_hours / 24))
        except Exception:
            return 0.5

    def _parse_time(self, ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts)
        except Exception:
            return datetime.now()
