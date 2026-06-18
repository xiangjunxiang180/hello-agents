import math
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from hello_agents.memory.base import BaseMemory, MemoryConfig, MemoryItem


class SemanticMemory(BaseMemory):
    """
    语义记忆
    
    特点：
    - Qdrant向量检索 + Neo4j图检索混合架构
    - 自动提取实体和关系，构建知识图谱
    - 支持多跳关系推理
    
    对应人类认知：语义记忆，存储概念、规则、通用知识
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self._init_storage()
        self._init_nlp()

    def _init_storage(self):
        """初始化Qdrant和Neo4j"""
        from hello_agents.memory.storage.qdrant_store import (
            QdrantVectorStore
        )
        from hello_agents.memory.storage.neo4j_store import (
            Neo4jGraphStore
        )
        from hello_agents.memory.embedding import get_dimension

        self.vector_store = QdrantVectorStore(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key,
            collection_name=self.config.qdrant_collection,
            vector_size=get_dimension(384)
        )
        self.graph_store = Neo4jGraphStore(
            uri=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password
        )

    def _init_nlp(self):
        """
        初始化spaCy NLP模型（中英文）
        用于从文本中自动提取实体和关系
        """
        self.nlp_zh = None
        self.nlp_en = None

        try:
            import spacy
            try:
                self.nlp_zh = spacy.load("zh_core_web_sm")
                print("[SemanticMemory] 中文NLP模型加载成功")
            except Exception:
                print("[SemanticMemory] 中文NLP模型未安装，跳过")

            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                print("[SemanticMemory] 英文NLP模型加载成功")
            except Exception:
                print("[SemanticMemory] 英文NLP模型未安装，跳过")
        except ImportError:
            print("[SemanticMemory] spaCy未安装，实体提取不可用")

    def add(self, memory_item: MemoryItem) -> str:
        """
        添加语义记忆
        1. 生成向量 → 存Qdrant
        2. 提取实体和关系 → 存Neo4j
        """
        from hello_agents.memory.embedding import embed_query

        embedding = embed_query(memory_item.content)

        metadata = {
            "memory_id": memory_item.id,
            "content": memory_item.content,
            "memory_type": "semantic",
            "importance": memory_item.importance,
            "timestamp": memory_item.timestamp.isoformat(),
            **memory_item.metadata
        }

        # 存Qdrant
        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[metadata],
            ids=[memory_item.id]
        )

        # 提取实体和关系，存Neo4j
        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(
            memory_item.content, entities
        )

        for entity in entities:
            self.graph_store.add_entity(
                entity_id=entity["id"],
                name=entity["name"],
                entity_type=entity["type"],
                memory_id=memory_item.id
            )

        for rel in relations:
            self.graph_store.add_relation(
                source_name=rel["source"],
                target_name=rel["target"],
                relation_type=rel["type"],
                memory_id=memory_item.id
            )

        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[MemoryItem]:
        """
        混合检索：向量检索 + 图检索，结果融合排序
        
        评分公式：
        (向量相似度×0.7 + 图相似度×0.3) × (0.8 + 重要性×0.4)
        """
        from hello_agents.memory.embedding import embed_query

        query_vec = embed_query(query)

        # 向量检索
        vec_hits = self.vector_store.search_similar(
            query_vector=query_vec,
            limit=limit * 2,
            where={"memory_type": "semantic"}
        )

        # 图检索：按关键词找相关实体，再找关联记忆
        graph_memory_ids = set()
        keywords = query.split()[:3]   # 取前3个词做关键词
        for kw in keywords:
            results = self.graph_store.search_by_keyword(kw, limit=5)
            for r in results:
                graph_memory_ids.add(r.get("memory_id", ""))

        # 融合评分
        combined: Dict[str, Dict] = {}

        for hit in vec_hits:
            mid = hit["metadata"].get("memory_id", hit["id"])
            combined[mid] = {
                **hit["metadata"],
                "vector_score": float(hit["score"]),
                "graph_score": 0.0
            }

        # 图检索命中的记忆给予图分加成
        for mid in graph_memory_ids:
            if mid in combined:
                combined[mid]["graph_score"] = 0.6
            # 图检索独有的结果这里暂不额外处理
            # （向量检索已覆盖大部分相关内容）

        # 计算最终分数
        scored = []
        for mid, data in combined.items():
            vec_s = data.get("vector_score", 0.0)
            graph_s = data.get("graph_score", 0.0)
            importance = float(data.get("importance", 0.5))

            # 书中公式
            base = vec_s * 0.7 + graph_s * 0.3
            weight = 0.8 + importance * 0.4
            final = base * weight

            memory_item = MemoryItem(
                id=mid,
                content=data.get("content", ""),
                memory_type="semantic",
                importance=importance,
                timestamp=self._parse_time(
                    data.get("timestamp", "")
                ),
                metadata={
                    k: v for k, v in data.items()
                    if k not in (
                        "memory_id", "content", "memory_type",
                        "importance", "timestamp",
                        "vector_score", "graph_score"
                    )
                }
            )
            scored.append((final, memory_item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def clear(self):
        pass  # 向量数据库清理较复杂，实际使用中按需实现

    # ── 实体和关系提取 ────────────────────────

    def _extract_entities(
        self, text: str
    ) -> List[Dict[str, str]]:
        """
        用spaCy从文本中提取命名实体
        例如："张三是Python开发者" → [{"name":"张三","type":"PERSON"}]
        没有spaCy时退化为简单规则提取
        """
        entities = []
        seen = set()

        # 尝试spaCy提取
        nlp = self.nlp_zh or self.nlp_en
        if nlp:
            try:
                doc = nlp(text[:500])   # 限制长度避免超时
                for ent in doc.ents:
                    if ent.text not in seen:
                        entities.append({
                            "id": f"entity_{len(entities)}",
                            "name": ent.text,
                            "type": ent.label_
                        })
                        seen.add(ent.text)
                return entities
            except Exception:
                pass

        # 退化：用正则提取大写开头的词（英文）或2-4字短语（中文）
        for word in re.findall(r'[A-Z][a-z]+', text):
            if word not in seen and len(word) > 2:
                entities.append({
                    "id": f"entity_{len(entities)}",
                    "name": word,
                    "type": "UNKNOWN"
                })
                seen.add(word)

        return entities[:10]    # 最多10个实体

    def _extract_relations(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict[str, str]]:
        """
        从实体列表中提取两两之间的关系
        简化实现：实体共现于同一文本即认为存在关联
        """
        relations = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                relations.append({
                    "source": entities[i]["name"],
                    "target": entities[j]["name"],
                    "type": "CO_OCCURS"
                })
        return relations[:5]    # 最多5条关系

    def _parse_time(self, timestamp_str: str) -> datetime:
        try:
            return datetime.fromisoformat(timestamp_str)
        except Exception:
            return datetime.now()
