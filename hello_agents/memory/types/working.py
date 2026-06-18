import math
from datetime import datetime, timedelta
from typing import List, Dict

from hello_agents.memory.base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """
    工作记忆
    
    特点：
    - 纯内存存储，无持久化，重启后清空
    - 容量有限（默认50条），超出后清理优先级最低的
    - TTL机制（默认60分钟），过期自动删除
    - 混合检索：TF-IDF向量化 + 关键词匹配
    
    对应人类认知：短期记忆/工作区，处理当前任务的信息暂存区
    """

    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.max_capacity = config.working_memory_capacity  # 默认50
        self.max_age_minutes = config.working_memory_ttl    # 默认60分钟
        self.memories: List[MemoryItem] = []                # 内存存储

    def add(self, memory_item: MemoryItem) -> str:
        """
        添加工作记忆
        添加前先清理过期条目，再检查容量
        """
        self._expire_old_memories()

        if len(self.memories) >= self.max_capacity:
            # 容量已满，移除优先级最低的（重要性最低且最旧的）
            self._remove_lowest_priority_memory()

        self.memories.append(memory_item)
        return memory_item.id

    def retrieve(
        self, query: str, limit: int = 5, **kwargs
    ) -> List[MemoryItem]:
        """
        混合检索：TF-IDF向量 + 关键词匹配
        
        评分公式：
        base_relevance = 向量相似度×0.7 + 关键词得分×0.3
        final_score = base_relevance × 时间衰减 × (0.8 + 重要性×0.4)
        """
        self._expire_old_memories()

        if not self.memories:
            return []

        # 尝试TF-IDF向量检索
        vector_scores = self._try_tfidf_search(query)

        scored = []
        for memory in self.memories:
            vec_score = vector_scores.get(memory.id, 0.0)
            kw_score = self._keyword_score(query, memory.content)

            base = (
                vec_score * 0.7 + kw_score * 0.3
                if vec_score > 0 else kw_score
            )
            time_decay = self._time_decay(memory.timestamp)
            importance_weight = 0.8 + memory.importance * 0.4

            final = base * time_decay * importance_weight
            if final > 0:
                scored.append((final, memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def clear(self):
        self.memories.clear()

    # ── 内部方法 ──────────────────────────────

    def _expire_old_memories(self):
        """清理超过TTL的记忆"""
        cutoff = datetime.now() - timedelta(minutes=self.max_age_minutes)
        self.memories = [
            m for m in self.memories if m.timestamp >= cutoff
        ]

    def _remove_lowest_priority_memory(self):
        """移除综合优先级最低的记忆（重要性低+时间旧）"""
        if not self.memories:
            return
        scored = []
        for m in self.memories:
            age_minutes = (
                datetime.now() - m.timestamp
            ).total_seconds() / 60
            priority = m.importance - age_minutes / (
                self.max_age_minutes * 10
            )
            scored.append((priority, m))
        scored.sort(key=lambda x: x[0])
        # 移除优先级最低的
        to_remove = scored[0][1]
        self.memories = [
            m for m in self.memories if m.id != to_remove.id
        ]

    def _time_decay(self, timestamp: datetime) -> float:
        """
        时间衰减：记忆越旧得分越低
        24小时内保持较高分，之后指数衰减
        """
        age_hours = (
            datetime.now() - timestamp
        ).total_seconds() / 3600
        return max(0.1, math.exp(-0.1 * age_hours / 24))

    def _keyword_score(self, query: str, content: str) -> float:
        """关键词匹配得分：命中词数 / 查询总词数"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        if not query_words:
            return 0.0
        hits = query_words & content_words
        return len(hits) / len(query_words)

    def _try_tfidf_search(
        self, query: str
    ) -> Dict[str, float]:
        """
        尝试用TF-IDF做向量相似度检索
        失败则返回空字典，退化为纯关键词匹配
        """
        try:
            from hello_agents.memory.embedding import TFIDFEmbedding
            embedder = TFIDFEmbedding()
            texts = [m.content for m in self.memories]
            all_vecs = embedder.encode(texts)
            query_vec = embedder.encode(query)[0]

            scores = {}
            for memory, vec in zip(self.memories, all_vecs):
                # 余弦相似度
                dot = sum(a * b for a, b in zip(query_vec, vec))
                na = math.sqrt(sum(a * a for a in query_vec))
                nb = math.sqrt(sum(b * b for b in vec))
                sim = dot / (na * nb) if na > 0 and nb > 0 else 0.0
                scores[memory.id] = sim
            return scores
        except Exception:
            return {}
