import os
import uuid
from typing import List, Dict, Any, Optional


class QdrantVectorStore:
    """
    Qdrant 向量数据库封装
    
    Qdrant的核心概念：
    - Collection（集合）：类似关系数据库的表，存储同类向量
    - Point（点）：一条记录，包含id、vector（向量）、payload（元数据）
    - search_similar：用余弦相似度找最近邻
    """

    def __init__(
        self,
        url: str = None,
        api_key: str = None,
        collection_name: str = "hello_agents_vectors",
        vector_size: int = 384
    ):
        self.url = url or os.getenv("QDRANT_URL", "")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", "")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化Qdrant客户端并确保集合存在"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import (
                Distance, VectorParams, PointStruct
            )

            if self.url and self.api_key:
                self.client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
                )
                print(f"[Qdrant] 连接云服务: {self.url}")
            else:
                # 本地内存模式，测试用
                self.client = QdrantClient(":memory:")
                print("[Qdrant] 使用内存模式")

            self._ensure_collection()

        except Exception as e:
            print(f"[Qdrant] 初始化失败: {e}")
            self.client = None

    def _ensure_collection(self):
        """确保集合存在，不存在则创建"""
        from qdrant_client.models import Distance, VectorParams

        collections = [
            c.name for c in self.client.get_collections().collections
        ]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE   # 使用余弦距离
                )
            )
            print(f"[Qdrant] 创建集合: {self.collection_name}")
        else:
            print(f"[Qdrant] 使用已有集合: {self.collection_name}")

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: List[str] = None
    ) -> List[str]:
        """
        批量存入向量
        
        vectors:  每条记忆的向量表示
        metadata: 对应的元数据（content、memory_type、importance等）
        ids:      可选，指定ID；不传则自动生成
        """
        if not self.client:
            self._init_client()
        if not self.client:
            return []

        from qdrant_client.models import PointStruct

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        points = []
        for i, (vec, meta, pid) in enumerate(
            zip(vectors, metadata, ids)
        ):
            # 维度对齐：防止向量维度与集合不匹配
            if len(vec) != self.vector_size:
                if len(vec) < self.vector_size:
                    vec = vec + [0.0] * (self.vector_size - len(vec))
                else:
                    vec = vec[:self.vector_size]

            points.append(
                PointStruct(id=pid, vector=vec, payload=meta)
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return ids

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = None,
        where: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索
        qdrant-client >= 1.7 使用 query_points 替代旧版 search
        """
        if not self.client:
            self._init_client()
        if not self.client:
            return []
     

        # 维度对齐
        if len(query_vector) != self.vector_size:
            if len(query_vector) < self.vector_size:
                query_vector = query_vector + [0.0] * (
                    self.vector_size - len(query_vector)
                )
            else:
                query_vector = query_vector[:self.vector_size]

        # 构建元数据过滤器
        query_filter = None
        if where:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in where.items()
            ]
            query_filter = Filter(must=conditions)

        try:
            # 新版 API：query_points（qdrant-client >= 1.7）
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )
            points = response.points

        except AttributeError:
            # 极旧版本兜底（不应该走到这里）
            return []

        return [
            {
                "id": str(p.id),
                "score": p.score,
                "metadata": p.payload or {}
            }
            for p in points
        ]

    def delete_vectors(self, ids: List[str]):
        """删除指定ID的向量"""
        if not self.client or not ids:
            return
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids)
        )

    def count(self) -> int:
        """返回集合中的向量总数"""
        if not self.client:
            return 0
        try:
            # 新版 API 用 get_collection 获取 points_count
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            try:
                # 备用方法
                result = self.client.count(
                    collection_name=self.collection_name
                )
                return result.count or 0
            except Exception:
                return 0
