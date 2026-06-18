import os
from typing import List, Dict, Any, Optional


class Neo4jGraphStore:
    """
    Neo4j 图数据库封装
    
    图数据库的核心概念：
    - Node（节点）：代表实体，如"Python"、"张三"
    - Relationship（关系）：节点之间的连接，如"张三 USES Python"
    - Cypher：Neo4j的查询语言，类似SQL但专为图设计
    
    语义记忆用它存储知识图谱，支持多跳关系查询，
    找到纯向量检索发现不了的隐含关联。
    """

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None
    ):
        self.uri = uri or os.getenv("NEO4J_URI", "")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.driver = None
        self._init_driver()

    def _init_driver(self):
        """初始化Neo4j连接"""
        if not self.uri or not self.password:
            print("[Neo4j] 未配置连接信息，图存储不可用")
            return

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # 验证连接
            self.driver.verify_connectivity()
            print(f"[Neo4j] 连接成功: {self.uri}")
            self._create_indexes()
        except Exception as e:
            print(f"[Neo4j] 连接失败: {e}")
            self.driver = None

    def _create_indexes(self):
        """创建索引，加速实体查找"""
        indexes = [
            "CREATE INDEX entity_name IF NOT EXISTS "
            "FOR (e:Entity) ON (e.name)",
            "CREATE INDEX memory_id IF NOT EXISTS "
            "FOR (m:Memory) ON (m.memory_id)",
        ]
        with self.driver.session() as session:
            for idx in indexes:
                try:
                    session.run(idx)
                except Exception:
                    pass
        print("[Neo4j] 索引创建完成")

    def add_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        memory_id: str,
        properties: Dict = None
    ):
        """
        添加实体节点
        使用 MERGE 避免重复创建同名实体
        """
        if not self.driver:
            return

        props = properties or {}
        with self.driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {name: $name, type: $type})
                ON CREATE SET e.entity_id = $entity_id,
                              e.created_at = timestamp()
                SET e.last_seen = timestamp()
                WITH e
                MERGE (m:Memory {memory_id: $memory_id})
                MERGE (e)-[:APPEARS_IN]->(m)
                """,
                name=name,
                type=entity_type,
                entity_id=entity_id,
                memory_id=memory_id
            )

    def add_relation(
        self,
        source_name: str,
        target_name: str,
        relation_type: str,
        memory_id: str
    ):
        """
        添加实体间的关系
        例如：("张三", "Python", "USES")
        """
        if not self.driver:
            return

        with self.driver.session() as session:
            session.run(
                """
                MATCH (s:Entity {name: $source})
                MATCH (t:Entity {name: $target})
                MERGE (s)-[r:RELATION {type: $rel_type}]->(t)
                ON CREATE SET r.memory_id = $memory_id,
                              r.created_at = timestamp()
                """,
                source=source_name,
                target=target_name,
                rel_type=relation_type,
                memory_id=memory_id
            )

    def search_by_keyword(
        self, keyword: str, limit: int = 10
    ) -> List[Dict]:
        """
        通过关键词搜索相关实体及其关联记忆
        返回包含该关键词的实体节点以及它们 APPEARS_IN 的记忆ID
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:APPEARS_IN]->(m:Memory)
                WHERE toLower(e.name) CONTAINS toLower($keyword)
                RETURN e.name AS entity, e.type AS type,
                       m.memory_id AS memory_id
                LIMIT $limit
                """,
                keyword=keyword,
                limit=limit
            )
            return [dict(r) for r in result]

    def get_related_memories(
        self, memory_id: str, hops: int = 2
    ) -> List[str]:
        """
        找与指定记忆通过实体关联的其他记忆ID
        hops：图上跳数，hops=2表示"朋友的朋友"级别的关联
        """
        if not self.driver:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (m1:Memory {memory_id: $memory_id})
                      <-[:APPEARS_IN]-(e:Entity)
                      -[:APPEARS_IN]->(m2:Memory)
                WHERE m2.memory_id <> $memory_id
                RETURN DISTINCT m2.memory_id AS memory_id
                LIMIT 20
                """,
                memory_id=memory_id
            )
            return [r["memory_id"] for r in result]

    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
