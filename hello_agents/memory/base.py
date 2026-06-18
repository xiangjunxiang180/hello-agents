import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional


@dataclass
class MemoryItem:
    """标准化记忆项"""
    content: str                                    # 记忆内容
    memory_type: str = "working"                    # 记忆类型：working/episodic/semantic/perceptual
    importance: float = 0.5                         # 重要性 0.0~1.0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 唯一ID
    timestamp: datetime = field(default_factory=datetime.now)   # 创建时间
    metadata: Dict[str, Any] = field(default_factory=dict)      # 额外元数据


@dataclass
class MemoryConfig:
    """记忆系统配置，从环境变量读取"""

    # 工作记忆参数
    working_memory_capacity: int = 50       # 最大条数
    working_memory_ttl: int = 60            # 存活时间（分钟）

    # 向量数据库（Qdrant）
    qdrant_url: str = field(
        default_factory=lambda: os.getenv("QDRANT_URL", "")
    )
    qdrant_api_key: str = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY", "")
    )
    qdrant_collection: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION", "hello_agents_vectors")
    )

    # 图数据库（Neo4j）
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )

    # SQLite 本地数据库路径
    database_path: str = "./memory_data/memory.db"

    # 相似度阈值
    similarity_threshold: float = 0.7


class BaseMemory(ABC):
    """所有记忆类型的抽象基类，定义统一接口"""

    def __init__(self, config: MemoryConfig):
        self.config = config

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆，返回记忆ID"""
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索记忆，返回最相关的记忆列表"""
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass
