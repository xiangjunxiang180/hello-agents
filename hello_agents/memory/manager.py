from datetime import datetime
from typing import List, Dict, Optional, Any

from hello_agents.memory.base import MemoryConfig, MemoryItem


class MemoryManager:
    """
    记忆管理器 — 统一调度协调器
    
    职责：
    1. 初始化并持有四种记忆类型的实例
    2. 根据 memory_type 参数将操作路由到对应模块
    3. 对外暴露统一的 add_memory / retrieve_memories 接口
    
    MemoryTool → MemoryManager → WorkingMemory/EpisodicMemory/...
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False
    ):
        # 从环境变量加载配置
        from dotenv import load_dotenv
        load_dotenv()

        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.memory_types: Dict = {}

        # 按需初始化各记忆类型
        if enable_working:
            from hello_agents.memory.types.working import WorkingMemory
            self.memory_types["working"] = WorkingMemory(self.config)
            print("[MemoryManager] WorkingMemory 已启用")

        if enable_episodic:
            from hello_agents.memory.types.episodic import EpisodicMemory
            self.memory_types["episodic"] = EpisodicMemory(self.config)
            print("[MemoryManager] EpisodicMemory 已启用")

        if enable_semantic:
            from hello_agents.memory.types.semantic import SemanticMemory
            self.memory_types["semantic"] = SemanticMemory(self.config)
            print("[MemoryManager] SemanticMemory 已启用")

        if enable_perceptual:
            from hello_agents.memory.types.perceptual import (
                PerceptualMemory
            )
            self.memory_types["perceptual"] = PerceptualMemory(
                self.config
            )
            print("[MemoryManager] PerceptualMemory 已启用")

    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> str:
        """
        添加一条记忆
        
        会自动补充 user_id 到 metadata，
        保证不同用户的记忆在同一个向量集合中可以按 user_id 过滤。
        """
        if memory_type not in self.memory_types:
            # 类型不存在时降级到工作记忆
            memory_type = "working"

        meta = metadata or {}
        meta["user_id"] = self.user_id

        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now(),
            metadata=meta
        )

        return self.memory_types[memory_type].add(item)

    def retrieve_memories(
        self,
        query: str,
        limit: int = 5,
        memory_types: List[str] = None,
        min_importance: float = 0.1,
        **kwargs
    ) -> List[MemoryItem]:
        """
        跨类型检索记忆
        
        memory_types=None 时搜索所有已启用类型，
        结果合并后按重要性×相关性综合排序。
        """
        types_to_search = memory_types or list(self.memory_types.keys())
        all_results: List[MemoryItem] = []

        for mtype in types_to_search:
            if mtype not in self.memory_types:
                continue
            try:
                results = self.memory_types[mtype].retrieve(
                    query=query,
                    limit=limit,
                    user_id=self.user_id,
                    **kwargs
                )
                # 过滤低重要性结果
                results = [
                    r for r in results
                    if r.importance >= min_importance
                ]
                all_results.extend(results)
            except Exception as e:
                print(f"[MemoryManager] {mtype} 检索失败: {e}")

        # 合并去重（按ID），按重要性排序
        seen = set()
        unique = []
        for item in all_results:
            if item.id not in seen:
                seen.add(item.id)
                unique.append(item)

        unique.sort(key=lambda x: x.importance, reverse=True)
        return unique[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """获取各记忆类型的统计信息"""
        stats = {"user_id": self.user_id, "types": {}}
        for mtype, module in self.memory_types.items():
            try:
                if hasattr(module, "doc_store"):
                    count = module.doc_store.count(memory_type=mtype)
                elif hasattr(module, "memories"):
                    count = len(module.memories)
                else:
                    count = -1
                stats["types"][mtype] = {"count": count}
            except Exception:
                stats["types"][mtype] = {"count": -1}
        return stats

    def clear_all(self):
        """清空所有记忆"""
        for module in self.memory_types.values():
            try:
                module.clear()
            except Exception as e:
                print(f"[MemoryManager] 清空失败: {e}")
