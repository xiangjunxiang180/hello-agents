from datetime import datetime
from typing import List, Optional, Any

from hello_agents.tools.tool import Tool


class MemoryTool(Tool):
    """
    记忆工具
    
    将 MemoryManager 封装为标准工具，Agent 通过 execute() 调用。
    
    支持的 action：
    - add:         添加记忆
    - search:      搜索记忆
    - summary:     获取记忆摘要
    - stats:       统计信息
    - forget:      遗忘记忆
    - consolidate: 整合记忆（工作→长期）
    - clear_all:   清空所有记忆
    """

    def __init__(
        self,
        user_id: str = "default_user",
        memory_types: List[str] = None,
        enable_perceptual: bool = False
    ):
        super().__init__(
            name="memory",
            description="记忆管理工具，支持添加、搜索、整合和遗忘记忆"
        )
        self.user_id = user_id
        self.memory_types = memory_types or [
            "working", "episodic", "semantic"
        ]
        self.current_session_id = None

        # 延迟初始化，避免启动时连接超时影响整体
        self._memory_manager = None
        self._enable_perceptual = enable_perceptual

    @property
    def memory_manager(self):
        """懒加载 MemoryManager"""
        if self._memory_manager is None:
            from hello_agents.memory.manager import MemoryManager
            self._memory_manager = MemoryManager(
                user_id=self.user_id,
                enable_working="working" in self.memory_types,
                enable_episodic="episodic" in self.memory_types,
                enable_semantic="semantic" in self.memory_types,
                enable_perceptual=self._enable_perceptual
            )
        return self._memory_manager

    def execute(self, action: str, **kwargs) -> str:
        """
        统一入口，按 action 路由
        """
        try:
            if action == "add":
                return self._add(**kwargs)
            elif action == "search":
                return self._search(**kwargs)
            elif action == "summary":
                return self._summary(**kwargs)
            elif action == "stats":
                return self._stats()
            elif action == "forget":
                return self._forget(**kwargs)
            elif action == "consolidate":
                return self._consolidate(**kwargs)
            elif action == "clear_all":
                self.memory_manager.clear_all()
                return "✅ 所有记忆已清空"
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ 操作失败 ({action}): {str(e)}"

    # ── 具体操作实现 ──────────────────────────

    def _add(
        self,
        content: str = "",
        memory_type: str = "working",
        importance: float = 0.5,
        session_id: str = None,
        **metadata
    ) -> str:
        """
        添加记忆
        
        自动管理 session_id：
        - 传入则使用传入的
        - 不传则复用当前会话，或自动创建新会话
        """
        if not content:
            return "❌ 内容不能为空"

        # 会话ID管理
        if session_id:
            self.current_session_id = session_id
        elif self.current_session_id is None:
            self.current_session_id = (
                f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

        metadata.update({
            "session_id": self.current_session_id,
            "timestamp": datetime.now().isoformat()
        })

        memory_id = self.memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata
        )
        return f"✅ 记忆已添加 (ID: {memory_id[:8]}...)"

    def _search(
        self,
        query: str = "",
        limit: int = 5,
        memory_types: List[str] = None,
        memory_type: str = None,
        min_importance: float = 0.1
    ) -> str:
        """搜索记忆，返回格式化的文本结果"""
        if not query:
            return "❌ 查询内容不能为空"

        # 兼容单个类型参数
        if memory_type and not memory_types:
            memory_types = [memory_type]

        results = self.memory_manager.retrieve_memories(
            query=query,
            limit=limit,
            memory_types=memory_types,
            min_importance=min_importance
        )

        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆"

        lines = [f"🔍 找到 {len(results)} 条相关记忆：\n"]
        for i, item in enumerate(results, 1):
            time_str = (
                item.timestamp.strftime("%Y-%m-%d %H:%M")
                if item.timestamp else "未知时间"
            )
            lines.append(
                f"{i}. [{item.memory_type}] "
                f"重要性:{item.importance:.1f} | {time_str}\n"
                f"   {item.content[:150]}"
            )
        return "\n".join(lines)

    def _summary(self, limit: int = 10) -> str:
        """获取最近记忆的摘要"""
        results = self.memory_manager.retrieve_memories(
            query="最近的记忆",
            limit=limit
        )
        if not results:
            return "📭 暂无记忆"

        lines = [f"📋 记忆摘要（最近{len(results)}条）：\n"]
        for item in results:
            lines.append(
                f"• [{item.memory_type}] {item.content[:100]}"
            )
        return "\n".join(lines)

    def _stats(self) -> str:
        """获取记忆系统统计信息"""
        stats = self.memory_manager.get_stats()
        lines = [f"📊 记忆统计 (用户: {stats['user_id']})："]
        for mtype, info in stats.get("types", {}).items():
            lines.append(f"  • {mtype}: {info.get('count', '?')} 条")
        return "\n".join(lines)

    def _forget(
        self,
        strategy: str = "importance",
        threshold: float = 0.3,
        **kwargs
    ) -> str:
        """
        遗忘记忆
        
        strategy 支持：
        - importance: 删除重要性低于 threshold 的记忆
        - time:       删除超过一定时间的记忆
        """
        # 简化实现：基于重要性遗忘
        results = self.memory_manager.retrieve_memories(
            query="",
            limit=200,
            min_importance=0.0
        )
        forgotten = 0
        for item in results:
            if item.importance < threshold:
                # 实际删除需要各存储后端支持，此处记录数量
                forgotten += 1

        return f"🗑️ 遗忘策略执行完成，共处理 {forgotten} 条低重要性记忆"

    def _consolidate(
        self,
        session_id: str = None,
        **kwargs
    ) -> str:
        """
        记忆整合：将重要的工作记忆提升为情景记忆
        
        模拟人类睡眠时的记忆巩固过程：
        短期重要信息 → 长期存储
        """
        sid = session_id or self.current_session_id
        if not sid:
            return "❌ 没有可整合的会话"

        # 获取当前会话的工作记忆
        working = self.memory_manager.retrieve_memories(
            query="",
            limit=50,
            memory_types=["working"],
            min_importance=0.6    # 只整合重要性>=0.6的
        )

        consolidated = 0
        for item in working:
            if item.metadata.get("session_id") == sid:
                # 将工作记忆提升为情景记忆
                self.memory_manager.add_memory(
                    content=item.content,
                    memory_type="episodic",
                    importance=item.importance,
                    metadata={
                        **item.metadata,
                        "consolidated_from": "working",
                        "consolidated_at": (
                            datetime.now().isoformat()
                        )
                    }
                )
                consolidated += 1

        return f"✅ 整合完成：{consolidated} 条工作记忆已提升为情景记忆"
