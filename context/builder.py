# hello_agents/context/builder.py
# ContextBuilder：GSSC 流水线实现
# Gather(汇集) → Select(选择) → Structure(结构化) → Compress(压缩)

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List


# ─────────────────────────────────────────────
# 核心数据结构
# ─────────────────────────────────────────────

@dataclass
class ContextPacket:
    """
    候选信息包 —— 系统中信息的基本流转单元
    
    每条候选信息都封装为 ContextPacket，
    统一结构简化了后续评分和排序逻辑。
    """
    content: str                            # 信息内容
    timestamp: datetime                     # 时间戳，用于新近性计算
    token_count: int                        # Token 数量，用于预算控制
    relevance_score: float = 0.5            # 相关性分数 0.0~1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # 确保相关性分数在有效范围内
        self.relevance_score = max(0.0, min(1.0, self.relevance_score))


@dataclass
class ContextConfig:
    """
    上下文构建配置
    
    关键参数说明：
    - reserve_ratio：为系统指令预留的比例，确保角色定义不被挤掉
    - recency_weight + relevance_weight 必须等于 1.0
    """
    max_tokens: int = 3000                  # 最大 token 数量
    reserve_ratio: float = 0.2             # 为系统指令预留的比例
    min_relevance: float = 0.1             # 最低相关性阈值，低于此值过滤掉
    enable_compression: bool = True        # 是否启用兜底压缩
    recency_weight: float = 0.3            # 新近性权重
    relevance_weight: float = 0.7          # 相关性权重

    def __post_init__(self):
        assert 0.0 <= self.reserve_ratio <= 1.0, \
            "reserve_ratio 必须在 [0, 1] 范围内"
        assert 0.0 <= self.min_relevance <= 1.0, \
            "min_relevance 必须在 [0, 1] 范围内"
        assert abs(self.recency_weight + self.relevance_weight - 1.0) < 1e-6, \
            "recency_weight + relevance_weight 必须等于 1.0"


# ─────────────────────────────────────────────
# ContextBuilder 主类
# ─────────────────────────────────────────────

class ContextBuilder:
    """
    上下文构建器 —— GSSC 流水线
    
    将上下文构建分解为四个清晰阶段：
    1. Gather：从多个来源汇集候选信息
    2. Select：按相关性+新近性评分，在 token 预算内贪心选择
    3. Structure：组织为固定骨架的结构化模板
    4. Compress：超限时兜底压缩，保持结构完整性
    """

    def __init__(
        self,
        memory_tool=None,
        rag_tool=None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()

    # ── 对外接口 ──────────────────────────────

    def build(
        self,
        user_query: str,
        conversation_history: Optional[List] = None,
        system_instructions: Optional[str] = None,
        custom_packets: Optional[List[ContextPacket]] = None
    ) -> str:
        """
        构建优化的上下文 —— 对外唯一入口
        
        完整流程：Gather → Select → Structure → Compress
        """
        # 计算可用 token 预算
        available_tokens = int(
            self.config.max_tokens * (1 - self.config.reserve_ratio)
        )

        # Step1: Gather 汇集
        packets = self._gather(
            user_query=user_query,
            conversation_history=conversation_history,
            system_instructions=system_instructions,
            custom_packets=custom_packets
        )

        # Step2: Select 选择
        selected = self._select(
            packets=packets,
            user_query=user_query,
            available_tokens=available_tokens
        )

        # Step3: Structure 结构化
        context = self._structure(
            selected_packets=selected,
            user_query=user_query
        )

        # Step4: Compress 兜底压缩
        if self.config.enable_compression:
            context = self._compress(context, self.config.max_tokens)

        return context

    # ── Step1: Gather ─────────────────────────

    def _gather(
        self,
        user_query: str,
        conversation_history=None,
        system_instructions=None,
        custom_packets=None
    ) -> List[ContextPacket]:
        """
        汇集所有候选信息
        
        来源优先级：
        1. 系统指令（最高，始终保留）
        2. 记忆系统检索结果
        3. RAG 知识库检索结果
        4. 对话历史（最近5条）
        5. 自定义信息包（调用者传入）
        """
        packets = []

        # 1. 系统指令 —— 最高优先级，不参与评分
        if system_instructions:
            packets.append(ContextPacket(
                content=system_instructions,
                timestamp=datetime.now(),
                token_count=self._count_tokens(system_instructions),
                relevance_score=1.0,
                metadata={"type": "system_instruction", "priority": "high"}
            ))

        # 2. 从记忆系统检索相关记忆
        if self.memory_tool:
            try:
                memory_results = self.memory_tool.execute(
                    "search",
                    query=user_query,
                    limit=10,
                    min_importance=0.3
                )
                memory_packets = self._parse_memory_results(
                    memory_results, user_query
                )
                packets.extend(memory_packets)
            except Exception as e:
                print(f"[WARNING] 记忆检索失败: {e}")

        # 3. 从 RAG 系统检索相关知识
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.execute(
                    "search",
                    query=user_query,
                    limit=5,
                    min_score=0.3
                )
                rag_packets = self._parse_rag_results(
                    rag_results, user_query
                )
                packets.extend(rag_packets)
            except Exception as e:
                print(f"[WARNING] RAG 检索失败: {e}")

        # 4. 对话历史（仅保留最近5条）
        if conversation_history:
            recent = conversation_history[-5:]
            for msg in recent:
                content = (
                    f"{msg.role}: {msg.content}"
                    if hasattr(msg, 'role') else str(msg)
                )
                packets.append(ContextPacket(
                    content=content,
                    timestamp=(
                        msg.timestamp
                        if hasattr(msg, 'timestamp')
                        else datetime.now()
                    ),
                    token_count=self._count_tokens(content),
                    relevance_score=0.6,
                    metadata={"type": "conversation_history"}
                ))

        # 5. 自定义信息包
        if custom_packets:
            packets.extend(custom_packets)

        print(f"[ContextBuilder] 汇集了 {len(packets)} 个候选信息包")
        return packets

    # ── Step2: Select ─────────────────────────

    def _select(
        self,
        packets: List[ContextPacket],
        user_query: str,
        available_tokens: int
    ) -> List[ContextPacket]:
        """
        选择最相关的信息包
        
        算法：
        1. 系统指令单独处理（始终保留）
        2. 其余信息按 综合分 = 相关性×0.7 + 新近性×0.3 排序
        3. 贪心填充：按分数从高到低，直到 token 预算耗尽
        """
        # 分离系统指令和其他信息
        system_packets = [
            p for p in packets
            if p.metadata.get("type") == "system_instruction"
        ]
        other_packets = [
            p for p in packets
            if p.metadata.get("type") != "system_instruction"
        ]

        # 计算系统指令消耗
        system_tokens = sum(p.token_count for p in system_packets)
        remaining_tokens = available_tokens - system_tokens

        if remaining_tokens <= 0:
            print("[WARNING] 系统指令已占满所有 token 预算")
            return system_packets

        # 为其他信息计算综合分数
        scored = []
        for packet in other_packets:
            # 若相关性为默认值 0.5，重新计算
            if packet.relevance_score == 0.5:
                packet.relevance_score = self._calculate_relevance(
                    packet.content, user_query
                )

            recency = self._calculate_recency(packet.timestamp)
            combined = (
                self.config.relevance_weight * packet.relevance_score +
                self.config.recency_weight * recency
            )

            # 过滤低相关性信息
            if packet.relevance_score >= self.config.min_relevance:
                scored.append((combined, packet))

        # 按综合分降序排序
        scored.sort(key=lambda x: x[0], reverse=True)

        # 贪心填充
        selected = system_packets.copy()
        current_tokens = system_tokens

        for score, packet in scored:
            if current_tokens + packet.token_count <= available_tokens:
                selected.append(packet)
                current_tokens += packet.token_count
            else:
                break

        print(
            f"[ContextBuilder] 选择了 {len(selected)} 个信息包，"
            f"共 {current_tokens} tokens"
        )
        return selected

    def _calculate_relevance(self, content: str, query: str) -> float:
        """
        Jaccard 相似度计算相关性
        生产环境可替换为向量相似度
        """
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        if not query_words:
            return 0.0
        intersection = content_words & query_words
        union = content_words | query_words
        return len(intersection) / len(union) if union else 0.0

    def _calculate_recency(self, timestamp: datetime) -> float:
        """
        指数衰减模型：24小时内保持高分，之后逐渐衰减
        """
        age_hours = (datetime.now() - timestamp).total_seconds() / 3600
        decay_factor = 0.1
        score = math.exp(-decay_factor * age_hours / 24)
        return max(0.1, min(1.0, score))

    # ── Step3: Structure ──────────────────────

    def _structure(
        self,
        selected_packets: List[ContextPacket],
        user_query: str
    ) -> str:
        """
        将选中信息组织为固定骨架的结构化模板
        
        模板分区：
        [Role & Policies] → [Task] → [Evidence] → [Context] → [Output]
        
        固定骨架的价值：便于调试、A/B测试、评估
        """
        system_instructions = []
        evidence = []
        context = []

        for packet in selected_packets:
            ptype = packet.metadata.get("type", "general")
            if ptype == "system_instruction":
                system_instructions.append(packet.content)
            elif ptype in ["rag_result", "knowledge"]:
                evidence.append(packet.content)
            else:
                context.append(packet.content)

        sections = []

        if system_instructions:
            sections.append(
                "[Role & Policies]\n" + "\n".join(system_instructions)
            )

        sections.append(f"[Task]\n{user_query}")

        if evidence:
            sections.append("[Evidence]\n" + "\n---\n".join(evidence))

        if context:
            sections.append("[Context]\n" + "\n".join(context))

        sections.append("[Output]\n请基于以上信息，提供准确、有据的回答。")

        return "\n\n".join(sections)

    # ── Step4: Compress ───────────────────────

    def _compress(self, context: str, max_tokens: int) -> str:
        """
        兜底压缩：超限时按分区截断，保持结构完整性
        
        策略：按分区从前到后贪心保留，
        预算不足时截断当前分区并标注 "[... 内容已压缩 ...]"
        """
        current_tokens = self._count_tokens(context)
        if current_tokens <= max_tokens:
            return context

        print(
            f"[ContextBuilder] 上下文超限"
            f"({current_tokens} > {max_tokens})，执行压缩"
        )

        sections = context.split("\n\n")
        compressed = []
        total = 0

        for section in sections:
            section_tokens = self._count_tokens(section)
            if total + section_tokens <= max_tokens:
                compressed.append(section)
                total += section_tokens
            else:
                remaining = max_tokens - total
                if remaining > 50:
                    truncated = self._truncate_text(section, remaining)
                    compressed.append(truncated + "\n[... 内容已压缩 ...]")
                break

        return "\n\n".join(compressed)

    # ── 工具方法 ──────────────────────────────

    def _count_tokens(self, text: str) -> int:
        """
        估算 token 数量
        简单实现：中文字符按1字=1token，英文按空格分词
        """
        if not text:
            return 0
        cjk = sum(
            1 for ch in text
            if '\u4e00' <= ch <= '\u9fff' or
               '\u3400' <= ch <= '\u4dbf'
        )
        non_cjk = len(text.split()) - cjk // 2
        return max(cjk + non_cjk, len(text) // 4)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """按 token 数截断文本"""
        words = text.split()
        result = []
        count = 0
        for word in words:
            word_tokens = self._count_tokens(word)
            if count + word_tokens > max_tokens:
                break
            result.append(word)
            count += word_tokens
        return " ".join(result)

    def _parse_memory_results(
        self, results: str, query: str
    ) -> List[ContextPacket]:
        """将记忆检索的文本结果解析为 ContextPacket 列表"""
        packets = []
        if not results or "未找到" in results:
            return packets

        lines = results.split("\n")
        current_content = []

        for line in lines:
            if line.strip():
                current_content.append(line)
            elif current_content:
                content = "\n".join(current_content)
                packets.append(ContextPacket(
                    content=f"记忆: {content}",
                    timestamp=datetime.now(),
                    token_count=self._count_tokens(content),
                    relevance_score=0.7,
                    metadata={"type": "memory"}
                ))
                current_content = []

        if current_content:
            content = "\n".join(current_content)
            packets.append(ContextPacket(
                content=f"记忆: {content}",
                timestamp=datetime.now(),
                token_count=self._count_tokens(content),
                relevance_score=0.7,
                metadata={"type": "memory"}
            ))

        return packets[:5]  # 最多取5条

    def _parse_rag_results(
        self, results: str, query: str
    ) -> List[ContextPacket]:
        """将 RAG 检索的文本结果解析为 ContextPacket 列表"""
        packets = []
        if not results or "未找到" in results:
            return packets

        # RAG 结果按 \n\n 分隔
        chunks = results.split("\n\n")
        for chunk in chunks[:3]:  # 最多取3块
            if chunk.strip():
                packets.append(ContextPacket(
                    content=chunk.strip(),
                    timestamp=datetime.now(),
                    token_count=self._count_tokens(chunk),
                    relevance_score=0.8,
                    metadata={"type": "rag_result"}
                ))
        return packets
