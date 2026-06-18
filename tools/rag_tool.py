import os
from typing import Dict, Any, Optional, List

from hello_agents.tools.tool import Tool


class RAGTool(Tool):
    """
    RAG 工具
    
    将 RAGPipeline 封装为标准工具。
    
    支持的 action：
    - add_document: 加载文件到知识库
    - add_text:     直接添加文本到知识库
    - search:       向量检索
    - ask:          智能问答（检索+LLM生成）
    - stats:        知识库统计
    - clear:        清空知识库
    """

    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        qdrant_url: str = None,
        qdrant_api_key: str = None,
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default"
    ):
        super().__init__(
            name="rag",
            description=(
                "知识库检索工具，支持多格式文档加载和智能问答"
            )
        )
        self.knowledge_base_path = knowledge_base_path
        os.makedirs(knowledge_base_path, exist_ok=True)

        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "")
        self.qdrant_api_key = (
            qdrant_api_key or os.getenv("QDRANT_API_KEY", "")
        )
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace

        self._pipeline = None

    @property
    def pipeline(self):
        """懒加载 RAGPipeline"""
        if self._pipeline is None:
            from hello_agents.memory.rag.pipeline import RAGPipeline
            self._pipeline = RAGPipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace
            )
        return self._pipeline

    def execute(self, action: str, **kwargs) -> Any:
        """统一入口"""
        try:
            if action == "add_document":
                return self._add_document(**kwargs)
            elif action == "add_text":
                return self._add_text(**kwargs)
            elif action == "search":
                return self._search(**kwargs)
            elif action == "ask":
                return self._ask(**kwargs)
            elif action == "stats":
                return self._stats()
            elif action == "clear":
                return "⚠️ 清空操作请直接在Qdrant控制台操作"
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ RAG操作失败 ({action}): {str(e)}"

    def _add_document(
        self,
        file_path: str = "",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        **kwargs
    ) -> Dict:
        """加载文档到知识库"""
        if not file_path:
            return {"success": False, "error": "file_path不能为空"}
        return self.pipeline.add_document(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def _add_text(
        self,
        text: str = "",
        document_id: str = None,
        **kwargs
    ) -> str:
        """直接添加文本"""
        if not text:
            return "❌ 文本内容不能为空"
        result = self.pipeline.add_text(text, document_id=document_id)
        return (
            f"✅ 文本已添加 "
            f"(ID: {result['document_id'][:8]}..., "
            f"分块数: {result['chunks']})"
        )

    def _search(
        self,
        query: str = "",
        limit: int = 5,
        min_score: float = 0.1,
        **kwargs
    ) -> str:
        """向量检索，返回格式化结果"""
        if not query:
            return "❌ 查询内容不能为空"

        hits = self.pipeline.search(
            query=query, limit=limit, min_score=min_score
        )

        if not hits:
            return f"🔍 未找到与 '{query}' 相关的内容"

        lines = [f"🔍 找到 {len(hits)} 条相关内容：\n"]
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            content = meta.get("content", "")[:200]
            score = hit["score"]
            heading = meta.get("heading_path", "")
            lines.append(
                f"{i}. 相似度: {score:.3f}"
                + (f" | 来源: {heading}" if heading else "")
                + f"\n   {content}"
            )
        return "\n".join(lines)

    def _ask(
        self,
        question: str = "",
        limit: int = 5,
        enable_advanced_search: bool = True,
        enable_mqe: bool = True,
        enable_hyde: bool = True,
        **kwargs
    ) -> str:
        """智能问答"""
        if not question:
            return "❌ 问题不能为空"
        return self.pipeline.ask(
            question=question,
            limit=limit,
            enable_advanced_search=enable_advanced_search,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde
        )

    def _stats(self) -> str:
        """知识库统计"""
        stats = self.pipeline.get_stats()
        return (
            f"📊 知识库统计\n"
            f"  命名空间: {stats['namespace']}\n"
            f"  集合: {stats['collection']}\n"
            f"  总块数: {stats['total_chunks']}"
        )
