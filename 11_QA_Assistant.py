# 11_QA_Assistant.py
# 智能文档问答助手 —— 第八章 8.4 节完整实现

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()


# ─────────────────────────────────────────────────────────
# 核心助手类：PDFLearningAssistant
# 封装 RAGTool 和 MemoryTool 的调用逻辑
# ─────────────────────────────────────────────────────────

class PDFLearningAssistant:
    """
    智能文档问答助手
    
    设计思路：
    - RAGTool  负责文档的向量化存储和智能检索
    - MemoryTool 负责学习过程的记忆管理
    - 两者通过 PDFLearningAssistant 协调配合
    """

    def __init__(self, user_id: str = "default_user"):
        """
        初始化学习助手
        
        user_id: 用户ID，实现不同用户数据隔离
        """
        self.user_id = user_id
        self.session_id = (
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # 初始化工具（懒加载，第一次使用时连接）
        from hello_agents.tools.memory_tool import MemoryTool
        from hello_agents.tools.rag_tool import RAGTool

        # MemoryTool：用 user_id 隔离不同用户的记忆空间
        self.memory_tool = MemoryTool(user_id=user_id)

        # RAGTool：用 rag_namespace 隔离不同用户的知识库
        self.rag_tool = RAGTool(rag_namespace=f"pdf_{user_id}")

        # 学习统计
        self.stats = {
            "session_start": datetime.now(),
            "documents_loaded": 0,
            "questions_asked": 0,
            "concepts_learned": 0
        }

        # 当前加载的文档名称
        self.current_document = None

    # ── 步骤1：加载PDF文档 ─────────────────────────────────

    def load_document(self, pdf_path: str) -> Dict[str, Any]:
        """
        加载PDF文档到知识库
        
        内部流程：
        RAGTool: MarkItDown转换 → 智能分块 → 向量化 → 存入Qdrant
        MemoryTool: 记录"加载文档"这个事件到情景记忆
        """
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "message": f"文件不存在: {pdf_path}"
            }

        start_time = time.time()

        # 【RAGTool】处理PDF
        result = self.rag_tool.execute(
            "add_document",
            file_path=pdf_path,
            chunk_size=1000,      # 每块约1000 tokens
            chunk_overlap=200     # 块间重叠200 tokens，保持连续性
        )

        process_time = time.time() - start_time

        if result.get("success", False):
            self.current_document = os.path.basename(pdf_path)
            self.stats["documents_loaded"] += 1

            # 【MemoryTool】记录到情景记忆
            # 用情景记忆的原因：这是一个有时间戳的具体事件
            self.memory_tool.execute(
                "add",
                content=f"加载了文档《{self.current_document}》",
                memory_type="episodic",
                importance=0.9,
                event_type="document_loaded",
                session_id=self.session_id
            )

            return {
                "success": True,
                "message": (
                    f"✅ 加载成功！共 {result.get('chunks', '?')} 个分块"
                    f"（耗时: {process_time:.1f}秒）"
                ),
                "document": self.current_document,
                "chunks": result.get("chunks", 0)
            }
        else:
            return {
                "success": False,
                "message": f"❌ 加载失败: {result.get('error', '未知错误')}"
            }

    # ── 步骤2：智能问答 ────────────────────────────────────

    def ask(
        self,
        question: str,
        use_advanced_search: bool = True
    ) -> str:
        """
        向文档提问
        
        内部流程（书中五步执行流程）：
        1. 将问题记录到工作记忆（当前会话上下文）
        2. RAGTool 执行高级检索（MQE + HyDE）
        3. 将问答事件记录到情景记忆（学习历程）
        4. 返回 LLM 基于检索结果生成的答案
        """
        if not self.current_document:
            try:
                stats = self.rag_tool.execute("stats")
                if "总块数: 0" in str(stats):
                    return "⚠️ 请先加载文档！"
                self.current_document = "已加载文档"
            except Exception:
                return "⚠️ 请先加载文档！"
   
        

        # 【MemoryTool】记录问题到工作记忆
        # 用工作记忆的原因：这是当前任务的临时上下文
        self.memory_tool.execute(
            "add",
            content=f"提问: {question}",
            memory_type="working",
            importance=0.6,
            session_id=self.session_id
        )

        # 【RAGTool】执行高级检索 + LLM生成答案
        # enable_mqe:  多查询扩展，提升召回率 30%-50%
        # enable_hyde: 假设文档嵌入，改善检索精度
        answer = self.rag_tool.execute(
            "ask",
            question=question,
            limit=5,
            enable_advanced_search=use_advanced_search,
            enable_mqe=use_advanced_search,
            enable_hyde=use_advanced_search
        )

        # 【MemoryTool】记录到情景记忆
        self.memory_tool.execute(
            "add",
            content=f"关于'{question}'的学习",
            memory_type="episodic",
            importance=0.7,
            event_type="qa_interaction",
            session_id=self.session_id
        )

        self.stats["questions_asked"] += 1
        return answer

    # ── 步骤3：记忆管理 ────────────────────────────────────

    def add_note(
        self,
        content: str,
        concept: Optional[str] = None
    ):
        """
        添加学习笔记
        
        用语义记忆的原因：笔记是抽象的概念知识，
        不是具体事件（情景），也不是临时上下文（工作记忆）
        """
        self.memory_tool.execute(
            "add",
            content=content,
            memory_type="semantic",
            importance=0.8,
            concept=concept or "general",
            session_id=self.session_id
        )
        self.stats["concepts_learned"] += 1

    def recall(self, query: str, limit: int = 5) -> str:
        """
        回顾学习历程
        跨所有记忆类型检索，返回格式化文本
        """
        result = self.memory_tool.execute(
            "search",
            query=query,
            limit=limit
        )
        return result

    # ── 步骤4：智能路由（RAG + Memory 协同） ──────────────

    def smart_query(self, query: str) -> str:
        """
        智能路由：自动判断使用 RAG 还是 Memory
        
        判断逻辑：
        - 包含文档相关词 → RAG检索文档
        - 包含记忆/历史相关词 → Memory检索
        - 其他 → 优先RAG，失败则Memory
        """
        doc_keywords = [
            "文档", "内容", "介绍", "说明", "章节",
            "什么是", "怎么", "如何", "为什么"
        ]
        memory_keywords = [
            "我问过", "之前", "历史", "记录",
            "笔记", "学过", "学习过"
        ]

        q_lower = query.lower()

        use_memory = any(kw in q_lower for kw in memory_keywords)
        use_rag = any(kw in q_lower for kw in doc_keywords)

        if use_memory and not use_rag:
            return self.recall(query)
        elif self.current_document:
            return self.ask(query)
        else:
            return "⚠️ 请先加载文档，或使用'回顾'功能查看学习历程。"

    # ── 步骤5：统计与报告 ──────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取当前会话统计信息"""
        duration = (
            datetime.now() - self.stats["session_start"]
        ).total_seconds()
        return {
            "会话时长": f"{duration:.0f}秒",
            "加载文档": self.stats["documents_loaded"],
            "提问次数": self.stats["questions_asked"],
            "学习笔记": self.stats["concepts_learned"],
            "当前文档": self.current_document or "未加载"
        }

    def generate_report(self, save_to_file: bool = True) -> Dict[str, Any]:
        """
        生成学习报告
        汇总 MemoryTool 和 RAGTool 的统计信息，导出为 JSON
        """
        memory_summary = self.memory_tool.execute("summary", limit=10)
        rag_stats = self.rag_tool.execute("stats")

        duration = (
            datetime.now() - self.stats["session_start"]
        ).total_seconds()

        report = {
            "session_info": {
                "session_id": self.session_id,
                "user_id": self.user_id,
                "start_time": self.stats["session_start"].isoformat(),
                "duration_seconds": duration
            },
            "learning_metrics": {
                "documents_loaded": self.stats["documents_loaded"],
                "questions_asked": self.stats["questions_asked"],
                "concepts_learned": self.stats["concepts_learned"]
            },
            "memory_summary": memory_summary,
            "rag_status": rag_stats
        }

        if save_to_file:
            report_file = f"learning_report_{self.session_id}.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(
                    report, f,
                    ensure_ascii=False, indent=2, default=str
                )
            report["report_file"] = report_file
            print(f"[Report] 已保存到: {report_file}")

        return report


# ─────────────────────────────────────────────────────────
# Gradio Web 界面
# ─────────────────────────────────────────────────────────

def build_gradio_app():
    """
    构建 Gradio Web 界面
    
    五个标签页对应书中五步执行流程：
    1. 初始化助手
    2. 加载文档（RAGTool）
    3. 智能问答（RAGTool + MemoryTool）
    4. 学习笔记（MemoryTool）
    5. 学习报告（统计汇总）
    """
    import gradio as gr

    # 全局助手实例（Gradio 会话共享）
    assistant_state: Dict[str, Optional[PDFLearningAssistant]] = {
        "instance": None
    }

    # ── Tab1：初始化助手 ───────────────────────────────────

    def init_assistant(user_id: str) -> str:
        uid = user_id.strip() or "default_user"
        try:
            assistant_state["instance"] = PDFLearningAssistant(
                user_id=uid
            )
            return (
                f"✅ 助手初始化成功！\n"
                f"用户ID: {uid}\n"
                f"会话ID: {assistant_state['instance'].session_id}\n\n"
                f"请切换到「加载文档」标签页上传PDF。"
            )
        except Exception as e:
            return f"❌ 初始化失败: {str(e)}"

    # ── Tab2：加载文档 ─────────────────────────────────────

    def load_pdf(pdf_file) -> str:
        if assistant_state["instance"] is None:
            return "⚠️ 请先在「初始化助手」标签页完成初始化。"
        if pdf_file is None:
            return "⚠️ 请先上传PDF文件。"

        # Gradio 上传文件后返回临时路径
        pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
        result = assistant_state["instance"].load_document(pdf_path)

        if result["success"]:
            return (
                f"{result['message']}\n"
                f"文档: {result['document']}\n"
                f"分块数: {result.get('chunks', '?')}\n\n"
                f"✅ 文档已就绪，可以开始提问！"
            )
        else:
            return result["message"]

    # ── Tab3：智能问答 ─────────────────────────────────────

    def ask_question(
    question: str,
    use_advanced: bool,
    history: list
    ):
        if assistant_state["instance"] is None:
            msg = "⚠️ 请先初始化助手。"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": msg})
            return history, ""

        if not question.strip():
            return history, ""

        answer = assistant_state["instance"].ask(
            question=question,
            use_advanced_search=use_advanced
        )
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    # ── Tab4：学习笔记 ─────────────────────────────────────

    def save_note(note_content: str, concept: str) -> str:
        if assistant_state["instance"] is None:
            return "⚠️ 请先初始化助手。"
        if not note_content.strip():
            return "⚠️ 笔记内容不能为空。"

        assistant_state["instance"].add_note(
            content=note_content,
            concept=concept.strip() or None
        )
        return f"✅ 笔记已保存到语义记忆！\n概念标签: {concept or '通用'}"

    def recall_memories(query: str) -> str:
        if assistant_state["instance"] is None:
            return "⚠️ 请先初始化助手。"
        return assistant_state["instance"].recall(query, limit=5)

    # ── Tab5：统计与报告 ────────────────────────────────────

    def show_stats() -> str:
        if assistant_state["instance"] is None:
            return "⚠️ 请先初始化助手。"
        stats = assistant_state["instance"].get_stats()
        lines = ["📊 当前学习统计\n"]
        for k, v in stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def gen_report() -> str:
        if assistant_state["instance"] is None:
            return "⚠️ 请先初始化助手。"
        report = assistant_state["instance"].generate_report(
            save_to_file=True
        )
        lines = [
            "📋 学习报告已生成\n",
            f"会话ID: {report['session_info']['session_id']}",
            f"用户ID: {report['session_info']['user_id']}",
            f"会话时长: {report['session_info']['duration_seconds']:.0f}秒",
            "",
            "📈 学习指标:",
            f"  加载文档: {report['learning_metrics']['documents_loaded']}",
            f"  提问次数: {report['learning_metrics']['questions_asked']}",
            f"  学习笔记: {report['learning_metrics']['concepts_learned']}",
            "",
            "🧠 记忆摘要:",
            report.get("memory_summary", ""),
            "",
            "📚 知识库状态:",
            str(report.get("rag_status", "")),
        ]
        if "report_file" in report:
            lines.append(f"\n💾 完整报告已保存: {report['report_file']}")
        return "\n".join(lines)

    # ── 构建 UI ───────────────────────────────────────────

    with gr.Blocks(
        title="智能文档问答助手",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            "# 📚 智能文档问答助手\n"
            "> 基于 HelloAgents RAGTool + MemoryTool 构建 | 第八章 8.4节"
        )

        # Tab1：初始化
        with gr.Tab("① 初始化助手"):
            gr.Markdown(
                "**首先完成初始化**，加载向量数据库、嵌入模型等资源。"
            )
            user_id_input = gr.Textbox(
                label="用户ID",
                placeholder="输入你的用户ID（默认: default_user）",
                value="default_user"
            )
            init_btn = gr.Button("🚀 初始化助手", variant="primary")
            init_output = gr.Textbox(label="初始化结果", lines=6)
            init_btn.click(
                fn=init_assistant,
                inputs=[user_id_input],
                outputs=[init_output]
            )

        # Tab2：加载文档
        with gr.Tab("② 加载文档"):
            gr.Markdown(
                "上传PDF文件，系统会自动完成：\n"
                "MarkItDown转换 → 智能分块 → 向量化 → 存入Qdrant"
            )
            pdf_input = gr.File(
                label="上传PDF文件",
                file_types=[".pdf"]
            )
            load_btn = gr.Button("📄 加载文档", variant="primary")
            load_output = gr.Textbox(label="加载结果", lines=6)
            load_btn.click(
                fn=load_pdf,
                inputs=[pdf_input],
                outputs=[load_output]
            )

        # Tab3：智能问答
        with gr.Tab("③ 智能问答"):
            gr.Markdown(
                "向文档提问，支持 **MQE多查询扩展** 和 **HyDE假设文档嵌入**。"
            )
            chatbot = gr.Chatbot(label="对话历史", height=400)
            with gr.Row():
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的问题...",
                    scale=4
                )
                ask_btn = gr.Button("发送", variant="primary", scale=1)
            advanced_check = gr.Checkbox(
                label="启用高级检索（MQE + HyDE）",
                value=True
            )
            ask_btn.click(
                fn=ask_question,
                inputs=[question_input, advanced_check, chatbot],
                outputs=[chatbot, question_input]
            )
            question_input.submit(
                fn=ask_question,
                inputs=[question_input, advanced_check, chatbot],
                outputs=[chatbot, question_input]
            )

        # Tab4：学习笔记
        with gr.Tab("④ 学习笔记"):
            gr.Markdown(
                "记录学习笔记（存入**语义记忆**），并可检索历史学习记录。"
            )
            with gr.Row():
                note_input = gr.Textbox(
                    label="笔记内容",
                    placeholder="记录你的学习心得、概念理解...",
                    lines=4,
                    scale=3
                )
                concept_input = gr.Textbox(
                    label="概念标签",
                    placeholder="如: Transformer, RAG, 注意力机制",
                    scale=1
                )
            save_btn = gr.Button("💾 保存笔记", variant="primary")
            note_output = gr.Textbox(label="保存结果", lines=2)
            save_btn.click(
                fn=save_note,
                inputs=[note_input, concept_input],
                outputs=[note_output]
            )

            gr.Markdown("---\n**🔍 回顾学习历程**")
            recall_input = gr.Textbox(
                label="检索关键词",
                placeholder="输入关键词检索历史记忆..."
            )
            recall_btn = gr.Button("🔍 检索记忆")
            recall_output = gr.Textbox(label="检索结果", lines=8)
            recall_btn.click(
                fn=recall_memories,
                inputs=[recall_input],
                outputs=[recall_output]
            )

        # Tab5：统计与报告
        with gr.Tab("⑤ 学习报告"):
            gr.Markdown(
                "查看本次学习统计，并生成详细的 JSON 学习报告。"
            )
            with gr.Row():
                stats_btn = gr.Button("📊 查看统计", variant="secondary")
                report_btn = gr.Button("📋 生成报告", variant="primary")
            report_output = gr.Textbox(
                label="统计 / 报告",
                lines=20
            )
            stats_btn.click(fn=show_stats, outputs=[report_output])
            report_btn.click(fn=gen_report, outputs=[report_output])

    return demo


# ─────────────────────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("智能文档问答助手 —— HelloAgents 第八章 8.4节")
    print("=" * 50)
    app = build_gradio_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,           # 本地访问，不创建公网链接
        show_error=True
    )
    # 启动后访问 http://localhost:7860
