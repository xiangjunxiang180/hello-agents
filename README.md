# Hello Agents

一个从零构建的 AI 智能体框架，实现了记忆系统、RAG检索增强生成和上下文工程等核心能力。

## 功能模块

**记忆系统**
- 四种记忆类型：工作记忆、情景记忆、语义记忆、感知记忆
- 向量存储（Qdrant）+ 图数据库（Neo4j）+ 本地SQLite 混合架构
- 支持跨类型检索，基于相关性和重要性自动排序

**RAG 检索增强生成**
- 支持 PDF、Word、Excel 等多格式文档加载
- 智能分块，保持语义完整性
- MQE 多查询扩展 + HyDE 假设文档嵌入，提升检索精度

**上下文工程**
- GSSC 流水线：Gather → Select → Structure → Compress
- 基于相关性和新近性的信息评分与筛选
- 结构化笔记系统（NoteTool），支持跨会话状态管理

**智能文档问答助手**
- 基于 Gradio 的 Web 应用
- 上传文档后即可提问，LLM 基于检索内容生成回答
- 集成记忆系统，支持学习笔记和历史回顾

## 技术栈

- Python 3.9
- Qdrant · Neo4j · SQLite
- sentence-transformers
- Gradio
- 通义千问 API

## 快速开始

配置 `.env` 文件：

```env
API_KEY=你的通义千问Key
BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MODEL=qwen-plus
QDRANT_URL=你的Qdrant地址
QDRANT_API_KEY=你的Qdrant Key
NEO4J_URI=你的Neo4j地址
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=你的密码
EMBED_MODEL_TYPE=local
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

安装依赖并启动问答助手：

```bash
pip install qdrant-client sentence-transformers gradio pdfminer.six neo4j spacy pyyaml openai
python 11_QA_Assistant.py
```

访问 http://127.0.0.1:7860
