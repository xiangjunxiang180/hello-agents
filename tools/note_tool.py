# hello_agents/tools/note_tool.py
# NoteTool：结构化笔记工具
# 格式：Markdown + YAML 前置元数据
# 支持操作：create / read / update / search / list / summary / delete

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from hello_agents.tools.tool import Tool


class NoteTool(Tool):
    """
    结构化笔记工具
    
    设计理念：
    - Markdown + YAML 混合格式，人类可读 + 机器可解析
    - 每条笔记是独立的 .md 文件，天然支持 Git 版本控制
    - 维护 notes_index.json 索引文件，支持快速检索
    - 无需数据库，轻量级的状态追踪
    
    笔记类型（note_type）：
    - task_state:  阶段性任务状态和进度
    - conclusion:  重要结论和发现
    - blocker:     阻塞问题（最高优先级）
    - action:      下一步行动计划
    - reference:   重要参考资料
    - general:     通用笔记（默认）
    """

    def __init__(self, workspace: str = "./notes"):
        super().__init__(
            name="note",
            description="结构化笔记工具，支持创建、检索和管理长期记忆笔记"
        )
        self.workspace = os.path.abspath(workspace)
        os.makedirs(self.workspace, exist_ok=True)

        # 加载或初始化索引文件
        self.index_path = os.path.join(self.workspace, "notes_index.json")
        self.index: Dict[str, Dict] = self._load_index()

        print(f"[NoteTool] 初始化完成: {self.workspace}")
        print(f"[NoteTool] 已有笔记: {len(self.index)} 条")

    # ── 对外统一入口 ──────────────────────────

    def execute(self, action: str, **kwargs) -> Any:
        """
        统一入口，按 action 路由
        
        支持的 action：
        create / read / update / search / list / summary / delete
        """
        try:
            if action == "create":
                return self._create_note(**kwargs)
            elif action == "read":
                return self._read_note(**kwargs)
            elif action == "update":
                return self._update_note(**kwargs)
            elif action == "search":
                return self._search_notes(**kwargs)
            elif action == "list":
                return self._list_notes(**kwargs)
            elif action == "summary":
                return self._summary()
            elif action == "delete":
                return self._delete_note(**kwargs)
            else:
                return f"❌ 不支持的操作: {action}"
        except Exception as e:
            return f"❌ NoteTool 操作失败 ({action}): {str(e)}"

    # ── create：创建笔记 ──────────────────────

    def _create_note(
        self,
        title: str,
        content: str,
        note_type: str = "general",
        tags: Optional[List[str]] = None
    ) -> str:
        """
        创建笔记
        
        文件名即ID，格式：note_YYYYMMDD_HHMMSS_序号.md
        内容格式：YAML 前置元数据 + Markdown 正文
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        note_id = f"note_{timestamp}_{len(self.index)}"

        metadata = {
            "id": note_id,
            "title": title,
            "type": note_type,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # 构建完整 Markdown 文件
        md_content = self._build_markdown(metadata, content)

        # 保存文件
        file_path = os.path.join(self.workspace, f"{note_id}.md")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # 更新索引
        metadata["file_path"] = file_path
        self.index[note_id] = metadata
        self._save_index()

        print(f"[NoteTool] 创建笔记: {title} ({note_type})")
        return note_id

    # ── read：读取笔记 ────────────────────────

    def _read_note(self, note_id: str) -> Dict:
        """读取笔记内容，返回元数据+正文字典"""
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        file_path = self.index[note_id]["file_path"]
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = f.read()

        metadata, content = self._parse_markdown(raw)
        return {"metadata": metadata, "content": content}

    # ── update：更新笔记 ──────────────────────

    def _update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """更新笔记的标题、内容、类型或标签"""
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        note = self._read_note(note_id)
        metadata = note["metadata"]
        old_content = note["content"]

        if title:
            metadata["title"] = title
        if note_type:
            metadata["type"] = note_type
        if tags is not None:
            metadata["tags"] = tags
        if content is not None:
            old_content = content

        metadata["updated_at"] = datetime.now().isoformat()

        # 重新写入文件
        md_content = self._build_markdown(metadata, old_content)
        file_path = metadata.get(
            "file_path",
            self.index[note_id]["file_path"]
        )
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        # 同步索引
        metadata["file_path"] = file_path
        self.index[note_id] = metadata
        self._save_index()

        return f"✅ 笔记已更新: {metadata['title']}"

    # ── search：搜索笔记 ──────────────────────

    def _search_notes(
        self,
        query: str,
        limit: int = 10,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        关键词搜索笔记
        在标题和正文中搜索，支持按类型和标签过滤
        按更新时间倒序返回
        """
        results = []
        query_lower = query.lower()

        for note_id, metadata in self.index.items():
            # 类型过滤
            if note_type and metadata.get("type") != note_type:
                continue
            # 标签过滤
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue

            try:
                note = self._read_note(note_id)
                content = note["content"]
                title = metadata.get("title", "")

                if (query_lower in title.lower() or
                        query_lower in content.lower()):
                    results.append({
                        "note_id": note_id,
                        "title": title,
                        "type": metadata.get("type"),
                        "tags": metadata.get("tags", []),
                        "content": content,
                        "updated_at": metadata.get("updated_at")
                    })
            except Exception as e:
                print(f"[WARNING] 读取笔记 {note_id} 失败: {e}")
                continue

        results.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return results[:limit]

    # ── list：列出笔记 ────────────────────────

    def _list_notes(
        self,
        note_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        列出笔记元数据（不含正文），按更新时间倒序
        支持按类型和标签过滤
        """
        results = []

        for note_id, metadata in self.index.items():
            if note_type and metadata.get("type") != note_type:
                continue
            if tags:
                note_tags = set(metadata.get("tags", []))
                if not note_tags.intersection(tags):
                    continue
            results.append(metadata)

        results.sort(
            key=lambda x: x.get("updated_at", ""), reverse=True
        )
        return results[:limit]

    # ── summary：统计摘要 ─────────────────────

    def _summary(self) -> Dict:
        """
        返回笔记统计信息：
        总数、按类型分布、最近5条
        """
        type_counts: Dict[str, int] = {}
        for metadata in self.index.values():
            nt = metadata.get("type", "general")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        recent = sorted(
            self.index.values(),
            key=lambda x: x.get("updated_at", ""),
            reverse=True
        )[:5]

        return {
            "total_notes": len(self.index),
            "type_distribution": type_counts,
            "recent_notes": [
                {
                    "id": n["id"],
                    "title": n.get("title", ""),
                    "type": n.get("type"),
                    "updated_at": n.get("updated_at")
                }
                for n in recent
            ]
        }

    # ── delete：删除笔记 ──────────────────────

    def _delete_note(self, note_id: str) -> str:
        """删除笔记文件和索引记录"""
        if note_id not in self.index:
            raise ValueError(f"笔记不存在: {note_id}")

        file_path = self.index[note_id]["file_path"]
        if os.path.exists(file_path):
            os.remove(file_path)

        title = self.index[note_id].get("title", note_id)
        del self.index[note_id]
        self._save_index()

        return f"✅ 笔记已删除: {title}"

    # ── 内部工具方法 ──────────────────────────

    def _build_markdown(self, metadata: Dict, content: str) -> str:
        """构建 YAML + Markdown 混合格式文件内容"""
        try:
            import yaml
            yaml_header = yaml.dump(
                metadata, allow_unicode=True, sort_keys=False
            )
        except ImportError:
            # yaml 不可用时用简单格式
            lines = [f"{k}: {v}" for k, v in metadata.items()]
            yaml_header = "\n".join(lines) + "\n"

        return f"---\n{yaml_header}---\n\n{content}"

    def _parse_markdown(self, raw: str) -> Tuple[Dict, str]:
        """解析 Markdown 文件，分离 YAML 元数据和正文"""
        parts = raw.split('---\n', 2)

        if len(parts) >= 3:
            yaml_str = parts[1]
            content = parts[2].strip()
            try:
                import yaml
                metadata = yaml.safe_load(yaml_str) or {}
            except Exception:
                metadata = {}
        else:
            metadata = {}
            content = raw.strip()

        return metadata, content

    def _load_index(self) -> Dict:
        """从磁盘加载索引文件"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self):
        """将索引持久化到磁盘"""
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)
