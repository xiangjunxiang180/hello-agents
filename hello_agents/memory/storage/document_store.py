import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional


class SQLiteDocumentStore:
    """
    SQLite 本地文档存储
    
    情景记忆和感知记忆需要持久化，系统重启后数据不丢失。
    SQLite是零配置的本地数据库，不需要额外的服务器进程。
    
    存储结构：
    - memories 表：存储所有记忆的原始内容和元数据
    - 用 JSON 序列化 metadata 字段
    """

    def __init__(self, db_path: str = "./memory_data/memory.db"):
        self.db_path = db_path
        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """创建数据库表和索引"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance  REAL DEFAULT 0.5,
                    timestamp   TEXT NOT NULL,
                    metadata    TEXT DEFAULT '{}'
                )
            """)
            # 建立索引加速按类型和时间查询
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_type "
                "ON memories(memory_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_time "
                "ON memories(timestamp)"
            )
            conn.commit()
        print(f"[SQLite] 初始化完成: {self.db_path}")

    def save(self, memory_id: str, content: str,
             memory_type: str, importance: float,
             timestamp: datetime, metadata: Dict) -> str:
        """保存一条记忆到SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, content, memory_type, importance, timestamp, metadata)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    memory_id,
                    content,
                    memory_type,
                    importance,
                    timestamp.isoformat(),
                    json.dumps(metadata, ensure_ascii=False)
                )
            )
            conn.commit()
        return memory_id

    def load(self, memory_id: str) -> Optional[Dict]:
        """按ID查询单条记忆"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def query(
        self,
        memory_type: str = None,
        limit: int = 100,
        min_importance: float = 0.0
    ) -> List[Dict]:
        """
        按类型和重要性批量查询
        按时间倒序返回（最新的在前）
        """
        sql = "SELECT * FROM memories WHERE importance >= ?"
        params = [min_importance]

        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()

        return [self._row_to_dict(r) for r in rows]

    def delete(self, memory_id: str):
        """删除指定记忆"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM memories WHERE id = ?", (memory_id,)
            )
            conn.commit()

    def count(self, memory_type: str = None) -> int:
        """统计记忆条数"""
        with sqlite3.connect(self.db_path) as conn:
            if memory_type:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type=?",
                    (memory_type,)
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM memories"
                ).fetchone()
        return row[0] if row else 0

    def _row_to_dict(self, row) -> Dict:
        """将SQLite行转为字典"""
        return {
            "id": row[0],
            "content": row[1],
            "memory_type": row[2],
            "importance": row[3],
            "timestamp": row[4],
            "metadata": json.loads(row[5] or "{}")
        }
