import os
import re
import uuid
from typing import List, Dict, Any


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _is_cjk(ch: str) -> bool:
    """判断是否为中日韩字符"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    """
    估算Token数量（中英文混合）
    CJK字符：每字=1 token
    非CJK：按空白分词
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = len([t for t in text.split() if t])
    return cjk + non_cjk


def _get_markitdown():
    """获取MarkItDown实例（懒加载）"""
    try:
        from markitdown import MarkItDown
        return MarkItDown()
    except ImportError:
        return None


def _fallback_reader(path: str) -> str:
    """MarkItDown不可用时的备用文本读取"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


# ─────────────────────────────────────────────
# 核心函数：文档转Markdown
# ─────────────────────────────────────────────

def convert_to_markdown(path: str) -> str:
    """
    将文档转为文本
    PDF 优先用 pdfminer，其他格式用 MarkItDown
    """
    if not os.path.exists(path):
        print(f"[Document] 文件不存在: {path}")
        return ""

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return _process_pdf(path)

    # 非PDF用MarkItDown
    md = _get_markitdown()
    if md:
        try:
            result = md.convert(path)
            text = getattr(result, "text_content", None)
            if text and text.strip():
                return text
        except Exception as e:
            print(f"[Document] MarkItDown失败: {e}")

    return _fallback_reader(path)


def _process_pdf(path: str) -> str:
    """
    PDF文本提取
    优先 pdfminer（速度快），失败则降级
    """
    # 方案1：pdfminer（推荐，纯文本提取，最快）
    try:
        from pdfminer.high_level import extract_text
        print(f"[Document] 使用pdfminer提取PDF...")
        text = extract_text(path)
        if text and text.strip():
            print(f"[Document] pdfminer提取成功: {len(text)} 字符")
            return text
    except ImportError:
        print("[Document] pdfminer未安装，尝试其他方案")
    except Exception as e:
        print(f"[Document] pdfminer失败: {e}")

    # 方案2：pypdf（备用）
    try:
        import pypdf
        print(f"[Document] 使用pypdf提取PDF...")
        reader = pypdf.PdfReader(path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(text)
        result = "\n\n".join(pages)
        if result.strip():
            print(f"[Document] pypdf提取成功: {len(result)} 字符")
            return result
    except ImportError:
        pass
    except Exception as e:
        print(f"[Document] pypdf失败: {e}")

    # 方案3：直接读取文本
    return _fallback_reader(path)


# ─────────────────────────────────────────────
# 核心函数：智能分块
# ─────────────────────────────────────────────

def split_markdown(text: str) -> List[Dict]:
    """
    按Markdown标题层次分割段落
    
    原理：利用 #/##/### 等标题标记识别语义边界，
    保持每个分块的主题完整性。
    同时记录 heading_path（面包屑路径），如"第三章 > 3.1节"
    """
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush():
        if not buf:
            return
        content = "\n".join(buf).strip()
        if content:
            paragraphs.append({
                "content": content,
                "heading_path": (
                    " > ".join(heading_stack)
                    if heading_stack else None
                )
            })
        buf.clear()

    for line in lines:
        if line.strip().startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            title = line.lstrip("#").strip()
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level - 1]
            heading_stack.append(title)
        elif line.strip() == "":
            flush()
        else:
            buf.append(line)

    flush()

    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None}]

    return paragraphs


def chunk_document(
    text: str,
    chunk_tokens: int = 500,
    overlap_tokens: int = 50
) -> List[Dict]:
    """
    智能分块主函数
    
    流程：
    1. split_markdown → 按标题分割段落
    2. 按Token数量合并段落 → 控制块大小
    3. 保留重叠部分 → 保持上下文连续性
    
    chunk_tokens:   每块目标Token数（默认500）
    overlap_tokens: 块间重叠Token数（默认50），避免边界信息丢失
    """
    paragraphs = split_markdown(text)
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0

    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1

        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            # 当前批次已满，生成一个chunk
            content = "\n\n".join(x["content"] for x in cur)
            heading = next(
                (x["heading_path"] for x in reversed(cur)
                 if x.get("heading_path")),
                None
            )
            chunks.append({
                "id": str(uuid.uuid4()),
                "content": content,
                "heading_path": heading,
                "token_count": cur_tokens
            })

            # 构建重叠部分（从当前批次末尾保留一部分）
            if overlap_tokens > 0:
                kept, kept_tokens = [], 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur, cur_tokens = [], 0

    # 处理最后一批
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        heading = next(
            (x["heading_path"] for x in reversed(cur)
             if x.get("heading_path")),
            None
        )
        chunks.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "heading_path": heading,
            "token_count": cur_tokens
        })

    print(f"[Document] 分块完成：{len(chunks)} 块")
    return chunks
