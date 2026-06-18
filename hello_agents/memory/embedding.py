import os
import math
import re
from typing import List, Union
import numpy as np


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
# 嵌入方案一：百炼 DashScope API
# ─────────────────────────────────────────────

class DashScopeEmbedding:
    """
    使用阿里云百炼平台的text-embedding-v3模型
    需要在.env中配置 EMBED_API_KEY
    """

    def __init__(self, model_name: str = "text-embedding-v3"):
        self.model_name = model_name
        self.api_key = os.getenv("EMBED_API_KEY", "")
        self.dimension = 1024   # text-embedding-v3 输出维度

    def encode(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """将文本编码为向量列表"""
        import dashscope
        from dashscope import TextEmbedding

        if isinstance(texts, str):
            texts = [texts]

        dashscope.api_key = self.api_key
        response = TextEmbedding.call(
            model=self.model_name,
            input=texts
        )
        embeddings = [
            item["embedding"]
            for item in response.output["embeddings"]
        ]
        return embeddings


# ─────────────────────────────────────────────
# 嵌入方案二：本地 sentence-transformers 模型
# ─────────────────────────────────────────────

class LocalTransformerEmbedding:
    """
    使用本地下载的 sentence-transformers 模型
    你已经安装了 sentence-transformers，这是主用方案
    模型首次使用时会自动从HuggingFace下载到本地缓存
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self._model = None          # 懒加载，第一次encode时才初始化
        self.dimension = 384        # all-MiniLM-L6-v2 输出384维

    def _load_model(self):
        """懒加载模型，避免启动时占用过多时间"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[Embedding] 加载本地模型: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"[Embedding] 模型加载完成，维度: {self.dimension}")

    def encode(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """将文本编码为向量列表"""
        global _embedder_instance
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        # sentence-transformers 返回 numpy array，转为 Python list
        embeddings = self._model.encode(texts)
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [
            e.tolist() if hasattr(e, "tolist") else list(e)
            for e in embeddings
        ]


# ─────────────────────────────────────────────
# 嵌入方案三：TF-IDF 兜底方案
# ─────────────────────────────────────────────

class TFIDFEmbedding:
    """
    不依赖任何外部模型的轻量级嵌入方案
    当前两种方案都不可用时自动启用
    原理：用词频-逆文档频率构建稀疏向量，维度固定为512
    精度较低，但保证系统在任何环境下都能运行
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vocab: dict = {}       # 词汇表：词 → 索引
        self.idf: dict = {}         # 每个词的IDF值

    def _tokenize(self, text: str) -> List[str]:
        """简单分词：按空格和标点切分"""
        return re.findall(r'\w+', text.lower())

    def _update_vocab(self, texts: List[str]):
        """根据当前文本集合更新词汇表"""
        all_tokens = set()
        for text in texts:
            all_tokens.update(self._tokenize(text))
        for token in all_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) % self.dimension

    def encode(
        self, texts: Union[str, List[str]]
    ) -> List[List[float]]:
        """将文本编码为TF-IDF向量"""
        if isinstance(texts, str):
            texts = [texts]

        self._update_vocab(texts)
        result = []

        for text in texts:
            tokens = self._tokenize(text)
            vec = [0.0] * self.dimension

            # 统计词频
            tf: dict = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1

            # 填入向量
            for token, count in tf.items():
                if token in self.vocab:
                    idx = self.vocab[token]
                    vec[idx] += count / max(len(tokens), 1)

            # L2归一化
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]

            result.append(vec)

        return result


# ─────────────────────────────────────────────
# 统一工厂函数：按优先级自动选择嵌入方案
# ─────────────────────────────────────────────

_embedder_instance = None   # 全局单例，避免重复加载模型


def get_text_embedder():
    """
    获取嵌入模型实例（全局单例）
    
    选择优先级：
    1. EMBED_MODEL_TYPE=dashscope → 百炼API
    2. EMBED_MODEL_TYPE=local     → 本地Transformer（你的情况）
    3. 以上都失败                 → TF-IDF兜底
    """
    global _embedder_instance

    if _embedder_instance is not None:
        return _embedder_instance

    model_type = os.getenv("EMBED_MODEL_TYPE", "local")
    model_name = os.getenv(
        "EMBED_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    if model_type == "dashscope":
        try:
            embedder = DashScopeEmbedding(model_name)
            _embedder_instance = embedder
            return _embedder_instance
        except Exception as e:
            print(f"[Embedding] 百炼API不可用: {e}，尝试本地模型")

    if model_type in ("local", "transformer"):
        try:
            embedder = LocalTransformerEmbedding(model_name)
            _embedder_instance = embedder
            return _embedder_instance
        except Exception as e:
            print(f"[Embedding] 本地模型不可用: {e}，使用TF-IDF兜底")

    # 最终兜底
    print("[Embedding] 使用TF-IDF兜底方案")
    _embedder_instance = TFIDFEmbedding()
    return _embedder_instance


def get_dimension(default: int = 384) -> int:
    """获取当前嵌入模型的输出维度"""
    embedder = get_text_embedder()
    return getattr(embedder, "dimension", default)


def embed_query(text: str) -> List[float]:
    """
    对单条文本编码，返回一维向量
    这是最常用的便捷函数，检索时直接调用
    """
    embedder = get_text_embedder()
    result = embedder.encode(text)

    # encode可能返回二维列表（批量），取第一条
    if result and isinstance(result[0], list):
        return result[0]
    return result
