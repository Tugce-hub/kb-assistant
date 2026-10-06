"""Local ONNX embedding + cross-encoder models via fastembed.

Running locally keeps repository content inside our network boundary (no
third-party embedding API sees source code) and makes indexing free. Models are
loaded lazily and cached per process.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=4)
def _embedder(model: str):
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model)


@lru_cache(maxsize=4)
def _cross_encoder(model: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model)


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def embed_passages(model: str, texts: list[str], batch_size: int = 32) -> np.ndarray:
    vecs = list(_embedder(model).passage_embed(texts, batch_size=batch_size))
    return _normalize(np.asarray(vecs, dtype=np.float32))


def embed_query(model: str, text: str) -> np.ndarray:
    vec = next(iter(_embedder(model).query_embed(text)))
    return _normalize(np.asarray([vec], dtype=np.float32))[0]


def rerank_scores(model: str, query: str, docs: list[str]) -> list[float]:
    return [float(s) for s in _cross_encoder(model).rerank(query, docs)]
