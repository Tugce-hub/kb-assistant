"""Hybrid retrieval: BM25 + dense vectors, fused with RRF, then cross-encoder rerank.

Why each stage exists (numbers in README):
* BM25 nails exact identifiers (`DEFAULT_MAX_REDIRECTS`, `trust_env`) that
  small embedding models blur together.
* Dense vectors catch paraphrases ("stop following env proxy settings").
* Reciprocal Rank Fusion merges the two without having to calibrate BM25
  scores against cosine similarities.
* The cross-encoder reads query and chunk together and fixes the ordering of
  the fused pool; it is the most expensive stage, so it only sees `rerank_pool`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from kbassist.config import Settings
from kbassist.index.embeddings import embed_query, rerank_scores
from kbassist.index.store import Index
from kbassist.models import Hit
from kbassist.observability import Trace
from kbassist.security.acl import ACL, Principal

Mode = Literal["bm25", "vector", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("bm25", "vector", "hybrid", "hybrid_rerank")


def _top(scores: np.ndarray, mask: np.ndarray, n: int) -> list[int]:
    scores = np.where(mask, scores, -np.inf)
    n = min(n, int(mask.sum()))
    if n <= 0:
        return []
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return [int(i) for i in idx if np.isfinite(scores[i])]


def rrf(rankings: dict[str, list[int]], k: int = 60) -> list[tuple[int, float, dict[str, int]]]:
    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for r, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + r)
            ranks.setdefault(doc, {})[name] = r
    order = sorted(fused, key=lambda d: -fused[d])
    return [(d, fused[d], ranks[d]) for d in order]


class Retriever:
    def __init__(self, index: Index, settings: Settings, acl: ACL) -> None:
        self.index = index
        self.settings = settings
        self.acl = acl
        self.paths = index.paths
        self.embedding_model = index.manifest["embedding_model"]
        self.reranker_model = settings.get("reranker.model")
        self._masks: dict[frozenset[str], np.ndarray] = {}

    def _mask(self, principal: Principal) -> np.ndarray:
        if principal.groups not in self._masks:
            self._masks[principal.groups] = self.acl.mask(self.paths, principal)
        return self._masks[principal.groups]

    def search(
        self,
        query: str,
        principal: Principal,
        mode: Mode = "hybrid_rerank",
        k: int | None = None,
        trace: Trace | None = None,
    ) -> list[Hit]:
        trace = trace or Trace()
        k = k or self.settings.get("retrieval.top_k", 6)
        n = self.settings.get("retrieval.candidates_per_retriever", 30)
        mask = self._mask(principal)
        rankings: dict[str, list[int]] = {}

        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            with trace.span("bm25"):
                rankings["bm25"] = _top(self.index.bm25.scores(query), mask, n)
        if mode in ("vector", "hybrid", "hybrid_rerank"):
            with trace.span("embed_query"):
                q = embed_query(self.embedding_model, query)
            with trace.span("vector"):
                rankings["vector"] = _top(self.index.embeddings @ q, mask, n)

        with trace.span("fuse"):
            fused = rrf(rankings, self.settings.get("retrieval.rrf_k", 60))

        if mode != "hybrid_rerank" or not self.settings.get("reranker.enabled", True):
            return [
                Hit(self.index.chunks[d], score, {**r, "fused": i})
                for i, (d, score, r) in enumerate(fused[:k], start=1)
            ]

        pool = fused[: self.settings.get("retrieval.rerank_pool", 20)]
        # Cross-encoder cost grows with pool size x text length; the head of a chunk
        # (path, title, first lines) carries most of the relevance signal.
        max_chars = self.settings.get("retrieval.rerank_max_chars")
        with trace.span("rerank"):
            scores = rerank_scores(
                self.reranker_model,
                query,
                [self.index.chunks[d].search_text()[:max_chars] for d, _, _ in pool],
            )
        order = sorted(range(len(pool)), key=lambda i: -scores[i])
        return [
            Hit(self.index.chunks[pool[i][0]], scores[i], {**pool[i][2], "fused": i + 1, "rerank": rank})
            for rank, i in enumerate(order[:k], start=1)
        ]
