"""BM25 (Okapi) with a code-aware tokenizer.

Identifiers are indexed both whole and split, so `max_keepalive_connections`
matches queries for "keepalive connections" and `HTTPStatusError` matches
"status error". Implemented directly (≈60 lines) instead of pulling in a
dependency, and stored as a plain inverted index that pickles quickly.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_TOKEN = re.compile(r"\w+", re.UNICODE)

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from how i if in is it its of on or
    the this that to what when where which who why will with you your we our
    there their them then than these those into out about via not no""".split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        if low in STOPWORDS or len(low) < 2:
            continue
        tokens.append(low)
        parts = [p.lower() for piece in raw.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            tokens.extend(p for p in parts if len(p) > 1 and p not in STOPWORDS)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, docs: list[str]) -> "BM25Index":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, doc in enumerate(docs):
            tf = Counter(tokenize(doc))
            lengths.append(sum(tf.values()))
            for term, count in tf.items():
                postings[term].append((i, count))
        self.postings = dict(postings)
        self.doc_len = np.asarray(lengths, dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(lengths) else 0.0
        n = len(docs)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }
        return self

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_len), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idx = np.fromiter((d for d, _ in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((c for _, c in plist), dtype=np.float32, count=len(plist))
            out[idx] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[idx])
        return out
