"""On-disk index: chunks.jsonl + embeddings.npy + bm25.pkl + manifest.json.

At this corpus size (low thousands of chunks) exact cosine search over a numpy
matrix is ~1 ms and has perfect recall, so a vector database would add an
operational dependency without improving anything. The `vector` stage in
retrieval.py is the one place Qdrant/pgvector would plug in.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kbassist.config import Settings
from kbassist.index.bm25 import BM25Index
from kbassist.index.embeddings import embed_passages
from kbassist.ingest.chunking import ChunkParams, chunk_file
from kbassist.ingest.github import iter_files, resolved_commit, sync_source
from kbassist.models import Chunk
from kbassist.security.pii import redact

log = logging.getLogger(__name__)


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    bm25: BM25Index
    manifest: dict

    @property
    def paths(self) -> list[str]:
        return [c.path for c in self.chunks]


def build_index(settings: Settings, sync: bool = True) -> Index:
    params = ChunkParams(**settings.get("chunking", {}))
    repos_dir = settings.path("paths.repos_dir")
    chunks: list[Chunk] = []
    commits: dict[str, str] = {}
    redactions = 0
    for source in settings.sources:
        checkout = sync_source(source, repos_dir) if sync else repos_dir / source.name
        commits[source.name] = resolved_commit(checkout)
        for rel, text in iter_files(source, checkout):
            # Secrets/PII never enter the index: redact before chunking/embedding.
            clean, n = redact(text)
            redactions += n
            chunks.extend(chunk_file(source.name, rel, clean, params))
    log.info("chunked %d files into %d chunks", len({c.path for c in chunks}), len(chunks))

    model = settings.get("embedding.model")
    t0 = time.perf_counter()
    embeddings = embed_passages(model, [c.search_text() for c in chunks], settings.get("embedding.batch_size", 32))
    embed_s = time.perf_counter() - t0
    bm25 = BM25Index().fit([c.search_text() for c in chunks])

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_model": model,
        "dim": int(embeddings.shape[1]),
        "num_chunks": len(chunks),
        "num_files": len({(c.source, c.path) for c in chunks}),
        "commits": commits,
        "sources": {s.name: s.url for s in settings.sources},
        "redactions": redactions,
        "embed_seconds": round(embed_s, 2),
    }
    return Index(chunks, embeddings, bm25, manifest)


def save_index(index: Index, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in index.chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", index.embeddings)
    with (index_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(index.bm25, f)
    (index_dir / "manifest.json").write_text(json.dumps(index.manifest, indent=2), encoding="utf-8")


def load_index(index_dir: Path) -> Index:
    if not (index_dir / "manifest.json").exists():
        raise FileNotFoundError(f"No index at {index_dir}. Run `python -m kbassist.cli index` first.")
    chunks = [
        Chunk(**json.loads(line))
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    embeddings = np.load(index_dir / "embeddings.npy")
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)  # our own artifact, written by save_index
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return Index(chunks, embeddings, bm25, manifest)
