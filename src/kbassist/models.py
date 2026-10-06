from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    id: str
    source: str          # source name from settings (e.g. "httpx")
    path: str            # repo-relative POSIX path
    kind: str            # "doc" | "code" | "config"
    title: str           # heading breadcrumb or qualified symbol name
    start_line: int
    end_line: int
    text: str

    def search_text(self) -> str:
        """What gets embedded / BM25-indexed: path and title add strong lexical signal."""
        return f"{self.path}\n{self.title}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    # Rank of this chunk in each stage, e.g. {"bm25": 3, "vector": 1, "rrf": 1, "rerank": 2}
    ranks: dict[str, int] = field(default_factory=dict)
