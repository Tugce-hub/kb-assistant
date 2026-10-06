"""KnowledgeService: the single entry point used by the CLI, MCP server, Slack bot and HTTP API.

Every surface goes through the same ACL check, PII redaction, tracing and audit
logging, so security properties don't depend on which front-end was used.
"""

from __future__ import annotations

import logging
from functools import cached_property

from kbassist.config import Settings, get_settings
from kbassist.generation import Answer, Generator
from kbassist.index.store import Index, load_index
from kbassist.models import Chunk, Hit
from kbassist.observability import Trace
from kbassist.retrieval import Mode, Retriever
from kbassist.security.acl import ACL, Principal
from kbassist.security.audit import AuditLog
from kbassist.security.pii import redact_text

log = logging.getLogger(__name__)


class AccessDenied(Exception):
    pass


class KnowledgeService:
    def __init__(self, settings: Settings | None = None, index: Index | None = None) -> None:
        self.settings = settings or get_settings()
        self._index = index
        self.acl = ACL.from_file(self.settings.path("acl_file"))
        self.audit = AuditLog(self.settings.path("paths.audit_log"))
        self.pricing = self.settings.get("llm.pricing", {})

    @cached_property
    def index(self) -> Index:
        return self._index or load_index(self.settings.path("paths.index_dir"))

    @cached_property
    def retriever(self) -> Retriever:
        return Retriever(self.index, self.settings, self.acl)

    @cached_property
    def generator(self) -> Generator:
        return Generator(self.settings)

    def principal(self, user_id: str) -> Principal:
        return self.acl.principal(user_id)

    # ------------------------------------------------------------------ links

    def permalink(self, chunk: Chunk) -> str:
        url = self.index.manifest["sources"].get(chunk.source, "").removesuffix(".git")
        commit = self.index.manifest["commits"].get(chunk.source, "HEAD")
        if "github.com" not in url:
            return f"{chunk.path}#L{chunk.start_line}-L{chunk.end_line}"
        return f"{url}/blob/{commit}/{chunk.path}#L{chunk.start_line}-L{chunk.end_line}"

    # ------------------------------------------------------------------ search

    def search(self, query: str, user_id: str, k: int | None = None, mode: Mode = "hybrid_rerank") -> tuple[list[Hit], Trace]:
        trace = Trace()
        principal = self.principal(user_id)
        with trace.span("retrieval_total"):
            hits = self.retriever.search(query, principal, mode=mode, k=k, trace=trace)
        self.audit.write(
            "search",
            user=user_id,
            question=query,
            groups=sorted(principal.groups),
            mode=mode,
            chunks=[h.chunk.id for h in hits],
            paths=[h.chunk.path for h in hits],
            **trace.to_dict(self.pricing),
        )
        return hits, trace

    def ask(self, question: str, user_id: str, channel: str | None = None) -> tuple[Answer, Trace]:
        trace = Trace()
        principal = self.principal(user_id)
        with trace.span("total"):
            with trace.span("retrieval_total"):
                hits = self.retriever.search(question, principal, trace=trace)
            answer = self.generator.answer(question, hits, trace)
            answer.text = redact_text(answer.text)
        self.audit.write(
            "ask",
            user=user_id,
            question=question,
            answer=answer.text,
            channel=channel,
            groups=sorted(principal.groups),
            answerable=answer.answerable,
            retrieved=[h.chunk.id for h in hits],
            cited=[h.chunk.id for h in answer.cited_hits],
            cited_paths=[h.chunk.path for h in answer.cited_hits],
            model=answer.model,
            stop_reason=answer.stop_reason,
            **trace.to_dict(self.pricing),
        )
        return answer, trace

    # --------------------------------------------------------------- documents

    def get_document(self, path: str, user_id: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Return (redacted) indexed content for `path`, if the caller may read it."""
        principal = self.principal(user_id)
        if not self.acl.can_read(principal, path):
            self.audit.write("get_document_denied", user=user_id, path=path)
            raise AccessDenied(path)
        chunk = next((c for c in self.index.chunks if c.path == path), None)
        if chunk is None:  # only files selected for indexing are readable
            raise FileNotFoundError(path)
        file = self.settings.path("paths.repos_dir") / chunk.source / path
        lines = redact_text(file.read_text(encoding="utf-8")).split("\n")
        end_line = min(end_line or len(lines), len(lines))
        body = "\n".join(lines[start_line - 1 : end_line])
        self.audit.write("get_document", user=user_id, path=path, start_line=start_line, end_line=end_line)
        return body

    def list_sources(self) -> dict:
        m = self.index.manifest
        return {k: m[k] for k in ("sources", "commits", "num_files", "num_chunks", "embedding_model", "built_at")}
