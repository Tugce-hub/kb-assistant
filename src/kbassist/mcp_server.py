"""MCP server exposing the knowledge base as tools.

Any MCP client (Claude Code, Claude Desktop, Cursor, an internal agent) can use
it. Tools are deliberately split so an agent can do its own multi-step
retrieval: `search_knowledge_base` -> `read_file` for more context ->
answer, while `ask_knowledge_base` is the one-shot path that reuses our
grounded-answer prompt.

Identity: over stdio the caller is the local user, configured with
KB_MCP_USER (mapped to groups by config/acl.yaml). Behind an HTTP transport the
same field would come from the authenticated token.

Run:
    python -m kbassist.mcp_server                # stdio
    python -m kbassist.mcp_server --http 8765    # streamable HTTP
"""

from __future__ import annotations

import argparse
import os

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from kbassist.service import AccessDenied, KnowledgeService

mcp = MCPServer(
    "kb-assistant",
    instructions=(
        "Search and read the company's indexed GitHub repositories (docs and code). "
        "Prefer search_knowledge_base, then read_file around the best hits; cite "
        "results with the returned GitHub permalinks."
    ),
)
_svc: KnowledgeService | None = None
READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def svc() -> KnowledgeService:
    global _svc
    if _svc is None:
        _svc = KnowledgeService()
    return _svc


def _user() -> str:
    return os.environ.get("KB_MCP_USER", "mcp-local")


@mcp.tool(annotations=READ_ONLY)
def search_knowledge_base(query: str, k: int = 6) -> list[dict]:
    """Hybrid search (BM25 + embeddings + reranker) over indexed repository docs and code.

    Use for any question about how our code or documentation works. Returns the
    top-k chunks with path, line range, a GitHub permalink and the chunk text.
    Write the query the way the answer would be phrased, and include exact
    identifiers (function, class, config key names) when you know them.
    """
    k = max(1, min(k, 20))
    hits, trace = svc().search(query, _user(), k=k)
    return [
        {
            "rank": i,
            "path": h.chunk.path,
            "lines": f"{h.chunk.start_line}-{h.chunk.end_line}",
            "title": h.chunk.title,
            "kind": h.chunk.kind,
            "score": round(h.score, 4),
            "url": svc().permalink(h.chunk),
            "text": h.chunk.text,
        }
        for i, h in enumerate(hits, start=1)
    ]


@mcp.tool(annotations=READ_ONLY)
def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """Read an indexed file (or a line range of it) by repository-relative path.

    Use after search_knowledge_base to see surrounding context. Only files in
    the index that you have permission to read are available.
    """
    try:
        return svc().get_document(path, _user(), start_line, end_line)
    except AccessDenied:
        return f"Access denied: you do not have permission to read {path}."
    except FileNotFoundError:
        return f"Not found in the index: {path}"


@mcp.tool(annotations=READ_ONLY)
def ask_knowledge_base(question: str) -> dict:
    """Answer a question end-to-end with citations, grounded only in indexed sources.

    Returns `answerable: false` instead of guessing when the sources don't cover it.
    """
    answer, trace = svc().ask(question, _user(), channel="mcp")
    return {
        "answerable": answer.answerable,
        "answer": answer.text,
        "sources": [
            {"n": i, "path": h.chunk.path, "url": svc().permalink(h.chunk)}
            for i, h in enumerate(answer.hits, start=1)
            if i in answer.citations
        ],
        "trace": trace.to_dict(svc().pricing),
    }


@mcp.tool(annotations=READ_ONLY)
def list_sources() -> dict:
    """List indexed repositories, pinned commits and index statistics."""
    return svc().list_sources()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, help="serve streamable HTTP on this port instead of stdio")
    args = parser.parse_args()
    if args.http:
        mcp.run(transport="streamable-http", port=args.http)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
