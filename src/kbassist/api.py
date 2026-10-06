"""HTTP API (FastAPI) for other internal services and for k8s health checks.

Identity comes from the `X-User` header, which in production is set by the
authenticating gateway / service mesh, never by the end client.

    uvicorn kbassist.api:app --port 8080
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kbassist.retrieval import Mode
from kbassist.service import AccessDenied, KnowledgeService

svc = KnowledgeService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the index and ONNX models before the pod reports ready; the first
    # query would otherwise pay ~20 s of model loading.
    svc.search("warmup", "eval-runner", k=1)
    yield


app = FastAPI(title="kb-assistant", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    k: int = 6
    mode: Mode = "hybrid_rerank"


class AskRequest(BaseModel):
    question: str


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.get("/readyz")
def readyz() -> dict:
    return {"ok": True, "chunks": len(svc.index.chunks), "commits": svc.index.manifest["commits"]}


@app.post("/search")
def search(req: SearchRequest, x_user: str = Header(...)) -> dict:
    hits, trace = svc.search(req.query, x_user, k=min(req.k, 20), mode=req.mode)
    return {
        "hits": [
            {"path": h.chunk.path, "lines": [h.chunk.start_line, h.chunk.end_line], "title": h.chunk.title,
             "score": h.score, "url": svc.permalink(h.chunk), "text": h.chunk.text}
            for h in hits
        ],
        "trace": trace.to_dict(),
    }


@app.post("/ask")
def ask(req: AskRequest, x_user: str = Header(...)) -> dict:
    answer, trace = svc.ask(req.question, x_user, channel="api")
    return {
        "answerable": answer.answerable,
        "answer": answer.text,
        "sources": [svc.permalink(h.chunk) for h in answer.cited_hits],
        "trace": trace.to_dict(svc.pricing),
    }


@app.get("/documents/{path:path}")
def document(path: str, x_user: str = Header(...), start_line: int = 1, end_line: int | None = None) -> dict:
    try:
        return {"path": path, "text": svc.get_document(path, x_user, start_line, end_line)}
    except AccessDenied:
        raise HTTPException(403, "forbidden")
    except FileNotFoundError:
        raise HTTPException(404, "not indexed")
