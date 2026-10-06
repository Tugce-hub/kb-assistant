"""Fast unit tests: no models, no network, no API key."""

from __future__ import annotations

import numpy as np

from kbassist.globs import match
from kbassist.index.bm25 import BM25Index, tokenize
from kbassist.ingest.chunking import ChunkParams, chunk_file
from kbassist.models import Chunk
from kbassist.retrieval import rrf
from kbassist.security.acl import ACL
from kbassist.security.pii import redact

P = ChunkParams(max_chars=400, min_chars=40, overlap_chars=60)


# ----------------------------------------------------------------- chunking


def test_markdown_breadcrumbs_and_fenced_hashes():
    md = (
        "# Timeouts\n\nIntro text that is long enough to stand alone as a section here.\n\n"
        "## Fine tuning\n\n```bash\n# this is a shell comment, not a heading\nexport X=1\n```\n"
        "More text about connect, read, write and pool timeouts in detail.\n"
    )
    chunks = chunk_file("s", "docs/t.md", md, P)
    titles = [c.title for c in chunks]
    assert "Timeouts > Fine tuning" in titles
    assert not any("shell comment" in t for t in titles)


def test_markdown_line_numbers_point_at_source():
    md = "\n\n# A\n\n" + "alpha line\n" * 10 + "# B\n\n" + "beta line\n" * 10
    lines = md.split("\n")
    for c in chunk_file("s", "x.md", md, P):
        first = c.text.split("\n")[0]
        assert lines[c.start_line - 1] == first


def test_python_chunks_by_symbol():
    src = (
        "import os\n\nDEFAULT = 5\n\n\n"
        "def helper(x):\n    return x + 1\n\n\n"
        "class Client:\n    def get(self):\n        return 1\n"
    )
    chunks = chunk_file("s", "pkg/mod.py", src, P)
    titles = {c.title for c in chunks}
    assert "pkg.mod.helper" in titles
    assert "pkg.mod.Client" in titles
    assert all(c.kind == "code" for c in chunks)


def test_oversized_chunks_are_split_with_overlap():
    md = "# Big\n\n" + "\n".join(f"line {i} " + "x" * 40 for i in range(60))
    chunks = chunk_file("s", "big.md", md, P)
    assert len(chunks) > 1
    assert all(len(c.text) <= P.max_chars + 60 for c in chunks)
    assert chunks[1].start_line <= chunks[0].end_line  # overlap


# ---------------------------------------------------------------------- bm25


def test_tokenizer_splits_identifiers_but_keeps_them_whole():
    toks = tokenize("max_keepalive_connections HTTPStatusError")
    assert "max_keepalive_connections" in toks
    assert {"keepalive", "connections", "status", "error", "httpstatuserror"} <= set(toks)


def test_bm25_ranks_exact_identifier_first():
    docs = ["timeouts are five seconds", "DEFAULT_MAX_REDIRECTS = 20", "redirect handling in general"]
    s = BM25Index().fit(docs).scores("DEFAULT_MAX_REDIRECTS")
    assert int(np.argmax(s)) == 1


# ----------------------------------------------------------------------- rrf


def test_rrf_rewards_agreement():
    fused = rrf({"bm25": [1, 2, 3], "vector": [2, 9, 1]}, k=60)
    assert fused[0][0] == 2  # ranked 2nd and 1st beats 1st and 3rd
    assert fused[0][2] == {"bm25": 2, "vector": 1}


# ----------------------------------------------------------------------- pii


def test_redacts_secrets_and_turkish_pii():
    text = (
        "token = 'ghp_" + "a" * 36 + "' and key AKIAABCDEFGHIJKLMNOP\n"
        "card 4111 1111 1111 1111, tckn 10000000146, iban TR330006100519786457841326\n"
        "mail ali.veli@paribu.com phone +90 532 123 45 67"
    )
    out, n = redact(text)
    for secret in ("ghp_", "AKIA", "4111", "10000000146", "TR3300", "ali.veli", "532 123"):
        assert secret not in out, secret
    assert n >= 7


def test_does_not_redact_ordinary_numbers_or_example_emails():
    text = "DEFAULT_MAX_REDIRECTS = 20; port 12345678901; user@example.com; 1234 5678 9012 3456"
    out, n = redact(text)
    assert out == text and n == 0


# ----------------------------------------------------------------------- acl


def _acl() -> ACL:
    return ACL(
        rules=[{"pattern": "docs/**", "groups": ["everyone"]}, {"pattern": "src/**", "groups": ["eng"]}],
        default_groups=["eng"],
        users={"*": ["everyone"], "U_ENG": ["everyone", "eng"]},
    )


def test_acl_first_match_and_default():
    acl = _acl()
    anon, eng = acl.principal("U_SOMEONE"), acl.principal("U_ENG")
    assert acl.can_read(anon, "docs/a/b.md")
    assert not acl.can_read(anon, "src/secret.py")
    assert not acl.can_read(anon, "ops/runbook.md")  # default_groups
    assert acl.can_read(eng, "src/secret.py")


def test_acl_mask_excludes_restricted_chunks_from_retrieval():
    from kbassist.index.store import Index
    from kbassist.retrieval import Retriever

    chunks = [
        Chunk("1", "s", "docs/public.md", "doc", "t", 1, 1, "redirect policy is public"),
        Chunk("2", "s", "src/secret.py", "code", "t", 1, 1, "redirect policy secret implementation"),
    ]
    index = Index(chunks, np.zeros((2, 4), dtype=np.float32), BM25Index().fit([c.search_text() for c in chunks]),
                  {"embedding_model": "none"})

    class S:
        def get(self, key, default=None):
            return default

    r = Retriever(index, S(), _acl())
    hits = r.search("redirect secret implementation", _acl().principal("anon"), mode="bm25", k=5)
    assert [h.chunk.path for h in hits] == ["docs/public.md"]


def test_glob_double_star():
    assert match("docs/advanced/ssl.md", "docs/**/*.md")
    assert match("docs/index.md", "docs/**/*.md")
    assert not match("httpx/docs.md", "docs/**/*.md")
