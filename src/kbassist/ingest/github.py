"""Fetch GitHub repositories at a pinned ref and enumerate indexable files."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from kbassist.config import Source
from kbassist.globs import match_any

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 400_000


def _git(*args: str, cwd: Path | None = None) -> str:
    env = dict(os.environ)
    # Private repos: GITHUB_TOKEN is passed via an auth header, never written into
    # the remote URL, so it does not end up in .git/config or in logs.
    token = env.get("GITHUB_TOKEN")
    extra: list[str] = []
    if token:
        extra = ["-c", f"http.extraHeader=Authorization: Bearer {token}"]
    result = subprocess.run(
        ["git", *extra, *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def sync_source(source: Source, repos_dir: Path) -> Path:
    """Clone (or update) `source` and check out its pinned ref. Returns the checkout path."""
    dest = repos_dir / source.name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("cloning %s", source.url)
        _git("clone", "--filter=blob:none", "--no-checkout", source.url, str(dest))
    current = _git("rev-parse", "HEAD", cwd=dest) if (dest / ".git" / "HEAD").exists() else ""
    if not current.startswith(source.ref):
        _git("fetch", "--depth", "1", "origin", source.ref, cwd=dest)
        _git("checkout", "--force", source.ref, cwd=dest)
    return dest


def resolved_commit(checkout: Path) -> str:
    return _git("rev-parse", "HEAD", cwd=checkout)


def iter_files(source: Source, checkout: Path):
    """Yield (relative_posix_path, text) for files selected by include/exclude globs."""
    for path in sorted(checkout.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(checkout).as_posix()
        if not match_any(rel, source.include) or match_any(rel, source.exclude):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            log.warning("skipping large file %s", rel)
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("skipping non-utf8 file %s", rel)
