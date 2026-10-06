"""Append-only JSONL audit log: who asked what, what they were shown, what it cost.

The question is stored redacted; the answer is stored as a hash plus the cited
chunk ids, which is enough to reconstruct "which documents did user X see"
without keeping a second copy of potentially sensitive generated text.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from kbassist.security.pii import redact_text

_lock = threading.Lock()


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, *, user: str, question: str | None = None, answer: str | None = None, **fields: Any) -> None:
        record: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
            "event": event,
            "user": user,
        }
        if question is not None:
            record["question"] = redact_text(question)
        if answer is not None:
            record["answer_sha256"] = hashlib.sha256(answer.encode()).hexdigest()
            record["answer_chars"] = len(answer)
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
