"""Permission-aware retrieval.

ACLs are enforced *inside* retrieval (chunks the caller cannot see are masked
before ranking), never by asking the LLM to ignore them. A restricted chunk
therefore can't leak through the answer, through citations, or through the
MCP `search` tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from kbassist.globs import match


@dataclass(frozen=True)
class Principal:
    id: str
    groups: frozenset[str]


class ACL:
    def __init__(self, rules: list[dict], default_groups: list[str], users: dict[str, list[str]]):
        self.rules = [(r["pattern"], frozenset(r["groups"])) for r in rules]
        self.default_groups = frozenset(default_groups)
        self.users = {k: frozenset(v) for k, v in users.items()}

    @classmethod
    def from_file(cls, path: Path) -> "ACL":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(data.get("rules", []), data.get("default_groups", []), data.get("users", {}))

    def principal(self, user_id: str) -> Principal:
        groups = self.users.get(user_id, self.users.get("*", frozenset()))
        return Principal(user_id, groups)

    def required_groups(self, path: str) -> frozenset[str]:
        for pattern, groups in self.rules:
            if match(path, pattern):
                return groups
        return self.default_groups

    def can_read(self, principal: Principal, path: str) -> bool:
        return bool(self.required_groups(path) & principal.groups)

    def mask(self, paths: list[str], principal: Principal) -> np.ndarray:
        """Boolean visibility mask aligned with `paths` (the index's chunk order)."""
        return np.fromiter((self.can_read(principal, p) for p in paths), dtype=bool, count=len(paths))
