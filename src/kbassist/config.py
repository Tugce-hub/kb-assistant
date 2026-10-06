"""Settings loaded from config/settings.yaml with KB_* environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    ref: str
    include: list[str]
    exclude: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any]
    sources: list[Source]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        p = Path(self.get(dotted))
        return p if p.is_absolute() else ROOT / p


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    # KB_LLM__MODEL=claude-sonnet-5-5  ->  raw["llm"]["model"] = "claude-sonnet-5-5"
    for key, value in os.environ.items():
        if not key.startswith("KB_") or "__" not in key:
            continue
        parts = key[3:].lower().split("__")
        node = raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    path = Path(os.environ.get("KB_CONFIG", ROOT / "config" / "settings.yaml"))
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _apply_env_overrides(raw)
    sources = [Source(**s) for s in raw.get("sources", [])]
    return Settings(raw=raw, sources=sources)
