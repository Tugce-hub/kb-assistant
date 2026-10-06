"""Lightweight tracing and cost accounting.

Each request gets a Trace with named spans (ms) and token usage. The trace is
written to the audit log and returned by the API/MCP tools, and the eval
harness aggregates the same numbers, so offline and online metrics come from
one code path. Swapping in OpenTelemetry/Langfuse means exporting `Trace`.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Usage:
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.model = other.model or self.model
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens

    def cost_usd(self, pricing: dict[str, dict[str, float]]) -> float:
        p = pricing.get(self.model)
        if not p:
            return 0.0
        return (
            self.input_tokens * p["input"]
            + self.output_tokens * p["output"]
            + self.cache_read_input_tokens * p.get("cache_read", p["input"])
            + self.cache_creation_input_tokens * p.get("cache_write", p["input"])
        ) / 1_000_000


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    spans_ms: dict[str, float] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)

    @contextmanager
    def span(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.spans_ms[name] = self.spans_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1000

    def to_dict(self, pricing: dict[str, dict[str, float]] | None = None) -> dict:
        d = {
            "trace_id": self.trace_id,
            "spans_ms": {k: round(v, 1) for k, v in self.spans_ms.items()},
            "usage": vars(self.usage),
        }
        if pricing is not None:
            d["cost_usd"] = round(self.usage.cost_usd(pricing), 6)
        return d
