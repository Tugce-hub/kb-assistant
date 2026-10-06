"""LLM backends behind one interface: structured (JSON-schema) output + usage.

* AnthropicLLM - Claude via the official SDK (`messages.parse`, effort, refusal fallbacks).
* OllamaLLM    - a local open-weights model via Ollama's REST API. Free and fully
                 offline: no repository content leaves the machine.

Pick with `llm.provider` in config/settings.yaml (or KB_LLM__PROVIDER=ollama).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from kbassist.config import Settings
from kbassist.observability import Usage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class LLMResult:
    parsed: BaseModel | None
    text: str
    usage: Usage
    model: str
    stop_reason: str | None


class LLM(Protocol):
    model: str

    def structured(self, system: str, user: str, schema: type[T], effort: str | None = None) -> LLMResult: ...


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 4000, default_effort: str = "low", client=None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.default_effort = default_effort

    def structured(self, system: str, user: str, schema: type[T], effort: str | None = None) -> LLMResult:
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": effort or self.default_effort},
            # On a safety-classifier refusal, re-run on Anthropic's recommended fallback model.
            extra_headers={"anthropic-beta": FALLBACK_BETA},
            extra_body={"fallbacks": "default"},
        )
        u = resp.usage
        usage = Usage(
            model=resp.model,
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        parsed = None if resp.stop_reason == "refusal" else resp.parsed_output
        return LLMResult(parsed, text, usage, resp.model, resp.stop_reason)


class OllamaLLM:
    def __init__(self, model: str, base_url: str = "http://localhost:11434", num_ctx: int = 8192,
                 max_tokens: int = 1500, timeout: float = 600.0, transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def structured(self, system: str, user: str, schema: type[T], effort: str | None = None) -> LLMResult:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            # Ollama constrains decoding to the JSON schema (grammar-based), so even a
            # small model returns parseable output.
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {
                "temperature": 0,
                # The default context window would silently truncate our sources.
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        resp = self.http.post("/api/chat", json=body)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        usage = Usage(model=self.model, input_tokens=data.get("prompt_eval_count", 0) or 0,
                      output_tokens=data.get("eval_count", 0) or 0)
        try:
            parsed = schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError):
            log.warning("ollama returned invalid JSON (done_reason=%s)", data.get("done_reason"))
            parsed = None
        return LLMResult(parsed, text, usage, self.model, data.get("done_reason"))


def make_llm(settings: Settings, role: str = "generator") -> LLM:
    """role: 'generator' or 'judge'."""
    provider = settings.get("llm.provider", "anthropic")
    cfg = settings.get(f"llm.{provider}", {})
    model = cfg["judge_model"] if role == "judge" else cfg["model"]
    if provider == "anthropic":
        return AnthropicLLM(model, settings.get("llm.max_tokens", 4000), cfg.get("effort", "low"))
    if provider == "ollama":
        return OllamaLLM(model, cfg.get("base_url", "http://localhost:11434"), cfg.get("num_ctx", 8192),
                         cfg.get("max_tokens", 1500))
    raise ValueError(f"unknown llm.provider: {provider}")
