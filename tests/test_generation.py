"""Generator + LLM backends against fakes (no API key, no Ollama needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx

from kbassist.generation import AnswerSchema, Generator
from kbassist.llm import AnthropicLLM, OllamaLLM
from kbassist.models import Chunk, Hit


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _response(parsed=None, stop_reason="end_turn", content=()):
    usage = SimpleNamespace(input_tokens=1200, output_tokens=150, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, content=list(content), usage=usage, model="claude-opus-5-5")


def _hits(n=3):
    return [Hit(Chunk(str(i), "s", f"docs/{i}.md", "doc", f"T{i}", 1, 5, f"text {i}"), 1.0) for i in range(n)]


def _claude(response):
    messages = FakeMessages(response)
    llm = AnthropicLLM("claude-opus-5-5", client=SimpleNamespace(messages=messages))
    return Generator(settings=None, llm=llm), messages


def test_anthropic_request_shape_and_citation_filtering():
    gen, messages = _claude(_response(AnswerSchema(answerable=True, answer="Use X [1][7]", citations=[1, 7, 1])))
    ans = gen.answer("q?", _hits())
    call = messages.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is AnswerSchema
    assert call["output_config"] == {"effort": "low"}
    assert call["extra_body"] == {"fallbacks": "default"}
    assert "thinking" not in call and "temperature" not in call
    assert '<source id="1" path="docs/0.md"' in call["messages"][0]["content"]
    assert ans.citations == [1]  # out-of-range 7 dropped, duplicates removed
    assert ans.usage.input_tokens == 1200


def test_refusal_is_not_treated_as_answer():
    gen, _ = _claude(_response(None, stop_reason="refusal"))
    ans = gen.answer("q?", _hits())
    assert not ans.answerable and ans.citations == []


def test_no_hits_skips_llm_call():
    gen, messages = _claude(_response(None))
    ans = gen.answer("q?", [])
    assert not ans.answerable and messages.calls == []


def _ollama(reply: str):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": reply}, "prompt_eval_count": 900,
                                         "eval_count": 80, "done_reason": "stop"})

    llm = OllamaLLM("qwen2.5:3b", transport=httpx.MockTransport(handler))
    return Generator(settings=None, llm=llm), seen


def test_ollama_sends_schema_and_context_window():
    gen, seen = _ollama(json.dumps({"answerable": True, "answer": "5 seconds [2]", "citations": [2]}))
    ans = gen.answer("default timeout?", _hits())
    body = seen["body"]
    assert body["format"]["properties"].keys() >= {"answerable", "answer", "citations"}
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0
    assert body["stream"] is False
    assert ans.answerable and ans.citations == [2] and ans.usage.output_tokens == 80


def test_ollama_invalid_json_degrades_gracefully():
    gen, _ = _ollama("not json")
    ans = gen.answer("q?", _hits())
    assert not ans.answerable and ans.text == "not json"
