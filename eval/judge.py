"""LLM-as-judge for groundedness (hallucination) and correctness.

The judge decomposes the answer into atomic factual claims and checks each one
against the exact sources the generator saw, so "hallucination" means
"not supported by the retrieved context" - measurable and independent of
whether the claim happens to be true in the world. Correctness is graded
separately against the hand-written reference facts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kbassist.generation import format_sources
from kbassist.llm import LLM
from kbassist.models import Hit
from kbassist.observability import Usage

JUDGE_SYSTEM = """\
You are a strict evaluator of a retrieval-augmented assistant. You get a question, \
reference facts written by a human, the numbered sources the assistant was given, \
and the assistant's answer.

1. Split the answer into atomic factual claims about the software (ignore \
pleasantries, hedges and statements like "the sources don't say X").
2. For each claim decide if it is SUPPORTED by the sources (explicitly stated or \
directly implied). Claims that are true in general but absent from the sources are \
UNSUPPORTED. Code snippets count as claims about the API they use.
3. Grade correctness against the reference facts: "correct" if the answer contains \
the key facts and nothing contradicting them, "partial" if some key facts are \
missing or slightly off, "incorrect" if wrong or missing the point. For a question \
whose reference says the assistant should decline, "correct" means it clearly said \
the information is not available and did not invent an answer.
4. abstained = true if the answer's main message is that it cannot answer from \
the available sources.
"""


class JudgeVerdict(BaseModel):
    claims: list[str] = Field(description="Atomic factual claims made by the answer.")
    unsupported_claims: list[str] = Field(description="Subset of claims not supported by the sources.")
    correctness: Literal["correct", "partial", "incorrect"]
    abstained: bool
    rationale: str = Field(description="One or two sentences.")


def judge(llm: LLM, question: str, reference: str, hits: list[Hit], answer: str) -> tuple[JudgeVerdict, Usage]:
    user = (
        f"<question>\n{question}\n</question>\n\n"
        f"<reference_facts>\n{reference}\n</reference_facts>\n\n"
        f"{format_sources(hits)}\n\n"
        f"<assistant_answer>\n{answer}\n</assistant_answer>"
    )
    result = llm.structured(JUDGE_SYSTEM, user, JudgeVerdict, effort="medium")
    if not isinstance(result.parsed, JudgeVerdict):
        raise RuntimeError(f"judge returned no verdict (stop_reason={result.stop_reason})")
    return result.parsed, result.usage
