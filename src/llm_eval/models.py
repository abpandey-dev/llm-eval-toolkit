"""Data models for suites, test cases and results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Assertion(BaseModel):
    """One check applied to a model output, e.g. {"type": "contains", "value": "refund"}."""

    type: str
    value: Any = None
    # Optional extras used by some assertion types (e.g. case_sensitive, rubric).
    options: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class

    id: str
    prompt: str
    system: str | None = None
    tags: list[str] = Field(default_factory=list)
    assertions: list[Assertion] = Field(default_factory=list)


class Suite(BaseModel):
    name: str
    description: str = ""
    system: str | None = None  # default system prompt for every case
    cases: list[TestCase]


class Completion(BaseModel):
    """What a provider returns for one prompt."""

    text: str
    latency_ms: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    refused: bool = False
    refusal_category: str | None = None


class AssertionResult(BaseModel):
    type: str
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    id: str
    tags: list[str]
    prompt: str
    output: str
    passed: bool
    latency_ms: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    assertions: list[AssertionResult] = Field(default_factory=list)


class SuiteResult(BaseModel):
    suite: str
    provider: str
    cases: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.cases)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0
