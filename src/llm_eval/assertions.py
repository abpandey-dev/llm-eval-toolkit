"""Assertion library. Each check takes the completion and returns an AssertionResult.

Deterministic checks (contains, regex, json, length, latency) are cheap and run first in spirit;
`llm_judge` uses a model to grade fuzzy qualities (helpfulness, tone, correctness against a rubric).
"""

from __future__ import annotations

import json
import re
from typing import Callable

from llm_eval.models import Assertion, AssertionResult, Completion
from llm_eval.providers import Provider

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "description": "1 (fails the rubric) to 5 (fully meets it)"},
        "passed": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["score", "passed", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You are a strict QA reviewer grading an AI assistant's answer against a rubric. "
    "Judge only what the rubric asks. Score 1-5; passed is true only if score >= the threshold given."
)


def _norm(text: str, a: Assertion) -> tuple[str, str]:
    value = str(a.value)
    if a.options.get("case_sensitive", False):
        return text, value
    return text.lower(), value.lower()


def check_contains(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    values = a.value if isinstance(a.value, list) else [a.value]
    text = c.text if a.options.get("case_sensitive") else c.text.lower()
    missing = [v for v in values if (str(v) if a.options.get("case_sensitive") else str(v).lower()) not in text]
    return AssertionResult(type=a.type, passed=not missing,
                           detail="all present" if not missing else f"missing: {missing}")


def check_not_contains(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    values = a.value if isinstance(a.value, list) else [a.value]
    text = c.text if a.options.get("case_sensitive") else c.text.lower()
    found = [v for v in values if (str(v) if a.options.get("case_sensitive") else str(v).lower()) in text]
    return AssertionResult(type=a.type, passed=not found,
                           detail="none present" if not found else f"found forbidden: {found}")


def check_equals(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    text, value = _norm(c.text.strip(), a)
    return AssertionResult(type=a.type, passed=text == value.strip(), detail=f"expected {a.value!r}")


def check_regex(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    flags = 0 if a.options.get("case_sensitive") else re.IGNORECASE
    ok = re.search(str(a.value), c.text, flags) is not None
    return AssertionResult(type=a.type, passed=ok, detail=f"pattern {a.value!r} {'matched' if ok else 'not found'}")


def check_json_valid(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    try:
        data = json.loads(_strip_fences(c.text))
    except json.JSONDecodeError as e:
        return AssertionResult(type=a.type, passed=False, detail=f"invalid JSON: {e.msg}")
    required = a.value or []
    if required:
        if not isinstance(data, dict):
            return AssertionResult(type=a.type, passed=False, detail="JSON is not an object")
        missing = [k for k in required if k not in data]
        if missing:
            return AssertionResult(type=a.type, passed=False, detail=f"missing keys: {missing}")
    return AssertionResult(type=a.type, passed=True, detail="valid JSON")


def check_max_length(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    words = len(c.text.split())
    limit = int(a.value)
    return AssertionResult(type=a.type, passed=words <= limit, detail=f"{words} words (limit {limit})")


def check_max_latency_ms(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    limit = float(a.value)
    return AssertionResult(type=a.type, passed=c.latency_ms <= limit,
                           detail=f"{c.latency_ms:.0f} ms (limit {limit:.0f} ms)")


def check_not_refused(c: Completion, a: Assertion, _: Provider | None) -> AssertionResult:
    return AssertionResult(type=a.type, passed=not c.refused,
                           detail="answered" if not c.refused else f"refused ({c.refusal_category})")


def check_llm_judge(c: Completion, a: Assertion, provider: Provider | None) -> AssertionResult:
    if provider is None:
        return AssertionResult(type=a.type, passed=False, detail="no judge provider configured")
    threshold = int(a.options.get("threshold", 4))
    prompt = (
        f"Rubric:\n{a.value}\n\nPassing threshold: score >= {threshold}\n\n"
        f"Question asked:\n{a.options.get('_prompt', '')}\n\nAnswer to grade:\n<answer>\n{c.text}\n</answer>"
    )
    try:
        verdict = provider.complete_json(prompt, JUDGE_SCHEMA, system=JUDGE_SYSTEM)
    except Exception as e:  # judge failure should fail the assertion, not crash the run
        return AssertionResult(type=a.type, passed=False, detail=f"judge error: {e}")
    score = int(verdict.get("score", 0))
    passed = score >= threshold
    return AssertionResult(type=a.type, passed=passed,
                           detail=f"score {score}/5 (need {threshold}): {verdict.get('reason', '')}")


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()


CHECKS: dict[str, Callable[[Completion, Assertion, Provider | None], AssertionResult]] = {
    "contains": check_contains,
    "not_contains": check_not_contains,
    "equals": check_equals,
    "regex": check_regex,
    "json_valid": check_json_valid,
    "max_length": check_max_length,
    "max_latency_ms": check_max_latency_ms,
    "not_refused": check_not_refused,
    "llm_judge": check_llm_judge,
}


def evaluate(c: Completion, a: Assertion, judge: Provider | None = None) -> AssertionResult:
    check = CHECKS.get(a.type)
    if check is None:
        return AssertionResult(type=a.type, passed=False, detail=f"unknown assertion type {a.type!r}")
    return check(c, a, judge)
