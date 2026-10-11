"""Load a YAML suite and run every case through a provider, then grade it."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_eval.assertions import evaluate
from llm_eval.models import CaseResult, Suite, SuiteResult
from llm_eval.providers import Provider


def load_suite(path: str | Path) -> Suite:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Suite.model_validate(data)


def run_suite(suite: Suite, provider: Provider, judge: Provider | None = None,
              tags: list[str] | None = None) -> SuiteResult:
    """Run each case; a case passes only if every assertion passes. Errors fail the case, not the run."""
    judge = judge or provider
    results: list[CaseResult] = []
    for case in suite.cases:
        if tags and not set(tags) & set(case.tags):
            continue
        system = case.system or suite.system
        try:
            completion = provider.complete(case.prompt, system=system)
        except Exception as e:
            results.append(CaseResult(id=case.id, tags=case.tags, prompt=case.prompt, output="",
                                      passed=False, latency_ms=0, model="", error=f"{type(e).__name__}: {e}"))
            continue

        checks = []
        for a in case.assertions:
            if a.type == "llm_judge":
                a = a.model_copy(update={"options": {**a.options, "_prompt": case.prompt}})
            checks.append(evaluate(completion, a, judge))

        results.append(CaseResult(
            id=case.id, tags=case.tags, prompt=case.prompt, output=completion.text,
            passed=all(r.passed for r in checks), latency_ms=completion.latency_ms,
            model=completion.model, input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens, assertions=checks,
        ))
    return SuiteResult(suite=suite.name, provider=provider.name, cases=results)
