"""Providers turn a prompt into a Completion. Swap them to test different models or run offline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import anthropic
import yaml

from llm_eval.models import Completion

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, system: str | None = None) -> Completion: ...

    def complete_json(self, prompt: str, schema: dict, system: str | None = None) -> dict: ...


class AnthropicProvider:
    """Calls Claude through the official Anthropic SDK.

    Credentials come from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile).
    Server-side refusal fallbacks are enabled (`fallbacks="default"`), so a declined request is
    re-run on Anthropic's recommended fallback model instead of failing the case outright.
    """

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "medium", max_tokens: int = 16000):
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def _create(self, prompt: str, system: str | None, **extra):
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
            **extra,
        )
        if system:
            kwargs["system"] = system
        return self.client.beta.messages.create(**kwargs)

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        start = time.perf_counter()
        response = self._create(prompt, system, output_config={"effort": self.effort})
        latency_ms = (time.perf_counter() - start) * 1000

        refused = response.stop_reason == "refusal"
        category = response.stop_details.category if refused and response.stop_details else None
        text = "".join(b.text for b in response.content if b.type == "text")
        return Completion(
            text=text,
            latency_ms=latency_ms,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            refused=refused,
            refusal_category=category,
        )

    def complete_json(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        response = self._create(
            prompt,
            system,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("Judge request was refused")
        text = "".join(b.text for b in response.content if b.type == "text")
        return json.loads(text)


class MockProvider:
    """Returns canned outputs from a YAML/JSON file keyed by case prompt. No network, no cost.

    Used for unit tests, CI, and demoing the toolkit without an API key. Judge calls return the
    `judge` entry for the matching output, or a pass by default.
    """

    name = "mock"

    def __init__(self, responses: dict[str, str] | None = None, judge: dict[str, dict] | None = None,
                 model: str = "mock-model", latency_ms: float = 5.0):
        self.responses = responses or {}
        self.judge = judge or {}
        self.model = model
        self.latency_ms = latency_ms

    @classmethod
    def from_file(cls, path: str | Path) -> "MockProvider":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(responses=data.get("responses", {}), judge=data.get("judge", {}))

    def complete(self, prompt: str, system: str | None = None) -> Completion:
        text = self.responses.get(prompt, "")
        return Completion(text=text, latency_ms=self.latency_ms, model=self.model,
                          input_tokens=len(prompt.split()), output_tokens=len(text.split()))

    def complete_json(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        for marker, verdict in self.judge.items():
            if marker in prompt:
                return verdict
        return {"score": 5, "passed": True, "reason": "mock judge default pass"}


def get_provider(name: str, **kwargs) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(**{k: v for k, v in kwargs.items() if k in ("model", "effort") and v})
    if name == "mock":
        path = kwargs.get("mock_file")
        return MockProvider.from_file(path) if path else MockProvider()
    raise ValueError(f"Unknown provider: {name!r} (expected 'anthropic' or 'mock')")
