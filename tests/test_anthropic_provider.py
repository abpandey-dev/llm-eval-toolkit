"""AnthropicProvider tests with a fake client - verifies request shape and response parsing, no API calls."""

from types import SimpleNamespace as NS

from llm_eval.providers import FALLBACK_BETA, AnthropicProvider


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def make_provider(response, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    p = AnthropicProvider()
    fake = FakeMessages(response)
    p.client = NS(beta=NS(messages=fake))
    return p, fake


def response(text="", stop_reason="end_turn", stop_details=None):
    return NS(
        content=[NS(type="thinking", thinking=""), NS(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
        model="claude-opus-5-5",
        usage=NS(input_tokens=12, output_tokens=7),
    )


def test_complete_sends_model_effort_and_fallbacks(monkeypatch):
    p, fake = make_provider(response("EMI is an equated monthly instalment."), monkeypatch)
    c = p.complete("What is EMI?", system="be brief")
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "medium"}
    assert call["fallbacks"] == "default" and call["betas"] == [FALLBACK_BETA]
    assert call["system"] == "be brief"
    assert "thinking" not in call  # Opus 5.5 rejects disabled thinking; we leave it at the default
    assert c.text == "EMI is an equated monthly instalment." and not c.refused
    assert (c.input_tokens, c.output_tokens) == (12, 7)


def test_refusal_is_recorded_not_raised(monkeypatch):
    p, _ = make_provider(response("", "refusal", NS(category="cyber", explanation="x")), monkeypatch)
    c = p.complete("something")
    assert c.refused and c.refusal_category == "cyber"


def test_complete_json_uses_structured_output(monkeypatch):
    p, fake = make_provider(response('{"score": 5, "passed": true, "reason": "ok"}'), monkeypatch)
    out = p.complete_json("grade this", {"type": "object"})
    assert out["score"] == 5
    assert fake.calls[0]["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
