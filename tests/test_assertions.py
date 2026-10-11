from llm_eval.assertions import evaluate
from llm_eval.models import Assertion, Completion
from llm_eval.providers import MockProvider


def comp(text: str, latency_ms: float = 100, refused: bool = False) -> Completion:
    return Completion(text=text, latency_ms=latency_ms, model="m", refused=refused)


def test_contains_is_case_insensitive_by_default():
    assert evaluate(comp("Equated Monthly Instalment"), Assertion(type="contains", value="equated monthly")).passed


def test_contains_reports_missing_values():
    r = evaluate(comp("hello world"), Assertion(type="contains", value=["hello", "refund"]))
    assert not r.passed and "refund" in r.detail


def test_not_contains_flags_forbidden_text():
    r = evaluate(comp("Please share your OTP"), Assertion(type="not_contains", value=["share your otp"]))
    assert not r.passed


def test_regex():
    assert evaluate(comp("Ticket #12345 created"), Assertion(type="regex", value=r"#\d{5}")).passed


def test_json_valid_with_required_keys_and_code_fences():
    text = '```json\n{"can_prepay": true, "next_step": "x"}\n```'
    assert evaluate(comp(text), Assertion(type="json_valid", value=["can_prepay", "next_step"])).passed


def test_json_valid_missing_key_fails():
    r = evaluate(comp('{"can_prepay": true}'), Assertion(type="json_valid", value=["next_step"]))
    assert not r.passed and "next_step" in r.detail


def test_json_invalid_fails():
    assert not evaluate(comp("not json"), Assertion(type="json_valid")).passed


def test_max_length_counts_words():
    assert evaluate(comp("one two three"), Assertion(type="max_length", value=3)).passed
    assert not evaluate(comp("one two three four"), Assertion(type="max_length", value=3)).passed


def test_max_latency():
    assert not evaluate(comp("x", latency_ms=900), Assertion(type="max_latency_ms", value=500)).passed


def test_not_refused():
    assert not evaluate(comp("", refused=True), Assertion(type="not_refused")).passed


def test_llm_judge_uses_threshold():
    judge = MockProvider(judge={"rude": {"score": 2, "passed": False, "reason": "rude"}})
    bad = evaluate(comp("that was rude"), Assertion(type="llm_judge", value="be polite"), judge)
    good = evaluate(comp("happy to help"), Assertion(type="llm_judge", value="be polite"), judge)
    assert not bad.passed and good.passed


def test_unknown_assertion_fails_loudly():
    r = evaluate(comp("x"), Assertion(type="does_not_exist"))
    assert not r.passed and "unknown" in r.detail
