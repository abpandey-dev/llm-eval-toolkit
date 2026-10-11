from pathlib import Path

from llm_eval.cli import main
from llm_eval.models import Completion
from llm_eval.providers import MockProvider
from llm_eval.runner import load_suite, run_suite

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SUITE = EXAMPLES / "loan_support_suite.yaml"
MOCK = EXAMPLES / "mock_responses.yaml"


def test_example_suite_runs_with_one_expected_failure():
    result = run_suite(load_suite(SUITE), MockProvider.from_file(MOCK))
    assert result.total == 5
    failed = [c.id for c in result.cases if not c.passed]
    assert failed == ["penalty-tone"]


def test_tag_filter():
    result = run_suite(load_suite(SUITE), MockProvider.from_file(MOCK), tags=["smoke"])
    assert {c.id for c in result.cases} == {"emi-definition", "never-asks-for-otp"}


def test_provider_error_fails_case_not_run():
    class Boom(MockProvider):
        def complete(self, prompt, system=None) -> Completion:
            raise TimeoutError("upstream timeout")

    result = run_suite(load_suite(SUITE), Boom(), tags=["format"])
    assert result.cases[0].passed is False and "TimeoutError" in result.cases[0].error


def test_cli_exit_code_gates_on_pass_rate(tmp_path):
    args = ["run", str(SUITE), "--provider", "mock", "--mock-file", str(MOCK), "--out", str(tmp_path)]
    assert main(args) == 1  # 4/5 < default 100%
    assert main(args + ["--min-pass-rate", "0.8"]) == 0
    assert (tmp_path / "results.json").exists() and (tmp_path / "report.md").exists()
