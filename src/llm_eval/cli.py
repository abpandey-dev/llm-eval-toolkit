"""`llmeval run suite.yaml` - exits non-zero when the pass rate is below the threshold (CI gate)."""

from __future__ import annotations

import argparse
import sys

from llm_eval.providers import DEFAULT_MODEL, get_provider
from llm_eval.report import console, write_reports
from llm_eval.runner import load_suite, run_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llmeval", description="Run an LLM eval suite.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run a YAML suite")
    run.add_argument("suite", help="Path to suite YAML")
    run.add_argument("--provider", default="anthropic", choices=["anthropic", "mock"])
    run.add_argument("--model", default=DEFAULT_MODEL, help=f"Model id (default {DEFAULT_MODEL})")
    run.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    run.add_argument("--mock-file", help="Canned responses for --provider mock")
    run.add_argument("--tags", nargs="*", help="Only run cases with any of these tags")
    run.add_argument("--out", default="reports", help="Directory for results.json and report.md")
    run.add_argument("--min-pass-rate", type=float, default=1.0,
                     help="Fail (exit 1) below this pass rate, 0-1 (default 1.0)")
    args = parser.parse_args(argv)

    suite = load_suite(args.suite)
    provider = get_provider(args.provider, model=args.model, effort=args.effort, mock_file=args.mock_file)
    result = run_suite(suite, provider, tags=args.tags)

    print(console(result))
    for p in write_reports(result, args.out):
        print(f"wrote {p}")
    return 0 if result.pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
