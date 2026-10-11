"""LLM Eval Toolkit: test LLM apps the way an SDET tests software."""

from llm_eval.models import AssertionResult, CaseResult, Suite, TestCase
from llm_eval.runner import load_suite, run_suite

__all__ = ["AssertionResult", "CaseResult", "Suite", "TestCase", "load_suite", "run_suite"]
__version__ = "0.1.0"
