"""Console, JSON and Markdown reports."""

from __future__ import annotations

from pathlib import Path

from llm_eval.models import SuiteResult


def console(result: SuiteResult) -> str:
    lines = [f"\nSuite: {result.suite}  (provider: {result.provider})", "-" * 60]
    for c in result.cases:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"[{mark}] {c.id:<32} {c.latency_ms:>7.0f} ms")
        if c.error:
            lines.append(f"       error: {c.error}")
        for a in c.assertions:
            if not a.passed:
                lines.append(f"       x {a.type}: {a.detail}")
    lines.append("-" * 60)
    lines.append(f"{result.passed}/{result.total} passed ({result.pass_rate:.0%})")
    return "\n".join(lines)


def markdown(result: SuiteResult) -> str:
    out = [
        f"# Eval report: {result.suite}",
        "",
        f"**Provider:** {result.provider} · **Pass rate:** {result.passed}/{result.total} ({result.pass_rate:.0%})",
        "",
        "| Case | Result | Latency (ms) | Failed checks |",
        "|---|---|---|---|",
    ]
    for c in result.cases:
        failed = "; ".join(f"{a.type}: {a.detail}" for a in c.assertions if not a.passed) or (c.error or "")
        out.append(f"| {c.id} | {'✅' if c.passed else '❌'} | {c.latency_ms:.0f} | {failed.replace('|', '/')} |")
    return "\n".join(out) + "\n"


def write_reports(result: SuiteResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "results.json"
    md_path = out / "report.md"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    return [json_path, md_path]
