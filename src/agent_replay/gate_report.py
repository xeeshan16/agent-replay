"""Gate reports: Markdown for PR comments and job summaries, JUnit XML for CI test views.

Both formats carry text from agent runs (tasks, tool names, scorecard messages), which is
untrusted. Markdown cells go through ``report.cell``; XML is built with ElementTree (which
escapes markup) after stripping the control characters XML 1.0 cannot represent at all, so a
hostile tool output can neither break the table nor produce a JUnit file CI refuses to parse.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from agent_replay.gate import CaseResult, GateReport, Outcome
from agent_replay.report import cell

_ICON = {
    Outcome.PASS: "pass",
    Outcome.CHANGED: "changed",
    Outcome.INCONCLUSIVE: "inconclusive",
    Outcome.FAIL: "FAIL",
    Outcome.ERROR: "ERROR",
}
_XML_INVALID = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _xml_text(text: str) -> str:
    return _XML_INVALID.sub("�", text)


def _case_detail(case: CaseResult) -> str:
    worst = max(case.samples, key=lambda s: list(Outcome).index(s.outcome))
    if worst.error:
        return worst.error
    if worst.failures:
        return worst.failures[0]
    if worst.first_divergence:
        return f"diverged: {worst.first_divergence}"
    return ""


def render_markdown(report: GateReport) -> str:
    counts = report.outcome_counts()
    verdict = "PASSED" if report.passed else "BREACHED"
    rate, wil, delta = report.success_rate, report.success_rate_wilson, report.success_delta
    pct = f"{rate.confidence:.0%}"
    lines = [
        f"### Agent gate: {verdict}",
        "",
        f"Suite `{cell(report.suite, 60)}`, candidate `{cell(report.candidate, 60)}`, "
        f"{len(report.cases)} cases x {report.repeats} repeat(s). "
        f"Replay: {cell(report.replay, 100)}. Provider: {cell(report.provider, 100)}.",
        "",
        "| Outcome | pass | changed | inconclusive | fail | error |",
        "|---------|------|---------|--------------|------|-------|",
        "| Cases | " + " | ".join(str(counts[o.value]) for o in Outcome) + " |",
        "",
        f"Success rate {rate.estimate:.3f} (baseline {report.baseline_success_rate:.3f}); "
        f"{pct} CI bootstrap [{rate.low:.3f}, {rate.high:.3f}], Wilson [{wil.low:.3f}, "
        f"{wil.high:.3f}]. Change vs baseline {delta.estimate:+.3f}, {pct} CI "
        f"[{delta.low:+.3f}, {delta.high:+.3f}].",
        "",
        "| Threshold | Result | Value | Limit | Detail |",
        "|-----------|--------|-------|-------|--------|",
    ]
    for check in report.checks:
        lines.append(
            f"| {check.name} | {'ok' if check.passed else 'BREACH'} | {check.value} "
            f"| {check.limit} | {cell(check.message, 120)} |"
        )
    lines.extend(
        [
            "",
            "| Case | Outcome | Success | Detail |",
            "|------|---------|---------|--------|",
        ]
    )
    for case in report.cases:
        successes = sum(s.success for s in case.samples)
        lines.append(
            f"| {cell(case.id, 40)} | {_ICON[case.outcome]} | {successes}/{len(case.samples)} "
            f"| {cell(_case_detail(case), 100)} |"
        )
    review = report.cases_with(Outcome.CHANGED)
    if review:
        lines.extend(
            [
                "",
                f"**Changed, needs review ({len(review)}):** "
                + ", ".join(f"`{cell(c.id, 40)}`" for c in review)
                + ". The scorecard passed but behaviour differs from the baseline; "
                "run `agent-replay diff` on the case to see where.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def render_junit(report: GateReport) -> str:
    """One ``testsuite`` of cases (``fail`` -> failure, ``error`` -> error) and one of
    thresholds, so a CI test view shows both what broke and why the gate breached."""
    root = ET.Element("testsuites", name="agent-replay gate")
    cases_el = ET.SubElement(
        root,
        "testsuite",
        name=f"gate.{report.suite}",
        tests=str(len(report.cases)),
        failures=str(len(report.cases_with(Outcome.FAIL))),
        errors=str(len(report.cases_with(Outcome.ERROR))),
        skipped="0",
    )
    props = ET.SubElement(cases_el, "properties")
    for name, value in (
        ("candidate", report.candidate),
        ("provider", report.provider),
        ("replay", report.replay),
        ("repeats", str(report.repeats)),
        ("verdict", report.verdict),
    ):
        ET.SubElement(props, "property", name=name, value=_xml_text(value))
    for case in report.cases:
        test = ET.SubElement(
            cases_el, "testcase", classname=f"gate.{report.suite}", name=_xml_text(case.id)
        )
        detail = _xml_text(_case_detail(case))
        lines = [f"task: {case.task}"] + [
            f"repeat {s.repeat}: {s.outcome.value}"
            + (f" ({s.first_divergence})" if s.first_divergence else "")
            + "".join(f"\n  {f}" for f in s.failures)
            + (f"\n  {s.error}" if s.error else "")
            for s in case.samples
        ]
        body = _xml_text("\n".join(lines))
        if case.outcome is Outcome.FAIL:
            ET.SubElement(test, "failure", message=detail, type="scorecard").text = body
        elif case.outcome is Outcome.ERROR:
            ET.SubElement(test, "error", message=detail, type="harness").text = body
        else:
            note = "changed, needs review\n" if case.outcome is Outcome.CHANGED else ""
            ET.SubElement(test, "system-out").text = _xml_text(note) + body

    breaches = [c for c in report.checks if not c.passed]
    thresholds_el = ET.SubElement(
        root,
        "testsuite",
        name="gate.thresholds",
        tests=str(len(report.checks)),
        failures=str(len(breaches)),
        errors="0",
        skipped="0",
    )
    for check in report.checks:
        test = ET.SubElement(
            thresholds_el, "testcase", classname="gate.thresholds", name=check.name
        )
        if not check.passed:
            ET.SubElement(test, "failure", message=_xml_text(check.message), type="threshold")
        else:
            ET.SubElement(test, "system-out").text = _xml_text(check.message)
    ET.indent(root)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"
