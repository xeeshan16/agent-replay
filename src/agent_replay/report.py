"""Markdown rendering of a ``TrajectoryDiff`` for terminals, PR comments and CI summaries.

The output is GitHub-flavoured Markdown: a verdict line, the first divergence with its decision
step, the aligned action table and the outcome. Cell content is escaped and truncated so that
untrusted tool output cannot break the table or inject markup into a PR comment.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import JsonValue

from agent_replay.canonical import canonical_json
from agent_replay.diff import Action, AlignedPair, Change, DiffOp, Divergence, TrajectoryDiff

_OP_MARK = {DiffOp.MATCH: "=", DiffOp.SUBSTITUTE: "~", DiffOp.DELETE: "-", DiffOp.INSERT: "+"}
_CELL = 90


def cell(text: str, width: int = _CELL) -> str:
    """Escape ``text`` for a Markdown table cell: one line, no pipes, no HTML, bounded length."""
    flat = " ".join(text.split())
    if len(flat) > width:
        flat = flat[: width - 1] + "…"
    for char, escaped in (("\\", "\\\\"), ("|", "\\|"), ("<", "&lt;"), (">", "&gt;"), ("`", "'")):
        flat = flat.replace(char, escaped)
    return flat


def _value(value: JsonValue) -> str:
    return value if isinstance(value, str) else canonical_json(value)


def _change(change: Change) -> str:
    if change.kind == "added":
        return f"{change.path}: + {_value(change.after)}"
    if change.kind == "removed":
        return f"{change.path}: - {_value(change.before)}"
    return f"{change.path}: {_value(change.before)} → {_value(change.after)}"


def _action(action: Action | None) -> str:
    if action is None:
        return ""
    step = "—" if action.step_index is None else str(action.step_index)
    flags = []
    if action.side_effect:
        flags.append("write")
    if not action.executed:
        flags.append("not executed")
    elif action.is_error:
        flags.append("error")
    if action.served is not None and action.served.value not in {"exact", "normalized"}:
        flags.append(action.served.value)
    # Flags first: they matter more than the arguments when a long cell is truncated.
    prefix = f"[{', '.join(flags)}] " if flags else ""
    return f"{prefix}{step}: {action.label()}"


def _row(index: int, pair: AlignedPair) -> str:
    detail = "; ".join(_change(c) for c in pair.changes[:3])
    if len(pair.changes) > 3:  # noqa: PLR2004 - three changes fit a cell
        detail += f"; +{len(pair.changes) - 3} more"
    cells = (
        str(index),
        _OP_MARK[pair.op],
        cell(_action(pair.baseline)),
        cell(_action(pair.candidate)),
        pair.reason.value if pair.reason else "",
        cell(detail),
    )
    return "| " + " | ".join(cells) + " |"


def _first(divergence: Divergence) -> Iterable[str]:
    where = []
    if divergence.baseline_step is not None:
        where.append(f"baseline step {divergence.baseline_step}")
    if divergence.candidate_step is not None:
        where.append(f"candidate step {divergence.candidate_step}")
    if divergence.decision_step is not None:
        where.append(f"decided at candidate model step {divergence.decision_step}")
    yield (
        f"**First divergence:** `{divergence.reason.value}`: {cell(divergence.summary, 200)}"
        + (f" ({', '.join(where)})" if where else "")
    )
    if divergence.side_effect:
        yield ""
        yield "> **Involves a side-effecting tool.**"
    for change in divergence.changes[:5]:
        yield f"- {cell(_change(change), 200)}"


def render_markdown(diff: TrajectoryDiff, *, title: str | None = None) -> str:
    lines: list[str] = [f"### {cell(title or 'Trajectory diff', 120)}", ""]
    lines.append(f"`{cell(diff.baseline_id, 80)}` → `{cell(diff.candidate_id, 80)}`")
    lines.append("")
    verdict = "identical" if diff.identical else f"diverged ({len(diff.divergences)} differences)"
    lines.append(
        f"**Verdict:** {verdict} · similarity {diff.similarity:.3f} · "
        f"edit distance {diff.distance:.3f}"
    )
    first = diff.first_divergence
    if first is not None:
        lines.append("")
        lines.extend(_first(first))
        reasons = ", ".join(f"{k} x{v}" for k, v in diff.reason_counts().items())
        lines.extend(["", f"**All differences:** {reasons}"])

    lines.extend(
        [
            "",
            "| # | Op | Baseline | Candidate | Reason | Detail |",
            "|---|----|----------|-----------|--------|--------|",
        ]
    )
    if diff.alignment:
        lines.extend(_row(i, pair) for i, pair in enumerate(diff.alignment))
    else:
        lines.append("| | | (no tool calls) | (no tool calls) | | |")

    out, usage = diff.outcome, diff.usage
    lines.extend(
        [
            "",
            "| Outcome | Baseline | Candidate |",
            "|---------|----------|-----------|",
            f"| Status | {out.baseline_status} | {out.candidate_status} |",
            f"| Answer | {cell(out.baseline_output or '')} | {cell(out.candidate_output or '')} |",
            "",
            (
                f"Candidate vs baseline: {usage.steps:+d} steps, {usage.tool_calls:+d} tool calls, "
                f"{usage.input_tokens:+d} input / {usage.output_tokens:+d} output tokens, "
                f"{usage.cost_usd:+.6f} USD."
            ),
        ]
    )
    if out.candidate_error:
        lines.extend(["", f"Candidate error: `{cell(out.candidate_error, 200)}`"])
    lines.extend(["", f"Normaliser: `{cell(', '.join(diff.normalizer) or 'none', 200)}`", ""])
    return "\n".join(lines)
