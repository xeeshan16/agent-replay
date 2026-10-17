from __future__ import annotations

from datetime import UTC, datetime

from agent_replay.diff import diff_trajectories
from agent_replay.model import (
    LLMRequest,
    LLMResponse,
    LLMStep,
    Message,
    RunStatus,
    ToolCall,
    ToolResult,
    ToolStep,
    Trajectory,
)
from agent_replay.report import cell, render_markdown

T0 = datetime(2026, 10, 17, tzinfo=UTC)


def _run(output: str, answer: str) -> Trajectory:
    call = ToolCall(id="c1", name="search", arguments={"q": "a|b"})
    request = LLMRequest(model="m", messages=(Message(role="user", content="q"),))
    return Trajectory(
        trajectory_id="t",
        agent_name="a",
        task_input="q",
        final_output=answer,
        status=RunStatus.SUCCESS,
        started_at=T0,
        steps=(
            LLMStep(
                index=0,
                started_at=T0,
                request=request,
                response=LLMResponse(model="m", tool_calls=(call,)),
            ),
            ToolStep(
                index=1,
                started_at=T0,
                call=call,
                result=ToolResult(tool_call_id="c1", name="search", output=output),
            ),
            LLMStep(
                index=2,
                started_at=T0,
                request=request,
                response=LLMResponse(model="m", content=answer),
            ),
        ),
    )


def test_cell_escapes_markup_and_bounds_length() -> None:
    assert cell("a | b\nc") == "a \\| b c"
    assert cell("<script>") == "&lt;script&gt;"
    assert cell("`x`") == "'x'"
    assert len(cell("x" * 500, 20)) == 20


def test_untrusted_tool_output_cannot_break_the_table() -> None:
    hostile = "| injected | row |\n\n# heading <img src=x>"
    md = render_markdown(diff_trajectories(_run("ok", "fine"), _run(hostile, "| also |")))
    table_rows = [line for line in md.splitlines() if line.startswith("| ")]
    for row in table_rows:
        unescaped = row.replace("\\|", "")
        assert unescaped.count("|") in {4, 7}, row  # outcome table or alignment table
    assert "<img" not in md
    assert "\n# heading" not in md


def test_identical_render() -> None:
    md = render_markdown(diff_trajectories(_run("ok", "fine"), _run("ok", "fine")), title="T")
    assert md.startswith("### T")
    assert "**Verdict:** identical" in md
    assert "First divergence" not in md
