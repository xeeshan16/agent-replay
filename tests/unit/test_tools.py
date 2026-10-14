from __future__ import annotations

import asyncio

import pytest
from pydantic import JsonValue

from agent_replay.model import ToolCall
from agent_replay.tools import Toolbox


def _call(name: str, **arguments: JsonValue) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


async def test_sync_and_async_tools_return_results() -> None:
    box = Toolbox()

    def add(a: int, b: int) -> JsonValue:
        """Add two numbers."""
        return a + b

    async def echo(text: str) -> JsonValue:
        return text

    box.add(add)
    box.add(echo, side_effect=True)
    assert (await box.execute(_call("add", a=1, b=2))).output == 3
    assert (await box.execute(_call("echo", text="hi"))).output == "hi"
    assert [s.name for s in box.specs()] == ["add", "echo"]
    spec = box.spec("add")
    assert spec is not None
    assert spec.description == "Add two numbers."
    echo_spec = box.spec("echo")
    assert echo_spec is not None
    assert echo_spec.side_effect


async def test_tool_exceptions_become_error_results() -> None:
    box = Toolbox()

    def boom() -> JsonValue:
        raise LookupError("missing")

    box.add(boom)
    result = await box.execute(_call("boom"))
    assert result.is_error
    assert result.error_type == "LookupError"
    assert result.output == "missing"
    assert result.tool_call_id == "c1"


async def test_unknown_tool_and_timeout_are_errors() -> None:
    box = Toolbox(timeout_s=0.01)

    async def slow() -> JsonValue:
        await asyncio.sleep(1)
        return None

    box.add(slow)
    unknown = await box.execute(_call("nope"))
    assert unknown.is_error
    assert unknown.error_type == "ToolNotFoundError"
    timed_out = await box.execute(_call("slow"))
    assert timed_out.error_type == "TimeoutError"


def test_duplicate_registration_is_rejected() -> None:
    box = Toolbox()

    def f() -> JsonValue:
        return None

    box.add(f)
    with pytest.raises(ValueError, match="already registered"):
        box.add(f)
