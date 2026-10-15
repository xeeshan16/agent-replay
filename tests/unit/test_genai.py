from __future__ import annotations

import pytest

from agent_replay import genai
from agent_replay.model import FinishReason, LLMResponse, Message, ToolCall, ToolSpec

HISTORY = (
    Message(role="system", content="Be helpful."),
    Message(role="user", content="Refund ORD-1001"),
    Message(
        role="assistant",
        content="Looking it up.",
        tool_calls=(ToolCall(id="c1", name="lookup_order", arguments={"order_id": "ORD-1001"}),),
    ),
    Message(role="tool", content='{"status":"delivered"}', tool_call_id="c1", name="lookup_order"),
)


def test_input_round_trip_splits_out_system_instructions() -> None:
    instructions, history = genai.encode_input(HISTORY)
    assert instructions is not None
    assert "system" not in history
    assert genai.decode_input(instructions, history) == HISTORY


def test_system_message_not_first_stays_inline() -> None:
    messages = (Message(role="user", content="hi"), Message(role="system", content="late"))
    instructions, history = genai.encode_input(messages)
    assert instructions is None
    assert genai.decode_input(None, history) == messages


def test_output_round_trip_maps_finish_reason_to_semconv() -> None:
    response = LLMResponse(
        model="m",
        tool_calls=(ToolCall(id="c2", name="issue_refund", arguments={"amount": 1.5}),),
        finish_reason=FinishReason.TOOL_CALLS,
    )
    raw = genai.encode_output(response)
    assert '"finish_reason":"tool_call"' in raw
    content, calls, finish = genai.decode_output(raw)
    assert (content, calls, finish) == ("", response.tool_calls, FinishReason.TOOL_CALLS)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("stop", FinishReason.STOP),
        ("end_turn", FinishReason.STOP),
        ("tool_use", FinishReason.TOOL_CALLS),
        ("max_tokens", FinishReason.LENGTH),
        ("content_filter", None),
        (None, None),
    ],
)
def test_finish_reasons_from_other_providers(value: str | None, expected: object) -> None:
    assert genai.parse_finish_reason(value) == expected


def test_tool_call_arguments_may_be_a_json_string() -> None:
    raw = (
        '[{"role":"assistant","parts":[{"type":"tool_call","id":"x","name":"t",'
        '"arguments":"{\\"a\\": 1}"}]}]'
    )
    _, calls, _ = genai.decode_output(raw)
    assert calls == (ToolCall(id="x", name="t", arguments={"a": 1}),)


def test_tool_definitions_round_trip_with_side_effect_flag() -> None:
    specs = (
        ToolSpec(name="read", description="r", parameters={"type": "object"}),
        ToolSpec(name="write", description="w", side_effect=True),
    )
    assert genai.decode_tool_definitions(genai.encode_tool_definitions(specs)) == specs


@pytest.mark.parametrize(
    "raw",
    [
        '[{"role":"user","parts":[{"type":"text","content":"trunc',  # truncated by a length limit
        '{"role":"user"}',  # not a list
        '[{"parts":[]}]',  # no role
        '[{"role":"robot","parts":[]}]',
    ],
)
def test_malformed_input_messages_raise(raw: str) -> None:
    with pytest.raises(genai.ContentDecodeError):
        genai.decode_input(None, raw)


def test_text_message_helpers() -> None:
    raw = genai.text_message("user", "hello")
    assert genai.first_text(raw, "x") == "hello"
    assert genai.first_text(genai.text_message("user", ""), "x") == ""
