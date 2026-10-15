"""Encode and decode content attributes of the OpenTelemetry GenAI semantic conventions.

Content capture is opt-in in the conventions (prompts and tool payloads can hold sensitive data),
so these attributes are only written when the recorder is asked to capture content. When they are
present, they carry everything needed to rebuild a replayable trajectory from spans alone:

* ``gen_ai.system_instructions``: system prompt as a list of parts.
* ``gen_ai.input.messages`` / ``gen_ai.output.messages``: chat history as role + parts, where a
  part is ``text``, ``tool_call`` or ``tool_call_response``.
* ``gen_ai.tool.definitions``: the tools offered to the model.
* ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``: tool payloads.

Two extensions are added where the conventions have no field and replay needs one: the tool name
on ``tool_call_response`` parts (``name``) and a ``side_effect`` flag on tool definitions. Readers
that do not know them ignore them.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import JsonValue

from agent_replay.canonical import canonical_json
from agent_replay.model import FinishReason, LLMResponse, Message, ToolCall, ToolSpec

# Conventions use singular "tool_call"; the internal model follows the OpenAI-style plural.
_FINISH_TO_SEMCONV = {
    FinishReason.STOP: "stop",
    FinishReason.TOOL_CALLS: "tool_call",
    FinishReason.LENGTH: "length",
}
_FINISH_FROM_SEMCONV = {
    "stop": FinishReason.STOP,
    "end_turn": FinishReason.STOP,
    "tool_call": FinishReason.TOOL_CALLS,
    "tool_calls": FinishReason.TOOL_CALLS,
    "tool_use": FinishReason.TOOL_CALLS,
    "length": FinishReason.LENGTH,
    "max_tokens": FinishReason.LENGTH,
}


class ContentDecodeError(ValueError):
    """A content attribute was present but not in the expected shape."""


def dumps(value: object) -> str:
    """Serialise an attribute payload. Canonical so equal content gives equal attributes."""
    return canonical_json(value)


def loads(raw: str, attribute: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        # Most often the SDK's attribute length limit truncated the value.
        raise ContentDecodeError(f"{attribute} is not valid JSON ({exc.msg})") from exc


# --- messages -------------------------------------------------------------------------------


def _message_parts(message: Message) -> list[dict[str, JsonValue]]:
    if message.role == "tool":
        part: dict[str, JsonValue] = {
            "type": "tool_call_response",
            "id": message.tool_call_id,
            "response": message.content,
        }
        if message.name is not None:
            part["name"] = message.name
        return [part]
    parts: list[dict[str, JsonValue]] = []
    if message.content:
        parts.append({"type": "text", "content": message.content})
    parts.extend(
        {"type": "tool_call", "id": c.id, "name": c.name, "arguments": c.arguments}
        for c in message.tool_calls
    )
    return parts


def encode_input(messages: Sequence[Message]) -> tuple[str | None, str]:
    """Split a request's history into (system instructions, input messages) attribute values."""
    system = [m for m in messages if m.role == "system"]
    rest = [m for m in messages if m.role != "system"]
    if len(system) > 1 or (system and messages[0].role != "system"):
        # Unusual layouts keep system messages inline so ordering is preserved exactly.
        system, rest = [], list(messages)
    instructions = (
        dumps([{"type": "text", "content": m.content} for m in system]) if system else None
    )
    history = dumps([{"role": m.role, "parts": _message_parts(m)} for m in rest])
    return instructions, history


def _text(parts: Sequence[dict[str, Any]]) -> str:
    return "".join(str(p.get("content", "")) for p in parts if p.get("type") == "text")


def _tool_calls(parts: Sequence[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls = []
    for p in parts:
        if p.get("type") != "tool_call":
            continue
        arguments = p.get("arguments") or {}
        if isinstance(arguments, str):  # some instrumentations send arguments as a JSON string
            arguments = loads(arguments, "tool_call.arguments")
        if not isinstance(arguments, dict):
            raise ContentDecodeError("tool_call.arguments must be an object")
        calls.append(ToolCall(id=str(p.get("id") or ""), name=str(p["name"]), arguments=arguments))
    return tuple(calls)


def _decode_message(raw: Any) -> Message:
    if not isinstance(raw, dict) or "role" not in raw:
        raise ContentDecodeError("message must be an object with a role")
    parts = raw.get("parts") or []
    if not isinstance(parts, list):
        raise ContentDecodeError("message parts must be a list")
    role = raw["role"]
    responses = [p for p in parts if isinstance(p, dict) and p.get("type") == "tool_call_response"]
    if role == "tool" or responses:
        if len(responses) != 1:
            raise ContentDecodeError("a tool message must carry exactly one tool_call_response")
        part = responses[0]
        response = part.get("response")
        return Message(
            role="tool",
            content=response if isinstance(response, str) else canonical_json(response),
            tool_call_id=part.get("id"),
            name=part.get("name"),
        )
    if role not in {"system", "user", "assistant"}:
        raise ContentDecodeError(f"unsupported message role {role!r}")
    return Message(role=role, content=_text(parts), tool_calls=_tool_calls(parts))


def decode_input(instructions: str | None, history: str) -> tuple[Message, ...]:
    messages: list[Message] = []
    if instructions is not None:
        parts = loads(instructions, "gen_ai.system_instructions")
        if not isinstance(parts, list):
            raise ContentDecodeError("gen_ai.system_instructions must be a list of parts")
        messages.append(Message(role="system", content=_text(parts)))
    items = loads(history, "gen_ai.input.messages")
    if not isinstance(items, list):
        raise ContentDecodeError("gen_ai.input.messages must be a list")
    messages.extend(_decode_message(item) for item in items)
    return tuple(messages)


def encode_output(response: LLMResponse) -> str:
    message = Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
    return dumps(
        [
            {
                "role": "assistant",
                "parts": _message_parts(message),
                "finish_reason": _FINISH_TO_SEMCONV[response.finish_reason],
            }
        ]
    )


def decode_output(raw: str) -> tuple[str, tuple[ToolCall, ...], FinishReason | None]:
    """Return (text, tool calls, finish reason) of the first output choice."""
    items = loads(raw, "gen_ai.output.messages")
    if not isinstance(items, list):
        raise ContentDecodeError("gen_ai.output.messages must be a list")
    if not items:
        return "", (), None
    first = items[0]
    if not isinstance(first, dict):
        raise ContentDecodeError("output message must be an object")
    parts = first.get("parts") or []
    return _text(parts), _tool_calls(parts), parse_finish_reason(first.get("finish_reason"))


def parse_finish_reason(value: object) -> FinishReason | None:
    return _FINISH_FROM_SEMCONV.get(str(value)) if value is not None else None


def text_message(role: str, text: str) -> str:
    """A single-message attribute value, used for an agent's task input and final answer."""
    parts = [{"type": "text", "content": text}] if text else []
    return dumps([{"role": role, "parts": parts}])


def first_text(raw: str, attribute: str) -> str | None:
    items = loads(raw, attribute)
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return None
    return _text(items[0].get("parts") or [])


# --- tools ----------------------------------------------------------------------------------


def encode_tool_definitions(specs: Sequence[ToolSpec]) -> str:
    return dumps(
        [
            {
                "type": "function",
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "side_effect": s.side_effect,
            }
            for s in specs
        ]
    )


def decode_tool_definitions(raw: str) -> tuple[ToolSpec, ...]:
    items = loads(raw, "gen_ai.tool.definitions")
    if not isinstance(items, list):
        raise ContentDecodeError("gen_ai.tool.definitions must be a list")
    specs = []
    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            raise ContentDecodeError("tool definition must be an object with a name")
        specs.append(
            ToolSpec(
                name=str(item["name"]),
                description=str(item.get("description") or ""),
                parameters=item.get("parameters") or {},
                side_effect=bool(item.get("side_effect", False)),
            )
        )
    return tuple(specs)
