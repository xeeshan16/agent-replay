"""The trajectory data model.

A *trajectory* is the ordered list of steps one agent run took: every model call (request and
response) and every tool call (arguments and result). It is the unit that is recorded, stored,
replayed, diffed and scored. The shapes deliberately mirror the OpenTelemetry GenAI semantic
conventions (``chat`` and ``execute_tool`` operations inside an ``invoke_agent`` span) so that
spans emitted by other frameworks can be mapped onto the same model.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from agent_replay.canonical import fingerprint

SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolCall(_Frozen):
    id: str
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)

    def key(self) -> str:
        """Content key: tool name plus canonical arguments, independent of the call ID."""
        return fingerprint("tool", self.name, self.arguments)


class ToolSpec(_Frozen):
    name: str
    description: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    side_effect: bool = False


class Message(_Frozen):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


class Usage(_Frozen):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=round(self.cost_usd + other.cost_usd, 10),
        )


class LLMRequest(_Frozen):
    model: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float = 0.0

    def key(self) -> str:
        """Content key used to match a replayed request against the recording."""
        return fingerprint("llm", self.model_dump(mode="json"))


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"


class LLMResponse(_Frozen):
    model: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.STOP
    usage: Usage = Usage()


class ToolResult(_Frozen):
    tool_call_id: str
    name: str
    output: JsonValue = None
    is_error: bool = False
    error_type: str | None = None


class _StepBase(_Frozen):
    index: int
    started_at: datetime
    duration_ms: float = 0.0


class LLMStep(_StepBase):
    kind: Literal["llm"] = "llm"
    request: LLMRequest
    response: LLMResponse

    def semantic(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content": self.response.content,
            "tool_calls": [
                {"name": c.name, "arguments": c.arguments} for c in self.response.tool_calls
            ],
            "finish_reason": self.response.finish_reason.value,
        }


class ToolStep(_StepBase):
    kind: Literal["tool"] = "tool"
    call: ToolCall
    result: ToolResult
    side_effect: bool = False

    def semantic(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.call.name,
            "arguments": self.call.arguments,
            "output": self.result.output,
            "is_error": self.result.is_error,
        }


Step = Annotated[LLMStep | ToolStep, Field(discriminator="kind")]


class RunStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    MAX_STEPS = "max_steps"


class Trajectory(_Frozen):
    schema_version: int = SCHEMA_VERSION
    trajectory_id: str
    agent_name: str
    task_input: str
    final_output: str | None = None
    status: RunStatus
    error: str | None = None
    started_at: datetime
    duration_ms: float = 0.0
    steps: tuple[Step, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def llm_steps(self) -> tuple[LLMStep, ...]:
        return tuple(s for s in self.steps if isinstance(s, LLMStep))

    @property
    def tool_steps(self) -> tuple[ToolStep, ...]:
        return tuple(s for s in self.steps if isinstance(s, ToolStep))

    @property
    def usage(self) -> Usage:
        total = Usage()
        for step in self.llm_steps:
            total = total + step.response.usage
        return total

    def semantic_steps(self) -> list[dict[str, Any]]:
        return [step.semantic() for step in self.steps]

    def fingerprint(self) -> str:
        """Digest of what the run *did*, ignoring IDs, timestamps and durations.

        Two runs with equal fingerprints made the same decisions, called the same tools with the
        same arguments, saw the same outputs and returned the same answer.
        """
        return fingerprint(
            self.task_input, self.status.value, self.final_output, self.semantic_steps()
        )

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, raw: str | bytes) -> Trajectory:
        return cls.model_validate_json(raw)
