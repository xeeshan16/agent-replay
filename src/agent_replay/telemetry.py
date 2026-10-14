"""OpenTelemetry helpers using the GenAI semantic conventions.

Span names and attribute keys follow the (incubating) ``gen_ai.*`` conventions so traces from
this library line up with traces from other instrumented agent frameworks, and so the ingest path
can map third-party spans back onto the trajectory model.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.trace import Tracer

TRACER_NAME = "agent_replay"

# Attribute keys (OpenTelemetry GenAI semantic conventions).
OPERATION_NAME = "gen_ai.operation.name"
PROVIDER_NAME = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
RESPONSE_MODEL = "gen_ai.response.model"
RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
AGENT_NAME = "gen_ai.agent.name"
CONVERSATION_ID = "gen_ai.conversation.id"
TOOL_NAME = "gen_ai.tool.name"
TOOL_CALL_ID = "gen_ai.tool.call.id"
ERROR_TYPE = "error.type"

# Project-specific attributes.
COST_USD = "agent_replay.cost_usd"
STEP_INDEX = "agent_replay.step.index"
REPLAY_MODE = "agent_replay.replay.mode"
TOOL_SIDE_EFFECT = "agent_replay.tool.side_effect"


def get_tracer() -> Tracer:
    return trace.get_tracer(TRACER_NAME)
