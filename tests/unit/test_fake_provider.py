from __future__ import annotations

from agent_replay.accounting import ModelPrice, PriceTable, estimate_tokens
from agent_replay.model import FinishReason, LLMRequest, Message
from agent_replay.providers.fake import FakeLLM, FakeTurn


def _request(text: str = "refund ORD-1001") -> LLMRequest:
    return LLMRequest(
        model="fake-tool-caller-1",
        messages=(
            Message(role="system", content="be helpful"),
            Message(role="user", content=text),
        ),
    )


async def test_same_request_yields_identical_response() -> None:
    llm = FakeLLM(lambda _req: FakeTurn.call("lookup_order", order_id="ORD-1001"))
    first = await llm.complete(_request())
    second = await llm.complete(_request())
    assert first == second
    assert first.finish_reason is FinishReason.TOOL_CALLS
    assert first.tool_calls[0].id.startswith("call_")
    assert llm.calls == 2


async def test_tool_call_ids_depend_on_request_content() -> None:
    llm = FakeLLM(lambda _req: FakeTurn.call("lookup_order", order_id="ORD-1001"))
    first = await llm.complete(_request("a"))
    second = await llm.complete(_request("b"))
    assert first.tool_calls[0].id != second.tool_calls[0].id


async def test_usage_and_cost_are_accounted() -> None:
    prices = PriceTable({"fake-tool-caller-1": ModelPrice(input_per_mtok=1.0, output_per_mtok=2.0)})
    llm = FakeLLM(lambda _req: FakeTurn.answer("All done here."), prices=prices)
    response = await llm.complete(_request())
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.output_tokens == estimate_tokens("All done here.") == 4
    # two messages: 4 overhead each + "be helpful" (2) + "refund ORD-1001" (4)
    assert response.usage.input_tokens == 14
    assert response.usage.cost_usd == round((14 * 1.0 + 4 * 2.0) / 1_000_000, 10)


async def test_unknown_model_is_priced_at_zero() -> None:
    llm = FakeLLM(lambda _req: FakeTurn.answer("x"), prices=PriceTable({}))
    response = await llm.complete(_request())
    assert response.usage.cost_usd == 0.0
    assert response.usage.input_tokens > 0
