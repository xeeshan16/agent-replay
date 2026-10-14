from __future__ import annotations

import asyncio
import random

import pytest

from agent_replay.errors import ProviderError, ProviderTimeoutError
from agent_replay.model import LLMRequest, LLMResponse, Message
from agent_replay.providers.resilience import ResilientLLM, RetryPolicy

REQUEST = LLMRequest(model="m", messages=(Message(role="user", content="hi"),))


class FlakyLLM:
    def __init__(self, failures: list[BaseException], delay_s: float = 0.0) -> None:
        self._failures = failures
        self._delay_s = delay_s
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "flaky"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._failures:
            raise self._failures.pop(0)
        return LLMResponse(model=request.model, content="ok")


class SleepRecorder:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_retries_retryable_errors_then_succeeds() -> None:
    inner = FlakyLLM([ProviderError("429", retryable=True), ProviderError("503", retryable=True)])
    sleep = SleepRecorder()
    llm = ResilientLLM(inner, RetryPolicy(max_attempts=3), sleep=sleep, rng=random.Random(7))
    response = await llm.complete(REQUEST)
    assert response.content == "ok"
    assert inner.calls == 3
    assert len(sleep.delays) == 2


async def test_non_retryable_error_fails_fast() -> None:
    inner = FlakyLLM([ProviderError("bad request", retryable=False)])
    sleep = SleepRecorder()
    llm = ResilientLLM(inner, RetryPolicy(max_attempts=5), sleep=sleep)
    with pytest.raises(ProviderError, match="bad request"):
        await llm.complete(REQUEST)
    assert inner.calls == 1
    assert sleep.delays == []


async def test_timeouts_are_retried_and_reported() -> None:
    inner = FlakyLLM([], delay_s=1.0)
    llm = ResilientLLM(inner, RetryPolicy(max_attempts=2, timeout_s=0.01), sleep=SleepRecorder())
    with pytest.raises(ProviderTimeoutError, match="timed out"):
        await llm.complete(REQUEST)
    assert inner.calls == 2


def test_backoff_is_bounded_full_jitter() -> None:
    policy = RetryPolicy(base_delay_s=0.5, max_delay_s=2.0)
    rng = random.Random(1)
    for attempt in range(10):
        delay = policy.backoff(attempt, rng)
        assert 0.0 <= delay <= min(2.0, 0.5 * 2**attempt)


def test_rejects_invalid_policy() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        ResilientLLM(FlakyLLM([]), RetryPolicy(max_attempts=0))
