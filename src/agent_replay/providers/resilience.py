"""Timeouts and retries with exponential backoff and full jitter around any provider."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_replay.errors import ProviderError, ProviderTimeoutError
from agent_replay.logs import get_logger
from agent_replay.model import LLMRequest, LLMResponse
from agent_replay.providers.base import LLMProvider

_log = get_logger("providers.resilience")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    timeout_s: float = 30.0

    def backoff(self, attempt: int, rng: random.Random) -> float:
        """Full-jitter backoff: uniform in [0, min(max, base * 2**attempt)]."""
        ceiling = min(self.max_delay_s, self.base_delay_s * (2**attempt))
        return rng.uniform(0.0, ceiling)


class ResilientLLM:
    """Wraps a provider with a per-attempt timeout and bounded retries on retryable errors.

    Non-retryable ``ProviderError`` (bad request, auth) fails fast. Model calls are not
    idempotent in general, but a completion that never returned has no observable side effect,
    so retrying a timed-out or rate-limited call is safe.
    """

    def __init__(
        self,
        inner: LLMProvider,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        policy = policy or RetryPolicy()
        if policy.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._inner = inner
        self._policy = policy
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not cryptography

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    async def complete(self, request: LLMRequest) -> LLMResponse:
        last_error: ProviderError | None = None
        for attempt in range(self._policy.max_attempts):
            try:
                async with asyncio.timeout(self._policy.timeout_s):
                    return await self._inner.complete(request)
            except TimeoutError:
                last_error = ProviderTimeoutError(
                    f"{self.provider_name} timed out after {self._policy.timeout_s}s"
                )
            except ProviderError as exc:
                if not exc.retryable:
                    raise
                last_error = exc
            if attempt + 1 < self._policy.max_attempts:
                delay = self._policy.backoff(attempt, self._rng)
                _log.warning(
                    "provider call failed, retrying",
                    extra={
                        "ctx": {
                            "provider": self.provider_name,
                            "attempt": attempt + 1,
                            "delay_s": round(delay, 3),
                            "error": str(last_error),
                        }
                    },
                )
                await self._sleep(delay)
        assert last_error is not None  # noqa: S101 - loop runs at least once
        raise last_error
