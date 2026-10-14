"""The provider interface every model backend implements.

Agents, the recorder and the replayer depend only on this protocol, so a deterministic fake, a
cassette-backed replayer and real network providers are interchangeable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_replay.model import LLMRequest, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
