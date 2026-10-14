"""Exception hierarchy. Callers catch ``AgentReplayError`` to handle anything raised here."""

from __future__ import annotations


class AgentReplayError(Exception):
    """Base class for all errors raised by agent_replay."""


class ProviderError(AgentReplayError):
    """A model provider failed."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ProviderTimeoutError(ProviderError):
    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=True)


class ToolNotFoundError(AgentReplayError):
    pass


class CassetteMissError(AgentReplayError):
    """A replayed run asked for an interaction that was never recorded."""

    def __init__(self, kind: str, key: str, detail: str) -> None:
        super().__init__(f"no recorded {kind} interaction for key {key}: {detail}")
        self.kind = kind
        self.key = key
        self.detail = detail


class SideEffectBlockedError(AgentReplayError):
    """A tool with side effects was invoked in a context where live execution is forbidden."""
