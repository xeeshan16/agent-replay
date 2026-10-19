"""Build a real provider from environment variables (all optional; the fake is the default).

    AGENT_REPLAY_LLM_PROVIDER   fake | openai | azure | ollama | anthropic   (default: fake)
    AGENT_REPLAY_LLM_MODEL      model or deployment name sent to the provider
    AGENT_REPLAY_LLM_BASE_URL   override the endpoint (vLLM, a proxy, an Azure deployment URL)
    AGENT_REPLAY_LLM_API_KEY    falls back to OPENAI_API_KEY, AZURE_OPENAI_API_KEY or
                                ANTHROPIC_API_KEY for the matching provider
    AGENT_REPLAY_LLM_API_VERSION  Azure OpenAI ``api-version``
    AGENT_REPLAY_LLM_TIMEOUT_S, AGENT_REPLAY_LLM_MAX_ATTEMPTS   per-attempt timeout and retries

The returned provider is wrapped in ``ResilientLLM``. Prices for real models are not bundled
(they change); pass a ``PriceTable`` or the scorecard's ``[prices]`` table to get costs, otherwise
token counts are still recorded and ratios fall back to tokens.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum

from agent_replay.accounting import DEFAULT_PRICES, PriceTable
from agent_replay.errors import AgentReplayError
from agent_replay.providers.base import LLMProvider
from agent_replay.providers.http import AnthropicLLM, OpenAICompatibleLLM
from agent_replay.providers.resilience import ResilientLLM, RetryPolicy

PREFIX = "AGENT_REPLAY_LLM_"


class ProviderKind(StrEnum):
    FAKE = "fake"
    OPENAI = "openai"
    AZURE = "azure"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"


class ProviderConfigError(AgentReplayError):
    pass


def provider_kind(env: Mapping[str, str] | None = None) -> ProviderKind:
    env = os.environ if env is None else env
    raw = env.get(f"{PREFIX}PROVIDER", "fake").strip().lower() or "fake"
    try:
        return ProviderKind(raw)
    except ValueError:
        choices = ", ".join(k.value for k in ProviderKind)
        raise ProviderConfigError(f"{PREFIX}PROVIDER={raw!r}; expected one of {choices}") from None


def provider_from_env(
    env: Mapping[str, str] | None = None, *, prices: PriceTable = DEFAULT_PRICES
) -> LLMProvider | None:
    """The configured real provider, or ``None`` when the fake is selected."""
    env = os.environ if env is None else env
    kind = provider_kind(env)
    if kind is ProviderKind.FAKE:
        return None
    model = env.get(f"{PREFIX}MODEL") or None
    base_url = env.get(f"{PREFIX}BASE_URL") or None
    timeout = float(env.get(f"{PREFIX}TIMEOUT_S", "60"))
    attempts = int(env.get(f"{PREFIX}MAX_ATTEMPTS", "3"))

    def key(*fallbacks: str, required: bool = True) -> str | None:
        for name in (f"{PREFIX}API_KEY", *fallbacks):
            if env.get(name):
                return env[name]
        if required:
            raise ProviderConfigError(f"{kind.value} needs {PREFIX}API_KEY or {fallbacks[0]}")
        return None

    inner: LLMProvider
    if kind is ProviderKind.ANTHROPIC:
        inner = AnthropicLLM(
            api_key=key("ANTHROPIC_API_KEY") or "",
            model=model,
            base_url=base_url or "https://api.anthropic.com/v1",
            prices=prices,
            timeout_s=timeout,
        )
    elif kind is ProviderKind.AZURE:
        if not base_url:
            raise ProviderConfigError(
                f"azure needs {PREFIX}BASE_URL, e.g. "
                "https://<resource>.openai.azure.com/openai/deployments/<deployment>"
            )
        inner = OpenAICompatibleLLM(
            base_url=base_url,
            api_key=key("AZURE_OPENAI_API_KEY"),
            model=model,
            azure=True,
            api_version=env.get(f"{PREFIX}API_VERSION", "2024-10-21"),
            prices=prices,
            timeout_s=timeout,
        )
    elif kind is ProviderKind.OLLAMA:
        inner = OpenAICompatibleLLM(
            base_url=base_url or "http://localhost:11434/v1",
            api_key=key("OLLAMA_API_KEY", required=False),
            model=model,
            prices=prices,
            timeout_s=timeout,
        )
    else:
        inner = OpenAICompatibleLLM(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=key("OPENAI_API_KEY"),
            model=model,
            prices=prices,
            timeout_s=timeout,
        )
    return ResilientLLM(inner, RetryPolicy(max_attempts=attempts, timeout_s=timeout))
