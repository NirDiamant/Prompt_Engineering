"""
Multi-provider LLM helper for Prompt Engineering tutorials.

Supports multiple LLM providers through a unified interface using LangChain's
ChatOpenAI class (since many providers offer OpenAI-compatible APIs).

Supported providers:
    - openai:  OpenAI GPT models (default)
    - minimax: MiniMax M2.5 / M2.7 models via OpenAI-compatible API

Usage:
    from utils.llm_provider import get_llm

    # Use default provider (OpenAI)
    llm = get_llm()

    # Use MiniMax
    llm = get_llm(provider="minimax")

    # Auto-detect from LLM_PROVIDER env var
    llm = get_llm()  # reads LLM_PROVIDER env var if set

Environment variables:
    LLM_PROVIDER      - Provider name ("openai" or "minimax"), default "openai"
    OPENAI_API_KEY     - API key for OpenAI
    MINIMAX_API_KEY    - API key for MiniMax
"""

import os
from typing import Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

SUPPORTED_PROVIDERS = ("openai", "minimax")

# Provider defaults: (base_url, default_model, api_key_env, temperature_min)
_PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": None,  # uses langchain default
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature_min": 0.0,
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.5",
        "api_key_env": "MINIMAX_API_KEY",
        "temperature_min": 0.0,
    },
}


def get_provider_name() -> str:
    """Return the active provider name from ``LLM_PROVIDER`` env var.

    Defaults to ``"openai"`` when the variable is unset.  If
    ``MINIMAX_API_KEY`` is set but ``OPENAI_API_KEY`` is not, auto-selects
    ``"minimax"``.
    """
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider and provider in SUPPORTED_PROVIDERS:
        return provider

    # Auto-detect: prefer MiniMax when only its key is available
    if os.getenv("MINIMAX_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "minimax"

    return "openai"


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    **kwargs,
) -> ChatOpenAI:
    """Create a LangChain ``ChatOpenAI`` instance for the chosen provider.

    Parameters
    ----------
    provider : str, optional
        One of ``"openai"`` or ``"minimax"``.  When *None*, falls back to
        :func:`get_provider_name` (reads ``LLM_PROVIDER`` env var).
    model : str, optional
        Model identifier.  When *None*, uses the provider's default model.
    temperature : float
        Sampling temperature.  Clamped to the provider's minimum if needed.
    max_tokens : int
        Maximum number of tokens to generate.
    **kwargs
        Forwarded to ``ChatOpenAI``.

    Returns
    -------
    ChatOpenAI
        A ready-to-use chat model instance.

    Raises
    ------
    ValueError
        If the provider is not recognised or no API key is found.
    """
    if provider is None:
        provider = get_provider_name()
    provider = provider.strip().lower()

    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Choose from: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    defaults = _PROVIDER_DEFAULTS[provider]
    api_key = os.getenv(defaults["api_key_env"], "")
    if not api_key:
        raise ValueError(
            f"API key not found. Set the {defaults['api_key_env']} "
            f"environment variable."
        )

    model = model or defaults["default_model"]
    temperature = max(temperature, defaults["temperature_min"])

    init_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "api_key": api_key,
        **kwargs,
    }

    if defaults["base_url"] is not None:
        init_kwargs["base_url"] = defaults["base_url"]

    return ChatOpenAI(**init_kwargs)
