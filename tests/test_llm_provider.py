"""Unit tests for utils.llm_provider module."""

import os
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.llm_provider import (
    get_llm,
    get_provider_name,
    SUPPORTED_PROVIDERS,
    _PROVIDER_DEFAULTS,
)


# ---------------------------------------------------------------------------
# get_provider_name tests
# ---------------------------------------------------------------------------

class TestGetProviderName:
    """Tests for provider auto-detection logic."""

    def test_default_is_openai(self):
        """Default provider should be openai when no env vars are set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("MINIMAX_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            assert get_provider_name() == "openai"

    def test_explicit_openai(self):
        """LLM_PROVIDER=openai should select openai."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=False):
            assert get_provider_name() == "openai"

    def test_explicit_minimax(self):
        """LLM_PROVIDER=minimax should select minimax."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "minimax"}, clear=False):
            assert get_provider_name() == "minimax"

    def test_explicit_minimax_uppercase(self):
        """LLM_PROVIDER should be case-insensitive."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "MiniMax"}, clear=False):
            assert get_provider_name() == "minimax"

    def test_auto_detect_minimax_when_only_minimax_key(self):
        """Auto-detect minimax when MINIMAX_API_KEY is set but OPENAI_API_KEY is not."""
        env = {"MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("LLM_PROVIDER", None)
            os.environ.pop("OPENAI_API_KEY", None)
            assert get_provider_name() == "minimax"

    def test_prefers_openai_when_both_keys(self):
        """When both API keys are set and LLM_PROVIDER is unset, default to openai."""
        env = {"MINIMAX_API_KEY": "mm-key", "OPENAI_API_KEY": "oai-key"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("LLM_PROVIDER", None)
            assert get_provider_name() == "openai"

    def test_unknown_provider_falls_through(self):
        """An unknown LLM_PROVIDER value should fall through to default logic."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}, clear=True):
            os.environ.pop("MINIMAX_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            assert get_provider_name() == "openai"

    def test_whitespace_handling(self):
        """LLM_PROVIDER with extra whitespace should still work."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "  minimax  "}, clear=False):
            assert get_provider_name() == "minimax"


# ---------------------------------------------------------------------------
# get_llm tests
# ---------------------------------------------------------------------------

class TestGetLlm:
    """Tests for LLM instance creation."""

    @patch("utils.llm_provider.ChatOpenAI")
    def test_openai_defaults(self, mock_chat):
        """OpenAI provider should use correct defaults."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            get_llm(provider="openai")
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args[1]
            assert kwargs["model"] == "gpt-4o-mini"
            assert kwargs["temperature"] == 0.0
            assert kwargs["api_key"] == "test-key"
            assert "base_url" not in kwargs

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_defaults(self, mock_chat):
        """MiniMax provider should use correct base_url and model."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False):
            get_llm(provider="minimax")
            mock_chat.assert_called_once()
            kwargs = mock_chat.call_args[1]
            assert kwargs["model"] == "MiniMax-M2.5"
            assert kwargs["base_url"] == "https://api.minimax.io/v1"
            assert kwargs["api_key"] == "mm-key"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_custom_model(self, mock_chat):
        """Custom model parameter should override the default."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False):
            get_llm(provider="minimax", model="MiniMax-M2.7")
            kwargs = mock_chat.call_args[1]
            assert kwargs["model"] == "MiniMax-M2.7"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_custom_temperature(self, mock_chat):
        """Custom temperature should be passed through."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            get_llm(provider="openai", temperature=0.7)
            kwargs = mock_chat.call_args[1]
            assert kwargs["temperature"] == 0.7

    @patch("utils.llm_provider.ChatOpenAI")
    def test_temperature_clamped_to_min(self, mock_chat):
        """Negative temperature should be clamped to provider minimum."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            get_llm(provider="openai", temperature=-1.0)
            kwargs = mock_chat.call_args[1]
            assert kwargs["temperature"] == 0.0

    @patch("utils.llm_provider.ChatOpenAI")
    def test_max_tokens(self, mock_chat):
        """Custom max_tokens should be passed through."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            get_llm(provider="openai", max_tokens=2048)
            kwargs = mock_chat.call_args[1]
            assert kwargs["max_tokens"] == 2048

    @patch("utils.llm_provider.ChatOpenAI")
    def test_extra_kwargs(self, mock_chat):
        """Extra keyword arguments should be forwarded to ChatOpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False):
            get_llm(provider="openai", top_p=0.9)
            kwargs = mock_chat.call_args[1]
            assert kwargs["top_p"] == 0.9

    def test_missing_api_key_raises(self):
        """Missing API key should raise ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("MINIMAX_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            with pytest.raises(ValueError, match="API key not found"):
                get_llm(provider="openai")

    def test_unsupported_provider_raises(self):
        """Unsupported provider name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_llm(provider="nonexistent_provider")

    @patch("utils.llm_provider.ChatOpenAI")
    def test_auto_detect_minimax(self, mock_chat):
        """When provider=None and only MINIMAX_API_KEY is set, use minimax."""
        env = {"MINIMAX_API_KEY": "mm-key"}
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("LLM_PROVIDER", None)
            get_llm()
            kwargs = mock_chat.call_args[1]
            assert kwargs["base_url"] == "https://api.minimax.io/v1"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_m27_model(self, mock_chat):
        """MiniMax M2.7 model should work correctly."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False):
            get_llm(provider="minimax", model="MiniMax-M2.7")
            kwargs = mock_chat.call_args[1]
            assert kwargs["model"] == "MiniMax-M2.7"
            assert kwargs["base_url"] == "https://api.minimax.io/v1"

    @patch("utils.llm_provider.ChatOpenAI")
    def test_minimax_highspeed_model(self, mock_chat):
        """MiniMax M2.5-highspeed model should work correctly."""
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False):
            get_llm(provider="minimax", model="MiniMax-M2.5-highspeed")
            kwargs = mock_chat.call_args[1]
            assert kwargs["model"] == "MiniMax-M2.5-highspeed"


# ---------------------------------------------------------------------------
# Provider configuration tests
# ---------------------------------------------------------------------------

class TestProviderConfig:
    """Tests for provider configuration constants."""

    def test_supported_providers_tuple(self):
        """SUPPORTED_PROVIDERS should be a tuple of strings."""
        assert isinstance(SUPPORTED_PROVIDERS, tuple)
        assert all(isinstance(p, str) for p in SUPPORTED_PROVIDERS)

    def test_openai_in_providers(self):
        assert "openai" in SUPPORTED_PROVIDERS

    def test_minimax_in_providers(self):
        assert "minimax" in SUPPORTED_PROVIDERS

    def test_each_provider_has_defaults(self):
        """Every supported provider must have an entry in _PROVIDER_DEFAULTS."""
        for provider in SUPPORTED_PROVIDERS:
            assert provider in _PROVIDER_DEFAULTS

    def test_defaults_have_required_keys(self):
        """Each provider's defaults must have base_url, default_model, api_key_env."""
        required_keys = {"base_url", "default_model", "api_key_env", "temperature_min"}
        for provider, defaults in _PROVIDER_DEFAULTS.items():
            assert required_keys.issubset(defaults.keys()), (
                f"Provider {provider!r} missing keys: "
                f"{required_keys - defaults.keys()}"
            )

    def test_minimax_base_url(self):
        assert _PROVIDER_DEFAULTS["minimax"]["base_url"] == "https://api.minimax.io/v1"

    def test_minimax_default_model(self):
        assert _PROVIDER_DEFAULTS["minimax"]["default_model"] == "MiniMax-M2.5"

    def test_minimax_api_key_env(self):
        assert _PROVIDER_DEFAULTS["minimax"]["api_key_env"] == "MINIMAX_API_KEY"
