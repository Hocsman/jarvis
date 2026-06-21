"""The LLM API key can be sourced from an environment variable.

``llm_api_key_env`` names an env var (e.g. "OPENROUTER_API_KEY"); the
parser resolves ``llm_api_key`` from it so the secret never has to sit
in config.json. Falls back to the literal ``llm_api_key`` field when
the env var is unset or no name is given.
"""

from __future__ import annotations

import json

import pytest


def _write_cfg(tmp_path, monkeypatch, data: dict):
    p = tmp_path / "jarvis.json"
    p.write_text(json.dumps(data))
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(p))


class TestLlmApiKeyEnv:

    @pytest.mark.unit
    def test_key_resolved_from_env_var(self, tmp_path, monkeypatch) -> None:
        _write_cfg(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://openrouter.ai/api/v1",
            "llm_api_key_env": "OPENROUTER_API_KEY",
            "llm_chat_model": "deepseek/deepseek-v4-flash",
        })
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromenv")

        from jarvis.config import load_settings
        s = load_settings()
        assert s.llm_api_key == "sk-or-fromenv"
        assert s.llm_api_key_env == "OPENROUTER_API_KEY"

    @pytest.mark.unit
    def test_missing_env_var_leaves_key_empty(self, tmp_path, monkeypatch) -> None:
        _write_cfg(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_api_key_env": "OPENROUTER_API_KEY",
        })
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from jarvis.config import load_settings
        s = load_settings()
        assert s.llm_api_key == ""

    @pytest.mark.unit
    def test_env_var_overrides_literal_key(self, tmp_path, monkeypatch) -> None:
        """When both are present, the env var wins (the literal field is
        the fallback, not the override)."""
        _write_cfg(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_api_key": "sk-or-literal",
            "llm_api_key_env": "OPENROUTER_API_KEY",
        })
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromenv")

        from jarvis.config import load_settings
        s = load_settings()
        assert s.llm_api_key == "sk-or-fromenv"

    @pytest.mark.unit
    def test_literal_key_used_when_no_env_name(self, tmp_path, monkeypatch) -> None:
        """No ``llm_api_key_env`` -> the literal field is used as-is
        (backwards compatible with configs that store the key inline)."""
        _write_cfg(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_api_key": "sk-or-literal",
        })
        from jarvis.config import load_settings
        s = load_settings()
        assert s.llm_api_key == "sk-or-literal"

    @pytest.mark.unit
    def test_resolved_key_reaches_backend(self, tmp_path, monkeypatch) -> None:
        """The env-resolved key propagates through the factory to the
        underlying OpenAI-compatible backend."""
        _write_cfg(tmp_path, monkeypatch, {
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://openrouter.ai/api/v1",
            "llm_api_key_env": "OPENROUTER_API_KEY",
            "llm_chat_model": "deepseek/deepseek-v4-flash",
        })
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fromenv")

        from jarvis.config import load_settings
        from jarvis.llm.factory import get_llm_backend
        from jarvis.llm.redacting import RedactingBackend

        s = load_settings()
        backend = get_llm_backend(s)
        inner = backend.inner if isinstance(backend, RedactingBackend) else backend
        assert inner._api_key == "sk-or-fromenv"
