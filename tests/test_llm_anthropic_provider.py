"""Routing and configuration for ``llm_provider: anthropic``.

Which backend each call reaches, which key goes with it, and what the
settings loader makes of an Anthropic configuration. The wire shape itself
is covered in ``test_llm_anthropic.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import pytest


@dataclass
class _Cfg:
    llm_provider: str = "anthropic"
    llm_base_url: str = ""
    llm_api_key: str = "sk-ant-test"
    llm_chat_model: str = "claude-haiku-5-5"
    llm_extra_body: Dict[str, Any] = field(default_factory=dict)
    auto_redact_before_cloud: bool = False
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "gemma4:e2b"
    ollama_embed_model: str = "nomic-embed-text"


def _inner(backend):
    from jarvis.llm.redacting import RedactingBackend

    return backend.inner if isinstance(backend, RedactingBackend) else backend


class TestChatRouting:
    def test_the_provider_builds_the_anthropic_backend(self):
        from jarvis.llm import AnthropicBackend, get_llm_backend

        assert isinstance(get_llm_backend(_Cfg()), AnthropicBackend)

    def test_a_claude_model_stays_on_anthropic(self):
        from jarvis.llm import AnthropicBackend, get_auxiliary_backend

        assert isinstance(get_auxiliary_backend(_Cfg(), "claude-haiku-5-5"), AnthropicBackend)

    @pytest.mark.parametrize("local_tag", ["qwen2.5:3b", "gemma4:e2b", "llama3"])
    def test_a_local_tag_runs_on_ollama(self, local_tag):
        """Claude model IDs carry no vendor slash, so the slash test that
        separates OpenRouter IDs from local tags cannot work here: a name
        is Anthropic's when it is a Claude model ID."""
        from jarvis.llm import OllamaBackend, get_auxiliary_backend

        backend = get_auxiliary_backend(_Cfg(), local_tag)

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_a_stale_base_url_never_receives_the_anthropic_key(self):
        """``llm_base_url`` left from an OpenAI-compatible setup (OpenRouter
        here) must not become the Anthropic endpoint: the key would go to
        a third party."""
        from jarvis.llm import get_llm_backend

        backend = _inner(get_llm_backend(_Cfg(llm_base_url="https://openrouter.ai/api/v1")))

        assert backend.base_url == "https://api.anthropic.com"

    def test_remote_egress_is_redacted_when_the_setting_is_on(self):
        from jarvis.llm import AnthropicBackend, get_llm_backend
        from jarvis.llm.redacting import RedactingBackend

        backend = get_llm_backend(_Cfg(auto_redact_before_cloud=True))

        assert isinstance(backend, RedactingBackend)
        assert isinstance(backend.inner, AnthropicBackend)

    def test_a_private_pin_stays_local(self):
        from jarvis.llm import OllamaBackend, get_private_backend

        assert isinstance(get_private_backend(_Cfg(), "qwen2.5:3b"), OllamaBackend)


class TestEmbeddingRouting:
    def test_embeddings_run_on_ollama(self):
        """Anthropic has no embeddings endpoint."""
        from jarvis.llm import OllamaBackend, get_embedding_backend

        backend = get_embedding_backend(_Cfg())

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_a_remote_embedding_server_does_not_inherit_the_anthropic_key(self):
        from jarvis.llm import OpenAICompatibleBackend, get_embedding_backend

        backend = _inner(get_embedding_backend(_Cfg(
            embedding_provider="openai_compatible",
            embedding_base_url="https://embeddings.example/v1",
        )))

        assert isinstance(backend, OpenAICompatibleBackend)
        assert backend._api_key is None


class TestSettings:
    def _load(self, tmp_path, monkeypatch, cfg: dict):
        cfg_path = tmp_path / "config.json"
        cfg.setdefault("_config_version", 2)
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        from jarvis.config import load_settings
        return load_settings()

    def test_the_provider_is_accepted(self, tmp_path, monkeypatch):
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "anthropic", "llm_chat_model": "claude-haiku-5-5"})

        assert settings.llm_provider == "anthropic"
        assert settings.llm_chat_model == "claude-haiku-5-5"

    def test_the_key_is_read_from_the_named_variable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("YUBA_TEST_ANTHROPIC_KEY", "sk-ant-from-env")
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "anthropic", "llm_api_key_env": "YUBA_TEST_ANTHROPIC_KEY"})

        assert settings.llm_api_key == "sk-ant-from-env"

    def test_a_claude_pin_is_kept_for_an_auxiliary_role(self, tmp_path, monkeypatch):
        import jarvis.config as config

        monkeypatch.setattr(config, "is_ollama_reachable", lambda *a, **k: False)
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "anthropic", "llm_chat_model": "claude-sonnet-5-5",
            "intent_judge_model": "claude-haiku-5-5"})

        assert settings.intent_judge_model == "claude-haiku-5-5"

    def test_a_local_pin_is_kept_while_ollama_answers(self, tmp_path, monkeypatch):
        import jarvis.config as config

        monkeypatch.setattr(config, "is_ollama_reachable", lambda *a, **k: True)
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "anthropic", "llm_chat_model": "claude-haiku-5-5",
            "tool_router_model": "qwen2.5:3b"})

        assert settings.tool_router_model == "qwen2.5:3b"

    def test_a_local_pin_falls_back_to_the_chat_model_without_ollama(self, tmp_path, monkeypatch):
        import jarvis.config as config

        monkeypatch.setattr(config, "is_ollama_reachable", lambda *a, **k: False)
        settings = self._load(tmp_path, monkeypatch, {
            "llm_provider": "anthropic", "llm_chat_model": "claude-haiku-5-5",
            "tool_router_model": "qwen2.5:3b"})

        assert settings.tool_router_model == "claude-haiku-5-5"
