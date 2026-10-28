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

    def test_a_stale_base_url_never_receives_the_anthropic_key(self, monkeypatch):
        """``llm_base_url`` left from an OpenAI-compatible setup (OpenRouter
        here) must not become the Anthropic endpoint: the key would go to
        a third party. Checked on the request the factory's backend sends."""
        import anthropic
        import httpx2
        from jarvis.llm import get_llm_backend
        from jarvis.llm.factory import clear_backend_cache

        hosts = []

        def answer(request):
            hosts.append(request.url.host)
            return httpx2.Response(200, json={
                "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})

        real = anthropic.Anthropic

        def on_memory(**kwargs):
            kwargs["http_client"] = httpx2.Client(transport=httpx2.MockTransport(answer))
            return real(**kwargs)

        monkeypatch.setattr(anthropic, "Anthropic", on_memory)
        clear_backend_cache()

        get_llm_backend(_Cfg(llm_base_url="https://openrouter.ai/api/v1")).direct(
            "claude-haiku-5-5", "sys", "hello")
        clear_backend_cache()

        assert hosts == ["api.anthropic.com"]

    def test_remote_egress_is_redacted_when_the_setting_is_on(self):
        from jarvis.llm import AnthropicBackend, get_llm_backend
        from jarvis.llm.redacting import RedactingBackend

        backend = get_llm_backend(_Cfg(auto_redact_before_cloud=True))

        assert isinstance(backend, RedactingBackend)
        assert isinstance(backend.inner, AnthropicBackend)

    def test_a_private_pin_stays_local(self):
        from jarvis.llm import OllamaBackend, get_private_backend

        assert isinstance(get_private_backend(_Cfg(), "qwen2.5:3b", "qwen2.5:3b"), OllamaBackend)

    def test_an_unpinned_private_prompt_goes_where_its_model_lives(self):
        """Reminders, routines, goals and the journal reader use the small
        chain when nothing is pinned. On Anthropic Ollama always runs (it
        embeds), so a local judge such as the default ``gemma4:e2b`` is
        kept: the prompt must go to Ollama, not to Anthropic with a tag
        Anthropic does not serve."""
        from jarvis.llm import AnthropicBackend, OllamaBackend, get_private_backend

        assert isinstance(get_private_backend(_Cfg(), "", "gemma4:e2b"), OllamaBackend)
        assert isinstance(_inner(get_private_backend(_Cfg(), "", "claude-haiku-5-5")), AnthropicBackend)


class TestTheAnthropicKeyStaysWithAnthropic:
    """``llm_api_key`` and ``llm_api_key_env`` are shared by the remote
    providers, so a provider switch can leave an Anthropic key in place for
    another one. Whatever the configuration, it never leaves for another host."""

    def _sent_headers(self, monkeypatch, backend_call):
        import requests

        seen = []

        def post(self, url, json=None, headers=None, timeout=None, **kwargs):
            seen.append(dict(headers or {}))
            raise requests.exceptions.Timeout("offline")

        monkeypatch.setattr(requests.Session, "post", post)
        backend_call()
        return seen

    def test_not_to_an_openai_compatible_server(self, monkeypatch, capsys):
        from jarvis.llm import get_llm_backend
        from jarvis.llm.factory import clear_backend_cache

        clear_backend_cache()
        cfg = _Cfg(llm_provider="openai_compatible", llm_base_url="https://openrouter.ai/api/v1",
                   llm_api_key="sk-ant-api03-secret", llm_chat_model="google/gemini-2.5-flash")
        seen = self._sent_headers(monkeypatch, lambda: get_llm_backend(cfg).direct(
            "google/gemini-2.5-flash", "sys", "hello"))

        assert seen and "Authorization" not in seen[0]
        assert "sk-ant-api03-secret" not in capsys.readouterr().out

    def test_not_to_an_embedding_server(self, monkeypatch):
        from jarvis.llm import get_embedding_backend
        from jarvis.llm.factory import clear_backend_cache

        clear_backend_cache()
        cfg = _Cfg(llm_provider="openai_compatible", llm_base_url="https://openrouter.ai/api/v1",
                   llm_api_key="sk-ant-api03-secret", embedding_provider="openai_compatible")
        seen = self._sent_headers(monkeypatch, lambda: get_embedding_backend(cfg).embed("t", "m"))

        assert seen and "Authorization" not in seen[0]

    def test_another_providers_key_is_not_sent_to_anthropic(self, monkeypatch, capsys):
        """The reverse switch: an OpenRouter key left in the shared field."""
        import anthropic
        from jarvis.llm import get_llm_backend
        from jarvis.llm.factory import clear_backend_cache

        built = []
        monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs: built.append(kwargs))
        clear_backend_cache()

        answer = get_llm_backend(_Cfg(llm_api_key="sk-or-v1-secret")).direct(
            "claude-haiku-5-5", "sys", "hello")
        clear_backend_cache()

        assert answer is None
        assert built == []
        assert "sk-or-v1-secret" not in capsys.readouterr().out

    def test_anthropics_own_openai_compatible_endpoint_still_gets_it(self, monkeypatch):
        from jarvis.llm import get_llm_backend
        from jarvis.llm.factory import clear_backend_cache

        clear_backend_cache()
        cfg = _Cfg(llm_provider="openai_compatible", llm_base_url="https://api.anthropic.com/v1/",
                   llm_api_key="sk-ant-api03-secret", llm_chat_model="anthropic/claude")
        seen = self._sent_headers(monkeypatch, lambda: get_llm_backend(cfg).direct(
            "anthropic/claude", "sys", "hello"))

        assert seen and seen[0].get("Authorization") == "Bearer sk-ant-api03-secret"


class TestEmbeddingRouting:
    def test_embeddings_run_on_ollama(self):
        """Anthropic has no embeddings endpoint."""
        from jarvis.llm import OllamaBackend, get_embedding_backend

        backend = get_embedding_backend(_Cfg())

        assert isinstance(backend, OllamaBackend)
        assert backend.base_url == "http://127.0.0.1:11434"

    def test_a_remote_embedding_server_does_not_inherit_the_anthropic_key(self, monkeypatch):
        import requests
        from jarvis.llm import get_embedding_backend
        from jarvis.llm.factory import clear_backend_cache

        seen = []

        def post(self, url, json=None, headers=None, timeout=None, **kwargs):
            seen.append(dict(headers or {}))
            raise requests.exceptions.Timeout("offline")

        monkeypatch.setattr(requests.Session, "post", post)
        clear_backend_cache()
        get_embedding_backend(_Cfg(
            llm_api_key="a-key-without-the-anthropic-prefix",
            embedding_provider="openai_compatible",
            embedding_base_url="https://embeddings.example/v1",
        )).embed("text", "model")

        assert seen and "Authorization" not in seen[0]


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
