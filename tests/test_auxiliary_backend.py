"""Tests for auxiliary LLM backend dispatch.

Auxiliary tasks (intent judge, tool router, evaluator) can run on a local
Ollama model even when the main chat provider is a remote OpenAI-compatible
server (e.g. OpenRouter). A bare model name without a slash indicates a local
model tag and is routed to Ollama, while a namespaced tag (vendor/model) or a
local OpenAI-compatible endpoint stays on the configured LLM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import pytest

from src.jarvis.llm import get_auxiliary_backend
from src.jarvis.llm.ollama import OllamaBackend
from src.jarvis.llm.openai_compatible import OpenAICompatibleBackend
from src.jarvis.llm.redacting import RedactingBackend


@dataclass
class _DummySettings:
    llm_provider: str = "ollama"
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    auto_redact_before_cloud: bool = False
    llm_extra_body: dict = None


def test_ollama_provider_always_uses_ollama():
    cfg = _DummySettings(llm_provider="ollama")
    backend_bare = get_auxiliary_backend(cfg, "qwen2.5:3b")
    backend_slash = get_auxiliary_backend(cfg, "meta-llama/llama-3.1-8b")

    assert isinstance(backend_bare, OllamaBackend)
    assert isinstance(backend_slash, OllamaBackend)


def test_local_openai_compatible_endpoint_keeps_openai_backend():
    cfg = _DummySettings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1234/v1",
    )
    backend_bare = get_auxiliary_backend(cfg, "qwen2.5:3b")
    backend_slash = get_auxiliary_backend(cfg, "meta-llama/llama-3.1-8b")

    assert isinstance(backend_bare, (OpenAICompatibleBackend, RedactingBackend))
    assert isinstance(backend_slash, (OpenAICompatibleBackend, RedactingBackend))


def test_remote_openai_compatible_routes_bare_tag_to_ollama():
    cfg = _DummySettings(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
    )
    backend_bare = get_auxiliary_backend(cfg, "qwen2.5:3b")

    # Bare tag has no slash: must route to local Ollama
    assert isinstance(backend_bare, OllamaBackend)
    assert backend_bare.base_url == "http://127.0.0.1:11434"


def test_remote_openai_compatible_routes_namespaced_tag_to_cloud():
    cfg = _DummySettings(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
    )
    backend_slash = get_auxiliary_backend(cfg, "openai/gpt-oss-120b")

    # Namespaced tag has a slash: must stay on remote OpenAI-compatible backend
    assert isinstance(backend_slash, (OpenAICompatibleBackend, RedactingBackend))


def test_empty_model_falls_back_to_llm_backend():
    cfg = _DummySettings(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
    )
    backend_empty = get_auxiliary_backend(cfg, "")
    backend_none = get_auxiliary_backend(cfg, None)

    assert isinstance(backend_empty, (OpenAICompatibleBackend, RedactingBackend))
    assert isinstance(backend_none, (OpenAICompatibleBackend, RedactingBackend))


# ── Shim routing ──────────────────────────────────────────────────────────
#
# Every auxiliary call site owns a thin ``call_llm_direct`` shim. The shim
# must hand its model to the dispatcher: a model-blind ``get_llm_backend``
# sends a bare local tag to the remote endpoint, which answers HTTP 400, and
# the auxiliary task silently dies on every call.


def _remote_provider_settings():
    return _DummySettings(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
    )


@pytest.fixture
def direct_calls(monkeypatch):
    """Record which backend class received a direct() call, and the model."""
    calls = []

    def _ollama_direct(self, chat_model, system_prompt, user_content, **kwargs):
        calls.append(("ollama", chat_model))
        return "ok"

    def _openai_direct(self, chat_model, system_prompt, user_content, **kwargs):
        calls.append(("openai", chat_model))
        return "ok"

    monkeypatch.setattr(OllamaBackend, "direct", _ollama_direct)
    monkeypatch.setattr(OpenAICompatibleBackend, "direct", _openai_direct)
    return calls


def test_planner_shim_routes_a_bare_model_to_local_ollama(direct_calls):
    from src.jarvis.reply import planner

    result = planner.call_llm_direct(
        cfg=_remote_provider_settings(), chat_model="qwen2.5:3b",
        system_prompt="s", user_content="u",
    )

    assert result == "ok"
    assert direct_calls == [("ollama", "qwen2.5:3b")]


def test_planner_shim_keeps_a_namespaced_model_on_the_remote_backend(direct_calls):
    from src.jarvis.reply import planner

    result = planner.call_llm_direct(
        cfg=_remote_provider_settings(), chat_model="deepseek/deepseek-v4-flash",
        system_prompt="s", user_content="u",
    )

    assert result == "ok"
    assert direct_calls == [("openai", "deepseek/deepseek-v4-flash")]


def test_graph_ops_shim_routes_a_bare_model_to_local_ollama(direct_calls):
    from src.jarvis.memory import graph_ops

    result = graph_ops.call_llm_direct(
        cfg=_remote_provider_settings(), chat_model="qwen2.5:3b",
        system_prompt="s", user_content="u",
    )

    assert result == "ok"
    assert direct_calls == [("ollama", "qwen2.5:3b")]


def test_evaluator_shim_routes_a_bare_model_to_local_ollama(direct_calls):
    from src.jarvis.reply import evaluator

    result = evaluator.call_llm_direct(
        cfg=_remote_provider_settings(), chat_model="qwen2.5:3b",
        system_prompt="s", user_content="u",
    )

    assert result == "ok"
    assert direct_calls == [("ollama", "qwen2.5:3b")]


def test_weather_place_extractor_routes_its_bare_model_to_local_ollama(direct_calls):
    from types import SimpleNamespace

    from src.jarvis.tools.builtin import weather

    cfg = SimpleNamespace(
        llm_provider="openai_compatible",
        llm_base_url="https://openrouter.ai/api/v1",
        ollama_base_url="http://127.0.0.1:11434",
        auto_redact_before_cloud=False,
        llm_extra_body=None,
        tool_router_model="qwen2.5:3b",
    )

    place = weather._extract_place_from_user_text("quel temps à Paris", cfg)

    assert place == "ok"
    assert direct_calls == [("ollama", "qwen2.5:3b")]
