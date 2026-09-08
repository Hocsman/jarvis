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
    assert backend_bare._base_url == "http://127.0.0.1:11434"


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
