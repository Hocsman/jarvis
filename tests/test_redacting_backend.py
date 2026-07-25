"""Tests for the RedactingBackend decorator + factory wiring.

The decorator scrubs secret-shaped tokens from prompts before they
reach the inner (cloud) backend. The factory wraps the
OpenAI-compatible backend with it when ``auto_redact_before_cloud``
is set (the default), and never wraps the local Ollama backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from jarvis.llm.backend import LLMBackend
from jarvis.llm.redacting import RedactingBackend


class _SpyBackend(LLMBackend):
    """Records the arguments it was called with so tests can assert
    the text was scrubbed before delegation."""

    def __init__(self) -> None:
        self.calls: Dict[str, Any] = {}

    def direct(self, chat_model, system_prompt, user_content, timeout_sec=10.0,
               thinking=False, num_ctx=4096, temperature=None):
        self.calls["direct"] = {"system": system_prompt, "user": user_content}
        return "ok"

    def streaming(self, chat_model, system_prompt, user_content, on_token=None,
                  timeout_sec=30.0, thinking=False):
        self.calls["streaming"] = {"system": system_prompt, "user": user_content}
        return "ok"

    def chat(self, chat_model, messages, timeout_sec=30.0, extra_options=None,
             tools=None, thinking=False, on_token=None):
        self.calls["chat"] = {"messages": messages}
        return {"message": {"role": "assistant", "content": "ok"}}

    def embed(self, text, model, timeout_sec=15.0):
        self.calls["embed"] = {"text": text}
        return [0.0]

    def list_models(self, timeout_sec=5.0):
        self.calls["list_models"] = True
        return ["m"]


# A user prompt carrying secret-shaped tokens the scrub must mask.
_DIRTY = "email me at alice@example.com, my AWS key is AKIAIOSFODNN7EXAMPLE"


class TestRedactingBackendScrubs:

    @pytest.mark.unit
    def test_direct_scrubs_system_and_user(self) -> None:
        spy = _SpyBackend()
        RedactingBackend(spy).direct("m", _DIRTY, _DIRTY)
        sent = spy.calls["direct"]
        assert "alice@example.com" not in sent["user"]
        assert "AKIAIOSFODNN7EXAMPLE" not in sent["user"]
        assert "alice@example.com" not in sent["system"]

    @pytest.mark.unit
    def test_streaming_scrubs(self) -> None:
        spy = _SpyBackend()
        RedactingBackend(spy).streaming("m", "sys", _DIRTY)
        assert "alice@example.com" not in spy.calls["streaming"]["user"]

    @pytest.mark.unit
    def test_chat_scrubs_each_message(self) -> None:
        spy = _SpyBackend()
        msgs = [
            {"role": "system", "content": "you are a butler"},
            {"role": "user", "content": _DIRTY},
        ]
        RedactingBackend(spy).chat("m", msgs)
        sent = spy.calls["chat"]["messages"]
        assert "alice@example.com" not in sent[1]["content"]
        # Original list must not be mutated in place.
        assert "alice@example.com" in msgs[1]["content"]

    @pytest.mark.unit
    def test_chat_preserves_non_string_content(self) -> None:
        """Multimodal / structured content blocks pass through untouched."""
        spy = _SpyBackend()
        block = [{"type": "image", "data": "..."}]
        msgs = [{"role": "user", "content": block}]
        RedactingBackend(spy).chat("m", msgs)
        assert spy.calls["chat"]["messages"][0]["content"] is block

    @pytest.mark.unit
    def test_embed_scrubs(self) -> None:
        spy = _SpyBackend()
        RedactingBackend(spy).embed(_DIRTY, "embed-model")
        assert "alice@example.com" not in spy.calls["embed"]["text"]

    @pytest.mark.unit
    def test_list_models_passthrough(self) -> None:
        spy = _SpyBackend()
        assert RedactingBackend(spy).list_models() == ["m"]


# ── Factory wiring ──────────────────────────────────────────────────────


@dataclass
class _Cfg:
    llm_provider: str = "openai_compatible"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = "sk-or-test"
    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_provider: str = "ollama"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    auto_redact_before_cloud: bool = True


class TestFactoryRedactionWiring:

    @pytest.mark.unit
    def test_openai_compatible_wrapped_when_flag_on(self) -> None:
        from jarvis.llm.factory import get_llm_backend
        backend = get_llm_backend(_Cfg(auto_redact_before_cloud=True))
        assert isinstance(backend, RedactingBackend), (
            "openai_compatible chat backend must be wrapped in "
            "RedactingBackend when auto_redact_before_cloud is on"
        )

    @pytest.mark.unit
    def test_openai_compatible_not_wrapped_when_flag_off(self) -> None:
        from jarvis.llm.factory import get_llm_backend
        from jarvis.llm.openai_compatible import OpenAICompatibleBackend
        backend = get_llm_backend(_Cfg(auto_redact_before_cloud=False))
        assert isinstance(backend, OpenAICompatibleBackend)
        assert not isinstance(backend, RedactingBackend)

    @pytest.mark.unit
    def test_ollama_never_wrapped(self) -> None:
        """Local Ollama prompts never leave the machine — no redaction
        overhead even when the flag is on."""
        from jarvis.llm.factory import get_llm_backend
        from jarvis.llm.ollama import OllamaBackend
        cfg = _Cfg(llm_provider="ollama", auto_redact_before_cloud=True)
        backend = get_llm_backend(cfg)
        assert isinstance(backend, OllamaBackend)
        assert not isinstance(backend, RedactingBackend)

    @pytest.mark.unit
    def test_local_embedding_not_wrapped(self) -> None:
        """Embeddings pinned to Ollama (our default for the cloud-chat
        setup) stay local and unwrapped even with redaction on."""
        from jarvis.llm.factory import get_embedding_backend
        from jarvis.llm.ollama import OllamaBackend
        cfg = _Cfg(embedding_provider="ollama", auto_redact_before_cloud=True)
        backend = get_embedding_backend(cfg)
        assert isinstance(backend, OllamaBackend)
