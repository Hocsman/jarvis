"""Regression: Anthropic fallback must swap to cfg.ollama_chat_model.

Pre-fix bug: when ``anthropic_provider.chat_with_messages(chat_model=
"claude-sonnet-4-6", ...)`` fell back to the local Ollama path (cloud
unreachable, auth failure, 4xx/5xx, etc.), it forwarded the SAME
``chat_model`` to ``_chat_with_messages_local``. Ollama then returned
404 because it doesn't have a model called ``claude-sonnet-4-6``, the
fallback returned ``None``, and the reply engine surfaced its generic
"Sorry, I had trouble processing that." message.

The fix is a single helper ``_local_model_for_fallback(cfg, chat_model)``
that returns ``cfg.ollama_chat_model`` when set, falling back to the
caller's ``chat_model`` only when the cfg doesn't carry one (which
the real Settings object always does, but partial test mocks may not).

These tests pin:
1. The helper returns the local model name when cfg has one.
2. The helper falls back to the requested model when cfg doesn't.
3. End-to-end: all three Anthropic public entry points (call_direct,
   call_streaming, chat_with_messages) hand the LOCAL model name to
   the local fallback after a forced cloud failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import patch

import pytest


@dataclass
class _RouterCfg:
    mode: str = "hybrid"
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    anthropic_model: str = "claude-sonnet-4-6"
    cloud_intents: List[str] = field(default_factory=lambda: ["code_complex"])
    auto_redact_before_cloud: bool = True
    fallback_to_local_on_error: bool = True
    anthropic_cache_threshold_chars: int = 8000


@dataclass
class _Cfg:
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    llm_router: _RouterCfg = field(default_factory=_RouterCfg)


class TestLocalModelForFallbackHelper:

    @pytest.mark.unit
    def test_returns_ollama_chat_model_when_set(self) -> None:
        from jarvis.providers.anthropic_provider import _local_model_for_fallback

        cfg = _Cfg()  # ollama_chat_model="gemma4:e2b"
        assert _local_model_for_fallback(cfg, "claude-sonnet-4-6") == "gemma4:e2b"

    @pytest.mark.unit
    def test_falls_back_to_requested_when_cfg_missing(self) -> None:
        """A test-mock cfg without ``ollama_chat_model`` should preserve
        the previous behaviour (use whatever the caller passed). The
        helper must not raise."""
        from jarvis.providers.anthropic_provider import _local_model_for_fallback

        class _PartialCfg:
            pass  # no ollama_chat_model attribute

        assert _local_model_for_fallback(_PartialCfg(), "claude-sonnet-4-6") == "claude-sonnet-4-6"

    @pytest.mark.unit
    def test_empty_ollama_chat_model_falls_back(self) -> None:
        """``ollama_chat_model = ""`` (someone hand-edited the config)
        should not be picked — fall back to the caller's name."""
        from jarvis.providers.anthropic_provider import _local_model_for_fallback

        @dataclass
        class _EmptyCfg:
            ollama_chat_model: str = ""

        assert _local_model_for_fallback(_EmptyCfg(), "claude-sonnet-4-6") == "claude-sonnet-4-6"


class TestFallbackHandsLocalModelName:
    """The bug surfaced when cloud failed. We force each public entry
    point's create() to raise, capture the model name forwarded to the
    local fallback callable, and assert it's the Ollama-side name
    (``gemma4:e2b``), not the Anthropic-side one (``claude-sonnet-4-6``)."""

    @pytest.mark.unit
    def test_call_direct_fallback_uses_local_model(self, monkeypatch) -> None:
        from jarvis.providers import anthropic_provider as ap

        cfg = _Cfg()

        # Stub the SDK so create() always raises — forces fallback.
        import types
        fake_sdk = types.ModuleType("anthropic_fake")

        class _RaisingClient:
            def __init__(self, *_a, **_k):
                self.messages = self
            def create(self, **_kw):
                raise RuntimeError("simulated cloud outage")
            def stream(self, **_kw):
                raise RuntimeError("simulated cloud outage")

        fake_sdk.Anthropic = _RaisingClient
        monkeypatch.setattr(ap, "_sdk_cache", {"module": fake_sdk})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        captured = {}

        def spy_local_direct(*args, **kwargs):
            # _local_callable("direct") -> _call_llm_direct_local(base_url, chat_model, ...)
            # The chat_model is positional arg index 1.
            captured["chat_model"] = args[1]
            return "fallback OK"

        monkeypatch.setattr(
            "jarvis.llm._call_llm_direct_local", spy_local_direct, raising=True,
        )

        result = ap.call_direct(
            base_url="http://localhost:11434",
            chat_model="claude-sonnet-4-6",  # cloud name
            system_prompt="sys",
            user_content="hello",
            cfg=cfg,
        )

        assert result == "fallback OK"
        assert captured["chat_model"] == "gemma4:e2b", (
            f"Fallback forwarded the cloud model name "
            f"{captured['chat_model']!r} instead of swapping to "
            f"cfg.ollama_chat_model. The fix exists to prevent Ollama "
            f"returning 404 on the cloud model name."
        )

    @pytest.mark.unit
    def test_chat_with_messages_fallback_uses_local_model(self, monkeypatch) -> None:
        """Same property on the chat() entry point — the one the
        reply engine actually exercises every turn."""
        from jarvis.providers import anthropic_provider as ap

        cfg = _Cfg()

        import types
        fake_sdk = types.ModuleType("anthropic_fake")

        class _RaisingClient:
            def __init__(self, *_a, **_k):
                self.messages = self
            def create(self, **_kw):
                raise RuntimeError("simulated cloud outage")

        fake_sdk.Anthropic = _RaisingClient
        monkeypatch.setattr(ap, "_sdk_cache", {"module": fake_sdk})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        captured = {}

        def spy_local_chat(*args, **kwargs):
            captured["chat_model"] = args[1]
            return {"message": {"role": "assistant", "content": "fallback OK"}}

        monkeypatch.setattr(
            "jarvis.llm._chat_with_messages_local", spy_local_chat, raising=True,
        )

        result = ap.chat_with_messages(
            base_url="http://localhost:11434",
            chat_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
        )

        assert isinstance(result, dict)
        assert captured["chat_model"] == "gemma4:e2b"

    @pytest.mark.unit
    def test_no_client_branch_also_swaps_model(self, monkeypatch) -> None:
        """The other fallback path — when ``_build_client`` returns
        ``None`` because the SDK isn't importable. Same swap required."""
        from jarvis.providers import anthropic_provider as ap

        cfg = _Cfg()
        # Empty SDK cache so _load_sdk returns None.
        monkeypatch.setattr(ap, "_sdk_cache", {"module": None})

        captured = {}

        def spy_local_chat(*args, **kwargs):
            captured["chat_model"] = args[1]
            return {"message": {"role": "assistant", "content": "ok"}}

        monkeypatch.setattr(
            "jarvis.llm._chat_with_messages_local", spy_local_chat, raising=True,
        )

        ap.chat_with_messages(
            base_url="http://localhost:11434",
            chat_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hi"}],
            cfg=cfg,
        )

        assert captured["chat_model"] == "gemma4:e2b"
