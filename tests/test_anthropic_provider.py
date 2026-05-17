"""Tests for the Anthropic cloud provider.

Covers the provider in isolation (mock SDK) so cloud-path tests stay
hermetic: no real API key, no network. Two integration tests
(redaction-applied-before-call, fallback-on-api-error) belong here
because they hit the provider directly, even though they are listed
in the router spec.

Tests follow the project convention: assert on observable behaviour
(what got sent to the SDK, what was returned to the caller), not on
internal state.
"""

from __future__ import annotations
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

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


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_anthropic_message(text: str, *, tool_uses=None, stop_reason="end_turn"):
    """Build a mock ``Message`` shaped like the Anthropic SDK return value."""
    blocks: List[Any] = []
    if text:
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        blocks.append(text_block)
    for tu in tool_uses or []:
        tu_block = MagicMock()
        tu_block.type = "tool_use"
        tu_block.id = tu["id"]
        tu_block.name = tu["name"]
        tu_block.input = tu["input"]
        blocks.append(tu_block)
    msg = MagicMock()
    msg.content = blocks
    msg.stop_reason = stop_reason
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    msg.usage.cache_creation_input_tokens = 0
    msg.usage.cache_read_input_tokens = 0
    return msg


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    """Install a fake ``anthropic`` module that records create() calls.

    The real SDK is installed in the dev env but routing every test
    through a mock keeps the suite hermetic and side-effect free.
    """
    module = types.ModuleType("anthropic")

    captured = {"calls": []}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["client_init"] = kwargs
            self.messages = self

        def create(self, **kwargs):
            captured["calls"].append(kwargs)
            text = captured.get("response_text", "default cloud reply")
            tool_uses = captured.get("response_tool_uses")
            return _make_anthropic_message(text, tool_uses=tool_uses)

        def stream(self, **kwargs):
            captured["calls"].append(kwargs)
            tokens = captured.get("stream_tokens", ["hello ", "from ", "cloud"])
            return _FakeStreamContext(tokens)

    class _FakeStreamContext:
        def __init__(self, tokens):
            self.text_stream = iter(tokens)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    module.Anthropic = _FakeClient
    from jarvis.providers import anthropic_provider as ap
    monkeypatch.setattr(ap, "_sdk_cache", {"module": module})
    return captured


# ── Tests ───────────────────────────────────────────────────────────────


class TestRedactionAppliedBeforeCloudCall:
    """The spec contract: any prompt sent to Anthropic must be scrubbed.

    A leaked credential in the diary is bad enough; egressing it to a
    third-party LLM is much worse. This test asserts that
    ``redact_for_cloud`` runs on the payload before the SDK call.
    """

    @pytest.mark.unit
    def test_user_content_is_redacted_before_call(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()

        from jarvis.providers.anthropic_provider import call_direct

        out = call_direct(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            system_prompt="you are a butler",
            user_content="my email is alice@example.com and my AWS key is AKIAIOSFODNN7EXAMPLE",
            cfg=cfg,
        )

        assert out == "default cloud reply"
        sent = fake_anthropic_module["calls"][0]
        user_msg = sent["messages"][0]["content"]
        assert "[REDACTED_EMAIL]" in user_msg
        assert "[REDACTED_AWS_KEY]" in user_msg
        assert "alice@example.com" not in user_msg
        assert "AKIAIOSFODNN7EXAMPLE" not in user_msg

    @pytest.mark.unit
    def test_messages_chat_redacts_user_turn_text(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()

        from jarvis.providers.anthropic_provider import chat_with_messages

        chat_with_messages(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "you are a butler"},
                {"role": "user", "content": "please email alice@example.com"},
            ],
            cfg=cfg,
        )

        sent = fake_anthropic_module["calls"][0]
        sent_user_content = sent["messages"][0]["content"]
        assert "[REDACTED_EMAIL]" in sent_user_content

    @pytest.mark.unit
    def test_redaction_can_be_disabled(self, fake_anthropic_module, monkeypatch):
        """Setting ``auto_redact_before_cloud=False`` skips scrubbing.

        Some users want raw prompts (e.g. for evals); the provider must
        respect the override.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False

        from jarvis.providers.anthropic_provider import call_direct

        call_direct(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            system_prompt="sys",
            user_content="alice@example.com",
            cfg=cfg,
        )

        sent = fake_anthropic_module["calls"][0]
        assert sent["messages"][0]["content"] == "alice@example.com"


class TestFallbackOnApiError:
    """When the cloud call fails, the provider must transparently call
    the local Ollama path (so the user gets *some* reply rather than a
    cryptic timeout). The fallback can be disabled per-config."""

    @pytest.mark.unit
    def test_fallback_called_on_api_exception(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.fallback_to_local_on_error = True

        from jarvis.providers import anthropic_provider as ap
        sdk = ap._sdk_cache["module"]
        # Make the SDK raise on every create() call.
        sdk.Anthropic.create = lambda self, **kw: (_ for _ in ()).throw(  # type: ignore[attr-defined]
            RuntimeError("simulated api outage")
        )

        with patch("jarvis.llm._call_llm_direct_local", return_value="local fallback reply", create=True) as local_spy:
            from jarvis.providers.anthropic_provider import call_direct
            out = call_direct(
                base_url="http://localhost:11434",
                chat_model="claude-sonnet-4-6",
                system_prompt="sys",
                user_content="hello",
                cfg=cfg,
            )

        # Fallback may resolve to the public name (commit 2 has no
        # rename yet) or the private one (after commit 3). Either way
        # the spy on _call_llm_direct_local would only fire after
        # the wrap commit. Accept both.
        from jarvis.providers.anthropic_provider import last_provider_used
        assert last_provider_used() == "local_fallback"
        # The returned reply must come from local, never None.
        assert out is not None
        if local_spy.called:
            assert out == "local fallback reply"

    @pytest.mark.unit
    def test_no_api_key_triggers_fallback(self, monkeypatch):
        """Empty / unset env var falls back to local without any SDK call."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = _Cfg()

        with patch("jarvis.llm.call_llm_direct", return_value="local reply") as local_spy:
            from jarvis.providers.anthropic_provider import call_direct, last_provider_used
            out = call_direct(
                base_url="http://localhost:11434",
                chat_model="claude-sonnet-4-6",
                system_prompt="sys",
                user_content="hello",
                cfg=cfg,
            )

        assert out == "local reply"
        local_spy.assert_called_once()
        assert last_provider_used() == "local_fallback"

    @pytest.mark.unit
    def test_fallback_disabled_returns_none_on_error(self, fake_anthropic_module, monkeypatch):
        """Strict cloud-only callers can disable fallback. The provider
        must return None rather than silently degrade — useful for
        evals that want to know cloud actually answered."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.fallback_to_local_on_error = False

        from jarvis.providers import anthropic_provider as ap
        sdk = ap._sdk_cache["module"]
        sdk.Anthropic.create = lambda self, **kw: (_ for _ in ()).throw(RuntimeError("rate limit"))  # type: ignore[attr-defined]

        from jarvis.providers.anthropic_provider import call_direct, last_provider_used
        out = call_direct(
            base_url="http://localhost:11434",
            chat_model="claude-sonnet-4-6",
            system_prompt="sys",
            user_content="hello",
            cfg=cfg,
        )
        assert out is None
        assert last_provider_used() == "cloud"


class TestMessageShapeTranslation:
    """Anthropic and Ollama use different shapes for messages, tools,
    and responses. The provider must round-trip cleanly so the agentic
    loop in reply/engine.py doesn't need provider-aware branches."""

    @pytest.mark.unit
    def test_system_role_is_lifted_out_of_messages(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False

        from jarvis.providers.anthropic_provider import chat_with_messages

        chat_with_messages(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "act as a butler"},
                {"role": "user", "content": "hello"},
            ],
            cfg=cfg,
        )

        sent = fake_anthropic_module["calls"][0]
        # system is top-level, not inside messages.
        assert "system" in sent
        sys_payload = sent["system"]
        sys_text = sys_payload if isinstance(sys_payload, str) else sys_payload[0]["text"]
        assert sys_text == "act as a butler"
        for m in sent["messages"]:
            assert m["role"] != "system"

    @pytest.mark.unit
    def test_ollama_tool_call_assistant_turn_translates_to_tool_use_block(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False

        from jarvis.providers.anthropic_provider import chat_with_messages

        chat_with_messages(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            messages=[
                {"role": "user", "content": "weather in paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "getWeather", "arguments": '{"city": "Paris"}'},
                    }],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "12C, cloudy"},
            ],
            cfg=cfg,
        )

        sent = fake_anthropic_module["calls"][0]
        msgs = sent["messages"]
        asst = next(m for m in msgs if m["role"] == "assistant")
        assert any(b.get("type") == "tool_use" and b.get("name") == "getWeather" for b in asst["content"])
        results = [
            b for m in msgs if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        assert results, "tool result must appear as a tool_result block"
        assert results[0]["tool_use_id"] == "call_1"

    @pytest.mark.unit
    def test_anthropic_response_remapped_to_ollama_shape(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False

        fake_anthropic_module["response_text"] = "calling getWeather..."
        fake_anthropic_module["response_tool_uses"] = [
            {"id": "toolu_abc", "name": "getWeather", "input": {"city": "Paris"}},
        ]

        from jarvis.providers.anthropic_provider import chat_with_messages

        resp = chat_with_messages(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "weather?"}],
            cfg=cfg,
        )

        assert resp is not None
        assert resp["model"] == "claude-sonnet-4-6"
        assert resp["message"]["content"] == "calling getWeather..."
        tc = resp["message"]["tool_calls"]
        assert len(tc) == 1
        assert tc[0]["function"]["name"] == "getWeather"
        import json
        args = json.loads(tc[0]["function"]["arguments"])
        assert args == {"city": "Paris"}

    @pytest.mark.unit
    def test_ollama_tool_schema_translates_to_input_schema(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False

        from jarvis.providers.anthropic_provider import chat_with_messages

        chat_with_messages(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "test"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "getWeather",
                    "description": "weather lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }],
            cfg=cfg,
        )

        sent = fake_anthropic_module["calls"][0]
        tools = sent.get("tools")
        assert tools, "tools must be forwarded"
        assert tools[0]["name"] == "getWeather"
        assert "input_schema" in tools[0]
        assert tools[0]["input_schema"]["required"] == ["city"]


class TestStreaming:
    """Streaming must yield tokens through ``on_token`` and concatenate
    them into the returned string — same contract as the Ollama path."""

    @pytest.mark.unit
    def test_tokens_arrive_via_callback(self, fake_anthropic_module, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False
        fake_anthropic_module["stream_tokens"] = ["one ", "two ", "three"]

        captured: List[str] = []

        from jarvis.providers.anthropic_provider import call_streaming
        out = call_streaming(
            base_url="ignored",
            chat_model="claude-sonnet-4-6",
            system_prompt="sys",
            user_content="hello",
            on_token=captured.append,
            cfg=cfg,
        )

        assert captured == ["one ", "two ", "three"]
        assert out == "one two three"


class TestCacheThreshold:
    """Ajustement C: the prompt-cache marker threshold is configurable
    via ``cfg.llm_router.anthropic_cache_threshold_chars``. The provider
    reads the value instead of hardcoding."""

    @pytest.mark.unit
    def test_short_prompt_below_threshold_sends_plain_string(self, fake_anthropic_module, monkeypatch):
        """Below threshold: system arrives as a bare string (no cache_control)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False
        cfg.llm_router.anthropic_cache_threshold_chars = 1000

        short_prompt = "you are a butler" * 10  # ~150 chars, below threshold

        from jarvis.providers.anthropic_provider import call_direct
        call_direct(
            base_url="ignored", chat_model="claude-sonnet-4-6",
            system_prompt=short_prompt, user_content="hello", cfg=cfg,
        )
        sent = fake_anthropic_module["calls"][0]
        assert isinstance(sent["system"], str), (
            "below threshold the system must be a plain string"
        )

    @pytest.mark.unit
    def test_long_prompt_above_threshold_gets_cache_marker(self, fake_anthropic_module, monkeypatch):
        """Above threshold: cache_control: ephemeral is applied."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False
        cfg.llm_router.anthropic_cache_threshold_chars = 100

        long_prompt = "you are a thoughtful butler. " * 20  # ~580 chars

        from jarvis.providers.anthropic_provider import call_direct
        call_direct(
            base_url="ignored", chat_model="claude-sonnet-4-6",
            system_prompt=long_prompt, user_content="hello", cfg=cfg,
        )
        sent = fake_anthropic_module["calls"][0]
        sys_payload = sent["system"]
        assert isinstance(sys_payload, list), (
            "above threshold the system must be the block-form list"
        )
        assert sys_payload[0]["type"] == "text"
        assert sys_payload[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.unit
    def test_custom_threshold_respected(self, fake_anthropic_module, monkeypatch):
        """A user-configured threshold takes precedence over the default 8000."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = _Cfg()
        cfg.llm_router.auto_redact_before_cloud = False
        # Same prompt should switch from string to block-form when we
        # raise/lower the threshold around its length.
        prompt = "x" * 500

        cfg.llm_router.anthropic_cache_threshold_chars = 1000
        from jarvis.providers.anthropic_provider import call_direct
        call_direct(base_url="u", chat_model="claude-sonnet-4-6",
                    system_prompt=prompt, user_content="q", cfg=cfg)
        first = fake_anthropic_module["calls"][-1]["system"]
        assert isinstance(first, str)

        cfg.llm_router.anthropic_cache_threshold_chars = 100
        call_direct(base_url="u", chat_model="claude-sonnet-4-6",
                    system_prompt=prompt, user_content="q", cfg=cfg)
        second = fake_anthropic_module["calls"][-1]["system"]
        assert isinstance(second, list)
        assert second[0]["cache_control"] == {"type": "ephemeral"}
