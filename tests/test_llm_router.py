"""Tests for the hybrid LLM router.

Covers the contracts that must hold regardless of which provider is wired:
- ``local_only`` mode never invokes the classifier (zero-overhead privacy
  contract).
- Hybrid routing for non-cloud intents stays on the local callable.
- Redaction wraps the project's existing ``utils.redact.redact`` rules.
- LRU cache hits avoid a second classifier call for the same prompt.

Provider-dependent contracts (cloud routing, fallback on API error,
end-to-end redaction at egress) live in the integration test file
since they require the Anthropic provider to exist.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import pytest

from jarvis import llm_router
from jarvis.llm_router import (
    Intent,
    MODE_LOCAL_ONLY,
    MODE_HYBRID,
    RouterDecision,
    classify_intent,
    clear_decision_cache,
    redact_for_cloud,
    route,
)


@dataclass
class _RouterCfg:
    """Minimal stand-in for ``Settings.llm_router``."""
    mode: str = MODE_LOCAL_ONLY
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"
    anthropic_model: str = "claude-sonnet-4-6"
    cloud_intents: List[str] = field(default_factory=lambda: [
        "code_complex", "multi_step_reasoning", "tool_use_chain",
    ])
    auto_redact_before_cloud: bool = True
    fallback_to_local_on_error: bool = True
    anthropic_cache_threshold_chars: int = 8000


@dataclass
class _Cfg:
    """Minimal cfg object for router tests."""
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    intent_judge_model: str = "gemma4:e2b"
    tool_router_model: str = ""
    llm_digest_timeout_sec: float = 8.0
    llm_router: _RouterCfg = field(default_factory=_RouterCfg)


@pytest.fixture(autouse=True)
def _reset_router_cache():
    """Each test starts with a clean decision cache so cache-related
    assertions are not polluted by ordering."""
    clear_decision_cache()
    yield
    clear_decision_cache()


class TestLocalOnlyZeroOverhead:
    """The privacy contract: ``local_only`` MUST NOT classify, ever.

    A user who stays on default settings should pay literally zero
    overhead for the router being installed — no LLM call, no cache
    touch, no cloud module import.
    """

    @pytest.mark.unit
    def test_local_only_never_classifies(self):
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_LOCAL_ONLY

        with patch("jarvis.llm_router._call_classifier_llm") as classifier_spy:
            callable_, model = route("write me a bash script", None, cfg, call_kind="chat")

        assert classifier_spy.call_count == 0, (
            "local_only must short-circuit before any classifier call"
        )
        assert model == "gemma4:e2b"
        assert callable_ is not None

    @pytest.mark.unit
    def test_local_only_skips_classification_for_cloud_intent_phrasing(self):
        """Even an obvious code-complex prompt stays local in local_only mode."""
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_LOCAL_ONLY

        with patch("jarvis.llm_router._call_classifier_llm") as classifier_spy:
            for prompt in (
                "refactor this Python class into composition pattern",
                "debug the segfault in my Rust async runtime",
                "compare options A B and C across latency cost privacy",
            ):
                _ = route(prompt, ["previous context line"], cfg)
        assert classifier_spy.call_count == 0

    @pytest.mark.unit
    def test_local_only_returns_existing_ollama_callable(self):
        """The returned callable must be the unmodified Ollama function so
        existing behaviour is byte-identical for default users."""
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_LOCAL_ONLY

        from jarvis import llm as _llm
        callable_, _model = route("hello", None, cfg, call_kind="chat")
        # In commit 1 the wrapping hasn't happened yet, so the local
        # callable must be the public ``chat_with_messages`` itself.
        # Once integration lands, ``_chat_with_messages_local`` will
        # be preferred; either way the underlying behaviour is the
        # original Ollama implementation.
        assert callable_ in (
            getattr(_llm, "_chat_with_messages_local", None),
            _llm.chat_with_messages,
        )


class TestHybridRoutesChatToLocal:
    """Hybrid mode: ``casual_chat`` and ``simple_query`` stay local."""

    @pytest.mark.unit
    def test_casual_chat_intent_routes_to_local(self):
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "casual_chat", "confidence": 0.9},
        ):
            callable_, model = route("hey jarvis what's up", None, cfg)
        assert model == "gemma4:e2b"
        from jarvis import llm as _llm
        assert callable_ in (
            getattr(_llm, "_chat_with_messages_local", None),
            _llm.chat_with_messages,
        )

    @pytest.mark.unit
    def test_simple_query_intent_routes_to_local(self):
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "simple_query", "confidence": 0.8},
        ):
            _callable, model = route("what time is it", None, cfg)
        assert model == "gemma4:e2b"

    @pytest.mark.unit
    def test_classifier_failure_falls_back_to_local(self):
        """When the classifier returns None (timeout, bad JSON, unknown
        intent), the router MUST default to local — never bill the user
        for a cloud call we have no signal to justify."""
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        with patch("jarvis.llm_router._call_classifier_llm", return_value=None):
            _callable, model = route("anything goes here", None, cfg)
        assert model == "gemma4:e2b"


class TestRedactionDelegation:
    """``redact_for_cloud`` must reuse the project's redaction pipeline,
    never re-implement scrub rules. Drift would mean cloud egress sees
    secrets that the diary already masks."""

    @pytest.mark.unit
    def test_redaction_masks_email(self):
        scrubbed = redact_for_cloud("please contact alice@example.com about the meeting")
        assert "[REDACTED_EMAIL]" in scrubbed
        assert "alice@example.com" not in scrubbed

    @pytest.mark.unit
    def test_redaction_masks_aws_key(self):
        scrubbed = redact_for_cloud(
            "the access key is AKIAIOSFODNN7EXAMPLE for the bucket"
        )
        assert "[REDACTED_AWS_KEY]" in scrubbed
        assert "AKIA" not in scrubbed

    @pytest.mark.unit
    def test_redaction_delegates_to_project_redact(self):
        """If utils.redact.redact gains a new pattern, redact_for_cloud
        must pick it up automatically — meaning it really calls into
        the same function, not a local copy."""
        with patch("jarvis.llm_router.redact", return_value="SCRUBBED") as project_redact:
            out = redact_for_cloud("any input")
        assert out == "SCRUBBED"
        project_redact.assert_called_once_with("any input")

    @pytest.mark.unit
    def test_redaction_handles_empty_input(self):
        assert redact_for_cloud("") == ""

    @pytest.mark.unit
    def test_redaction_masks_slack_token(self):
        """Slack xoxb-/xoxp- tokens must never survive redaction. The
        keyword-anchored credentials pattern matches first when the
        sentence contains ``token:``, but the underlying token bytes
        get scrubbed regardless of which label the project uses. This
        test asserts the privacy guarantee (raw token gone), not the
        bucket name (which is an implementation choice in utils.redact)."""
        scrubbed = redact_for_cloud("connection string xoxb-12345abcdefg67890ABC")
        assert "xoxb-" not in scrubbed
        assert "[REDACTED" in scrubbed  # any redacted label is fine


class TestIntentCacheHits:
    """LRU(256) on prompt+context+model hash so the same query inside one
    conversation never pays the classifier twice."""

    @pytest.mark.unit
    def test_second_identical_call_uses_cache(self):
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        prompt = "refactor the auth middleware into a smaller surface"
        context = ["assistant: previous answer", "user: keep going"]

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ) as classifier_spy:
            d1 = classify_intent(prompt, context, cfg)
            d2 = classify_intent(prompt, context, cfg)
        assert classifier_spy.call_count == 1, (
            "second identical classification must hit the LRU cache"
        )
        assert d1.reason == d2.reason == "code_complex"
        assert d1.intent_score == d2.intent_score == pytest.approx(0.95)

    @pytest.mark.unit
    def test_different_prompts_miss_cache(self):
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "simple_query", "confidence": 0.6},
        ) as classifier_spy:
            classify_intent("what time is it", None, cfg)
            classify_intent("what's the weather", None, cfg)
            classify_intent("convert 42c to fahrenheit", None, cfg)
        assert classifier_spy.call_count == 3, (
            "distinct prompts must each trigger a classification"
        )

    @pytest.mark.unit
    def test_context_change_invalidates_cache(self):
        """A prompt's meaning depends on surrounding dialogue, so the
        same query string with new context is a new cache key."""
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "casual_chat", "confidence": 0.5},
        ) as classifier_spy:
            classify_intent("yes", ["assistant: ready?"], cfg)
            classify_intent("yes", ["assistant: shall we delete the database"], cfg)
        assert classifier_spy.call_count == 2

    @pytest.mark.unit
    def test_classifier_model_change_invalidates_cache(self):
        """If the warm model is swapped (e.g. user reconfigures
        ``tool_router_model``), prior decisions must NOT be served from
        cache — the new model may classify differently."""
        cfg = _Cfg()
        cfg.llm_router.mode = MODE_HYBRID
        prompt = "compare yarn vs npm vs pnpm for monorepos"
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "multi_step_reasoning", "confidence": 0.8},
        ) as classifier_spy:
            cfg.intent_judge_model = "gemma4:e2b"
            classify_intent(prompt, None, cfg)
            cfg.intent_judge_model = "phi3:mini"
            classify_intent(prompt, None, cfg)
        assert classifier_spy.call_count == 2


class TestRouterDecisionShape:
    """Surface checks on the public dataclass."""

    @pytest.mark.unit
    def test_decision_is_immutable(self):
        d = RouterDecision(provider="local", model="m", reason="r", intent_score=0.5)
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            d.provider = "cloud"  # type: ignore[misc]

    @pytest.mark.unit
    def test_intent_enum_values(self):
        # Defensive: spec lists exactly these six labels. Any drift here
        # would silently change the classifier prompt vs route() mapping.
        assert {i.value for i in Intent} == {
            "casual_chat",
            "simple_query",
            "code_complex",
            "multi_step_reasoning",
            "tool_use_chain",
            "stop_command",
        }
