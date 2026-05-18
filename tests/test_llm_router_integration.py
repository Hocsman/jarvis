"""End-to-end integration tests for the hybrid LLM router.

These exercise the public ``jarvis.llm`` surface so the wrapping in
``llm.py`` is verified together with the router and the provider.
Local-only unit tests live in ``test_llm_router.py``; provider-only
unit tests live in ``test_anthropic_provider.py``. This file covers
the contracts that cross all three layers.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _RouterCfg:
    mode: str = "hybrid"
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
    db_path: str = ":memory:"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    intent_judge_model: str = "gemma4:e2b"
    tool_router_model: str = ""
    llm_digest_timeout_sec: float = 8.0
    llm_router: _RouterCfg = field(default_factory=_RouterCfg)


@pytest.fixture
def tmp_db_path():
    """Real disk-backed SQLite so telemetry writes survive a connect."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _reset_router_state():
    from jarvis import llm_router, llm, llm_router_telemetry
    llm_router.clear_decision_cache()
    llm.set_router_cfg_for_tests(None)
    llm_router_telemetry._reset_initialised_paths()
    yield
    llm.set_router_cfg_for_tests(None)
    llm_router.clear_decision_cache()
    llm_router_telemetry._reset_initialised_paths()


@pytest.fixture
def fake_anthropic(monkeypatch):
    module = types.ModuleType("anthropic")
    captured: dict = {"calls": []}

    def _msg(text="cloud reply"):
        m = MagicMock()
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        m.content = [text_block]
        m.stop_reason = "end_turn"
        m.usage.input_tokens = 200
        m.usage.output_tokens = 80
        m.usage.cache_creation_input_tokens = 0
        m.usage.cache_read_input_tokens = 0
        return m

    class FakeClient:
        def __init__(self, **kw):
            self.messages = self

        def create(self, **kw):
            captured["calls"].append(kw)
            if captured.get("raise"):
                raise RuntimeError("simulated outage")
            return _msg(captured.get("response_text", "cloud reply"))

    module.Anthropic = FakeClient

    from jarvis.providers import anthropic_provider as ap
    monkeypatch.setattr(ap, "_sdk_cache", {"module": module})
    return captured


class TestHybridRoutesCodeToCloud:
    """In hybrid mode, a code_complex query must reach the cloud provider
    end-to-end through ``jarvis.llm.call_llm_direct``."""

    @pytest.mark.unit
    def test_code_complex_query_lands_on_anthropic(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm

        llm.set_router_cfg_for_tests(_Cfg())

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ):
            out = llm.call_llm_direct(
                "http://localhost:11434",
                "gemma4:e2b",
                "you are a butler",
                "refactor this Python class to use composition",
            )

        assert out == "cloud reply"
        assert len(fake_anthropic["calls"]) == 1, (
            "code_complex intent should fire exactly one Anthropic call"
        )
        assert fake_anthropic["calls"][0]["model"] == "claude-sonnet-4-6"

    @pytest.mark.unit
    def test_casual_chat_does_not_call_cloud(self, fake_anthropic, monkeypatch):
        """The other half of the routing contract: chat-shaped queries
        must stay on Ollama even when the cloud is available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm

        llm.set_router_cfg_for_tests(_Cfg())

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "casual_chat", "confidence": 0.9},
        ), patch(
            "jarvis.llm._call_llm_direct_local", return_value="local reply"
        ) as local_spy:
            out = llm.call_llm_direct(
                "http://localhost:11434", "gemma4:e2b",
                "sys", "hey what's up",
            )
        assert out == "local reply"
        assert len(fake_anthropic["calls"]) == 0
        local_spy.assert_called_once()


class TestEndToEndRedaction:
    """The privacy contract on the full call chain: a query with secrets
    that the router classifies as cloud-bound must reach Anthropic
    *redacted*, never raw."""

    @pytest.mark.unit
    def test_secrets_redacted_before_egress(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm

        llm.set_router_cfg_for_tests(_Cfg())

        secret_prompt = (
            "Help me debug this auth flow. My token is AKIAIOSFODNN7EXAMPLE "
            "and the user is alice@example.com."
        )
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ):
            llm.call_llm_direct(
                "http://localhost:11434", "gemma4:e2b",
                "you are a butler", secret_prompt,
            )

        sent_user_content = fake_anthropic["calls"][0]["messages"][0]["content"]
        assert "AKIAIOSFODNN7EXAMPLE" not in sent_user_content
        assert "alice@example.com" not in sent_user_content
        assert "[REDACTED_AWS_KEY]" in sent_user_content
        assert "[REDACTED_EMAIL]" in sent_user_content


class TestFallbackEndToEnd:
    """When the cloud call fails inside the routed path, the user still
    gets a reply from local Ollama."""

    @pytest.mark.unit
    def test_cloud_failure_falls_back_to_local(self, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm

        llm.set_router_cfg_for_tests(_Cfg())
        fake_anthropic["raise"] = True

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.9},
        ), patch(
            "jarvis.llm._call_llm_direct_local", return_value="local fallback reply"
        ) as local_spy:
            out = llm.call_llm_direct(
                "http://localhost:11434", "gemma4:e2b",
                "sys", "refactor my code",
            )

        assert out == "local fallback reply"
        local_spy.assert_called_once()


class TestLocalOnlyZeroOverheadEndToEnd:
    """Default users (mode=local_only) must see byte-identical behaviour
    to the pre-router world: the router never touches their requests."""

    @pytest.mark.unit
    def test_local_only_bypasses_router_entirely(self, monkeypatch):
        from jarvis import llm

        cfg = _Cfg()
        cfg.llm_router.mode = "local_only"
        llm.set_router_cfg_for_tests(cfg)

        with patch(
            "jarvis.llm_router._call_classifier_llm"
        ) as classifier_spy, patch(
            "jarvis.llm._call_llm_direct_local", return_value="ollama reply"
        ) as local_spy:
            out = llm.call_llm_direct(
                "http://localhost:11434", "gemma4:e2b",
                "sys", "anything",
            )

        assert out == "ollama reply"
        assert classifier_spy.call_count == 0
        local_spy.assert_called_once()

    @pytest.mark.unit
    def test_no_telemetry_written_in_local_only(self, tmp_db_path, monkeypatch):
        """Privacy promise: local_only users get zero new state on disk."""
        from jarvis import llm, llm_router_telemetry

        cfg = _Cfg(db_path=tmp_db_path)
        cfg.llm_router.mode = "local_only"
        llm.set_router_cfg_for_tests(cfg)

        with patch(
            "jarvis.llm._call_llm_direct_local", return_value="ok"
        ):
            llm.call_llm_direct("u", "m", "s", "u")

        rows = llm_router_telemetry.fetch_recent(tmp_db_path, limit=10)
        assert rows == [], "local_only mode must not write telemetry rows"


class TestTelemetryRecording:
    """Hybrid / cloud_only modes record one telemetry row per call.

    Privacy invariant: no prompt or response text is persisted — only
    metrics."""

    @pytest.mark.unit
    def test_one_row_per_cloud_call(self, tmp_db_path, fake_anthropic, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm, llm_router_telemetry

        cfg = _Cfg(db_path=tmp_db_path)
        llm.set_router_cfg_for_tests(cfg)

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ):
            llm.chat_with_messages(
                "u", "gemma4:e2b",
                [{"role": "user", "content": "refactor the auth class"}],
            )

        rows = llm_router_telemetry.fetch_recent(tmp_db_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["provider"] == "cloud"
        assert row["model"] == "claude-sonnet-4-6"
        assert row["intent"] == "code_complex"
        assert row["tokens_in"] == 200
        assert row["tokens_out"] == 80
        # Sonnet 4.6: 200 in * 3.00/1M + 80 out * 15.00/1M = 0.0006 + 0.0012 = 0.0018
        assert row["cost_estimate_usd"] == pytest.approx(0.0018, abs=1e-6)
        assert isinstance(row["latency_ms"], int)
        # Privacy: no prompt or reply column exists at all.
        assert "prompt" not in row
        assert "response" not in row
        assert "content" not in row

    @pytest.mark.unit
    def test_no_prompt_text_persisted(self, tmp_db_path, fake_anthropic, monkeypatch):
        """Independent verification: every column value in the row must
        not contain any substring of the original prompt."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm

        cfg = _Cfg(db_path=tmp_db_path)
        llm.set_router_cfg_for_tests(cfg)

        marker = "PRIVATE_MARKER_SHOULD_NEVER_PERSIST"
        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ):
            llm.chat_with_messages(
                "u", "gemma4:e2b",
                [{"role": "user", "content": f"please debug {marker}"}],
            )

        with sqlite3.connect(tmp_db_path) as conn:
            cur = conn.execute("SELECT * FROM llm_router_stats")
            for row in cur.fetchall():
                for cell in row:
                    if isinstance(cell, str):
                        assert marker not in cell, (
                            f"prompt content leaked into telemetry cell: {cell!r}"
                        )

    @pytest.mark.unit
    def test_fallback_row_labelled_local_fallback(self, tmp_db_path, fake_anthropic, monkeypatch):
        """Fallback path must be distinguishable from native local calls
        in telemetry so cloud reliability can be measured separately."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm, llm_router_telemetry

        cfg = _Cfg(db_path=tmp_db_path)
        llm.set_router_cfg_for_tests(cfg)
        fake_anthropic["raise"] = True

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.95},
        ), patch(
            "jarvis.llm._chat_with_messages_local",
            return_value={
                "message": {"content": "local"},
                "prompt_eval_count": 50,
                "eval_count": 10,
            },
        ):
            llm.chat_with_messages(
                "u", "gemma4:e2b",
                [{"role": "user", "content": "refactor"}],
            )

        rows = llm_router_telemetry.fetch_recent(tmp_db_path)
        assert len(rows) == 1
        assert rows[0]["provider"] == "local_fallback"
        # Local fallback bills at zero (Ollama is free), even though
        # the model name retained is the cloud one (intended target).
        assert rows[0]["cost_estimate_usd"] == 0.0

    @pytest.mark.unit
    def test_pricing_unknown_model_returns_zero_cost(self, tmp_db_path, fake_anthropic, monkeypatch):
        """Adjustment A contract: an unknown model ID must NOT crash
        the cost estimator. Telemetry should record the row with
        cost_estimate_usd=0 (best-effort).
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        from jarvis import llm, llm_router_telemetry

        cfg = _Cfg(db_path=tmp_db_path)
        # Configure a model name that pricing.py does not know about.
        cfg.llm_router.anthropic_model = "claude-future-9-9-not-yet-priced"
        llm.set_router_cfg_for_tests(cfg)

        with patch(
            "jarvis.llm_router._call_classifier_llm",
            return_value={"intent": "code_complex", "confidence": 0.9},
        ):
            llm.chat_with_messages(
                "u", "gemma4:e2b",
                [{"role": "user", "content": "anything"}],
            )

        rows = llm_router_telemetry.fetch_recent(tmp_db_path)
        assert len(rows) == 1
        assert rows[0]["provider"] == "cloud"
        assert rows[0]["model"] == "claude-future-9-9-not-yet-priced"
        # Tokens are still recorded, only the cost is zeroed out.
        assert rows[0]["tokens_in"] == 200
        assert rows[0]["tokens_out"] == 80
        assert rows[0]["cost_estimate_usd"] == 0.0

    @pytest.mark.unit
    def test_estimate_cost_directly_unknown_model(self):
        """Lower-level sanity check on the pricing API: an unknown
        model returns 0.0 from estimate_cost_usd as well, not just
        through the full call chain."""
        from jarvis import llm_router_telemetry as _telem
        assert _telem.estimate_cost_usd(
            model="completely-made-up-model-id",
            tokens_in=10_000,
            tokens_out=5_000,
        ) == 0.0


class TestSettingsParsing:
    """The new ``llm_router`` config section must round-trip through
    ``load_settings()`` so a user's config.json is honoured."""

    @pytest.mark.unit
    def test_default_settings_are_local_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "nonexistent.json"))
        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.llm_router.mode == "local_only"
        assert settings.llm_router.anthropic_model == "claude-sonnet-4-6"
        assert settings.llm_router.auto_redact_before_cloud is True
        assert settings.llm_router.fallback_to_local_on_error is True
        assert settings.llm_router.anthropic_cache_threshold_chars == 8000

    @pytest.mark.unit
    def test_hybrid_mode_round_trips(self, monkeypatch, tmp_path):
        import json
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({
            "llm_router": {
                "mode": "hybrid",
                "anthropic_model": "claude-opus-4-7",
                "cloud_intents": ["code_complex"],
                "auto_redact_before_cloud": False,
                "anthropic_cache_threshold_chars": 2048,
            }
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.llm_router.mode == "hybrid"
        assert settings.llm_router.anthropic_model == "claude-opus-4-7"
        assert settings.llm_router.cloud_intents == ["code_complex"]
        assert settings.llm_router.auto_redact_before_cloud is False
        assert settings.llm_router.anthropic_cache_threshold_chars == 2048

    @pytest.mark.unit
    def test_invalid_mode_falls_back_to_local_only(self, monkeypatch, tmp_path):
        """Defensive: a typo in mode must not yank a user into the cloud."""
        import json
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"llm_router": {"mode": "definitely_not_a_mode"}}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        from jarvis.config import load_settings
        settings = load_settings()
        assert settings.llm_router.mode == "local_only"
