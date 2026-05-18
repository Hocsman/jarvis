"""Phase 2B obligatory regression tests.

Three behaviours pinned here:

1. Whisper transcribe calls all pass ``language=None`` (auto-detect).
   The 2026-05-18 recon found this was already correct in the code,
   but a future "let's force English for speed" temptation would silently
   break multilingual workflows. This test makes that regression visible.

2. The anglophone default workflow is unchanged. Users with no
   ``response_language`` lock + default Piper voice path still get the
   English TTS clause appended to their system prompt. The Phase 2B
   refactor must NOT alter that.

3. The hybrid router maps French queries to the right provider purely
   by intent (language is not part of the router's signal). FR
   complex query -> cloud; FR trivial query -> local. Both rest on
   the same routing table that EN uses — and that's the right
   behaviour. This test pins it so we notice if a future refactor
   introduces language-aware routing without an intentional spec
   change.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import MagicMock

import pytest


# ── Fixtures ────────────────────────────────────────────────────────────


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
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    response_language: str = "français"
    tts_engine: str = "piper"
    tts_piper_model_path: str = "/voices/fr_FR-siwis-medium.onnx"
    tts_voices: dict = field(default_factory=lambda: {
        "fr": "/voices/fr_FR-siwis-medium.onnx",
        "en": "/voices/en_US-amy-medium.onnx",
    })
    llm_router: _RouterCfg = field(default_factory=_RouterCfg)


# ── Tests ───────────────────────────────────────────────────────────────


class TestWhisperAutoLanguageDetection:
    """The transcribe calls must keep auto-detection on. Forcing
    ``language="en"`` would silently break multilingual workflows
    (the FR user wouldn't notice until the locale-aware tools start
    serving English Wikipedia for French queries)."""

    @pytest.mark.unit
    def test_all_transcribe_calls_use_language_none(self) -> None:
        """Every ``.transcribe(..., language=...)`` call in
        listener.py must pass ``language=None`` (the literal value).
        A future ``language="en"`` would be a silent regression on
        the entire multilingual stack."""
        root = pathlib.Path(__file__).resolve().parents[1] / "src"
        listener_path = root / "jarvis" / "listening" / "listener.py"
        text = listener_path.read_text(encoding="utf-8")

        # Find every `language=<value>` keyword argument that appears
        # inside a `.transcribe(` call. We pin the value at None.
        transcribe_blocks = re.findall(
            r"\.transcribe\s*\([^)]*?language\s*=\s*([^,)\s]+)",
            text,
            flags=re.DOTALL,
        )
        # If the regex finds nothing it means the call pattern changed
        # and the test needs updating — fail loudly rather than silently
        # pass (which would let a real regression slip through).
        assert transcribe_blocks, (
            "Expected to find at least one .transcribe(..., language=...) "
            "call in listener.py. The call pattern probably changed and "
            "this regression test needs an update."
        )
        non_none = [v for v in transcribe_blocks if v.strip() != "None"]
        assert non_none == [], (
            f"Whisper transcribe calls must pass language=None for "
            f"auto-detect. Found non-None: {non_none}. A future "
            f"language='en' would break the multilingual workflow."
        )

    @pytest.mark.unit
    def test_detected_language_capture_logic_present(self) -> None:
        """After the transcribe call, the listener must capture
        ``result['language']`` into ``_last_detected_language``. This
        is what feeds the FR-specific tools (Wikipedia FR locale,
        per-language Piper voice) downstream."""
        root = pathlib.Path(__file__).resolve().parents[1] / "src"
        text = (root / "jarvis" / "listening" / "listener.py").read_text(encoding="utf-8")
        # Two write sites: MLX and faster-whisper paths.
        assert "self._last_detected_language = detected" in text, (
            "Listener must capture Whisper's detected language into "
            "_last_detected_language for downstream consumers."
        )


class TestEnglishWorkflowUnchanged:
    """Anglophone default users (no response_language, default Piper
    voice, default config) must keep the English TTS clause they had
    before Phase 2B. The refactor was only meant to make the clause
    *conditional* — not to remove it for the default case."""

    @pytest.mark.unit
    def test_default_config_still_appends_english_clause(self) -> None:
        """Fresh install: no response_language, no tts_voices, no
        tts_piper_model_path. _tts_language_constraint must return
        the English-lock clause (legacy behaviour preserved)."""
        from jarvis.reply.engine import _tts_language_constraint

        @dataclass
        class _DefaultCfg:
            response_language: str = ""
            tts_engine: str = "piper"
            tts_piper_model_path: str = ""

        clause = _tts_language_constraint(_DefaultCfg())
        assert clause is not None, (
            "Default anglophone config must still emit the EN clause — "
            "removing it would break TTS quality for the upstream "
            "user base."
        )
        assert "Respond in English" in clause

    @pytest.mark.unit
    def test_explicit_english_voice_still_appends_clause(self) -> None:
        """User who explicitly set an EN Piper voice but no
        response_language: clause still present (consistency with
        the default-empty case)."""
        from jarvis.reply.engine import _tts_language_constraint

        @dataclass
        class _EnCfg:
            response_language: str = ""
            tts_engine: str = "piper"
            tts_piper_model_path: str = "/voices/en_US-amy-medium.onnx"

        clause = _tts_language_constraint(_EnCfg())
        assert clause is not None
        assert "Respond in English" in clause

    @pytest.mark.unit
    def test_select_tts_voice_legacy_scalar_unchanged(self) -> None:
        """A pre-Phase-2B config with only ``tts_piper_model_path`` set
        (no ``tts_voices`` map) must resolve to that same scalar path
        — zero migration burden on existing installs."""
        from jarvis.config import select_tts_voice

        @dataclass
        class _LegacyCfg:
            tts_voices: dict = field(default_factory=dict)
            tts_piper_model_path: str = "/voices/en_US-amy-medium.onnx"
            response_language: str = ""

        # Cold start, no detected language, no map -> legacy scalar.
        resolved = select_tts_voice(_LegacyCfg(), None)
        assert resolved == "/voices/en_US-amy-medium.onnx"


class TestFrenchPromptWithHybridRouting:
    """The router classifies on intent, not language. A French
    'code_complex' query routes to cloud Sonnet; a French
    'casual_chat' routes to local Ollama. The cloud/local split is
    determined by the intent classifier, not by any FR-specific
    branch — which is the correct behaviour and what we want to pin."""

    @pytest.mark.unit
    def test_french_complex_query_routes_cloud(self, monkeypatch) -> None:
        """FR query classified as ``code_complex`` must pick the
        Anthropic provider callable + Sonnet model."""
        from jarvis import llm_router as r

        cfg = _Cfg()

        # Inject a deterministic classifier so the test doesn't depend
        # on a live Ollama instance.
        monkeypatch.setattr(
            r,
            "classify_intent",
            lambda *a, **kw: r.RouterDecision(
                provider="cloud", model="claude-sonnet-4-6",
                reason="code_complex 0.95", intent_score=0.95,
            ),
        )

        callable_, model = r.route(
            "écris-moi une fonction Python qui factorise un nombre",
            None, cfg, call_kind="chat",
        )
        assert "anthropic_provider" in (callable_.__module__ or ""), (
            f"FR code_complex must route to cloud; got module "
            f"{callable_.__module__!r}"
        )
        assert model == "claude-sonnet-4-6"

    @pytest.mark.unit
    def test_french_trivial_query_routes_local(self, monkeypatch) -> None:
        """FR query classified as ``casual_chat`` stays local — the
        cloud is reserved for intents that justify the cost."""
        from jarvis import llm_router as r

        cfg = _Cfg()

        monkeypatch.setattr(
            r,
            "classify_intent",
            lambda *a, **kw: r.RouterDecision(
                provider="local", model="gemma4:e2b",
                reason="casual_chat 0.95", intent_score=0.95,
            ),
        )

        callable_, model = r.route(
            "salut, comment ça va ?",
            None, cfg, call_kind="chat",
        )
        # Local callable lives in jarvis.llm, not the provider module.
        assert "anthropic_provider" not in (callable_.__module__ or ""), (
            f"FR casual_chat must stay local; got cloud module "
            f"{callable_.__module__!r}"
        )
        assert model == "gemma4:e2b"

    @pytest.mark.unit
    def test_router_is_language_agnostic(self, monkeypatch) -> None:
        """Same intent, FR vs EN -> same routing decision. The router
        sees intent labels, not language. Pinning this guarantees we
        don't accidentally introduce a language-aware branch (which
        would be a spec change worth its own discussion)."""
        from jarvis import llm_router as r

        cfg = _Cfg()
        calls: list[str] = []

        def fake_classify(prompt, *a, **kw):
            calls.append(prompt)
            return r.RouterDecision(
                provider="cloud", model="claude-sonnet-4-6",
                reason="code_complex 0.95", intent_score=0.95,
            )

        monkeypatch.setattr(r, "classify_intent", fake_classify)

        callable_fr, model_fr = r.route(
            "écris une fonction Python qui détecte les nombres premiers",
            None, cfg, call_kind="chat",
        )
        callable_en, model_en = r.route(
            "write a Python function that detects prime numbers",
            None, cfg, call_kind="chat",
        )
        assert callable_fr is callable_en
        assert model_fr == model_en
        assert len(calls) == 2  # both passed through the classifier
