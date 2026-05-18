"""Behaviour tests for ``_tts_language_constraint`` in reply.engine.

The clause used to be unconditional ("Always respond in English…") which
poisoned multilingual workflows: even with a French Piper voice loaded
and ``response_language="français"`` set, the LLM still received an
order to switch back to English. The gated helper replaces it with a
config-aware decision; these tests pin that decision matrix down so
the regression cannot reappear.

Tests assert on observable shape (clause present / absent, and when
present, lacks the historical "Always respond in English" phrasing),
not on internal call counts or implementation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class _Cfg:
    response_language: str = ""
    tts_engine: str = "piper"
    tts_piper_model_path: str = ""


class TestTtsLanguageConstraint:
    """The helper decides whether to append a TTS-locale clause to the
    system prompt based on three signals: explicit language lock in
    config, the TTS engine in use, and the locale of the loaded Piper
    voice."""

    @pytest.mark.unit
    def test_response_language_set_returns_none(self) -> None:
        """A non-empty ``response_language`` means the persona prompt
        already carries the language directive; adding a second TTS
        clause would be redundant noise and risks conflicting wording
        (which is exactly what bit the FR workflow before)."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="français", tts_engine="piper", tts_piper_model_path="/x/en_US-amy.onnx")
        assert _tts_language_constraint(cfg) is None

    @pytest.mark.unit
    def test_piper_french_voice_no_lock(self) -> None:
        """When Piper loads a French voice, the model must be free to
        match the user's language. Locking it to English would make
        Piper try to pronounce English phonemes through a French
        voice — exactly what the bug was."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="piper", tts_piper_model_path="/voices/fr_FR-siwis-medium.onnx")
        assert _tts_language_constraint(cfg) is None

    @pytest.mark.unit
    def test_piper_german_voice_no_lock(self) -> None:
        """The locale sniff isn't FR-specific — any non-EN Piper locale
        should release the lock."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="piper", tts_piper_model_path="/voices/de_DE-thorsten.onnx")
        assert _tts_language_constraint(cfg) is None

    @pytest.mark.unit
    def test_piper_english_voice_returns_lock_clause(self) -> None:
        """The historical behaviour must survive for anglophone users
        with default config: Piper EN voice + no response_language →
        the English lock clause is still appended."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="piper", tts_piper_model_path="/voices/en_US-amy-medium.onnx")
        clause = _tts_language_constraint(cfg)
        assert clause is not None
        assert "Respond in English" in clause

    @pytest.mark.unit
    def test_piper_missing_voice_returns_lock_clause(self) -> None:
        """Empty / missing path is the legacy default for fresh installs.
        Until the user picks a voice we cannot know its locale, so we
        keep the conservative EN clause to preserve current behaviour."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="piper", tts_piper_model_path="")
        clause = _tts_language_constraint(cfg)
        assert clause is not None
        assert "Respond in English" in clause

    @pytest.mark.unit
    def test_chatterbox_returns_lock_clause(self) -> None:
        """Chatterbox still ships English-only voices, so the clause
        is unconditional for that engine (until/unless multilingual
        Chatterbox voices appear)."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="chatterbox", tts_piper_model_path="")
        clause = _tts_language_constraint(cfg)
        assert clause is not None
        assert "Respond in English" in clause

    @pytest.mark.unit
    def test_clause_does_not_use_legacy_wording(self) -> None:
        """The new clause must not contain the legacy phrase
        ``"Always respond in English regardless of the language..."``.
        That exact wording is what
        ``test_no_force_english_in_system_prompt_source`` greps the
        source for; a regression here would also flip that test."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="piper", tts_piper_model_path="")
        clause = _tts_language_constraint(cfg) or ""
        assert "Always respond in English regardless" not in clause
        # The new wording must explain WHY (the voice locale), not just
        # impose a rule out of the blue — this is what survives Sonnet's
        # tendency to push back on bare imperatives.
        assert "TTS voice" in clause or "voice" in clause

    @pytest.mark.unit
    def test_unknown_engine_returns_none(self) -> None:
        """Engines we don't recognise (future additions, custom hooks)
        get no clause. Conservative default: don't pretend to know what
        the voice supports."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="", tts_engine="elevenlabs", tts_piper_model_path="")
        assert _tts_language_constraint(cfg) is None

    @pytest.mark.unit
    def test_response_language_wins_over_english_voice(self) -> None:
        """Explicit user lock beats voice sniff: response_language="français"
        + EN voice → no clause appended (the persona handles it, and
        the user has explicitly accepted the mismatch — Piper will
        mispronounce, that's their config call to make)."""
        from jarvis.reply.engine import _tts_language_constraint

        cfg = _Cfg(response_language="français", tts_engine="piper", tts_piper_model_path="/x/en_US-amy.onnx")
        assert _tts_language_constraint(cfg) is None


class TestNoForceEnglishInSource:
    """Code-shape regression: the legacy unconditional phrase must not
    re-appear anywhere in the runtime source tree. Documentation files
    (*.md) are allowed to reference it historically."""

    @pytest.mark.unit
    def test_no_force_english_in_system_prompt_source(self) -> None:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1] / "src"
        offenders = []
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            if "Always respond in English regardless" in text:
                offenders.append(str(path.relative_to(root)))
        assert offenders == [], (
            f"Legacy English-forcing phrase reappeared in: {offenders}. "
            f"Use _tts_language_constraint(cfg) instead — it gates the "
            f"clause on response_language and the Piper voice locale."
        )


class TestPiperVoiceLanguageSniff:
    """Direct tests of the filename → locale sniff. Useful for users
    who name their voices oddly."""

    @pytest.mark.unit
    @pytest.mark.parametrize("path,expected", [
        ("/voices/fr_FR-siwis-medium.onnx", "fr"),
        ("/voices/en_US-amy-medium.onnx", "en"),
        ("/voices/de_DE-thorsten-medium.onnx", "de"),
        ("/voices/es_ES-mls.onnx", "es"),
        ("fr_FR-siwis-medium.onnx", "fr"),  # no directory prefix
        ("", None),
        ("/voices/custom-voice.onnx", None),  # doesn't match the convention
        ("/voices/EN_US-amy-medium.onnx", "EN"),  # case preserved by regex match
    ])
    def test_sniff(self, path: str, expected: Optional[str]) -> None:
        from jarvis.reply.engine import _piper_voice_language

        result = _piper_voice_language(path)
        # The helper lowercases; uppercase test case must come back lower
        if expected is not None:
            assert result == expected.lower()
        else:
            assert result is None
