"""Behaviour tests for ``config.select_tts_voice``.

The helper picks which Piper voice file to load given the user's
``tts_voices`` map, ``response_language`` lock, ``tts_piper_model_path``
legacy scalar, and the latest detected language from Whisper.

We assert the *resolved path string* — that's the observable output
the TTS engine sees. We don't pin internal state or call counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pytest


@dataclass
class _Cfg:
    tts_voices: dict = field(default_factory=dict)
    tts_piper_model_path: Optional[str] = None
    response_language: str = ""


class TestSelectTtsVoice:
    """Decision matrix:

    detected_lang in map  | response_language match | legacy scalar | -> result
    """

    @pytest.mark.unit
    def test_detected_language_wins_over_everything(self) -> None:
        """Live Whisper detection is the most current signal — it must
        take precedence over the config lock and legacy fallbacks."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(
            tts_voices={"en": "/voices/en.onnx", "fr": "/voices/fr.onnx"},
            tts_piper_model_path="/legacy/en_only.onnx",
            response_language="english",
        )
        assert select_tts_voice(cfg, "fr") == "/voices/fr.onnx"

    @pytest.mark.unit
    def test_response_language_used_when_detected_missing(self) -> None:
        """First turn or silent context: no detected_language yet, so
        ``response_language`` provides the cold-start hint."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(
            tts_voices={"en": "/voices/en.onnx", "fr": "/voices/fr.onnx"},
            response_language="français",
        )
        assert select_tts_voice(cfg, None) == "/voices/fr.onnx"

    @pytest.mark.unit
    def test_response_language_2char_code_lookup(self) -> None:
        """``response_language="English"`` must resolve to the ``"en"``
        key, not get sniffed for an exact ``"english"`` key — the map
        keys are 2-letter ISO codes by convention."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(
            tts_voices={"en": "/voices/en.onnx", "fr": "/voices/fr.onnx"},
            response_language="English",
        )
        assert select_tts_voice(cfg, None) == "/voices/en.onnx"

    @pytest.mark.unit
    def test_detected_language_with_uppercase(self) -> None:
        """Whisper sometimes returns codes with mixed case. The lookup
        must lowercase + truncate to 2 chars before indexing."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(tts_voices={"fr": "/voices/fr.onnx"})
        assert select_tts_voice(cfg, "FR") == "/voices/fr.onnx"
        assert select_tts_voice(cfg, "fr-FR") == "/voices/fr.onnx"

    @pytest.mark.unit
    def test_missing_key_falls_through_to_first_entry(self) -> None:
        """User has a tts_voices map but Whisper detected a language
        not in it (e.g. Spanish on a config that only has en/fr).
        Rather than crashing, return the first map entry — predictable
        and deterministic via insertion order."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(tts_voices={"en": "/voices/en.onnx", "fr": "/voices/fr.onnx"})
        # First insertion was "en"
        assert select_tts_voice(cfg, "es") == "/voices/en.onnx"

    @pytest.mark.unit
    def test_empty_map_falls_back_to_legacy_scalar(self) -> None:
        """Backwards-compat: existing configs only set
        ``tts_piper_model_path``. Without ``tts_voices``, we must
        return the legacy scalar unchanged."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(tts_voices={}, tts_piper_model_path="/legacy/voice.onnx")
        assert select_tts_voice(cfg, "fr") == "/legacy/voice.onnx"
        assert select_tts_voice(cfg, None) == "/legacy/voice.onnx"

    @pytest.mark.unit
    def test_returns_none_when_nothing_configured(self) -> None:
        """Fresh install: no map, no scalar. Helper returns ``None``
        so the caller can decide (warn + use Piper default, or skip
        the voice entirely)."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(tts_voices={}, tts_piper_model_path=None)
        assert select_tts_voice(cfg, "fr") is None
        assert select_tts_voice(cfg, None) is None

    @pytest.mark.unit
    def test_empty_string_detected_language_is_treated_as_none(self) -> None:
        """Some upstream paths pass ``""`` instead of ``None`` for
        ``detected_language``. Both must behave the same — fall through
        to the config lock / map first entry."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(tts_voices={"fr": "/voices/fr.onnx"}, response_language="français")
        assert select_tts_voice(cfg, "") == "/voices/fr.onnx"

    @pytest.mark.unit
    def test_response_language_unmapped_falls_through(self) -> None:
        """``response_language="russisch"`` (no ``ru`` key in map) must
        not surface a None — fall through to first map entry."""
        from jarvis.config import select_tts_voice

        cfg = _Cfg(
            tts_voices={"en": "/voices/en.onnx", "fr": "/voices/fr.onnx"},
            response_language="Russian",  # "ru" not in map
        )
        assert select_tts_voice(cfg, None) == "/voices/en.onnx"


class TestConfigParser:
    """The parser turns the raw JSON into the typed Settings field.
    We pin the coercion rules so a malformed config can't pollute
    ``select_tts_voice`` at call time."""

    @pytest.mark.unit
    def test_keys_lowercased_and_truncated(self, tmp_path, monkeypatch) -> None:
        """Keys must be lowercased and truncated to 2 chars so the
        runtime lookup (``detected_language[:2].lower()``) hits."""
        import json

        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({
            "tts_voices": {
                "EN": "/voices/en.onnx",
                "fr-FR": "/voices/fr.onnx",
                "DE_DE": "/voices/de.onnx",
            }
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        # Force re-import via the same env path
        from jarvis import config as cfg_mod
        settings = cfg_mod.load_settings()

        assert settings.tts_voices == {
            "en": "/voices/en.onnx",
            "fr": "/voices/fr.onnx",
            "de": "/voices/de.onnx",
        }

    @pytest.mark.unit
    def test_non_string_values_dropped(self, tmp_path, monkeypatch) -> None:
        """Numeric / nested / empty values must be silently dropped
        rather than blowing up at config load."""
        import json

        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({
            "tts_voices": {
                "en": "/voices/en.onnx",
                "fr": 123,           # not a string -> dropped
                "de": "",            # empty -> dropped
                "es": "/voices/es.onnx",
            }
        }))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis import config as cfg_mod
        settings = cfg_mod.load_settings()

        assert "en" in settings.tts_voices
        assert "es" in settings.tts_voices
        assert "fr" not in settings.tts_voices
        assert "de" not in settings.tts_voices

    @pytest.mark.unit
    def test_default_is_empty_dict(self, tmp_path, monkeypatch) -> None:
        """A config that doesn't set ``tts_voices`` at all must load
        with ``tts_voices == {}`` — never None or missing — so
        ``select_tts_voice`` can iterate without a guard."""
        import json

        cfg_path = tmp_path / "jarvis.json"
        cfg_path.write_text(json.dumps({"whisper_model": "medium"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        from jarvis import config as cfg_mod
        settings = cfg_mod.load_settings()

        assert settings.tts_voices == {}
