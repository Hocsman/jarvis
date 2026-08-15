"""Behaviour tests for the Kokoro TTS engine and its wiring.

These pin the engine-selection contract and the KokoroTTS interface
without loading the model (slow + a Hugging Face download): the actual
synthesis quality is verified manually / live. The point here is that
``tts_engine: "kokoro"`` resolves to a KokoroTTS configured from the
``tts_kokoro_*`` settings, that the engine honours the same interface as
the other engines (so the listener's loopback guard and interruption
work unchanged), and that the other engines still resolve correctly.
"""

from __future__ import annotations

import pytest

from src.jarvis.output.tts import (
    create_tts_engine,
    KokoroTTS,
    PiperTTS,
    _find_espeak_library,
)


class TestKokoroInterface:
    def test_has_same_interface_as_other_engines(self):
        tts = KokoroTTS(enabled=False)
        for name in ("start", "stop", "speak", "interrupt", "is_speaking",
                     "get_last_spoken_text"):
            assert callable(getattr(tts, name)), f"missing {name}"

    def test_disabled_is_a_no_op(self):
        # Disabled must never load the pipeline or touch audio hardware.
        tts = KokoroTTS(enabled=False)
        tts.start()
        tts.speak("bonjour")
        assert tts.is_speaking() is False
        tts.interrupt()
        tts.stop()
        assert tts._pipe is None  # never initialised

    def test_stores_config(self):
        tts = KokoroTTS(enabled=True, voice="ff_siwis", lang_code="f", speed=1.1)
        assert tts.voice == "ff_siwis"
        assert tts.lang_code == "f"
        assert tts.speed == 1.1
        assert tts._sample_rate == 24000  # Kokoro's fixed output rate

    def test_blank_values_fall_back_to_french_defaults(self):
        tts = KokoroTTS(enabled=True, voice="", lang_code="", speed=0)
        assert tts.voice == "ff_siwis"
        assert tts.lang_code == "f"
        assert tts.speed == 1.0


class TestEngineSelection:
    def test_kokoro_engine_resolves_to_kokoro(self):
        tts = create_tts_engine(
            engine="kokoro", enabled=False,
            kokoro_voice="ff_siwis", kokoro_lang_code="f", kokoro_speed=1.2,
        )
        assert isinstance(tts, KokoroTTS)
        assert tts.voice == "ff_siwis"
        assert tts.lang_code == "f"
        assert tts.speed == 1.2

    def test_kokoro_selection_is_case_insensitive(self):
        assert isinstance(create_tts_engine(engine="KOKORO", enabled=False), KokoroTTS)

    def test_piper_still_default(self):
        assert isinstance(create_tts_engine(engine="piper", enabled=False), PiperTTS)
        assert isinstance(create_tts_engine(engine="unknown", enabled=False), PiperTTS)


class TestEspeakDiscovery:
    def test_returns_path_or_none(self):
        # Platform-dependent: assert the contract (str path that exists, or
        # None) rather than a specific location.
        result = _find_espeak_library()
        assert result is None or (isinstance(result, str) and result)


class TestConfigWiring:
    def test_defaults_present(self):
        from src.jarvis.config import load_settings
        s = load_settings()
        assert s.tts_kokoro_voice == "ff_siwis"
        assert s.tts_kokoro_lang_code == "f"
        assert isinstance(s.tts_kokoro_speed, float)

    def test_kokoro_is_an_accepted_engine_value(self, tmp_path, monkeypatch):
        # Regression: the tts_engine validator must not reject "kokoro" and
        # silently fall back to "piper".
        import json
        from src.jarvis.config import load_settings
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"tts_engine": "kokoro"}))
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg))
        assert load_settings().tts_engine == "kokoro"
