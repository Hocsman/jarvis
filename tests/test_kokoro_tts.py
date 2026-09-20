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

import os
import sys
import types

import pytest

import src.jarvis.output.tts as tts_module
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

    def test_start_preloads_pipeline_module_safely(self, monkeypatch):
        """On Windows the heavy Kokoro import happens on the calling thread,
        before any worker starts — concurrent DLL loading is what deadlocks
        there — and the espeak-ng library path is resolved first."""
        events = []

        class _RecordingModule(types.ModuleType):
            def __getattr__(self, name):
                events.append(("import", name))
                return type(name, (), {})

        class MockThread:
            def __init__(self, target=None, daemon=False, name=None):
                events.append(("thread", name))

            def start(self):
                pass

        monkeypatch.setitem(sys.modules, "kokoro", _RecordingModule("kokoro"))
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(tts_module, "_find_espeak_library",
                            lambda: "/usr/lib/libespeak-ng.so.1")
        monkeypatch.setattr("threading.Thread", MockThread)
        saved_espeak = os.environ.pop("PHONEMIZER_ESPEAK_LIBRARY", None)
        try:
            KokoroTTS(enabled=True).start()

            assert ("import", "KPipeline") in events
            assert ("thread", "kokoro-init") in events
            assert events.index(("import", "KPipeline")) < events.index(
                ("thread", "kokoro-init"))
            assert (os.environ["PHONEMIZER_ESPEAK_LIBRARY"]
                    == "/usr/lib/libespeak-ng.so.1")
        finally:
            os.environ.pop("PHONEMIZER_ESPEAK_LIBRARY", None)
            if saved_espeak is not None:
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = saved_espeak

    def test_start_skips_the_pre_import_off_windows(self, monkeypatch):
        """The loader-lock deadlock the pre-import prevents is
        Windows-specific; elsewhere start() keeps the pure background
        warm-up and never imports Kokoro on the calling thread."""
        events = []

        class _RecordingModule(types.ModuleType):
            def __getattr__(self, name):
                events.append(("import", name))
                return type(name, (), {})

        class MockThread:
            def __init__(self, target=None, daemon=False, name=None):
                events.append(("thread", name))

            def start(self):
                pass

        monkeypatch.setitem(sys.modules, "kokoro", _RecordingModule("kokoro"))
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("threading.Thread", MockThread)

        KokoroTTS(enabled=True).start()

        assert ("import", "KPipeline") not in events
        assert ("thread", "kokoro-init") in events


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
        cfg.write_text(json.dumps({"tts_engine": "kokoro"}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg))
        assert load_settings().tts_engine == "kokoro"
