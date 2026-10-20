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

    def test_init_warms_the_synthesis_path(self, monkeypatch):
        """Initialisation ends with one throwaway synthesis, so the first
        real reply does not pay the model's first-call cost (CUDA kernel
        loading alone can take tens of seconds). The warm-up audio is
        discarded: nothing is queued, played or remembered."""
        calls = []

        class FakePipeline:
            def __init__(self, lang_code=None):
                pass

            def __call__(self, text, voice=None, speed=None):
                calls.append((text, voice, speed))
                yield None, None, [0.0] * 2400

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KPipeline = FakePipeline
        monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

        tts = KokoroTTS(enabled=True, voice="ff_siwis,af_bella", lang_code="f", speed=1.1)
        assert tts._ensure_initialized() is True

        assert calls == [(tts._WARMUP_TEXT, "ff_siwis,af_bella", 1.1)]
        assert tts.get_last_spoken_text() == ""
        assert tts.is_speaking() is False

    def test_a_failed_warmup_does_not_fail_init(self, monkeypatch):
        """A warm-up that raises leaves the pipeline loaded: synthesis
        errors keep surfacing at speak time, as without the warm-up."""

        class BrokenPipeline:
            def __init__(self, lang_code=None):
                pass

            def __call__(self, text, voice=None, speed=None):
                raise RuntimeError("boom")
                yield

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KPipeline = BrokenPipeline
        monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

        tts = KokoroTTS(enabled=True)
        assert tts._ensure_initialized() is True
        assert tts._pipe is not None
        assert tts._init_error is None


class TestKokoroVoiceBlends:
    """``tts_kokoro_voice`` syntax: a plain name, an equal blend
    (``a,b``, handled by Kokoro itself), or a weighted blend
    (``a:0.8,b:0.2``, built by the engine as a weighted mean of the voice
    packs, normalised by the sum of the weights)."""

    def _fake_kokoro(self, monkeypatch, voices_seen, packs=None):
        class FakePipeline:
            def __init__(self, lang_code=None):
                pass

            def load_single_voice(self, name):
                return (packs or {}).get(name, 1.0)

            def __call__(self, text, voice=None, speed=None):
                voices_seen.append(voice)
                yield None, None, [0.0] * 2400

        fake_kokoro = types.ModuleType("kokoro")
        fake_kokoro.KPipeline = FakePipeline
        monkeypatch.setitem(sys.modules, "kokoro", fake_kokoro)

    def test_plain_voice_stays_a_string(self, monkeypatch):
        seen = []
        self._fake_kokoro(monkeypatch, seen)
        tts = KokoroTTS(enabled=True, voice="ff_siwis,af_bella")
        assert tts._ensure_initialized() is True
        assert seen == ["ff_siwis,af_bella"]
        assert tts._voice_blend is None

    def test_weighted_blend_is_the_normalised_weighted_mean(self, monkeypatch):
        seen = []
        self._fake_kokoro(monkeypatch, seen, packs={"ff_siwis": 10.0, "af_bella": 20.0})
        tts = KokoroTTS(enabled=True, voice="ff_siwis:0.8,af_bella:0.2")
        assert tts._ensure_initialized() is True
        # (0.8*10 + 0.2*20) / 1.0 — and the warm-up already rides the blend
        assert seen == [pytest.approx(12.0)]

    def test_weights_need_not_sum_to_one(self, monkeypatch):
        seen = []
        self._fake_kokoro(monkeypatch, seen, packs={"ff_siwis": 10.0, "af_bella": 20.0})
        tts = KokoroTTS(enabled=True, voice="ff_siwis:4,af_bella:1")
        assert tts._ensure_initialized() is True
        assert seen == [pytest.approx(12.0)]  # (4*10 + 1*20) / 5

    def test_a_part_without_weight_counts_as_one(self, monkeypatch):
        seen = []
        self._fake_kokoro(monkeypatch, seen, packs={"ff_siwis": 10.0, "af_bella": 20.0})
        tts = KokoroTTS(enabled=True, voice="ff_siwis:3,af_bella")
        assert tts._ensure_initialized() is True
        assert seen == [pytest.approx(12.5)]  # (3*10 + 1*20) / 4

    def test_an_unparseable_weight_falls_back_to_the_raw_string(self, monkeypatch):
        seen = []
        self._fake_kokoro(monkeypatch, seen)
        tts = KokoroTTS(enabled=True, voice="ff_siwis:abc")
        assert tts._ensure_initialized() is True
        assert seen == ["ff_siwis:abc"]
        assert tts._voice_blend is None


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
