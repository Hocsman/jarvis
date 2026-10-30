"""Whisper's language can be pinned.

Field trace (2026-10-30): Whisper guessed the language of every utterance
on its own. Out of silence it wrote "ご視聴ありがとうございました",
"Продолжение следует..." and "Thank you."; a spoken "stop" came back as
"Стоп.", which no stop word matches. A user who speaks one language can
name it, and Whisper then transcribes only that one. Left empty, it keeps
guessing, which is what a multilingual user needs.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from test_dictation import _make_engine


@pytest.mark.unit
class TestTheSetting:

    def test_it_guesses_by_default(self):
        from jarvis.config import get_default_config

        assert get_default_config()["whisper_language"] == ""

    @pytest.mark.parametrize("written, read", [
        ("fr", "fr"), (" FR ", "fr"), ("", ""), ("auto", ""),
    ])
    def test_it_is_read_from_the_config_file(self, tmp_path, monkeypatch, written, read):
        from jarvis.config import load_settings

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"whisper_language": written}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

        assert load_settings().whisper_language == read

    def test_it_is_on_the_whisper_settings_page(self):
        from desktop_app.settings_window import FIELD_METADATA

        field = {f.key: f for f in FIELD_METADATA}["whisper_language"]
        assert field.category == "whisper"


def _listener_transcribing(language):
    from test_hot_window_input import _create_listener

    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.whisper_language = language
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    listener.model.transcribe.return_value = (
        [SimpleNamespace(text=" Bonjour.", avg_logprob=-0.1, no_speech_prob=0.01)],
        SimpleNamespace(language=language or "ja"))
    listener._process_transcript = MagicMock()
    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
    listener.is_speech_active = True
    return listener


@pytest.mark.unit
class TestTheListener:

    @pytest.mark.parametrize("language, passed", [("fr", "fr"), ("", None)])
    def test_whisper_hears_the_language_named(self, language, passed):
        listener = _listener_transcribing(language)

        listener._finalize_utterance()

        assert listener.model.transcribe.call_args.kwargs["language"] == passed
        listener.state_manager.stop()


@pytest.mark.unit
class TestDictation:

    @pytest.mark.parametrize("language, passed", [("fr", "fr"), ("", None)])
    def test_dictation_hears_the_language_named(self, language, passed):
        model = MagicMock()
        model.transcribe.return_value = ([SimpleNamespace(text=" bonjour ")], MagicMock())
        engine = _make_engine(
            whisper_model_ref=lambda: model,
            whisper_backend_ref=lambda: "faster-whisper",
            cfg=SimpleNamespace(whisper_language=language),
        )

        engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert model.transcribe.call_args.kwargs["language"] == passed
