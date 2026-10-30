"""Whisper's language can be pinned.

Field trace (2026-10-30): Whisper guessed the language of every utterance
on its own. Out of silence it wrote "ご視聴ありがとうございました",
"Продолжение следует..." and "Thank you."; a spoken "stop" came back as
"Стоп.", which no stop word matches. A user who speaks one language can
name it, and Whisper then transcribes only that one. Left empty, it keeps
guessing, which is what a multilingual user needs.

A code Whisper does not know would fail every transcription and leave her
deaf, so it is announced and read as nothing named.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_dictation import _make_engine


def _settings_with(tmp_path, monkeypatch, entries):
    from jarvis.config import load_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


@pytest.mark.unit
class TestTheSetting:

    def test_with_nothing_named_whisper_detects(self, tmp_path, monkeypatch):
        assert _settings_with(tmp_path, monkeypatch, {}).whisper_language is None

    @pytest.mark.parametrize("written, read", [
        ("fr", "fr"), (" FR ", "fr"), ("fr-FR", "fr"), ("fr_fr", "fr"), ("yue", "yue"),
        ("", None), ("auto", None), ("french", None), ("xx", None),
    ])
    def test_it_is_read_from_the_config_file(self, tmp_path, monkeypatch, written, read):
        assert _settings_with(tmp_path, monkeypatch, {"whisper_language": written}).whisper_language == read

    def test_a_language_whisper_does_not_know_is_announced(self, tmp_path, monkeypatch, capsys):
        import jarvis.config as config

        monkeypatch.setattr(config, "_announced_languages", set())
        _settings_with(tmp_path, monkeypatch, {"whisper_language": "french"})

        assert "french" in capsys.readouterr().out

    @pytest.mark.parametrize("language, read", [("fr", None), ("en", "en")])
    def test_an_english_only_model_hears_english(self, tmp_path, monkeypatch, language, read):
        """It transcribes English whatever it is told, and would report the
        named language for English text."""
        settings = _settings_with(tmp_path, monkeypatch,
                                  {"whisper_model": "medium.en", "whisper_language": language})

        assert settings.whisper_language == read

    def test_the_list_is_whisper_s_own(self):
        tokenizer = pytest.importorskip("faster_whisper.tokenizer")
        from jarvis.config import WHISPER_LANGUAGES

        assert tuple(WHISPER_LANGUAGES) == tuple(tokenizer._LANGUAGE_CODES)

    def test_the_settings_page_offers_detect_and_every_language(self):
        from desktop_app.settings_window import FIELD_METADATA
        from jarvis.config import WHISPER_LANGUAGES

        field = {f.key: f for f in FIELD_METADATA}["whisper_language"]
        assert field.category == "whisper"
        assert [value for value, _label in field.choices] == [""] + list(WHISPER_LANGUAGES)


def _listener_transcribing(language, backend="faster-whisper"):
    from test_hot_window_input import _create_listener

    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.whisper_language = language
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = backend, "cuda"
    listener._mlx_model_repo = "mlx-community/whisper"
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

    @pytest.mark.parametrize("language", ["fr", None])
    def test_whisper_hears_the_language_named(self, language):
        listener = _listener_transcribing(language)

        listener._finalize_utterance()

        assert listener.model.transcribe.call_args.kwargs["language"] == language
        listener.state_manager.stop()

    @pytest.mark.parametrize("language", ["fr", None])
    def test_so_does_mlx_whisper(self, language):
        listener = _listener_transcribing(language, backend="mlx")
        mlx = MagicMock()
        mlx.transcribe.return_value = {"text": "Bonjour.", "language": language or "ja",
                                       "segments": [{"text": "Bonjour.", "avg_logprob": -0.1,
                                                     "no_speech_prob": 0.01}]}

        with patch("jarvis.listening.listener.mlx_whisper", mlx, create=True):
            listener._finalize_utterance()

        assert mlx.transcribe.call_args.kwargs["language"] == language
        listener.state_manager.stop()


@pytest.mark.unit
class TestDictation:

    @pytest.mark.parametrize("language", ["fr", None])
    def test_dictation_hears_the_language_named(self, language):
        model = MagicMock()
        model.transcribe.return_value = ([SimpleNamespace(text=" bonjour ")], MagicMock())
        engine = _make_engine(
            whisper_model_ref=lambda: model,
            whisper_backend_ref=lambda: "faster-whisper",
            whisper_language=language,
        )

        engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert model.transcribe.call_args.kwargs["language"] == language

    @pytest.mark.parametrize("language", ["fr", None])
    def test_so_does_mlx_dictation(self, language):
        mlx = MagicMock()
        mlx.transcribe.return_value = {"text": "bonjour"}
        engine = _make_engine(
            whisper_backend_ref=lambda: "mlx",
            mlx_repo_ref=lambda: "mlx-community/whisper",
            whisper_language=language,
        )

        with patch.dict(sys.modules, {"mlx_whisper": mlx}):
            engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert mlx.transcribe.call_args.kwargs["language"] == language
