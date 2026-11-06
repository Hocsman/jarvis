"""Whisper's voice detector runs when `whisper_vad` says so.

Field trace: out of an utterance that held no speech (a breath, a chair,
the room), Whisper wrote a sentence: a subtitle credit, "Merci.", a
phrase nobody said. `large-v3-turbo` reports a `no_speech_prob` of 0.0
on those, so the no-speech filter never sees them, and in the hot window
each one reached her as if he had spoken. faster-whisper's own voice
detector (Silero, `vad_filter`) cuts the stretches that hold no speech
before the decoder hears them, so an utterance of noise comes back
empty. The setting (`whisper_vad`, on by default, "Use VAD Filter" on
the settings page) existed and nothing read it.

The listener and dictation both honour it. MLX Whisper has no voice
detector; on it the setting changes nothing.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_dictation import _make_engine

INVENTED = "Sous-titrage réalisé par la communauté d'Amara.org"


def _whisper_on(audio_holds):
    """A faster-whisper model hearing an utterance that holds *audio_holds*
    (a sentence, or "" for an utterance of noise).

    Without its voice detector it writes a sentence into noise, as the
    real one did; with it, it finds no speech in noise and hears nothing.
    A sentence that was said is heard either way.
    """
    def transcribe(audio, **kwargs):
        if audio_holds:
            text = audio_holds
        elif kwargs.get("vad_filter"):
            return [], SimpleNamespace(language="fr")
        else:
            text = INVENTED
        return ([SimpleNamespace(text=" " + text, avg_logprob=-0.1, no_speech_prob=0.0)],
                SimpleNamespace(language="fr"))

    model = MagicMock()
    model.transcribe.side_effect = transcribe
    return model


def _listener(whisper_vad, audio_holds="", backend="faster-whisper"):
    from test_hot_window_input import _create_listener

    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.whisper_vad = whisper_vad
    cfg.whisper_language = "fr"
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = backend, "cuda"
    listener._mlx_model_repo = "mlx-community/whisper"
    listener.model = _whisper_on(audio_holds)
    listener._process_transcript = MagicMock()
    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
    listener.is_speech_active = True
    return listener


def _heard(listener):
    """What the listener went on with, or None when it went on with nothing."""
    listener.state_manager.stop()
    if not listener._process_transcript.called:
        return None
    return listener._process_transcript.call_args.args[0]


@pytest.mark.unit
class TestTheListener:

    def test_an_utterance_of_noise_is_heard_as_nothing(self):
        listener = _listener(whisper_vad=True)

        listener._finalize_utterance()

        assert _heard(listener) is None

    def test_his_sentence_is_still_heard(self):
        listener = _listener(whisper_vad=True, audio_holds="Orla, quelle heure est-il ?")

        listener._finalize_utterance()

        assert _heard(listener) == "Orla, quelle heure est-il ?"

    def test_with_the_setting_off_whisper_hears_the_whole_utterance(self):
        listener = _listener(whisper_vad=False)

        listener._finalize_utterance()

        assert _heard(listener) == INVENTED

    def test_mlx_whisper_is_not_given_an_argument_it_does_not_take(self):
        listener = _listener(whisper_vad=True, backend="mlx")
        mlx = MagicMock()
        mlx.transcribe.return_value = {"text": "Bonjour.", "language": "fr",
                                       "segments": [{"text": "Bonjour.", "avg_logprob": -0.1,
                                                     "no_speech_prob": 0.01}]}

        with patch("jarvis.listening.listener.mlx_whisper", mlx, create=True):
            listener._finalize_utterance()

        assert _heard(listener) == "Bonjour."
        assert "vad_filter" not in mlx.transcribe.call_args.kwargs


@pytest.mark.unit
class TestDictation:

    def _engine(self, whisper_vad, audio_holds=""):
        return _make_engine(
            whisper_model_ref=lambda model=_whisper_on(audio_holds): model,
            whisper_backend_ref=lambda: "faster-whisper",
            whisper_language="fr",
            cfg=SimpleNamespace(whisper_vad=whisper_vad),
        )

    def test_a_press_with_nothing_said_pastes_nothing(self):
        engine = self._engine(whisper_vad=True)

        assert engine._transcribe(np.zeros(16000, dtype=np.float32)) == ""

    def test_what_he_dictates_is_still_heard(self):
        engine = self._engine(whisper_vad=True, audio_holds="Le rendez-vous est à neuf heures.")

        assert engine._transcribe(np.zeros(16000, dtype=np.float32)) == \
            "Le rendez-vous est à neuf heures."

    def test_with_the_setting_off_whisper_hears_the_whole_recording(self):
        engine = self._engine(whisper_vad=False)

        assert engine._transcribe(np.zeros(16000, dtype=np.float32)) == INVENTED

    def test_with_no_configuration_the_detector_runs_as_the_default_says(self):
        from jarvis.config import get_default_config

        engine = _make_engine(
            whisper_model_ref=lambda model=_whisper_on(""): model,
            whisper_backend_ref=lambda: "faster-whisper",
        )

        heard = engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert heard == ("" if get_default_config()["whisper_vad"] else INVENTED)


@pytest.mark.unit
class TestTheSetting:

    def test_it_is_on_by_default(self):
        from jarvis.config import get_default_config

        assert get_default_config()["whisper_vad"] is True

    @pytest.mark.parametrize("written", [True, False])
    def test_it_is_read_from_the_config_file(self, tmp_path, monkeypatch, written):
        import json
        from jarvis.config import load_settings

        path = tmp_path / "config.json"
        path.write_text(json.dumps({"whisper_vad": written}), encoding="utf-8")
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

        assert load_settings().whisper_vad is written

