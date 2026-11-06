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

A detector that fails (its model files missing from a build, onnxruntime
absent) fails on every call. Left on, it would take every utterance with
it and leave her deaf, so the utterance is heard again without it, the
failure is announced once for the whole process, and neither the listener
nor dictation asks for the detector again. A failure that is not the
detector's (a passing GPU error) leaves it on.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from test_dictation import _make_engine

INVENTED = "Sous-titrage réalisé par la communauté d'Amara.org"


DETECTOR_MISSING = "[ONNXRuntimeError] : 3 : NO_SUCHFILE : silero_encoder_v5.onnx failed"


GPU_ERROR = "CUDA failed with error out of memory"


def _whisper_on(audio_holds, detector_fails=False, decoder_fails=False, fails_once=False,
                silence_in="fr"):
    """A faster-whisper model hearing an utterance that holds *audio_holds*
    (a sentence, or "" for an utterance of noise).

    Without its voice detector it writes a sentence into noise, as the
    real one did; with it, it finds no speech in noise and hears nothing,
    yet still names a language for it (*silence_in*), as the real one does
    with no language set: the one it guesses from the silence it pads the
    emptied utterance with. A sentence that was said is heard either way.
    With *detector_fails*, asking for the detector raises what a build
    without its model files raises; with *decoder_fails*, every call
    raises, detector or not; with *fails_once*, its first call raises a
    passing GPU error.
    """
    calls = []

    def transcribe(audio, **kwargs):
        calls.append(kwargs)
        if decoder_fails or (fails_once and len(calls) == 1):
            raise RuntimeError(GPU_ERROR)
        if detector_fails and kwargs.get("vad_filter"):
            raise RuntimeError(DETECTOR_MISSING)
        if audio_holds:
            text = audio_holds
        elif kwargs.get("vad_filter"):
            return [], SimpleNamespace(language=silence_in)
        else:
            text = INVENTED
        return ([SimpleNamespace(text=" " + text, avg_logprob=-0.1, no_speech_prob=0.0)],
                SimpleNamespace(language="fr"))

    model = MagicMock()
    model.transcribe.side_effect = transcribe
    return model


def _listener(whisper_vad, audio_holds="", backend="faster-whisper", **whisper):
    from test_hot_window_input import _create_listener

    listener, _tts = _create_listener()
    cfg = listener.cfg
    cfg.whisper_vad = whisper_vad
    cfg.whisper_language = "fr"
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = backend, "cuda"
    listener._mlx_model_repo = "mlx-community/whisper"
    listener.model = _whisper_on(audio_holds, **whisper)
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


def _hear_next(listener, audio_holds="", **whisper):
    """Hand the listener another utterance, heard by a model in the same
    state, and return what it went on with."""
    listener._process_transcript.reset_mock()
    listener.model = _whisper_on(audio_holds, **whisper)
    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
    listener.is_speech_active = True
    listener._finalize_utterance()
    if not listener._process_transcript.called:
        return None
    return listener._process_transcript.call_args.args[0]


def _announcements(out):
    """The lines of console output that say the voice detector failed."""
    return [line for line in out.splitlines() if "voice detector" in line.lower()]


def _asked_for_the_detector(model):
    """Whether *model* was ever called with the voice detector."""
    return any(call.kwargs.get("vad_filter") for call in model.transcribe.call_args_list)


def _silero_that_hears_nothing(audio):
    """Silero's model as faster-whisper calls it: a batch of audio cut in
    512-sample windows in, one speech probability per window out."""
    assert audio.ndim == 2 and audio.shape[1] % 512 == 0
    return np.zeros((audio.shape[0], audio.shape[1] // 512), dtype=np.float32)


def _a_detector_that_loads():
    """faster-whisper's detector where onnxruntime and its model files are
    present. Only Silero's model is stood in for (the suite refuses
    onnxruntime): faster-whisper's own code around it runs, with the
    arguments the loader gives it."""
    return patch("faster_whisper.vad.get_vad_model", return_value=_silero_that_hears_nothing)


def _a_detector_that_cannot_load():
    """faster-whisper's detector where onnxruntime is missing."""
    return patch("faster_whisper.vad.get_vad_model", side_effect=RuntimeError(
        "Applying the VAD filter requires the onnxruntime package"))


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

    def test_an_utterance_of_noise_leaves_the_language_last_heard(self):
        """With no language set, Whisper names one even for an utterance the
        detector emptied. The question being collected keeps the language
        it was asked in."""
        listener = _listener(whisper_vad=True, audio_holds="Orla, quelle heure est-il ?")
        listener.cfg.whisper_language = None
        listener._finalize_utterance()

        _hear_next(listener, "", silence_in="en")
        listener.state_manager.stop()

        assert listener._last_detected_language == "fr"

    def test_on_mlx_an_utterance_heard_as_nothing_leaves_the_language_last_heard(self):
        listener = _listener(whisper_vad=True, backend="mlx")
        listener.cfg.whisper_language = None
        mlx = MagicMock()
        mlx.transcribe.side_effect = [
            {"text": "Bonjour.", "language": "fr",
             "segments": [{"text": "Bonjour.", "avg_logprob": -0.1, "no_speech_prob": 0.01}]},
            {"text": "", "language": "en", "segments": []},
        ]

        with patch("jarvis.listening.listener.mlx_whisper", mlx, create=True):
            listener._finalize_utterance()
            listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
            listener.is_speech_active = True
            listener._finalize_utterance()
        listener.state_manager.stop()

        assert listener._last_detected_language == "fr"

    def test_on_mlx_the_log_puts_nothing_down_to_a_detector(self):
        """MLX Whisper has no voice detector, so an utterance it heard as
        nothing is not put down to one."""
        listener = _listener(whisper_vad=True, backend="mlx")
        mlx = MagicMock()
        mlx.transcribe.return_value = {"text": "", "language": "fr", "segments": []}
        lines = []

        with patch("jarvis.listening.listener.mlx_whisper", mlx, create=True), \
                patch("jarvis.listening.listener.debug_log",
                      side_effect=lambda message, *_a, **_k: lines.append(message)):
            listener._finalize_utterance()
        listener.state_manager.stop()

        heard_as_nothing = [line for line in lines if "heard as nothing" in line]
        assert heard_as_nothing
        assert "voice detector on" not in heard_as_nothing[0]


@pytest.mark.unit
class TestDictation:

    def _engine(self, whisper_vad, audio_holds="", **whisper):
        return _make_engine(
            whisper_model_ref=lambda model=_whisper_on(audio_holds, **whisper): model,
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

    def test_a_detector_that_fails_does_not_take_his_dictation_with_it(self, capsys):
        model = _whisper_on("Le rendez-vous est à neuf heures.", detector_fails=True)
        engine = _make_engine(whisper_model_ref=lambda: model,
                              whisper_backend_ref=lambda: "faster-whisper",
                              cfg=SimpleNamespace(whisper_vad=True))

        first = engine._transcribe(np.zeros(16000, dtype=np.float32))
        model.transcribe.reset_mock()
        second = engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert first == second == "Le rendez-vous est à neuf heures."
        assert len(_announcements(capsys.readouterr().out)) == 1
        assert not _asked_for_the_detector(model)

    def test_a_detector_the_listener_saw_fail_is_not_asked_for_again(self, capsys):
        """One process, one detector: the listener announced it at start-up,
        and his first dictation neither asks for it nor announces it again."""
        pytest.importorskip("faster_whisper.vad")
        from jarvis.dictation.dictation_engine import DictationEngine

        _start_listener(whisper_vad=True, detector_loads=False)
        model = _whisper_on("Le rendez-vous est à neuf heures.", detector_fails=True)
        engine = DictationEngine(whisper_model_ref=lambda: model,
                                 whisper_backend_ref=lambda: "faster-whisper",
                                 mlx_repo_ref=lambda: None,
                                 cfg=SimpleNamespace(whisper_vad=True))

        heard = engine._transcribe(np.zeros(16000, dtype=np.float32))

        assert heard == "Le rendez-vous est à neuf heures."
        assert len(_announcements(capsys.readouterr().out)) == 1
        assert not _asked_for_the_detector(model)


@pytest.mark.unit
class TestADetectorThatFails:
    """Silero's model files missing from a build, or onnxruntime absent,
    fail every call that asks for the detector."""

    def test_his_sentence_is_heard_without_it(self):
        listener = _listener(whisper_vad=True, audio_holds="Orla, quelle heure est-il ?",
                             detector_fails=True)

        listener._finalize_utterance()

        assert _heard(listener) == "Orla, quelle heure est-il ?"

    def test_the_failure_is_announced_once(self, capsys):
        listener = _listener(whisper_vad=True, audio_holds="Orla, quelle heure est-il ?",
                             detector_fails=True)

        listener._finalize_utterance()
        later = _hear_next(listener, "Orla, tu es là ?", detector_fails=True)
        listener.state_manager.stop()

        assert later == "Orla, tu es là ?"
        assert len(_announcements(capsys.readouterr().out)) == 1
        assert not _asked_for_the_detector(listener.model)

    def test_a_passing_gpu_error_is_not_blamed_on_a_detector_that_loads(self, capsys):
        """The call with the detector also detects the language on the GPU;
        a passing error there, gone on the call without it, is not the
        detector's, so it stays on and the next utterance of noise is
        still heard as nothing."""
        listener = _listener(whisper_vad=True, audio_holds="Orla, quelle heure est-il ?",
                             fails_once=True)

        with _a_detector_that_loads():
            listener._finalize_utterance()
            first = listener._process_transcript.call_args.args[0]
            after_noise = _hear_next(listener, "")
        listener.state_manager.stop()

        assert first == "Orla, quelle heure est-il ?"
        assert after_noise is None
        assert _asked_for_the_detector(listener.model)
        assert _announcements(capsys.readouterr().out) == []

    def test_a_model_that_takes_only_the_language_still_hears(self, capsys):
        listener = _listener(whisper_vad=True)

        def transcribe(audio, language=None):
            return ([SimpleNamespace(text=" Orla, tu es là ?", avg_logprob=-0.1,
                                     no_speech_prob=0.0)], SimpleNamespace(language="fr"))

        listener.model = MagicMock()
        listener.model.transcribe.side_effect = transcribe
        listener._finalize_utterance()

        assert _heard(listener) == "Orla, tu es là ?"
        assert _announcements(capsys.readouterr().out) == []

    def test_a_failure_that_is_not_the_detector_s_leaves_the_detector_on(self, capsys):
        """A decoder that fails with and without the detector says nothing
        about the detector: it stays on, and the next utterance of noise
        is still heard as nothing."""
        listener = _listener(whisper_vad=True, decoder_fails=True)

        listener._finalize_utterance()
        nothing_after_failure = listener._process_transcript.called is False
        after_noise = _hear_next(listener, "")
        listener.state_manager.stop()

        assert nothing_after_failure
        assert after_noise is None
        assert _announcements(capsys.readouterr().out) == []


def _start_listener(whisper_vad, detector_loads):
    """Run the listener's start-up up to the microphone, with a Whisper
    model that loads and a voice detector that loads or not.

    The detector is faster-whisper's own (`faster_whisper.vad`), with
    Silero's model stood in for: the suite refuses onnxruntime.
    """
    from test_voice_listener import _create_mock_config

    cfg = _create_mock_config()
    cfg.whisper_vad = whisper_vad
    cfg.whisper_language = "fr"
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    cfg.ollama_chat_model = cfg.llm_chat_model = cfg.ollama_base_url = ""
    cfg.intent_judge_model = ""
    loads = _a_detector_that_loads() if detector_loads else _a_detector_that_cannot_load()
    with loads, \
            patch("jarvis.listening.listener.sys") as mock_sys, \
            patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True), \
            patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False), \
            patch("jarvis.listening.listener.WhisperModel", return_value=_whisper_on("")), \
            patch("jarvis.listening.listener.sd") as mock_sd:
        mock_sys.platform = "linux"
        mock_sd.query_devices.return_value = [{"name": "Test Mic", "max_input_channels": 1}]
        mock_sd.InputStream.side_effect = RuntimeError("stop here")
        from jarvis.listening.listener import VoiceListener

        listener = VoiceListener(MagicMock(), cfg, MagicMock(), MagicMock())
        listener.run()
    listener.state_manager.stop()
    return listener


@pytest.mark.unit
class TestStartUp:

    @pytest.fixture(autouse=True)
    def _faster_whisper(self):
        pytest.importorskip("faster_whisper.vad")

    def test_a_detector_that_cannot_load_is_announced_at_start_up(self, capsys):
        _start_listener(whisper_vad=True, detector_loads=False)

        assert len(_announcements(capsys.readouterr().out)) == 1

    def test_a_detector_that_loads_is_not_announced(self, capsys):
        _start_listener(whisper_vad=True, detector_loads=True)

        assert _announcements(capsys.readouterr().out) == []

    def test_with_the_setting_off_the_detector_is_not_loaded(self, capsys):
        _start_listener(whisper_vad=False, detector_loads=False)

        assert _announcements(capsys.readouterr().out) == []

    def test_once_announced_the_first_utterance_is_heard_without_it(self, capsys):
        listener = _start_listener(whisper_vad=True, detector_loads=False)
        listener._process_transcript = MagicMock()

        heard = _hear_next(listener, "Orla, quelle heure est-il ?", detector_fails=True)
        listener.state_manager.stop()

        assert heard == "Orla, quelle heure est-il ?"
        assert len(_announcements(capsys.readouterr().out)) == 1
        assert not _asked_for_the_detector(listener.model)

    def test_a_restart_in_the_same_process_tries_the_detector_again(self, capsys):
        """The bundled app restarts her inside its own process (Settings,
        Restart Jarvis). A detector that could not load at the first start
        and loads now is used again: noise is heard as nothing."""
        _start_listener(whisper_vad=True, detector_loads=False)
        listener = _start_listener(whisper_vad=True, detector_loads=True)
        listener._process_transcript = MagicMock()

        with _a_detector_that_loads():
            heard = _hear_next(listener, "")
        listener.state_manager.stop()

        assert heard is None
        assert len(_announcements(capsys.readouterr().out)) == 1

    def test_a_detector_that_still_cannot_load_is_announced_at_each_start(self, capsys):
        _start_listener(whisper_vad=True, detector_loads=False)
        _start_listener(whisper_vad=True, detector_loads=False)

        assert len(_announcements(capsys.readouterr().out)) == 2


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

