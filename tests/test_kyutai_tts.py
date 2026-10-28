"""Behaviour tests for the Kyutai TTS engine and its wiring.

The model is replaced by a stand-in that has the surface the engine uses
(``moshi.models.loaders`` and ``moshi.models.tts``), and the speakers by a
stream that plays into memory. What these tests pin is what the engine
does with them: when it loads, what it plays, what it promises the
listener (``tts.spec.md``, "Kyutai"). How the voice sounds is checked by
ear.
"""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from jarvis.output import tts as tts_module

SAMPLE_RATE = 24000
SAMPLES_PER_FRAME = 1920  # 80 ms of audio at SAMPLE_RATE


# ── Stand-ins ─────────────────────────────────────────────────────────


class _Pcm:
    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Mimi:
    sample_rate = SAMPLE_RATE

    def streaming(self, batch_size):
        return contextlib.nullcontext()

    def decode(self, codes):
        return _Pcm(np.full((1, 1, SAMPLES_PER_FRAME), 0.25, dtype=np.float32))


class Stage:
    """The fake model, the fake speakers, and what the tests want to know."""

    def __init__(self):
        self.load_attempts = 0
        self.load_error = None
        self.load_gate = None     # a model load waits on it
        self.load_started = threading.Event()
        self.loaded = []          # keyword arguments of each model load
        self.voices = []          # names asked of the voice repository
        self.conditioned = []     # voices a synthesis was conditioned on
        self.scripts = []         # texts the model was asked to speak
        self.frames = 25          # audible frames per synthesis
        self.delay = 2            # frames carrying -1 before the audio
        self.emitted = 0          # audible frames produced, all syntheses
        self.fail_with = None
        self.gate = None          # a synthesis waits on it before its first frame
        self.entered = threading.Event()
        self.env_at_import = []   # NO_TORCH_COMPILE whenever the package is used
        self.streams = []
        self.first_sounds = []
        self.captures = []        # (graph, keyword arguments) of each CUDA graph capture
        self.compile_module = None  # the stand-in for moshi.utils.compile

    @property
    def seconds(self):
        return self.frames * SAMPLES_PER_FRAME / SAMPLE_RATE


class _Package(types.ModuleType):
    """A module that notes the environment each time something is taken
    from it, which is when the real package would read its switches."""

    def __init__(self, name, attrs, stage):
        super().__init__(name)
        self._attrs = attrs
        self._stage = stage

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            value = self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None
        self._stage.env_at_import.append(os.environ.get("NO_TORCH_COMPILE"))
        return value


def _install_fake_moshi(monkeypatch, stage):
    class CheckpointInfo:
        @staticmethod
        def from_hf_repo(repo):
            return SimpleNamespace(repo=repo)

    class TTSModel:
        def __init__(self):
            self.mimi = _Mimi()
            self.delay_steps = stage.delay

        @classmethod
        def from_checkpoint_info(cls, info, **kwargs):
            stage.load_attempts += 1
            stage.load_started.set()
            if stage.load_gate is not None:
                stage.load_gate.wait(10)
            if stage.load_error is not None:
                raise stage.load_error
            stage.loaded.append(kwargs)
            return cls()

        def get_voice_path(self, name):
            stage.voices.append(name)
            return "voice-file:" + name

        def prepare_script(self, script, padding_between=0):
            stage.scripts.append(script[0])
            return ["entry"]

        def make_condition_attributes(self, voices, cfg_coef=None):
            stage.conditioned.append([str(v) for v in voices])
            return "condition"

        def generate(self, all_entries, conditions, on_frame=None):
            stage.entered.set()
            if stage.gate is not None:
                stage.gate.wait(10)
            if stage.fail_with is not None:
                raise stage.fail_with
            # moshi captures its CUDA graphs while it generates, through the
            # `cuda` of moshi.utils.compile as it is at that moment.
            cuda = getattr(stage.compile_module, "cuda", None)
            if cuda is not None:
                cuda.graph("a graph in generate")
            for _ in range(stage.delay):
                on_frame(np.full((1, 33, 1), -1, dtype=np.int64))
            for _ in range(stage.frames):
                on_frame(np.zeros((1, 33, 1), dtype=np.int64))
                stage.emitted += 1
            return SimpleNamespace(frames=[])

    loaders = _Package("moshi.models.loaders", {"CheckpointInfo": CheckpointInfo}, stage)
    tts = _Package(
        "moshi.models.tts",
        {"TTSModel": TTSModel, "DEFAULT_DSM_TTS_REPO": "stand-in/tts"},
        stage,
    )
    class _Cuda:
        """What moshi.utils.compile calls ``cuda``: torch.cuda, as far as graphs go."""

        CUDAGraph = object()

        def graph(self, graph, **kwargs):
            stage.captures.append((graph, kwargs))
            return graph

    compile_module = types.ModuleType("moshi.utils.compile")
    compile_module.cuda = _Cuda()
    stage.compile_module = compile_module
    utils_module = types.ModuleType("moshi.utils")
    utils_module.compile = compile_module

    monkeypatch.setitem(sys.modules, "moshi", types.ModuleType("moshi"))
    monkeypatch.setitem(sys.modules, "moshi.utils", utils_module)
    monkeypatch.setitem(sys.modules, "moshi.utils.compile", compile_module)
    monkeypatch.setitem(sys.modules, "moshi.models", types.ModuleType("moshi.models"))
    monkeypatch.setitem(sys.modules, "moshi.models.loaders", loaders)
    monkeypatch.setitem(sys.modules, "moshi.models.tts", tts)


def _install_fake_sounddevice(monkeypatch, stage):
    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    class OutputStream:
        def __init__(self, samplerate, channels, dtype, blocksize, callback):
            self.samplerate = samplerate
            self.dtype = dtype
            self.blocksize = blocksize
            self.callback = callback
            self.active = False
            self.started = False
            self.finished = False
            self.aborted = False
            self.audio = []
            stage.streams.append(self)

        def start(self):
            self.started = True
            self.active = True
            threading.Thread(target=self._pump, daemon=True).start()

        def _pump(self):
            try:
                while not self.aborted:
                    block = np.zeros((self.blocksize, 1), dtype=np.float32)
                    try:
                        self.callback(block, self.blocksize, None, None)
                    except CallbackStop:
                        self.audio.append(block[:, 0].copy())
                        break
                    except CallbackAbort:
                        break
                    self.audio.append(block[:, 0].copy())
                    time.sleep(0.005)
            finally:
                self.finished = True
                self.active = False

        def abort(self):
            self.aborted = True
            self.active = False

        def close(self):
            pass

    module = types.ModuleType("sounddevice")
    module.OutputStream = OutputStream
    module.CallbackStop = CallbackStop
    module.CallbackAbort = CallbackAbort
    monkeypatch.setitem(sys.modules, "sounddevice", module)


def _unset_env(monkeypatch, name):
    # setenv first so the variable is put back (or removed) after the test
    # whatever the engine did to it.
    monkeypatch.setenv(name, "placeholder")
    monkeypatch.delenv(name)


def _until(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def stage(monkeypatch):
    s = Stage()
    _install_fake_moshi(monkeypatch, s)
    _install_fake_sounddevice(monkeypatch, s)
    monkeypatch.setattr(tts_module.KyutaiTTS, "_cuda_problem", staticmethod(lambda: None))
    monkeypatch.setattr(tts_module.KyutaiTTS, "_triton_available", staticmethod(lambda: True))
    monkeypatch.setattr(tts_module, "_sound_started", lambda: s.first_sounds.append(
        bool(s.streams) and s.streams[-1].started))
    _unset_env(monkeypatch, "NO_TORCH_COMPILE")
    _unset_env(monkeypatch, "NO_CUDA_GRAPH")
    return s


@pytest.fixture
def make_engine(stage):
    made = []

    def _make(**kwargs):
        engine = tts_module.KyutaiTTS(enabled=True, **kwargs)
        made.append(engine)
        return engine

    yield _make
    for engine in made:
        engine.stop()


def _speak(engine, text, **kwargs):
    """Speak one item and wait for it; returns (heard, durations)."""
    done = threading.Event()
    durations = []
    engine.speak(text, completion_callback=done.set,
                 duration_callback=durations.append, **kwargs)
    return done.wait(5), durations


# ── The surface ───────────────────────────────────────────────────────


class TestKyutaiInterface:
    def test_has_same_interface_as_other_engines(self):
        tts = tts_module.KyutaiTTS(enabled=False)
        for name in ("start", "stop", "speak", "interrupt", "is_speaking",
                     "get_last_spoken_text"):
            assert callable(getattr(tts, name)), f"missing {name}"

    def test_disabled_is_a_no_op(self, stage):
        tts = tts_module.KyutaiTTS(enabled=False)
        tts.start()
        tts.speak("bonjour")
        assert tts.is_speaking() is False
        tts.interrupt()
        tts.stop()
        assert stage.load_attempts == 0, "a disabled engine must never load the model"
        assert stage.streams == []

    def test_stores_its_voice(self):
        assert tts_module.KyutaiTTS(enabled=True, voice="some/voice.wav").voice == "some/voice.wav"

    def test_blank_voice_falls_back_to_the_default(self):
        assert tts_module.KyutaiTTS(enabled=True, voice="").voice == tts_module.KYUTAI_DEFAULT_VOICE
        assert tts_module.KyutaiTTS(enabled=True, voice="  ").voice == tts_module.KYUTAI_DEFAULT_VOICE


# ── From the settings to the engine ───────────────────────────────────


def _settings_from(config):
    from jarvis.config import load_settings

    with patch("jarvis.config._load_json", return_value={"_config_version": 1, **config}), \
         patch("jarvis.config._save_json", return_value=True):
        return load_settings()


class TestWiring:
    @pytest.mark.parametrize("name", ["kyutai", "KYUTAI", "Kyutai"])
    def test_factory_resolves_the_name_in_any_case(self, name):
        tts = tts_module.create_tts_engine(engine=name, enabled=False, kyutai_voice="a/b.wav")
        assert isinstance(tts, tts_module.KyutaiTTS)
        assert tts.voice == "a/b.wav"

    def test_other_engines_still_resolve(self):
        assert isinstance(tts_module.create_tts_engine(engine="piper", enabled=False),
                          tts_module.PiperTTS)
        assert isinstance(tts_module.create_tts_engine(engine="kokoro", enabled=False),
                          tts_module.KokoroTTS)
        assert isinstance(tts_module.create_tts_engine(engine="nothing", enabled=False),
                          tts_module.PiperTTS)

    def test_the_config_layer_keeps_kyutai_as_a_valid_engine(self):
        assert _settings_from({"tts_engine": "Kyutai"}).tts_engine == "kyutai"

    def test_the_default_voice_in_the_config_is_the_engines_default(self):
        from jarvis.config import get_default_config

        assert get_default_config()["tts_kyutai_voice"] == tts_module.KYUTAI_DEFAULT_VOICE

    def test_a_voice_in_the_config_file_reaches_the_engine_the_daemon_builds(self):
        from jarvis import daemon

        settings = _settings_from({
            "tts_engine": "kyutai",
            "tts_enabled": True,
            "tts_kyutai_voice": "unmute-prod-website/developpeuse-3.wav",
        })

        engine = daemon.build_tts_engine(settings)

        assert type(engine).__name__ == "KyutaiTTS"
        assert engine.voice == "unmute-prod-website/developpeuse-3.wav"

    def test_a_blank_voice_in_the_config_file_means_the_default(self):
        settings = _settings_from({"tts_engine": "kyutai", "tts_kyutai_voice": "  "})
        assert settings.tts_kyutai_voice == tts_module.KYUTAI_DEFAULT_VOICE


# ── Initialisation ────────────────────────────────────────────────────


class TestInitialisation:
    def test_loads_the_model_on_the_gpu(self, stage, make_engine):
        tts = make_engine()
        assert tts._ensure_initialized() is True
        assert stage.loaded and stage.loaded[0]["device"] == "cuda"

    def test_without_a_gpu_it_fails_with_a_reason_and_loads_nothing(
            self, stage, make_engine, monkeypatch):
        monkeypatch.setattr(tts_module.KyutaiTTS, "_cuda_problem",
                            staticmethod(lambda: "no NVIDIA GPU was found"))
        tts = make_engine()

        assert tts._ensure_initialized() is False

        assert "no NVIDIA GPU was found" in tts._init_error
        assert stage.load_attempts == 0

    def test_without_the_package_it_fails_and_names_it(self, stage, make_engine, monkeypatch):
        monkeypatch.setitem(sys.modules, "moshi.models.loaders", None)
        tts = make_engine()

        assert tts._ensure_initialized() is False

        assert "moshi" in tts._init_error

    def test_a_model_that_fails_to_load_is_not_retried(self, stage, make_engine):
        stage.load_error = RuntimeError("out of memory")
        tts = make_engine()

        assert tts._ensure_initialized() is False
        assert tts._ensure_initialized() is False

        assert stage.load_attempts == 1
        assert "out of memory" in tts._init_error

    def test_initialisation_ends_with_one_throwaway_synthesis(self, stage, make_engine):
        tts = make_engine()

        assert tts._ensure_initialized() is True

        assert stage.scripts == [tts._WARMUP_TEXT]
        assert tts.get_last_spoken_text() == ""
        assert tts.is_speaking() is False
        assert tts._q.empty()
        assert stage.streams == [], "the warm-up audio must be discarded, not played"

    def test_a_failed_warmup_does_not_fail_initialisation(self, stage, make_engine):
        stage.fail_with = RuntimeError("boom")
        tts = make_engine()

        assert tts._ensure_initialized() is True
        assert tts._init_error is None


class TestCompileSwitch:
    """``torch.compile`` needs Triton, which has no Windows build. The package
    reads its switch when it is first imported."""

    def test_without_triton_compilation_is_off_before_the_package_is_used(
            self, stage, make_engine, monkeypatch):
        monkeypatch.setattr(tts_module.KyutaiTTS, "_triton_available", staticmethod(lambda: False))

        make_engine()._ensure_initialized()

        assert stage.env_at_import, "the package was never used"
        assert all(seen == "1" for seen in stage.env_at_import)

    def test_the_cuda_graphs_are_never_switched_off(self, stage, make_engine, monkeypatch):
        monkeypatch.setattr(tts_module.KyutaiTTS, "_triton_available", staticmethod(lambda: False))

        make_engine()._ensure_initialized()

        assert "NO_CUDA_GRAPH" not in os.environ

    def test_with_triton_the_switch_is_left_alone(self, stage, make_engine):
        make_engine()._ensure_initialized()

        assert "NO_TORCH_COMPILE" not in os.environ

    def test_a_value_the_user_set_is_left_alone(self, stage, make_engine, monkeypatch):
        monkeypatch.setattr(tts_module.KyutaiTTS, "_triton_available", staticmethod(lambda: False))
        monkeypatch.setenv("NO_TORCH_COMPILE", "their-choice")

        make_engine()._ensure_initialized()

        assert os.environ["NO_TORCH_COMPILE"] == "their-choice"

    @pytest.mark.parametrize(("platform", "imports_on_the_calling_thread"),
                             [("win32", True), ("linux", False)])
    def test_start_imports_the_package_on_the_calling_thread_on_windows_only(
            self, stage, make_engine, monkeypatch, platform, imports_on_the_calling_thread):
        """Loading PyTorch's DLLs from two threads at once deadlocks the
        Windows loader, so there the import happens before any worker, and
        after the compile switch is set."""
        monkeypatch.setattr(tts_module.KyutaiTTS, "_triton_available", staticmethod(lambda: False))
        monkeypatch.setattr(sys, "platform", platform)

        class _NoThread:
            def __init__(self, target=None, daemon=False, name=None):
                pass

            def start(self):
                pass

            def join(self, timeout=None):
                pass

        monkeypatch.setattr("threading.Thread", _NoThread)

        make_engine().start()

        if imports_on_the_calling_thread:
            assert stage.env_at_import and all(seen == "1" for seen in stage.env_at_import)
        else:
            assert stage.env_at_import == []


class TestVoice:
    def test_a_name_is_resolved_through_the_voice_repository(self, stage, make_engine):
        make_engine(voice="cml-tts/fr/some-voice.wav")._ensure_initialized()

        assert stage.voices == ["cml-tts/fr/some-voice.wav"]
        assert stage.conditioned[-1] == ["voice-file:cml-tts/fr/some-voice.wav"]

    def test_a_safetensors_path_is_used_as_it_is(self, stage, make_engine, tmp_path):
        embedding = tmp_path / "mine.safetensors"
        embedding.write_bytes(b"x")

        make_engine(voice=str(embedding))._ensure_initialized()

        assert stage.voices == []
        assert stage.conditioned[-1] == [str(embedding)]

    def test_a_path_that_does_not_exist_fails_and_is_not_replaced(
            self, stage, make_engine, tmp_path):
        missing = tmp_path / "gone.safetensors"
        tts = make_engine(voice=str(missing))

        assert tts._ensure_initialized() is False

        assert str(missing) in tts._init_error
        assert stage.voices == []
        assert stage.load_attempts == 0, "the model was loaded for a voice file that is not there"

    @pytest.mark.parametrize("name", [
        "cml-tts/fr/577_394_000070-0001_enhanced.wav",
        "unmute-prod-website/developpeuse-3.wav",
        "ears/p001/freeform_speech_01.wav",
    ])
    def test_the_names_of_the_voice_repository_are_accepted(self, stage, make_engine, name):
        tts = make_engine(voice=name)

        assert tts._ensure_initialized() is True

        assert stage.voices == [name]

    @pytest.mark.parametrize("value", [
        r"C:\Users\Alice\Documents\me.wav",
        "/home/alice/me.wav",
        "~/me.wav",
        "hf://some-org/some-repo/x.wav",
        "file:///etc/hosts",
        "folder\\me.wav",
        "a/../b.wav",
        "a//b.wav",
    ])
    def test_a_value_that_is_not_a_repository_name_never_reaches_the_network(
            self, stage, make_engine, value):
        """A local path would travel inside the request for the voice file,
        with the user's name and folders in it."""
        tts = make_engine(voice=value)

        assert tts._ensure_initialized() is False

        assert tts._init_error
        assert stage.voices == [], "the voice repository was asked for a local path"
        assert stage.load_attempts == 0, "the model was loaded for a voice that cannot be used"


# ── What is played, and what is promised ──────────────────────────────


class TestPlayback:
    def test_the_audio_reaches_the_speakers_at_the_codec_rate(self, stage, make_engine):
        tts = make_engine()

        heard, _ = _speak(tts, "Bonjour.")

        assert heard
        stream = stage.streams[-1]
        assert stream.samplerate == SAMPLE_RATE
        assert stream.dtype == "float32"
        audio = np.concatenate(stream.audio)
        assert np.count_nonzero(audio) == stage.frames * SAMPLES_PER_FRAME

    def test_frames_before_the_audio_starts_are_not_played(self, stage, make_engine):
        stage.delay = 5
        tts = make_engine()

        _speak(tts, "Bonjour.")

        audio = np.concatenate(stage.streams[-1].audio)
        assert np.count_nonzero(audio) == stage.frames * SAMPLES_PER_FRAME

    def test_the_duration_is_exact_and_known_before_playback_starts(self, stage, make_engine):
        tts = make_engine()
        seen = []
        done = threading.Event()

        tts.speak("Bonjour.", completion_callback=done.set,
                  duration_callback=lambda d: seen.append((d, len(stage.streams))))
        assert done.wait(5)

        duration, streams_open = seen[-1]
        assert duration == pytest.approx(stage.seconds)
        assert streams_open == 0

    def test_the_completion_callback_fires_once_the_audio_has_played_to_its_end(
            self, stage, make_engine):
        tts = make_engine()
        finished_at_completion = []
        done = threading.Event()

        def completed():
            finished_at_completion.append(stage.streams[-1].finished)
            done.set()

        tts.speak("Bonjour.", completion_callback=completed)

        assert done.wait(5)
        assert finished_at_completion == [True]

    def test_the_first_sound_is_marked_once_the_stream_is_running(self, stage, make_engine):
        tts = make_engine()

        _speak(tts, "Bonjour.")

        assert stage.first_sounds == [True]

    def test_the_text_is_spoken_without_its_markup(self, stage, make_engine):
        tts = make_engine()

        _speak(tts, "**Bonjour** toi.")

        assert stage.scripts[-1] == "Bonjour toi."
        assert tts.get_last_spoken_text() == "Bonjour toi."

    def test_the_language_it_is_given_changes_nothing(self, stage, make_engine):
        tts = make_engine()

        heard, _ = _speak(tts, "Bonjour.", language="de")

        assert heard
        assert stage.scripts[-1] == "Bonjour."

    def test_the_end_of_reply_marker_fires_after_everything_queued_before_it(
            self, stage, make_engine):
        tts = make_engine()
        order = []
        done = threading.Event()

        tts.speak("Une.", completion_callback=lambda: order.append("une"))
        tts.speak("Deux.", completion_callback=lambda: order.append("deux"))
        tts.speak("", completion_callback=lambda: (order.append("fin"), done.set()))

        assert done.wait(5)
        assert order == ["une", "deux", "fin"]
        assert stage.scripts[-2:] == ["Une.", "Deux."]


class TestFailures:
    def _marker(self, tts):
        """Fires once the worker has got past everything queued before it."""
        reached = threading.Event()
        tts.speak("", completion_callback=reached.set)
        return reached

    def test_a_synthesis_that_raises_settles_nothing_and_the_next_item_is_spoken(
            self, stage, make_engine):
        tts = make_engine()
        assert tts._ensure_initialized()
        said = []
        stage.fail_with = RuntimeError("boom")
        tts.speak("Perdu.", completion_callback=lambda: said.append("perdu"))
        assert self._marker(tts).wait(5)
        assert said == []
        assert tts.is_speaking() is False

        stage.fail_with = None
        heard, _ = _speak(tts, "Retrouvé.")

        assert heard

    def test_a_synthesis_that_produces_no_audio_settles_nothing(self, stage, make_engine):
        tts = make_engine()
        assert tts._ensure_initialized()
        stage.frames = 0
        said = []

        tts.speak("Rien.", completion_callback=lambda: said.append(1))

        assert self._marker(tts).wait(5)
        assert said == []
        assert stage.streams == []

    def test_an_engine_that_could_not_start_says_nothing_and_settles_nothing(
            self, stage, make_engine, monkeypatch):
        monkeypatch.setattr(tts_module.KyutaiTTS, "_cuda_problem",
                            staticmethod(lambda: "no NVIDIA GPU was found"))
        tts = make_engine()
        said = []

        tts.speak("Rappel.", completion_callback=lambda: said.append(1))

        assert self._marker(tts).wait(5)
        assert said == []
        assert stage.streams == []


class TestInterrupt:
    def test_items_queued_before_an_interrupt_are_dropped_and_later_ones_spoken(
            self, stage, make_engine):
        tts = make_engine()
        tts._thread = object()  # as if started: speak() queues without starting a worker
        called = []

        tts.speak("Avant.", completion_callback=lambda: called.append("avant"))
        tts.speak("Aussi avant.")
        tts.speak("", completion_callback=lambda: called.append("marqueur"))
        tts.interrupt()
        tts._thread = None
        heard, _ = _speak(tts, "Après.")

        assert heard
        assert called == []
        assert "Avant." not in stage.scripts and "Aussi avant." not in stage.scripts
        assert stage.scripts[-1] == "Après."

    def test_a_synthesis_in_flight_is_abandoned_at_its_next_frame(self, stage, make_engine):
        tts = make_engine()
        assert tts._ensure_initialized()
        stage.frames = 200
        stage.gate = threading.Event()
        stage.entered.clear()
        said = []

        tts.speak("Longue.", completion_callback=lambda: said.append(1))
        assert stage.entered.wait(5)
        tts.interrupt()
        stage.gate.set()

        assert _until(lambda: not tts.is_speaking())
        assert stage.emitted < stage.frames
        assert said == []
        assert stage.streams == []

    def test_an_interrupt_during_playback_stops_it_and_settles_nothing(self, stage, make_engine):
        tts = make_engine()
        assert tts._ensure_initialized()
        stage.frames = 200
        said = []

        tts.speak("Longue.", completion_callback=lambda: said.append(1))
        assert _until(lambda: stage.streams and stage.streams[-1].started)
        tts.interrupt()

        assert _until(lambda: not tts.is_speaking())
        assert stage.streams[-1].aborted
        assert said == []

    def test_an_interrupt_with_nothing_to_stop_is_harmless(self, stage, make_engine):
        tts = make_engine()

        for _ in range(5):
            tts.interrupt()

        heard, _ = _speak(tts, "Toujours là.")
        assert heard

    def test_stop_ends_the_worker_even_after_interrupting(self, stage, make_engine):
        tts = make_engine()
        _speak(tts, "Un.")

        tts.stop()

        assert tts._thread is None

    def test_a_stop_during_the_load_abandons_the_item_before_it_is_synthesised(
            self, stage, make_engine):
        """The first item waits for the model to load, which can take minutes.
        A stop that lands meanwhile has to cost the GPU nothing."""
        stage.load_gate = threading.Event()
        tts = make_engine()
        said = []

        tts.speak("Périmée.", completion_callback=lambda: said.append(1))
        assert stage.load_started.wait(5)
        assert _until(tts.is_speaking)

        tts.interrupt()
        stage.load_gate.set()

        assert _until(lambda: not tts.is_speaking())
        assert stage.scripts == [tts._WARMUP_TEXT]
        assert said == []


class TestLifecycle:
    def test_start_loads_the_model_in_the_background_before_anything_is_said(
            self, stage, make_engine):
        stage.load_gate = threading.Event()
        tts = make_engine()

        tts.start()

        assert stage.load_started.wait(5), "start() did not begin loading"
        assert "kyutai-init" in [t.name for t in threading.enumerate()]
        assert tts._q.empty() and stage.streams == []
        stage.load_gate.set()
        assert _until(lambda: tts._initialized)
        assert stage.load_attempts == 1

    def test_starting_twice_leaves_the_first_worker_in_place(self, stage, make_engine):
        tts = make_engine()
        tts.start()
        worker = tts._thread
        assert _until(lambda: tts._initialized)

        tts.start()

        assert tts._thread is worker
        assert stage.load_attempts == 1

    def test_a_sentence_arriving_during_the_load_waits_for_it_and_the_model_loads_once(
            self, stage, make_engine):
        stage.load_gate = threading.Event()
        tts = make_engine()
        done = threading.Event()

        tts.speak("Arrivée pendant le chargement.", completion_callback=done.set)
        assert stage.load_started.wait(5)
        assert _until(tts.is_speaking), "the worker never took the sentence up"
        assert not done.is_set()
        stage.load_gate.set()

        assert done.wait(5)
        assert stage.load_attempts == 1
        assert stage.scripts == [tts._WARMUP_TEXT, "Arrivée pendant le chargement."]

    def test_an_engine_that_cannot_start_says_why_and_does_not_try_again(
            self, stage, make_engine, monkeypatch, capsys):
        asked = []
        monkeypatch.setattr(
            tts_module.KyutaiTTS, "_cuda_problem",
            staticmethod(lambda: asked.append(1) or "no NVIDIA GPU was found"))
        tts = make_engine()
        reached = threading.Event()

        tts.speak("Une.")
        tts.speak("Deux.")
        tts.speak("", completion_callback=reached.set)

        assert reached.wait(5)
        assert "no NVIDIA GPU was found" in capsys.readouterr().out
        assert asked == [1]


class TestGraphCapture:
    """See tts.spec.md, Kyutai, "Graph capture"."""

    @staticmethod
    def _loaded(make_engine, stage):
        """An engine whose model has loaded, and the captures made on the way
        (the warm-up synthesis), which are set aside."""
        tts = make_engine()
        assert tts._ensure_initialized() is True
        made_while_loading = list(stage.captures)
        stage.captures.clear()
        return tts, made_while_loading

    def test_the_capture_made_while_the_model_warms_up_is_already_per_thread(
            self, stage, make_engine):
        _, made_while_loading = self._loaded(make_engine, stage)

        assert made_while_loading == [
            ("a graph in generate", {"capture_error_mode": "thread_local"})
        ]

    def test_the_capture_made_for_each_sentence_is_per_thread_too(self, stage, make_engine):
        tts, _ = self._loaded(make_engine, stage)

        heard, _ = _speak(tts, "Bonjour.")

        assert heard
        assert stage.captures == [
            ("a graph in generate", {"capture_error_mode": "thread_local"})
        ]

    def test_a_capture_mode_moshi_passes_by_keyword_is_kept(self, stage, make_engine):
        self._loaded(make_engine, stage)

        stage.compile_module.cuda.graph("a graph", capture_error_mode="relaxed")

        assert stage.captures == [("a graph", {"capture_error_mode": "relaxed"})]

    def test_everything_else_moshi_asks_of_cuda_is_untouched(self, stage, make_engine):
        original = stage.compile_module.cuda
        self._loaded(make_engine, stage)

        assert stage.compile_module.cuda.CUDAGraph is original.CUDAGraph

    def test_a_second_engine_keeps_the_wrapper_the_first_one_installed(self, stage, make_engine):
        self._loaded(make_engine, stage)
        first = stage.compile_module.cuda

        make_engine()._ensure_initialized()

        assert stage.compile_module.cuda is first

    def test_a_compile_module_with_no_cuda_to_replace_is_left_alone_and_still_loads(
            self, stage, make_engine, monkeypatch):
        monkeypatch.delattr(stage.compile_module, "cuda")
        tts = make_engine()

        assert tts._ensure_initialized() is True
        assert tts._init_error is None
        assert not hasattr(stage.compile_module, "cuda")

    def test_moshis_own_capture_code_asks_torch_for_a_per_thread_capture(self, monkeypatch):
        """Runs moshi's real ``CUDAGraphed`` against a stand-in for torch's CUDA
        module, so it fails if moshi stops reading ``cuda`` from its module
        globals. Not run where moshi is not installed (it is optional)."""
        compile_module = pytest.importorskip("moshi.utils.compile")
        monkeypatch.delenv("NO_CUDA_GRAPH", raising=False)
        modes = []

        class Cuda:
            class CUDAGraph:
                def replay(self):
                    pass

            @staticmethod
            @contextlib.contextmanager
            def graph(graph, **kwargs):
                modes.append(kwargs.get("capture_error_mode"))
                yield

        monkeypatch.setattr(compile_module, "cuda", Cuda)
        tts_module.KyutaiTTS._capture_graphs_per_thread()

        compile_module.CUDAGraphed(lambda x: x, warmup_steps=0)(1)

        assert modes == ["thread_local"]


class TestHardware:
    """A PyTorch that cannot be loaded, one built without CUDA and a machine
    without a GPU are three different fixes, so they are told apart."""

    @staticmethod
    def _fake_torch(monkeypatch, *, cuda_build, available):
        torch = types.ModuleType("torch")
        torch.version = SimpleNamespace(cuda="12.8" if cuda_build else None)
        torch.cuda = SimpleNamespace(is_available=lambda: available)
        monkeypatch.setitem(sys.modules, "torch", torch)

    def test_a_usable_gpu_is_no_problem(self, monkeypatch):
        self._fake_torch(monkeypatch, cuda_build=True, available=True)

        assert tts_module.KyutaiTTS._cuda_problem() is None

    def test_a_pytorch_that_cannot_be_loaded_is_reported_with_its_own_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        reason = tts_module.KyutaiTTS._cuda_problem()

        assert reason and "torch" in reason.lower()

    def test_a_pytorch_built_without_cuda_is_not_reported_as_a_missing_gpu(self, monkeypatch):
        self._fake_torch(monkeypatch, cuda_build=False, available=False)
        no_build = tts_module.KyutaiTTS._cuda_problem()
        self._fake_torch(monkeypatch, cuda_build=True, available=False)
        no_gpu = tts_module.KyutaiTTS._cuda_problem()

        assert no_build and no_gpu and no_build != no_gpu
        assert "CUDA" in no_build
