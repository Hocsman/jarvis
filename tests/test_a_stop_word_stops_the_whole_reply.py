"""A stop word stops the reply, not just the sentence being said.

`tts.spec.md` ("Interrupting") and `streaming.spec.md` ("Interruption
still stops everything") promise that `interrupt()`:

- stops the item playing;
- discards every item queued before the call, so that none of them is
  heard (nor synthesised, by an engine that synthesises one item at a
  time);
- abandons an item still being synthesised;
- fires no completion callback for any of them, the end-of-reply marker
  included, because a hot window opened by an interrupted reply would
  invite an answer to something she did not finish saying;
- still speaks what is queued after the call.

A reply streams into the queue a sentence at a time, and a stop word can
land at any point: with every chunk, the tail and the marker already
waiting, or with only some of them queued. The engine's queue drops what
it holds; the listener queues nothing more of that reply
(tests/test_elle_entend_pendant_qu_elle_reflechit.py).

The tests run each real engine over a stand-in for its model and its
speakers, and observe what was asked of the model, what reached the
speakers, and which callbacks fired.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.output import tts as tts_module

ENGINES = ["piper", "kokoro", "chatterbox", "kyutai"]


class Rig:
    """The model and the speakers of one engine, and what they were asked."""

    def __init__(self):
        self.engine = None
        self.synthesised: list[str] = []  # texts the model was asked to speak
        self.heard: list[str] = []        # texts whose audio reached the speakers
        self.aborted = 0                  # playbacks cut short
        self.languages: list = []         # voices asked for, by language (Piper)
        self.gate: threading.Event | None = None  # synthesis waits on it
        self.entered = threading.Event()  # a synthesis is waiting on the gate
        self.hold_playback = False        # playback lasts until it is aborted
        self.identify_audio = False       # a stream reads whose audio it is fed (Kyutai)
        self.prepares_ahead = False       # the next item is synthesised while one plays (Kyutai)
        self.current = ""

    def begin(self, text: str) -> None:
        self.synthesised.append(text)
        self.current = text

    def pause(self) -> None:
        gate = self.gate
        if gate is not None:
            self.entered.set()
            gate.wait(5)

    def played(self) -> None:
        self.heard.append(self.current)

    def sounddevice(self):
        rig = self

        class _Stream:
            def __init__(self, **kwargs):
                self.active = False
                self.callback = kwargs.get("callback")
                self.blocksize = kwargs.get("blocksize", 1024)

            def start(self):
                if rig.identify_audio:
                    self._start_reading_the_feed()
                    return
                rig.played()
                self.active = rig.hold_playback

            def _start_reading_the_feed(self):
                """An engine that prepares the next item while one plays cannot be
                told apart by the text it began last: the value of the samples it
                feeds the stream says which item they belong to."""
                block = np.zeros((self.blocksize, 1), dtype=np.float32)
                try:
                    self.callback(block, self.blocksize, None, None)
                except (_Abort, _Stop):
                    pass
                number = int(round(float(block[0, 0]) * 100))
                if number:
                    rig.heard.append(rig.synthesised[number - 1])
                if rig.hold_playback:
                    self.active = True
                    return
                for _ in range(10000):  # what a device does: take everything fed
                    try:
                        self.callback(block, self.blocksize, None, None)
                    except (_Abort, _Stop):
                        break

            def abort(self):
                rig.aborted += 1
                self.active = False

            def close(self):
                pass

        class _Abort(Exception):
            pass

        class _Stop(Exception):
            pass

        return SimpleNamespace(OutputStream=_Stream, CallbackAbort=_Abort, CallbackStop=_Stop)

    def start_worker(self) -> None:
        engine = self.engine
        engine._thread = threading.Thread(
            target=engine._run, daemon=True, name="test-tts-worker")
        engine._thread.start()

    def close(self) -> None:
        if self.gate is not None:
            self.gate.set()
        self.hold_playback = False
        self.engine.stop()


def _silent_state(monkeypatch, cls):
    monkeypatch.setattr(cls, "_notify_speaking_state", lambda self, speaking: None)


def _piper_rig(monkeypatch, tmp_path) -> Rig:
    rig = Rig()
    chunk = SimpleNamespace(audio_int16_array=np.zeros(160, dtype=np.int16))

    class _Voice:
        config = SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, syn_config):
            rig.begin(text)
            yield chunk
            rig.pause()
            yield chunk

    monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
    monkeypatch.setitem(sys.modules, "piper.config", SimpleNamespace(
        SynthesisConfig=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "sounddevice", rig.sounddevice())
    _silent_state(monkeypatch, tts_module.PiperTTS)

    engine = tts_module.PiperTTS(enabled=True, model_path=str(tmp_path / "voix.onnx"))
    engine._voice = _Voice()
    engine._sample_rate = 22050
    engine._initialized = True
    original = engine._voice_for

    def _recording(language):
        rig.languages.append(language)
        return original(language)

    engine._voice_for = _recording
    rig.engine = engine
    return rig


def _kokoro_rig(monkeypatch) -> Rig:
    rig = Rig()

    def _pipe(text, voice=None, speed=1.0):
        rig.begin(text)
        yield ("g", "p", np.zeros(240, dtype=np.float32))
        rig.pause()
        yield ("g", "p", np.zeros(240, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "sounddevice", rig.sounddevice())
    _silent_state(monkeypatch, tts_module.KokoroTTS)

    engine = tts_module.KokoroTTS(enabled=True)
    engine._pipe = _pipe
    engine._initialized = True
    rig.engine = engine
    return rig


def _chatterbox_rig(monkeypatch) -> Rig:
    rig = Rig()

    class _Music:
        busy = False

        def load(self, path):
            pass

        def play(self):
            rig.played()
            self.busy = rig.hold_playback

        def stop(self):
            rig.aborted += 1
            self.busy = False

        def get_busy(self):
            return self.busy

    pygame = SimpleNamespace(
        mixer=SimpleNamespace(init=lambda **kwargs: None, quit=lambda: None, music=_Music()),
        time=SimpleNamespace(wait=lambda ms: time.sleep(0.01)),
    )

    def _generate(text, **kwargs):
        rig.begin(text)
        rig.pause()
        return SimpleNamespace(shape=(1, 2400))

    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace(save=lambda *a, **k: None))
    _silent_state(monkeypatch, tts_module.ChatterboxTTS)

    engine = tts_module.ChatterboxTTS(enabled=True)
    engine._model = SimpleNamespace(sr=24000, generate=_generate)
    engine._initialized = True
    rig.engine = engine
    return rig


def _kyutai_rig(monkeypatch) -> Rig:
    rig = Rig()
    frame = np.zeros((1, 33, 1), dtype=np.int64)

    class _Pcm:
        def cpu(self):
            return self

        def numpy(self):
            # The number of the item being synthesised, so that a stream can
            # tell whose audio it was fed (one hundredth each: samples are
            # clipped to [-1, 1]).
            return np.full((1, 1, 480), len(rig.synthesised) / 100.0, dtype=np.float32)

    class _Mimi:
        sample_rate = 24000

        def streaming(self, batch_size):
            return contextlib.nullcontext()

        def decode(self, codes):
            return _Pcm()

    class _Model:
        mimi = _Mimi()

        def prepare_script(self, script, padding_between=0):
            rig.begin(script[0])
            return ["entry"]

        def make_condition_attributes(self, voices, cfg_coef=None):
            return "condition"

        def generate(self, entries, conditions, on_frame=None):
            on_frame(frame)
            rig.pause()
            on_frame(frame)
            return SimpleNamespace(frames=[])

    rig.identify_audio = True
    rig.prepares_ahead = True
    monkeypatch.setitem(sys.modules, "sounddevice", rig.sounddevice())
    _silent_state(monkeypatch, tts_module.KyutaiTTS)

    engine = tts_module.KyutaiTTS(enabled=True)
    engine._model = _Model()
    engine._voice_path = "voice"
    engine._initialized = True
    rig.engine = engine
    return rig


@pytest.fixture(params=ENGINES)
def rig(request, monkeypatch, tmp_path):
    if request.param == "piper":
        made = _piper_rig(monkeypatch, tmp_path)
    elif request.param == "kokoro":
        made = _kokoro_rig(monkeypatch)
    elif request.param == "chatterbox":
        made = _chatterbox_rig(monkeypatch)
    else:
        made = _kyutai_rig(monkeypatch)
    made.start_worker()
    yield made
    made.close()


@pytest.fixture
def piper(monkeypatch, tmp_path):
    made = _piper_rig(monkeypatch, tmp_path)
    made.start_worker()
    yield made
    made.close()


def _until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _speak_and_wait(engine, text, **kwargs):
    """Say ``text`` and return once the worker has finished with it."""
    done = threading.Event()
    engine.speak(text, completion_callback=done.set, **kwargs)
    return done.wait(5)


# ── The stop word lands while a sentence is being synthesised ───────────


@pytest.mark.unit
def test_a_stop_word_mid_synthesis_drops_everything_queued_behind_the_sentence(rig):
    engine = rig.engine
    fired = []
    rig.gate = threading.Event()

    engine.speak("Première phrase.", completion_callback=lambda: fired.append("première"))
    assert rig.entered.wait(5), "the first sentence never reached the model"
    engine.speak("Deuxième phrase.")
    engine.speak("Troisième phrase.", completion_callback=lambda: fired.append("troisième"))
    engine.speak("", completion_callback=lambda: fired.append("fin de réponse"))

    engine.interrupt()
    rig.gate.set()

    assert _speak_and_wait(engine, "Autre réponse.")
    assert fired == [], "an interrupted reply settled something"
    assert rig.heard == ["Autre réponse."]
    assert "Deuxième phrase." not in rig.synthesised
    assert "Troisième phrase." not in rig.synthesised


@pytest.mark.unit
def test_the_sentence_being_synthesised_when_the_stop_word_lands_is_never_heard(rig):
    engine = rig.engine
    rig.gate = threading.Event()

    engine.speak("Première phrase.")
    assert rig.entered.wait(5)
    engine.interrupt()
    rig.gate.set()

    assert _speak_and_wait(engine, "Autre réponse.")
    assert rig.heard == ["Autre réponse."]


# ── The stop word lands while a sentence is playing ─────────────────────


@pytest.mark.unit
def test_a_stop_word_during_playback_cuts_the_sound_and_drops_what_follows(rig):
    engine = rig.engine
    fired = []
    rig.hold_playback = True

    engine.speak("Première phrase.", completion_callback=lambda: fired.append("première"))
    assert _until(lambda: rig.heard == ["Première phrase."]), "playback never started"
    engine.speak("Deuxième phrase.")
    engine.speak("Troisième phrase.", completion_callback=lambda: fired.append("troisième"))
    engine.speak("", completion_callback=lambda: fired.append("fin de réponse"))

    engine.interrupt()
    assert _until(lambda: not engine.is_speaking()), "the sound never stopped"
    rig.hold_playback = False

    assert _speak_and_wait(engine, "Autre réponse.")
    assert rig.aborted, "the playing sentence was not cut"
    assert fired == []
    assert rig.heard == ["Première phrase.", "Autre réponse."]
    if not rig.prepares_ahead:
        # An engine that prepares the next item while one plays may already
        # have synthesised some of them when the stop word lands: it is what
        # it does with them, not hearing them, that the contract is about.
        assert rig.synthesised == ["Première phrase.", "Autre réponse."]


@pytest.mark.unit
def test_the_end_of_reply_marker_of_an_interrupted_reply_does_not_open_the_hot_window(rig):
    engine = rig.engine
    window = []
    rig.hold_playback = True

    engine.speak("Dernière phrase.")
    assert _until(lambda: rig.heard == ["Dernière phrase."])
    engine.speak("", completion_callback=lambda: window.append("ouverte"))

    engine.interrupt()
    assert _until(lambda: not engine.is_speaking())
    rig.hold_playback = False

    assert _speak_and_wait(engine, "Autre réponse.")
    assert window == []


# ── What comes after the stop word is spoken ────────────────────────────


@pytest.mark.unit
def test_a_reply_that_follows_a_stop_word_is_spoken_to_the_end_with_its_callbacks(rig):
    engine = rig.engine
    fired = []
    rig.gate = threading.Event()

    engine.speak("Première phrase.")
    assert rig.entered.wait(5)
    engine.interrupt()
    rig.gate.set()

    engine.speak("Réponse une.", duration_callback=lambda s: fired.append("durée"))
    engine.speak("Réponse deux.", completion_callback=lambda: fired.append("deux"))
    marker_reached = threading.Event()
    engine.speak("", completion_callback=lambda: (fired.append("fin"), marker_reached.set()))

    assert marker_reached.wait(5)
    assert rig.heard == ["Réponse une.", "Réponse deux."]
    assert fired == ["durée", "deux", "fin"]


@pytest.mark.unit
def test_a_second_stop_word_stops_the_reply_that_followed_the_first(rig):
    engine = rig.engine
    fired = []
    rig.hold_playback = True

    engine.interrupt()
    engine.speak("Réponse une.", completion_callback=lambda: fired.append("une"))
    assert _until(lambda: rig.heard == ["Réponse une."])
    engine.speak("Réponse deux.", completion_callback=lambda: fired.append("deux"))

    engine.interrupt()
    assert _until(lambda: not engine.is_speaking())
    rig.hold_playback = False

    assert _speak_and_wait(engine, "Dernière réponse.")
    assert fired == []
    assert rig.heard == ["Réponse une.", "Dernière réponse."]


# ── The stop word lands between two steps of the worker ─────────────────


class _InterruptsWhenTheWorkerPicksUpAnItem(threading.Event):
    """The speaking flag, which the worker sets the instant it takes an item
    up. The stop word lands there: after the item left the queue, before the
    engine has begun to speak it."""

    def __init__(self, engine):
        super().__init__()
        self._engine = engine
        self.armed = True

    def set(self):
        if self.armed:
            self._engine.interrupt()
            self.armed = False
        super().set()


@pytest.mark.unit
def test_a_stop_word_landing_as_the_worker_takes_an_item_up_is_not_lost(rig):
    engine = rig.engine
    fired = []
    engine._is_speaking = _InterruptsWhenTheWorkerPicksUpAnItem(engine)

    engine.speak("Déjà prise.", completion_callback=lambda: fired.append("prise"))

    assert _until(lambda: not engine._is_speaking.armed)
    assert _speak_and_wait(engine, "Autre réponse.")
    assert fired == []
    assert rig.heard == ["Autre réponse."]
    assert "Déjà prise." not in rig.synthesised


@pytest.mark.unit
def test_an_item_stopped_as_the_worker_takes_it_up_does_not_publish_speaking(rig):
    """The listener answers a stop word by publishing IDLE. A SPEAKING that
    followed it, from an item the stop had already cancelled, would stay on
    the orb until the next wake word."""
    engine = rig.engine
    published = []
    engine._notify_speaking_state = published.append
    engine._is_speaking = _InterruptsWhenTheWorkerPicksUpAnItem(engine)

    engine.speak("Déjà prise.")
    assert _until(lambda: not engine._is_speaking.armed)
    assert _speak_and_wait(engine, "Autre réponse.")

    # The engines also call the hook when speech ends, which publishes nothing.
    assert [speaking for speaking in published if speaking] == [True], (
        "only the sentence that was spoken says so"
    )


# ── Nothing to stop ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_stop_word_with_nothing_to_stop_swallows_nothing_that_follows(rig):
    engine = rig.engine

    for _ in range(3):
        engine.interrupt()

    assert _speak_and_wait(engine, "Première réponse.")
    assert rig.heard == ["Première réponse."]


@pytest.mark.unit
def test_stopping_the_engine_after_an_interrupt_ends_the_worker(rig):
    engine = rig.engine
    assert _speak_and_wait(engine, "Une phrase.")

    engine.interrupt()
    engine.stop()

    assert engine._thread is None


# ── Piper carries the language of each item through the same queue ──────


@pytest.mark.unit
def test_piper_still_speaks_each_item_in_the_language_it_was_queued_with(piper):
    engine = piper.engine

    assert _speak_and_wait(engine, "Il fait beau.", language="fr")
    engine.interrupt()
    assert _speak_and_wait(engine, "Es ist schön.", language="de")

    assert piper.languages == ["fr", "de"]
    assert piper.heard == ["Il fait beau.", "Es ist schön."]
