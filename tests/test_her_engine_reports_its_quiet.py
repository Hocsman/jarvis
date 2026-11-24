"""Her engine says how long it played quiet after her last sound.

A speech engine ends an utterance on a stretch of near silence, and the
moment its completion callback fires is when that stretch has played, not
when she stopped making a sound. His answer to the question she asks before
acting comes at once, inside that stretch, so the listener needs to know how
long it was (`quiet_tail`): each engine reads it from the audio it played.

The audio here is made up: a steady tone stands in for her voice, zeros or a
faint hiss for the quiet after it.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.output import tts as tts_module
from jarvis.output.tts import (
    QUIET_BELOW_LOUDEST_DB, QUIET_WINDOW_S, ChatterboxTTS, KokoroTTS, KyutaiTTS, PiperTTS,
    TrailingQuiet,
)

RATE = 16000


def _voice(seconds: float, level: float = 0.3) -> np.ndarray:
    """A tone at *level* for *seconds*: her voice, as far as loudness goes."""
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _quiet(seconds: float, level: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    return (level * rng.standard_normal(int(seconds * RATE))).astype(np.float32)


def _below_her_loudest(db: float) -> float:
    """The level of a hiss *db* decibels under the tone's own RMS."""
    return 0.3 / np.sqrt(2) * 10 ** (-db / 20)


def _read(*pieces, rate=RATE) -> float:
    quiet = TrailingQuiet(rate)
    for piece in pieces:
        quiet.add(piece)
    return quiet.seconds()


# ── Reading the quiet out of the audio ────────────────────────────────


def test_the_quiet_after_her_last_sound_is_measured():
    assert _read(np.concatenate([_voice(1.0), _quiet(0.8)])) == pytest.approx(0.8, abs=QUIET_WINDOW_S)


def test_audio_ending_on_her_voice_has_no_quiet():
    assert _read(_voice(1.0)) == pytest.approx(0.0, abs=QUIET_WINDOW_S)


def test_a_hiss_far_under_her_voice_is_quiet():
    hiss = _below_her_loudest(QUIET_BELOW_LOUDEST_DB + 10)

    assert _read(np.concatenate([_voice(1.0), _quiet(0.8, hiss)])) == pytest.approx(
        0.8, abs=QUIET_WINDOW_S)


def test_a_soft_sound_near_her_voice_is_still_her_sound():
    """A soft ending (a breath, a consonant) within the margin counts as
    sound: the quiet is measured short rather than long."""
    soft = _below_her_loudest(QUIET_BELOW_LOUDEST_DB - 10)

    assert _read(np.concatenate([_voice(1.0), _quiet(0.8, soft)])) == pytest.approx(
        0.0, abs=QUIET_WINDOW_S)


def test_a_sound_shorter_than_a_window_at_the_very_end_is_her_sound():
    """Her audio rarely ends on a whole window: a loud last bit too short
    for one still ends the quiet."""
    window = int(round(RATE * QUIET_WINDOW_S))
    audio = np.concatenate([_voice(1.0), _quiet(0.5), _voice(1.0)[: window // 4]])
    assert len(audio) % window

    assert _read(audio) == pytest.approx(0.0, abs=1e-9)


def test_audio_fed_in_pieces_reads_as_the_whole():
    whole = np.concatenate([_quiet(0.1), _voice(0.7), _quiet(0.05), _voice(0.3), _quiet(0.6)])
    pieces = np.array_split(whole, [333, 1000, 4097, 9001, 12000])

    assert _read(*pieces) == pytest.approx(_read(whole), abs=1e-9)
    assert _read(whole) == pytest.approx(0.6, abs=QUIET_WINDOW_S)


def test_whole_numbers_read_like_their_floats():
    """Piper plays 16-bit samples."""
    audio = np.concatenate([_voice(1.0), _quiet(0.5)])

    assert _read((audio * 32767).astype(np.int16)) == pytest.approx(_read(audio), abs=1e-9)


def test_audio_with_no_sound_is_quiet_throughout():
    assert _read(_quiet(0.4)) == pytest.approx(0.4, abs=1e-9)


def test_no_audio_has_no_quiet():
    assert _read() == 0.0


def test_audio_it_cannot_read_has_no_quiet():
    """Read on the way to her speakers, it never stops her speaking: her
    last sound is then when her reply was done."""
    assert _read(_voice(1.0), object(), _quiet(0.5)) == 0.0


# ── Each engine reports it ────────────────────────────────────────────


class _FakeSoundDevice(types.ModuleType):
    """Speakers that play each block the moment it is asked for."""

    class CallbackStop(Exception):
        pass

    class CallbackAbort(Exception):
        pass

    def __init__(self):
        super().__init__("sounddevice")
        device = self

        class OutputStream:
            def __init__(self, samplerate, channels, dtype, blocksize, callback):
                self.blocksize, self.callback, self.dtype = blocksize, callback, dtype
                self.active = False

            def start(self):
                self.active = True
                threading.Thread(target=self._pump, daemon=True).start()

            def _pump(self):
                try:
                    while self.active:
                        block = np.zeros((self.blocksize, 1), dtype=self.dtype)
                        try:
                            self.callback(block, self.blocksize, None, None)
                        except (device.CallbackStop, device.CallbackAbort):
                            break
                finally:
                    self.active = False

            def abort(self):
                self.active = False

            def close(self):
                pass

        self.OutputStream = OutputStream


@pytest.fixture
def speakers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSoundDevice())
    monkeypatch.setattr(tts_module, "_sound_started", lambda: None)


def _her_reply_then(quiet_seconds: float) -> np.ndarray:
    return np.concatenate([_voice(0.6), _quiet(quiet_seconds)])


@pytest.mark.parametrize("engine", [PiperTTS, KokoroTTS, ChatterboxTTS, KyutaiTTS])
def test_an_engine_that_played_nothing_reports_no_quiet(engine):
    assert engine(enabled=False).quiet_tail() == 0.0


def test_piper_reports_the_quiet_it_played(speakers):
    tts = PiperTTS(enabled=True)
    audio = (_her_reply_then(0.5) * 32767).astype(np.int16)
    voice = SimpleNamespace(synthesize=lambda text, config: [
        SimpleNamespace(audio_int16_array=audio)])
    tts._voice_for = lambda language: (voice, RATE)
    done = []
    tts._completion_callback = lambda: done.append(tts.quiet_tail())

    tts._speak_once("Une phrase.")

    assert done == [pytest.approx(0.5, abs=QUIET_WINDOW_S)]


def test_kokoro_reports_the_quiet_it_played(speakers):
    tts = KokoroTTS(enabled=True)
    tts._sample_rate = RATE
    tts._initialized = True
    tts._pipe = lambda text, voice=None, speed=None: iter([("g", "p", _her_reply_then(0.4))])
    done = []
    tts._completion_callback = lambda: done.append(tts.quiet_tail())

    tts._speak_once("Une phrase.")

    assert done == [pytest.approx(0.4, abs=QUIET_WINDOW_S)]


@pytest.fixture
def chatterbox_speakers(monkeypatch):
    """pygame and torchaudio as Chatterbox plays through them, each item
    played in a few polls."""
    def mixer_for_one_item():
        playing = {"polls": 3}

        def get_busy():
            playing["polls"] -= 1
            return playing["polls"] > 0

        return SimpleNamespace(load=lambda path: None, play=lambda: None, stop=lambda: None,
                               get_busy=get_busy)

    pygame = types.ModuleType("pygame")
    pygame.mixer = SimpleNamespace(init=lambda **kw: setattr(pygame.mixer, "music",
                                                             mixer_for_one_item()),
                                   quit=lambda: None, music=mixer_for_one_item())
    pygame.time = SimpleNamespace(wait=lambda ms: None)
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.save = lambda path, wav, sr: None
    monkeypatch.setitem(sys.modules, "pygame", pygame)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setattr(tts_module, "_sound_started", lambda: None)


def test_chatterbox_reports_the_quiet_it_played(chatterbox_speakers):
    tts = ChatterboxTTS(enabled=True)
    tts._initialized = True
    tts._model = SimpleNamespace(sr=RATE, generate=lambda text, **kw: _her_reply_then(0.3)[None, :])
    done = []
    tts._completion_callback = lambda: done.append(tts.quiet_tail())

    tts._speak_once("Une phrase.")

    assert done == [pytest.approx(0.3, abs=QUIET_WINDOW_S)]


def _cut_short_once_her_sound_starts(monkeypatch, tts):
    """The next item she says is interrupted once its sound has started:
    counting the interruption is `interrupt()`'s first step."""
    monkeypatch.setattr(tts_module, "_sound_started",
                        lambda: setattr(tts, "_epoch", tts._epoch + 1))


def test_a_reply_cut_short_leaves_the_quiet_of_the_last_one_heard(speakers, monkeypatch):
    """Interrupted means unheard: what it would have played says nothing of
    her last sound."""
    tts = PiperTTS(enabled=True)
    audio = (_her_reply_then(0.5) * 32767).astype(np.int16)
    voice = SimpleNamespace(synthesize=lambda text, config: [
        SimpleNamespace(audio_int16_array=audio)])
    tts._voice_for = lambda language: (voice, RATE)
    tts._speak_once("Une phrase.")
    heard = tts.quiet_tail()

    cut = (_her_reply_then(1.5) * 32767).astype(np.int16)
    voice.synthesize = lambda text, config: [SimpleNamespace(audio_int16_array=cut)]
    _cut_short_once_her_sound_starts(monkeypatch, tts)
    tts._speak_once("Une autre.", epoch=tts._epoch)

    assert heard == pytest.approx(0.5, abs=QUIET_WINDOW_S)
    assert tts.quiet_tail() == heard


def test_kokoro_cut_short_leaves_the_quiet_of_the_last_one_heard(speakers, monkeypatch):
    tts = KokoroTTS(enabled=True)
    tts._sample_rate = RATE
    tts._initialized = True
    tts._pipe = lambda text, voice=None, speed=None: iter([("g", "p", _her_reply_then(0.4))])
    tts._speak_once("Une phrase.")
    heard = tts.quiet_tail()

    tts._pipe = lambda text, voice=None, speed=None: iter([("g", "p", _her_reply_then(1.5))])
    _cut_short_once_her_sound_starts(monkeypatch, tts)
    tts._speak_once("Une autre.", epoch=tts._epoch)

    assert heard == pytest.approx(0.4, abs=QUIET_WINDOW_S)
    assert tts.quiet_tail() == heard


def test_chatterbox_cut_short_leaves_the_quiet_of_the_last_one_heard(chatterbox_speakers,
                                                                      monkeypatch):
    tts = ChatterboxTTS(enabled=True)
    tts._initialized = True
    tts._model = SimpleNamespace(sr=RATE, generate=lambda text, **kw: _her_reply_then(0.3)[None, :])
    tts._speak_once("Une phrase.")
    heard = tts.quiet_tail()

    tts._model.generate = lambda text, **kw: _her_reply_then(1.5)[None, :]
    _cut_short_once_her_sound_starts(monkeypatch, tts)
    tts._speak_once("Une autre.", epoch=tts._epoch)

    assert heard == pytest.approx(0.3, abs=QUIET_WINDOW_S)
    assert tts.quiet_tail() == heard


@pytest.mark.parametrize("engine", ["piper", "kokoro"])
def test_the_log_gives_no_quiet_for_a_reply_cut_short(speakers, monkeypatch, engine):
    """Interrupted once its sound has started: what it would have played is
    not her last sound, and the log does not give a quiet for it."""
    lines = []
    monkeypatch.setattr(tts_module, "debug_log", lambda message, *a, **k: lines.append(message))
    if engine == "piper":
        tts = PiperTTS(enabled=True)
        audio = (_her_reply_then(1.5) * 32767).astype(np.int16)
        voice = SimpleNamespace(synthesize=lambda text, config: [
            SimpleNamespace(audio_int16_array=audio)])
        tts._voice_for = lambda language: (voice, RATE)
    else:
        tts = KokoroTTS(enabled=True)
        tts._sample_rate = RATE
        tts._initialized = True
        tts._pipe = lambda text, voice=None, speed=None: iter([("g", "p", _her_reply_then(1.5))])
    _cut_short_once_her_sound_starts(monkeypatch, tts)

    tts._speak_once("Une autre.", epoch=tts._epoch)

    assert any("TTS complete" in line and "spoken=False" in line for line in lines), lines
    assert not any("quiet after her last sound" in line for line in lines), lines


def test_the_log_says_how_long_the_quiet_was(speakers, monkeypatch):
    lines = []
    monkeypatch.setattr(tts_module, "debug_log", lambda message, *a, **k: lines.append(message))
    tts = PiperTTS(enabled=True)
    audio = (_her_reply_then(0.5) * 32767).astype(np.int16)
    voice = SimpleNamespace(synthesize=lambda text, config: [
        SimpleNamespace(audio_int16_array=audio)])
    tts._voice_for = lambda language: (voice, RATE)

    tts._speak_once("Une phrase.")

    assert any("quiet after her last sound" in line for line in lines), lines
