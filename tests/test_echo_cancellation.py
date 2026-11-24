"""What she plays is taken out of what the microphone hears.

listening.spec.md, "Acoustic Echo Cancellation"; tts.spec.md, "Audio output".
"""

import queue
from types import SimpleNamespace

import numpy as np
import pytest
import sounddevice

import jarvis.listening.listener as listener_module
import jarvis.utils.echo_cancellation as echo_module
from jarvis.listening.listener import VoiceListener
from jarvis.listening.room_floor import RoomFloor
from jarvis.utils.audio_stream import open_output_stream
from jarvis.utils.echo_cancellation import EchoCanceller

pytestmark = pytest.mark.unit

CAPTURE_RATE = 16000


def _voice(seconds: float, rate: int, pitch: float, syllables: float) -> np.ndarray:
    """A voice-like sound defined in continuous time, so the same sound can
    be rendered at the rate she plays it and at the rate the microphone
    hears it: harmonics on a wavering pitch and a breath spread over the
    speech band, shaped into syllables. Without the breath, a sound made
    of harmonics alone is cancelled anywhere from 9 to 24 dB depending on
    its pitch; real speech, measured, 25 to 40."""
    t = np.arange(int(seconds * rate)) / rate
    f0 = pitch * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / rate
    harmonics = sum(np.sin(k * phase) / k for k in range(1, 25))
    rng = np.random.default_rng(int(pitch))
    breath = sum(np.sin(2 * np.pi * f * t + p)
                 for f, p in zip(rng.uniform(100, 7000, 150), rng.uniform(0, 2 * np.pi, 150)))
    envelope = np.clip(np.sin(2 * np.pi * syllables * t), 0, None) ** 2
    sound = (harmonics + 0.3 * breath / np.sqrt(75)) * envelope
    return (sound / np.max(np.abs(sound))).astype(np.float32)


def _db(x) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12))


class _Room:
    """The way from her speakers to the microphone: a delay, a few
    reflections, and a level close to his."""

    def __init__(self, delay_s: float, level_db: float):
        rng = np.random.default_rng(3)
        n = int(0.25 * CAPTURE_RATE)
        self.response = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.04 * CAPTURE_RATE)) * 0.15
        self.response[0] = 1.0
        self.delay = int(delay_s * CAPTURE_RATE)
        self.level_db = level_db

    def carry(self, played: np.ndarray, total: int, span: slice) -> np.ndarray:
        heard = np.convolve(played, self.response)[:total]
        heard = np.concatenate([np.zeros(self.delay), heard])[:total]
        return heard * 10 ** ((self.level_db - _db(heard[span])) / 20)


class _RecordingProcessor:
    """Stands in for WebRTC: keeps what it was given as her sound and
    takes out of the microphone whatever ``capture_gain`` says."""

    def __init__(self, rate, capture_gain=1.0, fail_on=None):
        self.rate = rate
        self.reference = []
        self.capture_gain = capture_gain
        self.fail_on = fail_on

    def reverse(self, pcm):
        if self.fail_on == "reverse":
            raise RuntimeError("render side broke")
        self.reference.append(pcm.copy())

    def capture(self, pcm):
        if self.fail_on == "capture":
            raise RuntimeError("capture side broke")
        return (pcm.astype(np.float32) * self.capture_gain).astype(np.int16)

    def heard_as_her(self) -> np.ndarray:
        if not self.reference:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.reference).astype(np.float32) / 32768


def _canceller_with(processor_kwargs=None, rate=CAPTURE_RATE, clock=None, wall_clock=None):
    made = []

    def factory(r):
        made.append(_RecordingProcessor(r, **(processor_kwargs or {})))
        return made[-1]

    clocks = {name: value for name, value in (("clock", clock), ("wall_clock", wall_clock)) if value}
    canceller = EchoCanceller(processor_factory=factory, **clocks)
    assert canceller.start(rate) is None
    canceller.capture_opened(rate, 1)
    return canceller, made


class _FakeOutput:
    """An output backend that keeps the callback it was given, so a test
    can play blocks through it the way PortAudio would."""

    CallbackStop = sounddevice.CallbackStop
    CallbackAbort = sounddevice.CallbackAbort

    def __init__(self):
        self.streams = []

    def OutputStream(self, **kwargs):
        stream = SimpleNamespace(closed=False, **kwargs)
        self.streams.append(stream)
        return stream


def _play(stream, blocks, dtype):
    """Run the stream's callback over ``blocks`` as PortAudio would; what
    the speakers would have received."""
    played = []
    for block in blocks:
        outdata = np.zeros((len(block), 1), dtype=dtype)

        def fill(out, frames, time_info, status, block=block):
            out[:, 0] = block

        stream.engine = fill
        stream.callback(outdata, len(block), None, None)
        played.append(outdata[:, 0].copy())
    return played


# --- The real canceller -------------------------------------------------------


def test_her_voice_played_through_the_room_is_taken_out_and_his_is_kept():
    """Her voice, played at 24 kHz in the engines' blocks, comes back
    through a room as loud as he is; the microphone is cleaned of it, and
    what he says on his own afterwards keeps its level."""
    canceller = EchoCanceller()
    assert canceller.start(CAPTURE_RATE) is None, "the WebRTC canceller should be available"
    canceller.capture_opened(CAPTURE_RATE, 1)

    play_rate, block = 24000, 1024
    her_seconds, total_seconds = 6.0, 9.0
    her_at = 0.5
    her_played = _voice(her_seconds, play_rate, pitch=210.0, syllables=3.0) * 0.4
    her_in_air = _voice(her_seconds, CAPTURE_RATE, pitch=210.0, syllables=3.0) * 0.4

    total = int(total_seconds * CAPTURE_RATE)
    span = slice(int(her_at * CAPTURE_RATE), int(her_at * CAPTURE_RATE) + len(her_in_air))
    timeline = np.zeros(total, dtype=np.float32)
    timeline[span] = her_in_air
    mic = _Room(delay_s=0.12, level_db=-31).carry(timeline, total, span)
    his = _voice(1.5, CAPTURE_RATE, pitch=110.0, syllables=4.0)
    his *= 10 ** ((-24 - _db(his)) / 20)
    his_at = int(7.0 * CAPTURE_RATE)
    mic[his_at:his_at + len(his)] += his
    mic = mic.astype(np.float32)

    tap = canceller.reference_tap(play_rate)
    blocks = [her_played[i:i + block] for i in range(0, len(her_played), block)]
    next_block, cleaned = 0, []
    capture_block = CAPTURE_RATE * 20 // 1000
    for start in range(0, total - capture_block + 1, capture_block):
        now = start / CAPTURE_RATE
        # PortAudio asks for each of her blocks ahead of its sound.
        while next_block < len(blocks) and her_at + next_block * block / play_rate <= now + 0.02:
            tap(blocks[next_block].reshape(-1, 1))
            next_block += 1
        cleaned.append(canceller.clean(mic[start:start + capture_block].reshape(-1, 1))[:, 0])
    out = np.concatenate(cleaned)

    settled = slice(span.start + CAPTURE_RATE, span.stop)  # past its first second
    his_span = slice(his_at, his_at + len(his))
    assert _db(mic[settled]) - _db(out[settled]) >= 10, "her voice was not taken out"
    assert abs(_db(out[his_span]) - _db(mic[his_span])) <= 1.5, "his voice did not keep its level"


# --- Failing open -------------------------------------------------------------


def test_without_the_package_the_microphone_is_read_as_captured():
    def missing(rate):
        raise ImportError("No module named 'livekit'")

    canceller = EchoCanceller(processor_factory=missing)
    reason = canceller.start(CAPTURE_RATE)
    canceller.capture_opened(CAPTURE_RATE, 1)

    assert reason and "livekit" in reason
    block = np.linspace(-.2, .2, 320, dtype=np.float32).reshape(-1, 1)
    np.testing.assert_array_equal(canceller.clean(block), block)
    canceller.reference_tap(24000)(np.ones((1024, 1), dtype=np.float32))  # never raises
    assert not canceller.active


@pytest.mark.parametrize("side", ["capture", "reverse"])
def test_a_canceller_that_fails_while_running_lets_the_microphone_through_and_says_so_once(capsys, side):
    canceller, _ = _canceller_with({"fail_on": side, "capture_gain": 0.0})
    tap = canceller.reference_tap(CAPTURE_RATE)
    block = np.full((320, 1), 0.1, dtype=np.float32)

    for _ in range(3):
        tap(np.full((320, 1), 0.3, dtype=np.float32))
        out = canceller.clean(block)

    np.testing.assert_array_equal(out, block)
    assert not canceller.active
    assert capsys.readouterr().out.count("Echo cancellation unavailable") == 1


@pytest.mark.parametrize("rate,channels", [(44100, 1), (16000, 2)])
def test_a_capture_it_cannot_clean_is_read_as_captured(rate, channels):
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    canceller.capture_opened(rate, channels)
    block = np.full((rate * 20 // 1000, channels), 0.1, dtype=np.float32)

    np.testing.assert_array_equal(canceller.clean(block), block)


def test_nothing_is_taken_out_until_it_is_started():
    canceller = EchoCanceller(processor_factory=lambda rate: _RecordingProcessor(rate, capture_gain=0.0))
    canceller.capture_opened(CAPTURE_RATE, 1)
    block = np.full((320, 1), 0.1, dtype=np.float32)

    np.testing.assert_array_equal(canceller.clean(block), block)


# --- What she plays is the reference --------------------------------------------


def test_what_a_stream_plays_is_the_canceller_reference():
    canceller, made = _canceller_with()
    backend = _FakeOutput()
    stream = open_output_stream(backend, samplerate=CAPTURE_RATE, channels=1, dtype="int16",
                                blocksize=320, callback=lambda *a: stream.engine(*a),
                                echo_canceller=canceller)
    rng = np.random.default_rng(1)
    blocks = [(rng.uniform(-.5, .5, 320) * 32767).astype(np.int16) for _ in range(5)]

    played = _play(stream, blocks, "int16")

    np.testing.assert_array_equal(
        (made[-1].heard_as_her() * 32768).astype(np.int16), np.concatenate(played))


def test_a_stream_at_another_rate_reaches_the_canceller_at_the_capture_rate():
    canceller, made = _canceller_with()
    backend = _FakeOutput()
    rate = 22050
    stream = open_output_stream(backend, samplerate=rate, channels=1, dtype="float32",
                                blocksize=1024, callback=lambda *a: stream.engine(*a),
                                echo_canceller=canceller)
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(rate) / rate)).astype(np.float32)

    _play(stream, [tone[i:i + 1024] for i in range(0, rate, 1024)], "float32")

    her = made[-1].heard_as_her()
    assert abs(len(her) - CAPTURE_RATE) <= CAPTURE_RATE // 100, "one second played is one second heard"
    spectrum = np.abs(np.fft.rfft(her * np.hanning(len(her))))
    assert abs(np.argmax(spectrum) * CAPTURE_RATE / len(her) - 1000) < 10, "a 1 kHz tone stays 1 kHz"


def test_the_last_block_is_handed_over_and_an_aborted_one_is_not():
    canceller, made = _canceller_with()
    backend = _FakeOutput()
    ending = {"raise": sounddevice.CallbackStop}

    def engine(out, frames, time_info, status):
        out[:, 0] = 0.25
        raise ending["raise"]()

    stream = open_output_stream(backend, samplerate=CAPTURE_RATE, channels=1, dtype="float32",
                                blocksize=320, callback=engine, echo_canceller=canceller)
    with pytest.raises(sounddevice.CallbackStop):
        stream.callback(np.zeros((320, 1), dtype=np.float32), 320, None, None)
    after_stop = len(made[-1].heard_as_her())

    ending["raise"] = sounddevice.CallbackAbort
    with pytest.raises(sounddevice.CallbackAbort):
        stream.callback(np.zeros((320, 1), dtype=np.float32), 320, None, None)

    assert after_stop == 320, "the block a stream ends on is played, so it is her sound"
    assert len(made[-1].heard_as_her()) == 320, "an aborted block is never played"


def test_her_voice_plays_on_when_the_canceller_fails():
    canceller, _ = _canceller_with({"fail_on": "reverse"})
    backend = _FakeOutput()
    stream = open_output_stream(backend, samplerate=CAPTURE_RATE, channels=1, dtype="float32",
                                blocksize=320, callback=lambda *a: stream.engine(*a),
                                echo_canceller=canceller)
    blocks = [np.full(320, 0.2, dtype=np.float32)] * 3

    played = _play(stream, blocks, "float32")

    np.testing.assert_array_equal(np.concatenate(played), np.concatenate(blocks))
    assert not canceller.active


def test_a_stream_without_a_callback_is_opened_as_asked():
    backend = _FakeOutput()
    stream = open_output_stream(backend, samplerate=CAPTURE_RATE, channels=1, dtype="float32")
    assert not hasattr(stream, "callback")


# --- The debug log ------------------------------------------------------------


def test_the_debug_log_says_how_much_was_taken_out_once_she_is_silent(monkeypatch):
    now = [0.0]
    lines = []
    monkeypatch.setattr(echo_module, "debug_log", lambda message, category="debug": lines.append(message))
    canceller, _ = _canceller_with({"capture_gain": 0.5}, clock=lambda: now[0])
    tap = canceller.reference_tap(CAPTURE_RATE)
    block = np.full((320, 1), 0.1, dtype=np.float32)

    for _ in range(50):  # one second of her voice
        tap(np.full((320, 1), 0.3, dtype=np.float32))
        canceller.clean(block)
        now[0] += 0.02
    said_while_she_played = [line for line in lines if "Echo cancelled" in line]
    now[0] += 1.2
    canceller.clean(block)

    reports = [line for line in lines if "Echo cancelled" in line]
    assert not said_while_she_played
    assert len(reports) == 1 and "6.0 dB" in reports[0]


# --- Whether it took her reply out ---------------------------------------------


def _handed_her_sound_at(when):
    """A canceller cleaning the capture, which her stream handed one block
    of her sound at *when*, on the wall clock."""
    canceller, _ = _canceller_with(wall_clock=lambda: when)
    canceller.reference_tap(CAPTURE_RATE)(np.full((320, 1), 0.3, dtype=np.float32))
    return canceller


def test_a_reply_handed_over_as_it_played_was_taken_out():
    # Her last block is handed over before it is heard, so before the
    # reply finishes playing.
    canceller = _handed_her_sound_at(100.0)

    assert canceller.took_out(100.3)


def test_a_reply_it_was_never_handed_was_not_taken_out():
    canceller, _ = _canceller_with()

    assert not canceller.took_out(100.3)


def test_a_reply_that_finished_well_after_the_last_sound_handed_over_was_not():
    # The tune handed over, then a reply played where it does not see it.
    canceller = _handed_her_sound_at(100.0)

    assert not canceller.took_out(100.0 + echo_module.HANDED_OVER_WITHIN_SECONDS + 1.0)


def test_a_capture_it_does_not_clean_had_nothing_taken_out():
    canceller = _handed_her_sound_at(100.0)
    canceller.capture_opened(44100, 1)

    assert not canceller.took_out(100.3)


def test_a_canceller_turned_off_took_nothing_out():
    canceller = _handed_her_sound_at(100.0)
    canceller.stop()

    assert not canceller.took_out(100.3)


# --- The listener -------------------------------------------------------------


def _listener(rate=CAPTURE_RATE):
    obj = VoiceListener.__new__(VoiceListener)
    obj.cfg = SimpleNamespace(voice_min_energy=0.0045, voice_debug=False, voice_echo_cancellation=True)
    obj._samplerate = CAPTURE_RATE
    obj._stream_samplerate = rate
    obj._frame_samples = rate * 20 // 1000
    obj._pending_audio = None
    obj._room_floor = RoomFloor(frame_ms=20)
    obj._frame_captured_at = 0.0
    obj.tts = obj._tune_player = None
    obj._vad_error_logged = False
    obj._vad = None
    obj._should_stop = obj._dictation_active = False
    obj._callback_count = 0
    obj._audio_q = queue.Queue(maxsize=4)
    obj._reset_audio_health(now=0)
    return obj


def test_the_listener_queues_the_microphone_cleaned_of_her(monkeypatch):
    canceller, _ = _canceller_with({"capture_gain": 0.0})  # all of it was her
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener()

    obj._on_audio(np.full((320, 1), 0.1, dtype=np.float32), 320, None, None)

    _, queued = obj._audio_q.get_nowait()
    assert not np.any(queued)


def test_the_listener_tells_the_canceller_the_capture_it_opened(monkeypatch):
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener(48000)
    obj._selected_input = lambda devices: {"device": 3}
    monkeypatch.setattr(listener_module.sd, "query_devices", lambda *a, **k: [])
    monkeypatch.setattr(listener_module, "_open_input_stream",
                        lambda rate, frame_ms, kwargs, callback=None: (object(), 48000, 2))

    obj._open_capture(20, announce=False)
    block = np.full((960, 2), 0.1, dtype=np.float32)
    obj._on_audio(block, 960, None, None)

    _, queued = obj._audio_q.get_nowait()
    np.testing.assert_array_equal(queued, block)


@pytest.mark.parametrize("setting,available,expected", [
    (True, True, "🧹 Echo cancellation"),
    (True, False, "⚠️  Echo cancellation unavailable"),
    (False, True, None),
])
def test_start_up_says_whether_her_voice_is_taken_out(monkeypatch, capsys, setting, available, expected):
    def factory(rate):
        if not available:
            raise ImportError("No module named 'livekit'")
        return _RecordingProcessor(rate)

    canceller = EchoCanceller(processor_factory=factory)
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener()
    obj.cfg.voice_echo_cancellation = setting

    obj._start_echo_cancellation()

    out = capsys.readouterr().out
    if expected is None:
        assert "Echo cancellation" not in out
    else:
        assert expected in out
    assert canceller.active == (setting and available)


@pytest.mark.parametrize("written,expected", [("{}", True), ('{"voice_echo_cancellation": false}', False)])
def test_the_setting_is_on_unless_he_turns_it_off(tmp_path, monkeypatch, written, expected):
    from jarvis.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(written, encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

    assert load_settings().voice_echo_cancellation is expected
