"""What she plays is taken out of what the microphone hears.

listening.spec.md, "Acoustic Echo Cancellation"; tts.spec.md, "Audio output".
"""

import queue
import threading
import time
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

# The cancellers this module's tests made, closed after each test.
_made = []


@pytest.fixture(autouse=True)
def _cancellers_closed():
    yield
    while _made:
        _made.pop().close()


def _new(**kwargs) -> EchoCanceller:
    canceller = EchoCanceller(**kwargs)
    _made.append(canceller)
    return canceller


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
    takes out of the microphone whatever ``capture_gain`` says. A frame
    of another length than its rate's 10 ms is refused, as WebRTC refuses
    it (there by ending the process). With a ``gate``, each microphone
    frame waits for it to open, ``entered`` set once one does."""

    def __init__(self, rate, capture_gain=1.0, fail_on=None, gate=None):
        self.rate = rate
        self.reference = []
        self.capture_gain = capture_gain
        self.fail_on = fail_on
        self.gate = gate
        self.entered = threading.Event()

    def _check(self, pcm):
        if len(pcm) != self.rate // 100:
            raise AssertionError(f"a frame of {len(pcm)} samples at {self.rate} Hz")

    def reverse(self, pcm):
        self._check(pcm)
        if self.fail_on == "reverse":
            raise RuntimeError("render side broke")
        self.reference.append(pcm.copy())

    def capture(self, pcm):
        self._check(pcm)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait()
        if self.fail_on == "capture":
            raise RuntimeError("capture side broke")
        return (pcm.astype(np.float32) * self.capture_gain).astype(np.int16)

    def heard_as_her(self) -> np.ndarray:
        if not self.reference:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.reference).astype(np.float32) / 32768


def _canceller_with(processor_kwargs=None, rate=CAPTURE_RATE, clock=None, factory=None):
    made = []

    def make(r):
        if factory is not None:
            factory(r)
        made.append(_RecordingProcessor(r, **(processor_kwargs or {})))
        return made[-1]

    canceller = _new(processor_factory=make, **({"clock": clock} if clock else {}))
    assert canceller.start(rate) is None
    canceller.capture_opened(rate, 1)
    assert canceller.settle()
    return canceller, made


class _Delivered:
    """Where the canceller delivers the microphone: each block with the
    moment it was captured, in the order delivered."""

    def __init__(self):
        self.blocks = []

    def __call__(self, captured_at, block):
        self.blocks.append((captured_at, block))

    @property
    def times(self):
        return [captured_at for captured_at, _ in self.blocks]

    def cleaned(self):
        """For each block delivered, whether nothing of it was left."""
        return [not np.any(block) for _, block in self.blocks]


def _mic(seconds=0.02, rate=CAPTURE_RATE, channels=1, level=0.1):
    return np.full((int(rate * seconds), channels), level, dtype=np.float32)


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


def _stream(canceller, rate, dtype="float32", blocksize=320, **kwargs):
    backend = _FakeOutput()
    stream = open_output_stream(backend, samplerate=rate, channels=1, dtype=dtype,
                                blocksize=blocksize, callback=lambda *a: stream.engine(*a),
                                echo_canceller=canceller, **kwargs)
    return stream


# --- The real canceller -------------------------------------------------------


def _through_the_room(*, his_at: float, his_seconds: float = 1.5):
    """Her voice, played at 24 kHz in the engines' blocks from 0.5 s to
    6.5 s, comes back through a room as loud as he is; he says something
    at *his_at*. Both go through the real canceller, the capture in 20 ms
    blocks, as one reply of hers. Returns what the microphone heard, what
    the canceller delivered, her span, his span and her reply's figure."""
    now = [0.0]
    canceller = _new(clock=lambda: now[0])
    assert canceller.start(CAPTURE_RATE) is None, "the WebRTC canceller should be available"
    canceller.capture_opened(CAPTURE_RATE, 1)

    play_rate, block = 24000, 1024
    her_seconds, total_seconds, her_at = 6.0, 9.0, 0.5
    her_played = _voice(her_seconds, play_rate, pitch=210.0, syllables=3.0) * 0.4
    her_in_air = _voice(her_seconds, CAPTURE_RATE, pitch=210.0, syllables=3.0) * 0.4

    total = int(total_seconds * CAPTURE_RATE)
    span = slice(int(her_at * CAPTURE_RATE), int(her_at * CAPTURE_RATE) + len(her_in_air))
    timeline = np.zeros(total, dtype=np.float32)
    timeline[span] = her_in_air
    mic = _Room(delay_s=0.12, level_db=-31).carry(timeline, total, span)
    his = _voice(his_seconds, CAPTURE_RATE, pitch=110.0, syllables=4.0)
    his *= 10 ** ((-24 - _db(his)) / 20)
    his_span = slice(int(his_at * CAPTURE_RATE), int(his_at * CAPTURE_RATE) + len(his))
    mic[his_span] += his
    mic = mic.astype(np.float32)

    tap = canceller.reference_tap(play_rate)
    blocks = [her_played[i:i + block] for i in range(0, len(her_played), block)]
    next_block, delivered = 0, _Delivered()
    capture_block = CAPTURE_RATE * 20 // 1000
    canceller.reply_started()
    for start in range(0, total - capture_block + 1, capture_block):
        now[0] = start / CAPTURE_RATE
        # PortAudio asks for each of her blocks ahead of its sound.
        while next_block < len(blocks) and her_at + next_block * block / play_rate <= now[0] + 0.02:
            tap(blocks[next_block].reshape(-1, 1))
            next_block += 1
        canceller.hear(now[0], mic[start:start + capture_block].reshape(-1, 1), delivered)
        assert canceller.settle()
    removed = canceller.reply_finished()
    out = np.concatenate([block[:, 0] for _, block in delivered.blocks])
    return mic, out, span, his_span, removed


def test_her_voice_played_through_the_room_is_taken_out_and_his_is_kept():
    """Her voice is taken out of the microphone; what he says on his own
    afterwards keeps its level, and her reply counts as taken out."""
    mic, out, span, his_span, removed = _through_the_room(his_at=7.0)

    settled = slice(span.start + CAPTURE_RATE, span.stop)  # past its first second
    assert _db(mic[settled]) - _db(out[settled]) >= 10, "her voice was not taken out"
    assert abs(_db(out[his_span]) - _db(mic[his_span])) <= 1.5, "his voice did not keep its level"
    assert removed >= echo_module.MIN_REMOVED_DB


def test_him_talking_over_the_end_of_her_reply_leaves_it_taken_out():
    """The field case: his question over her last words. The figure
    is read off her reply's frames, not lowered by the ones he shares."""
    *_, removed = _through_the_room(his_at=5.5)

    assert removed >= echo_module.MIN_REMOVED_DB


# --- The callbacks only hand over ---------------------------------------------


def test_the_callbacks_never_wait_on_the_canceller_s_work():
    """Its work stalled, her stream and the capture still hand their
    blocks over and return; the microphone arrives once it resumes."""
    gate = threading.Event()
    canceller, _ = _canceller_with({"capture_gain": 0.0, "gate": gate})
    tap = canceller.reference_tap(CAPTURE_RATE)
    delivered = _Delivered()

    def callbacks():
        for i in range(10):
            tap(np.full((320, 1), 0.3, dtype=np.float32))
            canceller.hear(float(i), _mic(), delivered)

    handing = threading.Thread(target=callbacks)
    handing.start()
    handing.join(timeout=2.0)
    returned_while_stalled = not handing.is_alive()
    gate.set()
    handing.join()

    assert returned_while_stalled
    assert canceller.settle()
    assert delivered.times == [float(i) for i in range(10)]
    assert all(delivered.cleaned())


def test_the_microphone_is_delivered_in_the_order_heard_with_its_capture_time():
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    delivered = _Delivered()

    for i in range(25):
        canceller.hear(100.0 + i * 0.02, _mic(), delivered)

    assert canceller.settle()
    assert delivered.times == [100.0 + i * 0.02 for i in range(25)]
    assert all(delivered.cleaned())


def test_a_canceller_never_started_delivers_the_microphone_as_it_is_heard():
    canceller = _new(processor_factory=lambda rate: _RecordingProcessor(rate, capture_gain=0.0))
    canceller.capture_opened(CAPTURE_RATE, 1)
    delivered = _Delivered()
    block = _mic()

    canceller.hear(5.0, block, delivered)

    assert delivered.blocks and delivered.blocks[0][0] == 5.0
    np.testing.assert_array_equal(delivered.blocks[0][1], block)


def test_a_canceller_fallen_behind_delivers_what_waits_as_captured_until_it_catches_up(monkeypatch):
    lines = []
    monkeypatch.setattr(echo_module, "debug_log", lambda message, category="debug": lines.append(message))
    now = [0.0]
    gate = threading.Event()
    canceller, made = _canceller_with({"capture_gain": 0.0, "gate": gate}, clock=lambda: now[0])
    delivered = _Delivered()

    canceller.hear(0.0, _mic(), delivered)
    assert made[-1].entered.wait(2.0)
    for i in range(1, 4):
        now[0] = i * 0.02
        canceller.hear(now[0], _mic(), delivered)
    now[0] = echo_module.BEHIND_SECONDS + 1.0  # they waited that long
    gate.set()
    assert canceller.settle()
    canceller.hear(now[0], _mic(), delivered)  # caught up
    assert canceller.settle()

    assert delivered.cleaned() == [True, False, False, False, True]
    assert sum("fell behind" in line for line in lines) == 1


# --- Failing open -------------------------------------------------------------


def test_without_the_package_the_microphone_is_read_as_captured():
    def missing(rate):
        raise ImportError("No module named 'livekit'")

    canceller = _new(processor_factory=missing)
    reason = canceller.start(CAPTURE_RATE)
    canceller.capture_opened(CAPTURE_RATE, 1)
    delivered = _Delivered()
    block = _mic()

    canceller.reference_tap(24000)(np.ones((1024, 1), dtype=np.float32))  # never raises
    canceller.hear(1.0, block, delivered)

    assert reason and "livekit" in reason
    assert canceller.settle()
    np.testing.assert_array_equal(delivered.blocks[0][1], block)
    assert not canceller.active


@pytest.mark.parametrize("side", ["capture", "reverse"])
def test_a_canceller_that_fails_while_running_lets_the_microphone_through_in_order_and_says_so_once(
        capsys, side):
    canceller, _ = _canceller_with({"fail_on": side, "capture_gain": 0.0})
    tap = canceller.reference_tap(CAPTURE_RATE)
    delivered = _Delivered()
    block = _mic()

    for i in range(5):
        tap(np.full((320, 1), 0.3, dtype=np.float32))
        canceller.hear(float(i), block, delivered)

    assert canceller.settle()
    assert delivered.times == [float(i) for i in range(5)]
    np.testing.assert_array_equal(delivered.blocks[-1][1], block)
    assert not canceller.active
    assert capsys.readouterr().out.count("Echo cancellation unavailable") == 1


@pytest.mark.parametrize("rate,channels", [(44100, 1), (16000, 2)])
def test_a_capture_it_cannot_clean_is_read_as_captured_and_said_once(capsys, rate, channels):
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    canceller.capture_opened(rate, channels)
    canceller.capture_opened(rate, channels)  # the same microphone, opened again
    delivered = _Delivered()
    block = _mic(rate=rate, channels=channels)

    canceller.hear(1.0, block, delivered)

    assert canceller.settle()
    np.testing.assert_array_equal(delivered.blocks[0][1], block)
    out = capsys.readouterr().out
    assert out.count("read as captured") == 1 and f"{rate} Hz" in out


def test_a_capture_it_cannot_clean_says_nothing_while_it_is_off(capsys):
    canceller = _new(processor_factory=lambda rate: _RecordingProcessor(rate))

    canceller.capture_opened(44100, 2)

    assert "Echo cancellation" not in capsys.readouterr().out


def test_nothing_is_taken_out_until_it_is_started():
    canceller = _new(processor_factory=lambda rate: _RecordingProcessor(rate, capture_gain=0.0))
    canceller.capture_opened(CAPTURE_RATE, 1)
    delivered = _Delivered()

    canceller.hear(1.0, _mic(), delivered)

    assert canceller.settle()
    assert delivered.cleaned() == [False]


# --- A capture opened at another rate -------------------------------------------


def test_a_capture_reopened_at_48_khz_is_cleaned_at_that_rate_while_she_plays():
    """Started for 16 kHz, the canceller meets a microphone opened at
    48 kHz while her stream plays on: from then on both sides reach the
    canceller in that rate's frames, never the old rate's."""
    canceller, made = _canceller_with({"capture_gain": 0.0})
    tap = canceller.reference_tap(24000)
    handed = [0]
    done = threading.Event()

    def her_stream():
        while not done.is_set():
            tap(np.full((1024, 1), 0.3, dtype=np.float32))
            handed[0] += 1
            time.sleep(0.001)

    def her_stream_hands_over(count):
        deadline = time.monotonic() + 2.0
        target = handed[0] + count
        while handed[0] < target and time.monotonic() < deadline:
            time.sleep(0.001)
        return handed[0] >= target

    stream = threading.Thread(target=her_stream)
    stream.start()
    assert her_stream_hands_over(3)
    canceller.capture_opened(48000, 1)
    delivered = _Delivered()
    for i in range(20):
        canceller.hear(float(i), _mic(rate=48000), delivered)
    assert her_stream_hands_over(3)
    done.set()
    stream.join()

    assert canceller.settle()
    assert canceller.active
    assert [p.rate for p in made] == [16000, 48000]
    assert len(made[-1].heard_as_her()) > 0
    assert all(delivered.cleaned()) and len(delivered.blocks) == 20


def test_a_canceller_it_cannot_make_at_the_capture_s_rate_lets_the_microphone_through(capsys):
    def refuses_48_khz(rate):
        if rate == 48000:
            raise RuntimeError("no canceller at 48 kHz")

    canceller, _ = _canceller_with({"capture_gain": 0.0}, factory=refuses_48_khz)
    canceller.capture_opened(48000, 1)
    delivered = _Delivered()
    block = _mic(rate=48000)

    canceller.hear(1.0, block, delivered)

    assert canceller.settle()
    np.testing.assert_array_equal(delivered.blocks[0][1], block)
    assert not canceller.active
    assert capsys.readouterr().out.count("Echo cancellation unavailable") == 1


# --- What she plays is the reference --------------------------------------------


def test_what_a_stream_plays_is_the_canceller_reference():
    canceller, made = _canceller_with()
    stream = _stream(canceller, CAPTURE_RATE, dtype="int16")
    rng = np.random.default_rng(1)
    blocks = [(rng.uniform(-.5, .5, 320) * 32767).astype(np.int16) for _ in range(5)]

    played = _play(stream, blocks, "int16")

    assert canceller.settle()
    np.testing.assert_array_equal(
        (made[-1].heard_as_her() * 32768).astype(np.int16), np.concatenate(played))


@pytest.mark.parametrize("play_rate,capture_rate", [(22050, 16000), (24000, 48000)])
def test_a_stream_at_another_rate_reaches_the_canceller_at_the_capture_rate(play_rate, capture_rate):
    canceller, made = _canceller_with(rate=capture_rate)
    stream = _stream(canceller, play_rate, blocksize=1024)
    tone = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(play_rate) / play_rate)).astype(np.float32)

    _play(stream, [tone[i:i + 1024] for i in range(0, play_rate, 1024)], "float32")

    assert canceller.settle()
    her = made[-1].heard_as_her()
    assert abs(len(her) - capture_rate) <= capture_rate // 100, "one second played is one second heard"
    spectrum = np.abs(np.fft.rfft(her * np.hanning(len(her))))
    assert abs(np.argmax(spectrum) * capture_rate / len(her) - 1000) < 10, "a 1 kHz tone stays 1 kHz"


def test_a_long_reply_at_three_times_the_capture_rate_keeps_its_length():
    """Thirty seconds played at 48 kHz in 1024-sample blocks are thirty
    seconds heard at 16 kHz, give or take one frame: no block gains a
    sample at the seam with the next."""
    canceller, made = _canceller_with()
    rate, seconds = 48000, 30
    stream = _stream(canceller, rate, blocksize=1024)
    sound = np.full(rate * seconds, 0.2, dtype=np.float32)

    _play(stream, [sound[i:i + 1024] for i in range(0, len(sound), 1024)], "float32")

    assert canceller.settle(timeout=30.0)
    assert abs(len(made[-1].heard_as_her()) - CAPTURE_RATE * seconds) <= CAPTURE_RATE // 100


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
    assert canceller.settle()
    after_stop = len(made[-1].heard_as_her())

    ending["raise"] = sounddevice.CallbackAbort
    with pytest.raises(sounddevice.CallbackAbort):
        stream.callback(np.zeros((320, 1), dtype=np.float32), 320, None, None)
    assert canceller.settle()

    assert after_stop == 320, "the block a stream ends on is played, so it is her sound"
    assert len(made[-1].heard_as_her()) == 320, "an aborted block is never played"


def test_her_voice_plays_on_when_the_canceller_fails():
    canceller, _ = _canceller_with({"fail_on": "reverse"})
    stream = _stream(canceller, CAPTURE_RATE)
    blocks = [np.full(320, 0.2, dtype=np.float32)] * 3

    played = _play(stream, blocks, "float32")

    np.testing.assert_array_equal(np.concatenate(played), np.concatenate(blocks))
    assert canceller.settle()
    assert not canceller.active


def test_the_thinking_tune_is_taken_out_but_is_not_her_reply(monkeypatch):
    """The tune plays while she thinks: it is taken out of the microphone
    like her voice, and what was taken out of it says nothing of her
    reply."""
    import sys
    import types

    import jarvis.utils.audio_stream as audio_stream_module
    from jarvis.output.tune_player import TunePlayer

    canceller, made = _canceller_with({"capture_gain": 0.0})
    monkeypatch.setattr(audio_stream_module, "ECHO_CANCELLER", canceller)
    opened = threading.Event()
    streams = []

    def output_stream(**kwargs):
        streams.append(SimpleNamespace(start=lambda: None, stop=lambda: None, abort=lambda: None,
                                       close=lambda: None, closed=False, **kwargs))
        opened.set()
        return streams[-1]

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = output_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    tune = TunePlayer(enabled=True)
    canceller.reply_started()
    tune.start_tune()
    try:
        assert opened.wait(5.0)
        stream = streams[-1]
        for i in range(3):
            outdata = np.zeros((stream.blocksize, 1), dtype=np.int16)
            stream.callback(outdata, stream.blocksize, None, None)
            canceller.hear(float(i), _mic(), _Delivered())
    finally:
        tune.stop_tune()

    assert canceller.settle()
    assert len(made[-1].heard_as_her()) > 0, "the tune is in the canceller's reference"
    assert canceller.reply_finished() is None


def test_a_stream_without_a_callback_is_opened_as_asked():
    backend = _FakeOutput()
    stream = open_output_stream(backend, samplerate=CAPTURE_RATE, channels=1, dtype="float32")
    assert not hasattr(stream, "callback")


# --- How much of her reply was taken out ----------------------------------------


def _gain_removing(db: float) -> float:
    return 10 ** (-db / 20)


def _her_reply(canceller, *, her_voice=True, handed_over=True, capture=(CAPTURE_RATE, 1),
               seconds=0.5):
    """One reply of hers: her stream hands her sound over (unless
    *handed_over* is False) while the microphone hears it. Returns the
    figure taken when it finishes."""
    canceller.reply_started()
    tap = canceller.reference_tap(CAPTURE_RATE, her_voice=her_voice)
    rate, channels = capture
    for i in range(int(seconds * 50)):
        if handed_over:
            tap(np.full((320, 1), 0.3, dtype=np.float32))
        canceller.hear(i * 0.02, _mic(rate=rate, channels=channels), _Delivered())
    assert canceller.settle()
    return canceller.reply_finished()


def test_her_reply_counts_as_taken_out_once_enough_of_it_was():
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(echo_module.MIN_REMOVED_DB + 4)})

    removed = _her_reply(canceller)

    assert removed == pytest.approx(echo_module.MIN_REMOVED_DB + 4, abs=0.2)
    assert canceller.took_her_reply_out()


def test_her_reply_with_too_little_taken_out_does_not_count():
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(echo_module.MIN_REMOVED_DB - 4)})

    _her_reply(canceller)

    assert not canceller.took_her_reply_out()


@pytest.mark.parametrize("how", [
    {"handed_over": False},
    {"her_voice": False},
    {"capture": (44100, 1)},
], ids=["reply-it-was-never-handed", "only-the-tune-played", "capture-it-does-not-clean"])
def test_her_reply_it_did_not_clean_has_no_figure(how):
    canceller, _ = _canceller_with({"capture_gain": 0.0})

    removed = _her_reply(canceller, **how)

    assert removed is None
    assert not canceller.took_her_reply_out()


def test_a_new_reply_has_no_figure_until_it_finishes():
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    _her_reply(canceller)

    canceller.reply_started()

    assert canceller.reply_removed_db is None
    assert not canceller.took_her_reply_out()


def test_a_canceller_turned_off_took_nothing_out():
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    _her_reply(canceller)

    canceller.stop()

    assert not canceller.took_her_reply_out()


# --- How much of her reply it is taking out, while it plays ----------------------


def _her_reply_so_far(canceller, *, blocks, her_voice=True, capture=(CAPTURE_RATE, 1)):
    """The first *blocks* 20 ms blocks of a reply of hers, heard by the
    microphone as her stream hands them over; the reply goes on."""
    canceller.reply_started()
    tap = canceller.reference_tap(CAPTURE_RATE, her_voice=her_voice)
    rate, channels = capture
    for i in range(blocks):
        tap(np.full((320, 1), 0.3, dtype=np.float32))
        canceller.hear(i * 0.02, _mic(rate=rate, channels=channels), _Delivered())
    assert canceller.settle()


def test_her_reply_is_being_taken_out_while_it_plays_once_enough_of_it_was():
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(echo_module.MIN_REMOVED_DB + 4)})

    _her_reply_so_far(canceller, blocks=25)

    assert canceller.taking_her_reply_out()


@pytest.mark.parametrize("db, blocks, how", [
    (echo_module.MIN_REMOVED_DB - 4, 25, {}),
    (40.0, echo_module.MIN_FRAMES_MEASURED // 2 - 1, {}),
    (40.0, 25, {"her_voice": False}),
    (40.0, 25, {"capture": (44100, 1)}),
], ids=["too-little-taken-out", "too-few-frames-yet", "only-the-tune-plays",
        "capture-it-does-not-clean"])
def test_her_reply_is_not_being_taken_out_while_it_plays(db, blocks, how):
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(db)})

    _her_reply_so_far(canceller, blocks=blocks, **how)

    assert not canceller.taking_her_reply_out()


@pytest.mark.parametrize("handed_over", [True, False], ids=["taken-out", "not-handed-over"])
def test_once_her_reply_has_finished_it_is_being_taken_out_as_its_figure_says(handed_over):
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(40.0)})

    _her_reply(canceller, handed_over=handed_over)

    assert canceller.taking_her_reply_out() is canceller.took_her_reply_out() is handed_over


def test_a_canceller_turned_off_is_taking_nothing_out():
    canceller, _ = _canceller_with({"capture_gain": _gain_removing(40.0)})
    _her_reply_so_far(canceller, blocks=25)

    canceller.stop()

    assert not canceller.taking_her_reply_out()


def test_the_debug_log_gives_what_was_taken_out_of_her_reply_when_it_finishes(monkeypatch):
    lines = []
    monkeypatch.setattr(echo_module, "debug_log", lambda message, category="debug": lines.append(message))
    canceller, _ = _canceller_with({"capture_gain": 0.5})

    canceller.reply_started()
    tap = canceller.reference_tap(CAPTURE_RATE)
    for i in range(50):  # one second of her voice
        tap(np.full((320, 1), 0.3, dtype=np.float32))
        canceller.hear(i * 0.02, _mic(), _Delivered())
    assert canceller.settle()
    said_while_she_played = [line for line in lines if "Echo cancelled" in line]
    removed = canceller.reply_finished()

    reports = [line for line in lines if "Echo cancelled" in line]
    assert not said_while_she_played
    assert removed == pytest.approx(-20 * np.log10(0.5), abs=0.1)
    assert len(reports) == 1 and f"{removed:.1f} dB" in reports[0]


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


def test_the_listener_queues_the_microphone_cleaned_of_her_with_its_capture_time(monkeypatch):
    canceller, _ = _canceller_with({"capture_gain": 0.0})  # all of it was her
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener()

    before = time.time()
    obj._on_audio(_mic(), 320, None, None)
    after = time.time()

    assert canceller.settle()
    captured_at, queued = obj._audio_q.get_nowait()
    assert before <= captured_at <= after
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
    block = _mic(rate=48000, channels=2)
    obj._on_audio(block, 960, None, None)

    assert canceller.settle()
    _, queued = obj._audio_q.get_nowait()
    np.testing.assert_array_equal(queued, block)


@pytest.mark.parametrize("available,expected", [
    (True, "🧹 Echo cancellation"),
    (False, "⚠️  Echo cancellation unavailable"),
])
def test_start_up_says_whether_her_voice_is_taken_out(monkeypatch, capsys, available, expected):
    def factory(rate):
        if not available:
            raise ImportError("No module named 'livekit'")
        return _RecordingProcessor(rate)

    canceller = _new(processor_factory=factory)
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener()

    obj._start_echo_cancellation()

    assert expected in capsys.readouterr().out
    assert canceller.active == available


def test_the_setting_off_turns_a_running_canceller_off_without_a_word(monkeypatch, capsys):
    canceller, _ = _canceller_with({"capture_gain": 0.0})
    monkeypatch.setattr(listener_module, "ECHO_CANCELLER", canceller)
    obj = _listener()
    obj.cfg.voice_echo_cancellation = False

    obj._start_echo_cancellation()
    delivered = _Delivered()
    canceller.hear(1.0, _mic(), delivered)

    assert "Echo cancellation" not in capsys.readouterr().out
    assert not canceller.active
    assert canceller.settle()
    assert delivered.cleaned() == [False]


@pytest.mark.parametrize("written,expected", [("{}", True), ('{"voice_echo_cancellation": false}', False)])
def test_the_setting_is_on_unless_he_turns_it_off(tmp_path, monkeypatch, written, expected):
    from jarvis.config import load_settings

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(written, encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

    assert load_settings().voice_echo_cancellation is expected
