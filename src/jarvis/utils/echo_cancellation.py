"""What she plays, taken out of what the microphone hears.

WebRTC's echo canceller (AEC3), through the ``livekit`` package's
``AudioProcessingModule``. Every output stream opened through
``open_output_stream`` hands each block it plays to ``reference_tap``; the
listener's capture callback hands each microphone block to ``hear``. The
callbacks only put a copy of the block in the canceller's inbox. Its own
thread takes the blocks in the order they came, feeds WebRTC, and delivers
the microphone cleaned with the moment it was captured. Everything runs on
the machine; nothing given to the canceller leaves it.

See ``listening.spec.md``, "Acoustic Echo Cancellation".
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from ..debug import debug_log

# The rates WebRTC's audio processing works at natively. A capture at any
# other rate is read as captured.
NATIVE_RATES = (8000, 16000, 32000, 48000)

# Her voice counts as playing for this long after a block of it was handed
# over: her voice engines' blocks are 46 ms at most, so a longer gap is a
# pause in her voice, not the space between two blocks.
PLAYING_FOR_SECONDS = 0.3
# A microphone block still waiting this long after it was handed over means
# the canceller's thread fell behind: what waits is delivered as captured,
# and her sound is dropped, until nothing waits.
BEHIND_SECONDS = 0.5
# Her reply counts as taken out of the microphone when at least this much
# was taken out of it (``reply_finished``).
MIN_REMOVED_DB = 10.0
# Her reply's figure: the 10 ms frames heard while her voice played, ranked
# by how much was taken out of each, read three quarters of the way up. Her
# pauses, and his voice over part of hers, sit low in that ranking; her
# voice taken out sits at the top.
REMOVAL_PERCENTILE = 75
# Fewer frames than this heard while her voice played make no figure.
MIN_FRAMES_MEASURED = 20
# The most frames one reply keeps for its figure: ten minutes of her voice.
MAX_FRAMES_MEASURED = 60_000

# What the inbox holds: her sound, the microphone, and the canceller's own
# changes, which take their place in the same order.
_HER, _MIC, _USE, _CAPTURE, _STOP, _MARK, _CLOSE = range(7)


def say_unavailable(reason: str) -> None:
    """The console line for a canceller that cannot run."""
    print(f"     ⚠️  Echo cancellation unavailable: {reason}", flush=True)


class _WebRtcProcessor:
    """One WebRTC audio processor, echo cancellation only, at one rate."""

    def __init__(self, rate: int):
        from livekit import rtc  # The import is what tells whether the package is there.

        self._rtc = rtc
        self._rate = rate
        self._apm = rtc.AudioProcessingModule(echo_cancellation=True)

    def _frame(self, pcm: np.ndarray):
        return self._rtc.AudioFrame(pcm.tobytes(), self._rate, 1, len(pcm))

    def reverse(self, pcm: np.ndarray) -> None:
        self._apm.process_reverse_stream(self._frame(pcm))

    def capture(self, pcm: np.ndarray) -> np.ndarray:
        frame = self._frame(pcm)
        self._apm.process_stream(frame)
        return np.frombuffer(bytes(frame.data), dtype=np.int16)


def _removed_db(frames: np.ndarray) -> Optional[float]:
    """How much came out of her reply, in dB, read from what came out of
    each of its *frames*: ``REMOVAL_PERCENTILE`` of them, None when there
    are fewer than ``MIN_FRAMES_MEASURED``."""
    if len(frames) < MIN_FRAMES_MEASURED:
        return None
    return float(np.percentile(frames, REMOVAL_PERCENTILE))


def _to_pcm(samples: np.ndarray) -> np.ndarray:
    """Float samples as 16-bit PCM, on the same scale ``_as_mono_float``
    reads PCM with, so an int16 block makes the round trip unchanged."""
    return np.clip(np.round(samples * 32768.0), -32768, 32767).astype(np.int16)


def _as_mono_float(block: np.ndarray) -> np.ndarray:
    """A block as a stream plays it (int16 or float, one channel or more),
    as one channel of float samples in [-1, 1]."""
    samples = np.asarray(block)
    if samples.dtype.kind in "iu":
        samples = samples.astype(np.float32) / 32768.0
    else:
        samples = samples.astype(np.float32, copy=False)
    if samples.ndim == 2:
        samples = samples[:, 0] if samples.shape[1] == 1 else samples.mean(axis=1)
    return samples.reshape(-1)


class _Resampler:
    """Brings one stream's sound from its rate to the capture rate, block by
    block, without a seam between blocks: a low-pass below the capture's
    Nyquist when the rate comes down, then linear interpolation."""

    TAPS = 33

    def __init__(self, from_rate: int, to_rate: int):
        self.from_rate = from_rate
        self.to_rate = to_rate
        self._step = from_rate / to_rate
        self._position = 0.0  # where the next output sample falls in ``_pending``
        self._pending = np.zeros(0, dtype=np.float32)
        if to_rate < from_rate:
            cutoff = 0.45 * to_rate / from_rate  # in cycles per input sample
            n = np.arange(self.TAPS) - (self.TAPS - 1) / 2
            taps = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(self.TAPS)
            self._taps = (taps / taps.sum()).astype(np.float32)
            self._history = np.zeros(self.TAPS - 1, dtype=np.float32)
        else:
            self._taps = None

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        if self.from_rate == self.to_rate:
            return samples
        if self._taps is not None:
            joined = np.concatenate([self._history, samples])
            self._history = joined[-(self.TAPS - 1):]
            samples = np.convolve(joined, self._taps, mode="valid").astype(np.float32)
        pending = np.concatenate([self._pending, samples])
        last = len(pending) - 1
        if last < self._position:
            self._pending = pending
            return np.zeros(0, dtype=np.float32)
        count = int(np.floor((last - self._position) / self._step)) + 1
        positions = self._position + self._step * np.arange(count)
        out = np.interp(positions, np.arange(len(pending)), pending).astype(np.float32)
        following = self._position + self._step * count
        # The next sample can fall past what is pending: the position then
        # carries the rest over to the next block.
        consumed = min(int(np.floor(following)), len(pending))
        self._pending = pending[consumed:]
        self._position = following - consumed
        return out


class _Stream:
    """One output stream as the canceller's thread keeps it: the rate it
    plays at, whether it is her voice, and its sound not yet a whole frame."""

    def __init__(self, rate: int, her_voice: bool):
        self.rate = rate
        self.her_voice = her_voice
        self.resampler: Optional[_Resampler] = None
        self.leftover = np.zeros(0, dtype=np.float32)


class EchoCanceller:
    """Takes what she plays out of what the microphone hears.

    Off until ``start``. Fails open: whatever goes wrong, the microphone is
    delivered as captured and her voice plays on.
    """

    def __init__(
        self,
        processor_factory: Optional[Callable[[int], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._factory = processor_factory or _WebRtcProcessor
        self._clock = clock
        # The callbacks append, the canceller's thread takes from the left.
        self._inbox: deque = deque()
        self._wake = threading.Event()
        self._thread_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._on = False
        self._failed: Optional[str] = None
        self._capture: Optional[tuple] = None
        self._said_unsupported: set = set()
        # Kept by the canceller's thread alone.
        self._processor = None
        self._rate: Optional[int] = None
        self._capture_ok = False
        self._voice_at = float("-inf")  # her voice last handed over, on ``clock``
        self._behind = False
        # Her reply's frames, written by the canceller's thread and read by
        # the thread that ends her reply.
        self._measure_lock = threading.Lock()
        self._reply_frames: deque = deque(maxlen=MAX_FRAMES_MEASURED)
        self._reply_removed_db: Optional[float] = None
        self._reply_playing = False

    # --- Life cycle -----------------------------------------------------

    def start(self, rate: int) -> Optional[str]:
        """Turn it on for a capture at ``rate``. None once it is on,
        otherwise why it is not."""
        # A capture at a rate it cannot clean is read as captured
        # (``capture_opened``); the processor still needs one to exist.
        native = rate if rate in NATIVE_RATES else 16000
        try:
            processor = self._factory(native)
        except Exception as exc:
            self._on = False
            self._failed = f"{type(exc).__name__}: {exc}"
            debug_log(f"echo canceller not created: {self._failed}", "voice")
            return self._failed
        self._failed = None
        self._ensure_thread()
        self._post((_USE, processor, native))
        if self._capture is not None:
            self._post((_CAPTURE, *self._capture))
        self._on = True
        debug_log(f"echo canceller on at {native} Hz", "voice")
        return None

    def stop(self) -> None:
        """Turn it off. What reaches it from then on is delivered as heard."""
        self._on = False
        if self._thread is not None:
            self._post((_STOP,))

    def close(self, timeout: float = 2.0) -> None:
        """Turn it off and end its thread, once what was handed to it before
        has been delivered; for the end of the process and for tests. What
        reaches it afterwards is delivered as heard, so a block handed over
        while it closes can be delivered before older ones."""
        with self._thread_lock:
            thread, self._thread = self._thread, None
        self._on = False
        if thread is None:
            return
        self._post((_CLOSE,))
        thread.join(timeout)
        # What reached the inbox after the thread had read its last item.
        while True:
            try:
                item = self._inbox.popleft()
            except IndexError:
                break
            if item[0] == _MIC:
                self._deliver(item[4], item[2], item[3])
            elif item[0] == _MARK:
                item[1].set()

    @property
    def active(self) -> bool:
        return self._on

    def capture_opened(self, rate: int, channels: int) -> None:
        """The capture the listener opened. One channel at a rate WebRTC
        works at natively is cleaned; any other is read as captured, which
        the console says once per format while it is on."""
        self._capture = (rate, channels)
        supported = channels == 1 and rate in NATIVE_RATES
        if not supported and self._on and (rate, channels) not in self._said_unsupported:
            self._said_unsupported.add((rate, channels))
            print(f"     ⚠️  Echo cancellation: this microphone ({rate} Hz, {channels} "
                  "channel(s)) is read as captured", flush=True)
            debug_log(f"echo canceller: a capture at {rate} Hz in {channels} channel(s) "
                      "is read as captured", "voice")
        if self._thread is not None:
            self._post((_CAPTURE, rate, channels))

    def settle(self, timeout: float = 2.0) -> bool:
        """Wait until everything handed to it so far has been dealt with.
        False if that took longer than *timeout* seconds."""
        if self._thread is None:
            return True
        done = threading.Event()
        self._post((_MARK, done))
        return done.wait(timeout)

    # --- What the callbacks hand over -----------------------------------

    def reference_tap(self, sample_rate: int, *, her_voice: bool = True) -> Callable[[np.ndarray], None]:
        """What one output stream hands its blocks to, as they are played.
        *her_voice* says whether it plays her voice, which her reply's
        figure is read against; the thinking tune does not. The tap only
        copies the block into the inbox, and never raises."""
        stream = _Stream(int(sample_rate), her_voice)
        inbox, wake, clock = self._inbox, self._wake, self._clock

        def tap(block: np.ndarray) -> None:
            if not self._on:
                return
            try:
                inbox.append((_HER, clock(), stream, np.array(block, copy=True)))
                wake.set()
            except Exception:
                pass  # Her voice plays on; nothing is said from a callback.

        return tap

    def hear(self, captured_at: float, block: np.ndarray,
             deliver: Callable[[float, np.ndarray], None]) -> None:
        """A microphone block, captured at *captured_at*, for ``deliver``.
        Once the canceller has started, its thread delivers every block in
        the order heard, cleaned when it can be; before, it is delivered
        here, as heard. The caller hands over a block it no longer writes."""
        if self._thread is None:
            deliver(captured_at, block)
            return
        self._inbox.append((_MIC, self._clock(), captured_at, block, deliver))
        self._wake.set()

    # --- Her reply ------------------------------------------------------

    def reply_started(self) -> None:
        """Her reply starts: its figure is read from the frames heard from
        now, and the last reply's is dropped."""
        with self._measure_lock:
            self._reply_frames.clear()
            self._reply_removed_db = None
            self._reply_playing = True

    def reply_finished(self) -> Optional[float]:
        """Her reply finished playing: how much was taken out of it, in dB
        (``REMOVAL_PERCENTILE`` of its frames heard while her voice
        played), None when fewer than ``MIN_FRAMES_MEASURED`` were. Kept
        until the next reply starts, and said in the debug log."""
        with self._measure_lock:
            frames = np.fromiter(self._reply_frames, dtype=np.float64)
            self._reply_frames.clear()
            self._reply_playing = False
        removed = _removed_db(frames)
        self._reply_removed_db = removed
        if self._on:
            if removed is None:
                debug_log(f"🧹 Echo cancellation: her reply not measured ({len(frames)} "
                          "frames heard while her voice played)", "voice")
            else:
                debug_log(f"🧹 Echo cancelled: {removed:.1f} dB of her reply "
                          f"({len(frames)} frames)", "voice")
        return removed

    @property
    def reply_removed_db(self) -> Optional[float]:
        """Her last finished reply's figure (``reply_finished``)."""
        return self._reply_removed_db

    def took_her_reply_out(self) -> bool:
        """Whether her last finished reply was taken out of the microphone:
        the canceller is on and at least ``MIN_REMOVED_DB`` came out."""
        removed = self._reply_removed_db
        return self._on and removed is not None and removed >= MIN_REMOVED_DB

    def taking_her_reply_out(self) -> bool:
        """Whether her reply is being taken out of the microphone: the
        canceller is on and, while her reply plays, at least
        ``MIN_REMOVED_DB`` came out of the frames of it heard so far, read
        as its figure is (``REMOVAL_PERCENTILE`` of at least
        ``MIN_FRAMES_MEASURED`` frames); once it has finished, its figure
        decides (``took_her_reply_out``)."""
        if not self._on:
            return False
        with self._measure_lock:
            playing = self._reply_playing
            frames = np.fromiter(self._reply_frames, dtype=np.float64) if playing else None
        if not playing:
            return self.took_her_reply_out()
        removed = _removed_db(frames)
        return removed is not None and removed >= MIN_REMOVED_DB

    # --- The canceller's thread -----------------------------------------

    def _ensure_thread(self) -> None:
        with self._thread_lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="echo-canceller", daemon=True)
                self._thread.start()

    def _post(self, item: tuple) -> None:
        self._inbox.append(item)
        self._wake.set()

    def _run(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            while True:
                try:
                    item = self._inbox.popleft()
                except IndexError:
                    break
                if item[0] == _CLOSE:
                    self._processor = None
                    return
                try:
                    self._take(item)
                except Exception as exc:
                    self._fail(f"{type(exc).__name__}: {exc}")
            self._behind = False

    def _take(self, item: tuple) -> None:
        kind = item[0]
        if kind == _MIC:
            self._take_microphone(*item[1:])
        elif kind == _HER:
            self._take_her_sound(*item[1:])
        elif kind == _USE:
            _, self._processor, self._rate = item
        elif kind == _CAPTURE:
            self._take_capture(*item[1:])
        elif kind == _STOP:
            self._processor = None
        elif kind == _MARK:
            item[1].set()

    def _take_capture(self, rate: int, channels: int) -> None:
        supported = channels == 1 and rate in NATIVE_RATES
        if supported and self._processor is not None and rate != self._rate:
            try:
                self._processor = self._factory(rate)
                self._rate = rate
            except Exception as exc:
                self._capture_ok = False
                self._fail(f"{type(exc).__name__}: {exc}")
                return
        self._capture_ok = supported

    def _take_her_sound(self, stamp: float, stream: _Stream, block: np.ndarray) -> None:
        processor = self._processor
        if processor is None or self._behind:
            return
        rate = self._rate
        if stream.resampler is None or stream.resampler.to_rate != rate:
            stream.resampler = _Resampler(stream.rate, rate)
            stream.leftover = np.zeros(0, dtype=np.float32)
        samples = np.concatenate([stream.leftover, stream.resampler(_as_mono_float(block))])
        frame = rate // 100
        whole = len(samples) // frame * frame
        stream.leftover = samples[whole:]
        try:
            for start in range(0, whole, frame):
                processor.reverse(_to_pcm(samples[start:start + frame]))
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return
        if stream.her_voice:
            self._voice_at = stamp

    def _take_microphone(self, stamp: float, captured_at: float, block: np.ndarray,
                         deliver: Callable[[float, np.ndarray], None]) -> None:
        """Deliver *block*, cleaned when it can be, and once only."""
        try:
            cleaned = self._cleaned(stamp, block)
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            cleaned = None
        if cleaned is None:
            self._deliver(deliver, captured_at, block)
            return
        self._deliver(deliver, captured_at, cleaned.reshape(np.shape(block)))
        if stamp - self._voice_at <= PLAYING_FOR_SECONDS:
            self._measure(np.asarray(block, dtype=np.float32).reshape(-1), cleaned, self._rate // 100)

    def _cleaned(self, stamp: float, block: np.ndarray) -> Optional[np.ndarray]:
        """*block* without her sound, as one channel, or None when it is
        read as captured: no canceller, a capture it does not clean, a
        block that is not whole frames, or its thread fallen behind."""
        processor = self._processor
        if processor is None or not self._capture_ok:
            return None
        late = self._clock() - stamp
        if self._behind or late > BEHIND_SECONDS:
            if not self._behind:
                self._behind = True
                debug_log(f"echo canceller fell behind by {late:.2f}s: the microphone goes "
                          "through as captured until it catches up", "voice")
            return None
        samples = np.asarray(block, dtype=np.float32).reshape(-1)
        frame = self._rate // 100
        if len(samples) % frame:
            return None
        cleaned = np.empty_like(samples)
        for start in range(0, len(samples), frame):
            out = processor.capture(_to_pcm(samples[start:start + frame]))
            cleaned[start:start + frame] = out.astype(np.float32) / 32768.0
        return cleaned

    def _measure(self, heard: np.ndarray, cleaned: np.ndarray, frame: int) -> None:
        """How much was taken out of each 10 ms frame heard while her voice
        played, kept for her reply's figure."""
        count = len(heard) // frame
        heard_energy = np.square(heard[:count * frame]).reshape(count, frame).sum(axis=1)
        cleaned_energy = np.square(cleaned[:count * frame]).reshape(count, frame).sum(axis=1)
        floor = frame * 1e-10  # silence on both sides is nothing taken out
        removed = 10 * np.log10((heard_energy + floor) / (cleaned_energy + floor))
        with self._measure_lock:
            self._reply_frames.extend(removed.tolist())

    @staticmethod
    def _deliver(deliver: Callable, captured_at: float, block: np.ndarray) -> None:
        try:
            deliver(captured_at, block)
        except Exception as exc:
            debug_log(f"echo canceller: a microphone block was not delivered: {exc}", "voice")

    # --- Failing open ---------------------------------------------------

    def _fail(self, reason: str) -> None:
        """Turn it off for an error while it runs, said once. Runs on the
        canceller's thread, never in a callback."""
        self._processor = None
        if self._failed is not None:
            return
        self._failed = reason
        self._on = False
        debug_log(f"echo canceller off after an error: {reason}", "voice")
        say_unavailable(reason)


# One per process: the engines' output streams and the listener's capture
# meet here.
ECHO_CANCELLER = EchoCanceller()
