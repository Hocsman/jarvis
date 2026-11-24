"""What she plays, taken out of what the microphone hears.

WebRTC's echo canceller (AEC3), through the ``livekit`` package's
``AudioProcessingModule``. Every output stream opened through
``open_output_stream`` hands what it plays to ``reference_tap``; the
listener's capture callback passes each microphone block through ``clean``.
Both run on the machine; nothing given to the canceller leaves it.

See ``listening.spec.md``, "Acoustic Echo Cancellation".
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

import numpy as np

from ..debug import debug_log

# The rates WebRTC's audio processing works at natively. A capture at any
# other rate is read as captured.
NATIVE_RATES = (8000, 16000, 32000, 48000)

# Her sound counts as playing for this long after a block of it was handed
# over: the engines' blocks are 43 ms at most, so a gap longer than this is
# a pause in her voice, not the space between two blocks.
PLAYING_FOR_SECONDS = 0.3
# Silence after her sound that closes one report in the debug log.
REPORT_AFTER_SECONDS = 1.0
# Her last block is handed over before it is heard: the output stream's
# latency (a few hundred milliseconds) and one engine block lie between. A
# reply whose last sound was handed over within this before it finished
# playing went through the canceller.
HANDED_OVER_WITHIN_SECONDS = 1.0


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
        consumed = int(np.floor(following))
        self._pending = pending[consumed:]
        self._position = following - consumed
        return out


class EchoCanceller:
    """Takes what she plays out of what the microphone hears.

    Off until ``start``. Fails open: whatever goes wrong, the microphone is
    read as captured and her voice plays on.
    """

    def __init__(
        self,
        processor_factory: Optional[Callable[[int], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self._factory = processor_factory or _WebRtcProcessor
        self._clock = clock
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._processor = None
        self._rate: Optional[int] = None
        self._capture_ok = False
        self._failed: Optional[str] = None
        self._said_unsupported: set = set()
        self._last_reference = float("-inf")
        # The same moment on the capture clock the listener stamps with.
        self._handed_over_at = float("-inf")
        self._heard_energy = 0.0
        self._cleaned_energy = 0.0

    # --- Life cycle -----------------------------------------------------

    def start(self, rate: int) -> Optional[str]:
        """Turn it on for a capture at ``rate``. None once it is on,
        otherwise why it is not."""
        with self._lock:
            self._failed = None
            self._processor = None
            # A capture at a rate it cannot clean is read as captured
            # (``capture_opened``); the processor still needs one to exist.
            self._rate = rate if rate in NATIVE_RATES else 16000
            try:
                self._processor = self._factory(self._rate)
            except Exception as exc:
                self._failed = f"{type(exc).__name__}: {exc}"
                debug_log(f"echo canceller not created: {self._failed}", "voice")
                return self._failed
            debug_log(f"echo canceller on at {self._rate} Hz", "voice")
            return None

    def stop(self) -> None:
        with self._lock:
            self._processor = None
            self._capture_ok = False

    @property
    def active(self) -> bool:
        return self._processor is not None and self._failed is None

    def took_out(self, finished: float) -> bool:
        """Whether a reply of hers that finished playing at *finished* (on
        the capture clock, ``time.time``) was taken out of the microphone:
        the capture is cleaned, and her sound was handed over in the
        ``HANDED_OVER_WITHIN_SECONDS`` before that. A reply played where no
        stream hands it over (Chatterbox) was not."""
        with self._lock:
            return (self.active and self._capture_ok
                    and self._handed_over_at >= finished - HANDED_OVER_WITHIN_SECONDS)

    def capture_opened(self, rate: int, channels: int) -> None:
        """The capture the listener opened. One channel at a rate WebRTC
        works at natively is cleaned; any other is read as captured."""
        with self._lock:
            supported = channels == 1 and rate in NATIVE_RATES
            if supported and self._processor is not None and rate != self._rate:
                try:
                    self._processor = self._factory(rate)
                    self._rate = rate
                except Exception as exc:
                    self._fail_locked(f"{type(exc).__name__}: {exc}")
                    return
            self._capture_ok = supported
            if not supported and (rate, channels) not in self._said_unsupported:
                self._said_unsupported.add((rate, channels))
                debug_log(f"echo canceller: a capture at {rate} Hz in {channels} channel(s) "
                          "is read as captured", "voice")

    # --- Her side -------------------------------------------------------

    def reference_tap(self, sample_rate: int) -> Callable[[np.ndarray], None]:
        """What one output stream hands its blocks to, as they are played.
        Never raises: her voice plays on whatever happens here."""
        state = {"resampler": None, "leftover": np.zeros(0, dtype=np.float32)}

        def tap(block: np.ndarray) -> None:
            if not self.active:
                return
            try:
                rate = self._rate
                resampler = state["resampler"]
                if resampler is None or resampler.to_rate != rate:
                    resampler = state["resampler"] = _Resampler(int(sample_rate), rate)
                    state["leftover"] = np.zeros(0, dtype=np.float32)
                samples = np.concatenate([state["leftover"], resampler(_as_mono_float(block))])
                frame = rate // 100
                whole = len(samples) // frame * frame
                state["leftover"] = samples[whole:]
                if whole:
                    with self._lock:
                        if self._processor is None:
                            return
                        for start in range(0, whole, frame):
                            self._processor.reverse(_to_pcm(samples[start:start + frame]))
                        self._last_reference = self._clock()
                        self._handed_over_at = self._wall_clock()
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")

        return tap

    # --- The microphone -------------------------------------------------

    def clean(self, block: np.ndarray) -> np.ndarray:
        """``block`` as the microphone captured it, without her sound. A
        block it cannot clean is returned as it came."""
        if not (self.active and self._capture_ok):
            return block
        try:
            samples = np.asarray(block, dtype=np.float32).reshape(-1)
            rate = self._rate
            frame = rate // 100
            if len(samples) % frame:
                return block
            out = np.empty_like(samples)
            with self._lock:
                if self._processor is None:
                    return block
                for start in range(0, len(samples), frame):
                    cleaned = self._processor.capture(_to_pcm(samples[start:start + frame]))
                    out[start:start + frame] = cleaned.astype(np.float32) / 32768.0
                last_reference = self._last_reference
            self._account(samples, out, last_reference)
            return out.reshape(np.shape(block))
        except Exception as exc:
            self._fail(f"{type(exc).__name__}: {exc}")
            return block

    def _account(self, heard: np.ndarray, cleaned: np.ndarray, last_reference: float) -> None:
        """How much was taken out while she played, said once she is silent."""
        quiet_for = self._clock() - last_reference
        if quiet_for <= PLAYING_FOR_SECONDS:
            self._heard_energy += float(np.sum(np.square(heard)))
            self._cleaned_energy += float(np.sum(np.square(cleaned)))
        elif quiet_for > REPORT_AFTER_SECONDS and self._heard_energy > 0:
            removed = 10 * np.log10((self._heard_energy + 1e-12) / (self._cleaned_energy + 1e-12))
            debug_log(f"🧹 Echo cancelled: {removed:.1f} dB while she played", "voice")
            self._heard_energy = self._cleaned_energy = 0.0

    # --- Failing open ---------------------------------------------------

    def _fail(self, reason: str) -> None:
        with self._lock:
            self._fail_locked(reason)

    def _fail_locked(self, reason: str) -> None:
        if self._failed is not None:
            return
        self._failed = reason
        self._processor = None
        debug_log(f"echo canceller off after an error: {reason}", "voice")
        print(f"     ⚠️  Echo cancellation unavailable: {reason}", flush=True)


# One per process: the engines' output streams and the listener's capture
# meet here.
ECHO_CANCELLER = EchoCanceller()
