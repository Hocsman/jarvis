"""Thread-safe audio ring buffer + FFT band analyser for the orb.

The orb widget consumes ``(rms, bass, mid, high)`` once per render
frame (60 Hz). Producers push raw mic chunks (via the listener
observer hook) and TTS output chunks (via the Piper output callback)
into the same bus. A single ``threading.Lock`` serialises writes and
the periodic FFT read. Lock-free would be marginal gain at 60 Hz on
this volume of data.

Buffer geometry
---------------
- 1 second of audio at 22050 Hz = 22050 float32 samples (~88 KB).
- Producers may push at different sample rates: the bus resamples
  via linear interpolation to a single internal rate at write time so
  the FFT band math is consistent. Quality is not a concern here, we
  only need band energy to drive a shader.

FFT bands
---------
- bass = 0..250 Hz       (low rumble + speech fundamentals)
- mid  = 250..2000 Hz    (voice intelligibility band)
- high = 2000+..nyquist  (sibilance, transients)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# Internal sample rate. Matches Piper's typical output (22050 Hz) so
# TTS chunks can be pushed without resampling. Mic input at 16 kHz
# is upsampled by linear interpolation at write time.
INTERNAL_SAMPLE_RATE = 22050

# FFT window. Power of two, ~23 ms at 22050 Hz so the FFT covers a
# full speech phoneme without too much spectral leakage.
FFT_WINDOW = 512

# Band edges in Hz.
BAND_BASS_HZ = 250.0
BAND_MID_HZ = 2000.0

# EMA smoothing coefficient. Higher alpha = more responsive, lower =
# more stable. 0.3 is a reasonable visual-feel compromise.
DEFAULT_EMA_ALPHA = 0.3


@dataclass(frozen=True)
class BandReading:
    rms: float
    bass: float
    mid: float
    high: float

    @classmethod
    def zero(cls) -> "BandReading":
        return cls(rms=0.0, bass=0.0, mid=0.0, high=0.0)


class _RingBuffer:
    """Single-writer-friendly numpy ring buffer.

    Multi-writer safe via the lock provided by the surrounding
    ``AudioBus``. The reader (FFT) takes the same lock briefly to
    snapshot the tail; the bulk of the lock-hold time is the memcpy
    of at most ``FFT_WINDOW`` samples (~4 KB) which is negligible.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0  # next index to write
        self._total_written = 0  # monotonically increasing

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        return self._total_written

    def write(self, samples: np.ndarray) -> None:
        """Append ``samples`` (mono float32) to the ring."""
        n = samples.shape[0]
        if n == 0:
            return
        if n >= self._capacity:
            # Caller pushed a chunk bigger than the whole window; keep
            # only the most recent capacity samples and reset.
            self._buf[:] = samples[-self._capacity:]
            self._write_pos = 0
            self._total_written += n
            return
        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = samples
        else:
            split = self._capacity - self._write_pos
            self._buf[self._write_pos:] = samples[:split]
            self._buf[: n - split] = samples[split:]
        self._write_pos = end % self._capacity
        self._total_written += n

    def tail(self, n: int) -> np.ndarray:
        """Return the most recent ``n`` samples as a contiguous copy.

        Returns silence (zeros) if the ring has not yet received ``n``
        samples since construction.
        """
        n = min(n, self._capacity)
        if self._total_written < n:
            return np.zeros(n, dtype=np.float32)
        start = (self._write_pos - n) % self._capacity
        end = start + n
        if end <= self._capacity:
            return self._buf[start:end].copy()
        first = self._buf[start:].copy()
        second = self._buf[: end - self._capacity]
        return np.concatenate([first, second])


class FFTAnalyser:
    """Compute ``(rms, bass, mid, high)`` from a windowed sample tail.

    Stateless except for the EMA cache so each instance can be reused
    across frames. ``analyse(window)`` returns the *smoothed* reading.
    """

    def __init__(
        self,
        sample_rate: int = INTERNAL_SAMPLE_RATE,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        self.sample_rate = sample_rate
        self.alpha = max(0.0, min(1.0, ema_alpha))
        self._prev = BandReading.zero()
        # Cache the Hanning window so we don't recompute per call.
        self._hann = np.hanning(FFT_WINDOW).astype(np.float32)
        # Cache band-edge bin indices on the rfft output of FFT_WINDOW.
        # rfft returns FFT_WINDOW//2 + 1 bins spaced sample_rate / FFT_WINDOW.
        bin_hz = sample_rate / FFT_WINDOW
        self._bass_lo = 0
        self._bass_hi = max(1, int(BAND_BASS_HZ / bin_hz))
        self._mid_hi = max(self._bass_hi + 1, int(BAND_MID_HZ / bin_hz))
        self._nyquist_bin = FFT_WINDOW // 2 + 1

    def reset(self) -> None:
        self._prev = BandReading.zero()

    def analyse(self, window: np.ndarray) -> BandReading:
        """Return an EMA-smoothed band reading for ``window`` (FFT_WINDOW
        mono float32 samples).

        ``window`` shorter than FFT_WINDOW is zero-padded at the front
        (we want the most recent samples to be the unpadded tail). Any
        non-finite input collapses the reading to zero (defensive: a
        NaN slipping in would otherwise propagate through the EMA and
        keep the orb stuck).
        """
        n = window.shape[0]
        if n == 0 or not np.all(np.isfinite(window)):
            return self._update(BandReading.zero())

        if n < FFT_WINDOW:
            padded = np.zeros(FFT_WINDOW, dtype=np.float32)
            padded[-n:] = window
            window = padded
        else:
            window = window[-FFT_WINDOW:]

        # RMS over the unwindowed signal so the level is true to the
        # incoming amplitude, not to the windowed energy.
        rms_raw = float(np.sqrt(np.mean(window * window)))

        # Spectrum on the Hann-windowed signal.
        windowed = window * self._hann
        spectrum = np.abs(np.fft.rfft(windowed))
        # Normalise so a sine at full scale gives ~1.0 in its bin.
        spectrum = spectrum * (2.0 / FFT_WINDOW)

        # Band amplitudes via cumulative energy then sqrt (energy ->
        # amplitude). Cumulative (sum) rather than mean so a single
        # strong frequency reads as strong even when its band is wide,
        # which matches how the shader uses these uniforms (a per-band
        # intensity in [0,1], not an average power). Clipped at 1.0.
        bass = float(np.sqrt(np.sum(spectrum[self._bass_lo:self._bass_hi] ** 2)))
        mid = float(np.sqrt(np.sum(spectrum[self._bass_hi:self._mid_hi] ** 2)))
        high = float(np.sqrt(np.sum(spectrum[self._mid_hi:self._nyquist_bin] ** 2)))

        raw = BandReading(
            rms=min(1.0, rms_raw),
            bass=min(1.0, bass),
            mid=min(1.0, mid),
            high=min(1.0, high),
        )
        return self._update(raw)

    def _update(self, raw: BandReading) -> BandReading:
        a = self.alpha
        smoothed = BandReading(
            rms=(1 - a) * self._prev.rms + a * raw.rms,
            bass=(1 - a) * self._prev.bass + a * raw.bass,
            mid=(1 - a) * self._prev.mid + a * raw.mid,
            high=(1 - a) * self._prev.high + a * raw.high,
        )
        self._prev = smoothed
        return smoothed


class AudioBus:
    """Façade that producers and the orb widget share.

    Threading model: ``push()`` and ``read_bands()`` take the same
    ``threading.Lock`` for the duration of their numpy copies. Lock
    contention is bounded: at 60 Hz read + 30 ms producer chunks,
    the lock is held for tens of microseconds per call.
    """

    def __init__(
        self,
        sample_rate: int = INTERNAL_SAMPLE_RATE,
        seconds: float = 1.0,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        capacity = max(FFT_WINDOW, int(sample_rate * seconds))
        self._ring = _RingBuffer(capacity)
        self._lock = threading.Lock()
        self._sample_rate = sample_rate
        self._analyser = FFTAnalyser(sample_rate=sample_rate, ema_alpha=ema_alpha)
        # Producers may inadvertently push a 2-D array; we flatten on
        # write since the listener callback can hand us (frames, 1).

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def push(self, chunk: np.ndarray, source_rate: Optional[int] = None) -> None:
        """Push a chunk of mono float32 samples into the ring.

        If ``source_rate`` differs from the bus' internal rate, the
        chunk is linearly resampled to ``self.sample_rate`` before
        writing. None means "trust the caller, sample rates match".
        """
        if chunk is None:
            return
        try:
            arr = np.asarray(chunk, dtype=np.float32)
        except Exception:
            return
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        if arr.size == 0:
            return
        if source_rate is not None and source_rate != self._sample_rate:
            arr = self._resample_linear(arr, source_rate, self._sample_rate)
        with self._lock:
            self._ring.write(arr)

    def read_bands(self) -> BandReading:
        """Snapshot the latest FFT window and return its smoothed
        band reading. Always cheap; called from the GL render thread."""
        with self._lock:
            window = self._ring.tail(FFT_WINDOW)
        return self._analyser.analyse(window)

    def reset(self) -> None:
        """Drop the smoothing state. The ring buffer keeps its samples."""
        self._analyser.reset()

    @staticmethod
    def _resample_linear(
        samples: np.ndarray,
        src_rate: int,
        dst_rate: int,
    ) -> np.ndarray:
        if src_rate == dst_rate or samples.size == 0:
            return samples
        n_in = samples.size
        n_out = max(1, int(n_in * dst_rate / src_rate))
        # Use float64 for the indexing math, then back to float32.
        x_out = np.linspace(0.0, n_in - 1, num=n_out, dtype=np.float64)
        # numpy.interp does linear interpolation efficiently.
        out = np.interp(x_out, np.arange(n_in), samples).astype(np.float32)
        return out
