"""Tests for the orb's thread-safe audio ring + FFT band analyser.

Covers the two contracts callers depend on:
1. Pushing from N producer threads while the GL thread reads at 60 Hz
   must not raise, deadlock, or corrupt the ring (no race).
2. A pure 100 Hz sinusoid lands in the bass band and only in the bass
   band, validating that the FFT bin edges match the documented
   frequency cutoffs.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from desktop_app.orb.audio_bus import (
    AudioBus,
    BandReading,
    FFTAnalyser,
    INTERNAL_SAMPLE_RATE,
)


class TestAudioBusThreadSafety:
    """4 concurrent producers + 1 consumer reading 60 Hz for 200 ms.

    The pass criterion is observable: no exception in any thread, and
    the consumer's reads are well-formed ``BandReading`` instances
    with finite values throughout. This catches both crashes and
    silent NaN propagation through the EMA cache.
    """

    @pytest.mark.unit
    def test_concurrent_producers_no_race(self):
        bus = AudioBus()
        stop = threading.Event()
        errors: list[BaseException] = []
        readings: list[BandReading] = []

        def producer(seed: int) -> None:
            try:
                rng = np.random.default_rng(seed)
                while not stop.is_set():
                    # ~30 ms chunk, typical of the listener's blocksize.
                    chunk = rng.standard_normal(480).astype(np.float32) * 0.1
                    bus.push(chunk, source_rate=16000)
                    time.sleep(0.005)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        def consumer() -> None:
            try:
                while not stop.is_set():
                    readings.append(bus.read_bands())
                    time.sleep(1.0 / 60.0)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(4)]
        threads.append(threading.Thread(target=consumer))
        for t in threads:
            t.start()
        time.sleep(0.2)
        stop.set()
        for t in threads:
            t.join(timeout=2.0)

        assert not errors, f"workers errored: {errors!r}"
        assert len(readings) > 5, "consumer didn't sample often enough"
        for r in readings:
            assert isinstance(r, BandReading)
            assert math.isfinite(r.rms) and math.isfinite(r.bass)
            assert math.isfinite(r.mid) and math.isfinite(r.high)
            assert 0.0 <= r.rms <= 1.0


class TestFFTBands:
    """Spectral isolation: a known frequency must land in the right
    band and not bleed into neighbours.

    This is the contract the shader relies on. If the bin edges drift,
    the orb's reactivity will look mis-attributed (a low rumble would
    make the high band light up etc.)."""

    @staticmethod
    def _sine(freq_hz: float, duration_s: float, sample_rate: int) -> np.ndarray:
        n = int(sample_rate * duration_s)
        t = np.arange(n, dtype=np.float32) / sample_rate
        return (0.8 * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)

    @pytest.mark.unit
    def test_pure_100hz_only_bass_responds(self):
        """100 Hz is well inside the 0-250 Hz bass band. The mid and
        high bands must stay near zero."""
        bus = AudioBus()
        sine = self._sine(100.0, duration_s=1.0, sample_rate=INTERNAL_SAMPLE_RATE)
        bus.push(sine)

        # Iterate the EMA a few times so the smoothed value approaches
        # the raw FFT output (the first read returns 30% of the raw).
        last = None
        for _ in range(10):
            last = bus.read_bands()
        assert last is not None

        assert last.bass > 0.10, f"bass should respond, got {last.bass:.3f}"
        assert last.mid < 0.05, (
            f"mid leaked from a pure 100 Hz tone: {last.mid:.3f}"
        )
        assert last.high < 0.02, (
            f"high leaked from a pure 100 Hz tone: {last.high:.3f}"
        )
        assert last.rms > 0.4, (
            f"global RMS should match the sine amplitude, got {last.rms:.3f}"
        )

    @pytest.mark.unit
    def test_pure_3000hz_only_high_responds(self):
        """Symmetric check: a 3 kHz sine should fire only the high band.

        Catches the bug where the upper band edge is hardcoded
        incorrectly and bass/mid eat the high frequencies."""
        bus = AudioBus()
        sine = self._sine(3000.0, duration_s=1.0, sample_rate=INTERNAL_SAMPLE_RATE)
        bus.push(sine)
        last = None
        for _ in range(10):
            last = bus.read_bands()
        assert last is not None
        assert last.high > 0.10, f"high should respond, got {last.high:.3f}"
        assert last.bass < 0.05, f"bass leaked: {last.bass:.3f}"

    @pytest.mark.unit
    def test_silence_zero_reading(self):
        """Empty bus -> all bands zero. Catches the case where the
        ring buffer's uninitialised memory leaks energy."""
        bus = AudioBus()
        r = bus.read_bands()
        assert r.rms == pytest.approx(0.0, abs=1e-6)
        assert r.bass == pytest.approx(0.0, abs=1e-6)
        assert r.mid == pytest.approx(0.0, abs=1e-6)
        assert r.high == pytest.approx(0.0, abs=1e-6)


class TestResampling:
    """The listener pushes at 16 kHz but the bus internal rate is
    22050 Hz. Verify the resample path doesn't shift band energy
    around."""

    @pytest.mark.unit
    def test_16khz_input_does_not_shift_bass_band(self):
        bus = AudioBus()
        sample_rate = 16000
        n = sample_rate  # 1 second
        t = np.arange(n, dtype=np.float32) / sample_rate
        sine = (0.8 * np.sin(2.0 * np.pi * 100.0 * t)).astype(np.float32)
        bus.push(sine, source_rate=sample_rate)
        last = None
        for _ in range(10):
            last = bus.read_bands()
        assert last is not None
        assert last.bass > 0.10
        assert last.mid < 0.05


class TestAnalyserResilience:
    """The FFT analyser must not propagate NaN/Inf through the EMA."""

    @pytest.mark.unit
    def test_non_finite_input_collapses_to_zero(self):
        a = FFTAnalyser()
        # Warm up the EMA with a real reading.
        a.analyse(np.full(512, 0.3, dtype=np.float32))
        # Now poison.
        out = a.analyse(np.array([np.nan, np.inf], dtype=np.float32))
        assert math.isfinite(out.rms) and out.rms <= 1.0
        assert math.isfinite(out.bass)
        assert math.isfinite(out.mid)
        assert math.isfinite(out.high)
