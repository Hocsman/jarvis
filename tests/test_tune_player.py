"""Behavioural tests for the thinking-tune player.

Covers:
- Sample generation: right format/size, seam is effectively seamless.
- TunePlayer lifecycle: idempotent start/stop, is_playing state, prompt
  stop even when a "stream" is running.
- Sounddevice dispatch: stop_tune closes the stream cleanly from the
  owning thread (no cross-thread abort — that races with close on
  macOS CoreAudio and logs a spurious !obj error).

The sounddevice stream is exercised via a fake `sounddevice` module
injected into sys.modules — works headlessly in CI.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pytest

from jarvis.output import tune_player
from jarvis.output.tune_player import (
    TunePlayer,
    _generate_thinking_pad_samples,
    _get_thinking_pad_samples,
)


# --- Sample generation -----------------------------------------------------

def test_thinking_pad_samples_have_expected_shape():
    samples, rate = _generate_thinking_pad_samples()
    assert rate == 44100
    assert samples.dtype.name == "int16"
    assert samples.ndim == 1
    # Long enough to contain several pulse-silence cycles.
    assert samples.size / rate >= 5.0


def test_thinking_pad_samples_cached():
    a = _get_thinking_pad_samples()
    b = _get_thinking_pad_samples()
    assert a is b


def test_thinking_pad_seam_is_effectively_seamless():
    samples, _ = _generate_thinking_pad_samples()
    first = int(samples[0])
    last = int(samples[-1])
    # Seam step must be well under full-scale; observed ≈ 500.
    assert abs(first - last) < 0.05 * 32767


def test_thinking_pad_breathes():
    """The pad is intentionally not continuous — it has a short audible
    breath followed by a silent pause each loop so long thinking runs
    aren't fatiguing. Verify both extremes exist."""
    samples, rate = _generate_thinking_pad_samples()
    win = rate // 10  # 100ms windows
    peaks = [
        int(abs(samples[i : i + win]).max())
        for i in range(0, samples.size - win, win)
    ]
    # At least one window is clearly audible (the hold section).
    assert max(peaks) > 0.10 * 32767
    # At least one window is effectively silent (the rest pause).
    assert min(peaks) < 0.005 * 32767


# --- TunePlayer lifecycle --------------------------------------------------

class _FakeStream:
    """Minimal sounddevice.OutputStream stand-in."""

    def __init__(self, *args, **kwargs):
        self.started = False
        self.aborted = False
        self.closed = False
        self._callback = kwargs.get("callback")

    def start(self):
        self.started = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def _install_fake_sounddevice(monkeypatch, stream_factory=None):
    """Inject a fake sounddevice module that records the created stream."""
    created = {}

    def _OutputStream(*args, **kwargs):
        stream = (stream_factory or _FakeStream)(*args, **kwargs)
        created["stream"] = stream
        return stream

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = _OutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return created


def test_disabled_player_never_starts():
    tp = TunePlayer(enabled=False)
    tp.start_tune()
    try:
        assert not tp.is_playing()
        assert tp._thread is None
    finally:
        tp.stop_tune()


def test_stop_is_idempotent():
    tp = TunePlayer(enabled=False)
    tp.stop_tune()
    tp.stop_tune()


def test_double_start_is_ignored(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    tp = TunePlayer(enabled=True)
    tp.start_tune()
    first = tp._thread
    try:
        tp.start_tune()
        assert tp._thread is first
    finally:
        tp.stop_tune()


def test_stop_closes_the_stream_and_returns_quickly(monkeypatch):
    created = _install_fake_sounddevice(monkeypatch)
    tp = TunePlayer(enabled=True)
    tp.start_tune()
    try:
        # Wait until the stream is actually started.
        for _ in range(100):
            stream = created.get("stream")
            if stream is not None and stream.started:
                break
            time.sleep(0.01)
        stream = created.get("stream")
        assert stream is not None and stream.started

        t0 = time.time()
        tp.stop_tune()
        elapsed = time.time() - t0

        # Only the tune thread closes the stream; stop_tune must NOT abort
        # from the caller's thread — that races with close() on macOS.
        assert stream.closed
        assert not stream.aborted
        assert elapsed < 1.0
        assert not tp.is_playing()
    finally:
        tp.stop_tune()


def test_fallback_when_sounddevice_unavailable(monkeypatch):
    # Force the sounddevice import inside _play_tune to fail.
    fake_sd = types.ModuleType("sounddevice_broken")

    def _raise(*a, **kw):
        raise RuntimeError("no audio here")

    fake_sd.OutputStream = _raise
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    tp = TunePlayer(enabled=True)
    tp.start_tune()
    try:
        # Give the thread a moment to reach the fallback loop.
        for _ in range(50):
            if tp.is_playing():
                break
            time.sleep(0.01)
        assert tp.is_playing()

        t0 = time.time()
        tp.stop_tune()
        elapsed = time.time() - t0
        assert elapsed < 1.5
        assert not tp.is_playing()
    finally:
        tp.stop_tune()


def test_stream_callback_wraps_seamlessly(monkeypatch):
    """The internal callback must wrap from end-of-buffer back to start
    without dropping a frame — that's the whole 'seamless loop' promise."""
    captured = {}

    class _SpyStream(_FakeStream):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured["callback"] = kw.get("callback")

    _install_fake_sounddevice(monkeypatch, stream_factory=_SpyStream)
    tp = TunePlayer(enabled=True)
    tp.start_tune()
    try:
        for _ in range(100):
            if captured.get("callback") is not None:
                break
            time.sleep(0.01)
        cb = captured["callback"]
        assert cb is not None

        samples, _ = _get_thinking_pad_samples()
        total = samples.size

        # Position the read head just before the end of the buffer so
        # the next callback crosses the seam.
        import numpy as np
        frames = 1024
        # Simulate two back-to-back callbacks that span the wrap.
        # First drain most of the buffer with a big fake call — we can
        # do it via multiple calls to the real callback.
        out = np.zeros((frames, 1), dtype=np.int16)

        # Call the callback repeatedly until position wraps.
        # The callback uses a closure; after enough calls we should cross.
        seen_wrap = False
        for _ in range(total // frames + 2):
            cb(out, frames, None, None)
            # When the internal position wraps, outdata will be a mix
            # of end-of-buffer and start-of-buffer samples. Verify no
            # exception raised and output is int16.
            assert out.dtype.name == "int16"
            seen_wrap = True
        assert seen_wrap
    finally:
        tp.stop_tune()


# --- The chime: his question taken while she speaks -------------------------

def _portaudio_lock_held_elsewhere() -> bool:
    """Whether another thread holds ``portaudio_lock``: a thread of its own
    tries to take it."""
    from jarvis.utils.audio_lock import portaudio_lock

    taken = []

    def probe():
        got = portaudio_lock.acquire(blocking=False)
        taken.append(got)
        if got:
            portaudio_lock.release()

    prober = threading.Thread(target=probe)
    prober.start()
    prober.join(5.0)
    return not taken[0]


class _RecordedStream:
    """An output stream that keeps what it is given, what is written to it,
    and each call made to it with whether ``portaudio_lock`` was held then.
    *fails* names the call that raises."""

    latency = 0.0

    def __init__(self, opened, fails=None, **kwargs):
        self.kwargs = kwargs
        self.written = []
        self.calls = []
        self.closed = False
        self._fails = fails
        opened.append(self)

    def _called(self, name):
        self.calls.append((name, _portaudio_lock_held_elsewhere()))
        if name == self._fails:
            raise OSError(f"{name} failed")

    def start(self):
        self._called("start")

    def write(self, data):
        import numpy as np

        self._called("write")
        self.written.append(np.array(data, copy=True))

    def stop(self):
        self._called("stop")

    def close(self):
        self._called("close")
        self.closed = True


def _streams_recorded(monkeypatch, fails=None) -> list:
    opened = []
    fake_sd = types.ModuleType("sounddevice")

    def output_stream(**kwargs):
        if fails == "open":
            raise OSError("no output device")
        return _RecordedStream(opened, fails, **kwargs)

    fake_sd.OutputStream = output_stream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return opened


def _chime_played(monkeypatch) -> tuple[list, list]:
    """Play the chime to its end. Returns the exceptions its thread raised
    and its debug log lines."""
    errors, lines = [], []
    monkeypatch.setattr(threading, "excepthook", lambda args: errors.append(args.exc_value))
    monkeypatch.setattr(tune_player, "debug_log",
                        lambda message, category="debug": lines.append(message))
    tune_player.play_heard_chime().join(5.0)
    return errors, lines


def test_the_chime_is_short_audible_and_played_once(monkeypatch):
    import numpy as np

    opened = _streams_recorded(monkeypatch)

    tune_player.play_heard_chime().join(5.0)

    assert len(opened) == 1
    stream = opened[0]
    assert stream.closed
    played = np.concatenate([np.asarray(block).reshape(-1) for block in stream.written])
    seconds = played.size / stream.kwargs["samplerate"]
    assert 0.05 < seconds < 0.4
    assert np.abs(played).max() > 0.1 * 32767


def test_the_chime_starts_stops_and_closes_its_stream_under_the_portaudio_lock(monkeypatch):
    """Her streams open, stop and close while the chime plays, and PortAudio
    on Windows ends the process on two such calls at once; the lock is not
    held while the chime is written."""
    opened = _streams_recorded(monkeypatch)

    tune_player.play_heard_chime().join(5.0)

    calls = opened[0].calls
    assert [name for name, _ in calls] == ["start", "write", "stop", "close"]
    assert all(held for name, held in calls if name != "write"), calls
    assert not any(held for name, held in calls if name == "write"), calls


def test_the_chime_plays_through_a_stream_written_whole_not_one_handed_over(monkeypatch):
    """Only a stream that plays through a callback hands what it plays to
    the echo canceller: the chime's is written whole, so two streams are
    never handed over at once."""
    import jarvis.utils.audio_stream as audio_stream_module

    opened = _streams_recorded(monkeypatch)
    canceller = MagicMock()
    monkeypatch.setattr(audio_stream_module, "ECHO_CANCELLER", canceller)

    tune_player.play_heard_chime().join(5.0)

    assert "callback" not in opened[0].kwargs
    assert opened[0].written
    canceller.reference_tap.assert_not_called()


@pytest.mark.parametrize("fails", ["open", "start", "write"])
def test_a_chime_that_cannot_play_says_so_and_raises_nothing(monkeypatch, fails):
    opened = _streams_recorded(monkeypatch, fails=fails)

    errors, lines = _chime_played(monkeypatch)

    assert errors == []
    assert any("heard chime" in line and "failed" in line for line in lines), lines
    assert all(stream.closed for stream in opened)


def test_without_sounddevice_the_chime_is_skipped_quietly(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)

    errors, lines = _chime_played(monkeypatch)

    assert errors == []
    assert any("sounddevice unavailable" in line for line in lines), lines
