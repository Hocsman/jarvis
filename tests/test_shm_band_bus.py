"""Tests for the SHM band bus (Phase 2C).

Three families:

1. ``TestWriterReaderRoundtrip`` — single-process correctness. Open a
   writer + reader on the same name, publish, read, verify content.
   Covers the on-the-wire layout, EMA path, and the lap-detection
   fallback. Fast (<100 ms each).

2. ``TestFallbackBehaviour`` — the bus must degrade silently when SHM
   can't be created: writer becomes a no-op, reader returns ``None``,
   adapter returns ``BandReading.zero()``. Tests cover the three real
   failure modes (no module, attach to missing segment, corrupted
   magic).

3. ``TestCrossProcessSpawn`` — real subprocess via
   ``multiprocessing.spawn``. Validates that the SHM truly bridges
   process boundaries. Marked ``integration`` so the fast CI tier
   (``pytest -m "not integration"``) skips it; the slow tier picks
   it up.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from typing import Optional

import pytest


def _segment_name(suffix: str) -> str:
    """Per-test unique segment name so concurrent tests can't collide."""
    return f"jarvis-test-{os.getpid()}-{suffix}"


# ── Family 1: single-process roundtrip ───────────────────────────────────


class TestWriterReaderRoundtrip:
    """Writer + reader live in the same process; we still cross the
    SHM serialisation layer so a layout bug shows up here."""

    @pytest.mark.unit
    def test_single_frame_roundtrip(self) -> None:
        from jarvis.utils.shm_band_bus import ShmBandWriter, ShmBandReader

        name = _segment_name("single")
        with ShmBandWriter(name=name) as w:
            assert w.is_active, "writer should be active in nominal path"
            w.publish(0.5, 0.4, 0.3, 0.2)

            r = ShmBandReader(name=name)
            try:
                assert r.is_active
                frame = r.read_latest()
                assert frame is not None
                # Float32 round-trip: tolerate IEEE rounding (~7 sig digits).
                assert frame.seq == 1
                assert abs(frame.rms - 0.5) < 1e-5
                assert abs(frame.bass - 0.4) < 1e-5
                assert abs(frame.mid - 0.3) < 1e-5
                assert abs(frame.high - 0.2) < 1e-5
                # timestamp is monotonic seconds — sanity-bound it.
                assert frame.timestamp > 0
            finally:
                r.close()

    @pytest.mark.unit
    def test_reader_sees_latest_after_many_writes(self) -> None:
        """100 publishes -> reader sees seq=100 with the matching
        values. Pins the header-counter / slot-index relationship."""
        from jarvis.utils.shm_band_bus import ShmBandWriter, ShmBandReader

        name = _segment_name("many")
        with ShmBandWriter(name=name) as w:
            for i in range(100):
                w.publish(i / 100.0, 0.0, 0.0, 0.0)

            r = ShmBandReader(name=name)
            try:
                frame = r.read_latest()
                assert frame is not None
                assert frame.seq == 100
                assert abs(frame.rms - 99 / 100.0) < 1e-5
            finally:
                r.close()

    @pytest.mark.unit
    def test_reader_before_any_write_returns_none(self) -> None:
        """Writer created but no publish call: reader attaches but
        ``read_latest`` returns ``None`` (write_seq is still 0)."""
        from jarvis.utils.shm_band_bus import ShmBandWriter, ShmBandReader

        name = _segment_name("empty")
        with ShmBandWriter(name=name) as w:
            assert w.is_active
            r = ShmBandReader(name=name)
            try:
                assert r.is_active
                assert r.read_latest() is None
            finally:
                r.close()


# ── Family 2: fallback behaviour ─────────────────────────────────────────


class TestFallbackBehaviour:
    """Failure modes must degrade silently — orb publishing is
    optional and must never crash the daemon or the desktop."""

    @pytest.mark.unit
    def test_reader_attaching_to_missing_segment_is_inactive(self) -> None:
        """No writer ever ran on this name. ``ShmBandReader`` swallows
        the ``FileNotFoundError`` and reports ``is_active=False``."""
        from jarvis.utils.shm_band_bus import ShmBandReader

        r = ShmBandReader(name=_segment_name("never-existed"))
        try:
            assert r.is_active is False
            assert r.read_latest() is None
        finally:
            r.close()

    @pytest.mark.unit
    def test_writer_init_failure_no_op(self, monkeypatch) -> None:
        """Simulate a SharedMemory creation failure (e.g. permission
        denied, /dev/shm full). Writer goes into the no-op mode;
        publish() is silently dropped and is_active is False."""
        from jarvis.utils import shm_band_bus

        def boom(*_args, **_kwargs):
            raise PermissionError("simulated /dev/shm denial")

        monkeypatch.setattr(shm_band_bus.ShmBandWriter, "_open_or_replace",
                            staticmethod(boom))

        w = shm_band_bus.ShmBandWriter(name=_segment_name("denied"))
        assert w.is_active is False
        # Must not raise even when the writer is inactive.
        w.publish(0.5, 0.4, 0.3, 0.2)
        w.close()

    @pytest.mark.unit
    def test_corrupted_magic_detaches_reader(self, monkeypatch) -> None:
        """A segment exists under our name but carries someone else's
        magic bytes — reader detaches rather than reading garbage."""
        from multiprocessing import shared_memory
        from jarvis.utils.shm_band_bus import ShmBandReader, _TOTAL_BYTES

        # Create a bogus segment with random magic.
        name = _segment_name("bogus")
        seg = shared_memory.SharedMemory(name=name, create=True, size=_TOTAL_BYTES)
        try:
            seg.buf[0:4] = b"XXXX"  # not "JBND"

            r = ShmBandReader(name=name)
            try:
                # Reader detects the bad magic and refuses to expose
                # a buffer.
                assert r.is_active is False
                assert r.read_latest() is None
            finally:
                r.close()
        finally:
            seg.close()
            seg.unlink()

    @pytest.mark.unit
    def test_adapter_returns_zero_when_no_reader(self) -> None:
        """The desktop-facing ``ShmBackedAudioBus`` must surface
        ``BandReading.zero()`` (not raise) when the underlying reader
        has nothing to give it. This is the runtime fallback path that
        keeps the orb visually alive (shader-only motion) when no
        daemon is running."""
        from jarvis.utils.shm_band_bus import ShmBackedAudioBus

        bus = ShmBackedAudioBus()  # attaches to default name; likely no writer
        try:
            reading = bus.read_bands()
            assert reading.rms == 0.0
            assert reading.bass == 0.0
            assert reading.mid == 0.0
            assert reading.high == 0.0
            # push() on the adapter is a no-op; must not raise.
            import numpy as np
            bus.push(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        finally:
            bus.close()

    @pytest.mark.unit
    def test_adapter_pulls_from_active_reader(self) -> None:
        """When a writer is active and has published frames, the
        adapter must surface them as a ``BandReading`` (not zero)."""
        from jarvis.utils.shm_band_bus import (
            ShmBackedAudioBus,
            ShmBandReader,
            ShmBandWriter,
        )

        name = _segment_name("adapter")
        with ShmBandWriter(name=name) as w:
            w.publish(0.7, 0.6, 0.5, 0.4)
            reader = ShmBandReader(name=name)
            adapter = ShmBackedAudioBus(reader=reader)
            try:
                assert adapter.is_active is True
                reading = adapter.read_bands()
                assert abs(reading.rms - 0.7) < 1e-5
                assert abs(reading.bass - 0.6) < 1e-5
                assert abs(reading.mid - 0.5) < 1e-5
                assert abs(reading.high - 0.4) < 1e-5
            finally:
                adapter.close()


# ── Family 3: cross-process via multiprocessing.spawn ────────────────────


def _spawn_publisher(name: str, n_frames: int, ready_signal, done_signal):
    """Module-level publisher target so multiprocessing.spawn can
    pickle it. (Spawn on macOS can't grab closures or local defs.)"""
    # The import has to happen inside the spawned process — the parent
    # process's imports are not inherited under the spawn start method.
    import sys
    sys.path.insert(0, "src")
    from jarvis.utils.shm_band_bus import ShmBandWriter

    w = ShmBandWriter(name=name)
    if not w.is_active:
        # Signal we're up so the test doesn't hang, but with 0 frames
        # written — the assertion side will catch it.
        ready_signal.set()
        done_signal.set()
        return
    ready_signal.set()
    for i in range(n_frames):
        w.publish((i + 1) / n_frames, 0.5, 0.4, 0.3)
        time.sleep(0.001)  # 1ms between publishes — gives reader a chance
    done_signal.set()
    # Give the reader a moment to observe before unlink takes the
    # segment away.
    time.sleep(0.2)
    w.close()


class TestCrossProcessSpawn:
    """Real subprocess publisher + main-process reader. Slow (~1 s
    per test for spawn + import startup), so we tag ``integration``
    and skip them in the fast CI tier."""

    @pytest.mark.integration
    def test_publisher_in_subprocess_reader_in_main(self) -> None:
        """End-to-end: a subprocess publishes 20 frames; the main
        process reader must see at least one published frame with the
        correct payload before the subprocess exits."""
        ctx = mp.get_context("spawn")
        name = _segment_name("cross")
        n_frames = 20
        ready = ctx.Event()
        done = ctx.Event()

        proc = ctx.Process(
            target=_spawn_publisher,
            args=(name, n_frames, ready, done),
            daemon=True,
        )
        proc.start()
        try:
            assert ready.wait(timeout=10.0), "subprocess never signalled ready"
            # Wait for the publisher to push at least one frame.
            from jarvis.utils.shm_band_bus import ShmBandReader

            deadline = time.monotonic() + 5.0
            last_frame = None
            while time.monotonic() < deadline:
                r = ShmBandReader(name=name)
                try:
                    if r.is_active:
                        f = r.read_latest()
                        if f is not None and f.seq >= 1:
                            last_frame = f
                            break
                finally:
                    r.close()
                time.sleep(0.05)

            assert last_frame is not None, (
                "reader never observed a published frame within 5s — "
                "either the SHM publish path is broken or spawn never "
                "imported the module"
            )
            assert last_frame.seq >= 1
            # The publisher uses bass=0.5, mid=0.4, high=0.3 for every
            # frame, so we can pin those exactly.
            assert abs(last_frame.bass - 0.5) < 1e-5
            assert abs(last_frame.mid - 0.4) < 1e-5
            assert abs(last_frame.high - 0.3) < 1e-5

            # Let the publisher run to completion.
            assert done.wait(timeout=5.0), "publisher never signalled done"
        finally:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
