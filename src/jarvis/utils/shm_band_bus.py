"""Cross-process FFT-band bus over ``multiprocessing.shared_memory``.

The desktop tray spawns the daemon as a subprocess, so the mic chunks
that arrive at ``jarvis.listening.listener._on_audio`` live in a
different memory space from the orb widget that wants to consume their
FFT bands. Phase 1 worked around this with a synthetic envelope (i.e.
no audio source at all); Phase 2C lets the daemon publish a small
fixed-size band frame to a shared segment, which the desktop reads
non-blocking once per render frame.

Design
------
- One named SHM segment per user. The name embeds ``os.getuid()`` so a
  second daemon (different OS user) cannot stomp the first.
- 16 byte header (magic + version + write counter) + 256-slot ring of
  64-byte frames. Total ≈ 16 KB, well under the 64 KB POSIX page
  default. The ring is dramatically over-sized for the access pattern
  (publisher 60 Hz, reader 60 Hz, slot lifetime ~16 ms × 256 = ~4 s)
  which is the point — the reader is allowed to lag many frames
  without ever seeing a torn write.
- Lock-free SPSC. The publisher atomically increments the header
  counter *after* writing the slot; the reader snapshots the counter,
  reads ``slot[(counter - 1) % N]``, then re-reads the counter to
  verify the slot wasn't lapped during the read. Lap detection
  uses ``counter_after - counter_before < N``.
- All writes are byte-level via ``numpy.frombuffer(buf, dtype=…)[i] = …``
  on a single field at a time. Numpy's assignment of a primitive into a
  pre-typed buffer is a single ``memcpy``; given CPython's GIL and
  POSIX memory ordering, that's enough atomicity for our use case
  (a torn read manifests as one frame of jitter, which the orb's EMA
  smoothing absorbs).

Frame layout (64 bytes, packed)
-------------------------------
::

    offset  size  field
    ------  ----  -----------------------------------------------
     0      8     seq         (uint64, == header counter at write)
     8      8     timestamp_s (float64, time.monotonic at write)
    16      4     rms         (float32, in [0, 1])
    20      4     bass        (float32, in [0, 1])
    24      4     mid         (float32, in [0, 1])
    28      4     high        (float32, in [0, 1])
    32     32     _padding    (reserved, zero)

Header (16 bytes)
-----------------
::

    offset  size  field
    ------  ----  -----------------------------------------------
     0      4     magic       (ASCII "JBND")
     4      4     version     (uint32, currently 1)
     8      8     write_seq   (uint64, monotonic; reader uses
                               this as the ring index source)

Failure modes & fallback
------------------------
- ``SharedMemory`` is not available on the platform (e.g. minimal
  Linux containers without ``/dev/shm``): the writer and reader
  factories return ``None``; callers degrade to the previous
  no-audio behaviour.
- A reader attaches before any writer has run: ``read_latest()``
  returns ``None`` (no published frame yet).
- The writer crashes mid-write: the reader's seq-recheck catches
  the inconsistency and returns ``None`` for that read (next read
  recovers).
- A stale segment from a previous run is left behind: the writer
  detects it by name conflict, unlinks it, and creates fresh.
"""

from __future__ import annotations

import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..debug import debug_log


# ── Constants ───────────────────────────────────────────────────────────

_MAGIC: bytes = b"JBND"
_VERSION: int = 1
_HEADER_BYTES: int = 16
_FRAME_BYTES: int = 64
_RING_SLOTS: int = 256
_TOTAL_BYTES: int = _HEADER_BYTES + _RING_SLOTS * _FRAME_BYTES  # 16 + 16384 = 16400

# Header field offsets within the buffer.
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_WRITE_SEQ = 8

# Frame field offsets within a slot.
_OFF_SLOT_SEQ = 0
_OFF_SLOT_TS = 8
_OFF_SLOT_RMS = 16
_OFF_SLOT_BASS = 20
_OFF_SLOT_MID = 24
_OFF_SLOT_HIGH = 28


def default_segment_name() -> str:
    """Return the per-user SHM segment name.

    ``getuid`` keeps two users on a shared box from clobbering each
    other. On Windows, ``getuid`` is unavailable; fall back to the
    process owner (``USERNAME`` env). The segment is single-instance
    per user: spawning a second daemon under the same user will
    replace the first's segment (writer reclaims on conflict).
    """
    try:
        uid = os.getuid()
        return f"jarvis-orb-bands-{uid}"
    except AttributeError:
        # Windows path — no getuid. Use USERNAME or fall back to PID.
        username = os.environ.get("USERNAME") or os.environ.get("USER") or str(os.getpid())
        return f"jarvis-orb-bands-{username}"


@dataclass(frozen=True)
class BandFrame:
    """One published band reading.

    Mirrors the on-the-wire layout. The desktop adapter turns this
    into a ``BandReading`` so the orb's existing API is unchanged.
    """
    seq: int          # writer's monotonic counter at the moment of the write
    timestamp: float  # time.monotonic() at the moment of the write
    rms: float
    bass: float
    mid: float
    high: float


# ── Internal helpers ────────────────────────────────────────────────────


def _slot_offset(idx: int) -> int:
    """Byte offset of slot ``idx`` (0-indexed) within the SHM buffer."""
    return _HEADER_BYTES + (idx % _RING_SLOTS) * _FRAME_BYTES


def _shared_memory_available() -> bool:
    """Cheap probe for ``multiprocessing.shared_memory`` support.

    The module exists since Python 3.8, but some sandboxed environments
    (alpine containers without /dev/shm, restricted seccomp) reject
    ``shm_open`` at create time. We attempt a tiny probe segment and
    immediately unlink it; failure -> we run in fallback mode.
    """
    try:
        from multiprocessing import shared_memory as _shm  # noqa: F401
    except ImportError:
        return False
    return True


# ── Writer ──────────────────────────────────────────────────────────────


class ShmBandWriter:
    """Owner of the SHM segment. Publishes band frames at the caller's
    cadence.

    Lifecycle:
        with ShmBandWriter() as w:
            w.publish(rms, bass, mid, high)
        # segment closed and unlinked on context exit

    Or manual:
        w = ShmBandWriter()
        w.publish(...)
        ...
        w.close()

    Construction:
        ShmBandWriter() — opens / replaces the per-user default segment.
        ShmBandWriter(name="custom") — for tests, distinct segments.

    If SHM creation fails (no module, permission denied, etc.), the
    writer enters a no-op ``is_active=False`` mode so the daemon
    continues running silently without orb audio publishing.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name or default_segment_name()
        self._shm = None
        self._buf: Optional[np.ndarray] = None
        self._write_seq: int = 0
        self._lock = threading.Lock()
        self._closed: bool = False

        if not _shared_memory_available():
            debug_log(
                "shm_band_bus: multiprocessing.shared_memory unavailable; "
                "orb audio publishing disabled",
                "orb",
            )
            return

        try:
            self._shm = self._open_or_replace(self._name, _TOTAL_BYTES)
        except Exception as exc:
            debug_log(
                f"shm_band_bus: failed to create segment {self._name!r}: {exc}",
                "orb",
            )
            self._shm = None
            return

        self._buf = np.ndarray((_TOTAL_BYTES,), dtype=np.uint8, buffer=self._shm.buf)
        self._init_header()

    @staticmethod
    def _open_or_replace(name: str, size: int):
        """Create the segment, replacing any stale one with the same
        name. Two daemons under the same user is a configuration
        mistake we resolve by letting the newer one win."""
        from multiprocessing import shared_memory as _shm
        try:
            return _shm.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            # Stale segment from a prior crashed daemon. Attach, unlink,
            # then create fresh — this is the only safe recovery.
            try:
                stale = _shm.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except Exception as inner:
                debug_log(
                    f"shm_band_bus: stale segment {name!r} cleanup failed: {inner}",
                    "orb",
                )
            return _shm.SharedMemory(name=name, create=True, size=size)

    def _init_header(self) -> None:
        """Write magic + version + zero write_seq."""
        assert self._buf is not None
        self._buf[_OFF_MAGIC:_OFF_MAGIC + 4] = np.frombuffer(_MAGIC, dtype=np.uint8)
        struct.pack_into("<I", self._buf.data, _OFF_VERSION, _VERSION)
        struct.pack_into("<Q", self._buf.data, _OFF_WRITE_SEQ, 0)

    @property
    def is_active(self) -> bool:
        """``True`` iff the writer holds a live SHM segment."""
        return self._shm is not None and not self._closed

    @property
    def name(self) -> str:
        return self._name

    def publish(self, rms: float, bass: float, mid: float, high: float) -> None:
        """Write one frame at the next ring slot, then bump the header
        counter so the reader can find it.

        Thread-safe: an internal lock serialises concurrent publishers
        (defensive — in practice there's one publisher thread).
        """
        if not self.is_active:
            return
        assert self._buf is not None
        with self._lock:
            seq = self._write_seq + 1
            slot_off = _slot_offset(seq - 1)
            ts = time.monotonic()
            # Pack the whole 64-byte frame into a single bytes object,
            # then memcpy into the slot in one shot. This is the
            # closest Python gets to an atomic update of the slot from
            # the reader's POV.
            frame = struct.pack(
                "<Qdffff32s",
                seq, ts, float(rms), float(bass), float(mid), float(high), b"",
            )
            self._buf[slot_off:slot_off + _FRAME_BYTES] = np.frombuffer(frame, dtype=np.uint8)
            # Header bump LAST: the reader uses this as the
            # "publication committed" signal.
            struct.pack_into("<Q", self._buf.data, _OFF_WRITE_SEQ, seq)
            self._write_seq = seq

    def close(self) -> None:
        """Release the segment. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._shm is None:
            return
        try:
            self._buf = None
            self._shm.close()
            self._shm.unlink()
        except Exception as exc:
            debug_log(f"shm_band_bus: close/unlink failed: {exc}", "orb")
        finally:
            self._shm = None

    def __enter__(self) -> "ShmBandWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the user forgot to call close. The
        # GC pass can race with finaliser ordering, hence the broad
        # except.
        try:
            self.close()
        except Exception:
            pass


# ── Reader ──────────────────────────────────────────────────────────────


class ShmBandReader:
    """Read-only attach to the writer's segment.

    Use ``read_latest()`` from any thread — it never blocks. Returns
    ``None`` when:
      - the segment doesn't exist yet (no writer started),
      - no frame has been published yet (write_seq is 0),
      - the slot was lapped during the read (rare; retry next frame).

    Construction:
        ShmBandReader() — attaches to the per-user default segment.
        ShmBandReader(name=...) — for tests.

    If the attach fails the reader enters ``is_active=False`` and
    ``read_latest()`` always returns ``None``.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name or default_segment_name()
        self._shm = None
        self._buf: Optional[np.ndarray] = None
        self._closed: bool = False

        if not _shared_memory_available():
            return

        try:
            from multiprocessing import shared_memory as _shm
            self._shm = _shm.SharedMemory(name=self._name, create=False)
        except FileNotFoundError:
            # Writer not up yet — perfectly normal at startup.
            self._shm = None
            return
        except Exception as exc:
            debug_log(
                f"shm_band_bus: reader attach to {self._name!r} failed: {exc}",
                "orb",
            )
            self._shm = None
            return

        # Sanity check magic before exposing the buffer; a colliding
        # segment from a different program would corrupt our reads.
        try:
            self._buf = np.ndarray((_TOTAL_BYTES,), dtype=np.uint8, buffer=self._shm.buf)
            magic = bytes(self._buf[_OFF_MAGIC:_OFF_MAGIC + 4])
            if magic != _MAGIC:
                debug_log(
                    f"shm_band_bus: bad magic {magic!r} on {self._name!r}; detaching",
                    "orb",
                )
                self._buf = None
                self._shm.close()
                self._shm = None
        except Exception as exc:
            debug_log(f"shm_band_bus: reader init failed: {exc}", "orb")
            self._buf = None
            if self._shm is not None:
                try:
                    self._shm.close()
                except Exception:
                    pass
                self._shm = None

    @property
    def is_active(self) -> bool:
        return self._buf is not None and not self._closed

    @property
    def name(self) -> str:
        return self._name

    def read_latest(self) -> Optional[BandFrame]:
        """Return the most recently published frame, or ``None``.

        Two paths return ``None``:
        - No publication yet (write_seq is 0).
        - Lap detected (writer wrote >= N frames between our reads of
          the header counter). The orb's EMA absorbs a missed frame.
        """
        if not self.is_active:
            return None
        assert self._buf is not None
        # Snapshot the publication counter before reading the slot.
        seq_before = struct.unpack_from("<Q", self._buf.data, _OFF_WRITE_SEQ)[0]
        if seq_before == 0:
            return None
        slot_off = _slot_offset(seq_before - 1)
        try:
            unpacked = struct.unpack_from("<Qdffff32s", self._buf.data, slot_off)
        except struct.error as exc:
            debug_log(f"shm_band_bus: frame unpack failed: {exc}", "orb")
            return None
        slot_seq, ts, rms, bass, mid, high, _pad = unpacked
        # Re-snapshot the publication counter. If the writer has
        # produced fewer than _RING_SLOTS frames since seq_before,
        # the slot we read is still the one we wanted.
        seq_after = struct.unpack_from("<Q", self._buf.data, _OFF_WRITE_SEQ)[0]
        if seq_after - seq_before >= _RING_SLOTS:
            return None
        # Also verify the slot's own seq matches what the header
        # claimed at read time — catches a torn write where the slot
        # was being overwritten as we read it.
        if slot_seq != seq_before:
            return None
        return BandFrame(
            seq=slot_seq,
            timestamp=ts,
            rms=rms,
            bass=bass,
            mid=mid,
            high=high,
        )

    def close(self) -> None:
        """Detach from the segment. Idempotent. Does NOT unlink —
        only the writer owns that."""
        if self._closed:
            return
        self._closed = True
        if self._shm is None:
            return
        try:
            self._buf = None
            self._shm.close()
        except Exception as exc:
            debug_log(f"shm_band_bus: reader close failed: {exc}", "orb")
        finally:
            self._shm = None

    def __enter__(self) -> "ShmBandReader":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
