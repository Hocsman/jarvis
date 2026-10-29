"""The talking face's frames, in shared memory between the renderer and front ends.

The renderer writes each frame with the moment it is to be on screen; a front
end reads the newest frame due and, by reading, tells the renderer someone is
watching. Neither ever waits for the other (`face.spec.md`, "Shared memory").
"""

import getpass
import hashlib
import os
import struct
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List, Optional

import numpy as np

from ...debug import debug_log

WIDTH = HEIGHT = 512
SLOTS = 24
STALE_SECONDS = 2.0      # a renderer that has not beaten for this long is gone
_RETRY_SECONDS = 1.0     # how often a reader looks again for a segment

_MAGIC = b"JFAC"
_VERSION = 1
# magic, version, width, height, slots | last sequence, viewer beat, renderer beat
_HEADER = struct.Struct("<4sIIII4xQdd")
_OFF_LAST_SEQ = struct.calcsize("<4sIIII4x")
_OFF_VIEWER = _OFF_LAST_SEQ + 8
_OFF_RENDERER = _OFF_VIEWER + 8
# sequence (0 while being written, or once a cut has withdrawn it), stamp
_SLOT = struct.Struct("<Qd")
_PIXELS = WIDTH * HEIGHT * 3
_SLOT_SIZE = _SLOT.size + _PIXELS
_SIZE = _HEADER.size + SLOTS * _SLOT_SIZE


def segment_name() -> str:
    """One segment per user of the machine, so two users never share a face."""
    try:
        who = getpass.getuser()
    except Exception:
        who = os.path.expanduser("~")
    return "jarvis_face_" + hashlib.sha1(who.encode("utf-8")).hexdigest()[:12]


def _slot_offset(index: int) -> int:
    return _HEADER.size + index * _SLOT_SIZE


def _owned(shm: shared_memory.SharedMemory) -> bool:
    """Whether this user alone can open the segment. Windows keeps named
    mappings per session; on POSIX they all live in one shared directory."""
    if sys.platform == "win32":
        return True
    st = os.fstat(shm._fd)
    return st.st_uid == os.getuid() and not st.st_mode & 0o077


@dataclass
class Frame:
    pixels: np.ndarray   # HEIGHT x WIDTH x 3, RGB, uint8
    stamp: float         # perf_counter time at which it is on screen
    seq: int


class FrameWriter:
    """The renderer's side: owns the segment and publishes frames."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or segment_name()
        self._shm = self._open()
        self._buf = self._shm.buf
        self._buf[:_SIZE] = bytes(_SIZE)
        _HEADER.pack_into(self._buf, 0, _MAGIC, _VERSION, WIDTH, HEIGHT, SLOTS, 0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._seq = 0
        self._generation = 0
        self._cuts: List[float] = []   # the cut that ended each generation

    def _open(self) -> shared_memory.SharedMemory:
        try:
            return shared_memory.SharedMemory(name=self.name, create=True, size=_SIZE)
        except FileExistsError:
            pass
        # A previous renderer's segment: kept alive by a front end that still
        # holds it (Windows), or left behind by a renderer that died (POSIX).
        shm = shared_memory.SharedMemory(name=self.name)
        if shm.size >= _SIZE and _owned(shm):
            return shm
        shm.close()
        if sys.platform == "win32":
            raise RuntimeError("the face's shared memory is held by another program")
        stale = shared_memory.SharedMemory(name=self.name)
        stale.close()
        stale.unlink()   # refused for another user's segment, which then stays refused
        return shared_memory.SharedMemory(name=self.name, create=True, size=_SIZE)

    def publish(self, rgb: np.ndarray, stamp: float, generation: int) -> bool:
        """Publish a frame drawn in ``generation``; False when a cut since then
        means it shows words she will not say."""
        with self._lock:
            if generation < self._generation and stamp >= min(self._cuts[generation:]):
                return False
            self._seq += 1
            off = _slot_offset(self._seq % SLOTS)
            struct.pack_into("<Q", self._buf, off, 0)
            np.frombuffer(self._buf, np.uint8, _PIXELS, off + _SLOT.size)[:] = np.asarray(rgb, np.uint8).reshape(-1)
            _SLOT.pack_into(self._buf, off, 0, stamp)
            struct.pack_into("<Q", self._buf, off, self._seq)
            struct.pack_into("<Q", self._buf, _OFF_LAST_SEQ, self._seq)
            return True

    def cut(self, at: float) -> int:
        """She stops talking at ``at``: frames stamped from then on are
        withdrawn, published or not. Returns the generation the frames drawn
        from now on carry."""
        with self._lock:
            self._cuts.append(at)
            self._generation += 1
            for slot in range(SLOTS):
                off = _slot_offset(slot)
                seq, stamp = _SLOT.unpack_from(self._buf, off)
                if seq and stamp >= at:
                    struct.pack_into("<Q", self._buf, off, 0)
            return self._generation

    def beat(self, now: float) -> None:
        struct.pack_into("<d", self._buf, _OFF_RENDERER, now)

    def viewer_age(self, now: float) -> float:
        seen = struct.unpack_from("<d", self._buf, _OFF_VIEWER)[0]
        return now - seen if seen > 0 else float("inf")

    def close(self) -> None:
        try:
            self._buf = None
            self._shm.close()
            self._shm.unlink()
        except Exception as e:
            debug_log(f"face: closing the frame segment failed: {e}", "face")


class FrameReader:
    """A front end's side: attaches while a renderer beats, and reads frames."""

    def __init__(self, name: Optional[str] = None) -> None:
        self._name = name
        self._shm = None
        self._buf = None
        self._next_try = 0.0

    def _attach(self) -> bool:
        if self._buf is not None:
            return True
        now = time.monotonic()
        if now < self._next_try:
            return False
        self._next_try = now + _RETRY_SECONDS
        try:
            shm = shared_memory.SharedMemory(name=self._name or segment_name())
        except (OSError, ValueError):
            # ValueError: an empty segment cannot be mapped (POSIX).
            return False
        if sys.platform != "win32":
            # Python 3.12 would unlink the renderer's segment when this process exits.
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        if (shm.size < _SIZE or not _owned(shm) or bytes(shm.buf[:4]) != _MAGIC
                or struct.unpack_from("<I", shm.buf, 4)[0] != _VERSION):
            shm.close()
            return False
        self._shm, self._buf = shm, shm.buf
        return True

    def _copy_pixels(self, slot: int) -> np.ndarray:
        off = _slot_offset(slot) + _SLOT.size
        return np.frombuffer(self._buf, np.uint8, _PIXELS, off).reshape(HEIGHT, WIDTH, 3).copy()

    def read(self, now: Optional[float] = None) -> Optional[Frame]:
        """The frame to show at ``now`` (default: the current time), or None."""
        if not self._attach():
            return None
        if now is None:
            now = time.perf_counter()
        buf = self._buf
        struct.pack_into("<d", buf, _OFF_VIEWER, now)
        if now - struct.unpack_from("<d", buf, _OFF_RENDERER)[0] > STALE_SECONDS:
            # Let go: the next renderer may bring a segment of its own.
            self.close()
            return None
        for _attempt in range(2):
            best = None
            for slot in range(SLOTS):
                seq, stamp = _SLOT.unpack_from(buf, _slot_offset(slot))
                if seq and stamp <= now and (best is None or (stamp, seq) > (best[1], best[0])):
                    best = (seq, stamp, slot)
            if best is None:
                return None
            seq, stamp, slot = best
            pixels = self._copy_pixels(slot)
            if _SLOT.unpack_from(buf, _slot_offset(slot)) == (seq, stamp):
                return Frame(pixels, stamp, seq)
        return None

    def close(self) -> None:
        if self._shm is not None:
            self._buf = None
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None


_reader: Optional[FrameReader] = None


def read_face_frame(now: Optional[float] = None) -> Optional[Frame]:
    """The talking face's frame to show now, or None when there is none."""
    global _reader
    if _reader is None:
        _reader = FrameReader()
    return _reader.read(now)
