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
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import numpy as np

from ...debug import debug_log

WIDTH = HEIGHT = 512
SLOTS = 24
STALE_SECONDS = 2.0      # a renderer silent this long is gone
_RETRY_SECONDS = 1.0     # how often a reader looks again for a segment that is not there

_MAGIC = b"JFAC"
_VERSION = 1
# magic, version, width, height, slots, last sequence, viewer beat, renderer beat, generation, cut time
_HEADER = struct.Struct("<4sIIII4xQddQd")
# sequence (0 while being written), stamp, generation
_SLOT = struct.Struct("<QdQ8x")
_PIXELS = WIDTH * HEIGHT * 3
_SLOT_SIZE = _SLOT.size + _PIXELS
_SIZE = _HEADER.size + SLOTS * _SLOT_SIZE

_OFF_LAST_SEQ = 24
_OFF_VIEWER = 32
_OFF_RENDERER = 40
_OFF_GENERATION = 48
_OFF_CUT = 56

_NAME: Optional[str] = None   # tests give each test its own segment


def segment_name() -> str:
    """One segment per user of the machine, so two users never share a face."""
    if _NAME:
        return _NAME
    try:
        who = getpass.getuser()
    except Exception:
        who = os.path.expanduser("~")
    return "jarvis_face_" + hashlib.sha1(who.encode("utf-8")).hexdigest()[:12]


def _slot_offset(index: int) -> int:
    return _HEADER.size + index * _SLOT_SIZE


@dataclass
class Frame:
    pixels: np.ndarray   # HEIGHT x WIDTH x 3, RGB, uint8
    stamp: float         # perf_counter time at which it is on screen
    seq: int


class FrameWriter:
    """The renderer's side: creates the segment and publishes frames."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or segment_name()
        try:
            self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=_SIZE)
        except FileExistsError:
            # A renderer that died left its segment behind (POSIX keeps it).
            old = shared_memory.SharedMemory(name=self.name)
            old.close()
            old.unlink()
            self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=_SIZE)
        self._buf = self._shm.buf
        self._buf[:_SIZE] = bytes(_SIZE)
        _HEADER.pack_into(self._buf, 0, _MAGIC, _VERSION, WIDTH, HEIGHT, SLOTS, 0, 0.0, 0.0, 0, float("inf"))
        self._seq = 0
        self._generation = 0

    def publish(self, rgb: np.ndarray, stamp: float, generation: int) -> int:
        self._seq += 1
        off = _slot_offset(self._seq % SLOTS)
        _SLOT.pack_into(self._buf, off, 0, 0.0, 0)
        np.frombuffer(self._buf, np.uint8, _PIXELS, off + _SLOT.size)[:] = np.asarray(rgb, np.uint8).reshape(-1)
        _SLOT.pack_into(self._buf, off, 0, stamp, generation)
        struct.pack_into("<Q", self._buf, off, self._seq)
        struct.pack_into("<Q", self._buf, _OFF_LAST_SEQ, self._seq)
        return self._seq

    def cut(self, at: float) -> int:
        """Frames drawn so far and stamped at or after ``at`` must not be shown.
        Returns the new generation, which the frames drawn from now on carry."""
        self._generation += 1
        struct.pack_into("<d", self._buf, _OFF_CUT, at)
        struct.pack_into("<Q", self._buf, _OFF_GENERATION, self._generation)
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
    """A front end's side: attaches when the segment exists and reads frames."""

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
        except (FileNotFoundError, OSError):
            return False
        if sys.platform != "win32":
            # Python 3.12 would unlink the renderer's segment when this process exits.
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(shm._name, "shared_memory")
            except Exception:
                pass
        if shm.size < _SIZE or bytes(shm.buf[:4]) != _MAGIC or struct.unpack_from("<I", shm.buf, 4)[0] != _VERSION:
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
        renderer = struct.unpack_from("<d", buf, _OFF_RENDERER)[0]
        if now - renderer > STALE_SECONDS:
            return None
        generation = struct.unpack_from("<Q", buf, _OFF_GENERATION)[0]
        cut = struct.unpack_from("<d", buf, _OFF_CUT)[0]
        for _attempt in range(2):
            best = None
            for slot in range(SLOTS):
                seq, stamp, gen = _SLOT.unpack_from(buf, _slot_offset(slot))
                if not seq or stamp > now or (gen < generation and stamp >= cut):
                    continue
                if best is None or (stamp, seq) > (best[1], best[0]):
                    best = (seq, stamp, slot)
            if best is None:
                return None
            seq, stamp, slot = best
            pixels = self._copy_pixels(slot)
            again = _SLOT.unpack_from(buf, _slot_offset(slot))
            if again[0] == seq and again[1] == stamp:
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
