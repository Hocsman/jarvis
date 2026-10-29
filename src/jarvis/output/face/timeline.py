"""Her audio laid out by the time it will be heard, and when to feed Ditto.

Pure logic with no GPU and no model: `process.py` runs it against the clock.
Every time here is a `time.perf_counter()` value (`face.spec.md`, "The clock").
"""

from dataclasses import dataclass
from math import ceil, gcd
from typing import List, Optional, Tuple

import numpy as np

RATE = 16000                 # what Ditto's audio encoder reads
FPS = 25
# Ditto reads each chunk as frames of 40 ms: three before the five it animates,
# then two after, and its audio encoder 80 samples more.
CHUNKSIZE = (3, 5, 2)
CHUNK_FRAMES = CHUNKSIZE[1]
CHUNK_SECONDS = CHUNK_FRAMES / FPS
PAST_SECONDS = CHUNKSIZE[0] / FPS
FUTURE_SECONDS = CHUNKSIZE[2] / FPS + 80 / RATE
CHUNK_SAMPLES = sum(CHUNKSIZE) * RATE // FPS + 80   # 6480
DELAY = 0.75                 # audio heard at H is handed over by H - DELAY

# Samples of the previous piece kept to resample the join with the next one; a
# multiple of every input rate's step, so the overlap maps to whole samples.
_JOIN_SECONDS = 0.01


def resample(x: np.ndarray, up: int, down: int) -> np.ndarray:
    """Resample by up/down: zeros inserted, a Kaiser-windowed sinc low-pass,
    every down-th sample kept (scipy's resample_poly with its defaults)."""
    widest = max(up, down)
    half = 10 * widest
    n = np.arange(-half, half + 1)
    taps = np.sinc(n / widest) * np.kaiser(2 * half + 1, 5.0)
    taps *= up / taps.sum()
    spread = np.zeros(len(x) * up, np.float64)
    spread[::up] = x
    filtered = np.convolve(spread, taps)[half:half + len(spread)]
    return filtered[::down][:ceil(len(x) * up / down)].astype(np.float32)


class Timeline:
    """Audio at 16 kHz, addressed by the time it will be heard.

    It holds ``seconds`` of audio from the earliest moment not yet forgotten;
    audio outside that span is dropped. Where pieces overlap, the later wins.
    """

    def __init__(self, seconds: float = 60.0) -> None:
        self._n = int(seconds * RATE)
        self._buf = np.zeros(self._n, np.float32)
        self._floor: Optional[int] = None      # first sample still held
        # (rate, end time, last input samples) of the previous piece, to join the next one smoothly
        self._tail: Optional[Tuple[int, float, np.ndarray]] = None

    @staticmethod
    def _pos(t: float) -> int:
        return int(round(t * RATE))

    def _clip(self, pos: int, count: int) -> Tuple[int, int, int]:
        """The part of [pos, pos + count) the ring holds: (first, last, offset into the request)."""
        floor = self._floor if self._floor is not None else pos
        first = max(pos, floor)
        last = min(pos + count, floor + self._n)
        return first, last, first - pos

    def _write(self, pos: int, samples: np.ndarray) -> None:
        if self._floor is None:
            self._floor = pos
        first, last, skip = self._clip(pos, len(samples))
        while first < last:
            i = first % self._n
            take = min(last - first, self._n - i)
            self._buf[i:i + take] = samples[skip:skip + take]
            first += take
            skip += take

    def _zero(self, first: int, last: int) -> None:
        while first < last:
            i = first % self._n
            take = min(last - first, self._n - i)
            self._buf[i:i + take] = 0.0
            first += take

    def lay(self, heard_at: float, samples: np.ndarray, rate: int) -> None:
        """Lay ``samples`` (mono, at ``rate``) so that the first is heard at ``heard_at``."""
        samples = np.asarray(samples, np.float32).reshape(-1)
        if not len(samples) or rate <= 0:
            return
        end = heard_at + len(samples) / rate
        start = heard_at
        if rate != RATE:
            g = gcd(rate, RATE)
            up, down = RATE // g, rate // g
            keep = max(down, int(_JOIN_SECONDS * rate) // down * down)
            # A piece that continues the previous one is resampled together with the
            # end of it, so the filter sees across the join. Each end of a resampled
            # block is spoilt by the filter's edge, so only the second half of the
            # overlap is written again: there the new block is past its own edge and
            # the previous one was at its own.
            if self._tail is not None and self._tail[0] == rate and abs(self._tail[1] - heard_at) < 0.5 / rate:
                before = self._tail[2][-keep:]
                joined = np.concatenate([before, samples])
                skip = len(before) * up // down // 2
                start = heard_at - len(before) / rate + skip / RATE
            else:
                joined = samples
                skip = 0
            self._tail = (rate, end, joined[-keep:].copy())
            samples = resample(joined, up, down)[skip:]
        else:
            self._tail = None
        self._write(self._pos(start), samples)

    def cut(self, at: float) -> None:
        """Nothing heard at or after ``at`` will be said: it becomes silence."""
        pos = self._pos(at)
        if self._floor is not None:
            first, last, _ = self._clip(pos, self._floor + self._n - pos)
            self._zero(first, last)
        self._tail = None

    def forget(self, before: float) -> None:
        """Drop what is heard before ``before``; the ring reuses that room."""
        pos = self._pos(before)
        if self._floor is None:
            self._floor = pos
            return
        if pos <= self._floor:
            return
        self._zero(self._floor, min(pos, self._floor + self._n))
        self._floor = pos

    def read(self, start: float, count: int) -> np.ndarray:
        out = np.zeros(count, np.float32)
        if self._floor is None:
            return out
        first, last, skip = self._clip(self._pos(start), count)
        while first < last:
            i = first % self._n
            take = min(last - first, self._n - i)
            out[skip:skip + take] = self._buf[i:i + take]
            first += take
            skip += take
        return out


@dataclass
class Step:
    feed: Optional[float] = None   # the chunk to feed now, by the stamp of its first frame
    wait: float = 0.0              # otherwise, how long nothing is due


class Feeder:
    """Decides, at each instant, whether to feed Ditto a chunk.

    It feeds continuously while someone watches and rests otherwise. A chunk
    is fed the moment the end of its window is one delay ahead of the clock,
    never earlier; a feeder that fell behind jumps to the first chunk it can
    still feed on time.
    """

    def __init__(self, timeline: Timeline, delay: float = DELAY) -> None:
        self.timeline = timeline
        self.delay = delay
        self.next_start: Optional[float] = None   # None while resting

    @staticmethod
    def stamps(start: float) -> List[float]:
        return [start + n / FPS for n in range(CHUNK_FRAMES)]

    @staticmethod
    def window(start: float) -> Tuple[float, float]:
        return start - PAST_SECONDS, start + CHUNK_SECONDS + FUTURE_SECONDS

    def due(self, start: float) -> float:
        return start + CHUNK_SECONDS + FUTURE_SECONDS - self.delay

    def earliest(self, now: float) -> float:
        """The first chunk that can still be fed on time."""
        return now + self.delay - CHUNK_SECONDS - FUTURE_SECONDS

    def step(self, now: float, watched: bool) -> Step:
        if not watched:
            self.next_start = None
            return Step(wait=0.05)
        if self.next_start is None:
            return Step(feed=self.earliest(now))
        start = self.next_start
        due = self.due(start)
        if now < due:
            return Step(wait=due - now)
        if now - due > CHUNK_SECONDS:
            start = self.earliest(now)
        return Step(feed=start)

    def fed(self, start: float) -> None:
        self.next_start = start + CHUNK_SECONDS
