"""Her audio laid out by the time it will be heard, and when to feed Ditto.

Pure logic with no GPU and no model: `process.py` runs it against the clock.
Every time here is a `time.perf_counter()` value (`face.spec.md`, "The clock").
"""

from dataclasses import dataclass
from math import gcd
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

RATE = 16000                 # what Ditto's audio encoder reads
FPS = 25
CHUNK_FRAMES = 5
CHUNK_SECONDS = CHUNK_FRAMES / FPS
PAST_SECONDS = 3 / FPS       # audio before a chunk's five frames that Ditto reads
FUTURE_SECONDS = 2 / FPS + 80 / RATE   # and after them
CHUNK_SAMPLES = round((PAST_SECONDS + CHUNK_SECONDS + FUTURE_SECONDS) * RATE)   # 6480
DELAY = 0.75                 # audio heard at H is handed over by H - DELAY
TAIL_SECONDS = 1.0           # feeding goes on this long after her last word, so the mouth closes
LEAD_SECONDS = 0.4           # and starts this long before her first one, so the motion is settled

# Samples of the previous piece kept to resample the join with the next one; a
# multiple of every input rate's step, so the overlap maps to whole samples.
_JOIN_SECONDS = 0.01


class Timeline:
    """Audio at 16 kHz, addressed by the time it will be heard.

    It holds ``seconds`` of audio from the earliest moment not yet forgotten;
    audio outside that span is dropped. Where pieces overlap, the later wins.
    """

    def __init__(self, seconds: float = 60.0) -> None:
        self._n = int(seconds * RATE)
        self._buf = np.zeros(self._n, np.float32)
        self._floor: Optional[int] = None      # first sample still held
        self._spans: List[Tuple[float, float]] = []   # where audio was laid, merged
        # (rate, end time, last input samples) of the previous piece, to join the next one smoothly
        self._tail: Optional[Tuple[int, float, np.ndarray]] = None

    # -- positions --------------------------------------------------------
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

    # -- what the renderer is handed --------------------------------------
    def lay(self, heard_at: float, samples: np.ndarray, rate: int) -> None:
        """Lay ``samples`` (mono, at ``rate``) so that the first is heard at ``heard_at``."""
        samples = np.asarray(samples, np.float32).reshape(-1)
        if not len(samples):
            return
        end = heard_at + len(samples) / rate
        start = heard_at
        if rate != RATE:
            g = gcd(rate, RATE)
            up, down = RATE // g, rate // g
            # A piece that continues the previous one is resampled together with the
            # end of it, so the filter sees across the join: the previous piece's last
            # samples, spoilt by the edge, are written again correctly.
            keep = max(down, int(_JOIN_SECONDS * rate) // down * down)
            # Each end of a resampled block is spoilt by the filter's edge, so only the
            # second half of the overlap is written again: there the new block is past
            # its own edge and the previous one was at its own.
            if self._tail is not None and self._tail[0] == rate and abs(self._tail[1] - heard_at) < 0.5 / rate:
                before = self._tail[2][-keep:]
                joined = np.concatenate([before, samples])
                skip = len(before) * up // down // 2
                start = heard_at - len(before) / rate + skip / RATE
            else:
                joined = samples
                skip = 0
            self._tail = (rate, end, joined[-keep:].copy())
            samples = resample_poly(joined, up, down).astype(np.float32)[skip:]
        else:
            self._tail = None
        self._write(self._pos(start), samples)
        self._add_span(heard_at, end)

    def cut(self, at: float) -> None:
        """Nothing heard at or after ``at`` will be said: it becomes silence."""
        pos = self._pos(at)
        if self._floor is not None:
            first, last, _ = self._clip(pos, self._floor + self._n - pos)
            self._zero(first, last)
        self._spans = [(a, min(b, at)) for a, b in self._spans if a < at]
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
        self._spans = [(max(a, before), b) for a, b in self._spans if b > before]

    # -- what the feeder reads --------------------------------------------
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

    def has_audio(self, start: float, end: float) -> bool:
        return any(a < end and b > start for a, b in self._spans)

    def last_audio_end(self) -> Optional[float]:
        return max((b for _a, b in self._spans), default=None)

    def _add_span(self, start: float, end: float) -> None:
        spans = sorted(self._spans + [(start, end)])
        merged = [spans[0]]
        for a, b in spans[1:]:
            if a <= merged[-1][1] + 1e-6:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self._spans = merged


@dataclass
class Step:
    feed: Optional[float] = None   # the chunk to feed now, by the stamp of its first frame
    wait: float = 0.0              # otherwise, how long nothing is due


class Feeder:
    """Decides, at each instant, whether to feed Ditto a chunk.

    While someone watches it feeds continuously; otherwise only around her
    audio, from LEAD_SECONDS before it to TAIL_SECONDS after it, and rests in
    between. A chunk is fed the moment the end of its window is one delay
    ahead of the clock, never earlier.
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

    def _wanted(self, start: float, watched: bool) -> bool:
        return watched or self.timeline.has_audio(start - TAIL_SECONDS, start + LEAD_SECONDS)

    def step(self, now: float, watched: bool) -> Step:
        if self.next_start is None:
            start = self.earliest(now)
            if self._wanted(start, watched):
                return Step(feed=start)
            return Step(wait=0.05)
        start = self.next_start
        due = self.due(start)
        if now < due:
            return Step(wait=due - now)
        if now - due > CHUNK_SECONDS:
            start = self.earliest(now)
        if not self._wanted(start, watched):
            self.next_start = None
            return Step(wait=0.05)
        return Step(feed=start)

    def fed(self, start: float) -> None:
        self.next_start = start + CHUNK_SECONDS
