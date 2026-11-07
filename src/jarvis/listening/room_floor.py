"""The room's quiet level: the floor the voice proof weighs an utterance
against (listening.spec.md, "And it takes a voice").

The floor is read when her reply starts, and the microphone's last second
then is rarely the room: the thinking tune plays until her first sentence
is handed over, the end of his question can be in it, and a reply from
elsewhere can start the moment another ends. A floor read there sits at
several times the room's level, and his soft answer right after her last
word falls under the bar it sets and is taken for her echo.

So the floor is read from what the microphone hears while nothing of hers
plays, over a stretch of quiet longer than any one utterance, at a low
percentile: his words and the room's knocks are loud and fill only part
of the stretch, so they sit above it.
"""

import threading
from collections import deque
from typing import Optional

# How much quiet the floor is read over. Longer than the longest utterance
# the voice detector keeps whole (`max_utterance_ms`, 12 s by default), so
# even a question that long leaves a fifth of the stretch to the room, twice
# what the percentile below needs; short enough that the floor follows the
# room within seconds of quiet when a fan or a television starts.
ROOM_FLOOR_WINDOW_SECONDS = 15.0

# The level the room stays above for nine-tenths of its quiet. His words, a
# cough, a door sit above it as long as they fill less than nine-tenths of
# the stretch. A steady room varies by a few percent from one frame to the
# next, so this lies within a few percent of its mean and the bar stays
# where the room puts it.
ROOM_FLOOR_PERCENTILE = 10

# Less quiet than this and the percentile stands on a handful of frames:
# there is no floor yet.
ROOM_FLOOR_MIN_SECONDS = 1.0

# A frame the listening loop reads later than this after the microphone
# delivered it is left out. The loop knows what plays when it reads a
# frame, not when the frame was captured, and it falls behind while Whisper
# or the judge holds it: what it then catches up on can be her voice or the
# tune, read once neither plays any more.
ROOM_FRAME_MAX_AGE_SECONDS = 0.5

# The floor before a second of quiet has been heard: the echo detector's
# default (`EchoDetector.track_tts_start`). A quiet room through a desk
# microphone sits below it, so until the room is heard the bar leans
# towards taking a soft voice for her echo rather than room noise for his.
FALLBACK_FLOOR = 0.0045


class RoomFloor:
    """The room's quiet level, kept frame by frame by the listening loop and
    read by the reply thread as her reply starts."""

    def __init__(self, frame_ms: int) -> None:
        frames_per_second = 1000.0 / frame_ms
        self._levels: deque = deque(maxlen=max(1, round(ROOM_FLOOR_WINDOW_SECONDS
                                                         * frames_per_second)))
        self._min_frames = max(1, round(ROOM_FLOOR_MIN_SECONDS * frames_per_second))
        self._frame_seconds = frame_ms / 1000.0
        # The loop adds a level while the reply thread reads them all: a
        # read must not see the deque change under it.
        self._lock = threading.Lock()

    def hear(self, level: float, captured_at: float, hers_playing: bool,
             now: float) -> None:
        """One frame's level, captured at *captured_at* and read by the loop
        at *now*. It counts as the room when nothing of hers plays as it is
        read (*hers_playing*: the thinking tune or her voice), it was read
        promptly enough for that to say what played when it was captured,
        and it holds a sound: samples of exactly nothing are a muted or
        missing input, not a quiet room."""
        if hers_playing or level <= 0.0:
            return
        if now - captured_at > ROOM_FRAME_MAX_AGE_SECONDS:
            return
        with self._lock:
            self._levels.append(level)

    def level(self) -> Optional[float]:
        """The room's quiet level, or None until a second of quiet has
        been heard."""
        with self._lock:
            levels = list(self._levels)
        if len(levels) < self._min_frames:
            return None
        ordered = sorted(levels)
        return ordered[len(ordered) * ROOM_FLOOR_PERCENTILE // 100]

    def seconds_heard(self) -> float:
        """How much quiet the floor stands on, in seconds."""
        with self._lock:
            return len(self._levels) * self._frame_seconds
