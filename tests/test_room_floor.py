"""The voice proof's floor is the room's quiet level.

The echo detector weighs an utterance against `energy_spike_threshold`
times a floor (listening.spec.md, "And it takes a voice"). That floor is
read when her reply starts, and the microphone's last second then is
rarely the room: the thinking tune plays until her first sentence is
handed over, the end of his question can still be in it, and a reply from
elsewhere can start the moment another ends. A floor read from it sat at
five times the room's level, and his soft answer right after her last
word fell under the bar and was taken for her echo.

These pin what the floor is made of: what the microphone hears while
nothing of hers plays, over a long stretch, at a low percentile.
"""

from __future__ import annotations

import pytest

from jarvis.config import get_default_config
from jarvis.listening.room_floor import (
    ROOM_FLOOR_MIN_SECONDS,
    ROOM_FLOOR_WINDOW_SECONDS,
    ROOM_FRAME_MAX_AGE_SECONDS,
    RoomFloor,
)

pytestmark = pytest.mark.unit

FRAME_MS = 20
ROOM = 0.001


class _Microphone:
    """Feeds a RoomFloor the way the listening loop does: one level per
    frame, captured on a clock that advances a frame at a time, read by the
    loop *late* seconds after capture, with or without her sound playing
    when the loop reads it."""

    def __init__(self, floor: RoomFloor) -> None:
        self.floor = floor
        self.clock = 1_000.0

    def hears(self, level: float, seconds: float, *, hers_playing: bool = False,
              late: float = 0.0) -> None:
        for _ in range(round(seconds * 1000 / FRAME_MS)):
            self.clock += FRAME_MS / 1000
            self.floor.hear(level, captured_at=self.clock, hers_playing=hers_playing,
                            now=self.clock + late)


def _room(seconds: float = 3.0):
    floor = RoomFloor(frame_ms=FRAME_MS)
    mic = _Microphone(floor)
    mic.hears(ROOM, seconds)
    return floor, mic


def test_a_quiet_room_is_its_own_floor():
    floor, _ = _room()

    assert floor.level() == pytest.approx(ROOM)


def test_the_thinking_tune_never_raises_it():
    """However long she thinks, the tune is not the room."""
    floor, mic = _room()

    mic.hears(ROOM * 5, ROOM_FLOOR_WINDOW_SECONDS * 2, hers_playing=True)

    assert floor.level() == pytest.approx(ROOM)


def test_her_voice_never_raises_it():
    """A reply from elsewhere said the moment hers ends reads the floor
    just after her voice: her voice is not the room either."""
    floor, mic = _room()

    mic.hears(ROOM * 30, ROOM_FLOOR_WINDOW_SECONDS * 2, hers_playing=True)

    assert floor.level() == pytest.approx(ROOM)


def test_his_own_words_never_raise_it():
    """He asked just before her reply, for as long as the voice detector
    keeps an utterance whole: what is left of the stretch is still enough
    room for the floor."""
    longest = get_default_config()["max_utterance_ms"] / 1000
    floor, mic = _room(seconds=ROOM_FLOOR_WINDOW_SECONDS)

    mic.hears(ROOM * 20, longest)

    assert floor.level() == pytest.approx(ROOM)


def test_frames_read_long_after_their_capture_do_not_count():
    """The loop knows what plays when it reads a frame, not when the frame
    was captured. Behind after Whisper or the judge held it, it catches up
    on audio she was saying while nothing plays any more: none of it is
    taken for the room."""
    floor, mic = _room(seconds=ROOM_FLOOR_MIN_SECONDS)

    mic.hears(ROOM * 30, ROOM_FLOOR_WINDOW_SECONDS * 2,
              late=ROOM_FRAME_MAX_AGE_SECONDS * 4)

    assert floor.level() == pytest.approx(ROOM)


def test_it_follows_the_room():
    """A fan starts: within a stretch of quiet the floor is the new room."""
    floor, mic = _room(seconds=ROOM_FLOOR_WINDOW_SECONDS)

    mic.hears(ROOM * 4, ROOM_FLOOR_WINDOW_SECONDS)

    assert floor.level() == pytest.approx(ROOM * 4)


def test_without_a_second_of_quiet_there_is_no_floor():
    floor, mic = _room(seconds=ROOM_FLOOR_MIN_SECONDS / 2)

    mic.hears(ROOM * 5, ROOM_FLOOR_WINDOW_SECONDS, hers_playing=True)

    assert floor.level() is None


def test_a_silent_microphone_is_not_a_room():
    """Samples of exactly nothing are a muted or missing input, not a
    quiet room, and give no floor."""
    floor = RoomFloor(frame_ms=FRAME_MS)

    _Microphone(floor).hears(0.0, ROOM_FLOOR_WINDOW_SECONDS)

    assert floor.level() is None


@pytest.mark.parametrize("frame_ms", [10, 20, 30])
def test_the_stretch_is_measured_in_seconds_whatever_the_frame(frame_ms):
    """The window and the minimum are durations: a shorter frame does not
    shorten the stretch the floor is read over."""
    floor = RoomFloor(frame_ms=frame_ms)
    clock = 0.0

    def hears(level, seconds):
        nonlocal clock
        for _ in range(round(seconds * 1000 / frame_ms)):
            clock += frame_ms / 1000
            floor.hear(level, captured_at=clock, hers_playing=False, now=clock)

    hears(ROOM, ROOM_FLOOR_MIN_SECONDS * 0.9)
    assert floor.level() is None

    hears(ROOM, ROOM_FLOOR_WINDOW_SECONDS)
    hears(ROOM * 20, ROOM_FLOOR_WINDOW_SECONDS * 0.8)
    assert floor.level() == pytest.approx(ROOM)
