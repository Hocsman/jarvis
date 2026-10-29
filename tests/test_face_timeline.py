"""The talking face's timeline and feeder (`src/jarvis/output/face/face.spec.md`,
"The clock", "Watching and resting", "Interrupting").

Pure logic: audio laid out by the time it will be heard, and the decision, at
each instant, to feed Ditto a chunk, wait for one to fall due, or rest.
"""

import numpy as np
import pytest

from jarvis.output.face import timeline as tl
from jarvis.output.face.timeline import Feeder, Timeline

RATE = tl.RATE
T0 = 1000.0   # perf_counter values are large; nothing may assume they start at 0


def _tone(seconds, rate, freq=220.0, start=0.0):
    t = start + np.arange(int(round(seconds * rate))) / rate
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestLayingAudio:
    def test_audio_is_read_back_at_the_time_it_will_be_heard(self):
        line = Timeline()
        sound = _tone(0.5, RATE)
        line.lay(T0 + 2.0, sound, RATE)

        before = line.read(T0 + 1.5, RATE // 2)
        during = line.read(T0 + 2.0, len(sound))

        assert not before.any()
        np.testing.assert_allclose(during, sound, atol=1e-6)

    def test_audio_at_another_rate_is_resampled_to_sixteen_kilohertz(self):
        line = Timeline()
        line.lay(T0, _tone(1.0, 24000, freq=300.0), 24000)

        got = line.read(T0 + 0.1, RATE // 2)
        want = _tone(0.5, RATE, freq=300.0, start=0.1)

        assert np.abs(got - want).max() < 0.02

    def test_audio_handed_over_in_small_pieces_reads_as_if_it_came_whole(self):
        # Kyutai hands over 80 ms at a time; the joins must not click.
        whole, pieces = Timeline(), Timeline()
        sound = _tone(1.0, 24000, freq=300.0)
        whole.lay(T0, sound, 24000)
        step = 1920
        for i in range(0, len(sound), step):
            pieces.lay(T0 + i / 24000, sound[i:i + step], 24000)

        a = whole.read(T0, RATE)
        b = pieces.read(T0, RATE)

        assert np.abs(a[:-64] - b[:-64]).max() < 1e-3

    def test_where_two_pieces_overlap_the_later_one_wins(self):
        line = Timeline()
        line.lay(T0, np.full(RATE, 0.25, np.float32), RATE)
        line.lay(T0 + 0.5, np.full(RATE // 4, -0.5, np.float32), RATE)

        assert line.read(T0 + 0.25, 1)[0] == pytest.approx(0.25)
        assert line.read(T0 + 0.6, 1)[0] == pytest.approx(-0.5)
        assert line.read(T0 + 0.8, 1)[0] == pytest.approx(0.25)

    def test_what_was_forgotten_reads_as_silence(self):
        line = Timeline()
        line.lay(T0, np.ones(RATE, np.float32), RATE)

        line.forget(T0 + 0.5)

        assert not line.read(T0, RATE // 2).any()
        assert line.read(T0 + 0.5, RATE // 2).all()

    def test_audio_beyond_what_the_timeline_holds_is_dropped_not_wrapped(self):
        line = Timeline(seconds=10.0)
        line.forget(T0)
        line.lay(T0 + 30.0, np.ones(RATE, np.float32), RATE)

        assert not line.read(T0 + 30.0, RATE).any()
        assert not line.read(T0, 10 * RATE).any()


class TestCutting:
    def test_a_cut_silences_everything_heard_from_then_on(self):
        line = Timeline()
        line.lay(T0, np.ones(2 * RATE, np.float32), RATE)

        line.cut(T0 + 1.0)

        assert line.read(T0, RATE).all()
        assert not line.read(T0 + 1.0, RATE).any()
        assert line.last_audio_end() == pytest.approx(T0 + 1.0)

    def test_audio_laid_after_a_cut_is_kept(self):
        # The reply that follows a stop word is spoken.
        line = Timeline()
        line.lay(T0, np.ones(RATE, np.float32), RATE)
        line.cut(T0 + 0.5)
        line.lay(T0 + 2.0, np.ones(RATE, np.float32), RATE)

        assert line.read(T0 + 2.0, RATE).all()


def _drive(feeder, line, until, watched, start=T0, tick=0.01):
    """Run the feeder on a fake clock; return the chunks fed, as (start, fed at)."""
    fed = []
    now = start
    while now < until:
        step = feeder.step(now, watched(now) if callable(watched) else watched)
        if step.feed is not None:
            fed.append((step.feed, now))
            feeder.fed(step.feed)
        now += tick
    return fed


class TestFeeding:
    def test_nothing_is_fed_while_no_one_watches_and_she_says_nothing(self):
        feeder = Feeder(Timeline())

        assert _drive(feeder, feeder.timeline, T0 + 5.0, watched=False) == []

    def test_while_watched_chunks_follow_each_other_every_two_hundred_milliseconds(self):
        feeder = Feeder(Timeline())

        fed = _drive(feeder, feeder.timeline, T0 + 3.0, watched=True)

        starts = [s for s, _ in fed]
        assert len(starts) >= 10
        assert np.allclose(np.diff(starts), tl.CHUNK_SECONDS)

    def test_a_chunk_is_fed_when_its_window_ends_one_delay_ahead_never_earlier(self):
        feeder = Feeder(Timeline())

        fed = _drive(feeder, feeder.timeline, T0 + 3.0, watched=True, tick=0.001)

        for start, at in fed:
            due = start + tl.CHUNK_SECONDS + tl.FUTURE_SECONDS - tl.DELAY
            assert due - 1e-9 <= at < due + 0.005

    def test_frames_are_stamped_with_the_time_their_audio_is_heard(self):
        stamps = Feeder.stamps(T0 + 2.0)

        assert stamps == pytest.approx([T0 + 2.0 + n / tl.FPS for n in range(tl.CHUNK_FRAMES)])

    def test_the_window_holds_what_ditto_reads_around_the_five_frames(self):
        start = T0 + 2.0
        first, end = Feeder.window(start)

        assert first == pytest.approx(start - tl.PAST_SECONDS)
        assert end == pytest.approx(start + tl.CHUNK_SECONDS + tl.FUTURE_SECONDS)
        assert round((end - first) * RATE) == tl.CHUNK_SAMPLES

    def test_audio_wakes_a_resting_face_in_time_to_draw_it(self):
        line = Timeline()
        heard = T0 + 3.0
        line.lay(heard, np.ones(RATE, np.float32), RATE)
        feeder = Feeder(line)

        fed = _drive(feeder, line, T0 + 6.0, watched=False)

        starts = [s for s, _ in fed]
        assert starts and starts[0] <= heard
        assert any(s <= heard + 0.9 < s + tl.CHUNK_SECONDS for s in starts)

    def test_after_her_last_word_the_face_rests_once_the_mouth_has_closed(self):
        line = Timeline()
        line.lay(T0 + 1.0, np.ones(RATE, np.float32), RATE)
        feeder = Feeder(line)

        fed = _drive(feeder, line, T0 + 10.0, watched=False)

        last = max(s for s, _ in fed)
        assert T0 + 2.0 + tl.TAIL_SECONDS - tl.CHUNK_SECONDS <= last < T0 + 2.0 + tl.TAIL_SECONDS + tl.CHUNK_SECONDS

    def test_a_viewer_leaving_lets_the_face_rest(self):
        feeder = Feeder(Timeline())

        fed = _drive(feeder, feeder.timeline, T0 + 6.0, watched=lambda now: now < T0 + 2.0)

        assert fed and max(at for _, at in fed) < T0 + 2.0 + 0.01

    def test_a_feeder_that_fell_far_behind_jumps_to_what_it_can_still_draw_on_time(self):
        feeder = Feeder(Timeline())
        _drive(feeder, feeder.timeline, T0 + 1.0, watched=True)

        late = feeder.step(T0 + 5.0, True)

        assert late.feed == pytest.approx(T0 + 5.0 + tl.DELAY - tl.CHUNK_SECONDS - tl.FUTURE_SECONDS)
