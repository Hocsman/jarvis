"""The talking face's frames in shared memory (`src/jarvis/output/face/face.spec.md`,
"Shared memory", "Watching and resting", "Interrupting").

The renderer writes frames stamped with the moment they are to be on screen; a
front end reads the newest one due, never one early, never a torn one, never
one showing words a cut took back.
"""

import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from jarvis.output.face import frames as fr
from jarvis.output.face.frames import FrameReader, FrameWriter

NOW = 5000.0


def _frame(value):
    return np.full((fr.HEIGHT, fr.WIDTH, 3), value, np.uint8)


@pytest.fixture
def writer():
    w = FrameWriter()
    w.beat(NOW)
    yield w
    w.close()


@pytest.fixture
def reader():
    r = FrameReader()
    yield r
    r.close()


class TestReading:
    def test_the_newest_frame_due_is_shown(self, writer, reader):
        for i in range(5):
            writer.publish(_frame(i), NOW + i * 0.04, 0)

        got = reader.read(NOW + 0.1)

        assert got.stamp == pytest.approx(NOW + 0.08)
        assert (got.pixels == 2).all()

    def test_a_frame_is_never_shown_before_its_stamp(self, writer, reader):
        writer.publish(_frame(7), NOW + 1.0, 0)

        assert reader.read(NOW + 0.5) is None

    def test_no_renderer_means_no_frame(self, reader):
        assert reader.read(NOW) is None

    def test_a_segment_that_cannot_be_mapped_is_not_used(self, monkeypatch):
        """On POSIX, an empty segment (a renderer killed while creating it,
        or one another user made) cannot be mapped at all."""
        def unmappable(*args, **kwargs):
            raise ValueError("cannot mmap an empty file")

        monkeypatch.setattr(fr.shared_memory, "SharedMemory", unmappable)

        assert FrameReader("jarvis_face_test_empty").read(NOW) is None

    def test_a_renderer_that_stopped_beating_shows_nothing(self, writer, reader):
        writer.publish(_frame(1), NOW, 0)

        assert reader.read(NOW + 0.1) is not None
        assert reader.read(NOW + fr.STALE_SECONDS + 1.0) is None

    def test_the_ring_keeps_frames_far_enough_ahead_of_the_clock(self, writer, reader):
        # The delay keeps under a second of frames ahead; the ring holds them all.
        for i in range(fr.SLOTS):
            writer.publish(_frame(i % 200), NOW + i * 0.04, 0)

        got = reader.read(NOW + (fr.SLOTS - 1) * 0.04)

        assert got.stamp == pytest.approx(NOW + (fr.SLOTS - 1) * 0.04)

    def test_a_copy_torn_by_a_write_is_thrown_away(self, writer, reader):
        writer.publish(_frame(1), NOW, 0)
        writer.publish(_frame(2), NOW + 0.04, 0)
        real_copy = reader._copy_pixels

        def torn(slot):
            pixels = real_copy(slot)
            if (pixels == 2).all():
                # The whole ring is rewritten, with frames not yet due, during the copy.
                for _ in range(fr.SLOTS):
                    writer.publish(_frame(9), NOW + 10.0, 0)
            return pixels

        reader._copy_pixels = torn

        assert reader.read(NOW + 0.05) is None


class TestFollowingTheRenderer:
    def test_a_reader_that_outlived_its_renderer_shows_the_next_one(self, monkeypatch, reader):
        monkeypatch.setattr(fr, "_RETRY_SECONDS", 0.0)
        first = FrameWriter()
        first.beat(NOW)
        first.publish(_frame(1), NOW, 0)
        assert (reader.read(NOW + 0.01).pixels == 1).all()

        first.close()
        second = FrameWriter()   # the reader still holds the first one's mapping
        try:
            later = NOW + 10.0
            second.beat(later)
            second.publish(_frame(2), later, 0)

            got = None
            for _ in range(3):
                got = reader.read(later + 0.01) or got
            assert got is not None and (got.pixels == 2).all()
        finally:
            second.close()


class TestCut:
    def test_frames_published_before_a_cut_and_stamped_after_it_are_withdrawn(self, writer, reader):
        for i in range(10):
            writer.publish(_frame(i), NOW + i * 0.1, 0)

        writer.cut(NOW + 0.45)
        got = reader.read(NOW + 0.8)

        assert got.stamp == pytest.approx(NOW + 0.4)

    def test_frames_drawn_before_a_cut_and_published_after_it_are_refused(self, writer, reader):
        writer.publish(_frame(1), NOW, 0)
        writer.cut(NOW + 0.45)

        assert writer.publish(_frame(2), NOW + 0.5, 0) is False
        assert writer.publish(_frame(3), NOW + 0.3, 0) is True
        assert reader.read(NOW + 0.6).stamp == pytest.approx(NOW + 0.3)

    def test_a_second_cut_does_not_bring_back_what_the_first_took(self, writer, reader):
        writer.cut(NOW + 0.4)
        writer.cut(NOW + 0.8)

        assert writer.publish(_frame(1), NOW + 0.5, 0) is False
        assert writer.publish(_frame(2), NOW + 0.9, 1) is False
        assert writer.publish(_frame(3), NOW + 0.5, 1) is True

    def test_frames_drawn_after_the_cut_are_shown(self, writer, reader):
        for i in range(10):
            writer.publish(_frame(i), NOW + i * 0.1, 0)
        generation = writer.cut(NOW + 0.45)

        writer.publish(_frame(42), NOW + 0.9, generation)

        assert (reader.read(NOW + 0.95).pixels == 42).all()


class TestWatching:
    def test_reading_tells_the_renderer_someone_is_watching(self, writer, reader):
        assert writer.viewer_age(NOW) == float("inf")

        reader.read(NOW)

        assert writer.viewer_age(NOW + 0.5) == pytest.approx(0.5)


def test_another_process_reads_what_the_renderer_published(writer):
    writer.publish(_frame(77), NOW, 0)
    code = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(fr.__file__).rsplit('jarvis', 1)[0]!r})
        from jarvis.output.face import frames
        r = frames.FrameReader({writer.name!r})
        f = r.read({NOW + 0.01})
        print(int(f.pixels[0, 0, 0]) if f is not None else 'none')
        r.close()
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)

    assert out.stdout.strip() == "77", out.stderr
    assert writer.viewer_age(NOW + 0.01) < 1.0
