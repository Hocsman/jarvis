"""The face window showing her talking face (`src/desktop_app/desktop_app.spec.md`,
"Face"; `src/jarvis/output/face/face.spec.md`, "Watching and resting").

The renderer is stood in for by a real ``FrameWriter`` on a segment of the
test's own, or by a reader that returns the frames a test hands it. What is
pinned: the window shows the frame due, fitted to the window with its
proportions kept; it shows the drawn face when no frame comes; and it reads,
which is what tells the renderer someone is watching, only while it is on
screen.
"""

from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from jarvis.output.face import frames as fr
from jarvis.output.face.frames import Frame, FrameReader, FrameWriter

GREEN = (0, 200, 0)
BLUE = (20, 40, 220)


def _pixels(rgb):
    return np.full((fr.HEIGHT, fr.WIDTH, 3), rgb, np.uint8)


def _until(qapp, condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if condition():
            return True
        time.sleep(0.005)
    qapp.processEvents()
    return condition()


def _wait(qapp, seconds):
    _until(qapp, lambda: False, seconds)


def _colour_at(window, x, y):
    colour = window.grab().toImage().pixelColor(x, y)
    return colour.red(), colour.green(), colour.blue()


def _centre(window):
    return _colour_at(window, window.width() // 2, window.height() // 2)


def _near(colour, rgb, tolerance=6):
    return all(abs(a - b) <= tolerance for a, b in zip(colour, rgb))


class Frames:
    """A reader handing out what the test gives it, counting its reads."""

    def __init__(self):
        self.frame = None
        self.reads = 0

    def show(self, rgb, seq=1):
        self.frame = Frame(_pixels(rgb), time.perf_counter(), seq)

    def stop(self):
        self.frame = None

    def __call__(self):
        self.reads += 1
        return self.frame


@pytest.fixture
def segment():
    name = "jarvis_face_test_" + uuid.uuid4().hex[:12]
    writer = FrameWriter(name)
    reader = FrameReader(name)
    writer.beat(time.perf_counter())
    yield writer, reader
    reader.close()
    writer.close()


@pytest.fixture
def window_with(qapp):
    from desktop_app.face_widget import FaceWindow

    made = []

    def make(read_frame):
        window = FaceWindow(read_frame=read_frame)
        made.append(window)
        return window

    yield make
    for window in made:
        window.hide()
        window.deleteLater()
    qapp.processEvents()


class TestShowingHerFace:
    def test_the_window_shows_the_frame_the_renderer_published(self, qapp, segment, window_with):
        writer, reader = segment
        writer.publish(_pixels(GREEN), time.perf_counter() - 0.01, 0)
        window = window_with(reader.read)

        window.show()

        assert _until(qapp, lambda: _near(_centre(window), GREEN))

    def test_each_new_frame_replaces_the_one_on_screen(self, qapp, segment, window_with):
        writer, reader = segment
        writer.publish(_pixels(GREEN), time.perf_counter() - 0.02, 0)
        window = window_with(reader.read)
        window.show()
        assert _until(qapp, lambda: _near(_centre(window), GREEN))

        writer.publish(_pixels(BLUE), time.perf_counter(), 0)

        assert _until(qapp, lambda: _near(_centre(window), BLUE))

    def test_a_frame_due_later_is_not_shown_before_its_moment(self, qapp, segment, window_with):
        writer, reader = segment
        writer.publish(_pixels(GREEN), time.perf_counter() - 0.02, 0)
        window = window_with(reader.read)
        window.show()
        assert _until(qapp, lambda: _near(_centre(window), GREEN))

        writer.publish(_pixels(BLUE), time.perf_counter() + 0.4, 0)
        _wait(qapp, 0.2)

        assert _near(_centre(window), GREEN)
        assert _until(qapp, lambda: _near(_centre(window), BLUE))

    def test_the_frame_fills_the_window_as_a_square_centred_in_it(self, qapp, window_with):
        frames = Frames()
        frames.show(GREEN)
        window = window_with(frames)
        window.resize(640, 420)
        window.show()
        assert _until(qapp, lambda: _near(_centre(window), GREEN))

        image = window.grab().toImage()
        row = [_near(_colour_at(window, x, image.height() // 2), GREEN) for x in range(image.width())]
        column = [_near(_colour_at(window, image.width() // 2, y), GREEN) for y in range(image.height())]
        left, right = row.index(True), len(row) - row[::-1].index(True)
        top, bottom = column.index(True), len(column) - column[::-1].index(True)

        assert abs((right - left) - (bottom - top)) <= 2           # its proportions kept
        assert bottom - top >= image.height() - 40                 # as large as the window allows
        assert abs(left - (image.width() - right)) <= 2            # centred
        assert not _near(_colour_at(window, 5, image.height() // 2), GREEN)


class TestTheDrawnFace:
    def test_with_no_renderer_the_window_shows_the_drawn_face(self, qapp, window_with):
        frames = Frames()
        window = window_with(frames)

        window.show()
        _wait(qapp, 0.1)

        assert frames.reads > 0
        assert not _near(_centre(window), GREEN)

    def test_it_holds_her_last_frame_for_a_second_then_goes_back_to_the_drawn_face(self, qapp, window_with):
        frames = Frames()
        frames.show(GREEN)
        window = window_with(frames)
        window.show()
        assert _until(qapp, lambda: _near(_centre(window), GREEN))

        frames.stop()
        _wait(qapp, 0.5)
        assert _near(_centre(window), GREEN)

        assert _until(qapp, lambda: not _near(_centre(window), GREEN), timeout=1.5)

    def test_her_face_comes_back_with_the_next_frame(self, qapp, window_with):
        frames = Frames()
        window = window_with(frames)
        window.show()
        _wait(qapp, 0.1)
        assert not _near(_centre(window), BLUE)

        frames.show(BLUE)

        assert _until(qapp, lambda: _near(_centre(window), BLUE))

    def test_a_window_opened_again_does_not_show_a_frame_from_before(self, qapp, window_with):
        frames = Frames()
        frames.show(GREEN)
        window = window_with(frames)
        window.show()
        assert _until(qapp, lambda: _near(_centre(window), GREEN))
        window.hide()
        frames.stop()

        window.show()
        qapp.processEvents()

        assert not _near(_centre(window), GREEN)


class TestReadingIsWatching:
    def test_on_screen_the_renderer_sees_a_viewer(self, qapp, segment, window_with):
        writer, reader = segment
        window = window_with(reader.read)

        window.show()

        assert _until(qapp, lambda: writer.viewer_age(time.perf_counter()) < 0.1)

    def test_it_reads_often_enough_to_show_each_frame_within_a_refresh(self, qapp, window_with):
        frames = Frames()
        window = window_with(frames)
        window.show()
        _wait(qapp, 0.05)
        before = frames.reads

        _wait(qapp, 0.5)

        assert frames.reads - before >= 20     # 60 a second asked; a loaded machine misses some

    def test_hidden_it_reads_nothing(self, qapp, window_with):
        frames = Frames()
        window = window_with(frames)
        window.show()
        assert _until(qapp, lambda: frames.reads > 0)

        window.hide()
        qapp.processEvents()
        after_hiding = frames.reads
        _wait(qapp, 0.3)

        assert frames.reads == after_hiding

    def test_minimised_it_reads_nothing(self, qapp, window_with):
        frames = Frames()
        window = window_with(frames)
        window.show()
        assert _until(qapp, lambda: frames.reads > 0)

        window.showMinimized()
        qapp.processEvents()
        after_minimising = frames.reads
        _wait(qapp, 0.3)

        assert frames.reads == after_minimising
        window.showNormal()
        assert _until(qapp, lambda: frames.reads > after_minimising)

    def test_hidden_the_renderer_stops_seeing_a_viewer(self, qapp, segment, window_with):
        writer, reader = segment
        window = window_with(reader.read)
        window.show()
        assert _until(qapp, lambda: writer.viewer_age(time.perf_counter()) < 0.1)

        window.hide()
        _wait(qapp, 0.3)

        assert writer.viewer_age(time.perf_counter()) > 0.25


class TestWhereTheFramesComeFrom:
    def test_by_default_the_window_shows_the_frames_jarvis_offers_a_front_end(self, qapp, monkeypatch):
        from desktop_app.face_widget import FaceWindow

        frames = Frames()
        frames.show(GREEN)
        monkeypatch.setattr(fr, "read_face_frame", lambda now=None: frames())
        window = FaceWindow()
        try:
            window.show()
            assert _until(qapp, lambda: _near(_centre(window), GREEN))
        finally:
            window.hide()
            window.deleteLater()

    def test_importing_the_window_loads_no_numpy(self):
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src"
        probe = (f"import sys; sys.path.insert(0, {str(src)!r}); "
                 "import desktop_app.face_widget; print('numpy' in sys.modules)")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                timeout=120, env=env)

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "False"
