"""The daemon's handle on the talking-face renderer (`src/jarvis/output/face/face.spec.md`,
"The process").

The renderer process is faked by a small script, so these tests run without a
GPU or a model. What they hold the handle to: it reports ready only after the
renderer said so, turns the face off for good on any failure, never raises to
its caller and never blocks it.
"""

import struct
import sys
import textwrap
import time

import numpy as np
import pytest

from jarvis.output.face.client import FaceRenderer, MSG_AUDIO, MSG_CUT, MSG_QUIT, HEADER, PORTRAIT_ENV


def _fake(body):
    """A command that runs ``body`` as the renderer, with the protocol's helpers."""
    script = textwrap.dedent("""
        import json, os, struct, sys, time
        HEADER = struct.Struct(%r)
        def say(**event):
            print(json.dumps(event), flush=True)
        def messages():
            while True:
                head = sys.stdin.buffer.read(HEADER.size)
                if len(head) < HEADER.size:
                    return
                kind, t, rate, n = HEADER.unpack(head)
                yield kind, t, rate, sys.stdin.buffer.read(n)
    """ % HEADER.format) + textwrap.dedent(body)
    return [sys.executable, "-c", script]


def _wait(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def made():
    renderers = []

    def make(body, **kwargs):
        r = FaceRenderer("portrait.png", command=_fake(body), **kwargs)
        renderers.append(r)
        r.start()
        return r

    yield make
    for r in renderers:
        r.stop()


class TestReadiness:
    def test_it_is_ready_only_once_the_renderer_says_so(self, made):
        r = made("""
            time.sleep(0.5)
            say(event="ready", delay=0.75)
            for _ in messages():
                pass
        """)

        assert not r.ready()
        assert _wait(r.ready)
        assert r.delay == pytest.approx(0.75)

    def test_a_renderer_that_fails_turns_the_face_off_with_its_reason(self, made):
        r = made("""
            say(event="failed", reason="no face in the portrait")
        """)

        assert _wait(lambda: r.failure is not None)
        assert not r.ready()
        assert "no face" in r.failure

    def test_a_renderer_that_dies_turns_the_face_off(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            time.sleep(0.3)
            sys.exit(3)
        """)

        assert _wait(r.ready)
        assert _wait(lambda: not r.ready(), timeout=10)
        assert r.failure

    def test_a_renderer_that_cannot_start_turns_the_face_off(self):
        r = FaceRenderer("portrait.png", command=["no-such-program-for-the-face"])
        r.start()
        try:
            assert not r.ready()
            assert r.failure
        finally:
            r.stop()


class TestSending:
    def test_audio_and_cuts_reach_the_renderer_in_order(self, made, tmp_path):
        log = tmp_path / "got.txt"
        r = made(f"""
            say(event="ready", delay=0.75)
            with open({str(log)!r}, "w") as f:
                for kind, t, rate, payload in messages():
                    f.write(f"{{kind}} {{t}} {{rate}} {{len(payload)}}\\n")
                    f.flush()
                    if kind == {MSG_QUIT}:
                        break
        """)
        assert _wait(r.ready)

        r.send_audio(12.5, np.zeros(480, np.float32), 24000)
        r.cut(13.0)
        r.stop()

        lines = log.read_text(encoding="utf-8").split("\n")
        assert lines[0] == f"{MSG_AUDIO} 12.5 24000 {480 * 4}"
        assert lines[1] == f"{MSG_CUT} 13.0 0 0"
        assert lines[2].startswith(f"{MSG_QUIT} ")

    def test_sending_never_blocks_even_when_the_renderer_stops_reading(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            time.sleep(60)
        """, stall_seconds=1.0)
        assert _wait(r.ready)

        t = time.monotonic()
        for _ in range(50):
            r.send_audio(1.0, np.zeros(48000, np.float32), 24000)
        assert time.monotonic() - t < 0.5

        assert _wait(lambda: not r.ready(), timeout=10)
        assert "reading" in r.failure

    def test_sending_to_a_face_that_is_off_does_nothing(self, made):
        r = made("""
            say(event="failed", reason="no GPU")
        """)
        assert _wait(lambda: r.failure is not None)

        r.send_audio(1.0, np.zeros(10, np.float32), 24000)
        r.cut(1.0)


class TestStartingAndStopping:
    def test_the_portrait_reaches_the_renderer_in_its_environment_not_its_command_line(self, made, tmp_path):
        log = tmp_path / "env.txt"
        r = made(f"""
            open({str(log)!r}, "w", encoding="utf-8").write(os.environ.get({PORTRAIT_ENV!r}, "") + "|" + " ".join(sys.argv))
            say(event="ready", delay=0.75)
            for _ in messages():
                pass
        """)
        assert _wait(r.ready)

        portrait, argv = log.read_text(encoding="utf-8").split("|")
        assert portrait == "portrait.png"
        assert "portrait.png" not in argv

    def test_the_packaged_app_says_why_instead_of_starting_a_child(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        r = FaceRenderer("portrait.png", command=[sys.executable, "-c", "raise SystemExit(5)"])

        r.start()
        try:
            assert r.failure and "source install" in r.failure
            assert r._proc is None
        finally:
            r.stop()

    def test_a_renderer_that_ignores_quit_is_killed_on_stop(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            time.sleep(60)
        """)
        assert _wait(r.ready)

        started = time.monotonic()
        r.stop(timeout=1.0)

        assert time.monotonic() - started < 5.0
        assert r._proc.poll() is not None

    def test_a_line_that_is_not_an_event_does_not_deafen_the_handle(self, made):
        r = made("""
            print(42, flush=True)
            print("[1, 2]", flush=True)
            say(event="ready", delay=0.75)
            for _ in messages():
                pass
        """)

        assert _wait(r.ready)


class TestWatching:
    def test_the_handle_follows_whether_someone_watches(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            say(event="watched", value=True)
            for _ in messages():
                pass
        """)

        assert _wait(r.watched)

    def test_nobody_watches_a_face_that_failed(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            say(event="watched", value=True)
            time.sleep(0.5)
            say(event="failed", reason="no GPU")
        """)

        assert _wait(r.watched)
        assert _wait(lambda: r.failure is not None)
        assert not r.watched()

    @pytest.mark.parametrize("delay", ["NaN", "Infinity", "5.0", "-1", '"soon"'])
    def test_an_impossible_delay_turns_the_face_off_rather_than_hold_her_voice(self, made, delay):
        r = made(f"""
            print('{{"event": "ready", "delay": {delay}}}', flush=True)
            for _ in messages():
                pass
        """)

        assert _wait(lambda: r.failure is not None)
        assert "delay" in r.failure
        assert not r.ready()


class TestNeverRaising:
    def test_what_cannot_be_sent_is_dropped_without_raising(self, made):
        r = made("""
            say(event="ready", delay=0.75)
            for _ in messages():
                pass
        """)
        assert _wait(r.ready)

        r.send_audio(None, object(), "fast")
        r.send_audio(1.0, np.zeros(10, np.float32), -5)
        r.cut("soon")

        assert r.ready()
