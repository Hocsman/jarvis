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

from jarvis.output.face.client import FaceRenderer, MSG_AUDIO, MSG_CUT, MSG_QUIT, HEADER


def _fake(body):
    """A command that runs ``body`` as the renderer, with the protocol's helpers."""
    script = textwrap.dedent("""
        import json, struct, sys, time
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


def test_the_header_is_what_the_renderer_reads():
    assert HEADER.format == "<BdII"
    assert struct.calcsize(HEADER.format) == HEADER.size
