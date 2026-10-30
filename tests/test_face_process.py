"""The talking-face renderer's logic (`src/jarvis/output/face/face.spec.md`, "The
process", "Watching and resting", "Her lips at rest", "Interrupting", "Failure").

Ditto is faked: these tests drive the renderer with a real timeline and a real
shared-memory writer, and stand in for the model's frames themselves.
"""

import importlib
import io
import json
import sys
import time

import numpy as np
import pytest

from jarvis.output.face import process
from jarvis.output.face import timeline as tl
from jarvis.output.face.client import HEADER, MSG_AUDIO, MSG_CUT, MSG_QUIT
from jarvis.output.face.frames import HEIGHT, WIDTH, FrameReader, FrameWriter


class FakeStream:
    """Ditto, as far as the renderer can tell: chunks in, frames out on request."""

    def __init__(self):
        self.chunks = []
        self.ctrl_info = {}       # frame index -> Ditto's per-frame controls
        self.worker_exception = None
        self.stopped = False

    def run_chunk(self, audio):
        self.chunks.append(audio)

    def stop(self):
        self.stopped = True


def _picture(value):
    return np.full((HEIGHT, WIDTH, 3), value, np.uint8)


def _message(kind, at=0.0, rate=0, samples=None):
    payload = b"" if samples is None else np.asarray(samples, "<f4").tobytes()
    return HEADER.pack(kind, at, rate, len(payload)) + payload


@pytest.fixture
def renderer():
    r = process.Renderer(FakeStream(), FrameWriter())
    yield r
    r.close()


@pytest.fixture
def reader():
    r = FrameReader()
    yield r
    r.close()


def _watch(renderer, reader, now):
    """A front end reading at ``now``: the renderer then counts as watched."""
    renderer.frames.beat(now)
    reader.read(now)


def _shown(renderer, reader, at):
    renderer.frames.beat(at)
    frame = reader.read(at)
    return None if frame is None else int(frame.pixels[0, 0, 0])


class TestListening:
    def test_audio_lands_on_the_timeline_at_the_time_it_will_be_heard(self, renderer):
        t = 1000.0
        renderer.listen(io.BytesIO(_message(MSG_AUDIO, t, tl.RATE, np.ones(tl.RATE)) + _message(MSG_QUIT)))

        assert renderer.timeline.read(t, tl.RATE).all()
        assert not renderer.timeline.read(t - 1.0, tl.RATE).any()

    def test_a_cut_silences_the_timeline_and_starts_a_new_generation(self, renderer):
        t = 1000.0
        before = renderer.generation
        renderer.listen(io.BytesIO(_message(MSG_AUDIO, t, tl.RATE, np.ones(2 * tl.RATE))
                                   + _message(MSG_CUT, t + 1.0) + _message(MSG_QUIT)))

        assert renderer.generation == before + 1
        assert renderer.timeline.read(t, tl.RATE).all()
        assert not renderer.timeline.read(t + 1.0, tl.RATE).any()

    @pytest.mark.parametrize("ending", [
        _message(MSG_QUIT),
        b"",                                                         # the daemon closed the pipe
        _message(MSG_AUDIO, 1.0, tl.RATE, np.ones(100))[:-7],        # it died mid-message
        HEADER.pack(9, 0.0, 0, 3) + b"abc" + b"\x01\x02",            # a header cut short
    ])
    def test_the_renderer_stops_when_the_daemon_stops_talking(self, renderer, ending):
        renderer.listen(io.BytesIO(ending))

        assert renderer.quit.is_set()

    def test_audio_it_cannot_lay_is_skipped_not_fatal(self, renderer):
        t = 1000.0
        renderer.listen(io.BytesIO(_message(MSG_AUDIO, t, 0, np.ones(10))
                                   + _message(MSG_AUDIO, t, tl.RATE, np.ones(tl.RATE)) + _message(MSG_QUIT)))

        assert renderer.timeline.read(t, tl.RATE).all()


class TestPublishing:
    def test_frames_ditto_hands_back_are_published_at_their_stamps(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        first = renderer.due[0].stamp

        for value in range(tl.CHUNK_FRAMES):
            renderer.on_frame(_picture(value + 1))

        assert len(renderer.stream.chunks) == 1 and len(renderer.stream.chunks[0]) == tl.CHUNK_SAMPLES
        assert _shown(renderer, reader, first) == 1
        assert _shown(renderer, reader, first + 1 / tl.FPS) == 2

    def test_warm_up_frames_are_never_shown(self, renderer, reader):
        now = time.perf_counter()
        renderer._feed(now + 1.0, publish=False)

        renderer.on_frame(_picture(9))

        assert _shown(renderer, reader, now + 2.0) is None

    def test_frames_still_inside_ditto_when_the_face_rests_are_never_shown(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        stamp = renderer.due[0].stamp

        renderer.tick(now + process.WATCH_SECONDS + 1.0)   # nobody has read since: the face rests
        renderer.on_frame(_picture(9))

        assert renderer.feeder.next_start is None
        assert _shown(renderer, reader, stamp + 0.01) is None

    def test_frames_drawn_before_a_cut_and_stamped_after_it_are_not_shown(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        stamps = [due.stamp for due in renderer.due]
        renderer.listen(io.BytesIO(_message(MSG_CUT, stamps[2]) + _message(MSG_QUIT)))

        for value in range(tl.CHUNK_FRAMES):
            renderer.on_frame(_picture(value + 1))

        assert _shown(renderer, reader, stamps[-1] + 0.01) == 2

    def test_frames_published_after_their_moment_are_counted_and_reported_at_the_rest(self, renderer, reader, capsys):
        now = time.perf_counter()
        renderer._feed(now - 5.0, publish=True)
        for _ in range(tl.CHUNK_FRAMES):
            renderer.on_frame(_picture(1))

        renderer.tick(now)   # nobody watches: the face rests

        assert f"{tl.CHUNK_FRAMES} frames after their moment" in capsys.readouterr().err


class TestFeeding:
    def test_nothing_is_fed_while_no_one_watches(self, renderer):
        now = time.perf_counter()
        for i in range(20):
            renderer.tick(now + i * 0.05)

        assert renderer.stream.chunks == []

    def test_a_chunk_is_skipped_while_ditto_owes_a_frame_already_due(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        start = renderer.feeder.next_start
        late = renderer.due[0].stamp + 2 / tl.FPS   # the first frame fed is still inside Ditto
        _watch(renderer, reader, late)

        renderer.tick(late)

        assert len(renderer.stream.chunks) == 1
        assert renderer.feeder.next_start is not None and renderer.feeder.next_start > start

    def test_feeding_goes_on_while_ditto_keeps_up(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        for _ in range(tl.CHUNK_FRAMES):
            renderer.on_frame(_picture(1))
        start = renderer.feeder.next_start
        _watch(renderer, reader, renderer.feeder.due(start))

        renderer.tick(renderer.feeder.due(start))

        assert len(renderer.stream.chunks) == 2


class TestHerLipsAtRest:
    """While she is silent her lips are the portrait's.

    Ditto drives the mouth from the audio it is fed, silence included, and
    drawn from silence the lips drift and then press in, as if she bit them.
    So each frame tells Ditto how much its lips follow the voice: fully while
    she is heard, not at all in silence, with a short ramp at either end so
    the pauses between words do not close her mouth.
    """

    T = 1000.0
    SPEECH = 0.07         # Kyutai's speech, per frame
    NOISE_FLOOR = 0.0003  # and its silences

    def _voice(self, renderer, start, seconds):
        renderer.timeline.lay(start, self.SPEECH * np.ones(int(seconds * tl.RATE), np.float32), tl.RATE)

    def _weights(self, renderer):
        """What Ditto is handed for each frame it has been fed, in order: the
        k-th frame it draws is the k-th frame fed."""
        fed = len(renderer.stream.chunks) * tl.CHUNK_FRAMES
        return [renderer.stream.ctrl_info.get(i, {}).get("vad_alpha", 1.0) for i in range(fed)]

    def _feed(self, renderer, chunks):
        for n in range(chunks):
            renderer._feed(self.T + n * tl.CHUNK_SECONDS, publish=True)
        return self._weights(renderer)

    def test_in_silence_the_lips_are_the_portrait_s(self, renderer):
        weights = self._feed(renderer, 4)

        assert weights == [0.0] * len(weights)

    def test_her_silences_are_silence_though_not_quite_zero(self, renderer):
        noise = np.random.default_rng(7).normal(0.0, self.NOISE_FLOOR, 3 * tl.RATE).astype(np.float32)
        renderer.timeline.lay(self.T - 1.0, noise, tl.RATE)

        weights = self._feed(renderer, 4)

        assert weights == [0.0] * len(weights)

    def test_while_she_is_heard_they_follow_her_voice(self, renderer):
        self._voice(renderer, self.T - 1.0, 3.0)

        weights = self._feed(renderer, 4)

        assert weights == [1.0] * len(weights)

    def test_after_she_stops_they_return_to_the_portrait_s_gradually(self, renderer):
        self._voice(renderer, self.T - 1.0, 1.2)   # her voice ends 0.2 s in

        weights = self._feed(renderer, 4)

        after = weights[int(0.2 * tl.FPS):]
        ramp = after[:process.LIPS_RELEASE_FRAMES]
        assert all(0.0 < w < 1.0 for w in ramp), after
        assert all(a > b for a, b in zip(ramp, ramp[1:])), after
        assert after[process.LIPS_RELEASE_FRAMES:] == [0.0] * (len(after) - process.LIPS_RELEASE_FRAMES)

    def test_they_start_to_follow_just_before_she_speaks(self, renderer):
        """Her voice starts on a chunk's first frame: the frames before it,
        in the chunk before, see it coming in what that chunk carries."""
        onset = 2 * tl.CHUNK_FRAMES
        self._voice(renderer, self.T + onset / tl.FPS, 1.0)

        weights = self._feed(renderer, 4)

        ramp = weights[onset - process.LIPS_ATTACK_FRAMES:onset]
        assert weights[onset - process.LIPS_ATTACK_FRAMES - 1] == 0.0
        assert all(0.0 < w < 1.0 for w in ramp), weights
        assert all(a < b for a, b in zip(ramp, ramp[1:])), weights
        assert weights[onset] == 1.0

    def test_a_pause_between_words_does_not_close_her_mouth(self, renderer):
        self._voice(renderer, self.T - 1.0, 1.2)
        self._voice(renderer, self.T + 0.32, 1.0)   # 120 ms without a sound

        weights = self._feed(renderer, 4)

        assert min(weights[int(0.2 * tl.FPS):int(0.32 * tl.FPS)]) > 0.0

    def test_each_frame_is_named_by_its_place_in_what_ditto_draws(self, renderer):
        """Ditto counts the frames it draws from the first chunk fed, warm-up
        included; the control must land on the frame it was meant for."""
        self._voice(renderer, self.T - 1.0, 1.0)
        renderer._feed(self.T - tl.CHUNK_SECONDS, publish=False)    # a warm-up chunk, voiced

        renderer._feed(self.T, publish=True)            # silent: every frame is held

        assert sorted(renderer.stream.ctrl_info) == list(range(tl.CHUNK_FRAMES, 2 * tl.CHUNK_FRAMES))

    def test_a_skipped_chunk_takes_no_place_in_what_ditto_draws(self, renderer, reader):
        now = time.perf_counter()
        _watch(renderer, reader, now)
        renderer.tick(now)
        late = renderer.due[0].stamp + 2 / tl.FPS
        _watch(renderer, reader, late)
        renderer.tick(late)                             # skipped: Ditto owes a frame already due
        for _ in range(tl.CHUNK_FRAMES):
            renderer.on_frame(_picture(1))
        start = renderer.feeder.next_start
        _watch(renderer, reader, renderer.feeder.due(start))

        renderer.tick(renderer.feeder.due(start))

        assert len(renderer.stream.chunks) == 2
        assert sorted(renderer.stream.ctrl_info) == list(range(2 * tl.CHUNK_FRAMES))

    def test_after_a_rest_her_lips_start_at_rest(self, renderer, reader):
        """Someone stopped watching while she spoke and comes back to her
        silence: nothing is left of the ramp that followed her voice."""
        now = time.perf_counter()
        self._voice(renderer, now - 1.0, 5.0)
        _watch(renderer, reader, now)
        renderer.tick(now)
        renderer.tick(now + 2 * process.WATCH_SECONDS)     # nobody watches: the face rests
        back = now + 20.0
        _watch(renderer, reader, back)

        renderer.tick(back)

        assert len(renderer.stream.chunks) == 2
        assert self._weights(renderer)[:tl.CHUNK_FRAMES] == [1.0] * tl.CHUNK_FRAMES
        assert self._weights(renderer)[tl.CHUNK_FRAMES:] == [0.0] * tl.CHUNK_FRAMES

    def _ctrl_vad(self, monkeypatch):
        """Ditto's lip control, loaded as the suite loads the face: without PyTorch."""
        monkeypatch.setitem(sys.modules, "torch", None)
        for name in ("jarvis.output.face.ditto.atomic_components.motion_stitch",
                     "jarvis.output.face.ditto.models.stitch_network"):
            monkeypatch.delitem(sys.modules, name, raising=False)
        return importlib.import_module("jarvis.output.face.ditto.atomic_components.motion_stitch").ctrl_vad

    def test_ditto_holds_the_lips_and_nothing_else_to_the_portrait_s(self, monkeypatch):
        """Every other keypoint stays as driven, so a silent face goes on blinking."""
        ctrl_vad = self._ctrl_vad(monkeypatch)
        lips = [6, 12, 14, 17, 19, 20]
        others = [k for k in range(21) if k not in lips]
        driven = np.arange(63, dtype=np.float32).reshape(1, 63) + 1.0
        portrait = {"exp": np.zeros((1, 63), np.float32)}

        held = ctrl_vad({"exp": driven.copy()}, portrait, 0.0)["exp"].reshape(21, 3)
        ramped = ctrl_vad({"exp": driven.copy()}, portrait, 0.25)["exp"].reshape(21, 3)

        assert not held[lips].any()
        assert np.allclose(ramped[lips], 0.25 * driven.reshape(21, 3)[lips])
        for blended in (held, ramped):
            assert (blended[others] == driven.reshape(21, 3)[others]).all()


class TestTellingTheHandle:
    def test_the_handle_hears_when_someone_starts_and_stops_watching(self, reader):
        events = []
        renderer = process.Renderer(FakeStream(), FrameWriter(), on_event=lambda **e: events.append(e))
        try:
            now = time.perf_counter()
            _watch(renderer, reader, now)
            renderer.tick(now)
            renderer.tick(now + 0.05)
            renderer.tick(now + process.WATCH_SECONDS + 1.0)
        finally:
            renderer.close()

        assert events == [{"event": "watched", "value": True}, {"event": "watched", "value": False}]


class TestStartingUp:
    def _events(self, capsys):
        return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]

    def test_no_portrait_is_a_failure_with_its_reason(self, monkeypatch, capsys):
        monkeypatch.delenv(process.PORTRAIT_ENV, raising=False)

        assert process.main([]) == 1
        assert self._events(capsys) == [{"event": "failed", "reason": "no portrait was named"}]

    def test_once_ready_the_renderer_tells_the_handle_who_watches(self, monkeypatch, capsys):
        monkeypatch.setenv(process.PORTRAIT_ENV, "portrait.png")
        monkeypatch.setattr(process, "_check_packages", lambda: None)
        built = process.Renderer(FakeStream(), FrameWriter())
        monkeypatch.setattr(process.Renderer, "build", classmethod(lambda cls, portrait, framing: built))
        monkeypatch.setattr(process.Renderer, "warm_up", lambda self: None)

        def run(self):
            now = time.perf_counter()
            viewer = FrameReader()
            self.frames.beat(now)
            viewer.read(now)
            viewer.close()
            self.tick(now)

        monkeypatch.setattr(process.Renderer, "run", run)
        monkeypatch.setattr(process.sys, "stdin", io.TextIOWrapper(io.BytesIO(b"")))

        assert process.main([]) == 0
        assert self._events(capsys) == [{"event": "ready", "delay": tl.DELAY},
                                        {"event": "watched", "value": True}]

    def test_a_missing_package_is_a_failure_with_its_reason(self, monkeypatch, capsys):
        monkeypatch.setenv(process.PORTRAIT_ENV, "portrait.png")

        def missing():
            raise RuntimeError("tensorrt-cu12 is not installed")

        monkeypatch.setattr(process, "_check_packages", missing)

        assert process.main([]) == 1
        assert self._events(capsys) == [{"event": "failed", "reason": "tensorrt-cu12 is not installed"}]
