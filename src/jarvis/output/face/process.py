"""The talking-face renderer: a child process of the daemon (`face.spec.md`).

usage: JARVIS_FACE_PORTRAIT=PATH python -m jarvis.output.face.process [--framing head|face]

The portrait comes in the environment, which only this user can read, rather
than on the command line, which every user can. The renderer reports on stdout,
one JSON object per line (``ready``, ``watched`` or ``failed``), reads audio, cuts and quit
from stdin (see ``client.HEADER``), and publishes frames to shared memory
(``frames.py``). Everything else goes to stderr, which its handle logs.
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from . import timeline as tl
from .client import HEADER, MSG_AUDIO, MSG_CUT, MSG_QUIT, PORTRAIT_ENV

CROP = {"head": 2.8, "face": 2.3}
STEPS = 10
WARM_UP_SECONDS = 3.0
WATCH_SECONDS = 2.0      # a viewer heartbeat younger than this means someone is watching
KEEP_SECONDS = 5.0       # audio heard longer ago than this is forgotten
# Her lips at rest: a frame whose audio has at most this RMS is silence. Kyutai's
# speech sits near 0.07 RMS per frame and its silences near 0.0003.
SILENCE_RMS = 0.005
LIPS_ATTACK_FRAMES = 2    # the lips start to follow her voice this many frames before it
LIPS_RELEASE_FRAMES = 6   # and return to the portrait's over this many after it


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _check_packages() -> None:
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"PyTorch cannot be loaded ({e})")
    if not torch.cuda.is_available():
        raise RuntimeError("the face needs an NVIDIA GPU with CUDA, and none is available to PyTorch")
    for module, package in (("tensorrt", "tensorrt-cu12"), ("cv2", "opencv-python-headless"), ("einops", "einops")):
        try:
            __import__(module)
        except Exception as e:
            raise RuntimeError(f"{package} is not installed ({e}); it is installed by hand, like moshi")


@dataclass
class _Due:
    """What the renderer knows of a frame it fed and Ditto has not handed back."""
    stamp: float
    generation: int
    epoch: int        # the stretch of feeding it belongs to
    publish: bool


class Renderer:
    """Feeds Ditto from the timeline and publishes what it draws.

    ``stream`` is Ditto's (``ditto/stream.py``) once set up with ``on_frame``,
    ``frames`` a ``FrameWriter``, ``on_event(**event)`` where the events for
    the handle go. Everything here is plain logic: ``build`` loads the model.
    """

    def __init__(self, stream, frames, delay: float = tl.DELAY, on_event=None) -> None:
        self.stream = stream
        self.frames = frames
        self.on_event = on_event or (lambda **event: None)
        self.watched = False
        self.lock = threading.Lock()
        self.timeline = tl.Timeline()
        self.feeder = tl.Feeder(self.timeline, delay)
        self.due = collections.deque()       # one _Due per frame fed, in order
        self.generation = 0
        self.epoch = 0
        self.late: Dict[int, int] = {}       # frames published after their stamp, per epoch
        self.skipped = 0
        self.quit = threading.Event()
        self.frames_fed = 0                   # frames fed to Ditto, which is how it names those it draws
        self._since_voice = LIPS_RELEASE_FRAMES + 1   # frames since she was heard: past the release, at rest

    @classmethod
    def build(cls, portrait: str, framing: str) -> "Renderer":
        from . import engines, portrait as portraits, weights
        from .ditto.stream import Stream
        from .frames import FrameWriter

        # The portrait first: a portrait that cannot be read costs no download.
        rgb = portraits.load(portrait)
        root = weights.models_dir()
        if weights.missing(root):
            _log("fetching the face's model files")
            weights.fetch(root)
        warp_engine, decoder_engine = engines.ensure_engines(root)
        warp, decode = engines.warp_and_decode(warp_engine, decoder_engine)
        stream = Stream(str(root / weights.CONFIG), str(root / weights.DATA_ROOT))
        renderer = cls(stream, FrameWriter())
        stream.setup(rgb, warp, decode, renderer.on_frame, crop_scale=CROP[framing], sampling_timesteps=STEPS)
        return renderer

    # -- what Ditto hands back ------------------------------------------------
    def on_frame(self, rgb: np.ndarray) -> None:
        due = self.due.popleft()
        if not due.publish or due.epoch != self.epoch:
            return
        late = time.perf_counter() > due.stamp
        if self.frames.publish(rgb, due.stamp, due.generation) and late:
            self.late[due.epoch] = self.late.get(due.epoch, 0) + 1

    # -- what the daemon hands over -------------------------------------------
    def listen(self, stdin) -> None:
        """Read messages until quit, the end of the pipe, or one that cannot be read."""
        try:
            while True:
                head = stdin.read(HEADER.size)
                if len(head) < HEADER.size:
                    return
                kind, at, rate, size = HEADER.unpack(head)
                payload = stdin.read(size) if size else b""
                if len(payload) < size:
                    return   # the daemon went away mid-message
                if kind == MSG_AUDIO and size % 4 == 0:
                    with self.lock:
                        self.timeline.lay(at, np.frombuffer(payload, "<f4"), rate)
                elif kind == MSG_CUT:
                    with self.lock:
                        self.timeline.cut(at)
                        self.generation = self.frames.cut(at)
                elif kind == MSG_QUIT:
                    return
        except Exception as e:
            _log(f"reading from the daemon failed: {e}")
        finally:
            self.quit.set()

    # -- feeding --------------------------------------------------------------
    def _lips(self, audio: np.ndarray) -> list:
        """How much each new frame's lips follow her voice: 1 while she is
        heard, 0 in silence, where Ditto drawing from silence would press them
        in, and a ramp between so the pauses between words keep her mouth."""
        per = tl.RATE // tl.FPS
        past = tl.CHUNKSIZE[0]
        ahead = tl.CHUNK_FRAMES + tl.CHUNKSIZE[2]   # the chunk carries two frames of what follows
        voiced = []
        for k in range(ahead):
            frame = audio[(past + k) * per:(past + k + 1) * per]
            voiced.append(float(np.sqrt(np.mean(np.square(frame)))) > SILENCE_RMS)
        weights = []
        for k in range(tl.CHUNK_FRAMES):
            if voiced[k]:
                self._since_voice = 0
                weights.append(1.0)
                continue
            self._since_voice += 1
            release = max(0.0, 1.0 - self._since_voice / (LIPS_RELEASE_FRAMES + 1))
            soon = next((j for j in range(1, LIPS_ATTACK_FRAMES + 1)
                         if k + j < ahead and voiced[k + j]), None)
            attack = 1.0 - soon / (LIPS_ATTACK_FRAMES + 1) if soon else 0.0
            weights.append(max(release, attack))
        return weights

    def _feed(self, start: float, publish: bool) -> None:
        first, _end = self.feeder.window(start)
        with self.lock:
            audio = self.timeline.read(first, tl.CHUNK_SAMPLES)
            generation = self.generation
        for stamp, lips in zip(self.feeder.stamps(start), self._lips(audio)):
            self.due.append(_Due(stamp, generation, self.epoch, publish))
            if lips < 1.0:
                # Ditto names a frame by its place among those it draws.
                self.stream.ctrl_info[self.frames_fed] = {"vad_alpha": lips}
            self.frames_fed += 1
        self.stream.run_chunk(audio)
        self.feeder.fed(start)

    def _behind(self, now: float) -> bool:
        """Whether a frame Ditto still owes should already be on screen."""
        for due in self.due:
            if due.epoch == self.epoch and due.publish:
                return now > due.stamp + 1 / tl.FPS
        return False

    def _rest(self) -> None:
        """Feeding stops: frames still inside Ditto will not be shown."""
        late = self.late.pop(self.epoch, 0)
        if late or self.skipped:
            _log(f"before this rest: {late} frames after their moment, {self.skipped} chunks skipped")
        self.skipped = 0
        self.epoch += 1
        self._since_voice = LIPS_RELEASE_FRAMES + 1   # whoever watches next finds her lips at rest

    def tick(self, now: float) -> float:
        """One turn of the feeding loop; returns how long to sleep."""
        self.frames.beat(now)
        watched = self.frames.viewer_age(now) < WATCH_SECONDS
        if watched != self.watched:
            # The speech engine holds her voice back for the face only while someone watches.
            self.watched = watched
            self.on_event(event="watched", value=watched)
        was_feeding = self.feeder.next_start is not None
        with self.lock:
            step = self.feeder.step(now, watched)
            self.timeline.forget(now - KEEP_SECONDS)
        if step.feed is None:
            if was_feeding and self.feeder.next_start is None:
                self._rest()
            return min(step.wait, 0.05)
        if self._behind(now):
            # Ditto is behind: frames for this chunk would come too late to show.
            self.skipped += 1
            self.feeder.fed(step.feed)
            return 0.0
        self._feed(step.feed, publish=True)
        return 0.0

    def warm_up(self, seconds: float = WARM_UP_SECONDS) -> None:
        """Ditto's first seconds run slowly; they are spent on silence nobody sees."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            step = self.feeder.step(time.perf_counter(), True)
            if step.feed is not None:
                self._feed(step.feed, publish=False)
            else:
                time.sleep(min(step.wait, 0.05))
        # Ready means the next frame fed comes out on time, so what is still inside drains first.
        drain = time.perf_counter() + 10.0
        while len(self.due) > tl.CHUNK_FRAMES and time.perf_counter() < drain:
            time.sleep(0.02)
        self.feeder.next_start = None
        self.epoch += 1

    def run(self) -> None:
        while not self.quit.is_set():
            if self.stream.worker_exception is not None:
                raise RuntimeError(f"Ditto failed while drawing: {self.stream.worker_exception}")
            wait = self.tick(time.perf_counter())
            if wait:
                time.sleep(wait)

    def close(self) -> None:
        self.stream.stop()
        self.frames.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framing", choices=sorted(CROP), default="head")
    args = parser.parse_args(argv)
    # Events go to the real stdout; anything a library prints goes to stderr, in
    # UTF-8 because a pipe on Windows would refuse an emoji.
    events = sys.stdout
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr

    def say(**event) -> None:
        events.write(json.dumps(event) + "\n")
        events.flush()

    try:
        portrait = os.environ.get(PORTRAIT_ENV)
        if not portrait:
            raise RuntimeError("no portrait was named")
        _check_packages()
        renderer = Renderer.build(portrait, args.framing)
        renderer.warm_up()
        renderer.on_event = say
    except Exception as e:
        say(event="failed", reason=str(e) or type(e).__name__)
        return 1
    say(event="ready", delay=tl.DELAY)
    listener = threading.Thread(target=renderer.listen, args=(sys.stdin.buffer,), daemon=True, name="face-listen")
    listener.start()
    try:
        renderer.run()
    except Exception as e:
        say(event="failed", reason=str(e) or type(e).__name__)
        return 1
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
