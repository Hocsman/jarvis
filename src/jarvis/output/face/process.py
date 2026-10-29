"""The talking-face renderer: a child process of the daemon (`face.spec.md`).

usage: python -m jarvis.output.face.process --portrait PATH [--framing head|face]

It reports on stdout, one JSON object per line (``ready`` or ``failed``), reads
audio, cuts and quit from stdin (see ``client.HEADER``), and publishes frames
to shared memory (``frames.py``). Everything else goes to stderr.
"""

import argparse
import collections
import json
import sys
import threading
import time

import numpy as np

from . import timeline as tl
from .client import HEADER, MSG_AUDIO, MSG_CUT, MSG_QUIT

CROP = {"head": 2.8, "face": 2.3}
OVERLAP = 75          # five new frames per sampling, so a frame waits for fewer
STEPS = 10
WARM_UP_SECONDS = 3.0
WATCH_SECONDS = 2.0   # a viewer heartbeat younger than this means someone is watching
KEEP_SECONDS = 5.0    # audio heard longer ago than this is forgotten


def _say(**event) -> None:
    sys.__stdout__.write(json.dumps(event) + "\n")
    sys.__stdout__.flush()


def _log(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _check_packages() -> None:
    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"PyTorch cannot be loaded ({e})")
    if not torch.cuda.is_available():
        raise RuntimeError("the face needs an NVIDIA GPU with CUDA, and none is available to PyTorch")
    for module, package in (("tensorrt", "tensorrt-cu12"), ("cv2", "opencv-python-headless")):
        try:
            __import__(module)
        except Exception as e:
            raise RuntimeError(f"{package} is not installed ({e}); it is installed by hand, like moshi")


class Renderer:
    def __init__(self, portrait: str, framing: str) -> None:
        from . import engines, portrait as portraits, weights
        from .ditto.stream import Stream
        from .frames import FrameWriter

        root = weights.models_dir()
        if weights.missing(root):
            _log("fetching the face's model files")
            weights.fetch(root)
        rgb = portraits.load(portrait)
        data_root = root / weights.DATA_ROOT
        warp_engine, decoder_engine = engines.ensure_engines(
            root, str(data_root / "models" / "warp_network.pth"), str(data_root / "models" / "decoder.pth"))
        warp, decode = engines.warp_and_decode(warp_engine, decoder_engine)

        self.lock = threading.Lock()
        self.timeline = tl.Timeline()
        self.feeder = tl.Feeder(self.timeline)
        self.stamps = collections.deque()   # (stamp, generation, publish) per frame fed, in order
        self.generation = 0
        self.late = 0
        self.frames = FrameWriter()
        self.stream = Stream(str(root / weights.CONFIG), str(data_root))
        self.stream.setup(rgb, warp, decode, self._on_frame, crop_scale=CROP[framing],
                          overlap_v2=OVERLAP, sampling_timesteps=STEPS)
        self.quit = threading.Event()

    def _on_frame(self, rgb: np.ndarray) -> None:
        stamp, generation, publish = self.stamps.popleft()
        if not publish:
            return
        if time.perf_counter() > stamp:
            self.late += 1
        self.frames.publish(rgb, stamp, generation)

    def _feed(self, start: float, publish: bool) -> None:
        first, _end = self.feeder.window(start)
        with self.lock:
            audio = self.timeline.read(first, tl.CHUNK_SAMPLES)
            generation = self.generation
        for stamp in self.feeder.stamps(start):
            self.stamps.append((stamp, generation, publish))
        self.stream.run_chunk(audio)
        self.feeder.fed(start)

    def warm_up(self) -> None:
        """Ditto's first seconds run slowly; they are spent on silence nobody sees."""
        end = time.perf_counter() + WARM_UP_SECONDS
        while time.perf_counter() < end:
            step = self.feeder.step(time.perf_counter(), True)
            if step.feed is not None:
                self._feed(step.feed, publish=False)
            else:
                time.sleep(min(step.wait, 0.05))

    def listen(self) -> None:
        """Read messages from the daemon until quit or the pipe closes."""
        stdin = sys.stdin.buffer
        while True:
            head = stdin.read(HEADER.size)
            if len(head) < HEADER.size:
                break
            kind, at, rate, size = HEADER.unpack(head)
            payload = stdin.read(size) if size else b""
            if kind == MSG_AUDIO:
                with self.lock:
                    self.timeline.lay(at, np.frombuffer(payload, "<f4"), rate)
            elif kind == MSG_CUT:
                with self.lock:
                    self.timeline.cut(at)
                    self.generation = self.frames.cut(at)
            elif kind == MSG_QUIT:
                break
        self.quit.set()

    def run(self) -> None:
        resting = True
        while not self.quit.is_set():
            if self.stream.worker_exception is not None:
                raise RuntimeError(f"Ditto failed while drawing: {self.stream.worker_exception}")
            now = time.perf_counter()
            self.frames.beat(now)
            watched = self.frames.viewer_age(now) < WATCH_SECONDS
            with self.lock:
                step = self.feeder.step(now, watched)
                self.timeline.forget(now - KEEP_SECONDS)
            if step.feed is not None:
                if resting:
                    resting = False
                    self.late = 0
                self._feed(step.feed, publish=True)
                continue
            if self.feeder.next_start is None and not resting:
                resting = True
                if self.late:
                    _log(f"{self.late} frames came after their moment before this rest")
            time.sleep(min(step.wait, 0.05))

    def close(self) -> None:
        self.stream.stop()
        self.frames.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portrait", required=True)
    parser.add_argument("--framing", choices=sorted(CROP), default="head")
    args = parser.parse_args(argv)
    # Anything a library prints must not land among the events, and a pipe on
    # Windows would refuse the emoji in the progress lines.
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    sys.stdout = sys.stderr

    try:
        _check_packages()
        renderer = Renderer(args.portrait, args.framing)
        renderer.warm_up()
    except Exception as e:
        _say(event="failed", reason=str(e) or type(e).__name__)
        return 1
    _say(event="ready", delay=tl.DELAY)
    listener = threading.Thread(target=renderer.listen, daemon=True, name="face-listen")
    listener.start()
    try:
        renderer.run()
    except Exception as e:
        _say(event="failed", reason=str(e) or type(e).__name__)
        return 1
    finally:
        renderer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
