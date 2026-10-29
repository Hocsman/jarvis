"""Check the talking face on this machine's GPU, end to end.

usage: python scripts/check_talking_face.py PORTRAIT WAV [--framing head|face] [--record OUT.mp4 --ffmpeg EXE]

Starts the real renderer the way the daemon does, hands it the WAV the way a
speech engine does (80 ms at a time, a little faster than real time, each piece
one delay before it is heard), and reads frames the way a front end does. It
reports whether every frame was on screen at its moment. With --record it also
writes what was shown, with the audio at the times it was heard, to an mp4.

The suite cannot run this: it needs the GPU, TensorRT and the model files.
"""

import argparse
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import soundfile as sf

from jarvis.output.face import timeline as tl
from jarvis.output.face.client import FaceRenderer
from jarvis.output.face.frames import FrameReader

PIECE = 0.08          # Kyutai hands over 80 ms at a time
SPEED = 1.25          # and makes it a little faster than it is played
JITTER = 0.15         # what a speech engine keeps in hand against a slow moment


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("portrait")
    p.add_argument("wav")
    p.add_argument("--framing", default="head")
    p.add_argument("--record")
    p.add_argument("--ffmpeg", default="ffmpeg")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    audio, rate = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(1)

    print("🎭 Talking face check", flush=True)
    renderer = FaceRenderer(args.portrait, args.framing)
    renderer.start()
    started = time.monotonic()
    while not renderer.ready():
        if renderer.failure:
            print(f"   ❌ {renderer.failure}", flush=True)
            renderer.stop()
            return 1
        time.sleep(0.2)
    print(f"   ✅ Ready in {time.monotonic() - started:.0f} s, delay {renderer.delay:.2f} s", flush=True)

    shown = []            # (stamp, seq, time shown)
    kept = []             # (stamp, jpeg) for --record
    stop = threading.Event()

    def front_end():
        import cv2

        reader = FrameReader()
        last = None
        while not stop.is_set():
            frame = reader.read()
            if frame is not None and frame.seq != last:
                shown.append((frame.stamp, frame.seq, time.perf_counter()))
                if args.record:
                    kept.append((frame.stamp, cv2.imencode(".jpg", cv2.cvtColor(frame.pixels, cv2.COLOR_RGB2BGR))[1]))
                last = frame.seq
            time.sleep(0.004)
        reader.close()

    viewer = threading.Thread(target=front_end, daemon=True)
    viewer.start()
    time.sleep(1.0)

    # The first piece is heard one delay and a jitter after it is handed over.
    heard0 = time.perf_counter() + renderer.delay + JITTER
    step = int(PIECE * rate)
    for i in range(0, len(audio), step):
        due = heard0 - renderer.delay - JITTER + (i / rate) / SPEED
        while time.perf_counter() < due:
            time.sleep(0.002)
        renderer.send_audio(heard0 + i / rate, audio[i:i + step], rate)
    end = heard0 + len(audio) / rate
    while time.perf_counter() < end + 1.5:
        time.sleep(0.05)
    stop.set()
    viewer.join()
    failure = renderer.failure
    renderer.stop()

    during = [(s, q, t) for s, q, t in shown if heard0 <= s <= end]
    lateness = np.array([t - s for s, _q, t in during]) * 1000
    seqs = [q for _s, q, _t in during]
    skipped = sum(b - a - 1 for a, b in zip(seqs, seqs[1:]) if b > a + 1)
    print("📊 While she spoke", flush=True)
    print(f"   🎞️  {len(during)} frames shown, {skipped} skipped", flush=True)
    if len(lateness):
        print(f"   ⏱️  shown {np.median(lateness):.1f} ms after their moment (p95 {np.percentile(lateness, 95):.1f}, "
              f"max {lateness.max():.1f})", flush=True)
        late = [s - heard0 for (s, _q, _t), ms in zip(during, lateness) if ms > 1000 / tl.FPS]
        if late:
            seconds = sorted({int(s) for s in late})
            print(f"   🐢 {len(late)} frames a frame or more late, in seconds {seconds} of the speech", flush=True)
    if failure:
        print(f"   ❌ {failure}", flush=True)

    if args.record and kept:
        import cv2

        with tempfile.TemporaryDirectory() as d:
            silent, wav = Path(d) / "v.mp4", Path(d) / "a.wav"
            first, last = kept[0][0], kept[-1][0]
            vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), tl.FPS, (512, 512))
            frames = iter(kept)
            nxt = next(frames)
            current = None
            for n in range(int((last - first) * tl.FPS) + 1):
                t = first + n / tl.FPS
                while nxt is not None and nxt[0] <= t + 1e-6:
                    current = cv2.imdecode(nxt[1], cv2.IMREAD_COLOR)
                    nxt = next(frames, None)
                vw.write(current)
            vw.release()
            track = np.zeros(int((last - first) * rate) + rate, np.float32)
            at = int((heard0 - first) * rate)
            if at >= 0:
                track[at:at + len(audio)] = audio[:len(track) - at]
            sf.write(wav, track, rate)
            subprocess.run([args.ffmpeg, "-loglevel", "error", "-y", "-i", str(silent), "-i", str(wav), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", args.record], check=True)
        print(f"   🎥 {args.record}", flush=True)
    return 0 if not failure else 1


if __name__ == "__main__":
    sys.exit(main())
