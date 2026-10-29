"""The daemon's handle on the talking-face renderer (`face.spec.md`, "The process").

The renderer runs in a child process so that its model, TensorRT and the GPU
memory they hold can fail without touching the voice. This handle starts it,
hands it audio and cuts, and watches it; it never raises to its caller and
never blocks it, and any failure turns the face off for the session.
"""

import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ...debug import debug_log

# One message: kind, a time, a sample rate, the number of payload bytes.
HEADER = struct.Struct("<BdII")
MSG_AUDIO, MSG_CUT, MSG_QUIT = 1, 2, 3


class FaceRenderer:
    def __init__(self, portrait: str, framing: str = "head", command: Optional[List[str]] = None,
                 stall_seconds: float = 5.0) -> None:
        self._command = command or [sys.executable, "-m", "jarvis.output.face.process",
                                    "--portrait", portrait, "--framing", framing]
        self._stall_seconds = stall_seconds
        self._proc: Optional[subprocess.Popen] = None
        self._outbox: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._ready = False
        self._stopping = False
        self._writing_since: Optional[float] = None
        self.delay: Optional[float] = None
        self.failure: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if getattr(sys, "frozen", False):
            self._fail("the face runs from a source install only: the packaged app carries no TensorRT")
            return
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # The child imports jarvis from the same tree as this module.
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[3])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        try:
            self._proc = subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, bufsize=0, creationflags=flags, env=env)
        except Exception as e:
            self._fail(f"the renderer could not start: {e}")
            return
        debug_log("face: renderer started", "face")
        for target, name in ((self._write_loop, "face-send"), (self._read_events, "face-events"),
                             (self._read_log, "face-log")):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the renderer to quit, and make sure it is gone."""
        with self._lock:
            self._stopping = True
        if self._proc is not None and self._proc.poll() is None and self.failure is None:
            self._outbox.put(HEADER.pack(MSG_QUIT, 0.0, 0, 0))
        self._outbox.put(None)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=timeout)
        for t in self._threads:
            t.join(timeout=timeout)
        self._ready = False

    def ready(self) -> bool:
        """True between the renderer's ``ready`` line and the first failure."""
        since = self._writing_since
        if since is not None and time.monotonic() - since > self._stall_seconds:
            self._fail("the renderer stopped reading")
        return self._ready and self.failure is None

    # -- what the speech engine hands over ----------------------------------
    def send_audio(self, heard_at: float, samples, rate: int) -> None:
        if not self.ready():
            return
        payload = np.asarray(samples, dtype="<f4").tobytes()
        self._outbox.put(HEADER.pack(MSG_AUDIO, heard_at, rate, len(payload)) + payload)

    def cut(self, at: float) -> None:
        if not self.ready():
            return
        self._outbox.put(HEADER.pack(MSG_CUT, at, 0, 0))

    # -- workers -----------------------------------------------------------
    def _fail(self, reason: str) -> None:
        with self._lock:
            if self.failure is not None or self._stopping:
                return
            self.failure = reason
            self._ready = False
        debug_log(f"face: off for this session, {reason}", "face")
        print(f"  ⚠️ Face: {reason}", flush=True)
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._outbox.put(None)

    def _write_loop(self) -> None:
        while True:
            message = self._outbox.get()
            if message is None or self.failure is not None:
                break
            self._writing_since = time.monotonic()
            try:
                self._proc.stdin.write(message)
            except Exception as e:
                self._fail(f"the pipe to the renderer broke: {e}")
                break
            finally:
                self._writing_since = None
        try:
            self._proc.stdin.close()
        except Exception:
            pass

    def _read_events(self) -> None:
        for raw in self._proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            try:
                event = json.loads(line)
            except ValueError:
                debug_log(f"face renderer: {line}", "face")
                continue
            if event.get("event") == "ready":
                self.delay = float(event.get("delay", 0.0))
                self._ready = self.failure is None
                debug_log(f"face: renderer ready, delay {self.delay:.2f}s", "face")
            elif event.get("event") == "failed":
                self._fail(str(event.get("reason") or "the renderer failed"))
        code = self._proc.wait()
        self._fail(f"the renderer exited (code {code})")

    def _read_log(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                debug_log(f"face renderer: {line}", "face")
