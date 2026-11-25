"""Whether his turn is over, read from his voice.

The model is Smart Turn v3.2 by Daily (BSD 2-Clause, code and weights):
an 8 MB int8 ONNX classifier that reads the last eight seconds of a turn,
as sound and not as text, and gives the probability that the speaker has
finished. It knows 23 languages and has no rule for any one of them.

Everything happens on this machine. The file comes from one pinned revision
of the Hugging Face repository, without a token, into a staging folder; it
is checked against its published SHA-256 there and moved into place only
when it matches, and a file at its final path is checked again before it is
loaded. After that first download nothing is sent anywhere.

The detector never raises and never blocks the listener: until the model is
loaded, and whenever it cannot be (no network on the first start, a file
that does not match, onnxruntime missing, a session that fails), it gives
no verdict, and the listener waits for its fixed silence.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..debug import debug_log

REPO = "pipecat-ai/smart-turn-v3"
REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
FILE = "smart-turn-v3.2-cpu.onnx"
SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
SIZE = 8679182

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
# Above this probability his turn is finished: the threshold the model's
# authors use.
FINISHED_ABOVE = 0.5
# How much of the silence that closed an utterance the model hears, in
# seconds. Measured on the same synthetic French speech, it tells a pause in
# the middle of a sentence apart more often than with the 0.4 s or 0.8 s the
# voice detector waits, for a few more finished sentences heard as unfinished
# (`scripts/turn_end_trial.py --closing`). A sentence cut short costs more
# than a longer wait.
CLOSING_SILENCE_HEARD = 0.2

# Whisper's log-mel spectrogram, as the model was trained on it.
_N_FFT = 400
_HOP = 160
_N_MELS = 80
_INPUT = "input_features"
# Threads the model may use. Two keep a verdict near 15 ms on a desktop CPU
# without taking cores from speech synthesis.
_THREADS = 2
_STAGING = ".staging"


def models_dir() -> Path:
    return Path("~/.local/share/jarvis/models/smart_turn").expanduser()


# ── The model file ────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _matches(path: Path) -> bool:
    return path.is_file() and path.stat().st_size == SIZE and _sha256(path) == SHA256


def fetch(root: Path, download: Optional[Callable[..., str]] = None) -> Path:
    """The model file in ``root``, downloaded when it is missing or does not
    match its published digest.

    Raises ``RuntimeError`` when the download fails or the file downloaded
    does not match. ``download`` defaults to ``huggingface_hub``'s, which
    writes under a temporary name and renames once complete.
    """
    final = root / FILE
    if _matches(final):
        return final
    if final.exists():
        debug_log("end of turn: the model file on disk does not match its digest, "
                  "fetching it again", "voice")
        final.unlink(missing_ok=True)
    if download is None:
        from huggingface_hub import hf_hub_download as download
    staging = root / _STAGING
    debug_log(f"end of turn: fetching {FILE} at {REVISION[:12]}", "voice")
    try:
        try:
            path = Path(download(repo_id=REPO, filename=FILE, revision=REVISION,
                                 local_dir=str(staging), token=False))
        except Exception as e:
            raise RuntimeError(f"could not download {FILE}: {e}") from e
        if not _matches(path):
            raise RuntimeError(f"{FILE} does not match its published digest")
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return final


# ── The audio the model reads ─────────────────────────────────────────


def last_window(sound: np.ndarray) -> np.ndarray:
    """The last ``WINDOW_SECONDS`` of *sound*, preceded by silence when it is
    shorter: the end of his turn always sits at the end of the window."""
    sound = np.asarray(sound, dtype=np.float32).reshape(-1)
    if sound.size >= WINDOW_SAMPLES:
        return sound[-WINDOW_SAMPLES:]
    return np.pad(sound, (WINDOW_SAMPLES - sound.size, 0))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Slaney's mel scale: linear below 1 kHz, logarithmic above."""
    hz = np.asarray(hz, dtype=np.float64)
    mel = 3.0 * hz / 200.0
    above = hz >= 1000.0
    mel[above] = 15.0 + np.log(hz[above] / 1000.0) * (27.0 / np.log(6.4))
    return mel


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    hz = 200.0 * mel / 3.0
    above = mel >= 15.0
    hz[above] = 1000.0 * np.exp(np.log(6.4) / 27.0 * (mel[above] - 15.0))
    return hz


def _mel_filters() -> np.ndarray:
    """Whisper's 80 triangular filters over the 201 bins of a 400-point FFT
    at 16 kHz, Slaney-normalised, as (bins, filters)."""
    bins = np.linspace(0, SAMPLE_RATE // 2, 1 + _N_FFT // 2)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(np.array([0.0]))[0],
                                   _hz_to_mel(np.array([8000.0]))[0], _N_MELS + 2))
    widths = np.diff(edges)
    slopes = edges[None, :] - bins[:, None]
    down = -slopes[:, :-2] / widths[:-1]
    up = slopes[:, 2:] / widths[1:]
    filters = np.maximum(0.0, np.minimum(down, up))
    return filters * (2.0 / (edges[2:] - edges[:-2]))[None, :]


_FILTERS = _mel_filters()
_WINDOW = np.hanning(_N_FFT + 1)[:-1]


def log_mel(sound: np.ndarray) -> np.ndarray:
    """The features the model reads for *sound* (16 kHz mono): the last
    window, normalised to zero mean and unit variance, as Whisper's log-mel
    spectrogram, (80, 800) float32."""
    x = last_window(sound).astype(np.float64)
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, _N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, _N_FFT)[::_HOP]
    power = np.abs(np.fft.rfft(frames * _WINDOW, axis=1)) ** 2
    mel = np.maximum(1e-10, power @ _FILTERS).T
    spec = np.log10(mel)[:, :-1]
    spec = np.maximum(spec, spec.max() - 8.0)
    return ((spec + 4.0) / 4.0).astype(np.float32)


# ── The detector ──────────────────────────────────────────────────────


def _open_onnx(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = _THREADS
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Idle threads sleep rather than spin between his utterances.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options,
                                providers=["CPUExecutionProvider"])


class TurnEndDetector:
    """Smart Turn on this machine. ``status`` is ``idle`` until loading
    starts, then ``loading``, then ``ready`` or ``unavailable`` (``why``
    says what went wrong)."""

    def __init__(self, root: Optional[Path] = None,
                 download: Optional[Callable[..., str]] = None,
                 open_session: Optional[Callable[[Path], object]] = None):
        self._root = root if root is not None else models_dir()
        self._download = download
        self._open_session = open_session or _open_onnx
        self._session = None
        self.status = "idle"
        self.why = ""

    def start(self) -> threading.Thread:
        """Fetch and load the model on a thread of its own."""
        self.status = "loading"
        thread = threading.Thread(target=self.load, name="turn-end-model", daemon=True)
        thread.start()
        return thread

    def load(self) -> bool:
        """Fetch the model if needed and open it. Never raises."""
        self.status = "loading"
        started = time.perf_counter()
        if not (self._root / FILE).is_file():
            try:
                print(f"     ⬇️  Downloading the end-of-turn model ({SIZE / 1e6:.1f} MB, once)",
                      flush=True)
            except Exception:
                pass
        try:
            session = self._open_session(fetch(self._root, self._download))
        except Exception as e:
            self.why = str(e) or type(e).__name__
            self.status = "unavailable"
            debug_log(f"end of turn: model unavailable ({self.why}), "
                      f"the fixed silence decides", "voice")
            return False
        self._session = session
        self.status = "ready"
        debug_log(f"end of turn: model ready in {(time.perf_counter() - started) * 1000:.0f} ms",
                  "voice")
        return True

    def probability(self, sound: Optional[np.ndarray]) -> Optional[float]:
        """How likely it is that his turn is over once *sound* (16 kHz mono,
        any length) has been said, or None when there is no verdict."""
        session = self._session
        if session is None or sound is None or np.size(sound) == 0:
            return None
        try:
            features = log_mel(sound)[None, ...]
            outputs = session.run(None, {_INPUT: features})
            probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        except Exception as e:
            debug_log(f"end of turn: no verdict ({type(e).__name__}: {e})", "voice")
            return None
        if not math.isfinite(probability):
            return None
        return probability
