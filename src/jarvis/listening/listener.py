"""
Voice Listener - Main orchestrator for voice capture and processing.

Coordinates audio capture, speech recognition, echo detection, and state management.
"""

from __future__ import annotations
import functools
import os
import threading
import time
import queue
import sys
import platform
from collections import deque
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, TYPE_CHECKING, Any
from datetime import datetime

from rapidfuzz import fuzz
from .audio_capture import UtteranceCapture
from .echo_detection import EchoDetector
from .room_floor import FALLBACK_FLOOR, ROOM_FLOOR_MIN_SECONDS, RoomFloor
from .state_manager import StateManager, ListeningState
from ..utils.audio_lock import open_stream_count, portaudio_lock, unguarded_open
from .wake_detection import is_wake_word_detected, extract_query_after_wake, is_stop_command
from .transcript_buffer import TranscriptBuffer, TranscriptSegment
from .whisper_turbo import is_faster_whisper_turbo_supported
from .whisper_vad import (
    detector_failed as voice_detector_failed, prepare_voice_detector,
    transcribe as transcribe_faster_whisper,
)
from .intent_judge import (
    IntentJudge,
    _is_low_power_mode_enabled,
    _ollama_keep_alive_for_power_mode,
    create_intent_judge,
    warm_up_chat_model,
)
from ..debug import debug_log
from ..latency import get_latency_journal
from ..state import JarvisState, get_jarvis_state
from ..utils.location import is_location_available

if TYPE_CHECKING:
    from ..memory.db import Database
    from ..memory.conversation import DialogueMemory

from ..utils.audio_stream import (
    DeviceListUnavailable,
    IndexNamesNoInput,
    InputDeviceNotFound,
    is_input_format_error,
    open_input_stream,
    refresh_input_devices,
    resolve_input_device,
)

def _open_input_stream(
    sample_rate: int,
    frame_ms: int,
    device_kwargs: dict[str, Any],
    *,
    callback: Optional[Callable] = None,
    serialise: bool = True,
) -> tuple[Any, int, int]:
    """Open mono first, then bounded native-rate/channel alternatives on the same input."""
    return open_input_stream(
        sample_rate,
        frame_ms,
        device_kwargs,
        callback=callback,
        serialise=serialise,
        sd_backend=sd,
        lock=portaudio_lock,
    )


# How much microphone audio the queue between the capture callback and the
# listening loop holds. The loop still waits on Whisper and on the intent
# judge; while it does, his words wait here instead of being dropped.
AUDIO_BACKLOG_SECONDS = 30.0

# While she only thinks, a stop word counts as a stop in an utterance of at
# most this many words; longer, it is a sentence that happens to hold one.
STOP_WORDS_WHILE_THINKING = 3

# Questions asked while a reply is on its way wait their turn, this many at
# most: past it the oldest is dropped, so a wake word repeated by a film or
# a video cannot stack up replies that would run minutes later.
MAX_WAITING_TURNS = 2

# While the selected microphone is missing, PortAudio's device list is read
# afresh every MICROPHONE_RESCAN_SECONDS for the first
# MICROPHONE_RESCAN_FAST_FOR_SECONDS of the wait, then every
# MICROPHONE_RESCAN_SLOW_SECONDS. A new loss, or a dictation press that misses
# the microphone, makes the readings frequent again.
MICROPHONE_RESCAN_SECONDS = 2.5
MICROPHONE_RESCAN_FAST_FOR_SECONDS = 60.0
MICROPHONE_RESCAN_SLOW_SECONDS = 15.0

# A reading of the device list held back this long, in seconds, by streams
# still open is said once on the console.
HELD_BACK_NOTICE_SECONDS = 60.0

# A stream that dies sooner than this, in seconds, after it opened waits for
# the scheduled reading of the device list rather than one at once, so a
# device that keeps dying is not reopened once a second.
STREAM_SETTLED_SECONDS = 5.0

# No microphone callback for this long, in seconds, and the input stream is
# taken for dead: closed, waited for and opened again. Twice the capture-health
# warning's five seconds, so a stall long enough to be reported is not yet
# enough to cycle the stream.
CAPTURE_LOST_SECONDS = 10.0

# How often, in seconds, the input stream is checked for death.
CAPTURE_CHECK_SECONDS = 1.0


class _Permission(Enum):
    """What the Windows microphone permission probe found."""

    OK = "ok"
    # Windows blocks the microphone: listening ends.
    BLOCKED = "blocked"
    # It failed for another reason: the microphone is waited for and probed again.
    FAILED = "failed"


def _reads_as_access_denied(error: Exception) -> bool:
    """Whether a failure to open or start the microphone says access is
    denied, which does not go away on its own."""
    message = str(error).lower()
    return "access" in message or "permission" in message


def _print_access_blocked_box() -> None:
    """What to change when Windows blocks the microphone."""
    print("", flush=True)
    print("  ┌─────────────────────────────────────────────────────────┐", flush=True)
    print("  │  🔒 MICROPHONE ACCESS BLOCKED BY WINDOWS               │", flush=True)
    print("  │                                                         │", flush=True)
    print("  │  To fix this:                                          │", flush=True)
    print("  │  1. Open Windows Settings                              │", flush=True)
    print("  │  2. Go to Privacy & security → Microphone              │", flush=True)
    print("  │  3. Turn ON 'Microphone access'                        │", flush=True)
    print("  │  4. Turn ON 'Let apps access your microphone'          │", flush=True)
    print("  │  5. Turn ON 'Let desktop apps access your microphone'  │", flush=True)
    print("  │                                                         │", flush=True)
    print("  │  Then restart Jarvis.                                  │", flush=True)
    print("  └─────────────────────────────────────────────────────────┘", flush=True)
    print("", flush=True)


def _vad_frame_ms(cfg) -> int:
    """The VAD frame length: 10, 20 or 30 ms, 20 for anything else."""
    frame_ms = int(getattr(cfg, "vad_frame_ms", 20))
    if frame_ms in (10, 20, 30):
        return frame_ms
    debug_log(f"Unsupported VAD frame duration {frame_ms}; using 20 ms", "voice")
    return 20


@dataclass
class _Turn:
    """One query handed to the reply thread, with what was true when it was
    handed over: the language he spoke, his exact words, when he stopped
    speaking, and the stop count. A stop counted after it means it is
    never said."""
    query: str
    language: Optional[str] = None
    heard: Optional[str] = None
    heard_at: float = 0.0
    epoch: int = 0


@dataclass
class _AskedQuestion:
    """A spoken question one of her replies ended on, and when he finished
    hearing it (None until he has)."""
    request_id: str
    turn: _Turn
    heard_at: Optional[float] = None

    def heard(self) -> None:
        self.heard_at = time.time()


def is_whisper_hallucination(no_speech_prob: float, threshold: float) -> bool:
    """Shared Whisper no-speech gate.

    Whisper can report high `avg_logprob` confidence on hallucinated phrases
    when the audio is silent or noise. `no_speech_prob` is an independent
    signal and must be checked first. Used by both the faster-whisper path
    (`_filter_noisy_segments`) and the MLX path (`_finalize_utterance`) so
    both backends apply identical policy.
    """
    return no_speech_prob >= threshold

# Audio processing imports (optional)
try:
    import sounddevice as sd
    import webrtcvad
    import numpy as np
except ImportError as e:
    sd = None
    webrtcvad = None
    np = None
    # Log import error for debugging
    print(f"  ⚠️  Audio import error: {e}", flush=True)
    print("     This may indicate PortAudio is not found", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys
except OSError as e:
    # PortAudio loading errors appear as OSError
    sd = None
    webrtcvad = None
    np = None
    print(f"  ❌ PortAudio initialisation failed: {e}", flush=True)
    print("     Please reinstall the application or check audio drivers", flush=True)
    import sys as _sys
    if _sys.platform == 'linux':
        print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
    del _sys

# Whisper backend imports - try MLX first on Apple Silicon, fall back to faster-whisper
MLX_WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _get_mic_permission_hint() -> str:
    """Return platform-appropriate microphone permission guidance."""
    if sys.platform == 'win32':
        return "Windows Settings > Privacy > Microphone > Allow apps to access"
    elif sys.platform == 'darwin':
        return "System Settings > Privacy & Security > Microphone"
    else:
        return "`pactl list sources` or audio settings for your desktop environment"

def _resample(audio, src_rate: int, dst_rate: int):
    """Resample a 1-D float32 numpy array from *src_rate* to *dst_rate*.

    Uses linear interpolation — fast and good enough for speech going into Whisper.
    """
    if src_rate == dst_rate or np is None:
        return audio
    ratio = dst_rate / src_rate
    n_out = round(len(audio) * ratio)
    indices = np.arange(n_out) / ratio
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def _setup_nvidia_dll_path() -> None:
    """Add NVIDIA CUDA DLL directories to PATH on Windows.

    The pip packages nvidia-cublas-cu12 and nvidia-cudnn-cu12 install DLLs
    under site-packages/nvidia/*/bin/ which isn't on PATH by default.
    PyInstaller bundles place them in {app}/cuda/. This function finds
    both locations and prepends them to PATH so ctypes.CDLL can find them.
    """
    import os

    dirs_to_add = []

    # 1. Check for NVIDIA pip packages in site-packages
    try:
        import nvidia.cublas  # type: ignore[import-untyped]
        for pkg_path in nvidia.cublas.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    try:
        import nvidia.cudnn  # type: ignore[import-untyped]
        for pkg_path in nvidia.cudnn.__path__:
            bin_dir = os.path.join(pkg_path, "bin")
            if os.path.isdir(bin_dir):
                dirs_to_add.append(bin_dir)
    except (ImportError, AttributeError):
        pass

    # 2. Check for CUDA DLLs in app directory (installed by install_cuda.ps1)
    # For frozen apps: check next to the executable (not _MEIPASS, since
    # CUDA libs are downloaded post-install, not bundled in the archive)
    if getattr(sys, "frozen", False):
        app_dir = os.path.dirname(sys.executable)
    else:
        app_dir = None

    if app_dir:
        cuda_dir = os.path.join(app_dir, "cuda")
        if os.path.isdir(cuda_dir):
            dirs_to_add.append(cuda_dir)

    # 3. Register DLL directories (must happen before ctypes.CDLL probes)
    # Use both os.add_dll_directory (for ctypes.CDLL) and PATH (for
    # subprocess/child processes). On Windows, PATH changes after process
    # start don't affect ctypes.CDLL search — add_dll_directory is needed.
    if dirs_to_add:
        current_path = os.environ.get("PATH", "")
        new_entries = os.pathsep.join(dirs_to_add)
        os.environ["PATH"] = new_entries + os.pathsep + current_path
        for d in dirs_to_add:
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            debug_log(f"added NVIDIA DLL path: {d}", "voice")


@functools.lru_cache(maxsize=None)
def _probe_cuda_available() -> tuple[bool, list[str]]:
    """Probe cuBLAS + cuDNN availability once per process and cache the result.

    The version ranges intentionally span more than the currently pinned
    versions in `installer/windows/install_cuda.ps1` (`cublas64_12.dll`,
    `cudnn_ops64_9.dll`) so a future installer bump doesn't silently fall
    back to CPU until this probe is updated too. A bump outside the
    existing range still requires widening these ranges — the relationship
    is by convention, not enforced.

    Cached because DLLs don't appear or disappear while the process is
    running, and the scan does up to 18 `LoadLibrary` calls on a miss.
    """
    _setup_nvidia_dll_path()

    missing_libs: list[str] = []
    cublas_found = False
    cudnn_found = False
    try:
        import ctypes

        for ver in range(20, 10, -1):
            try:
                ctypes.CDLL(f"cublas64_{ver}.dll")
                cublas_found = True
                debug_log(f"cuBLAS found (cublas64_{ver}.dll)", "voice")
                break
            except OSError:
                continue
        if not cublas_found:
            missing_libs.append("cuBLAS")

        for ver in range(15, 7, -1):
            try:
                ctypes.CDLL(f"cudnn_ops64_{ver}.dll")
                cudnn_found = True
                debug_log(f"cuDNN found (cudnn_ops64_{ver}.dll)", "voice")
                break
            except OSError:
                continue
        if not cudnn_found:
            missing_libs.append("cuDNN")
    except Exception as e:
        debug_log(f"CUDA library probe failed: {e}", "voice")

    return cublas_found and cudnn_found, missing_libs


def _probe_windows_cuda_libraries(device: str) -> tuple[str, list[str]]:
    """Return the device to use and any missing CUDA lib names.

    Short-circuits on non-Windows or non-CUDA device strings. Otherwise
    delegates to the cached `_probe_cuda_available()` so the expensive DLL
    scan only runs once per process lifetime.
    """
    if sys.platform != "win32" or device not in ("auto", "cuda"):
        return device, []

    available, missing_libs = _probe_cuda_available()
    if not available:
        return "cpu", missing_libs
    return device, []


def _print_cuda_unavailable_hint(missing_libs: list[str]) -> None:
    """Print the user-facing CUDA-missing message and recovery hint.

    The hint deliberately points at the tray action, not at "reinstall the
    app". The Inno Setup task only fires once and skips on stale marker
    files, so reinstalling without first deleting `{app}\\cuda` rarely
    fixes the underlying problem. The tray action re-runs install_cuda.ps1
    directly with UAC, which is the actual recovery path.
    """
    debug_log(f"CUDA libraries missing: {missing_libs}, forcing CPU mode", "voice")
    print("  ℹ️  CUDA not available, using CPU mode", flush=True)
    if missing_libs:
        print(f"     Missing: {', '.join(missing_libs)}", flush=True)
    print(
        "  💡 For GPU acceleration, click 'Reinstall GPU libraries' in the Jarvis tray menu",
        flush=True,
    )


try:
    if _is_apple_silicon():
        import mlx_whisper
        MLX_WHISPER_AVAILABLE = True
except Exception:
    mlx_whisper = None

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    # Catch broad: the faster-whisper import chain can raise ValueError
    # (e.g. "psutil.__spec__ is not set") in some environments.
    WhisperModel = None


# A heard word counts as one she said when it matches any word of her
# reply this closely. Deliberately loose, because ASR mangles echo:
# 'perlet' for 'perplexe' scores 71, 'sahas' for 'saas' 89. Measured on
# the day's real transcripts, the aggregate below is unchanged whether
# this sits at 70, 80 or 90 — the design does not rest on the number.
_ECHO_WORD_MATCH = 80

# Above this share of unmatched words, the utterance carries something
# she did not say. The two real echoes measured 0% and 18%; barge-in
# measured 62% and 100%. This sits in the gap.
_SPEECH_SHARE = 0.4


def _carries_speech_she_did_not_say(heard: str, spoken: str) -> bool:
    """True when enough of the utterance has no counterpart in her reply.

    The echo override needs proof that the user said something, and its
    old test was the *absence* of proof that they had not: `partial_ratio`
    below a bar. That ratio looks for a contiguous window, so a decimated
    echo — words dropped from the middle, others corrupted — stops
    matching and reads as new content. A similarity measure that failed to
    find its match is not evidence.

    Per word nothing separates the two ('même' against 'semaine' scores
    the same as a genuinely new word). Over the whole utterance it
    separates cleanly, so the share is what decides.

    Fails open: with nothing of hers to compare against, everything is
    the user's.
    """
    import re

    mots = re.findall(r"\w+", (heard or "").lower(), re.UNICODE)
    if not mots:
        return False
    siens = re.findall(r"\w+", (spoken or "").lower(), re.UNICODE)
    if not siens:
        return True

    neufs = sum(1 for m in mots if not _her_word(m, siens))
    return (neufs / len(mots)) > _SPEECH_SHARE


def _her_word(word: str, her_words: list) -> bool:
    """Whether *word* matches a word of her reply as closely as
    `_ECHO_WORD_MATCH` asks."""
    return max((fuzz.ratio(word, s) for s in her_words), default=0) >= _ECHO_WORD_MATCH


def _tokens_she_did_not_say(heard: str, spoken: str) -> list:
    """The tokens of *heard*, split on whitespace as the wake-word check
    splits it, that hold a word she did not say in her reply *spoken*: all
    of them when there is nothing of hers to compare against."""
    import re

    siens = re.findall(r"\w+", (spoken or "").lower(), re.UNICODE)
    tokens = (heard or "").lower().split()
    if not siens:
        return tokens
    return [t for t in tokens
            if not all(_her_word(m, siens) for m in re.findall(r"\w+", t, re.UNICODE))]


# A piece of an utterance is a sentence of her reply coming back when it is
# one run of that reply this closely. Measured piece by piece on two days of
# real transcripts, hers scored 87 and above and his 65 and below;
# corrections of his written in her own wording for the measure, 70 to 81.
_HER_SENTENCE_MATCH = 85

# Under this many words a piece is never hers: three words in a row are few
# enough for him to say by chance after a long reply of hers ("à la
# maison"), and what is taken out cannot be recovered.
_HER_SENTENCE_MIN_WORDS = 4


def _her_sentence(piece: str, spoken: str) -> bool:
    """Whether *piece*, one of the segments Whisper made of an utterance, is
    a sentence of her reply *spoken* coming back: long enough to tell, and
    one run of her reply.

    The run is read in one direction only. `partial_ratio` lays the shorter
    text along the longer, so against a reply shorter than the piece it
    would ask whether her reply is a run of the piece, and a segment
    holding her whole reply and then words of his would pass. Such a piece
    is weighed whole against her reply (`ratio`).
    """
    if len(piece.split()) < _HER_SENTENCE_MIN_WORDS:
        return False
    heard, said = piece.strip().lower(), spoken.strip().lower()
    run = fuzz.partial_ratio if len(heard) <= len(said) else fuzz.ratio
    return run(heard, said) >= _HER_SENTENCE_MATCH


def _without_her_sentences_at_the_edges(pieces: list, spoken: str, min_words: int) -> tuple:
    """*pieces*, the segments Whisper made of one utterance heard in her
    time, without the sentences of her reply *spoken* that open and close
    it. Returns what is kept and what was taken out.

    Nothing is taken out when what would be left is under *min_words* words
    or carries no words she did not say: a word or two after her sentences
    is what Whisper makes of the quiet after her voice, and split from them
    it would stand alone as words she never said. A sentence of hers between
    two of his stays where it is.
    """
    pieces = [piece for piece in pieces if piece and piece.strip()]
    if len(pieces) < 2 or not (spoken or "").strip():
        return pieces, []
    first, last = 0, len(pieces)
    while first < last and _her_sentence(pieces[first], spoken):
        first += 1
    while last > first and _her_sentence(pieces[last - 1], spoken):
        last -= 1
    his, hers = pieces[first:last], pieces[:first] + pieces[last:]
    left = " ".join(his)
    if (not hers or len(left.split()) < min_words
            or not _carries_speech_she_did_not_say(left, spoken)):
        return pieces, []
    return his, hers


_TRUNCATION_MARKS = ("...", "…")


def _query_is_an_unfinished_fragment(heard: str, query: str) -> bool:
    """True when the judge handed back the tail of a sentence still being said.

    A minimum-length guard cannot do this job — it is the wrong axis. The
    fragment observed in the field, "Quel est pour toi", is four words and
    seventeen characters; every legitimate short query is shorter, and
    "tell me more" is one the judge's own prompt names as valid. Any
    threshold catching the fragment kills those.

    Truncation is the signal, and Whisper writes it: the transcript ends
    in an ellipsis. On its own that is not enough, because the judge is
    told to resolve a topic-less question against earlier segments, and
    when it does the query is a new sentence rather than a slice of this
    one. So both have to hold: the utterance was cut off, and the query is
    still just a piece of it.

    Punctuation rather than vocabulary, so it does not privilege a
    language. Fails open: no ellipsis, no opinion.
    """
    import re

    heard = (heard or "").strip()
    query = (query or "").strip()
    if not heard or not query:
        return False
    if not heard.endswith(_TRUNCATION_MARKS):
        return False
    plat = re.sub(r"\s+", " ", heard.rstrip("… .").lower())
    return re.sub(r"\s+", " ", query.lower()) in plat


def _get_mlx_model_repo(model_name: str) -> str:
    """Get the MLX Community HuggingFace repo for a Whisper model."""
    # Map standard model names to MLX Community repos
    model_map = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "tiny.en": "mlx-community/whisper-tiny.en-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "base.en": "mlx-community/whisper-base.en-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "small.en": "mlx-community/whisper-small.en-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "medium.en": "mlx-community/whisper-medium.en-mlx",
        "large": "mlx-community/whisper-large-v3-mlx",
        "large-v2": "mlx-community/whisper-large-v2-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }
    return model_map.get(model_name, f"mlx-community/whisper-{model_name}-mlx")


# faster-whisper fetches exactly these files (faster_whisper/utils.py).
_WHISPER_ALLOW_PATTERNS = [
    "config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*",
]


def _predownload_whisper_snapshot(model_name: str) -> None:
    """Best-effort visible download of the faster-whisper snapshot.

    faster-whisper silences huggingface_hub's progress bars, and the desktop
    app never shows a TTY, so a multi-GB first-run download would print
    nothing for minutes. This surfaces progress. Failures are deliberately
    swallowed: WhisperModel's own cache/429/offline paths handle them.
    """
    try:
        from faster_whisper.utils import _MODELS as fw_models
        repo_id = fw_models.get(model_name, f"Systran/faster-whisper-{model_name}")
    except Exception:
        repo_id = f"Systran/faster-whisper-{model_name}"
    try:
        from ..utils.hf_download import download_snapshot_with_progress
        download_snapshot_with_progress(
            repo_id, _WHISPER_ALLOW_PATTERNS, f"Whisper '{model_name}'",
        )
    except Exception as e:
        debug_log(f"Whisper pre-download incomplete, WhisperModel takes over: {e}", "voice")


def _predownload_mlx_snapshot(model_name: str, repo_id: str) -> None:
    """Same visible download for the MLX path, which downloads inside
    mlx_whisper.transcribe with the same silence problem."""
    try:
        from ..utils.hf_download import download_snapshot_with_progress
        download_snapshot_with_progress(repo_id, None, f"MLX Whisper '{model_name}'")
    except Exception as e:
        debug_log(f"MLX Whisper pre-download incomplete, transcribe takes over: {e}", "voice")


def _clear_corrupted_whisper_cache(error_message: str) -> bool:
    """Clear a corrupted Whisper model cache directory.

    Parses the CTranslate2 error message to find the snapshot directory,
    then deletes the parent ``models--`` directory so the model can be
    re-downloaded cleanly (including blobs that may also be corrupt).

    Returns ``True`` if a cache directory was found and deleted.
    """
    import re
    import shutil

    # CTranslate2 error format:
    #   "Unable to open file 'model.bin' in model '/path/to/snapshots/hash'"
    match = re.search(
        r"unable to open file\s+'[^']+'\s+in model\s+'([^']+)'",
        error_message,
        re.IGNORECASE,
    )
    if not match:
        debug_log("could not parse cache path from error message", "voice")
        return False

    snapshot_path = match.group(1)

    # Walk up to the models-- directory
    # snapshot_path is e.g. .../models--Org--Name/snapshots/<hash>
    # We want to delete .../models--Org--Name entirely
    from pathlib import Path
    path = Path(snapshot_path)
    model_dir = None
    for parent in [path] + list(path.parents):
        if parent.name.startswith("models--"):
            model_dir = parent
            break

    if model_dir is None or not model_dir.is_dir():
        debug_log(f"could not locate models-- cache directory from: {snapshot_path}", "voice")
        return False

    try:
        shutil.rmtree(model_dir)
        debug_log(f"cleared corrupted Whisper cache: {model_dir}", "voice")
        return True
    except OSError as e:
        debug_log(f"failed to clear corrupted cache: {e}", "voice")
        return False


@contextmanager
def _serialised_stream(stream):
    """Like ``with stream:`` but with lifecycle calls under portaudio_lock.

    sounddevice's context manager calls start() on enter and stop()/close()
    on exit; those are the thread-unsafe PortAudio lifecycle operations that
    must be serialised process-wide (see jarvis.utils.audio_lock).
    """
    try:
        with portaudio_lock:
            stream.start()
        yield stream
    finally:
        with portaudio_lock:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass


class VoiceListener(threading.Thread):
    """Main voice listening thread that orchestrates all voice processing."""

    def __init__(self, db: "Database", cfg, tts: Optional[Any],
                 dialogue_memory: "DialogueMemory"):
        """
        Initialise voice listener.

        Args:
            db: Database instance for storage
            cfg: Configuration object
            tts: Text-to-speech engine (optional)
            dialogue_memory: Dialogue memory instance
        """
        super().__init__(daemon=True)

        self.db = db
        self.cfg = cfg
        # The journal times each spoken exchange from here: durations only,
        # never text (latency.spec.md).
        get_latency_journal().configure(db, cfg)
        self.tts = tts
        self.dialogue_memory = dialogue_memory
        self._should_stop = False
        self._dictation_active = False  # Pause flag set by dictation engine
        self._first_utterance = True  # Suppress turn separator before the very first transcription
        # ISO-639-1 code Whisper detected for the most recent utterance.
        # Updated at every successful transcription site (MLX + faster-
        # whisper) and read when a query is handed to the reply thread, so
        # downstream tools can pick locale-appropriate resources (e.g.
        # tr.wikipedia.org). The turn keeps its own copy: the loop goes on
        # transcribing while the reply is prepared.
        self._last_detected_language: Optional[str] = None

        # Audio processing components
        self._whisper_backend: Optional[str] = None  # "mlx" or "faster-whisper"
        self._whisper_device: Optional[str] = None  # "cpu" or "cuda" (resolved from CTranslate2)
        self._mlx_model_repo: Optional[str] = None  # For MLX backend
        self.model: Optional[Any] = None  # WhisperModel for faster-whisper, None for MLX
        self.transcribe_lock = threading.Lock()  # Shared lock for Whisper model access
        # Entries are (time captured, block). Blocks are one VAD frame long.
        self._audio_q: queue.Queue = queue.Queue(
            maxsize=int(AUDIO_BACKLOG_SECONDS * 1000 / _vad_frame_ms(cfg)))
        self._pre_roll: deque = deque()
        # When the frame being read was captured: the clock an utterance's
        # start and end are read from, however late the loop reads it.
        self._frame_captured_at = 0.0

        # The reply thread: runs one handed-over query at a time, and is the
        # only thread that speaks.
        self._turns: queue.Queue = queue.Queue()
        self._turn_in_hand: Optional[_Turn] = None
        self._replies: Optional[threading.Thread] = None
        # Turns handed over and not yet finished, the one in hand included.
        self._replies_owed = 0
        # Each stop counts one. A turn handed over before the count moved
        # is never said.
        self._stop_epoch = 0
        # The count when the latest turn was handed over. Turns run in the
        # order they were handed over, so while any is owed the latest one
        # is, and a stop since it was handed over leaves none to be said.
        self._latest_turn_epoch = 0
        # When the last utterance he was heard saying ended.
        self._last_heard_at = 0.0
        # The utterance being handled, whole, and when it ended. It becomes
        # his last words only if it is collected.
        self._hearing: tuple = (None, 0.0)
        # The spoken question her last reply ended on, if it did.
        self._asked: Optional[_AskedQuestion] = None
        # The loop starts the tune and the reply thread stops it: a player
        # replaced between the two would play on with nothing to stop it.
        self._tune_lock = threading.Lock()
        # Held while a sentence of a turn is handed to the speech engine,
        # while a turn is stopped and while the turns owed are counted, so
        # no sentence slips out after a stop.
        self._speech_lock = threading.Lock()

        # Audio callback monitoring (for debugging)
        self._callback_count = 0
        self._last_callback_log_time = 0
        self._pending_audio = None
        self._vad_error_logged = False
        self._reset_audio_health()

        # Set while the microphone is waited for: "missing" when it was not
        # connected, "lost" when its stream died or it would not open or
        # start. Cleared, with a console line, once audio arrives from it again.
        self._microphone_wait: Optional[str] = None
        # When the device list is next read afresh while waiting, and since
        # when the readings are frequent (monotonic).
        self._next_rescan_at = 0.0
        self._rescans_hurried_at = 0.0
        # Since when the readings have been held back by open streams
        # (monotonic), and whether the console was told.
        self._held_back_since: Optional[float] = None
        self._held_back_said = False
        # What has been said, on the console or in the log, since start-up
        # or since the wait began: each is said once (`_first_time`).
        self._said: set = set()

        # Voice activity detection
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames: list = []
        self._samplerate = int(getattr(self.cfg, "sample_rate", 16000))
        self._configure_frames()
        self._vad: Optional = None

        # Off unless he set the environment variable, and it says so when
        # it is on. Announced here rather than at stream-open so a failure
        # to query the audio device cannot swallow the notice.
        self._input_device_name = ""
        # How much of the current reply streaming already spoke. Read by
        # `_speak_reply` to know whether it has a tail to say or a whole
        # reply, and reset when the turn ends.
        self._streamed_chars = 0
        self._capture = UtteranceCapture.from_env()
        self._capture.announce()

        # Initialise VAD if available
        if webrtcvad is not None and bool(getattr(self.cfg, "vad_enabled", True)):
            try:
                self._vad = webrtcvad.Vad(int(getattr(self.cfg, "vad_aggressiveness", 2)))
            except Exception:
                self._vad = None

        # Initialise modular components
        self.echo_detector = EchoDetector(
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            energy_spike_threshold=float(getattr(self.cfg, "echo_energy_threshold", 2.0))
        )

        self.state_manager = StateManager(
            hot_window_seconds=float(getattr(self.cfg, "hot_window_seconds", 3.0)),
            echo_tolerance=float(getattr(self.cfg, "echo_tolerance", 0.3)),
            voice_collect_seconds=float(getattr(self.cfg, "voice_collect_seconds", 2.0)),
            max_collect_seconds=float(getattr(self.cfg, "voice_max_collect_seconds", 60.0))
        )

        # The room's quiet level, the floor of the echo detector's voice proof
        self._room_floor = RoomFloor(frame_ms=self._frame_ms)

        # Audio-level wake word detection timestamp
        self._wake_timestamp: Optional[float] = None

        # Rolling transcript buffer for context-aware processing
        # Used for both retention and context passed to intent judge
        self._buffer_duration = float(getattr(self.cfg, "transcript_buffer_duration_sec", 120.0))
        self._transcript_buffer = TranscriptBuffer(max_duration_sec=self._buffer_duration)
        debug_log(f"transcript buffer initialised ({self._buffer_duration}s)", "voice")

        # Intent judge (full context, larger model) - always used when available
        self._intent_judge = create_intent_judge(self.cfg)
        if self._intent_judge is not None:
            debug_log(f"intent judge initialised (model: {self._intent_judge.config.model})", "voice")
        else:
            debug_log("intent judge unavailable, using simple wake word detection", "voice")

        # Thinking tune player
        self._tune_player: Optional = None

    def stop(self) -> None:
        """Stop the voice listener. No turn handed over is answered after
        this: the reply thread finishes the engine call in hand, if any,
        unheard."""
        self._should_stop = True
        self.state_manager.stop()
        self._stop_thinking_tune()

    def _start_thinking_tune(self) -> None:
        """Start the thinking tune when processing a query."""
        with self._tune_lock:
            if (self.cfg.tune_enabled and
                self._tune_player is None and
                (self.tts is None or not self.tts.is_speaking())):
                from ..output.tune_player import TunePlayer
                self._tune_player = TunePlayer(enabled=True)
                self._tune_player.start_tune()

    def _silence_the_tune(self) -> bool:
        """Stop the thinking tune, if one plays. Whether one did."""
        with self._tune_lock:
            player, self._tune_player = self._tune_player, None
        if player is None:
            return False
        player.stop_tune()
        return True

    def _stop_thinking_tune(self) -> None:
        """Stop the thinking tune and revert the published state to IDLE."""
        if self._silence_the_tune():
            get_jarvis_state().set_state(JarvisState.IDLE)

    def _drop_the_early_beep(self) -> None:
        """An utterance turned out not to be for her: stop the beep it
        started. While a reply is on its way the tune is that reply's, and
        it plays on."""
        if not self._a_reply_is_on_its_way():
            self._stop_thinking_tune()

    def _silence_speech(self) -> None:
        """Stop her talking on a spoken stop.

        The engine drops the rest of the reply, and an interrupted reply is
        unheard, so the completion callback that would end the speech never
        fires. Nothing else takes the published state out of SPEAKING, so it
        is published here (`state.spec.md`).
        """
        self.tts.interrupt()
        get_jarvis_state().set_state(JarvisState.IDLE)

    def _is_thinking_tune_active(self) -> bool:
        """Check if thinking tune is currently active."""
        return self._tune_player is not None and self._tune_player.is_playing()

    def _she_is_heard(self) -> bool:
        """Whether the speakers play something of hers: the thinking tune
        or her voice."""
        return self._is_thinking_tune_active() or bool(
            self.tts is not None and self.tts.is_speaking())

    def _publish_listening_state(self) -> None:
        """Publish LISTENING as the assistant's state."""
        get_jarvis_state().set_state(JarvisState.LISTENING)

    def _speak_as_it_comes(self, turn: _Turn):
        """A token callback that speaks each sentence of ``turn`` as it
        closes.

        Returns ``None`` when there is nothing to speak into, so the
        caller can hand the engine no callback at all and keep the
        buffered path.

        The chunks carry no completion callback: the reply's one callback
        is attached at the end by `_speak_reply`, because it opens the hot
        window and doing that mid-reply would take her own next sentence
        for a follow-up. Every sentence carries the language he was heard
        in when he asked, so one reply is never spoken in two voices, and
        nothing more of the turn is said once it was stopped.
        """
        if not (self.tts and self.tts.enabled):
            return None

        from ..output.streaming import SentenceStreamer

        decoupeur = SentenceStreamer()
        self._streamed_chars = 0

        def _sur_jeton(delta: str) -> None:
            try:
                for phrase in decoupeur.feed(delta):
                    with self._speech_lock:
                        if self._turn_is_off(turn):
                            return
                        self.track_tts_start(phrase, continues=self._streamed_chars > 0)
                        self.tts.speak(phrase, language=turn.language)
                    self._streamed_chars += len(phrase) + 1
            except Exception as e:
                # Never let delivery break generation: the buffered path
                # at the end still says the whole reply.
                debug_log(f"streaming TTS chunk failed, falling back: {e}", "voice")

        return _sur_jeton

    def track_tts_start(self, tts_text: str, continues: bool = False) -> None:
        """Record *tts_text* with the echo detector as she is about to say it.

        The room's quiet level goes with it (`room_floor.py`): the detector
        takes it as the voice proof's floor when this text starts a reply,
        and a later sentence (*continues*) keeps its reply's floor. Read from
        what the microphone heard while neither the thinking tune nor her
        voice played, it is the room's even when the tune still plays as her
        first sentence is queued. Before `ROOM_FLOOR_MIN_SECONDS` of quiet
        has been heard there is no room to read, and the floor is
        `FALLBACK_FLOOR`.
        """
        if self.tts and self.tts.enabled:
            floor = self._room_floor.level()
            if floor is None:
                floor = FALLBACK_FLOOR
                if not continues:
                    debug_log(f"voice proof floor: {self._room_floor.seconds_heard():.1f}s of "
                              f"quiet heard, under {ROOM_FLOOR_MIN_SECONDS:g}s, so the "
                              f"default {FALLBACK_FLOOR:.4f}", "echo")

            # A talking face holds her voice back: her words come that much later.
            self.echo_detector.track_tts_start(tts_text, floor, continues=continues,
                                               start_delay=self.tts.voice_delay())

    def activate_hot_window(self) -> None:
        """Activate hot window after TTS completion."""
        debug_log("TTS completed, checking hot window activation", "voice")

        if not self.cfg.hot_window_enabled:
            debug_log("hot window disabled in config, skipping", "voice")
            return

        # Track TTS finish time for echo detection
        self.echo_detector.track_tts_finish()

        # While the microphone is waited for nothing can hear a follow-up: no
        # window opens, so LISTENING is never shown while nothing listens,
        # and her speech ends on IDLE, unless a dictation owns the state.
        if self._microphone_wait is not None:
            debug_log("no hot window: the microphone is waited for", "voice")
            if not self._dictation_active:
                get_jarvis_state().set_state(JarvisState.IDLE)
            return

        # A waiting spoken question buys more listening time than a
        # follow-up does. Set per window rather than at construction,
        # because it depends on what is pending right now.
        self.state_manager.hot_window_seconds = self.hot_window_duration()

        # Schedule delayed hot window activation
        debug_log(f"scheduling hot window activation (echo_tolerance={self.state_manager.echo_tolerance}s, hot_window={self.state_manager.hot_window_seconds}s)", "voice")
        self.state_manager.schedule_hot_window_activation(self.cfg.voice_debug)

    def _process_transcript(self, text: str, utterance_energy: float = 0.0, utterance_start_time: float = 0.0, utterance_end_time: float = 0.0, trailing_silence: float = 0.0) -> None:  # noqa: E501
        """
        Process a transcript from speech recognition.

        Args:
            text: Transcribed text from audio
            utterance_energy: Pre-calculated energy from the utterance frames
            trailing_silence: Seconds of silence the voice detector waited
                through before closing the utterance. ``utterance_end_time``
                is when it was closed; the user stopped speaking this much
                earlier, which is where the latency journal starts its clock.
        """
        # Kept whole. When a spoken question is waiting, the answer is
        # read from this rather than from the query the intent judge
        # extracts, because at that moment there is no request to find in
        # the speech — there is a reply to read. Only what is collected
        # becomes his last words (`_collect`): an utterance left out, such
        # as a phrase Whisper made out of silence, must not answer for him.
        heard_at = utterance_end_time or time.time()
        if text and text.strip():
            self._hearing = (text, heard_at)
            get_latency_journal().note_heard(utterance_end_time - trailing_silence)

        if not text or not text.strip():
            self._check_query_timeout(heard_at)
            return

        text_lower = text.strip().lower()

        # Reset wake timestamp — it must reflect only the current utterance.
        # If this utterance contains a wake word, the early-beep check below
        # will set it. Without this reset, a prior rejected wake-worded
        # utterance would vouch for subsequent unrelated utterances via the
        # `_wake_timestamp is not None` guard in the intent-judge accept path.
        self._wake_timestamp = None

        start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
        end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3] if utterance_end_time > 0 else "N/A"
        debug_log(f"heard: '{text}' (utterance from {start_time_str} to {end_time_str})", "voice")

        # Track if this input was received during TTS (for logging purposes)
        received_during_tts = self.tts and self.tts.is_speaking()

        # --- Early echo check + early beep ---
        # Check for echo BEFORE starting beep and BEFORE intent judge.
        # This prevents: false beeps on echo, intent judge blocking the audio
        # loop for seconds on echo, and hot window extending from echo resets.
        if not received_during_tts and not self._is_thinking_tune_active():
            in_hot_window = self.state_manager.was_speech_during_hot_window(
                utterance_start_time, utterance_end_time
            )
            if in_hot_window:
                # Fuzzy echo check — instant, no intent judge needed.
                # Only catches pure echo (transcript ≈ TTS text). Mixed
                # echo+speech chunks (user spoke over echo) go to the
                # intent judge which can extract the user's speech.
                last_tts_text = self.echo_detector._last_tts_text or ""
                if last_tts_text:
                    is_pure_echo, echo_score = self._is_pure_echo(text_lower, last_tts_text)
                    if is_pure_echo:
                        # Before rejecting, try to salvage user speech appended
                        # after the echo prefix. Whisper commonly merges the tail
                        # of TTS echo with the user's follow-up into a single
                        # transcript; without salvage, the user's real speech
                        # would be dropped before the intent judge ever sees it.
                        # Try exact-word cleanup first (cheapest, most precise),
                        # then fall back to the rightmost-boundary scan which
                        # handles Whisper mis-transcriptions at the echo/speech
                        # join ("explores" → "laws") that exact matching can't.
                        salvaged = self.echo_detector.cleanup_leading_echo(text_lower)
                        if salvaged == text_lower:
                            salvaged_alt = self.echo_detector.salvage_after_echo_tail(text_lower)
                            if salvaged_alt:
                                salvaged = salvaged_alt
                        # Require ≥ min_salvage_words to avoid treating Whisper's
                        # echo-tail hallucinations ("…regions like Steneti") as
                        # genuine user speech. The threshold lives on the echo
                        # detector so every salvage site shares one policy.
                        # The stripper removes the part it recognised; the
                        # remainder is only speech if it is not her echo
                        # (`_not_her_echo`) and its voice went on after hers
                        # (`_voice_stopped_with_hers`). Left as "whatever is
                        # left over", a decimated tail of her own sentence,
                        # or one Whisper rewrote, rides through as a query.
                        min_words = self.echo_detector.min_salvage_words
                        if (salvaged != text_lower
                                and len(salvaged.split()) >= min_words
                                and not self._voice_stopped_with_hers(
                                    utterance_end_time, trailing_silence)
                                and self._not_her_echo(salvaged, utterance_energy)):
                            debug_log(
                                f"salvaged user speech from hot-window echo+speech "
                                f"chunk: '{salvaged}'",
                                "voice",
                            )
                            print(
                                f"  ✂️ Stripped echo prefix, kept: \"{salvaged[:60]}"
                                f"{'...' if len(salvaged) > 60 else ''}\"",
                                flush=True,
                            )
                            self._transcript_buffer.update_last_segment_text(salvaged)
                            # text_lower now carries the salvaged query — the rest
                            # of _process_transcript reads from this variable.
                            text_lower = salvaged
                        else:
                            debug_log(f"🔇 Early echo rejection (score={echo_score}): \"{text_lower}\"", "voice")
                            print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                            return

                # Non-echo (or salvaged) in hot window — start beep
                self._start_thinking_tune()
                self._publish_listening_state()
                debug_log("early beep: hot window active", "voice")
            else:
                # Not in hot window — check for wake word
                if self._carries_the_wake_word(text_lower):
                    self._wake_timestamp = utterance_start_time
                    self._start_thinking_tune()
                    self._publish_listening_state()
                    debug_log("early beep: wake word detected", "voice")

        # Stop commands — while she speaks, while a reply is on its way, and
        # while his question is still being collected. Fast and text-based.
        # Outside her speech the stop has to be (nearly) all he said: a stop
        # word inside a longer sentence is a question, "Jarvis, how do I stop
        # the timer?", and waits its turn.
        speaking = bool(self.tts and self.tts.enabled and self.tts.is_speaking())
        if speaking or (self._something_to_stop()
                        and len(text_lower.split()) <= STOP_WORDS_WHILE_THINKING):
            stop_commands = getattr(self.cfg, "stop_commands", ["stop", "quiet", "shush", "silence", "enough", "shut up"])
            # The threshold travels with the words. It was configurable,
            # exported and settable, and never reached this call, so
            # `is_stop_command`'s own signature default decided how hard
            # it was to interrupt her — whatever he had chosen.
            if is_stop_command(
                text_lower, stop_commands,
                fuzzy_ratio=float(getattr(self.cfg, "stop_command_fuzzy_ratio", 0.8)),
            ):
                debug_log(f"stop command detected ({'speaking' if speaking else 'thinking'}): "
                          f"{text_lower} (energy: {utterance_energy:.4f})", "voice")
                self._stop_the_replies()
                return

        # Echo rejection — only while TTS is actively playing. After TTS
        # finishes, the intent judge handles everything (echo detection, hot
        # window follow-ups, etc.) using full transcript context + last TTS text.
        if speaking:
            # Echo rejection during active TTS
            should_reject = self.echo_detector.should_reject_as_echo(
                text_lower, utterance_energy, True,
                getattr(self.cfg, 'tts_rate', 200), utterance_start_time
            )
            if should_reject:
                # Try to salvage user speech appended after echo
                salvaged = self.echo_detector.cleanup_leading_echo_during_tts(
                    text_lower,
                    getattr(self.cfg, 'tts_rate', 200),
                    utterance_start_time,
                )
                min_words = self.echo_detector.min_salvage_words
                if (salvaged and salvaged.strip() and salvaged != text_lower
                        and len(salvaged.split()) >= min_words):
                    debug_log(f"salvaged user speech from echo during TTS: '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged
                else:
                    debug_log(f"echo rejected during TTS: '{text_lower[:50]}'", "echo")
                    print(f"  🔇 Heard (echo): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    return

        # Salvage user speech from merged echo+speech chunks.
        # When Whisper delivers a single transcript containing TTS echo followed by
        # user speech (e.g. "I can only provide... Well you can search for it"), the
        # echo portion was captured during TTS but the transcript arrives after TTS
        # finishes. Try to strip the leading echo and use just the user's speech.
        # Skip entirely if there's no prior TTS — nothing to match against.
        last_tts_text_for_salvage = self.echo_detector._last_tts_text or ""
        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
        # Use echo_tolerance as buffer — speaker/mic latency means the utterance
        # may start slightly after TTS finish yet still contain the echo.
        echo_tol = self.echo_detector.echo_tolerance
        if (last_tts_text_for_salvage and last_tts_finish > 0
                and utterance_start_time > 0
                and utterance_start_time < last_tts_finish + echo_tol):
            salvaged = self.echo_detector._salvage_suffix_from_echo(
                text_lower,
                getattr(self.cfg, 'tts_rate', 200),
                utterance_start_time,
            )
            # If the prefix-based salvage fails or truncates too aggressively
            # (Whisper-mangled echo boundary → exact cleanup misses; fuzzy
            # prefix iteration prefers shortest suffix), fall through to the
            # rightmost-boundary scan which recovers the full follow-up.
            boundary_salvaged = self.echo_detector.salvage_after_echo_tail(text_lower)
            if boundary_salvaged and (
                salvaged is None or salvaged == text_lower
                or len(boundary_salvaged.split()) > len(salvaged.split())
            ):
                salvaged = boundary_salvaged
            min_words = self.echo_detector.min_salvage_words
            if (salvaged and salvaged.strip() and salvaged != text_lower
                    and len(salvaged.split()) >= min_words):
                # A voice that stopped with hers left nothing of his to
                # salvage: the utterance stays whole, and marked.
                if self._voice_stopped_with_hers(utterance_end_time, trailing_silence):
                    debug_log(f"nothing salvaged, its voice stopped with hers: "
                              f"'{text_lower[:60]}'", "voice")
                else:
                    debug_log(f"salvaged user speech from merged echo+speech chunk: '{salvaged}'", "voice")
                    self._transcript_buffer.update_last_segment_text(salvaged)
                    text_lower = salvaged

        # Check hot window expiry
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

        # Intent judge — the single decision-maker for all post-TTS input.
        # Gets full transcript context, last TTS text, and hot window state.
        # Handles: echo detection, wake word queries, hot window follow-ups.
        # During active TTS, this block skips short utterances (<=3 words)
        # as those are handled by stop command detection above. While his
        # question is collected, Priority 5 still judges them: Whisper's
        # inventions are that short ("Sous-titrage ST' 501").
        is_speaking_now = self.tts and self.tts.is_speaking()
        intent_judgment = None
        judged_as_a_follow_up = False

        # Determine if this could be a hot window follow-up.
        # Only use formal hot window state — no time-based grace period.
        # The state manager already handles the timing (echo_tolerance
        # delay before activation, hot_window_seconds before expiry).
        # A generous grace period caused false hot window claims after
        # the user had already seen "Returning to wake word mode".
        could_be_hot_window = self.state_manager.was_speech_during_hot_window(
            utterance_start_time, utterance_end_time
        )

        # Use the upgraded intent judge if available (with full transcript context)
        # Allow during TTS for longer utterances (>3 words) that might be user responses
        word_count = len(text_lower.split())
        skip_intent_judge_during_tts = is_speaking_now and word_count <= 3

        # Gate the intent judge on an engagement signal. Without this check the
        # judge was called on every ambient utterance, blocking the audio loop
        # for up to `timeout_sec` on each background chatter — which could
        # cascade into UI freezes when many utterances queued up during a slow
        # or loaded Ollama. The judge adds value only when one of:
        #   1. A wake word was detected in the current utterance
        #   2. We are in (or pending) a hot window following TTS
        #   3. TTS is currently speaking (intent judge can catch responses / stops
        #      that the fast text-based stop command check missed)
        #   4. His question is still being collected: Priority 5 asks, once
        #      nothing else has taken the utterance
        has_engagement_signal = (
            self._wake_timestamp is not None
            or could_be_hot_window
            or is_speaking_now
        )

        if not has_engagement_signal and not self.state_manager.is_collecting():
            debug_log(
                f"skipping intent judge — no wake word, no hot window, no TTS, no question collected "
                f"(ambient: \"{text_lower[:40]}{'...' if len(text_lower) > 40 else ''}\")",
                "voice",
            )

        # The error back-off lives inside ``judge()``: during the cooldown it
        # returns None without touching the backend. The listener still
        # enters this block so the no-verdict branch below runs — it prints
        # the unavailability and keeps his words in the hot window. A judge that cannot
        # answer is not the same thing as speech that was never addressed to
        # us, and gating entry on ``available`` made the two identical for
        # thirty seconds at a time.
        if (
            not skip_intent_judge_during_tts
            and has_engagement_signal
            and self._intent_judge is not None
        ):
            judged_as_a_follow_up = could_be_hot_window
            # Get recent transcript segments for context (full buffer)
            context_segments = self._transcript_buffer.get_last_seconds(self._buffer_duration)

            # Get TTS context for echo detection
            last_tts_text = self.echo_detector._last_tts_text or ""
            last_tts_finish_time = self.echo_detector._last_tts_finish_time or 0.0

            # Words that could be her own voice coming back, as the "(during
            # TTS)" mark has it, read on the text as it stands once a
            # stripped echo prefix is gone. No hot-window override takes
            # them, whatever the judge's words for leaving them out. Her
            # voice's end is read before the judge's call, during which her
            # next reply can start, and an override's log line gives that
            # reading.
            stopped_with_hers = self._voice_stopped_with_hers(utterance_end_time,
                                                              trailing_silence)
            could_be_hers = (could_be_hot_window
                             and self._could_be_hers(text_lower, utterance_energy,
                                                     self._in_her_time(utterance_start_time),
                                                     stopped_with_hers, spoken=last_tts_text))

            intent_judgment = self._judge_timed(
                segments=context_segments,
                wake_timestamp=self._wake_timestamp,
                last_tts_text=last_tts_text,
                last_tts_finish_time=last_tts_finish_time,
                in_hot_window=could_be_hot_window,
                current_text=text_lower,
            )

            if intent_judgment is not None:
                # Log intent judge decision for user visibility
                mode_str = "hot window" if could_be_hot_window else "wake word"
                if intent_judgment.directed:
                    print(f"  🧠 Intent ({mode_str}): directed → \"{intent_judgment.query or text_lower}\"", flush=True)
                else:
                    print(f"  🧠 Intent ({mode_str}): not directed ({intent_judgment.reasoning})", flush=True)
            else:
                reason = self._intent_judge.last_failure_reason or "no segments or unavailable"
                print(f"  🧠 Intent judge: unavailable ({reason})", flush=True)
                debug_log(f"intent judge returned None — falling back ({reason})", "voice")
                # Hot window without a verdict: his words are kept as he said
                # them, and her reply coming back whole is set aside, as are
                # words that could be hers, unless they carry a wake word that
                # can only be his: that one leaves the decision to the
                # wake-word check, as it does behind a verdict.
                if could_be_hot_window:
                    is_pure_echo, echo_score = self._is_pure_echo(text_lower, last_tts_text)
                    his_wake_word = (could_be_hers and not is_pure_echo
                                     and self._carries_his_wake_word(text_lower, last_tts_text))
                    if is_pure_echo or (could_be_hers and not his_wake_word):
                        self._heard_as_her_echo(
                            text_lower, f"no verdict, score={echo_score}, "
                                        f"could be hers: {could_be_hers}")
                        return
                    if his_wake_word:
                        debug_log(f"🔔 wake word in the hot window, no verdict, the wake-word "
                                  f"check decides: \"{text_lower}\"", "voice")
                    else:
                        print(f"  🧠 Intent fallback: accepting hot window speech", flush=True)
                        debug_log(f"✅ Hot window fallback (judge unavailable): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self._collect(text_lower)
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

            if intent_judgment is not None:
                # If judge says stop command, interrupt TTS
                if intent_judgment.stop and (
                        (self.tts and self.tts.is_speaking()) or self._something_to_stop()):
                    debug_log(f"🛑 Intent judge detected stop command", "voice")
                    self._stop_the_replies()
                    return

                # In the hot window, text Whisper made out of silence is
                # nobody's speech. The overrides below exist to keep his
                # words, so none of them may answer it, and it is no context
                # for what he says next. A verdict that also says directed
                # falls to his side. Outside the window the judge only runs
                # on a wake word or while she speaks, and Whisper does not
                # invent her name, so the paths below keep the verdict.
                if intent_judgment.noise and not intent_judgment.directed and could_be_hot_window:
                    debug_log(f"🔇 Not speech (judge): \"{text_lower}\"", "voice")
                    self._transcript_buffer.mark_segment_processed(text_lower)
                    print(f"  🔇 Heard (not speech): \"{text_lower[:50]}"
                          f"{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    self._drop_the_early_beep()
                    return

                # In the hot window, what he says to someone else (a guest,
                # a caller, the people watching a live stream) is not for
                # her, and the overrides below would answer it as speech in
                # her window. It is still words someone said: it stays in
                # the transcript as context for what he asks her next, and
                # the window stays open. Her name in it is his call, as
                # everywhere, and a verdict that also says directed falls to
                # his side.
                if (intent_judgment.aside and not intent_judgment.directed
                        and could_be_hot_window
                        and not self._carries_the_wake_word(text_lower)):
                    debug_log(f"🔇 Said to someone else (judge): \"{text_lower}\"", "voice")
                    print(f"  🔇 Heard (not to her): \"{text_lower[:50]}"
                          f"{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                    self._drop_the_early_beep()
                    return

                # In the hot window a verdict the judge is unsure of weighs
                # what a sure one does: both go through the paths below, and
                # each of them says on the console what became of the
                # utterance. Outside it, an unsure verdict leaves the
                # decision to the wake word. The judge's confidence is "low"
                # whenever its answer leaves it out.
                decisive = intent_judgment.confidence == "high" or could_be_hot_window

                # The wake word is his call, in the hot window as outside it.
                # A verdict that extracts nothing from an utterance carrying
                # a wake word that can only be his, her reply coming back
                # whole aside, leaves the decision to the wake-word check,
                # which answers what was heard with the wake word taken out.
                # The overrides below would set a quiet answer of his aside
                # as words that could be hers. A wake word that could be
                # hers (see `_carries_his_wake_word`) proves nothing: those
                # overrides decide.
                if (could_be_hot_window
                        and not (intent_judgment.directed and intent_judgment.query)
                        and not self._is_pure_echo(text_lower, last_tts_text)[0]
                        and self._carries_his_wake_word(text_lower, last_tts_text)):
                    debug_log(f"🔔 wake word in the hot window, the wake-word check decides: "
                              f"\"{text_lower}\"", "voice")
                    decisive = False

                # Her name alone, outside the hot window: a directed verdict
                # that finds no question, in what he said or in its query,
                # once her name is taken out, decides nothing. The wake-word
                # check does, and waits for his question rather than answer
                # her name as if it were one. A question the judge reads in
                # his earlier words is his, and goes on below.
                if (not could_be_hot_window and intent_judgment.directed
                        and not self._what_follows_her_name(text_lower)
                        and not self._what_follows_her_name((intent_judgment.query or "").lower())):
                    debug_log(f"🔔 her name alone, the wake-word check decides "
                              f"({intent_judgment.confidence} verdict)", "voice")
                    intent_judgment = replace(intent_judgment, query="")
                    decisive = False

                # If directed with query, process it
                if intent_judgment.directed and intent_judgment.query:
                    # In wake word mode, verify the wake word is actually present
                    # The LLM sometimes hallucinates wake words that don't exist
                    if not could_be_hot_window:
                        has_wake_word = (self._wake_timestamp is not None
                                         or self._carries_the_wake_word(text_lower))
                        if not has_wake_word:
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed but no wake word found in '{text_lower[:50]}...' "
                                f"(reasoning: {intent_judgment.reasoning})",
                                "voice"
                            )
                            # Don't accept - fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{intent_judgment.query}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self._collect(intent_judgment.query)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window mode - no wake word needed, but check for echo.
                        # The mic can pick up Jarvis's own TTS output and Whisper
                        # transcribes it as user speech. Check fuzzy similarity.
                        # Only reject PURE echo — if the heard text is significantly
                        # longer than TTS, it contains user speech mixed with echo
                        # and the intent judge's extraction should be used instead.
                        is_pure_echo, echo_score = self._is_pure_echo(text_lower, last_tts_text)
                        if is_pure_echo:
                            # Also check judge's extracted query — if it matches
                            # TTS too, it's genuinely pure echo. If the query is
                            # different, the judge extracted real user speech.
                            query_echo_score = fuzz.partial_ratio(
                                intent_judgment.query.lower(),
                                last_tts_text.lower()
                            )
                            if query_echo_score >= self.echo_detector.PURE_ECHO_THRESHOLD:
                                self._heard_as_her_echo(
                                    text_lower, f"directed, score={echo_score}")
                                return
                            debug_log(
                                f"echo in text (score={echo_score}) but judge extracted "
                                f"non-echo query: \"{intent_judgment.query}\"", "voice"
                            )

                        # The intent judge is explicitly designed to prune echo
                        # and extract the actual user query — always prefer its
                        # output when present. Falling back to raw heard text
                        # leaks partially-salvaged echo fragments into tool
                        # calls (e.g. "…amount now? okay, what is his best
                        # song?" reaching webSearch verbatim). If the judge
                        # returns an empty query (rare), fall back to raw text.
                        judge_query = (intent_judgment.query or "").strip()
                        hot_query = judge_query or text_lower
                        if judge_query and judge_query.lower() != text_lower:
                            debug_log(
                                f"using judge query over heard text: "
                                f"\"{judge_query}\" (heard: \"{text_lower[:80]}\")",
                                "voice",
                            )
                        # Half a sentence waits for its other half. The
                        # segment is deliberately left unprocessed so the
                        # next utterance's judge call sees both halves and
                        # can compose the whole question.
                        if _query_is_an_unfinished_fragment(text_lower, hot_query):
                            debug_log(
                                f"⏳ holding unfinished fragment: \"{hot_query}\" "
                                f"(heard: \"{text_lower[:80]}\")", "voice",
                            )
                            print("  ⏳ Sentence still going, waiting for the rest",
                                  flush=True)
                            self._drop_the_early_beep()
                            return

                        debug_log(f"✅ Intent judge accepted ({intent_judgment.confidence}): \"{hot_query}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()

                        self._collect(hot_query)

                        # Start thinking tune and show processing message
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

                # If directed, decisively (see above), but no extracted query, use actual text
                # Per spec: "Hot window input should reflect what the user actually said"
                # This handles cases where intent judge correctly identifies directed speech
                # but fails to extract/synthesize a query (e.g., conversational follow-ups)
                if intent_judgment.directed and decisive:
                    # In wake word mode, verify the wake word is actually present
                    if not could_be_hot_window:
                        has_wake_word = (self._wake_timestamp is not None
                                         or self._carries_the_wake_word(text_lower))
                        if not has_wake_word:
                            print(f"  🧠 Intent override: no wake word found, ignoring", flush=True)
                            debug_log(
                                f"⚠️ Intent judge said directed (no query) but no wake word in '{text_lower[:50]}...'",
                                "voice"
                            )
                            # Fall through to wake word check
                        else:
                            debug_log(f"✅ Intent judge accepted (directed, {intent_judgment.confidence} confidence, using actual text): \"{text_lower}\"", "voice")
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self._collect(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return
                    else:
                        # Hot window: with nothing extracted, what he said
                        # is the query, unless it is her reply coming back
                        # whole or words that could be hers. Shown those
                        # "(during TTS)", a judge saying directed and
                        # extracting nothing is leaving them out.
                        is_pure_echo, echo_score = self._is_pure_echo(text_lower, last_tts_text)
                        if is_pure_echo or could_be_hers:
                            self._heard_as_her_echo(
                                text_lower, f"directed/no-query, score={echo_score}, "
                                            f"could be hers: {could_be_hers}")
                            return

                        debug_log(f"✅ Intent judge accepted (directed, {intent_judgment.confidence} confidence, using actual text): \"{text_lower}\"", "voice")
                        self.state_manager.cancel_hot_window_activation()
                        self._transcript_buffer.mark_segment_processed(text_lower)
                        self._clear_audio_buffers()
                        self._collect(text_lower)
                        self._start_thinking_tune()
                        try:
                            print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                        except Exception:
                            pass
                        return

                # If not directed, decisively (see above), check reasoning before rejecting
                if not intent_judgment.directed and decisive:
                    # A rejection whose reasoning says "echo" is the judge
                    # taking the words for her own voice coming back. The
                    # verdict is weighed: just after the window it is
                    # overridden (the grace below), in the hot window only on
                    # proof that he spoke, and elsewhere the wake-word check
                    # decides.
                    reasoning_lower = (intent_judgment.reasoning or "").lower()
                    if "echo" in reasoning_lower:
                        debug_log(
                            f"⚠️ Intent judge says echo: \"{text_lower}\"",
                            "voice"
                        )
                        # Check if utterance started shortly after hot window expired
                        # This catches cases where user started speaking just as hot window expired
                        # Use a 2-second grace period after the 3-second hot window
                        hot_window_grace = 2.0
                        last_tts_finish = self.echo_detector._last_tts_finish_time or 0.0
                        hot_window_end = last_tts_finish + self.state_manager.hot_window_seconds
                        time_after_hot_window = utterance_start_time - hot_window_end if utterance_start_time > 0 and hot_window_end > 0 else float('inf')

                        if 0 <= time_after_hot_window < hot_window_grace:
                            # Utterance started within grace period after hot window
                            debug_log(
                                f"✅ Accepting as directed: started {time_after_hot_window:.2f}s after hot window expired",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()

                            # Mark the current segment as processed to prevent re-extraction
                            self._transcript_buffer.mark_segment_processed(text_lower)

                            self._clear_audio_buffers()
                            self._collect(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Check could_be_hot_window (handles overlap: utterance
                        # started during TTS but extended into hot window span).
                        # The grace period above only checks utterance_start_time
                        # which is negative for overlapping utterances.
                        if could_be_hot_window:
                            # The judge read this as her own echo and said
                            # so. Overriding that verdict takes proof the
                            # user spoke (`_not_her_echo`), against the
                            # reply the judge was shown, not the failure of
                            # a contiguous ratio to find its match in a
                            # decimated echo; and words that could be hers,
                            # their voice stopping with hers among them,
                            # are never his.
                            is_pure_echo, echo_score = self._is_pure_echo(
                                text_lower, last_tts_text)
                            proofs = self._not_her_echo(
                                text_lower, utterance_energy, spoken=last_tts_text)
                            if is_pure_echo or could_be_hers or not proofs:
                                self._heard_as_her_echo(
                                    text_lower, f"echo verdict stands, score={echo_score}, "
                                                f"could be hers: {could_be_hers}, "
                                                f"words she did not say in a voice: {proofs}")
                                return
                            print(f"  🧠 Intent override: accepting hot window speech "
                                  f"(words she did not say, in a voice)", flush=True)
                            debug_log(
                                f"⚡ Overriding the echo verdict in hot window "
                                f"(echo_score={echo_score}, "
                                f"{self._not_hers_because(stopped_with_hers)}): \"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self._collect(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Otherwise fall through to wake word detection
                        debug_log(f"⏭️ Not near hot window ({time_after_hot_window:.2f}s after), falling through to wake word check", "voice")
                        # Continue to wake word detection below
                    else:
                        is_pure_echo, echo_score = self._is_pure_echo(text_lower, last_tts_text)

                        # Her reply coming back whole, which the early check
                        # should have caught, or words that could be hers,
                        # rejected in other words than "echo": no override
                        # takes them.
                        if could_be_hot_window and (is_pure_echo or could_be_hers):
                            self._heard_as_her_echo(
                                text_lower, f"rejected, score={echo_score}, "
                                            f"could be hers: {could_be_hers}")
                            return

                        if could_be_hot_window:
                            # Hot window + non-echo speech → user is talking to us.
                            # Override the intent judge rejection — small models
                            # sometimes reject valid follow-ups like "don't you
                            # already know that?" as not directed.
                            print(f"  🧠 Intent override: accepting hot window speech", flush=True)
                            debug_log(
                                f"⚡ Overriding intent judge in hot window "
                                f"(echo_score={echo_score}, reasoning={intent_judgment.reasoning}): "
                                f"\"{text_lower}\"",
                                "voice"
                            )
                            self.state_manager.cancel_hot_window_activation()
                            self._transcript_buffer.mark_segment_processed(text_lower)
                            self._clear_audio_buffers()
                            self._collect(text_lower)
                            self._start_thinking_tune()
                            try:
                                print(f"\n✨ Working on it: {self.state_manager.get_pending_query()}")
                            except Exception:
                                pass
                            return

                        # Outside hot window — trust rejection
                        debug_log(f"🚫 Intent judge rejected (not directed, {intent_judgment.confidence} confidence): \"{text_lower}\"", "voice")
                        self._drop_the_early_beep()
                        return
                else:
                    # Not decisive (see above): the wake word decides
                    debug_log(f"⏭️ Intent judge leaves it to the wake word ({intent_judgment.confidence}), checking wake word", "voice")

        # Priority 4: Wake word detection (fallback when intent judge unavailable/inconclusive)
        wake_word = getattr(self.cfg, "wake_word", "jarvis")

        wake_detected = self._carries_the_wake_word(text_lower)
        debug_log(f"wake word check: '{wake_word}' in '{text_lower}' → {wake_detected}", "voice")

        if wake_detected:
            # Cancel any pending hot window activation when new query starts
            self.state_manager.cancel_hot_window_activation()

            # Mark the current segment as processed to prevent re-extraction
            self._transcript_buffer.mark_segment_processed(text_lower)

            # Clear audio buffers to prevent concatenation issues
            self._clear_audio_buffers()

            self._collect(self._what_follows_her_name(text_lower))

            # Start thinking tune and show processing message. Her name with
            # nothing after it is a question with nothing in it yet: she
            # waits for it, and `_check_query_timeout` ends the wait. The
            # line is flushed: through a pipe it would otherwise wait for the
            # next one, and could show up once the wait is over.
            self._start_thinking_tune()
            pending = self.state_manager.get_pending_query()
            if not pending:
                debug_log("wake word with nothing after it: waiting for his question", "voice")
            try:
                print(f"\n✨ Working on it: {pending}" if pending else "\n👂 Waiting for your question...",
                      flush=True)
            except Exception:
                pass
            return

        # Priority 5: Collection mode handling. What he adds to his question
        # needs no wake word, as in the hot window, so Whisper's inventions
        # get this far too. The judge reads it as it reads a follow-up, with
        # his question before it; a verdict given above in wake-word mode
        # saw neither, so it is asked again. A stop it reads cancels the
        # question, a noise verdict leaves the utterance out, and anything
        # else completes the question. A phantom leaves the collection's
        # silence clock alone, so his question still goes when he stops.
        # Marked processed either way: left visible, the judge could restate
        # it in a later query.
        if self.state_manager.is_collecting():
            if not judged_as_a_follow_up and self._intent_judge is not None:
                intent_judgment = self._judge_timed(
                    segments=self._after_his_question(
                        self._transcript_buffer.get_last_seconds(self._buffer_duration),
                        text_lower),
                    wake_timestamp=None,
                    last_tts_text=self.echo_detector._last_tts_text or "",
                    last_tts_finish_time=self.echo_detector._last_tts_finish_time or 0.0,
                    in_hot_window=True,
                    current_text=text_lower,
                )
                if intent_judgment is None:
                    reason = self._intent_judge.last_failure_reason or "no segments or unavailable"
                    print(f"  🧠 Intent judge: unavailable ({reason})", flush=True)
                    debug_log(f"intent judge returned None while collecting — it joins ({reason})", "voice")
            self._transcript_buffer.mark_segment_processed(text_lower)
            if intent_judgment is not None and intent_judgment.stop:
                debug_log(f"🛑 Intent judge detected stop command while collecting", "voice")
                self._stop_the_replies()
                return
            if (intent_judgment is not None and intent_judgment.noise
                    and not intent_judgment.directed):
                debug_log(f"🔇 Not speech while collecting (judge): \"{text_lower}\"", "voice")
                print(f"  🔇 Heard (not speech): \"{text_lower[:50]}"
                      f"{'...' if len(text_lower) > 50 else ''}\"", flush=True)
                return
            self._collect(text_lower)
            return

        # Priority 6: Non-wake input (ignore)
        # Provide clear debug info about why input was ignored
        intent_info = ""
        if intent_judgment is not None:
            intent_info = f", intent={intent_judgment.directed}/{intent_judgment.confidence}"

        # Stop any early-started beep since we're not processing this input
        self._drop_the_early_beep()

        if received_during_tts:
            # User spoke during TTS but it wasn't a stop command - this is likely a response
            # to a TTS question that arrived before hot window activated
            debug_log(f"input ignored (during TTS, not a stop command{intent_info}): {text_lower}", "voice")
            try:
                print(f"  ⏳ Heard during TTS (waiting for hot window): \"{text_lower[:50]}{'...' if len(text_lower) > 50 else ''}\"", flush=True)
            except Exception:
                pass
        else:
            debug_log(f"input ignored (no wake word{intent_info}): {text_lower}", "voice")

    def _not_her_echo(self, heard: str, utterance_energy: float,
                      spoken: Optional[str] = None) -> bool:
        """True when *heard* cannot be only her own voice coming back: it
        carries words she did not say, loud enough to have been spoken.

        *spoken* is what she said: by default the reply the echo detector
        holds now. A decision on the judge's verdict passes the reply the
        judge was shown, which a reply starting during its call replaces.

        Two proofs, because either alone is fooled. Text alone takes a
        "Thank you." that Whisper makes of the room over her French reply
        for new words; energy alone takes a door slamming for a voice.
        Together they are fooled by her own voice when Whisper makes words
        she never said of it (a figure heard as an hour, a unit as a
        name), since her voice through the speakers clears the energy bar.
        After her last word, `_voice_stopped_with_hers` tells that one
        from him; while she speaks nothing does, and an invention of the
        room the judge names noise, which nothing overrides.
        """
        if spoken is None:
            spoken = self.echo_detector._last_tts_text or ""
        return (_carries_speech_she_did_not_say(heard, spoken)
                and self.echo_detector.utterance_carries_a_voice(utterance_energy))

    def _his_words_among_her_sentences(self, text: str, pieces: list) -> str:
        """*text*, an utterance heard in her time that Whisper made of
        *pieces*, without the sentences of her reply that open and close it
        (`_without_her_sentences_at_the_edges`), and the console told what
        was kept. Unchanged when nothing of it is hers to take out."""
        his, hers = _without_her_sentences_at_the_edges(
            pieces, self.echo_detector._last_tts_text or "",
            self.echo_detector.min_salvage_words)
        if not hers:
            return text
        kept = " ".join(his).strip()
        debug_log(f"✂️ her sentences at the edges left out: {[h.strip() for h in hers]}, "
                  f"kept: \"{kept}\"", "voice")
        print(f"  ✂️ Left out her own words, kept: \"{kept[:60]}"
              f"{'...' if len(kept) > 60 else ''}\"", flush=True)
        return kept

    def _in_her_time(self, utterance_start_time: float) -> bool:
        """Whether an utterance can hold her own voice coming back: she is
        speaking, or it started (the voice detector's pre-roll included)
        before her last word plus the echo tolerance."""
        if self.tts is not None and self.tts.is_speaking():
            return True
        finish = self.echo_detector._last_tts_finish_time
        return (finish > 0 and utterance_start_time > 0
                and utterance_start_time < finish + self.echo_detector.echo_tolerance)

    def _voice_stopped_with_hers(self, utterance_end_time: float,
                                 trailing_silence: float) -> Optional[bool]:
        """Whether the utterance's voice stopped before her last word plus
        the echo tolerance, on the capture clock: the detector closed it at
        *utterance_end_time* after *trailing_silence* seconds without a
        voice. Her voice coming back cannot outlast her voice, whatever
        Whisper makes of it; his answer begun over her last word goes on
        after it.

        None when it cannot be told: while she speaks her last word is not
        known yet, and an utterance whose end is not known says nothing.
        None counts as no, and the two proofs decide (`_not_her_echo`)."""
        if self.tts is not None and self.tts.is_speaking():
            return None
        finish = self.echo_detector._last_tts_finish_time
        if not (finish > 0 and utterance_end_time > 0):
            return None
        voice_ended = utterance_end_time - max(0.0, trailing_silence)
        return voice_ended < finish + self.echo_detector.echo_tolerance

    def _could_be_hers(self, heard: str, utterance_energy: float, in_her_time: bool,
                       stopped_with_hers: Optional[bool],
                       spoken: Optional[str] = None) -> bool:
        """Whether an utterance could be her own voice coming back, as the
        "(during TTS)" mark has it: heard in her time (*in_her_time*, as
        `_in_her_time` read it), and either its voice stopped with hers
        (*stopped_with_hers*, as `_voice_stopped_with_hers` read it),
        whatever its words, or nothing proves he spoke (`_not_her_echo`,
        against *spoken*).

        The caller reads her time and her voice's end once, and its log line
        gives that same reading: her reply can start or end while the words
        are weighed."""
        if not in_her_time:
            return False
        return (bool(stopped_with_hers)
                or not self._not_her_echo(heard, utterance_energy, spoken=spoken))

    @staticmethod
    def _not_hers_because(stopped_with_hers: Optional[bool]) -> str:
        """Why words heard in her time were taken for his, as the log gives
        it: words she did not say, in a voice, and that voice going on after
        hers when its end was read (`_voice_stopped_with_hers`). While she
        speaks, or with its end unknown, the two proofs alone decided."""
        if stopped_with_hers is False:
            return "words she did not say, in a voice that went on after hers"
        return "words she did not say, in a voice"

    def _is_pure_echo(self, heard: str, spoken: str) -> tuple:
        """Whether *heard* is her reply *spoken* coming back whole: one
        contiguous match (`partial_ratio`) at the detector's pure-echo
        threshold, and not so much longer than her reply that words of his
        must follow it. Returns the verdict and the score."""
        if not spoken:
            return False, 0
        score = fuzz.partial_ratio(heard, spoken.lower())
        her_words = len(spoken.split())
        return (score >= self.echo_detector.PURE_ECHO_THRESHOLD
                and len(heard.split()) <= max(her_words * 1.3, her_words + 3)), score

    def _carries_the_wake_word(self, text_lower: str) -> bool:
        """Whether the wake word or one of its aliases is in *text_lower*,
        matched as the configured fuzzy ratio allows."""
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fuzzy_ratio = float(getattr(self.cfg, "wake_fuzzy_ratio", 0.78))
        return is_wake_word_detected(text_lower, wake_word, list(aliases), fuzzy_ratio)

    def _what_follows_her_name(self, text_lower: str) -> str:
        """What *text_lower* says once the wake word and its aliases are
        taken out, or nothing when no letter or digit is left, in any script:
        "jarvis ?", "¿jarvis?" and "jarvis…" are her name alone."""
        wake_word = getattr(self.cfg, "wake_word", "jarvis")
        aliases = set(getattr(self.cfg, "wake_aliases", [])) | {wake_word}
        fragment = extract_query_after_wake(text_lower, wake_word, list(aliases))
        return fragment if any(ch.isalnum() for ch in fragment) else ""

    def _carries_his_wake_word(self, text_lower: str, spoken: str) -> bool:
        """Whether *text_lower* carries a wake word that can only be his
        call, given her reply *spoken*. The wake-word check finds one in
        *text_lower*, so it answers what it is handed; it finds none in her
        reply; and it still finds one once the tokens that are only her words
        are taken out.

        Her reply can name her, or say a word the check takes for her name
        ("avis" for "jarvis"), and coming back that word can take any
        spelling Whisper gives her name, so after such a reply no wake word
        heard can be told from hers. Whisper can also bend a word of hers
        into one ("servi" heard "servis", close to "jervis"): that token is
        still hers as far as `_her_word` reaches, which is word by word, so
        two of her words merged into one ("dave is" heard "davis") pass for
        his."""
        if not self._carries_the_wake_word(text_lower):
            return False
        if self._carries_the_wake_word((spoken or "").lower()):
            debug_log(f"🔔 wake word heard after a reply holding one, it decides nothing: "
                      f"\"{text_lower}\"", "voice")
            return False
        if not self._carries_the_wake_word(" ".join(_tokens_she_did_not_say(text_lower, spoken))):
            debug_log(f"🔔 wake word heard among her own words, it decides nothing: "
                      f"\"{text_lower}\"", "voice")
            return False
        return True

    def _heard_as_her_echo(self, text_lower: str, why: str) -> None:
        """Set an utterance heard in the hot window aside as her own voice
        coming back, and say so on the console."""
        debug_log(f"🔇 Echo in hot window ({why}): \"{text_lower}\"", "voice")
        print(f"  🔇 Heard (echo): \"{text_lower[:50]}"
              f"{'...' if len(text_lower) > 50 else ''}\"", flush=True)
        self._drop_the_early_beep()

    def _after_his_question(self, segments: list, current_text: str) -> list:
        """The transcript with the question being collected put back before
        the utterance being judged. Its segments are processed, so the judge
        would not see them, and an addition read alone, misheard ("et pour à
        près de main aussi"), looks like something Whisper invented. Priority
        5 reads only the verdict's stop and noise, never a query, so the
        question cannot be restated."""
        asked = self.state_manager.get_pending_query()
        if not asked:
            return segments
        at = next((i for i in range(len(segments) - 1, -1, -1)
                   if segments[i].text.lower() == current_text), len(segments))
        when = segments[at].start_time if at < len(segments) else time.time()
        question = TranscriptSegment(text=asked, start_time=when, end_time=when)
        return [*segments[:at], question, *segments[at:]]

    def _judge_timed(self, **kwargs):
        """Ask the intent judge, and tell the latency journal how long it took
        and what kind of answer it gave (an enumerated outcome, never text)."""
        started = time.monotonic()
        verdict = self._intent_judge.judge(**kwargs)
        if verdict is None:
            outcome = "no_verdict"
        else:
            outcome = "directed" if verdict.directed else "not_directed"
        get_latency_journal().note_judge(time.monotonic() - started, outcome)
        return verdict

    def _latency_models(self) -> dict:
        """The model identifiers a turn is timed against (names, not text)."""
        from ..reply.engine import resolve_tool_router_model

        return {
            "judge": getattr(self.cfg, "intent_judge_model", None),
            "router": resolve_tool_router_model(self.cfg),
            "chat": getattr(self.cfg, "llm_chat_model", None),
        }

    # ── The reply thread ──────────────────────────────────────────────
    #
    # A reply takes seconds to prepare and the microphone must be read
    # meanwhile, so the loop hands a query over and goes back to listening,
    # and this thread answers the queries one after the other. It is also
    # the only thread that speaks, so the echo detector's record of what she
    # last said has one writer.

    def _start_replies(self) -> None:
        """Start the reply thread, once."""
        if self._replies is None or not self._replies.is_alive():
            self._replies = threading.Thread(
                target=self._reply_loop, daemon=True, name="voice-replies")
            self._replies.start()

    def _reply_loop(self) -> None:
        while not self._should_stop:
            try:
                turn = self._turns.get(timeout=0.1)
            except queue.Empty:
                turn = None
            if turn is not None:
                self._take_turn(turn)
            try:
                self.drain_reply_queue()
            except Exception as e:
                debug_log(f"reply thread: queued reply failed: {e}", "voice")

    def _take_turn(self, turn: _Turn) -> None:
        """Answer one handed-over query, once her voice has stopped: a reply
        started while the last is still audible would replace the echo
        detector's record of it, and that one's window would open over the
        new one."""
        with self._speech_lock:
            self._turn_in_hand = turn
        try:
            while (self.tts is not None and self.tts.is_speaking()
                   and not self._turn_is_off(turn)):
                time.sleep(0.02)
            if self._turn_is_off(turn):
                debug_log(f"turn not answered, stopped: '{turn.query}'", "voice")
            elif self._may_answer_now(turn):
                self._dispatch_query(turn)
        except Exception as e:
            print(f"  ❌ Reply failed: {e}", flush=True)
            debug_log(f"reply thread: turn failed: {e}", "voice")
            self._stop_thinking_tune()
            get_jarvis_state().set_state(JarvisState.IDLE)
        finally:
            with self._speech_lock:
                self._turn_in_hand = None
                self._replies_owed -= 1

    def _turn_is_off(self, turn: _Turn) -> bool:
        """A stop was heard, or the listener stopped, since it was handed over."""
        return turn.epoch != self._stop_epoch or self._should_stop

    def _may_answer_now(self, turn: _Turn) -> bool:
        """Whether this turn may run now, given her spoken question.

        A spoken question is answered by the next voice turn, and only
        words heard after she asked it are an answer. A turn whose words he
        said before he heard it is set aside, so the question waits for
        what he says to it. A question he never heard (its reply was
        stopped) is withdrawn, and the turn runs.
        """
        asked = self._asked
        waiting = self._waiting_spoken_question()
        if waiting is None or asked is None or asked.request_id != waiting.request_id:
            return True
        if asked.heard_at is None and self._turn_is_off(asked.turn):
            from ..reply.engine import expire_waiting_question

            expire_waiting_question(self.db, self.dialogue_memory,
                                    why="its reply was stopped before he heard it")
            print("  🙋 Her question was never heard: withdrawn", flush=True)
            return True
        heard_at = asked.heard_at if asked.heard_at is not None else time.time()
        if turn.heard_at <= heard_at:
            shown = turn.query[:50] + ("..." if len(turn.query) > 50 else "")
            print(f"  ⏭️ Asked before her question, set aside: \"{shown}\"", flush=True)
            debug_log(f"turn set aside, its words predate her question: '{turn.query}'", "voice")
            return False
        return True

    def _waiting_spoken_question(self):
        """The spoken question a voice turn could answer, if one waits."""
        dm = self.dialogue_memory
        if dm is None or not hasattr(dm, "peek_pending"):
            return None
        try:
            from ..tools.confirmation import CHANNEL_PAROLE

            waiting = dm.peek_pending()
        except Exception as e:
            debug_log(f"waiting question not read: {e}", "voice")
            return None
        if (waiting is None or getattr(waiting, "channel", None) != CHANNEL_PAROLE
                or getattr(waiting, "origin", None) != "voix"):
            return None
        return waiting

    def _hand_over(self, query: str) -> None:
        """Give a complete query to the reply thread. Returns at once: a
        query asked while another is being answered waits its turn."""
        # The utterance just closed is consumed; what he says next is read
        # from the audio queue as it comes.
        self._clear_audio_buffers()
        with self._speech_lock:
            turn = _Turn(query, self._last_detected_language,
                         getattr(self, "_last_transcript", None),
                         heard_at=self._last_heard_at, epoch=self._stop_epoch)
            self._replies_owed += 1
            self._latest_turn_epoch = turn.epoch
            dropped = None
            if self._turns.qsize() >= MAX_WAITING_TURNS:
                try:
                    dropped = self._turns.get_nowait()
                    self._replies_owed -= 1
                except queue.Empty:
                    pass
            self._turns.put(turn)
        if dropped is not None:
            shown = dropped.query[:50] + ("..." if len(dropped.query) > 50 else "")
            print(f"  ⏭️ Too many questions waiting, dropped: \"{shown}\"", flush=True)
        debug_log(f"handing query to the reply thread: '{query}'", "voice")

    def _collect(self, query: str) -> None:
        """Start collecting his question, or add to the one being collected:
        a second sentence that reaches her before the first was handed over
        completes it, and she answers both. The utterance being handled
        becomes his last words, and the collection's silence is counted from
        when it ended, so neither Whisper nor the judge adds to it."""
        self._last_transcript, self._last_heard_at = self._hearing
        if self.state_manager.is_collecting():
            self.state_manager.add_to_collection(query, heard_at=self._last_heard_at)
        else:
            self.state_manager.start_collection(query, heard_at=self._last_heard_at)

    def _something_to_stop(self) -> bool:
        """A turn owed, or his question still being collected. A turn a stop
        has silenced counts while the engine finishes it: a stop he repeats
        then is still a stop, never a question."""
        return self._replies_owed > 0 or self.state_manager.is_collecting()

    def _a_reply_is_on_its_way(self) -> bool:
        """Whether a reply of hers will be said: a turn is being prepared or
        said, or waits to be, and no stop came after it. A stopped turn the
        engine is still finishing is never said, so it is not one."""
        return self._replies_owed > 0 and self._latest_turn_epoch == self._stop_epoch

    def _stop_the_replies(self) -> None:
        """A stop: nothing more of the reply in hand is said, the queries
        waiting their turn are dropped, and so is a question still being
        collected. The engine finishes the turn in hand on its own,
        unheard; replies handed over by other features are kept."""
        with self._speech_lock:
            self._stop_epoch += 1
            self._silence_speech()
        if self.state_manager.is_collecting():
            self.state_manager.clear_collection()
        self._stop_thinking_tune()
        debug_log("stopped: the reply in hand and the ones waiting are not said", "voice")

    def _dispatch_query(self, turn: _Turn) -> None:
        """Run one turn through the reply engine and say the reply."""
        query = turn.query
        debug_log(f"dispatching query: '{query}'", "voice")

        get_jarvis_state().set_state(JarvisState.THINKING)
        debug_log("state set to THINKING (dispatch_query)", "voice")

        # Import reply engine
        from ..reply.engine import run_reply_engine
        from ..daemon import query_lock

        journal = get_latency_journal()
        journal.begin_turn(self._latency_models())
        waiting_before = self._waiting_spoken_question()

        try:
            # Process the query (keep thinking tune playing during processing).
            # Hold the shared voice+text query lock so a voice query and a text
            # chat query cannot run the reply engine concurrently against the
            # same dialogue memory. Voice blocks while a text query finishes
            # rather than being dropped (see daemon.query_lock).
            try:
                with query_lock():
                    if self._turn_is_off(turn):
                        # Stopped while it waited for the lock: not run at all.
                        reply = None
                    else:
                        reply = run_reply_engine(
                            self.db, self.cfg, None, query, self.dialogue_memory,
                            language=turn.language,
                            origin="voix",
                            heard=turn.heard,
                            on_token=self._speak_as_it_comes(turn),
                            on_stage=journal.on_stage,
                        )
            except Exception as e:
                # The apology below is not this turn's reply, so its sound must
                # not be taken for the turn's first sound.
                journal.finish_turn(will_speak=False)
                # Log the error visibly - this should never happen silently
                print(f"\n  ❌ Reply engine error: {e}", flush=True)
                debug_log(f"reply engine exception: {e}", "voice")
                self._stop_thinking_tune()
                # Provide user feedback via TTS
                if self.tts and self.tts.enabled and not self._turn_is_off(turn):
                    self.tts.speak("Sorry, I encountered an error processing your request.")
                journal.flush()
                return

            asked = self._note_question_asked(turn, waiting_before)

            # Frozen now, written once the reply is queued: the write must not sit
            # between "reply ready" and "start speaking". A sound that starts in
            # between completes the row from memory; one that starts later
            # updates it.
            try:
                with self._speech_lock:
                    off = self._turn_is_off(turn)
                    will_speak = bool(reply and not off and self.tts
                                      and getattr(self.tts, "enabled", False))
                    journal.finish_turn(will_speak=will_speak)
                    if off:
                        debug_log("turn was stopped: nothing more of it is said", "voice")
                    else:
                        self._speak_reply(reply, on_spoken=asked.heard if asked else None,
                                          language=turn.language)
                if off:
                    if self._a_reply_is_on_its_way() or self.state_manager.is_collecting():
                        debug_log("stopped turn done: what he started since keeps the tune "
                                  "and the state", "voice")
                    else:
                        # The stop silenced the tune and published IDLE, but
                        # the THINKING this turn published as it reached the
                        # engine may have landed after it.
                        self._stop_thinking_tune()
                        get_jarvis_state().set_state(JarvisState.IDLE)
            finally:
                journal.flush()
        finally:
            self._streamed_chars = 0

    def _note_question_asked(self, turn: _Turn, waiting_before) -> Optional[_AskedQuestion]:
        """The spoken question this turn's reply ends on, if it raised one."""
        waiting = self._waiting_spoken_question()
        if waiting is None or (waiting_before is not None
                               and waiting_before.request_id == waiting.request_id):
            return None
        self._asked = _AskedQuestion(waiting.request_id, turn)
        return self._asked

    def _speak_reply(self, reply: Optional[str], on_spoken=None,
                     language: Optional[str] = None) -> None:
        """Say one reply, and do the echo bookkeeping that goes with it.

        The only place anything is spoken. A sentence said without being
        recorded here is one the echo detector will not recognise coming
        back, so she answers herself.

        ``on_spoken`` fires when the speech finished — not when it
        started, and not at all when it was interrupted, because
        interrupted means unheard. ``language`` is the one the reply is
        spoken in, each part of it.
        """
        if reply and self.tts and self.tts.enabled:
            # Stop thinking tune when TTS starts
            self._stop_thinking_tune()

            # TTS completion callback for hot window
            def _on_tts_complete():
                import time as _time
                debug_log(f"TTS completion callback triggered at {_time.time():.3f}", "voice")
                self.activate_hot_window()
                # After the window, so a slow or raising caller cannot
                # cost the user their chance to reply. This runs on the
                # audio thread: whatever it is doing, it is not worth the
                # microphone.
                if on_spoken is not None:
                    try:
                        on_spoken()
                    except Exception as e:
                        debug_log(f"delivery callback raised, ignoring: {e}", "voice")

            # Duration callback to update echo detector with exact timing (Piper only)
            def _on_duration_known(duration: float):
                debug_log(f"TTS exact duration: {duration:.2f}s", "voice")
                if self.echo_detector:
                    self.echo_detector._tts_exact_duration = duration

            # What streaming already said, if anything. The chunks are
            # spoken as they close; what is left here is the tail — and
            # when the last sentence ended on punctuation there is none,
            # so the completion callback rides the end-of-reply marker
            # instead. It has to fire exactly once, after everything:
            # firing per chunk would open the hot window mid-reply and
            # invite her own next sentence in as a follow-up.
            deja_dit = self._streamed_chars
            self._streamed_chars = 0

            if deja_dit:
                reste = reply[deja_dit:].strip()
                debug_log(f"TTS tail after streaming ({len(reste)} chars)", "voice")
                if reste:
                    self.track_tts_start(reste, continues=True)
                self.tts.speak(reste, completion_callback=_on_tts_complete,
                               duration_callback=_on_duration_known,
                               language=language)
                return

            # Track TTS start for echo detection with actual text
            self.track_tts_start(reply)
            debug_log(f"starting TTS for reply ({len(reply)} chars)", "voice")

            self.tts.speak(reply, completion_callback=_on_tts_complete,
                          duration_callback=_on_duration_known,
                          language=language)
        else:
            debug_log(f"no TTS output: reply={bool(reply)}, tts={bool(self.tts)}, enabled={getattr(self.tts, 'enabled', False) if self.tts else False}", "voice")
            # Stop thinking tune if no TTS response
            self._stop_thinking_tune()

    # ── Replies produced elsewhere ────────────────────────────────────
    #
    # A click-approved action runs on a resume worker. If that worker
    # spoke directly it would race the reply thread, which speaks outside
    # the block holding the query lock: two writes to the echo detector's
    # record of what was last said, and two hot-window activations. The
    # detector would then compare the next transcript against the wrong
    # sentence, which is how a user's answer gets deleted as an echo.

    def enqueue_reply(self, text: Optional[str], on_spoken=None) -> bool:
        """Hand a reply to the reply thread to speak. Safe from any thread.

        Returns whether it was taken. False is a *definitive* refusal —
        there is no engine that will ever say this — so the caller deals
        with it now rather than waiting for a delivery that cannot come.
        Queueing into a dead engine would hold the text forever and say
        it at some unrelated later moment.

        ``on_spoken`` fires once the speech actually finished, which is
        not the same thing as it being queued. Anything holding a promise
        needs that distinction: interrupted speech never reports, so it
        can be tried again.
        """
        if not text or not text.strip():
            return False
        if not self.tts or not getattr(self.tts, "enabled", False):
            debug_log("reply not queued: TTS unavailable", "voice")
            return False
        if getattr(self, "_reply_queue", None) is None:
            import queue as _queue

            self._reply_queue = _queue.Queue()
        self._reply_queue.put((text, on_spoken))
        debug_log(f"reply queued for speaking ({len(text)} chars)", "voice")
        return True

    def drain_reply_queue(self) -> None:
        """Say at most one queued reply. The reply thread only.

        One at a time, and only while nothing is already being said. The
        echo detector holds the record of one reply (its text, its start,
        its exact duration and the floor of its voice proof), so a second
        reply started while the first is still audible would replace it,
        and the rest of the first would be compared against the wrong
        sentence. Whatever is left waits for the reply thread's next pass,
        a few milliseconds away when it has no query to run.
        """
        q = getattr(self, "_reply_queue", None)
        if q is None:
            return
        if self.tts is not None and self.tts.is_speaking():
            return

        import queue as _queue

        try:
            text, on_spoken = q.get_nowait()
        except _queue.Empty:
            return
        # It has no utterance behind it: the language he was last heard in.
        self._speak_reply(text, on_spoken=on_spoken, language=self._last_detected_language)

    def _sounds_like_her_name(self, text: str) -> bool:
        """Whether a segment about to be dropped is the user calling her.

        `avg_logprob` is a poor judge of a proper noun the model has
        never seen. That is exactly why "Yuba" comes back as "Youba",
        "Nuba", "Juba" — and why the segment carrying it scores low and
        is thrown away, leaving the sentence that follows to arrive with
        no wake word in it and be ignored. She then looks broken, which
        is the one failure mode a wake word cannot afford.

        The wake word is configured, not a language pattern, and this is
        the same matcher the detector uses a moment later. Keeping the
        segment decides nothing: everything downstream still applies.
        """
        try:
            return self._carries_the_wake_word((text or "").lower())
        except Exception:
            return False

    def awaiting_spoken_answer(self) -> bool:
        """Whether a question is on the table that a sentence can settle.

        The confidence filter exists to stop her *acting* on a mumble.
        While this is true nothing acts on it: the approval judge reads
        the sentence and fails closed, so anything it cannot read as a
        clear yes is not an approval. The filter is therefore redundant
        here, and actively harmful — the answer to a yes/no question is
        one word, one word carries almost no context, and `avg_logprob`
        over one word is poor by construction. The shorter the answer the
        likelier it is dropped, and "oui" is as short as they come.
        """
        dm = getattr(self, "dialogue_memory", None)
        if dm is None or not hasattr(dm, "peek_pending"):
            return False
        try:
            from ..tools.confirmation import CHANNEL_PAROLE

            waiting = dm.peek_pending()
            return waiting is not None and waiting.channel == CHANNEL_PAROLE
        except Exception as e:
            debug_log(f"pending question not checked: {e}", "voice")
            return False

    def hot_window_duration(self) -> float:
        """How long to keep listening after she finishes speaking.

        The configured default is tuned for a follow-up ("tell me more").
        A person weighing whether to let something happen pauses first, so
        a waiting spoken question widens it. A gesture-only question does
        not: nothing said can settle one, and the extra listening would
        only collect an answer that has to be thrown away.
        """
        default = float(getattr(self.cfg, "hot_window_seconds", 3.0))
        dm = getattr(self, "dialogue_memory", None)
        if dm is None or not hasattr(dm, "peek_pending"):
            return default
        try:
            from ..tools.confirmation import CHANNEL_PAROLE

            waiting = dm.peek_pending()
            if waiting is not None and waiting.channel == CHANNEL_PAROLE:
                return float(getattr(self.cfg, "confirmation_hot_window_sec", 12.0))
        except Exception as e:
            debug_log(f"hot window duration fell back to default: {e}", "voice")
        return default

    def _calculate_audio_energy(self, frames: list) -> float:
        """Calculate RMS energy from audio frames."""
        if not frames or np is None:
            return 0.0
        try:
            audio_data = np.concatenate(frames)
            rms = float(np.sqrt(np.mean(np.square(audio_data))))
            return rms
        except Exception:
            return 0.0

    def _clear_audio_buffers(self) -> None:
        """Reset the utterance being built and the speech state.

        Call this on state transitions to prevent old audio from being
        incorrectly concatenated with new input.
        """
        self._utterance_frames = []
        self._pre_roll.clear()
        self._pending_audio = None
        self.is_speech_active = False
        self._silence_frames = 0

        # Clear wake detection state
        self._wake_timestamp = None

        # The audio queue is left alone: what it holds was said after the
        # utterance just handled, and is his next words, not stale audio.
        debug_log("audio buffers cleared", "voice")

    def _is_speech_frame(self, frame) -> bool:
        """Determine if audio frame contains speech."""
        if np is None:
            return True

        # The room's quiet level, for the echo detector's voice proof
        rms = float(np.sqrt(np.mean(np.square(frame))))
        self._room_floor.hear(rms, captured_at=self._frame_captured_at,
                              hers_playing=self._she_is_heard(), now=time.time())

        if self._vad is None:
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))

        # Use WebRTC VAD
        try:
            stream_rate = getattr(self, "_stream_samplerate", self._samplerate)
            vad_rate = 16000
            target_samples = int(vad_rate * getattr(self, "vad_frame_ms", 20) / 1000)
            if stream_rate == vad_rate:
                vad_audio = frame.flatten()
            else:
                vad_audio = _resample(frame.flatten(), stream_rate, vad_rate)
            if len(vad_audio) < target_samples:
                vad_audio = np.pad(vad_audio, (0, target_samples - len(vad_audio)))
            elif len(vad_audio) > target_samples:
                vad_audio = vad_audio[:target_samples]
            pcm16 = np.clip(vad_audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            return bool(self._vad.is_speech(pcm16, vad_rate))
        except Exception as exc:
            if not self._vad_error_logged:
                self._vad_error_logged = True
                debug_log(f"VAD rejected audio frame: {exc}", "voice")
                print("  ⚠️  Speech detection failed; using audio-level detection. Enable voice_debug for details.", flush=True)
            return rms >= float(getattr(self.cfg, "voice_min_energy", 0.0045))

    def _whisper_vad_setting(self) -> bool:
        """Whether `whisper_vad` asks for faster-whisper's own voice
        detector (Silero), which cuts what holds no speech before the
        decoder hears it.

        Out of an utterance of breath or room noise the decoder writes a
        sentence, which `large-v3-turbo` reports with a `no_speech_prob` of
        0.0, so the confidence filters after it cannot tell; with the
        detector, such an utterance comes back empty. MLX Whisper has no
        detector.
        """
        return bool(getattr(self.cfg, "whisper_vad", True))

    def _voice_detector_on(self) -> bool:
        """Whether the voice detector runs: the backend is faster-whisper
        (MLX has none), the setting asks for it, and it has not failed
        since the listener started (`whisper_vad.py`)."""
        return (self._whisper_backend != "mlx" and self._whisper_vad_setting()
                and not voice_detector_failed())

    def _filter_noisy_segments(self, segments):
        """Filter out low-confidence Whisper segments."""
        min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
        if self.awaiting_spoken_answer():
            # A question is waiting and only the judge decides. Keeping
            # this bar would make the spoken channel unanswerable: the
            # observed "Oui" scored 0.22 against a 0.3 floor, which is
            # what a one-word utterance scores.
            min_confidence = 0.0
        marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
        # Threshold above which a segment is considered non-speech (hallucination during silence).
        # Checked independently of avg_logprob because Whisper can be confident about a
        # hallucinated phrase even when no real speech is present.
        no_speech_threshold = getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
        filtered = []

        for seg in segments:
            # Hard filter: high no_speech_prob means no real speech regardless of logprob.
            if hasattr(seg, 'no_speech_prob') and is_whisper_hallucination(seg.no_speech_prob, no_speech_threshold):
                debug_log(
                    f"segment filtered (no_speech_prob={seg.no_speech_prob:.2f}): '{seg.text[:50]}'",
                    "voice",
                )
                continue

            confidence = None
            if hasattr(seg, 'avg_logprob'):
                confidence = min(1.0, max(0.0, (seg.avg_logprob + 1.0)))
            elif hasattr(seg, 'no_speech_prob'):
                confidence = 1.0 - seg.no_speech_prob

            if (confidence is not None and confidence < min_confidence
                    and self._sounds_like_her_name(seg.text)):
                debug_log(
                    f"segment kept despite confidence={confidence:.2f}: "
                    f"it carries her name", "voice",
                )
                filtered.append(seg)
                continue

            if confidence is not None and confidence < min_confidence:
                if confidence >= marginal_threshold:
                    # Marginal confidence - show in log viewer (not debug)
                    print(f"🔇 Low confidence ({confidence:.2f}): \"{seg.text.strip()[:50]}...\"", flush=True)
                else:
                    # Very low confidence - debug only
                    debug_log(f"segment filtered (confidence={confidence:.2f}): '{seg.text}'", "voice")
                continue

            filtered.append(seg)

        return filtered

    def _transcription_announcement(self, text: str) -> str:
        """The console line for one transcription — heard, or thrown away.

        The repetition guard picks the line, so a transcription it rejects
        is announced as discarded and never as heard. The rejection would
        otherwise reach only the debug log, which is off by default, and
        a console that reports "Heard" while the guard is catching every
        repetition says the opposite of what the system is doing: a log
        that cannot be trusted is a log nobody judges by.

        What is discarded says so, briefly, with its reason. Repeating
        the garbage under a different emoji would fix nothing.
        """
        if not text or not text.strip():
            return ""
        separator = "" if self._first_utterance else f"\n{'─' * 50}"
        self._first_utterance = False
        if self._is_repetitive_hallucination(text):
            apercu = text.strip()[:40]
            return f"{separator}\n  🔇 Discarded (repetition): \"{apercu}…\""
        return f'{separator}\n📝 Heard: "{text}"'

    def _is_repetitive_hallucination(self, text: str) -> bool:
        """
        Detect repetitive hallucinations that Whisper produces on quiet/ambiguous audio.

        Common patterns include repeated single words like "don't don't don't..."
        or repeated short phrases. Also detects character-level repetition patterns
        like "Jろ Jろ Jろ..." which may appear with or without spaces.

        Args:
            text: Transcribed text to check

        Returns:
            True if the text appears to be a hallucination
        """
        import re
        from collections import Counter

        if not text:
            return False

        text_stripped = text.strip()
        if len(text_stripped) < 6:
            return False

        # --- Phrase-level repetition detection ---
        # Whisper loops on phrases as readily as on words, and the word
        # rules below cannot see it: in the field trace that prompted
        # this — "There we go." thirteen times over air-conditioning
        # hum — no single word reaches half the tokens (0.33) and none
        # ever falls twice in a row, because two others sit between each
        # repetition.
        #
        # Three consecutive repeats, not two: he says things twice
        # ("oui oui, vas-y", "c'est bon, c'est bon") and a bar of two
        # would eat him mid-sentence.
        mots = re.findall(r"\w+", text_stripped.lower(), re.UNICODE)
        for n in range(2, 11):
            if len(mots) < n * 3:
                break
            for debut in range(len(mots) - n * 3 + 1):
                bloc = mots[debut:debut + n]
                repeats, pos = 1, debut + n
                while (pos + n <= len(mots)
                       and mots[pos:pos + n] == bloc):
                    repeats += 1
                    pos += n
                if repeats >= 3:
                    debug_log(
                        f"repetitive hallucination detected: phrase "
                        f"{' '.join(bloc)!r} repeated {repeats}x in "
                        f"'{text_stripped[:50]}...'", "voice")
                    return True

        # --- Character-level repetition detection ---
        # Remove all whitespace to detect patterns like "Jろ Jろ Jろ" or "JろJろJろ"
        text_no_space = re.sub(r'\s+', '', text_stripped.lower())

        # Look for repeating patterns of 1-5 characters appearing 3+ times consecutively
        # This catches "JろJろJろJろ" (pattern "Jろ" repeating)
        for pattern_len in range(1, 6):
            if len(text_no_space) < pattern_len * 3:
                continue

            # Check if text is mostly composed of a repeating pattern
            for start in range(pattern_len):
                pattern = text_no_space[start:start + pattern_len]
                if not pattern:
                    continue

                # Count how many times this pattern repeats consecutively from this start position
                remaining = text_no_space[start:]
                repeat_count = 0
                pos = 0
                while pos + pattern_len <= len(remaining) and remaining[pos:pos + pattern_len] == pattern:
                    repeat_count += 1
                    pos += pattern_len

                # If pattern repeats 4+ times and covers most of the string, it's a hallucination
                covered_chars = repeat_count * pattern_len
                coverage = covered_chars / len(text_no_space) if text_no_space else 0

                if repeat_count >= 4 and coverage >= 0.6:
                    debug_log(f"char-level repetition detected: pattern '{pattern}' repeats {repeat_count}x, coverage={coverage:.0%}", "voice")
                    return True

        # --- Word-level repetition detection (existing logic) ---
        words = text_stripped.lower().split()
        if len(words) < 4:
            return False

        # Strip punctuation from words for comparison (handles "word..." vs "word")
        clean_words = [re.sub(r'[^\w]', '', w) for w in words]
        clean_words = [w for w in clean_words if w]  # Remove empty strings

        if len(clean_words) < 4:
            return False

        word_counts = Counter(clean_words)
        most_common_word, most_common_count = word_counts.most_common(1)[0]

        # If a single word makes up more than 50% of all words and appears 4+ times
        if most_common_count >= 4 and most_common_count / len(clean_words) > 0.5:
            debug_log(f"repetitive hallucination detected: '{most_common_word}' repeated {most_common_count}x in '{text[:50]}...'", "voice")
            return True

        # Check for repeated consecutive sequences (e.g., "don don don" or "stop stop stop")
        # Look for any word repeated 3+ times consecutively
        consecutive_count = 1
        for i in range(1, len(clean_words)):
            if clean_words[i] == clean_words[i-1]:
                consecutive_count += 1
                if consecutive_count >= 3:
                    debug_log(f"consecutive repetition detected: '{clean_words[i]}' repeated {consecutive_count}+ times", "voice")
                    return True
            else:
                consecutive_count = 1

        return False

    def _check_query_timeout(self, now: Optional[float] = None) -> None:
        """Check if there's a pending query that has timed out, and check hot
        window expiry. ``now`` is when the audio being read was captured: the
        collection's silence is counted in that time, so audio the loop
        reads late, after a stall, still has his next words in time to
        complete the question. With no audio to read, the wall clock."""
        if self.state_manager.check_collection_timeout(now):
            query = self.state_manager.clear_collection()
            # No letter or digit, in any script, is nothing to answer: her
            # name alone, and at most the punctuation Whisper made out of the
            # quiet after it.
            if any(ch.isalnum() for ch in query):
                self._hand_over(query)
            else:
                self._nothing_after_the_wake_word()

        # Also check hot window expiry - this ensures the timeout is enforced
        # even when there's no audio being processed
        self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)

    def _nothing_after_the_wake_word(self) -> None:
        """His silence ran out on a question with nothing in it: her name
        alone, then nothing. Nothing is handed over, the console says so, and
        the published state goes back to what she is doing; while he
        dictates, the dictation's state stands. A reply on its way owns the
        tune, which plays on; otherwise the tune was this question's, and it
        stops."""
        reply_on_its_way = self._a_reply_is_on_its_way()
        if not reply_on_its_way:
            self._silence_the_tune()
        if self._dictation_active:
            state = None
        elif self.tts is not None and self.tts.is_speaking():
            state = JarvisState.SPEAKING
        elif reply_on_its_way:
            state = JarvisState.THINKING
        else:
            state = JarvisState.IDLE
        if state is not None:
            get_jarvis_state().set_state(state)
        debug_log("nothing after the wake word: nothing handed over, state "
                  f"{state.value if state is not None else 'left to the dictation'}", "voice")
        try:
            print("💤 Nothing after the wake word, returning to wake word mode\n", flush=True)
        except Exception:
            pass

    def _reset_audio_health(self, now=None):
        now = time.monotonic() if now is None else now
        self._audio_started = now
        self._last_audio_callback = now
        self._last_health_check = now
        self._last_capture_check = now
        self._audio_peak = 0.0
        self._audio_frames_seen = 0
        self._speech_frames_seen = 0
        self._audio_dropped = 0
        self._callback_status = ''
        self._audio_callback_error = ''
        self._audio_health_warning = None

    def _check_audio_health(self, now=None):
        """Report capture failures outside the real-time callback, without recording audio."""
        now = time.monotonic() if now is None else now
        if self._dictation_active:
            self._reset_audio_health(now)
            return
        if now - self._last_health_check < 5:
            return
        self._last_health_check = now
        warning = None
        if now - self._last_audio_callback >= 5:
            warning = 'No microphone callbacks in the last 5 seconds'
        elif self._audio_peak <= 1e-7 and now - self._audio_started >= 10:
            warning = 'Microphone is delivering silent samples'
        if warning and warning != self._audio_health_warning:
            print(f"  ⚠️  {warning}. Check the selected input, mute and recording permissions.", flush=True)
            if sys.platform.startswith('linux'):
                print("     🎤 Check PipeWire/PulseAudio recording-source routing (pavucontrol or wpctl status); avoid monitor/output sources.", flush=True)
        elif not warning and self._audio_health_warning:
            print("  ✅ Microphone audio is arriving again.", flush=True)
        self._audio_health_warning = warning
        if self._callback_status or self._audio_callback_error or self._audio_dropped:
            print(f"  ⚠️  Audio capture: {self._callback_status or self._audio_callback_error or 'queue full'}; {self._audio_dropped} blocks dropped.", flush=True)
            self._callback_status = self._audio_callback_error = ''
            self._audio_dropped = 0
        if self.cfg.voice_debug:
            debug_log(
                f"Audio capture: callbacks={self._callback_count}, frames={self._audio_frames_seen}, "
                f"speech_frames={self._speech_frames_seen}, peak={self._audio_peak:.6f}, "
                f"rate={self._stream_samplerate} Hz", "voice",
            )
        self._audio_peak = 0.0

    def _audio_frames(self, buf):
        """Keep native-rate frame boundaries across arbitrary callback block sizes."""
        if buf.ndim > 1:
            mono = buf[:, 0] if buf.shape[1] == 1 else buf.mean(axis=1)
        else:
            mono = buf.flatten()
        if mono.size:
            self._audio_peak = max(self._audio_peak, float(np.max(np.abs(mono))))
        if self._pending_audio is not None and self._pending_audio.size:
            mono = np.concatenate((self._pending_audio, mono))
        frame_samples = max(1, self._frame_samples)
        count = len(mono) // frame_samples
        end = count * frame_samples
        self._pending_audio = mono[end:].copy() if end < len(mono) else None
        self._audio_frames_seen += count
        return [mono[start:start + frame_samples] for start in range(0, end, frame_samples)]

    def _on_audio(self, indata, frames, time_info, status):
        """Audio callback from sounddevice."""
        try:
            self._last_audio_callback = time.monotonic()
            self._callback_count += 1
            if status:
                self._callback_status = str(status)
            if self._should_stop or self._dictation_active:
                return
            chunk = (indata.copy() if hasattr(indata, "copy") else indata)
            try:
                self._audio_q.put_nowait((time.time(), chunk))
            except queue.Full:
                self._audio_dropped += 1
        except Exception as exc:
            self._audio_callback_error = str(exc)

    def _determine_whisper_backend(self) -> str:
        """Determine which Whisper backend to use based on config and availability."""
        backend_pref = getattr(self.cfg, "whisper_backend", "auto")

        if backend_pref == "mlx":
            if MLX_WHISPER_AVAILABLE:
                return "mlx"
            debug_log("MLX Whisper requested but not available, falling back to faster-whisper", "voice")
            return "faster-whisper"

        if backend_pref == "faster-whisper":
            return "faster-whisper"

        # Auto mode: prefer MLX on Apple Silicon
        if MLX_WHISPER_AVAILABLE and _is_apple_silicon():
            return "mlx"

        return "faster-whisper"

    def _apply_whisper_load_success(
        self, model_name: str, try_device: str, try_compute: str,
        device: str, compute: str, cpu_threads: int,
        context: str = "",
    ) -> str:
        """Record state and print diagnostics after a successful Whisper model load.

        Returns the resolved device string.
        """
        ct2_model = getattr(self.model, "model", None)
        resolved_device = str(getattr(ct2_model, "device", try_device)).lower()
        debug_log(
            f"faster-whisper initialised{context}: name={model_name}, "
            f"device={resolved_device}, compute={try_compute}, "
            f"cpu_threads={cpu_threads}",
            "voice",
        )
        self._whisper_device = resolved_device

        if try_device != device and device in ("auto", "cuda"):
            print("     ⚠️  CUDA not available, using CPU (this may be slower)", flush=True)
            print("     💡 Tip: Install NVIDIA CUDA toolkit for faster speech recognition", flush=True)
        if try_compute != compute:
            print(f"     ⚠️  Using '{try_compute}' compute type ('{compute}' not supported)", flush=True)
        if resolved_device == "cpu":
            print(f"     ⚡ CPU mode: using {cpu_threads} threads with optimised decoding", flush=True)

        suffix = f" ({context})" if context else ""
        print(f"     🎤 Whisper '{model_name}' loaded on {resolved_device}{suffix}", flush=True)
        return resolved_device

    # How long "Listening!" may wait for the warm-ups, counted from when they
    # began. A slow or unreachable model server, or a voice still being
    # downloaded, must not keep her deaf: past it, the first engagement pays.
    _WARM_UP_BUDGET_SECONDS = 60.0

    def _await_warm_ups(self) -> None:
        """Wait, within one budget, for what the first engagement needs: the
        LLM warm-ups and the voice. She says she is listening only once she
        can answer out loud; a voice still warming would make the first
        reply after a launch wait for it."""
        deadline = (getattr(self, "_llm_warmup_started_at", None) or time.time()) \
            + self._WARM_UP_BUDGET_SECONDS
        warmup_threads = getattr(self, "_llm_warmup_threads", [])
        for t in warmup_threads:
            t.join(timeout=max(0.0, deadline - time.time()))
        still_warming = any(t.is_alive() for t in warmup_threads)

        if warmup_threads:
            results = getattr(self, "_llm_warmup_results", {})

            # Trailing space after ⚠️ intentional: the warning glyph renders
            # narrower than 🧠/💬, so the pad keeps columns aligned.
            def _print_status(role_key: str, label: str, ok_icon: str) -> None:
                entry = results.get(role_key)
                if entry is None:
                    return
                name, ok = entry
                icon = ok_icon if ok else "⚠️ "
                status = "ready" if ok else "warmup failed — will load on first use"
                print(f"     {icon} {label} '{name}' {status}", flush=True)

            _print_status("chat", "Chat model", "💬")
            _print_status("judge", "Intent judge", "🧠")
            _print_status("router", "Tool router", "🔧")

        if self.tts is not None and getattr(self.tts, "enabled", False):
            engine = self.cfg.tts_engine
            try:
                ready = self.tts.wait_until_ready(max(0.0, deadline - time.time()))
            except Exception as e:
                # Like every warm-up, the voice's never keeps her deaf: it
                # is taken for one still warming, and the first reply pays.
                debug_log(f"voice warm-up: wait failed ({type(e).__name__})", "voice")
                ready = None
            debug_log(f"voice warm-up: {ready}", "voice")
            if ready is True:
                print(f"     🔊 Voice '{engine}' ready", flush=True)
            elif ready is False:
                print(f"     ⚠️  Voice '{engine}' could not load", flush=True)
            else:
                still_warming = True

        if still_warming:
            debug_log(f"warm-ups still running after {self._WARM_UP_BUDGET_SECONDS:.0f}s "
                      f"— continuing without", "voice")
            print("     ⏳ Some models still warming — continuing anyway", flush=True)

    def _start_llm_warmup(self) -> list[threading.Thread]:
        """Pre-load the chat, intent judge and tool router models via the
        active backend.

        Warmup goes through ``LLMBackend.warm_up``, so it pages models into
        Ollama's resident memory on the Ollama path and opens the connection
        on a remote provider with one minimal, billed request. Starts up to
        three daemon threads concurrently so warmup overlaps with Whisper
        initialisation. One model is warmed once, whatever roles it plays:
        when chat and judge point at the same model, a single warmup covers
        both, and a router on the chat or judge model rides theirs. A router
        on a model of its own is warmed on its own prompt
        (``warm_up_tool_router``), which no other warm-up computes.

        Results land in ``self._llm_warmup_results`` keyed by role. The
        caller joins the returned threads with a shared deadline before
        announcing "Listening!" so the ready state actually means ready.
        """
        self._llm_warmup_results: dict[str, tuple[str, bool]] = {}

        if _is_low_power_mode_enabled(self.cfg):
            print("     🌱 Low power mode: LLM warmup skipped", flush=True)
            debug_log("low power mode enabled: skipping LLM warmup", "voice")
            return []

        chat_model = str(getattr(self.cfg, "llm_chat_model", "") or "").strip()
        chat_timeout = max(float(getattr(self.cfg, "llm_tools_timeout_sec", 8.0)), 60.0)
        judge = self._intent_judge
        judge_model = judge.config.model if judge is not None else ""
        shared_judge = bool(chat_model) and judge_model == chat_model

        # Tool router — warmed only under the LLM selection strategy, on the
        # prompt its first call will send: loading a model computes nothing,
        # and the router's system prompt and catalogue (a few thousand
        # tokens), left to the first exchange, would be read on a model that
        # has never run while the rest of start-up holds the GPU. A router
        # on the chat or judge model rides that model's warm-up instead:
        # their calls ask Ollama for another context size than the router's,
        # Ollama loads the model again for it and the computed prompt goes
        # with the old load, and two warm-ups of one model at two sizes
        # would race to decide which load stays. An empty
        # `tool_router_model` means "reuse the intent-judge model or the
        # chat model as a last resort"; `resolve_tool_router_model` is the
        # reply engine's own resolution, so warmup targets whatever the
        # engine will actually call.
        strategy = str(getattr(self.cfg, "tool_selection_strategy", "") or "").lower()
        from ..reply.engine import resolve_tool_router_model, warm_up_tool_router
        router_model = resolve_tool_router_model(self.cfg) if strategy == "llm" else ""
        shared_router = bool(router_model) and router_model in {chat_model, judge_model}

        threads: list[threading.Thread] = []

        if chat_model:
            def _warm_chat() -> None:
                ok = warm_up_chat_model(self.cfg, chat_model, timeout=chat_timeout)
                self._llm_warmup_results["chat"] = (chat_model, ok)
                # When chat and judge share a model, one warmup covers both.
                if shared_judge:
                    self._llm_warmup_results["judge"] = (chat_model, ok)
                if shared_router and router_model == chat_model:
                    self._llm_warmup_results["router"] = (chat_model, ok)

            threads.append(threading.Thread(target=_warm_chat, daemon=True, name="warmup-chat"))

        if judge is not None and not shared_judge:
            def _warm_judge() -> None:
                ok = judge.warm_up()
                self._llm_warmup_results["judge"] = (judge_model, ok)
                if shared_router and router_model == judge_model:
                    self._llm_warmup_results["router"] = (judge_model, ok)

            threads.append(threading.Thread(target=_warm_judge, daemon=True, name="warmup-judge"))

        if router_model and not shared_router:
            def _warm_router() -> None:
                ok = warm_up_tool_router(
                    self.cfg,
                    timeout_sec=chat_timeout,
                    keep_alive=_ollama_keep_alive_for_power_mode(self.cfg),
                )
                self._llm_warmup_results["router"] = (router_model, ok)

            threads.append(threading.Thread(target=_warm_router, daemon=True, name="warmup-router"))

        for t in threads:
            t.start()

        debug_log(
            f"LLM warmup started (chat={chat_model or 'n/a'}, "
            f"judge={judge_model or 'n/a'}, router={router_model or 'n/a'}, "
            f"shared_judge={shared_judge}, shared_router={shared_router}, "
            f"keep_alive={_ollama_keep_alive_for_power_mode(self.cfg)})",
            "voice",
        )
        return threads

    def _weather_example(self, wake_title: str) -> str:
        """Return the weather query example for the startup banner.

        Shows the plain form when a location source is configured, or the
        [your city] placeholder form so the user knows to supply a city.
        """
        location_enabled = getattr(self.cfg, "location_enabled", True)
        location_auto_detect = getattr(self.cfg, "location_auto_detect", True)
        location_ip_address = getattr(self.cfg, "location_ip_address", None)
        location_known = (
            location_enabled
            and (location_auto_detect or bool(location_ip_address))
            and is_location_available()
        )
        if location_known:
            return f"\"How's the weather, {wake_title}?\""
        return f"\"How's the weather in [your city], {wake_title}?\""

    def _check_microphone_permission(self, deferred: bool) -> _Permission:
        """Windows: whether the selected microphone can be recorded from,
        tried with a brief recording before capture opens.

        BLOCKED, with what to change printed, when Windows blocks it: the
        probe hangs past five seconds, or, at start-up, it fails with an
        error that reads as access denied. Listening then ends. Any other
        error is FAILED: the wait for the microphone begins here, if it is
        not on already, and the microphone is probed again once it is there.
        ``deferred`` is the probe of a microphone that was waited for and has
        just appeared: only a hang is BLOCKED then, since a device just
        plugged in can fail with the same codes before it is ready, and its
        error is said once per wait.

        This catches privacy settings that silently block audio access. A
        5-second timeout prevents indefinite hangs when Windows blocks the
        audio device at the system level without raising an error. Uses
        InputStream (not sd.rec) so the stream can be explicitly closed on
        timeout, avoiding resource leaks that could block later audio init.
        """
        try:
            if self._first_time("permission check"):
                print("  🔐 Checking microphone permission...", flush=True)
            mic_ok = threading.Event()
            mic_error: list = [None]

            def _mic_check():
                # Deliberately NOT under portaudio_lock: this probe's
                # open/start can hang indefinitely when Windows blocks
                # mic access at the system level (that is what the 5s
                # timeout below is for), and hanging while holding the
                # process-wide lock would freeze every other audio user
                # (listener, dictation, TTS). It runs before capture first
                # opens, or mid-session once a microphone that was waited
                # for appears. Mid-session it runs only on a tick when no
                # stream is open (`_wait_for_microphone`), so it does not
                # overlap her voice, the tune or a dictation, though one of
                # them may still open while it runs. The quick stop/close
                # after a successful start stays guarded. The microphone
                # is resolved and opened inside `unguarded_open`, so the
                # device list is not read afresh between the two, and the
                # stream is tracked, so it is never read afresh under it.
                stream = None
                try:
                    with unguarded_open():
                        with portaudio_lock:
                            stream_kwargs = self._selected_input(sd.query_devices())
                        if stream_kwargs is None:
                            raise OSError("the selected microphone is not in the device list")
                        stream, _, _ = _open_input_stream(
                            self._samplerate, 100, stream_kwargs, serialise=False,
                        )
                    stream.start()
                    time.sleep(0.15)
                    with portaudio_lock:
                        stream.stop()
                        stream.close()
                    stream = None
                    mic_ok.set()
                except Exception as exc:
                    mic_error[0] = exc
                    if stream is not None:
                        try:
                            with portaudio_lock:
                                stream.close()
                        except Exception:
                            pass

            check_thread = threading.Thread(target=_mic_check, daemon=True)
            check_thread.start()
            check_thread.join(timeout=5.0)

            if check_thread.is_alive():
                # Do NOT abort/close the stream from this thread: the
                # check thread may still be blocked inside start()/stop()
                # on it, and closing a stream under another thread's feet
                # is a native use-after-free that aborts the whole app on
                # Windows. Abandon it: the daemon check thread
                # will finish the stop/close itself if it ever unblocks.
                debug_log("microphone permission check timed out after 5s", "voice")
                print("  ❌ Microphone permission check timed out (access blocked by Windows)", flush=True)
                _print_access_blocked_box()
                return _Permission.BLOCKED
            elif mic_error[0] is not None:
                e = mic_error[0]
                error_str = str(e).lower()
                if not deferred and (
                    "unapproved" in error_str or "denied" in error_str
                    or "access" in error_str or "-9999" in error_str
                ):
                    print(f"  ❌ Microphone permission check failed: {e}", flush=True)
                    _print_access_blocked_box()
                    return _Permission.BLOCKED
                # The wait begins before the error is said, so the error
                # that starts it counts as said for it.
                self._say_microphone_lost(f"the permission probe failed: {e}")
                if self._first_time("permission failure"):
                    print(f"  ❌ Microphone permission check failed (trying again): {e}", flush=True)
                return _Permission.FAILED
            elif mic_ok.is_set():
                print("  ✅ Microphone permission OK", flush=True)
            else:
                print("  ⚠️  Microphone returned empty audio", flush=True)
        except Exception as e:
            debug_log(f"microphone permission check error: {e}", "voice")
            print(f"  ⚠️  Microphone check error: {e}", flush=True)
        return _Permission.OK

    def run(self) -> None:
        """Main voice listening loop."""
        if sd is None:
            debug_log("sounddevice not available", "voice")
            print("  ❌ Audio system not available - sounddevice failed to load", flush=True)
            return

        # Verify PortAudio is working by querying devices (catches Windows DLL issues)
        try:
            with portaudio_lock:
                devices = sd.query_devices()
            input_devices = [d for d in devices if d.get('max_input_channels', 0) > 0]
            debug_log(f"PortAudio initialised: {len(input_devices)} input device(s) found", "voice")
        except Exception as e:
            debug_log(f"PortAudio device query failed: {e}", "voice")
            print(f"  ❌ Audio system error: {e}", flush=True)
            print("     PortAudio may not be properly installed", flush=True)
            if sys.platform == 'linux':
                print("     On Linux, ensure PortAudio is installed: sudo apt install libportaudio2", flush=True)
            return

        # The same input for the permission probe and continuous capture. A
        # microphone that is not connected is waited for
        # (`_listen_until_stopped`), and no other input is ever opened in its
        # place. An index that names no input has no microphone to wait for.
        try:
            stream_kwargs = self._selected_input(devices)
        except IndexNamesNoInput:
            print("  ❌ Selected microphone not found. Choose an available input in Settings.", flush=True)
            debug_log("the configured input index names no input device", "voice")
            return
        if stream_kwargs is None:
            self._say_microphone_missing()

        # Windows: a microphone its privacy settings block is caught before
        # capture. One that is not connected yet, or that fails the probe
        # for another reason, is waited for and checked once it is there.
        permission_checked = sys.platform != 'win32'
        if sys.platform == 'win32' and stream_kwargs is not None:
            verdict = self._check_microphone_permission(deferred=False)
            if verdict is _Permission.BLOCKED:
                return
            permission_checked = verdict is _Permission.OK

        # Kick off LLM warmups in parallel with Whisper load so the first
        # user engagement doesn't pay cold-load cost on either model. All
        # warmup output (Whisper + LLMs) is indented under this header to
        # visually group the phase.
        print("  🔥 Warming up models...", flush=True)
        self._llm_warmup_started_at = time.time()
        self._llm_warmup_threads = self._start_llm_warmup()

        # Determine and initialise Whisper backend
        self._whisper_backend = self._determine_whisper_backend()
        model_name = getattr(self.cfg, "whisper_model", "small")

        # Validate large-v3-turbo support for faster-whisper backend
        if model_name == "large-v3-turbo" and self._whisper_backend != "mlx":
            if not is_faster_whisper_turbo_supported():
                debug_log(
                    "faster-whisper does not support large-v3-turbo, "
                    "falling back to large-v3", "voice",
                )
                print(
                    "  ⚠️  large-v3-turbo is not supported by the installed Whisper engine, "
                    "using large-v3 instead", flush=True,
                )
                model_name = "large-v3"

        if self._whisper_backend == "mlx":
            if not MLX_WHISPER_AVAILABLE:
                debug_log("MLX Whisper not available", "voice")
                print("  ❌ MLX Whisper not available. Install with: pip install mlx-whisper", flush=True)
                return

            self._mlx_model_repo = _get_mlx_model_repo(model_name)
            print(f"     🎤 Loading MLX Whisper '{model_name}' (Apple Silicon GPU)...", flush=True)

            max_retries = 4
            for attempt in range(max_retries + 1):
                try:
                    # Pre-load the model by doing a warmup transcription.
                    # Use low-amplitude noise (not silence) so the decoder actually runs —
                    # silent audio trips the no-speech short-circuit and leaves the decode
                    # path cold, so the first real utterance still pays the full cost.
                    if np is not None:
                        _predownload_mlx_snapshot(model_name, self._mlx_model_repo)
                        rng = np.random.default_rng(0)
                        warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                        _ = mlx_whisper.transcribe(
                            warmup_audio,
                            path_or_hf_repo=self._mlx_model_repo,
                            language=self.cfg.whisper_language,
                        )
                        debug_log(f"MLX Whisper model pre-loaded: repo={self._mlx_model_repo}", "voice")

                    print(f"     🎤 MLX Whisper '{model_name}' ready (Apple Silicon GPU)", flush=True)
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )
                    if is_rate_limited and attempt < max_retries:
                        wait = 2 ** (attempt + 1)
                        debug_log(f"rate limited loading MLX Whisper (attempt {attempt + 1}): {e}", "voice")
                        print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({attempt + 1}/{max_retries})...", flush=True)
                        time.sleep(wait)
                        continue
                    debug_log(f"failed to initialise MLX Whisper: {e}", "voice")
                    print(f"  ❌ Failed to initialise MLX Whisper: {e}", flush=True)
                    if is_rate_limited:
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                    return
        else:
            # faster-whisper backend
            if not FASTER_WHISPER_AVAILABLE:
                debug_log("faster-whisper not available", "voice")
                print("  ❌ faster-whisper not available. Install with: pip install faster-whisper", flush=True)
                return

            device = getattr(self.cfg, "whisper_device", "auto")
            compute = getattr(self.cfg, "whisper_compute_type", "int8")

            # On Windows, probe for CUDA runtime libraries before trying to
            # use them. faster-whisper/CTranslate2 lazily loads cuBLAS and
            # cuDNN during transcription, so without this check a model
            # that loaded fine on cuda will crash on the first audio chunk.
            resolved_device, missing_libs = _probe_windows_cuda_libraries(device)
            if missing_libs:
                _print_cuda_unavailable_hint(missing_libs)
            device = resolved_device

            # Build list of (device, compute_type) combinations to try
            # This handles both compute type fallbacks and CUDA -> CPU fallbacks
            configs_to_try = []

            # Start with preferred config
            compute_types = [compute]
            if compute == "int8":
                compute_types.extend(["float16", "float32"])
            elif compute == "float16":
                compute_types.append("float32")

            # Add preferred device with all compute types
            for ct in compute_types:
                configs_to_try.append((device, ct))

            # If device is "auto" or "cuda", add CPU fallback configs
            # This handles Windows without CUDA libraries
            if device in ("auto", "cuda"):
                for ct in compute_types:
                    configs_to_try.append(("cpu", ct))

            last_error = None
            used_device = device
            used_compute = compute
            for try_device, try_compute in configs_to_try:
                try:
                    cpu_threads = (os.cpu_count() or 4) if try_device in ("cpu", "auto") else 0
                    print(f"     🎤 Loading Whisper '{model_name}' (device={try_device}, compute={try_compute})...", flush=True)
                    _predownload_whisper_snapshot(model_name)
                    self.model = WhisperModel(
                        model_name, device=try_device, compute_type=try_compute,
                        cpu_threads=cpu_threads,
                    )
                    self._apply_whisper_load_success(
                        model_name, try_device, try_compute,
                        device, compute, cpu_threads,
                    )
                    used_device = try_device
                    used_compute = try_compute
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    error_str = str(e).lower()

                    # Check if this is a CUDA/GPU-related error that we should fall back from
                    is_cuda_error = any(x in error_str for x in [
                        "cuda", "cublas", "cudnn", "gpu", "nvidia",
                        ".dll is not found", "library", "ctypes"
                    ])
                    is_compute_error = any(x in error_str for x in [
                        "compute type", "int8", "float16"
                    ])

                    if is_cuda_error or is_compute_error:
                        debug_log(f"config ({try_device}, {try_compute}) failed, trying fallback: {e}", "voice")
                        continue

                    # Check for corrupted model cache (e.g. interrupted download)
                    is_corrupted_cache = "unable to open file" in error_str

                    if is_corrupted_cache:
                        debug_log(f"detected corrupted Whisper model cache: {e}", "voice")
                        print("  ⚠️  Whisper model cache appears corrupted, attempting recovery...", flush=True)

                        cache_cleared = _clear_corrupted_whisper_cache(str(e))
                        if cache_cleared:
                            try:
                                print(f"     🎤 Re-downloading Whisper '{model_name}'...", flush=True)
                                _predownload_whisper_snapshot(model_name)
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads,
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="recovered",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                break
                            except Exception as retry_e:
                                debug_log(f"retry after cache clear also failed: {retry_e}", "voice")
                                print(f"  ❌ Failed to load Whisper model after cache recovery: {retry_e}", flush=True)
                                continue
                        else:
                            debug_log("could not clear corrupted cache automatically", "voice")
                            print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                            print("  💡 Try manually deleting the Whisper model cache directory and restarting", flush=True)
                            continue
                    # Check for rate limiting (HTTP 429) — check string and response status code
                    # (HfHubHTTPError may carry the status on .response without "429" in str(e))
                    is_rate_limited = (
                        any(x in error_str for x in ["429", "too many requests", "rate limit"])
                        or getattr(getattr(e, "response", None), "status_code", None) == 429
                    )

                    if is_rate_limited:
                        _max_retries = 4
                        _backoff = 2
                        debug_log(f"rate limited loading Whisper model: {e}", "voice")
                        retry_succeeded = False
                        for retry_num in range(1, _max_retries + 1):
                            wait = _backoff ** retry_num
                            print(f"  ⏳ Rate limited by HuggingFace, retrying in {wait}s ({retry_num}/{_max_retries})...", flush=True)
                            time.sleep(wait)
                            try:
                                _predownload_whisper_snapshot(model_name)
                                self.model = WhisperModel(
                                    model_name, device=try_device, compute_type=try_compute,
                                    cpu_threads=cpu_threads,
                                )
                                self._apply_whisper_load_success(
                                    model_name, try_device, try_compute,
                                    device, compute, cpu_threads,
                                    context="rate-limit retry",
                                )
                                used_device = try_device
                                used_compute = try_compute
                                last_error = None
                                retry_succeeded = True
                                break
                            except Exception as retry_e:
                                debug_log(f"rate-limit retry {retry_num} failed: {retry_e}", "voice")
                                last_error = retry_e
                        if retry_succeeded:
                            break
                        debug_log(f"gave up after {_max_retries} rate-limit retries", "voice")
                        print(f"  ❌ Failed to load Whisper model after {_max_retries} retries: {last_error}", flush=True)
                        print("  💡 HuggingFace is rate limiting downloads. Please wait a few minutes and restart.", flush=True)
                        return
                    else:
                        # For other errors (model not found, etc.), don't try fallbacks
                        debug_log(f"failed to initialise faster-whisper: {e}", "voice")
                        print(f"  ❌ Failed to load Whisper model: {e}", flush=True)
                        return

            if last_error is not None:
                debug_log(f"failed to initialise faster-whisper with any config: {last_error}", "voice")
                print(f"  ❌ Failed to load Whisper model: {last_error}", flush=True)
                return

            # Warm up faster-whisper so the first real utterance doesn't pay
            # the cold-decode cost. Use low-amplitude noise rather than pure
            # silence — silence trips faster-whisper's no-speech short-circuit
            # and the decoder never actually runs. Mirror the real transcribe
            # parameters so beam search, language detection, and the timestamp
            # path are all exercised here instead of on the user's first word.
            if np is not None and self.model is not None:
                try:
                    cpu_mode = self._whisper_device == "cpu"
                    rng = np.random.default_rng(0)
                    warmup_audio = rng.standard_normal(self._samplerate).astype(np.float32) * 0.01
                    try:
                        segments_iter, _ = self.model.transcribe(
                            warmup_audio,
                            language=self.cfg.whisper_language,
                            vad_filter=False,
                            condition_on_previous_text=not cpu_mode,
                            without_timestamps=cpu_mode,
                        )
                    except TypeError:
                        segments_iter, _ = self.model.transcribe(warmup_audio, language=self.cfg.whisper_language)
                    for _ in segments_iter:
                        pass
                    debug_log("faster-whisper warmup transcription complete", "voice")
                except Exception as e:
                    debug_log(f"faster-whisper warmup failed: {e}", "voice")

            # The voice detector loads its own model on first use. The decode
            # above ran without it, which would have cut this noise away, so
            # it is loaded here; one that cannot load is announced now rather
            # than on his first words. Each start-up tries it afresh.
            if self.model is not None and self._whisper_vad_setting():
                prepare_voice_detector()

        self._await_warm_ups()

        # Audio parameters
        self._configure_frames()
        frame_ms = self._frame_ms

        debug_log(f"audio params: sample_rate={self._samplerate}, frame_ms={frame_ms}, frame_samples={self._frame_samples}", "voice")
        debug_log(f"VAD: enabled={bool(self._vad is not None)}, aggressiveness={getattr(self.cfg, 'vad_aggressiveness', 2)}", "voice")

        # Audio device setup
        if self.cfg.voice_debug:
            debug_log("available input devices:", "voice")
            try:
                with portaudio_lock:
                    listed = sd.query_devices()
                for idx, dev in enumerate(listed):
                    try:
                        max_in = int(dev.get("max_input_channels", 0))
                    except Exception:
                        max_in = 0
                    if max_in > 0:
                        name = dev.get("name")
                        rate = dev.get("default_samplerate")
                        debug_log(f"  [{idx}] {name} (channels={max_in}, default_sr={rate})", "voice")
            except Exception:
                pass

        self._stream_samplerate = self._samplerate
        self._listen_until_stopped(permission_checked)

    def _listen_until_stopped(self, permission_checked: bool) -> None:
        """Capture from the selected microphone until the listener stops.

        A microphone that is not connected, that will not open or start, or
        whose stream dies, is waited for and opened again
        (`_wait_for_microphone`); no other input is ever opened in its place.
        Listening ends only on an error that reads as access denied before any
        wait (`_keep_trying_after`) or a permission probe that hangs.
        ``permission_checked`` is False on Windows until the permission probe
        has passed: it then runs once the microphone is there.
        """
        frame_ms = self._frame_ms
        ready = False
        while not self._should_stop:
            if self._microphone_wait is not None:
                if not self._wait_for_microphone(needs_quiet=not permission_checked):
                    return
                if not permission_checked:
                    verdict = self._check_microphone_permission(deferred=True)
                    if verdict is _Permission.BLOCKED:
                        return
                    if verdict is _Permission.FAILED:
                        continue
                    permission_checked = True
            if self._first_time("device"):
                self._announce_input_device()
            try:
                stream = self._open_capture(frame_ms, announce=not ready)
            except Exception as e:
                if not self._keep_trying_after(e):
                    return
                continue
            if stream is None:
                self._say_microphone_missing()
                continue

            opened_at = time.monotonic()
            self._reset_audio_health(opened_at)
            lost = None
            started = False
            try:
                with _serialised_stream(stream):
                    # Verify stream is actually recording (helps catch permission issues)
                    if not stream.active:
                        with portaudio_lock:
                            stream.start()
                    started = True

                    # Show ready message only after stream is confirmed active
                    if not ready:
                        self._announce_listening()
                        ready = True

                    lost = self._read_microphone(stream)
                    if lost is not None:
                        # Aborted rather than stopped: a dead device has no
                        # buffers worth waiting for.
                        try:
                            with portaudio_lock:
                                stream.abort()
                        except Exception:
                            pass
            except Exception as e:
                if started:
                    raise
                # The stream would not start, and is closed.
                if not self._keep_trying_after(e):
                    return
                continue
            if lost is None:
                return
            self._say_microphone_lost(lost, lived_for=time.monotonic() - opened_at)
            self._forget_unfinished_audio()

    def _announce_listening(self) -> None:
        """The ready banner, then the published state and the reply thread:
        capture has started for the first time."""
        wake_word = getattr(self.cfg, "wake_word", "jarvis").lower()
        wake_title = wake_word.title()
        print(f"\n{'─' * 50}\n🎙️  Listening! Try:", flush=True)
        print(f"      {self._weather_example(wake_title)}", flush=True)
        print(f"      \"I just ate a Big Mac, {wake_title}.\"", flush=True)
        print(f"      \"What are you thinking, {wake_title}?\"", flush=True)
        print(f"      \"What do you know about me, {wake_title}?\"", flush=True)

        # Small-model disclaimer: SMALL models can't infer your intent
        # from vague prompts, but they can still execute complex flows
        # if you spell out the steps. Assume the model is dumb and lay
        # things out for it. Classification lives in model_variants so
        # it stays in sync when supported models change.
        from ..reply.prompts.model_variants import detect_model_size, ModelSize
        chat_model_name = str(getattr(self.cfg, "llm_chat_model", "") or "").strip()
        if chat_model_name and detect_model_size(chat_model_name) == ModelSize.SMALL:
            print(
                f"  ⚠️  Small model in use ({chat_model_name}). Assume it can't infer. Spell out the steps for anything more involved:",
                flush=True,
            )
            print(
                f"      \"Tell me tomorrow's weather, then find local events for tomorrow, then recommend ones that suit the weather, {wake_title}.\"",
                flush=True,
            )

        # Chrome MCP tip: the chrome MCP exposes a `navigate` tool that
        # takes a URL. Vague phrasing like "Open YouTube" forces the model
        # to guess a URL; "Navigate to youtube.com" maps directly to the
        # tool's argument and is more reliable on small models.
        try:
            from ..tools.registry import get_cached_mcp_tools
            mcp_tool_names = list(get_cached_mcp_tools().keys())
            has_chrome_mcp = any("chrome" in name.lower() for name in mcp_tool_names)
        except Exception:
            has_chrome_mcp = False
        if has_chrome_mcp:
            print(
                f"  🌐 Chrome MCP detected. Name the destination URL so the browser tool can act directly:",
                flush=True,
            )
            print(
                f"      \"Navigate to youtube.com, {wake_title}.\"",
                flush=True,
            )

        # Awake and ready, waiting for the wake word
        get_jarvis_state().set_state(JarvisState.IDLE)

        # Before the loop: a narration handed over by another thread is
        # said by the reply thread whether or not anyone speaks.
        self._start_replies()

    def _read_microphone(self, stream) -> Optional[str]:
        """Take what the microphone hears until the listener stops (None) or
        its stream is taken for dead (why)."""
        callbacks_at_open = self._callback_count
        while not self._should_stop:
            now = time.monotonic()
            self._check_audio_health(now)
            if self._microphone_wait is not None and self._callback_count > callbacks_at_open:
                self._say_microphone_back()
            lost = self._capture_lost(stream, now)
            if lost is not None:
                return lost

            try:
                entry = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                # Critical: Check timeouts even when no audio is being received
                # This ensures hot window expiry fires reliably
                self._check_query_timeout()
                continue

            if entry is None:
                # Reset marker
                self.is_speech_active = False
                self._silence_frames = 0
                self._utterance_frames = []
                self._pre_roll.clear()
                self._pending_audio = None
                continue

            if np is None:
                continue

            captured_at, item = entry
            for frame in self._audio_frames(item):
                self._take_frame(frame, captured_at)
        return None

    def _capture_lost(self, stream, now: float) -> Optional[str]:
        """Why the input stream is taken for dead, or None while it lives.

        Dead is no callback for ``CAPTURE_LOST_SECONDS``, or a stream that is
        no longer active or fails when asked. Looked at once every
        ``CAPTURE_CHECK_SECONDS``, and not while he dictates: the dictation
        pause resets the capture-health clock.
        """
        if self._dictation_active or now - self._last_capture_check < CAPTURE_CHECK_SECONDS:
            return None
        self._last_capture_check = now
        silent_for = now - self._last_audio_callback
        if silent_for >= CAPTURE_LOST_SECONDS:
            return f"no microphone callback for {silent_for:.0f} s"
        try:
            if not stream.active:
                return "the stream is no longer active"
        except Exception as exc:
            return f"the stream failed: {exc}"
        return None

    def _selected_input(self, devices) -> Optional[dict]:
        """The stream arguments that select the configured input in
        ``devices``, or None when it selects none there: a name or an index
        that matches no input, or no input at all for the system default."""
        try:
            device = resolve_input_device(self.cfg.voice_device, lambda: devices)
        except IndexNamesNoInput:
            raise
        except InputDeviceNotFound:
            return None
        if device is not None:
            return {'device': device}
        if not any(d.get('max_input_channels', 0) > 0 for d in devices):
            return None
        return {}

    def _wait_for_microphone(self, needs_quiet: bool = False) -> bool:
        """Read the device list afresh until it holds the selected
        microphone. False when the listener stops first.

        The readings come every ``MICROPHONE_RESCAN_SECONDS`` for the first
        ``MICROPHONE_RESCAN_FAST_FOR_SECONDS`` of the wait, then every
        ``MICROPHONE_RESCAN_SLOW_SECONDS``; a new loss, or a dictation press
        that misses the microphone (`microphone_missed`), makes them frequent
        again. The list is read afresh only while no stream is open in the
        process (`refresh_input_devices`); until then the old one is kept and
        the reading is tried again on the next one. With ``needs_quiet`` (the
        Windows permission probe, which opens without the lock, comes next)
        it returns only on a tick when no stream is open. Meanwhile her
        replies go on, and a question being collected or a hot window still
        ends on time.
        """
        self._start_replies()
        found = False
        while not self._should_stop:
            now = time.monotonic()
            if not found and now >= self._next_rescan_at:
                found = self._read_device_list_afresh(now)
            if found:
                if not needs_quiet or not open_stream_count():
                    return True
                if self._first_time("probe put off"):
                    debug_log("permission probe put off while a stream is open", "voice")
            self._check_query_timeout()
            time.sleep(0.1)
        return False

    def _read_device_list_afresh(self, now: float) -> bool:
        """One reading of the device list while the microphone is waited
        for, with the next one scheduled. Whether the list read holds it."""
        frequent = now - self._rescans_hurried_at < MICROPHONE_RESCAN_FAST_FOR_SECONDS
        self._next_rescan_at = now + (MICROPHONE_RESCAN_SECONDS if frequent
                                      else MICROPHONE_RESCAN_SLOW_SECONDS)
        try:
            devices = refresh_input_devices(sd)
        except DeviceListUnavailable as exc:
            if self._first_time("reading failed"):
                debug_log(f"device list not read afresh, PortAudio failed: {exc}", "voice")
            return False
        if devices is None:
            self._reading_held_back(now)
            return False
        if self._held_back_since is not None:
            debug_log("device list read afresh", "voice")
        self._held_back_since = None
        self._held_back_said = False
        return self._selected_input(devices) is not None

    def _reading_held_back(self, now: float) -> None:
        """A reading of the device list was held back by streams still open:
        logged when that starts, and said once on the console when it has
        lasted ``HELD_BACK_NOTICE_SECONDS`` (a stream left unclosed would
        keep the old list for as long)."""
        if self._held_back_since is None:
            self._held_back_since = now
            debug_log(f"device list not read afresh: {open_stream_count()} stream(s) open", "voice")
        elif not self._held_back_said and now - self._held_back_since >= HELD_BACK_NOTICE_SECONDS:
            self._held_back_said = True
            debug_log(f"device list still held back after {HELD_BACK_NOTICE_SECONDS:.0f} s: "
                      f"{open_stream_count()} stream(s) counted as open", "voice")
            print("  ⏳ Microphone check waits for other audio to finish...", flush=True)

    def microphone_missed(self) -> None:
        """A dictation press did not find the microphone. While it is waited
        for, the device list is read afresh every
        ``MICROPHONE_RESCAN_SECONDS`` again, the next reading at most that
        far away. Called from the dictation worker."""
        if self._microphone_wait is None:
            return
        now = time.monotonic()
        self._rescans_hurried_at = now
        self._next_rescan_at = min(self._next_rescan_at, now + MICROPHONE_RESCAN_SECONDS)
        debug_log("dictation missed the microphone: frequent readings again", "voice")

    def _open_capture(self, frame_ms: int, announce: bool):
        """Open the selected microphone, or None when the device list as it is
        now holds no such input. The input is resolved in the hold of
        portaudio_lock that opens it: another thread may read the device
        list afresh, and an index is only valid for the list it was read
        from."""
        with portaudio_lock:
            stream_kwargs = self._selected_input(sd.query_devices())
            if stream_kwargs is None:
                return None
            stream, self._stream_samplerate, channels = _open_input_stream(
                self._samplerate, frame_ms, stream_kwargs, callback=self._on_audio,
            )
        self._frame_samples = max(1, int(self._stream_samplerate * frame_ms / 1000))
        if announce and (self._stream_samplerate != self._samplerate or channels != 1):
            print(f"  🎤 Using {self._stream_samplerate} Hz, {channels} input channel(s); "
                  "converting to mono and resampling for speech recognition", flush=True)
        return stream

    def _announce_input_device(self) -> None:
        """Say which microphone she is about to listen to."""
        try:
            with portaudio_lock:
                stream_kwargs = self._selected_input(sd.query_devices())
                if stream_kwargs is None:
                    return
                if "device" in stream_kwargs:
                    device_name = sd.query_devices(stream_kwargs["device"]).get('name', 'Unknown')
                else:
                    try:
                        device_name = sd.query_devices(sd.default.device[0]).get('name', 'Unknown')
                    except Exception:
                        device_name = None
            if "device" in stream_kwargs:
                self._input_device_name = device_name
                debug_log(f"using input device: {device_name} (index {stream_kwargs['device']})", "voice")
                print(f"  🎤 Using audio device: {device_name}", flush=True)
            else:
                debug_log("using system default input device", "voice")
                if device_name is not None:
                    self._input_device_name = device_name
                    print(f"  🎤 Using default device: {device_name}", flush=True)
                else:
                    print("  🎤 Using system default input device", flush=True)
        except Exception:
            pass

    def _keep_trying_after(self, error: Exception) -> bool:
        """The selected microphone would not open or start. Whether to wait
        for it and try again, rather than stop listening.

        Before any wait, an error that reads as access denied stops listening,
        since it does not go away on its own; any other error starts the wait,
        as for a microphone lost while running. While waiting, every error is
        tried again on the next reading of the device list. The error is said
        once per wait.
        """
        if self._microphone_wait is None:
            if _reads_as_access_denied(error):
                self._report_capture_failure(error)
                return False
            self._say_microphone_lost(f"it would not open or start: {error}")
        if self._first_time("capture failure"):
            self._report_capture_failure(error)
        return True

    def _report_capture_failure(self, error: Exception) -> None:
        """Say why the microphone would not open or start."""
        error_msg = str(error).lower()
        debug_log(f"failed to open or start the input stream: {error}", "voice")

        # Provide helpful error messages for common issues
        if _reads_as_access_denied(error):
            print(f"  ❌ Microphone access denied. Please check: {_get_mic_permission_hint()}", flush=True)
        elif "device" in error_msg and ("use" in error_msg or "busy" in error_msg):
            print("  ❌ Microphone is being used by another application", flush=True)
        elif "device" in error_msg:
            print(f"  ❌ Failed to open microphone: {error}", flush=True)
            print("     Try selecting a different audio device in settings", flush=True)
        else:
            print(f"  ❌ Failed to start audio recording: {error}", flush=True)

    def _first_time(self, what: str) -> bool:
        """Whether ``what`` is yet to be said since start-up, or since the
        wait for the microphone began; it counts as said from now."""
        if what in self._said:
            return False
        self._said.add(what)
        return True

    def _begin_microphone_wait(self, kind: str) -> bool:
        """Start waiting for the microphone, "missing" or "lost", unless a
        wait is on already. The readings of the device list are frequent from
        now, and what the wait says is said afresh. Whether a wait began."""
        if self._microphone_wait is not None:
            return False
        self._microphone_wait = kind
        self._rescans_hurried_at = time.monotonic()
        self._held_back_since = None
        self._held_back_said = False
        self._said = set()
        return True

    def _say_microphone_missing(self) -> None:
        """Say once that the microphone is not connected and is waited for.
        Its name stays out of the line: a device name can carry a person's."""
        if not self._begin_microphone_wait("missing"):
            return
        self._next_rescan_at = 0.0
        print("  🎤 Microphone not connected: waiting for it...", flush=True)
        debug_log("the configured input is not in the device list: waiting for it", "voice")

    def _say_microphone_lost(self, why: str, lived_for: Optional[float] = None) -> None:
        """Say once that the microphone is lost and is waited for: its stream
        died after ``lived_for`` seconds, or it would not open or start.

        The device list is read afresh at once, unless the stream died within
        ``STREAM_SETTLED_SECONDS`` of opening: the scheduled reading is then
        kept, so a device that keeps dying is not reopened once a second."""
        if not self._begin_microphone_wait("lost"):
            return
        debug_log(f"microphone lost ({why}): waiting for it", "voice")
        if lived_for is None or lived_for >= STREAM_SETTLED_SECONDS:
            self._next_rescan_at = 0.0
        print("  ⚠️  Microphone lost: waiting for it to come back...", flush=True)

    def _say_microphone_back(self) -> None:
        """Audio arrives again from the microphone waited for."""
        word = "connected" if self._microphone_wait == "missing" else "back"
        self._microphone_wait = None
        print(f"  ✅ Microphone {word}: listening", flush=True)
        debug_log(f"microphone {word}: audio is arriving", "voice")

    def _forget_unfinished_audio(self) -> None:
        """Drop what a dead stream left behind: the blocks still queued and
        the utterance it cut off. Words heard after it reopens start afresh."""
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._clear_audio_buffers()

    def _configure_frames(self) -> None:
        """The frame length and the voice detector's limits, in frames."""
        self._frame_ms = frame_ms = _vad_frame_ms(self.cfg)
        self._frame_samples = max(1, int(self._samplerate * frame_ms / 1000))
        pre_roll_ms = int(getattr(self.cfg, "vad_pre_roll_ms", 240))
        endpoint_silence_ms = int(getattr(self.cfg, "endpoint_silence_ms", 800))
        max_utt_ms = int(getattr(self.cfg, "max_utterance_ms", 12000))
        tts_max_utt_ms = int(getattr(self.cfg, "tts_max_utterance_ms", 3000))

        self._pre_roll_max_frames = max(1, int(pre_roll_ms / frame_ms))
        self._endpoint_silence_frames = max(1, int(endpoint_silence_ms / frame_ms))
        # The maximum is picked per frame, by whether she is speaking.
        self._normal_max_utt_frames = max(1, int(max_utt_ms / frame_ms))
        self._tts_max_utt_frames = max(1, int(tts_max_utt_ms / frame_ms))

    def _take_frame(self, frame, captured_at: float) -> None:
        """One frame through the voice detector. ``captured_at`` is when the
        microphone delivered it: an utterance's start and end are read from
        it, however late the loop gets to the frame."""
        self._frame_captured_at = captured_at
        is_voice = self._is_speech_frame(frame)
        self._speech_frames_seen += int(is_voice)

        if not self.is_speech_active:
            if is_voice:
                self.is_speech_active = True

                # Backdate start time by pre-roll duration — the
                # actual speech onset was before VAD triggered.
                pre_roll_sec = len(self._pre_roll) * self._frame_ms / 1000.0
                utterance_start_time = captured_at - pre_roll_sec

                # Track utterance timing for echo detection
                self.echo_detector.track_utterance_timing(utterance_start_time, 0.0)

                # Seed with pre-roll
                if self._pre_roll:
                    self._utterance_frames.extend(list(self._pre_roll))
                self._utterance_frames.append(frame.copy())
                self._silence_frames = 0
            else:
                # Maintain pre-roll buffer
                self._pre_roll.append(frame.copy())
                while len(self._pre_roll) > self._pre_roll_max_frames:
                    try:
                        self._pre_roll.popleft()
                    except Exception:
                        break
        else:
            if is_voice:
                self._utterance_frames.append(frame.copy())
                self._silence_frames = 0
            else:
                self._silence_frames += 1
                # Use shorter timeout during TTS for quick stop command detection
                current_max_frames = (self._tts_max_utt_frames
                                      if (self.tts and self.tts.is_speaking())
                                      else self._normal_max_utt_frames)
                if (self._silence_frames >= self._endpoint_silence_frames
                        or len(self._utterance_frames) >= current_max_frames):
                    self._finalize_utterance()
                    self._pre_roll.clear()

        # Check for query timeouts, in the time this frame was captured
        self._check_query_timeout(captured_at)

    def _trailing_silence_seconds(self) -> float:
        """The silence at the end of the utterance being closed, in seconds:
        the silent frames the voice detector counted, each cut at the rate
        the microphone is read at, which is Whisper's own unless the device
        refused it."""
        frames = getattr(self, "_silence_frames", 0)
        frame_samples = getattr(self, "_frame_samples", 0)
        rate = getattr(self, "_stream_samplerate", None) or getattr(self, "_samplerate", 0)
        if not (frames and frame_samples and rate):
            return 0.0
        return frames * frame_samples / rate

    def _finalize_utterance(self) -> None:
        """Process completed utterance through speech recognition."""
        if np is None or not self._utterance_frames:
            self.is_speech_active = False
            self._silence_frames = 0
            self._utterance_frames = []
            return

        # When the frame that closed it was captured, which is when he
        # stopped, however long the loop took to get to it.
        utterance_end_time = self._frame_captured_at or time.time()
        # Read before the reset below: how long the detector waited in silence.
        trailing_silence = self._trailing_silence_seconds()
        utterance_start_time = self.echo_detector._utterance_start_time

        if self.cfg.voice_debug:
            utterance_duration = utterance_end_time - utterance_start_time if utterance_start_time > 0 else 0
            start_time_str = datetime.fromtimestamp(utterance_start_time).strftime('%H:%M:%S.%f')[:-3] if utterance_start_time > 0 else "N/A"
            end_time_str = datetime.fromtimestamp(utterance_end_time).strftime('%H:%M:%S.%f')[:-3]
            debug_log(f"utterance captured: duration={utterance_duration:.2f}s (started: {start_time_str}, ended: {end_time_str})", "voice")

        # Transcribe full audio - the intent judge will extract the relevant query
        try:
            audio = np.concatenate(self._utterance_frames, axis=0).flatten()
        except Exception:
            audio = None

        # Calculate energy before clearing frames for transcript processing
        utterance_energy = self._calculate_audio_energy(self._utterance_frames[-10:] if self._utterance_frames else [])

        # Reset state before processing
        self.is_speech_active = False
        self._silence_frames = 0
        self._utterance_frames = []

        if audio is None or audio.size == 0:
            return

        # Resample to Whisper's expected rate if the stream ran at a different rate
        stream_rate = getattr(self, "_stream_samplerate", self._samplerate)
        if stream_rate != self._samplerate:
            audio = _resample(audio, stream_rate, self._samplerate)

        # Filter short audio
        audio_duration = len(audio) / self._samplerate
        min_duration = getattr(self.cfg, "whisper_min_audio_duration", 0.3)
        if audio_duration < min_duration:
            debug_log(f"audio too short ({audio_duration:.2f}s < {min_duration}s), ignoring", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # The language Whisper reports for this utterance (the one named in
        # `whisper_language`, or the one it detected), kept below only if
        # the utterance is heard as something.
        detected: Optional[str] = None
        # The segments Whisper made of the utterance that the confidence
        # filter let through; none when it gave only a whole text.
        pieces: list = []

        # Speech recognition with appropriate backend
        try:
            if self._whisper_backend == "mlx":
                # MLX Whisper transcription
                with self.transcribe_lock:
                    result = mlx_whisper.transcribe(
                        audio,
                        path_or_hf_repo=self._mlx_model_repo,
                        language=self.cfg.whisper_language,
                    )
                detected = result.get("language")

                # Filter segments by confidence (MLX Whisper returns segments with avg_logprob)
                min_confidence = getattr(self.cfg, "whisper_min_confidence", 0.3)
                if self.awaiting_spoken_answer():
                    # Same reasoning as the faster-whisper path above.
                    min_confidence = 0.0
                marginal_threshold = min_confidence / 3  # Show user-visible log for marginal confidence
                no_speech_threshold = getattr(self.cfg, "whisper_no_speech_threshold", 0.5)
                segments = result.get("segments", [])

                if segments:
                    filtered_texts = []
                    for seg in segments:
                        avg_logprob = seg.get("avg_logprob", 0)
                        no_speech_prob = seg.get("no_speech_prob", 0)

                        # Convert avg_logprob to confidence (typically -1 to 0, so add 1)
                        confidence = min(1.0, max(0.0, avg_logprob + 1.0))
                        seg_text = seg.get("text", "").strip()

                        # Hard filter: high no_speech_prob means no real speech regardless of logprob.
                        if is_whisper_hallucination(no_speech_prob, no_speech_threshold):
                            debug_log(f"MLX segment filtered (no_speech_prob={no_speech_prob:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        if (confidence < min_confidence
                                and self._sounds_like_her_name(seg_text)):
                            debug_log(
                                f"MLX segment kept despite "
                                f"confidence={confidence:.2f}: it carries her "
                                f"name", "voice",
                            )
                            filtered_texts.append(seg.get("text", ""))
                            continue

                        if confidence < min_confidence:
                            if confidence >= marginal_threshold:
                                # Marginal confidence - show in log viewer (not debug)
                                print(f"🔇 Low confidence ({confidence:.2f}): \"{seg_text[:50]}...\"", flush=True)
                            else:
                                # Very low confidence - debug only
                                debug_log(f"MLX segment filtered (confidence={confidence:.2f}): '{seg_text[:50]}'", "voice")
                            continue

                        filtered_texts.append(seg.get("text", ""))

                    pieces = filtered_texts
                    text = " ".join(filtered_texts).strip()
                else:
                    # Fallback to full text if no segments
                    text = result.get("text", "").strip()
            else:
                # faster-whisper transcription
                # CPU mode: skip timestamps and disable context carry-over for speed
                cpu_mode = self._whisper_device == "cpu"
                with self.transcribe_lock:
                    segments_list, _info = transcribe_faster_whisper(
                        self.model, audio, language=self.cfg.whisper_language,
                        voice_detector=self._whisper_vad_setting(),
                        condition_on_previous_text=not cpu_mode,
                        without_timestamps=cpu_mode,
                    )
                # faster-whisper reports it on the info object; older API
                # variants may not carry the attribute.
                detected = getattr(_info, "language", None)
                filtered_segments = self._filter_noisy_segments(segments_list)
                pieces = [seg.text for seg in filtered_segments]
                text = " ".join(pieces).strip()
        except Exception as e:
            debug_log(f"transcription error: {e}", "voice")
            if sys.platform == 'win32':
                print(f"  ❌ Whisper error: {e}", flush=True)
            text = ""

        # Before the empty-transcript return, so the corpus keeps the
        # utterances that produced nothing: those are where a transducer
        # and an autoregressive decoder differ most.
        self._capture.save(
            audio, self._samplerate, text,
            model=str(getattr(self.cfg, "whisper_model", "")),
            device=self._input_device_name,
        )

        if not text or not text.strip():
            detector = ("none on mlx" if self._whisper_backend == "mlx"
                        else "on" if self._voice_detector_on() else "off")
            debug_log(f"utterance of {audio_duration:.1f}s heard as nothing "
                      f"(whisper_vad={self._whisper_vad_setting()}, "
                      f"voice detector {detector})", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # The language last heard, which the turn handed to the reply thread
        # carries so tools can pick locale-appropriate resources. Only an
        # utterance heard as something sets it: with no language set,
        # Whisper still names one for an utterance the voice detector
        # emptied, guessed from the silence it pads it with.
        if isinstance(detected, str) and detected:
            self._last_detected_language = detected

        annonce = self._transcription_announcement(text)
        if annonce:
            print(annonce, flush=True)

        # Filter out repetitive hallucinations (e.g., "don't don't don't...")
        if self._is_repetitive_hallucination(text):
            debug_log(f"rejected repetitive hallucination: '{text[:80]}...'", "voice")
            self.state_manager.check_hot_window_expiry(self.cfg.voice_debug)
            return

        # Add to transcript buffer for context-aware processing.
        # The "during TTS" mark tells the judge the words may be her own
        # voice coming back, which it never extracts. Timing says when that
        # can be: while she speaks, or for an utterance that started (the
        # voice detector's pre-roll included) before her last word plus the
        # echo tolerance. His answer to her question often starts inside
        # that tolerance, so timing does not decide alone: words she did not
        # say, loud enough to have been spoken, are never marked, unless
        # their voice stopped with hers (`_could_be_hers`). Both are read
        # once, so the line logged gives the reading that decided.
        in_her_time = self._in_her_time(utterance_start_time)
        if in_her_time:
            # Her sentences that open and close it go first, so everything
            # below reads his words alone.
            text = self._his_words_among_her_sentences(text, pieces)
        stopped_with_hers = self._voice_stopped_with_hers(utterance_end_time, trailing_silence)
        is_during_tts = self._could_be_hers(text, utterance_energy, in_her_time,
                                            stopped_with_hers)
        if in_her_time:
            if is_during_tts and stopped_with_hers:
                debug_log(f"🔇 marked as her echo (its voice stopped with hers, whatever "
                          f"its words): \"{text[:60]}\"", "voice")
            elif not is_during_tts:
                debug_log(f"🗣️ not marked as her echo "
                          f"({self._not_hers_because(stopped_with_hers)}): "
                          f"\"{text[:60]}\"", "voice")
        self._transcript_buffer.add(
            text=text,
            start_time=utterance_start_time,
            end_time=utterance_end_time,
            energy=utterance_energy,
            is_during_tts=is_during_tts,
        )

        # Process the transcript with pre-calculated energy and utterance timing
        self._process_transcript(
            text, utterance_energy, utterance_start_time, utterance_end_time,
            trailing_silence=trailing_silence,
        )
