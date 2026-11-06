"""faster-whisper with its voice detector, and without it once it has failed.

With `whisper_vad` on, faster-whisper's own voice detector (Silero) cuts the
stretches of an utterance that hold no speech before the decoder hears them.
The detector is an ONNX model of its own, read from faster-whisper's
``assets`` folder through onnxruntime. When either is missing (a build that
left the files out, an install without onnxruntime) every call that asks for
the detector fails, and a caller that kept asking would lose every utterance.

So a call that fails with the detector is made again without it. The
detector is blamed only when it then fails to load on its own: the failure
is announced once, and no caller in the process asks for it again. A call
that failed for another reason (a passing GPU error while Whisper detects
the language) leaves the detector on.

The detector's model is loaded once per process and shared, so its failure
is the process's: the listener and dictation, which share the Whisper model,
both call it through here and read the same state.
"""

from __future__ import annotations

import threading
from typing import Any, List, Optional, Tuple

from ..debug import debug_log

# The detector's failure in this process, once it has been seen.
_failure: Optional[Exception] = None
_failure_lock = threading.Lock()


def detector_failed() -> bool:
    """Whether the voice detector has failed in this process."""
    return _failure is not None


def detector_failure_line(error: Exception) -> str:
    """The console line announcing that the voice detector failed."""
    return (f"  ⚠️  Whisper's voice detector is unavailable ({type(error).__name__}): "
            f"transcribing without it, so phrases may be invented out of noise")


def _give_up(error: Exception) -> None:
    """Record the detector's failure and announce it, once per process."""
    global _failure
    with _failure_lock:
        if _failure is not None:
            return
        _failure = error
    debug_log(f"Whisper's voice detector failed, transcribing without it: {error}", "voice")
    print(detector_failure_line(error), flush=True)


def load_voice_detector() -> None:
    """Load the voice detector's model, so the first utterance does not pay
    for it (about 0.1 s). Raises what loading raises: onnxruntime missing,
    or the model files.

    faster-whisper keeps the loaded model for the life of the process, and
    every transcription that asks for the detector uses that one.
    """
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    get_speech_timestamps(np.zeros(512, dtype=np.float32), VadOptions())


def prepare_voice_detector() -> None:
    """Load the detector now, or announce now that it cannot load."""
    if detector_failed():
        return
    try:
        load_voice_detector()
        debug_log("Whisper's voice detector loaded", "voice")
    except Exception as error:
        _give_up(error)


def transcribe(model: Any, audio: Any, *, language: Optional[str], voice_detector: bool,
               **options: Any) -> Tuple[List[Any], Any]:
    """The segments faster-whisper hears in *audio* (a list) and its info.

    With *voice_detector* (the setting asks for it) and a detector that has
    not failed, the detector cuts what holds no speech before the decoder
    hears it. When that call fails and the same call without the detector
    works, the detector is loaded on its own: if that fails, it was the
    detector, and it is given up for the process (`_give_up`); if it loads,
    the failure was something else and it stays on. When the call without
    it fails too, that error is raised and the detector is not blamed. A
    model that takes none of these arguments (`TypeError`) is called with
    the language alone.
    """
    asked = voice_detector and not detector_failed()
    try:
        segments, info = model.transcribe(audio, language=language, vad_filter=asked, **options)
    except TypeError:
        segments, info = model.transcribe(audio, language=language)
        return list(segments), info
    except Exception as error:
        if not asked:
            raise
        debug_log(f"transcription with the voice detector failed ({error}); trying without it",
                  "voice")
        segments, info = model.transcribe(audio, language=language, vad_filter=False, **options)
        try:
            load_voice_detector()
            debug_log("the voice detector loads: the failure was not its own", "voice")
        except Exception as load_error:
            _give_up(load_error)
        return list(segments), info
    return list(segments), info
