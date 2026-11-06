"""Debug logging utilities for Jarvis."""
import os
import sys
import time
from typing import Optional
from .config import load_settings


_last_check_time: float = 0.0
_cached_voice_debug: Optional[bool] = None
_CACHE_TTL_SECONDS: float = 2.0


# Optional file sink. The desktop app spawns the daemon with its output
# piped into the in-app log viewer, which means nothing reaches a
# terminal or a file: diagnosing the voice path from outside the app is
# impossible without this. Set ``JARVIS_DEBUG_FILE`` to a path to also
# append there. Unset by default, so it costs nothing.
#
# The daemon's own entry (``jarvis.main``) mirrors its whole console into
# that file, and a stderr that says so (``mirrors_log_file``) has already
# put the line there. Appending is for the processes whose console is not
# mirrored: the desktop app's, and the daemon run in its thread.
_SINK_PATH = os.environ.get("JARVIS_DEBUG_FILE", "").strip()


def _append_to_sink(line: str) -> None:
    if not _SINK_PATH or getattr(sys.stderr, "mirrors_log_file", False) is True:
        return
    try:
        with open(_SINK_PATH, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:
        pass


def _is_debug_enabled() -> bool:
    global _last_check_time, _cached_voice_debug
    now = time.time()
    if _cached_voice_debug is None or (now - _last_check_time) > _CACHE_TTL_SECONDS:
        # Claim the slot before reloading: the settings parser itself logs
        # (a kept auxiliary pin announces itself), and a re-entrant call
        # here must see a settled answer rather than reload again.
        _last_check_time = now
        if _cached_voice_debug is None:
            _cached_voice_debug = False
        try:
            _cached_voice_debug = bool(load_settings().voice_debug)
        except Exception:
            _cached_voice_debug = False
    return bool(_cached_voice_debug)


def debug_log(message: str, category: str = "debug") -> None:
    """Unified debug logging function for Jarvis.

    Args:
        message: The debug message to log
        category: The log category (e.g., "debug", "voice", "echo", "tts", etc.)
    """
    if not _is_debug_enabled():
        return
    line = f"[{category:^10}] {message}"
    try:
        # One write, so the line stays whole beside other threads' lines.
        sys.stderr.write(line + "\n")
    except Exception:
        pass
    _append_to_sink(line)
