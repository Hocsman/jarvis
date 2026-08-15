"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

import os
import sys

from .daemon import main


class _Tee:
    """Write to the original stream and to a file at the same time.

    The desktop app spawns this module with both streams piped into its
    in-app log viewer, so nothing the daemon prints reaches a terminal or
    a file. That makes the voice path undiagnosable from outside the app:
    the one line that separates "she never heard you" from "she heard you
    and did nothing" is a plain ``print``. Setting ``JARVIS_DEBUG_FILE``
    to a path mirrors everything there. Unset by default.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        try:
            self._handle.write(data)
            self._handle.flush()
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._handle.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_sink() -> None:
    path = os.environ.get("JARVIS_DEBUG_FILE", "").strip()
    if not path:
        return
    try:
        handle = open(path, "a", encoding="utf-8", buffering=1)
    except Exception:
        return
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)


if __name__ == "__main__":
    _install_sink()
    main()
