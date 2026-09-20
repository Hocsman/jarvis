"""UTF-8 on the standard streams, for the Windows console.

Emoji are part of this project's command line output, and the Windows
console defaults to a code page that cannot carry them. Both entry points,
the daemon and the desktop app, need the same treatment, and the desktop app
imports the daemon in-process, so the setup runs more than once in one
process and has to be safe to repeat.
"""

from __future__ import annotations

import sys
from typing import Any


def force_utf8_stream(stream: Any) -> None:
    """Switch one stream to UTF-8 in place.

    The stream object and the buffer beneath it must survive: replacing a
    stream drops the last reference to whatever was there before, and its
    finaliser closes the buffer the replacement is still writing through.
    Reconfiguring mutates in place and is safe to repeat.

    A stream that offers no ``reconfigure`` is left exactly as it is.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError):
        pass


def force_utf8_console() -> None:
    """Apply it to stdout and stderr, where the console needs it.

    Frozen builds route their output to the crash log, which the desktop app
    already opens as UTF-8, so they are left alone.
    """
    if sys.platform != "win32" or getattr(sys, "frozen", False):
        return
    force_utf8_stream(sys.stdout)
    force_utf8_stream(sys.stderr)
