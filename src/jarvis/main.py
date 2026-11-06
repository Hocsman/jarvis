"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

import os
import sys

from .daemon import main


class _Tee:
    """The console, made safe to lose, mirrored into the log file when one is named.

    The desktop app spawns this module with both streams piped into its
    in-app log viewer, so nothing the daemon prints reaches a terminal or
    a file. That makes the voice path undiagnosable from outside the app:
    the one line that separates "she never heard you" from "she heard you
    and did nothing" is a plain ``print``. Setting ``JARVIS_DEBUG_FILE``
    to a path mirrors everything there. Unset by default.

    The pipe goes when the desktop app does, which can be before the daemon
    has finished shutting down. A console that fails is given up for the rest
    of the run, said once in the file, and never raises to the caller: a
    ``print`` must not be what stops the diary being saved. The file is
    written first, so it gets the line whatever the console does.
    """

    def __init__(self, stream, handle=None):
        self._stream = stream
        self._handle = handle
        self._console_lost = False
        self._mid_line = False
        self._owed_note: str = ""
        # Read by ``debug_log``, which leaves the file to this mirror rather
        # than write each of its lines there a second time.
        self.mirrors_log_file = handle is not None

    def write(self, data):
        self._to_file(data)
        self._to_console(self._stream.write, data)
        return len(data)

    def flush(self):
        self._to_console(self._stream.flush)
        if self._handle is not None:
            try:
                self._handle.flush()
            except Exception:
                pass

    def _to_file(self, data) -> None:
        if self._handle is None:
            return
        try:
            self._handle.write(data)
            self._handle.flush()
        except Exception:
            return
        if data:
            self._mid_line = not data.endswith("\n")
        self._write_owed_note()

    def _to_console(self, call, *args) -> None:
        if self._console_lost:
            return
        try:
            call(*args)
        except Exception as e:
            self._console_lost = True
            name = getattr(self._stream, "name", "") or "console"
            self._owed_note = f"🔌 {name} gone ({e}), the log file carries on alone\n"
            self._write_owed_note()

    def _write_owed_note(self) -> None:
        """When the console went, said in the file once the line in progress ends."""
        if not self._owed_note or self._mid_line or self._handle is None:
            return
        note, self._owed_note = self._owed_note, ""
        try:
            self._handle.write(note)
            self._handle.flush()
        except Exception:
            pass

    def reconfigure(self, *args, **kwargs):
        if hasattr(self._stream, "reconfigure"):
            return self._stream.reconfigure(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _install_sink() -> None:
    """Wrap both streams: safe to lose, and mirrored into ``JARVIS_DEBUG_FILE``
    when it names a file."""
    from .utils.console import force_utf8_console
    force_utf8_console()

    handle = None
    path = os.environ.get("JARVIS_DEBUG_FILE", "").strip()
    if path:
        try:
            handle = open(path, "a", encoding="utf-8", buffering=1)
        except Exception:
            handle = None
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)


if __name__ == "__main__":
    _install_sink()
    main()
