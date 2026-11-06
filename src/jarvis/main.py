"""
Jarvis Voice Assistant - Main Entry Point

A modular voice assistant with conversation memory, tool integration,
and natural language processing capabilities.
"""

import os
import sys
import threading

from .daemon import main


# One lock for both mirrored streams: they share the file, and a line is
# written to it whole, never spliced with another thread's.
_FILE_LOCK = threading.Lock()


def _encodable(text: str, encoding: str) -> str:
    """``text`` with what ``encoding`` cannot carry replaced, so that one
    unwritable character costs that character, not the line."""
    return text.encode(encoding or "utf-8", "replace").decode(encoding or "utf-8", "replace")


class _Tee:
    """The console, made safe to lose, mirrored into the log file when one is named.

    The desktop app spawns this module with both streams piped into its
    in-app log viewer, so nothing the daemon prints reaches a terminal or
    a file. That makes the voice path undiagnosable from outside the app:
    the one line that separates "she never heard you" from "she heard you
    and did nothing" is a plain ``print``. Setting ``JARVIS_DEBUG_FILE``
    to a path mirrors everything there. Unset by default.

    The pipe goes when the desktop app does, which can be before the daemon
    has finished shutting down. A console that is gone (an error from the
    pipe or the closed file beneath it, or no console at all) is given up for
    the rest of the run, said once in the file, and never raises to the
    caller: a ``print`` must not be what stops the diary being saved. A line
    the console cannot encode is written with what it cannot carry
    replaced, and any other refusal costs that one write. The file is written
    first, so it gets the line whatever the console does.

    Each thread's text reaches the file a whole line at a time, so lines from
    the listener, the speech engine and the reply workers never splice.
    """

    def __init__(self, stream, handle=None):
        self._stream = stream
        self._handle = handle
        self._console_lost = stream is None
        self._partial = threading.local()
        self._owed_note: str = ""
        # Read by ``debug_log``, which leaves the file to this mirror rather
        # than write each of its lines there a second time.
        self.mirrors_log_file = handle is not None

    def write(self, data):
        self._to_file(data)
        self._to_console("write", data)
        return len(data)

    def flush(self):
        self._to_console("flush")
        if self._handle is not None:
            with _FILE_LOCK:
                self._write_file(getattr(self._partial, "text", ""))
                self._partial.text = ""
                try:
                    self._handle.flush()
                except Exception:
                    pass

    def _to_file(self, data) -> None:
        if self._handle is None or not data:
            return
        text = getattr(self._partial, "text", "") + data
        whole, newline, rest = text.rpartition("\n")
        self._partial.text = rest
        if newline:
            with _FILE_LOCK:
                self._write_file(whole + newline)

    def _write_file(self, text: str) -> None:
        """Write ``text`` to the file, then the note owed when the console
        went, once that text ends a line. Called holding ``_FILE_LOCK``."""
        if text:
            try:
                try:
                    self._handle.write(text)
                except UnicodeEncodeError:
                    self._handle.write(_encodable(text, getattr(self._handle, "encoding", "utf-8")))
                self._handle.flush()
            except Exception:
                return
            if not text.endswith("\n"):
                return
        if self._owed_note:
            note, self._owed_note = self._owed_note, ""
            try:
                self._handle.write(note)
                self._handle.flush()
            except Exception:
                pass

    def _to_console(self, method: str, *args) -> None:
        if self._console_lost:
            return
        try:
            getattr(self._stream, method)(*args)
        except UnicodeEncodeError:
            self._retry_encodable(method, *args)
        except (OSError, ValueError) as e:
            # The pipe is broken or closed, or the file beneath it is.
            self._give_up_console(e)
        except Exception:
            pass

    def _retry_encodable(self, method: str, *args) -> None:
        if method != "write":
            return
        try:
            self._stream.write(_encodable(args[0], getattr(self._stream, "encoding", "utf-8")))
        except (OSError, ValueError) as e:
            self._give_up_console(e)
        except Exception:
            pass

    def _give_up_console(self, error: Exception) -> None:
        self._console_lost = True
        if self._handle is None:
            return
        name = getattr(self._stream, "name", "") or "console"
        self._owed_note = f"🔌 {name} gone ({error}), the log file carries on alone\n"
        if not getattr(self._partial, "text", ""):
            with _FILE_LOCK:
                self._write_file("")

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
            handle = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
        except Exception:
            handle = None
    # No console and no file (``pythonw``): printing already does nothing.
    if sys.stdout is not None or handle is not None:
        sys.stdout = _Tee(sys.stdout, handle)
    if sys.stderr is not None or handle is not None:
        sys.stderr = _Tee(sys.stderr, handle)


if __name__ == "__main__":
    _install_sink()
    main()
