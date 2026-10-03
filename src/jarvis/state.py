"""The assistant's overall state: what the voice pipeline is doing right now.

The listener, the reply engine, the speech engines and the dictation engine
publish it as they work (``IDLE``, ``LISTENING``, ``THINKING``, ``SPEAKING``,
...). Whatever shows the assistant to the user reads it: the desktop app's
orb, dashboard and face. The state belongs to the core, so it lives here, and
a front end imports it from ``jarvis``; ``jarvis`` imports no front end.

Publishing needs nothing but the standard library, so a headless run (no
desktop app, no Qt in the process) publishes exactly as a desktop run does.

The state crosses processes through a small file in the system temp
directory, because the daemon runs as a subprocess in a development checkout
and as a thread inside the bundled app; a file reads the same in both. The
file carries one enum value and nothing else.
"""

from __future__ import annotations

import os
import tempfile
import threading
from enum import Enum
from typing import Optional

from .debug import debug_log


class JarvisState(Enum):
    """What the assistant is doing, as shown to the user."""
    ASLEEP = "asleep"          # Daemon not started yet
    IDLE = "idle"              # Awake and ready, waiting for the wake word
    LISTENING = "listening"    # Actively listening (collecting or hot window)
    THINKING = "thinking"      # Processing a query
    SPEAKING = "speaking"      # Speaking a response
    DICTATING = "dictating"    # Hold-to-dictate recording active
    DICTATION_PROCESSING = "dictation_processing"  # Transcribing and pasting captured dictation


def _state_file_path() -> str:
    """Where the state is shared between processes."""
    return os.path.join(tempfile.gettempdir(), "jarvis_state")


class JarvisStateManager:
    """The process-wide holder of the assistant's state.

    The file is the source of truth for readers, so a reader in one process
    sees what a writer in another published; the in-memory value is the
    fallback for a file that cannot be read, and the only copy on a machine
    with no usable temp directory.

    A new manager always starts ``ASLEEP`` and writes that to the file: the
    file carries state across processes during a session, not across launches.
    """

    def __init__(self, state_file: Optional[str] = None):
        self._state = JarvisState.ASLEEP
        self._state_lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._state_file: Optional[str] = state_file
        if self._state_file is None:
            try:
                self._state_file = _state_file_path()
            except OSError as exc:
                debug_log(f"no usable temp directory, state is not shared across processes: {exc}", "state")
        self._write_state(JarvisState.ASLEEP)

    @property
    def state(self) -> JarvisState:
        """The current state, read from the file so another process's writes show."""
        if self._state_file is not None:
            try:
                with open(self._state_file, "r", encoding="utf-8") as handle:
                    return JarvisState(handle.read().strip())
            except (ValueError, OSError):
                # Missing, unreadable or unrecognised content: the last state
                # set in this process is the best answer left.
                pass

        with self._state_lock:
            return self._state

    def set_state(self, state: JarvisState) -> None:
        """Publish a new state (thread-safe, visible to other processes).

        Publishes are ordered: the memory value and the file are updated as
        one step, so the publish that started last is the one both end up
        holding. Readers are not held up by a publish in progress.

        Never raises on a file that cannot be written: publishing is a side
        channel of the voice pipeline and must not interrupt it.
        """
        with self._publish_lock:
            with self._state_lock:
                self._state = state
            self._write_state(state)

    def _write_state(self, state: JarvisState) -> None:
        if self._state_file is None:
            return
        try:
            with open(self._state_file, "w", encoding="utf-8") as handle:
                handle.write(state.value)
        except OSError as exc:
            debug_log(f"state file not writable, state is not shared across processes: {exc}", "state")


_jarvis_state_instance: Optional[JarvisStateManager] = None
_jarvis_state_lock = threading.Lock()


def get_jarvis_state() -> JarvisStateManager:
    """The process-wide state holder."""
    global _jarvis_state_instance
    with _jarvis_state_lock:
        if _jarvis_state_instance is None:
            _jarvis_state_instance = JarvisStateManager()
        return _jarvis_state_instance
