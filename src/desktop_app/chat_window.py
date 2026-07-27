"""
💬 Chat Window

A text chat interface for Jarvis, alongside the existing voice path. Voice
and text share one conversation (the daemon's global dialogue memory). See
``chat_window.spec.md`` for the full contract.

The window is created lazily by the system tray and kept alive for the
session. Daemon callback signals are marshalled onto the Qt main thread via
``ChatSignals`` so UI updates never touch the worker thread directly.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QCloseEvent, QShowEvent, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jarvis.debug import debug_log
from desktop_app.themes import COLORS, JARVIS_THEME_STYLESHEET


def get_hot_window_messages() -> list:
    """Thin wrapper around ``jarvis.daemon.get_hot_window_messages``.

    Lives at module scope so tests can monkeypatch
    ``desktop_app.chat_window.get_hot_window_messages`` without touching the
    daemon module (which the bundled and subprocess paths resolve
    differently).
    """
    from jarvis import daemon
    return daemon.get_hot_window_messages()


# ---------------------------------------------------------------------------
# Thread-safe signal bridge
# ---------------------------------------------------------------------------


class ChatSignals(QObject):
    """Marshals daemon-worker-thread callbacks onto the Qt main thread.

    The daemon fires ``on_start`` / ``on_complete`` / ``on_busy`` from its
    worker thread. The window connects these signals to slots so the actual
    UI mutation happens on the main thread.
    """

    started = pyqtSignal(str)
    completed = pyqtSignal(object)  # Optional[str]
    busy = pyqtSignal()


class ChatIpcSignals(QObject):
    """Marshals a raw ``__CHAT__:`` log line from the log-reader worker thread
    onto the Qt main thread.

    The desktop app's log reader runs on a plain ``threading.Thread`` and must
    not create widgets or parse IPC into widget mutations directly. It emits
    ``line_received`` (a queued cross-thread connection) and the main-thread
    slot calls ``ChatWindow.process_ipc_line``.
    """

    line_received = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------


_TRANSCRIPT_STYLE = f"""
    QPlainTextEdit {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 10px;
        font-family: '.AppleSystemUIFont', 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
"""

_INPUT_STYLE = f"""
    QPlainTextEdit {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 8px;
        font-family: '.AppleSystemUIFont', 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
    QPlainTextEdit:focus {{
        border-color: {COLORS['accent_primary']};
    }}
"""

_SEND_BTN_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['accent_primary']};
        color: #0a0b0f;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['accent_secondary']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['accent_muted']};
        color: {COLORS['text_muted']};
    }}
"""

_STOP_BTN_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['error']};
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['error_light']};
    }}
"""

_STATUS_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 12px;
        padding: 2px 4px;
    }}
"""

_DAEMON_STATUS_MESSAGES = {
    "starting": "Starting Jarvis...",
    "stopping": "Stopping Jarvis...",
    "stopped": "Start Listening from the tray to use chat.",
    "crashed": "Jarvis stopped unexpectedly. Start Listening to reconnect.",
}

_DAEMON_STATUS_PLACEHOLDERS = {
    "starting": "Jarvis is starting",
    "stopping": "Jarvis is stopping",
    "stopped": "Start Listening from the tray to use chat",
    "crashed": "Start Listening from the tray to reconnect chat",
    "running": "Type a message to Jarvis... (Enter to send, Shift+Enter for newline)",
}


class ChatWindow(QMainWindow):
    """Text chat window. Sends via ``jarvis.daemon.submit_text_query``.

    In subprocess mode the desktop app sets ``submit_fn`` to a callable that
    writes a ``__CHAT_QUERY__:`` line to the daemon's stdin, and feeds
    ``__CHAT__:`` events back into the window's signals. In bundled mode the
    default path calls the daemon directly with the window's signal emitters
    as callbacks.
    """

    def __init__(
        self, submit_fn=None, daemon_available: bool = True, cancel_fn=None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis Chat")
        self.setMinimumSize(520, 560)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_fn = submit_fn
        # Subprocess mode routes cancellation to the daemon the same way it
        # routes a submission. Without it, Stop sets a flag in this
        # process while the query runs in the other one.
        self._cancel_fn = cancel_fn
        # Set by Stop, cleared by the next send. The engine keeps running
        # after a cancel and its reply still arrives, so the window has to
        # decline the answer to an exchange the user walked away from.
        self._query_cancelled = False
        self._daemon_available = daemon_available
        self._daemon_status = "running" if daemon_available else "stopped"

        # Signal bridge: daemon worker -> Qt main thread.
        self.signals = ChatSignals()
        self.signals.started.connect(self._on_start)
        self.signals.completed.connect(self._on_complete)
        self.signals.busy.connect(self._on_busy)

        # --- Layout -----------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Transcript (read-only)
        self.transcript_widget = QPlainTextEdit()
        self.transcript_widget.setReadOnly(True)
        self.transcript_widget.setStyleSheet(_TRANSCRIPT_STYLE)
        layout.addWidget(self.transcript_widget, stretch=1)

        # Status indicator (display-only label)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_STATUS_STYLE)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Input row: input box + send + stop
        row = QHBoxLayout()
        row.setSpacing(8)

        self.input_widget = QPlainTextEdit()
        self.input_widget.setPlaceholderText(_DAEMON_STATUS_PLACEHOLDERS["running"])
        self.input_widget.setFixedHeight(64)
        self.input_widget.setStyleSheet(_INPUT_STYLE)
        self.input_widget.keyPressEvent = self._input_key_press  # type: ignore[method-assign]
        row.addWidget(self.input_widget, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(_SEND_BTN_STYLE)
        self.send_button.clicked.connect(self._send)
        row.addWidget(self.send_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(_STOP_BTN_STYLE)
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setVisible(False)
        row.addWidget(self.stop_button)

        layout.addLayout(row)

        self._query_in_flight = False
        # Whether the transcript has been seeded from the daemon's hot window.
        # Seeded once on first show so re-opening never duplicates turns.
        self._hot_window_seeded = False
        self.set_daemon_available(daemon_available)

    # --- Sending --------------------------------------------------------

    def _send(self) -> None:
        text = self.input_widget.toPlainText().strip()
        if not text:
            return
        if not self._daemon_available:
            self._append_system("Start Listening to use chat.")
            return

        # Echo the user message into the transcript immediately.
        self._append_user(text)
        self.input_widget.setPlainText("")

        self._query_cancelled = False
        self._set_thinking(True)

        if self._submit_fn is not None:
            # Subprocess mode: the desktop app routes the query to the daemon's
            # stdin and feeds __CHAT__: events back via the signals.
            self._submit_fn(text)
        else:
            # Bundled mode: call the daemon directly with our signal emitters.
            from jarvis import daemon

            daemon.submit_text_query(
                text,
                on_start=self.signals.started.emit,
                on_complete=self.signals.completed.emit,
                on_busy=self.signals.busy.emit,
            )

    def _stop(self) -> None:
        """Abandon the in-flight query. Never request_stop, which would
        tear down the whole voice assistant."""
        # Refuse the reply locally first: cancellation cannot unwind a
        # request already inside the engine, so the answer arrives either
        # way and this is what keeps it out of the transcript.
        self._query_cancelled = True

        if self._cancel_fn is not None:
            # Subprocess mode: the query lives in the daemon process.
            self._cancel_fn()
        else:
            from jarvis import daemon

            daemon.cancel_active_chat_query()

        # Reset the thinking indicator immediately so the user sees
        # feedback without waiting for the engine to finish.
        self._set_thinking(False)

    def set_daemon_available(self, available: bool) -> None:
        """Enable or disable chat submission based on daemon availability."""
        self.set_daemon_status("running" if available else "stopped")

    def set_daemon_status(self, status: str) -> None:
        """Reflect daemon lifecycle state in chat controls and status text."""
        if status != "running" and status not in _DAEMON_STATUS_MESSAGES:
            debug_log(f"unknown chat daemon status ignored: {status}", "chat")
            status = "stopped"

        self._daemon_status = status
        self._daemon_available = status == "running"
        if not self._daemon_available:
            self._query_in_flight = False
            self.stop_button.setVisible(False)
        self.input_widget.setEnabled(self._daemon_available)
        self.input_widget.setPlaceholderText(
            _DAEMON_STATUS_PLACEHOLDERS.get(
                status,
                _DAEMON_STATUS_PLACEHOLDERS["stopped"],
            )
        )
        self._refresh_status_label()
        self._refresh_send_button()

    # --- Daemon callback slots (run on the main thread via signals) -----

    def _on_start(self, _query: str) -> None:
        # The user message is already echoed in _send. We keep the thinking
        # indicator on; nothing extra to render for the start event in the MVP.
        self._set_thinking(True)

    def _on_complete(self, reply: Optional[str]) -> None:
        self._set_thinking(False)
        if self._query_cancelled:
            debug_log("chat reply dropped: the query was cancelled", "chat")
            self._query_cancelled = False
            return
        if reply:
            self._append_assistant(reply)

    def _on_busy(self) -> None:
        self._set_thinking(False)
        self._append_system("Jarvis is busy with another query already.")

    # --- Subprocess IPC entry point --------------------------------------

    def process_ipc_line(self, line: str) -> bool:
        """Parse a ``__CHAT__:`` event line and emit the matching signal.

        Mirrors ``DiaryUpdateDialog.process_log_line``: the caller (the log
        reader thread) forwards the raw line, and this method owns the JSON
        parse + signal emit. Returns True if the line was a chat event (even
        if malformed), False otherwise. Must be called on the Qt main thread
        (the caller marshals via a main-thread-owned signal).
        """
        from jarvis.daemon import CHAT_IPC_PREFIX
        if not line.startswith(CHAT_IPC_PREFIX):
            return False
        import json as _json
        try:
            payload = _json.loads(line[len(CHAT_IPC_PREFIX):])
        except Exception:
            debug_log(f"malformed {CHAT_IPC_PREFIX} line ignored", "chat")
            return True
        kind = payload.get("type")
        data = payload.get("data")
        if kind == "start":
            self.signals.started.emit(str(data) if data is not None else "")
        elif kind == "complete":
            self.signals.completed.emit(data)
        elif kind == "busy":
            self.signals.busy.emit()
        return True

    # --- Rendering helpers ----------------------------------------------

    def _append_user(self, text: str) -> None:
        self._append_line(f"👤 You: {text}")

    def _append_assistant(self, text: str) -> None:
        self._append_line(f"🤖 Jarvis: {text}")

    def _append_system(self, text: str) -> None:
        self._append_line(f"  ⏳ {text}")

    def _append_line(self, line: str) -> None:
        cursor = self.transcript_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self.transcript_widget.toPlainText():
            cursor.insertText("\n")
        cursor.insertText(line)
        self.transcript_widget.setTextCursor(cursor)

    def _set_thinking(self, thinking: bool) -> None:
        self._query_in_flight = thinking and self._daemon_available
        self.stop_button.setVisible(self._query_in_flight)
        self._refresh_status_label()
        self._refresh_send_button()

    def _refresh_send_button(self) -> None:
        self.send_button.setEnabled(self._daemon_available and not self._query_in_flight)

    def _refresh_status_label(self) -> None:
        if self._query_in_flight:
            self._status_label.setText("  Jarvis is thinking…")
            self._status_label.setVisible(True)
            return

        if self._daemon_status == "running":
            self._status_label.setText("")
            self._status_label.setVisible(False)
            return

        message = _DAEMON_STATUS_MESSAGES.get(
            self._daemon_status,
            _DAEMON_STATUS_MESSAGES["stopped"],
        )
        self._status_label.setText(f"  {message}")
        self._status_label.setVisible(True)

    # --- Input key handling ---------------------------------------------

    def _input_key_press(self, event) -> None:
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt as _Qt

        if (
            isinstance(event, QKeyEvent)
            and event.key() in (_Qt.Key.Key_Return, _Qt.Key.Key_Enter)
            and not (event.modifiers() & _Qt.KeyboardModifier.ShiftModifier)
        ):
            # Enter (or numpad Enter) sends; Shift+Enter inserts a newline.
            self._send()
            return
        # Default handling for all other keys.
        QPlainTextEdit.keyPressEvent(self.input_widget, event)

    # --- Lifecycle ------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        # Seed the transcript from the daemon's hot window once, on first show,
        # so a user who has been talking by voice sees their recent turns
        # instead of a blank panel. Runs only once per instance: re-showing
        # (from the tray or after a hide) must never duplicate turns. Fails
        # silently when the daemon accessor is unavailable (e.g. subprocess
        # mode before the bridge is wired) — the window just opens blank.
        if not self._hot_window_seeded:
            self._hot_window_seeded = True
            try:
                for msg in get_hot_window_messages():
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "user" and content:
                        self._append_user(content)
                    elif role == "assistant" and content:
                        self._append_assistant(content)
            except Exception as exc:
                debug_log(f"hot window replay failed: {exc}", "chat")
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Hide instead of destroying; the tray re-shows the same instance.
        # We intentionally do NOT call request_stop here — closing the chat
        # window does not stop the daemon or end the conversation. The explicit
        # hide() guarantees the window disappears regardless of how the close
        # is triggered (title bar button, ESC, tray toggle) and keeps the
        # instance alive so a reply that lands while hidden still lands here.
        self.hide()
        event.accept()
