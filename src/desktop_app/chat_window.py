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
from PyQt6.QtGui import QCloseEvent, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from desktop_app.themes import COLORS, JARVIS_THEME_STYLESHEET


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


class ChatWindow(QMainWindow):
    """Text chat window. Sends via ``jarvis.daemon.submit_text_query``.

    In subprocess mode the desktop app sets ``submit_fn`` to a callable that
    writes a ``__CHAT_QUERY__:`` line to the daemon's stdin, and feeds
    ``__CHAT__:`` events back into the window's signals. In bundled mode the
    default path calls the daemon directly with the window's signal emitters
    as callbacks.
    """

    def __init__(self, submit_fn=None) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis Chat")
        self.setMinimumSize(520, 560)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_fn = submit_fn

        # Signal bridge: daemon worker -> Qt main thread.
        self._signals = ChatSignals()
        self._signals.started.connect(self._on_start)
        self._signals.completed.connect(self._on_complete)
        self._signals.busy.connect(self._on_busy)

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

        # Status indicator
        self._status_label = QPushButton("")
        self._status_label.setStyleSheet(_STATUS_STYLE)
        self._status_label.setFlat(True)
        self._status_label.setCursor(Qt.CursorShape.ArrowCursor)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)

        # Input row: input box + send + stop
        row = QHBoxLayout()
        row.setSpacing(8)

        self.input_widget = QPlainTextEdit()
        self.input_widget.setPlaceholderText("Type a message to Jarvis… (Enter to send, Shift+Enter for newline)")
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

    # --- Sending --------------------------------------------------------

    def _send(self) -> None:
        text = self.input_widget.toPlainText().strip()
        if not text:
            return

        # Echo the user message into the transcript immediately.
        self._append_user(text)
        self.input_widget.setPlainText("")

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
                on_start=self._signals.started.emit,
                on_complete=self._signals.completed.emit,
                on_busy=self._signals.busy.emit,
            )

    def _stop(self) -> None:
        from jarvis import daemon

        daemon.request_stop()

    # --- Daemon callback slots (run on the main thread via signals) -----

    def _on_start(self, _query: str) -> None:
        # The user message is already echoed in _send. We keep the thinking
        # indicator on; nothing extra to render for the start event in the MVP.
        self._set_thinking(True)

    def _on_complete(self, reply: Optional[str]) -> None:
        self._set_thinking(False)
        if reply:
            self._append_assistant(reply)

    def _on_busy(self) -> None:
        self._set_thinking(False)
        self._append_system("Jarvis is busy with another query already.")

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
        self._query_in_flight = thinking
        self.stop_button.setVisible(thinking)
        self.send_button.setEnabled(not thinking)
        self._status_label.setVisible(thinking)
        if thinking:
            self._status_label.setText("  Jarvis is thinking…")

    # --- Input key handling ---------------------------------------------

    def _input_key_press(self, event) -> None:
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt as _Qt

        if isinstance(event, QKeyEvent) and event.key() == _Qt.Key.Key_Return and not (event.modifiers() & _Qt.KeyboardModifier.ShiftModifier):
            # Enter sends; Shift+Enter inserts a newline (default).
            self._send()
            return
        # Default handling for all other keys.
        QPlainTextEdit.keyPressEvent(self.input_widget, event)

    # --- Lifecycle ------------------------------------------------------

    def closeEvent(self, event: QCloseEvent) -> None:
        # Hide instead of destroying; the tray re-shows the same instance.
        # We intentionally do NOT call request_stop here — closing the chat
        # window does not stop the daemon or end the conversation.
        event.accept()
