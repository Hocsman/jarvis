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
from PyQt6.QtGui import QCloseEvent, QTextCursor
from PyQt6.QtWidgets import (
    QFrame,
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

# Height of the orb hero band at the top of the chat window. The orb
# renders centred within this strip (it sizes to min(width, height)).
_ORB_HERO_HEIGHT = 150


# ---------------------------------------------------------------------------
# Thread-safe signal bridge
# ---------------------------------------------------------------------------


def send_confirmation_decision(request_id: str, approved: bool) -> None:
    """Send one decision to the daemon, in whichever mode it is running.

    Bundled mode calls it directly — the daemon runs in-process, and
    ``cancel_active_chat_query`` is already called this way. Subprocess
    mode writes the one line on stdin that authorises an irreversible
    action; the daemon validates it there rather than trusting this side.

    Module-level so a test can replace it without reaching into a widget,
    and so both modes have one entry point rather than two that drift.
    """
    from jarvis import daemon

    writer = _decision_writer
    if writer is not None:
        writer(request_id, approved)
        return
    daemon.resolve_confirmation(request_id, approved)


# Set by the desktop app in subprocess mode, where the daemon is another
# process and decisions travel on its stdin.
_decision_writer = None


def set_decision_writer(writer) -> None:
    """Route decisions through the daemon's stdin (subprocess mode)."""
    global _decision_writer
    _decision_writer = writer


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

_CONFIRM_CARD_STYLE = f"""
    QFrame {{
        background-color: {COLORS['bg_tertiary']};
        border: 1px solid {COLORS['warning']};
        border-radius: 8px;
    }}
"""

_CONFIRM_TITLE_STYLE = f"""
    QLabel {{
        color: {COLORS['warning']};
        font-size: 13px;
        font-weight: 600;
        background: transparent;
        border: none;
    }}
"""

# Neither button is the attractive one. The card's own amber border
# already says "attention"; a filled call-to-action next to it would make
# approving the easy reflex, which is precisely backwards when approving
# is the irreversible half. Declining is quiet and neutral because it is
# the safe answer, not an alarming one; approving is outlined rather than
# filled, so it takes a deliberate look to find.
_CONFIRM_DECLINE_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        border-color: {COLORS['text_muted']};
    }}
    QPushButton:disabled {{
        color: {COLORS['text_muted']};
    }}
"""

_CONFIRM_APPROVE_STYLE = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLORS['warning']};
        border: 1px solid {COLORS['warning']};
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['bg_secondary']};
    }}
    QPushButton:disabled {{
        color: {COLORS['text_muted']};
        border-color: {COLORS['border']};
    }}
"""

_CONFIRM_DETAIL_STYLE = f"""
    QLabel {{
        color: {COLORS['text_primary']};
        font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
        font-size: 12px;
        background: transparent;
        border: none;
    }}
"""

_CONFIRM_HAZARD_STYLE = f"""
    QLabel {{
        color: {COLORS['error']};
        font-size: 12px;
        background: transparent;
        border: none;
    }}
"""

_CONFIRM_STATUS_STYLE = f"""
    QLabel {{
        color: {COLORS['text_muted']};
        font-size: 12px;
        background: transparent;
        border: none;
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

    def __init__(self, submit_fn=None, daemon_available: bool = True) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis Chat")
        self.setMinimumSize(520, 560)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_fn = submit_fn
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

        # Orb hero: the reactive orb sits at the top of the chat and is
        # the window's focal point. It pulses THINKING while a reply is
        # generating and settles to IDLE otherwise. Built defensively —
        # if the orb stack is unavailable for any reason, the chat still
        # works without it.
        self._orb = self._build_orb()
        if self._orb is not None:
            layout.addWidget(self._orb)

        # Transcript (read-only)
        self.transcript_widget = QPlainTextEdit()
        self.transcript_widget.setReadOnly(True)
        self.transcript_widget.setStyleSheet(_TRANSCRIPT_STYLE)
        layout.addWidget(self.transcript_widget, stretch=1)

        # Confirmation card. The only way a destructive action can be
        # approved — voice cannot grant one — so if this does not appear,
        # or its buttons do not work, such an action has no route to
        # happening and the user is left listening to an unanswerable
        # question.
        layout.addWidget(self._build_confirmation_card())

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
        self.set_daemon_available(daemon_available)

    # --- Confirmation ---------------------------------------------------

    def _build_confirmation_card(self) -> QFrame:
        """The card that asks permission for one action.

        Hidden until there is a question. Never modal: a user who wants
        to ask something else, or to say why they are hesitating, must
        not be stranded by a card.
        """
        card = QFrame()
        card.setStyleSheet(_CONFIRM_CARD_STYLE)
        card.setVisible(False)

        inner = QVBoxLayout(card)
        inner.setContentsMargins(12, 10, 12, 10)
        inner.setSpacing(6)

        self.confirmation_title = QLabel("🙋 Yuba demande ta permission")
        self.confirmation_title.setStyleSheet(_CONFIRM_TITLE_STYLE)
        inner.addWidget(self.confirmation_title)

        # The call, verbatim. This is the copy the user decides on: the
        # arguments are model output, and the ledger's copy is redacted
        # and whitespace-collapsed, so it cannot serve here.
        self.confirmation_detail = QLabel("")
        self.confirmation_detail.setStyleSheet(_CONFIRM_DETAIL_STYLE)
        self.confirmation_detail.setWordWrap(True)
        self.confirmation_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        inner.addWidget(self.confirmation_detail)

        self.confirmation_hazards = QLabel("")
        self.confirmation_hazards.setStyleSheet(_CONFIRM_HAZARD_STYLE)
        self.confirmation_hazards.setWordWrap(True)
        self.confirmation_hazards.setVisible(False)
        inner.addWidget(self.confirmation_hazards)

        self.confirmation_status = QLabel("")
        self.confirmation_status.setStyleSheet(_CONFIRM_STATUS_STYLE)
        self.confirmation_status.setVisible(False)
        inner.addWidget(self.confirmation_status)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        buttons.addStretch(1)

        # Declining is the default, so an absent-minded return key can
        # only ever be a no.
        self.confirmation_decline_button = QPushButton("Refuser")
        self.confirmation_decline_button.setStyleSheet(_CONFIRM_DECLINE_STYLE)
        self.confirmation_decline_button.setDefault(True)
        self.confirmation_decline_button.setAutoDefault(True)
        self.confirmation_decline_button.clicked.connect(
            lambda: self._decide_confirmation(False)
        )
        buttons.addWidget(self.confirmation_decline_button)

        self.confirmation_approve_button = QPushButton("Autoriser")
        self.confirmation_approve_button.setStyleSheet(_CONFIRM_APPROVE_STYLE)
        self.confirmation_approve_button.setDefault(False)
        self.confirmation_approve_button.setAutoDefault(False)
        self.confirmation_approve_button.clicked.connect(
            lambda: self._decide_confirmation(True)
        )
        buttons.addWidget(self.confirmation_approve_button)

        inner.addLayout(buttons)

        self.confirmation_card = card
        self._pending_request_id: Optional[str] = None
        return card

    def _show_confirmation(self, data: dict) -> None:
        """Put one question on screen."""
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            debug_log("confirm event without a request id ignored", "chat")
            return

        self._pending_request_id = request_id
        self.confirmation_detail.setText(str(data.get("shown") or ""))

        hazards = data.get("hazards") or []
        self.confirmation_hazards.setText("⚠️ " + " · ".join(str(h) for h in hazards))
        self.confirmation_hazards.setVisible(bool(hazards))

        self.confirmation_status.setVisible(False)
        self._set_confirmation_buttons(True)
        self.confirmation_card.setVisible(True)
        self.confirmation_decline_button.setFocus()

    def _set_confirmation_buttons(self, live: bool) -> None:
        self.confirmation_approve_button.setEnabled(live)
        self.confirmation_decline_button.setEnabled(live)

    def _decide_confirmation(self, approved: bool) -> None:
        """Send one decision, once."""
        request_id = self._pending_request_id
        if not request_id:
            return
        # Locked immediately: a second click before the daemon answers
        # would look to the user like the first one failed.
        self._set_confirmation_buttons(False)
        self.confirmation_status.setText("⏳ Transmission…")
        self.confirmation_status.setVisible(True)
        try:
            send_confirmation_decision(request_id, approved)
        except Exception as exc:
            debug_log(f"confirmation decision not sent: {exc}", "chat")
            self.confirmation_status.setText("⚠️ Décision non transmise. Réessaie.")
            self._set_confirmation_buttons(True)

    def _settle_confirmation(self, data: dict) -> None:
        """The question is over, one way or another."""
        if data.get("request_id") != self._pending_request_id:
            return
        outcome = str(data.get("outcome") or "")
        self._pending_request_id = None
        self.confirmation_card.setVisible(False)
        if outcome == "expiré":
            # Being told a decision was waiting and never told what became
            # of it is worse than being told it lapsed.
            self._append_line("⌛ La demande a expiré sans réponse.")
        elif outcome == "décliné":
            self._append_line("🚫 Refusé.")

    def _confirmation_not_taken(self, data: dict) -> None:
        """The click did not land. The question is still open."""
        if data.get("request_id") != self._pending_request_id:
            return
        outcome = str(data.get("outcome") or "")
        reasons = {
            "occupée": "⏳ Yuba est occupée. Réessaie dans un instant.",
            "arrêt": "🛑 Yuba s'arrête. Rien n'a été fait.",
            "inconnue": "❔ Cette demande n'est plus en attente.",
        }
        self.confirmation_status.setText(reasons.get(outcome, "⚠️ Décision non prise."))
        self.confirmation_status.setVisible(True)
        # Live again: the question is not settled, so the user must be
        # able to try. A correct decision that looks like a broken button
        # is the failure this whole channel exists to avoid.
        self._set_confirmation_buttons(outcome != "inconnue")

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if (event.key() == Qt.Key.Key_Escape
                and self._pending_request_id
                and self.confirmation_approve_button.isEnabled()):
            self._decide_confirmation(False)
            return
        super().keyPressEvent(event)

    # --- Orb hero -------------------------------------------------------

    def _build_orb(self):
        """Construct the embedded orb widget, or return ``None`` if the
        orb stack can't be loaded (so the chat degrades gracefully).

        The orb is sized to a fixed-height hero band and honours the
        ``ui.orb_particles_enabled`` config knob.
        """
        try:
            from desktop_app.orb.orb_widget import OrbWidget

            particles = True
            try:
                from jarvis.config import load_settings

                particles = bool(load_settings().ui.orb_particles_enabled)
            except Exception:
                pass  # default True if config can't be read

            orb = OrbWidget(particles_enabled=particles)
            # Relax the orb's own 320x320 minimum so it fits a slim hero
            # band; it renders centred at min(width, height).
            orb.setMinimumSize(0, 0)
            orb.setFixedHeight(_ORB_HERO_HEIGHT)
            return orb
        except Exception as exc:
            debug_log(f"chat orb unavailable, continuing without it: {exc}", "chat")
            return None

    def _set_orb_state(self, state_name: str) -> None:
        """Drive the embedded orb's state controller. No-op when the orb
        failed to load. ``state_name`` is an ``OrbState`` member name
        ("IDLE", "THINKING", ...)."""
        if self._orb is None:
            return
        try:
            from desktop_app.orb.state_controller import OrbState

            self._orb.state_controller().set_state(getattr(OrbState, state_name))
        except Exception as exc:
            debug_log(f"orb state set failed ({state_name}): {exc}", "chat")

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Resume the orb's render loop only while the window is visible —
        # no point burning 60 fps on a hidden window (also keeps the
        # laptop cooler when the chat is closed).
        if self._orb is not None:
            self._orb.resume_rendering()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._orb is not None:
            self._orb.pause_rendering()
        super().hideEvent(event)

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
        from jarvis import daemon

        # Cancel the in-flight chat query (drop the reply), NOT request_stop
        # which would tear down the whole voice assistant.
        daemon.cancel_active_chat_query()
        # Reset the thinking indicator immediately so the user sees feedback
        # without waiting for the engine to finish.
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
            # Daemon down: settle the orb to IDLE (no reply can be in
            # flight). We avoid an ERROR flare here — a stopped daemon is
            # an expected user action, not a fault.
            self._set_orb_state("IDLE")
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
        elif kind == "confirm":
            self._show_confirmation(data if isinstance(data, dict) else {})
        elif kind == "confirm_settled":
            self._settle_confirmation(data if isinstance(data, dict) else {})
        elif kind == "confirm_nack":
            self._confirmation_not_taken(data if isinstance(data, dict) else {})
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
        # The orb is the visual heartbeat of the reply: THINKING while a
        # query is in flight, back to IDLE once it settles.
        self._set_orb_state("THINKING" if self._query_in_flight else "IDLE")
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
