"""
💬 Chat Window

A text chat interface for Jarvis, alongside the existing voice path. Voice
and text share one conversation (the daemon's global dialogue memory). See
``chat_window.spec.md`` for the full contract.

The window is created lazily by the system tray and kept alive for the
session. Daemon callback signals are marshalled onto the Qt main thread via
``ChatSignals`` so UI updates never touch the worker thread directly.

There is exactly one conversation, like a text-message thread with a single
contact: no session list, no new-session button, nothing written to disk.
Every sent message carries a subtle rewind button that rolls the
conversation back to that message and regenerates a fresh reply.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QRectF
from PyQt6.QtGui import (
    QCloseEvent, QShowEvent, QPainter, QColor, QPen, QLinearGradient,
    QRadialGradient, QShortcut, QKeySequence,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
    QSizeGrip,
)

from jarvis.debug import debug_log
from desktop_app.themes import COLORS, JARVIS_THEME_STYLESHEET, CHAT_THEME_STYLESHEET

# Height of the orb hero band at the top of the chat window. The orb
# renders centred within this strip (it sizes to min(width, height)).
_ORB_HERO_HEIGHT = 150


class PhoneShell(QWidget):
    """Rounded, layered chassis drawn inside the translucent window."""

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(self.rect()).adjusted(3, 3, -3, -3)
        metal = QLinearGradient(rect.topLeft(), rect.bottomRight())
        for position, colour in (
            (0, 'text_muted'), (.18, 'bg_hover'), (.5, 'border'),
            (.8, 'bg_primary'), (1, 'text_muted'),
        ):
            metal.setColorAt(position, QColor(COLORS[colour]))
        painter.setPen(QPen(QColor(COLORS['border']), 1))
        painter.setBrush(metal)
        painter.drawRoundedRect(rect, 36, 36)
        painter.setBrush(QColor(COLORS['bg_primary']))
        painter.drawRoundedRect(rect.adjusted(4, 4, -4, -4), 32, 32)


class ChatTitleBar(QWidget):
    """Desktop window movement, with a fallback for unsupported platforms."""

    def mousePressEvent(self, event) -> None:
        self._drag_offset = None
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self.window().windowHandle()
            if handle is None or not handle.startSystemMove():
                self._drag_offset = event.globalPosition().toPoint() - self.window().pos()
            event.accept()

    def mouseMoveEvent(self, event) -> None:
        offset = getattr(self, '_drag_offset', None)
        if offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - offset)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            if window.isMaximized():
                window.showNormal()
            else:
                window.showMaximized()


class JarvisOrb(QWidget):
    """Resolution-independent amber core, with no external image assets."""

    def __init__(self, size: int, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAccessibleName('Jarvis')

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(self.width() / 100, self.height() / 100)
        glow = QRadialGradient(50, 50, 50)
        colour = QColor(COLORS['accent_primary'])
        colour.setAlpha(65)
        glow.setColorAt(0, colour)
        colour.setAlpha(0)
        glow.setColorAt(1, colour)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QRectF(0, 0, 100, 100))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLORS['accent_muted']), .8))
        painter.drawEllipse(QRectF(16, 16, 68, 68))
        painter.setPen(QPen(QColor(COLORS['accent_secondary']), 2))
        painter.drawArc(QRectF(22, 22, 56, 56), 25 * 16, 130 * 16)
        painter.drawArc(QRectF(22, 22, 56, 56), 205 * 16, 130 * 16)
        painter.setPen(QPen(
            QColor(COLORS['accent_secondary']), 3,
            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
        ))
        for x, height in ((38, 10), (46, 24), (54, 32), (62, 16)):
            painter.drawLine(x, 50 - height // 2, x, 50 + height // 2)


class ChatSizeGrip(QSizeGrip):
    """A visible resize affordance, including on platforms with blank grips."""

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(
            QColor(COLORS['text_muted']), 1,
            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
        ))
        painter.drawLine(5, 12, 12, 5)
        painter.drawLine(10, 12, 12, 10)


def send_confirmation_decision(request_id: str, approved: bool) -> None:
    """Send one decision to the daemon, in whichever mode it is running.

    Bundled mode calls it directly: the daemon runs in-process, and
    cancel_active_chat_query is already called this way. Subprocess
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
# Styles
# ---------------------------------------------------------------------------

_TRANSCRIPT_AREA_STYLE = f"""
    QScrollArea {{
        background-color: {COLORS['bg_primary']};
        border: none;
    }}
"""

_INPUT_STYLE = f"""
    QPlainTextEdit {{
        background-color: {COLORS['bg_secondary']};
        color: {COLORS['text_primary']};
        border: none;
        border-radius: 14px;
        padding: 8px 4px;
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
        color: {COLORS['bg_primary']};
        border: none;
        border-radius: 18px;
        padding: 0;
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
        color: {COLORS['text_primary']};
        border: none;
        border-radius: 18px;
        padding: 0;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['error_light']};
    }}
"""

# A subtle ghost button: SMS threads don't advertise actions, but the rewind
# affordance stays reachable next to each sent message.
_REWIND_BTN_STYLE = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLORS['text_muted']};
        border: none;
        border-radius: 12px;
        font-size: 13px;
        padding: 2px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['bg_hover']};
        color: {COLORS['accent_secondary']};
    }}
    QPushButton:disabled {{
        color: {COLORS['border']};
    }}
"""

_STATUS_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 12px;
        padding: 2px 4px;
    }}
"""

_HEADER_STATUS_STYLE = f"""
    QLabel {{
        color: {COLORS['text_muted']};
        font-size: 12px;
    }}
"""

# SMS-style bubbles: the user's messages sit on the right in the accent
# colour, Jarvis's replies on the left in a dark bubble. The corner nearest
# the sender is squared off, like a speech bubble.
_BUBBLE_STYLES = {
    "user": f"""
        QLabel {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {COLORS['accent_secondary']}, stop:1 {COLORS['accent_primary']});
            color: {COLORS['bg_primary']};
            border-radius: 18px;
            border-bottom-right-radius: 5px;
            padding: 12px 16px;
            font-size: 14px;
        }}
    """,
    "assistant": f"""
        QLabel {{
            background-color: {COLORS['bg_secondary']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 18px;
            border-bottom-left-radius: 5px;
            padding: 12px 16px;
            font-size: 14px;
        }}
    """,
}

_TIMESTAMP_STYLE = f"""
    QLabel {{
        color: {COLORS['text_muted']};
        font-size: 11px;
    }}
"""

_MESSAGE_TEXT_STYLES = {
    "system": f"color: {COLORS['text_muted']}; font-size: 12px;",
}

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
    "running": "Message Jarvis…",
}

_HEADER_STATUS_TEXTS = {
    "running": "Online",
    "starting": "Starting…",
    "stopping": "Stopping…",
    "stopped": "Offline",
    "crashed": "Offline",
}

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


class ChatWindow(QMainWindow):
    """Text chat window. Sends via ``jarvis.daemon.submit_text_query``.

    In subprocess mode the desktop app sets ``submit_fn`` to a callable that
    writes a ``__CHAT_QUERY__:`` line to the daemon's stdin, ``cancel_fn`` to
    the cancel line writer, and ``control_fn`` to a callable that writes the
    rewind line. In bundled mode the window calls the daemon directly.

    There is a single conversation, displayed like an SMS thread: no session
    list, no new-session button. The transcript maps 1:1 to the daemon's
    shared dialogue memory, so the voice path sees the same turns. Every
    sent message carries a subtle rewind button that truncates the
    conversation to before that message and regenerates a fresh reply.
    """

    def __init__(
        self,
        submit_fn=None,
        daemon_available: bool = True,
        cancel_fn=None,
        control_fn: Optional[Callable[[str, Optional[dict]], None]] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis Chat")
        self.setObjectName('chatWindow')
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # A portrait, phone-like window reads as a message thread. The tray
        # re-shows the same instance, so the size persists for the session.
        screen = self.screen()
        if screen is None:
            from PyQt6.QtWidgets import QApplication
            screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(480, available.width()), min(800, available.height()))
        else:
            self.resize(480, 800)
        # The floor below which the composer, the controls and a raised
        # confirmation card stop being usable; the grip cannot go under it.
        self.setMinimumSize(380, 560)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET + CHAT_THEME_STYLESHEET)
        self._submit_fn = submit_fn
        # Subprocess mode routes cancellation to the daemon the same way it
        # routes a submission. Without it, Stop sets a flag in this
        # process while the query runs in the other one.
        self._cancel_fn = cancel_fn
        # Subprocess mode routes rewind to the daemon's stdin. Bundled mode
        # calls the daemon module directly and leaves this None.
        self._control_fn = control_fn
        # Set by Stop, cleared by the next send. The engine keeps running
        # after a cancel and its reply still arrives, so the window has to
        # decline the answer to an exchange the user walked away from.
        self._query_cancelled = False
        # The user row echoed by the last send, until the daemon says
        # whether it took the turn. A row it did not take keeps its bubble
        # but loses its ordinal, so later turns keep counting in step with
        # the memory.
        self._sent_row: Optional[dict] = None
        # A rewind asked of the daemon and not yet answered:
        # (user_index, text, rows to keep). The transcript changes only on
        # the answer, so it cannot part ways with the memory on a refusal.
        self._rewind_pending: Optional[tuple] = None
        self._daemon_available = daemon_available
        self._daemon_status = "running" if daemon_available else "stopped"

        # The single conversation's transcript. In-memory only; nothing is
        # written to disk, and a fresh app run starts blank (the daemon's
        # dialogue memory owns the durable record).
        self._messages: list[dict] = []
        # The bubble labels on screen, in transcript order, so a resize
        # re-caps them without walking the whole widget tree.
        self._bubbles: list[QLabel] = []

        # Signal bridge: daemon worker -> Qt main thread.
        self.signals = ChatSignals()
        self.signals.started.connect(self._on_start)
        self.signals.completed.connect(self._on_complete)
        self.signals.busy.connect(self._on_busy)

        # --- Layout -----------------------------------------------------
        central = PhoneShell()
        central.setObjectName('phoneShell')
        self.setCentralWidget(central)
        shell_layout = QVBoxLayout(central)
        shell_layout.setContentsMargins(10, 10, 10, 10)
        surface = QWidget()
        surface.setObjectName('chatSurface')
        shell_layout.addWidget(surface)
        root = QVBoxLayout(surface)
        root.setContentsMargins(18, 10, 18, 10)
        root.setSpacing(12)

        self.title_bar = ChatTitleBar()
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(0, 0, 0, 0)
        self.minimise_button = QPushButton('−')
        self.close_button = QPushButton('×')
        for button, name, callback in (
            (self.minimise_button, 'Minimise chat', self.showMinimized),
            (self.close_button, 'Close chat', self.close),
        ):
            button.setObjectName('chatWindowControl')
            button.setFixedSize(36, 36)
            button.setAccessibleName(name)
            button.setToolTip(name)
            button.clicked.connect(callback)
        title_layout.addWidget(self.minimise_button)
        title_layout.addStretch()
        capsule = QLabel('J A R V I S  /  CHAT')
        capsule.setObjectName('chatCapsule')
        capsule.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        title_layout.addWidget(capsule)
        title_layout.addStretch()
        title_layout.addWidget(self.close_button)
        root.addWidget(self.title_bar)
        self._close_shortcut = QShortcut(QKeySequence.StandardKey.Close, self)
        self._close_shortcut.activated.connect(self.close)

        # Contact header, like the top of an SMS thread.
        header = QHBoxLayout()
        header.setSpacing(10)
        avatar = JarvisOrb(60)
        header.addWidget(avatar)
        name_col = QVBoxLayout()
        name_col.setSpacing(3)
        name_label = QLabel("Jarvis")
        name_label.setObjectName('chatName')
        name_col.addWidget(name_label)
        self._header_status = QLabel("")
        self._header_status.setStyleSheet(_HEADER_STATUS_STYLE)
        name_col.addWidget(self._header_status)
        header.addLayout(name_col)
        header.addStretch(1)
        companion = QLabel('YOUR\nCOMPANION')
        companion.setObjectName('chatEyebrow')
        companion.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(companion)
        root.addLayout(header)

        self._orb = self._build_orb()
        self.empty_state = QWidget()
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(8, 0, 8, 0)
        empty_layout.setSpacing(12)
        empty_layout.addStretch()
        if self._orb is not None:
            empty_layout.addWidget(self._orb)
        else:
            empty_layout.addWidget(JarvisOrb(120), alignment=Qt.AlignmentFlag.AlignCenter)
        for text, name in (
            ('A little space to think.', 'chatEmptyTitle'),
            ('Ask a question. Follow a thought.\nPick up where your voice left off.', 'chatEmptyDetail'),
        ):
            label = QLabel(text)
            label.setObjectName(name)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_layout.addWidget(label)
        empty_layout.addStretch()
        root.addWidget(self.empty_state, stretch=1)

        # Transcript: a scroll area whose container holds one row widget per
        # message, so sent messages can carry a rewind button. Rebuilt
        # atomically on rewind (see _render_transcript).
        self.transcript_widget = QScrollArea()
        self.transcript_widget.setWidgetResizable(True)
        self.transcript_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.transcript_widget.setStyleSheet(_TRANSCRIPT_AREA_STYLE)
        # The view follows the newest message only while the reader is at
        # the end. A resize or a re-wrap changes the range too, and a reader
        # who scrolled up to re-read something keeps their place through it.
        scroll_bar = self.transcript_widget.verticalScrollBar()
        self._follow_bottom = True
        scroll_bar.rangeChanged.connect(self._on_transcript_range_changed)
        scroll_bar.valueChanged.connect(self._on_transcript_value_changed)
        self._transcript_container = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript_container)
        self._transcript_layout.setContentsMargins(4, 4, 4, 4)
        self._transcript_layout.setSpacing(16)
        self._transcript_layout.addStretch(1)
        self.transcript_widget.setWidget(self._transcript_container)
        root.addWidget(self.transcript_widget, stretch=1)
        self.transcript_widget.hide()

        # Confirmation card. The only way a destructive action can be
        # approved: voice cannot grant one, so if this does not appear,
        # or its buttons do not work, such an action has no route to
        # happening and the user is left listening to an unanswerable
        # question.
        root.addWidget(self._build_confirmation_card())

        # Status indicator (display-only label)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_STATUS_STYLE)
        self._status_label.setWordWrap(True)
        self._status_label.setVisible(False)
        root.addWidget(self._status_label)

        # Input row: input box + send + stop
        composer = QWidget()
        composer.setObjectName('chatComposer')
        row = QHBoxLayout(composer)
        row.setContentsMargins(12, 10, 10, 10)
        row.setSpacing(8)

        self.input_widget = QPlainTextEdit()
        self.input_widget.setPlaceholderText(_DAEMON_STATUS_PLACEHOLDERS["running"])
        self.input_widget.setFixedHeight(58)
        self.input_widget.setAccessibleName('Message Jarvis')
        self.input_widget.setToolTip('Enter to send. Shift+Enter for a new line.')
        self.input_widget.setStyleSheet(_INPUT_STYLE)
        self.input_widget.keyPressEvent = self._input_key_press  # type: ignore[method-assign]
        row.addWidget(self.input_widget, stretch=1)

        self.send_button = QPushButton("↑")
        self.send_button.setFixedSize(38, 38)
        self.send_button.setAccessibleName('Send message')
        self.send_button.setToolTip('Send message (Enter)')
        self.send_button.setStyleSheet(_SEND_BTN_STYLE)
        self.send_button.clicked.connect(self._send)
        row.addWidget(self.send_button)

        self.stop_button = QPushButton("■")
        self.stop_button.setFixedSize(38, 38)
        self.stop_button.setAccessibleName('Stop response')
        self.stop_button.setToolTip('Stop response')
        self.stop_button.setStyleSheet(_STOP_BTN_STYLE)
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setVisible(False)
        row.addWidget(self.stop_button)

        root.addWidget(composer)
        hint = QLabel('ENTER TO SEND  ·  SHIFT + ENTER FOR A NEW LINE')
        hint.setObjectName('chatEyebrow')
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        root.addWidget(hint)
        footer = QHBoxLayout()
        footer.addSpacing(16)
        footer.addStretch()
        indicator = QLabel()
        indicator.setObjectName('chatHomeIndicator')
        indicator.setFixedSize(88, 4)
        footer.addWidget(indicator)
        footer.addStretch()
        grip = ChatSizeGrip(self)
        grip.setFixedSize(16, 16)
        grip.setToolTip('Resize chat')
        footer.addWidget(grip)
        root.addLayout(footer)

        self._query_in_flight = False
        # Whether the transcript has been seeded from the daemon's hot window.
        # Seeded once on first show so re-opening never duplicates turns.
        self._hot_window_seeded = False
        self.set_daemon_available(daemon_available)
        self.input_widget.setFocus()

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
        self.confirmation_detail.setTextFormat(Qt.TextFormat.PlainText)
        self.confirmation_detail.setWordWrap(True)
        self.confirmation_detail.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        inner.addWidget(self.confirmation_detail)

        self.confirmation_hazards = QLabel("")
        self.confirmation_hazards.setStyleSheet(_CONFIRM_HAZARD_STYLE)
        self.confirmation_hazards.setTextFormat(Qt.TextFormat.PlainText)
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
            self._append_system("⌛ La demande a expiré sans réponse.")
        elif outcome == "décliné":
            self._append_system("🚫 Refusé.")

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

    # --- Sending --------------------------------------------------------

    def _send(self) -> None:
        text = self.input_widget.toPlainText().strip()
        if not text:
            return
        if self._query_in_flight or not self._daemon_available:
            if not self._daemon_available:
                self._append_system("Start Listening to use chat.")
            return

        # Echo the user message into the transcript immediately.
        self._append_user(text)
        self._sent_row = self._messages[-1]
        self.input_widget.setPlainText("")

        self._query_cancelled = False
        self._set_thinking(True)
        self._submit(text)

    def _stop(self) -> None:
        """Abandon the in-flight query. Never request_stop, which would
        tear down the whole voice assistant."""
        # Refuse the reply locally first: cancellation cannot unwind a
        # request already inside the engine, so the answer arrives either
        # way and this is what keeps it out of the transcript.
        self._query_cancelled = True
        if self._pending_request_id:
            self._pending_request_id = None
            self.confirmation_card.setVisible(False)

        if self._cancel_fn is not None:
            # Subprocess mode: the query lives in the daemon process.
            self._cancel_fn()
        else:
            from jarvis import daemon

            daemon.cancel_active_chat_query()

        # Reset the thinking indicator immediately so the user sees
        # feedback without waiting for the engine to finish.
        self._set_thinking(False)

    def _rewind_to_user(self, user_index: int, text: str) -> None:
        """Roll the conversation back to before the ``user_index``-th user
        message and regenerate a fresh reply to it.

        The daemon is asked first and answers with a verdict: the memory is
        the conversation and the transcript only a view of it, so the view
        changes once the memory has, never before. Bundled mode answers in
        the call; subprocess mode answers with a ``rewound`` or
        ``rewind_nack`` event. The daemon anchors on the message text and
        uses the ordinal only to tell identical messages apart, so a turn
        the memory no longer holds is refused rather than approximated.
        Rewinding is disabled while a query or another rewind is in flight.
        """
        if not self._rewind_allowed():
            return
        keep_until = None
        for i, m in enumerate(self._messages):
            if m.get("kind") == "user" and m.get("user_index") == user_index:
                keep_until = i + 1
                break
        if keep_until is None:
            return
        self._rewind_pending = (user_index, text, keep_until)
        self._query_cancelled = False
        self._set_thinking(True)
        if self._control_fn is not None:
            # Subprocess mode: the memory lives in the daemon process.
            self._control_fn("rewind", {"user_index": user_index, "content": text})
            return
        from jarvis import daemon

        self._on_rewind_settled(user_index, daemon.rewind_chat_to_user(user_index, text))

    def _on_rewind_settled(self, user_index: int, rewound: bool) -> None:
        """The daemon's verdict on the rewind in flight."""
        pending = self._rewind_pending
        if pending is None or pending[0] != user_index:
            return
        self._rewind_pending = None
        _, text, keep_until = pending
        if not rewound:
            debug_log(f"chat rewind refused for user message {user_index}", "chat")
            self._set_thinking(False)
            self._append_system("Could not rewind to that message.")
            return
        if self._pending_request_id:
            # The daemon closed the question along with the turn that asked
            # it; its settlement is on its way and needs no second notice.
            self._pending_request_id = None
            self.confirmation_card.setVisible(False)
        self._messages = self._messages[:keep_until]
        self._render_transcript(self._messages)
        debug_log(
            f"chat rewound to user message {user_index}, {keep_until} rows kept", "chat"
        )
        if self._query_cancelled:
            # Stop was pressed while the daemon was answering: the memory
            # is rewound and the transcript matches it, nothing is asked.
            self._query_cancelled = False
            self._set_thinking(False)
            return
        # Regenerate: re-submit the same message for a fresh reply. The
        # message is already displayed, so no new echo is added.
        self._sent_row = self._messages[-1]
        self._set_thinking(True)
        self._submit(text)

    def _submit(self, text: str) -> None:
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

    def set_daemon_hooks(
        self,
        submit_fn=None,
        cancel_fn=None,
        control_fn=None,
    ) -> None:
        """Update backend submission, cancellation, and control callables."""
        self._submit_fn = submit_fn
        self._cancel_fn = cancel_fn
        self._control_fn = control_fn

    def set_daemon_available(self, available: bool) -> None:
        """Enable or disable chat submission based on daemon availability."""
        self.set_daemon_status("running" if available else "stopped")

    def set_daemon_status(self, status: str) -> None:
        """Reflect daemon lifecycle state in chat controls and status text."""
        if status != "running" and status not in _DAEMON_STATUS_MESSAGES:
            debug_log(f"unknown chat daemon status ignored: {status}", "chat")
            status = "stopped"

        was_running = self._daemon_status == "running"
        self._daemon_status = status
        self._daemon_available = status == "running"
        if not self._daemon_available:
            self._query_in_flight = False
            self._rewind_pending = None
            self.stop_button.setVisible(False)
            self._set_orb_state("IDLE")
        elif not was_running:
            # A daemon that comes back starts a fresh memory. The rows on
            # screen stay as history, but their ordinals named turns of a
            # memory that no longer exists, so they can no longer be rewound
            # to, and the next show may seed what the new memory holds.
            self._retire_rewind_anchors()
            self._hot_window_seeded = False
        self.input_widget.setEnabled(self._daemon_available)
        self.input_widget.setPlaceholderText(
            _DAEMON_STATUS_PLACEHOLDERS.get(
                status,
                _DAEMON_STATUS_PLACEHOLDERS["stopped"],
            )
        )
        self._refresh_status_label()
        self._refresh_send_button()
        self._refresh_header_status()

    # --- Daemon callback slots (run on the main thread via signals) -----

    def _on_start(self, _query: str) -> None:
        # The user message is already echoed in _send. We keep the thinking
        # indicator on; nothing extra to render for the start event in the MVP.
        self._set_thinking(True)

    def _on_complete(self, reply: Optional[str]) -> None:
        sent_row = self._sent_row
        self._sent_row = None
        self._set_thinking(False)
        if self._query_cancelled:
            # The engine ran to the end and stored the turn; only the
            # answer is declined, so the row keeps its ordinal.
            debug_log("chat reply dropped: the query was cancelled", "chat")
            self._query_cancelled = False
            return
        if reply:
            self._append_assistant(reply)
        elif sent_row is not None:
            # The engine gave up before storing the turn.
            self._retract_ordinal(sent_row)

    def _on_busy(self) -> None:
        sent_row = self._sent_row
        self._sent_row = None
        self._set_thinking(False)
        if sent_row is not None:
            self._retract_ordinal(sent_row)
        self._append_system("Jarvis is busy with another query already.")

    def _retract_ordinal(self, row: dict) -> None:
        """The daemon never took this turn: the bubble stays, the rewind
        button goes, and later turns keep counting in step with the memory."""
        if row.get("user_index") is None:
            return
        row["user_index"] = None
        self._render_transcript(self._messages)

    def _retire_rewind_anchors(self) -> None:
        """Rows that named turns of a memory that is gone keep their bubble
        and lose their rewind button."""
        changed = False
        for m in self._messages:
            if m.get("kind") == "user" and m.get("user_index") is not None:
                m["user_index"] = None
                changed = True
        if changed:
            self._render_transcript(self._messages)

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
        if not isinstance(payload, dict):
            debug_log(f"non-dict {CHAT_IPC_PREFIX} line ignored", "chat")
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
        elif kind in ("rewound", "rewind_nack"):
            user_index = data.get("user_index") if isinstance(data, dict) else None
            if isinstance(user_index, int):
                self._on_rewind_settled(user_index, kind == "rewound")
        return True

    # --- Rendering helpers ----------------------------------------------

    def _append_user(self, text: str) -> None:
        # Ordinal over the turns the memory holds: a row without one is a
        # message the daemon never took, or a turn of a memory that is gone.
        user_index = 1 + sum(
            1 for m in self._messages
            if m.get("kind") == "user" and m.get("user_index") is not None
        )
        self._append_message("user", text, user_index=user_index)

    def _append_assistant(self, text: str) -> None:
        self._append_message("assistant", text)

    def _append_system(self, text: str) -> None:
        self._append_message("system", text)

    def _append_message(self, kind: str, text: str, user_index: Optional[int] = None) -> None:
        """Add one message row to the transcript."""
        # Decided from state, not from what is on screen: a message can land
        # while the window is hidden, and the panel must still be gone when
        # the window is next shown.
        self.empty_state.hide()
        if self._orb is not None:
            self._orb.pause_rendering()
        self.transcript_widget.show()
        self._messages.append(
            {
                "kind": kind,
                "text": text,
                "user_index": user_index,
                "time": datetime.now().strftime("%H:%M"),
            }
        )
        # Append a single row (before the trailing stretch) instead of
        # rebuilding the whole transcript, so long sessions stay O(n).
        row = self._make_message_row(self._messages[-1])
        self._transcript_layout.insertWidget(
            self._transcript_layout.count() - 1, row
        )
        self._scroll_to_bottom()

    def _render_transcript(self, messages: list) -> None:
        """Rebuild the transcript rows atomically from ``messages``.

        Rebuilding (instead of incrementally appending) keeps rewind
        truncation trivially correct: the rendered rows always mirror the
        message list.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(16)
        is_empty = not messages
        self.empty_state.setVisible(is_empty)
        if self._orb is not None:
            if is_empty and self.isVisible():
                self._orb.resume_rendering()
            else:
                self._orb.pause_rendering()
        self.transcript_widget.setVisible(bool(messages))
        self._bubbles = []
        for m in messages:
            layout.addWidget(self._make_message_row(m))
        layout.addStretch(1)
        old = self.transcript_widget.takeWidget()
        self.transcript_widget.setWidget(container)
        self._transcript_container = container
        self._transcript_layout = layout
        if old is not None:
            old.hide()
            old.deleteLater()
        self._scroll_to_bottom()

    def _make_message_row(self, m: dict) -> QWidget:
        kind = m.get("kind", "system")
        text = m.get("text", "")
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        if kind in ("user", "assistant"):
            # Bubble with a timestamp underneath, aligned to the sender's edge.
            bubble = QLabel(text)
            bubble.setObjectName("bubble")
            bubble.setTextFormat(Qt.TextFormat.PlainText)
            bubble.setWordWrap(True)
            bubble.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            bubble.setStyleSheet(_BUBBLE_STYLES[kind])
            bubble.setMaximumWidth(self._bubble_width())
            self._bubbles.append(bubble)
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(bubble)
            time_label = QLabel(m.get("time") or "")
            time_label.setObjectName("timestamp")
            time_label.setStyleSheet(_TIMESTAMP_STYLE)
            column.addWidget(
                time_label,
                alignment=Qt.AlignmentFlag.AlignRight,
            )
            if kind == "user":
                # SMS puts the sender's messages on the right; the rewind
                # affordance sits quietly to the left of the bubble. A row
                # with no ordinal names no turn in the memory, so it has
                # nothing to rewind to.
                row.addStretch(1)
                user_index = m.get("user_index")
                if user_index is not None:
                    rewind_btn = QPushButton("⟲")
                    rewind_btn.setObjectName(f"rewind_{user_index}")
                    rewind_btn.setToolTip("Rewind to this message and regenerate")
                    rewind_btn.setStyleSheet(_REWIND_BTN_STYLE)
                    rewind_btn.setFixedSize(26, 26)
                    rewind_btn.setEnabled(self._rewind_allowed())
                    rewind_btn.clicked.connect(
                        lambda _checked=False, idx=user_index, txt=text:
                        self._rewind_to_user(idx, txt)
                    )
                    row.addWidget(
                        rewind_btn, alignment=Qt.AlignmentFlag.AlignVCenter
                    )
                row.addLayout(column)
            else:
                row.addLayout(column)
                row.addStretch(1)
        else:
            label = QLabel(f"  ⏳ {text}")
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            label.setStyleSheet(_MESSAGE_TEXT_STYLES["system"])
            row.addStretch(1)
            row.addWidget(label)
            row.addStretch(1)
        return row_widget

    def resizeEvent(self, event) -> None:
        # Keep bubbles at a phone-like share of the window width, so the
        # thread reads as SMS whether the window is narrow or maximised.
        # The cap depends on the width alone, so a height-only resize (a
        # grip drag straight down, a card raising) re-caps nothing.
        super().resizeEvent(event)
        if not hasattr(self, "_bubbles"):
            return
        if event.oldSize().width() == event.size().width():
            return
        max_w = self._bubble_width()
        for bubble in self._bubbles:
            bubble.setMaximumWidth(max_w)

    def _bubble_width(self) -> int:
        return max(120, int((self.width() - 64) * 0.82))

    def _scroll_to_bottom(self) -> None:
        """Keep the latest message visible after Qt settles the new row."""
        self._follow_bottom = True
        bar = self.transcript_widget.verticalScrollBar()
        bar.setValue(bar.maximum())
        # Inserting a message schedules a layout pass. At this point the
        # scrollbar maximum can still describe the transcript before the new
        # row: the range change that follows pins the view to the new end
        # (_on_transcript_range_changed), and this catches a row that changed
        # nothing the bar could notice.
        QTimer.singleShot(0, self._finish_scroll_to_bottom)

    def _finish_scroll_to_bottom(self) -> None:
        if self._follow_bottom:
            bar = self.transcript_widget.verticalScrollBar()
            bar.setValue(bar.maximum())

    def _on_transcript_value_changed(self, value: int) -> None:
        bar = self.transcript_widget.verticalScrollBar()
        self._follow_bottom = value >= bar.maximum()

    def _on_transcript_range_changed(self, _minimum: int, maximum: int) -> None:
        if self._follow_bottom:
            self.transcript_widget.verticalScrollBar().setValue(maximum)

    def transcript_text(self) -> str:
        """Plain-text rendering of the transcript (testing + copy).

        The bubbles carry no role prefixes in the UI: position and colour
        convey the sender, so the text dump is just the message bodies.
        """
        return "\n".join(m["text"] for m in self._messages)

    def _set_thinking(self, thinking: bool) -> None:
        self._query_in_flight = thinking and self._daemon_available
        self.stop_button.setVisible(self._query_in_flight)
        self._set_orb_state("THINKING" if self._query_in_flight else "IDLE")
        self._refresh_status_label()
        self._refresh_send_button()
        self._refresh_rewind_buttons()
        self._refresh_header_status()

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

    def _rewind_allowed(self) -> bool:
        return (
            self._daemon_available
            and not self._query_in_flight
            and self._rewind_pending is None
        )

    def _refresh_rewind_buttons(self) -> None:
        """Disable rewind while a query or a rewind is in flight or the
        daemon is down."""
        enabled = self._rewind_allowed()
        for btn in self.transcript_widget.findChildren(QPushButton):
            if btn.objectName().startswith("rewind_"):
                btn.setEnabled(enabled)

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

    def _refresh_header_status(self) -> None:
        """Contact-style presence line, like the header of an SMS thread."""
        if self._query_in_flight:
            self._header_status.setText("Typing…")
            return
        self._header_status.setText(
            _HEADER_STATUS_TEXTS.get(self._daemon_status, "Offline")
        )

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

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if (event.key() == Qt.Key.Key_Escape
                and self._pending_request_id
                and self.confirmation_approve_button.isEnabled()):
            self._decide_confirmation(False)
            return
        super().keyPressEvent(event)

    # --- Lifecycle ------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        # The orb lives in the introductory panel, which only an empty
        # transcript shows; decided from the model, so a message that landed
        # while the window was hidden counts.
        if self._orb is not None and not self._messages:
            self._orb.resume_rendering()
        # The first show of a daemon's life seeds the transcript from its hot
        # window, so a user who has been talking by voice sees their recent
        # turns instead of a blank panel. It runs once per daemon life:
        # re-showing (from the tray or after a hide) never duplicates turns.
        # In subprocess mode the memory lives in the other process and the
        # accessor returns nothing, so the window opens blank.
        if not self._hot_window_seeded and self._daemon_available:
            self._hot_window_seeded = True
            seeded = 0
            try:
                for msg in get_hot_window_messages():
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "user" and content:
                        self._append_user(content)
                    elif role == "assistant" and content:
                        self._append_assistant(content)
                    else:
                        continue
                    seeded += 1
            except Exception as exc:
                debug_log(f"hot window replay failed: {exc}", "chat")
            if seeded:
                debug_log(f"chat transcript seeded with {seeded} hot-window turns", "chat")
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._orb is not None:
            self._orb.pause_rendering()
        super().hideEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Hide instead of destroying; the tray re-shows the same instance.
        # We intentionally do NOT call request_stop here, closing the chat
        # window does not stop the daemon or end the conversation. The explicit
        # hide() guarantees the window disappears regardless of how the close
        # is triggered (title bar button, the platform close shortcut) and
        # keeps the instance alive so a reply that lands while hidden still
        # lands here.
        self.hide()
        event.accept()
