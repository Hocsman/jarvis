"""Frameless, translucent always-on-top window hosting the orb.

The window is intentionally minimal: it owns the GL widget, a drag
handler, a global hotkey for toggle visibility, and an optional DEV
badge that surfaces when the orb is running in subprocess dev mode
(no in-process daemon audio tap).

Cross-process detection
-----------------------
Phase 1 ships the bundled-mode reactive path only. When the orb is
launched standalone (``scripts/run_orb.sh``) it shares the daemon's
process and has direct memory access to the audio queue. When the
desktop_app is launched while a daemon subprocess runs the listener,
the orb cannot tap audio in real time and falls back to a synthetic
envelope. The badge tells the user which path is live.

Hotkey
------
``pynput.keyboard.GlobalHotKeys`` (project already depends on pynput
for dictation) registers ``cmd+shift+J`` on macOS / ``ctrl+shift+J``
elsewhere as a global toggle. A Qt ``QShortcut`` is not enough: it
fires only when the window has focus, which is never the case when
the window is hidden.
"""

from __future__ import annotations

import os
import platform
import sys
import threading
from typing import Any, Optional

# Qt is imported eagerly because the *window* is the unit of GUI we
# expose; tests for this module run in offscreen Qt mode and need the
# imports to land before fixtures execute.
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from .orb_widget import OrbWidget


# Default toggle key combo per platform. pynput names: <cmd> / <ctrl> /
# <shift>; lowercase letter is the key.
_TOGGLE_HOTKEY_MAC = "<cmd>+<shift>+j"
_TOGGLE_HOTKEY_OTHER = "<ctrl>+<shift>+j"


def _pynput_is_safe_on_this_platform() -> bool:
    """Return False on platforms where pynput's keyboard backend is
    known to crash the process.

    macOS 26+ (Tahoe): pynput's CGEventTap runs on a background thread
    that invokes TSM (Text Services Manager) APIs; macOS 26 enforces
    those APIs are called from the main dispatch queue and aborts the
    process via SIGTRAP otherwise. Same root cause as dictation issue
    #172. We disable the global hotkey on that platform and rely on
    the window-scoped Qt fallback.
    """
    if sys.platform != "darwin":
        return True
    try:
        mac_ver = platform.mac_ver()[0]
        major = int(mac_ver.split(".")[0]) if mac_ver else 0
    except (ValueError, IndexError):
        return True
    return major < 26


def _is_dev_mode() -> bool:
    """Return True if we're running in subprocess dev mode.

    Best-effort heuristic. We can't ask the daemon "are we the same
    process" so we sniff for the env var the desktop_app sets when it
    spawns the daemon. Override via ``JARVIS_ORB_FORCE_DEV=1`` /
    ``JARVIS_ORB_FORCE_PROD=1`` for testing.
    """
    if os.environ.get("JARVIS_ORB_FORCE_DEV") == "1":
        return True
    if os.environ.get("JARVIS_ORB_FORCE_PROD") == "1":
        return False
    # If the daemon was spawned by this same desktop_app process tree,
    # JARVIS_DAEMON_SUBPROCESS will be set on its env. We are the desktop
    # side, so we look for our own marker.
    return os.environ.get("JARVIS_DESKTOP_HAS_DAEMON_SUBPROCESS") == "1"


class OrbWindow(QMainWindow):
    """Top-level orb window.

    Public API:
    - ``show_orb()`` / ``hide_orb()`` / ``toggle_visibility()`` for
      programmatic control (also wired to the global hotkey).
    - ``trigger_error()`` proxies to the controller for Phase 1 manual
      ERROR demos.

    Lifecycle: register the global hotkey on first ``show()``,
    unregister on ``closeEvent``. The hotkey listener runs in its own
    pynput thread; we marshal state changes back onto the Qt thread
    via a ``pyqtSignal``.
    """

    # Signal used to marshal hotkey callbacks (pynput thread) onto the
    # Qt main thread. Qt connections with auto-connection type handle
    # the thread hop for us.
    _toggle_requested = pyqtSignal()

    DEFAULT_SIZE = 320

    def __init__(
        self,
        audio_bus: Optional[Any] = None,
        state_controller: Optional[Any] = None,
        particles_enabled: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # Frameless, translucent, always-on-top, hidden from dock on
        # macOS via the Tool flag.
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle("Jarvis Orb")
        self.resize(self.DEFAULT_SIZE, self.DEFAULT_SIZE)

        # Central widget hierarchy: GL widget + optional DEV badge.
        central = QWidget(self)
        # Transparent stylesheet so the GL widget shows the desktop
        # behind it where the orb has zero alpha.
        central.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._orb = OrbWidget(
            audio_bus=audio_bus,
            state_controller=state_controller,
            particles_enabled=particles_enabled,
            parent=central,
        )
        layout.addWidget(self._orb)

        self._dev_badge: Optional[QLabel] = None
        if _is_dev_mode():
            self._dev_badge = self._build_dev_badge(central)

        self.setCentralWidget(central)
        self._position_default()

        # Hotkey listener (pynput) is registered lazily on first show
        # to avoid surprising the system permission dialog when an
        # automated test only instantiates the window without showing.
        self._hotkey_listener_lock = threading.Lock()
        self._hotkey_listener: Optional[Any] = None
        self._toggle_requested.connect(self.toggle_visibility)

        # Window-scoped Qt fallback: works only while the orb has focus
        # (so it can't *show* a hidden window) but lets the user close
        # the orb with the same key combo on platforms where the global
        # pynput hotkey is disabled (macOS 26+).
        qt_combo = "Meta+Shift+J" if sys.platform == "darwin" else "Ctrl+Shift+J"
        self._qt_shortcut = QShortcut(QKeySequence(qt_combo), self)
        self._qt_shortcut.activated.connect(self.toggle_visibility)

        # Drag-to-move bookkeeping.
        self._drag_origin: Optional[QPoint] = None

    # ── Public API ─────────────────────────────────────────────────────

    def show_orb(self) -> None:
        self._ensure_hotkey_registered()
        self.show()

    def hide_orb(self) -> None:
        self.hide()

    def toggle_visibility(self) -> None:
        """Toggle show/hide. Safe to call from any thread (Qt routes
        through the connected signal when invoked from pynput)."""
        if self.isVisible():
            self.hide_orb()
        else:
            self.show_orb()

    def trigger_error(self) -> None:
        """Proxy: route a manual ERROR pulse through the orb widget's
        state controller. Phase 2 will wire this from the daemon's
        exception handlers; Phase 1 keeps it as a public test entry."""
        self._orb.trigger_error()

    # ── Qt overrides ───────────────────────────────────────────────────

    def showEvent(self, event) -> None:  # noqa: N802 (Qt API)
        # Register hotkey on first show.
        self._ensure_hotkey_registered()
        super().showEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        self._teardown_hotkey()
        super().closeEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            # Store the gap between cursor and window's top-left so the
            # drag tracks the cursor without "jumping" on the first move.
            self._drag_origin = event.globalPosition().toPoint() - self.pos()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # ── Internals ──────────────────────────────────────────────────────

    def _build_dev_badge(self, parent: QWidget) -> QLabel:
        """Bottom-right small label that flags subprocess dev mode."""
        badge = QLabel("DEV", parent)
        font = QFont("Menlo", 9)
        font.setBold(True)
        badge.setFont(font)
        badge.setStyleSheet(
            "QLabel { color: rgba(180, 180, 180, 180); background: transparent; }"
        )
        badge.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        # Position the badge inside the window via showEvent re-layout.
        badge.move(self.DEFAULT_SIZE - 34, self.DEFAULT_SIZE - 18)
        return badge

    def _position_default(self) -> None:
        """Centre the window on the primary screen."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.left() + (geo.width() - self.DEFAULT_SIZE) // 2
        y = geo.top() + (geo.height() - self.DEFAULT_SIZE) // 2
        self.move(x, y)

    def _hotkey_combo(self) -> str:
        return _TOGGLE_HOTKEY_MAC if sys.platform == "darwin" else _TOGGLE_HOTKEY_OTHER

    def _ensure_hotkey_registered(self) -> None:
        """Set up the global pynput hotkey listener once.

        Failures are non-fatal: a missing accessibility permission on
        macOS will silently mean "no global hotkey" but the rest of
        the orb keeps working. The window-scoped Qt shortcut still
        toggles the orb while it has focus on every platform.

        Skipped entirely on macOS 26+ where pynput is known to crash
        the process (TSM main-thread assertion, dictation issue #172).
        """
        with self._hotkey_listener_lock:
            if self._hotkey_listener is not None:
                return

            if not _pynput_is_safe_on_this_platform():
                print(
                    "  ⚠️  Orb global hotkey disabled on macOS 26+ "
                    "(pynput keyboard listener incompatibility); "
                    "use the window-focused Cmd+Shift+J inside the orb "
                    "or close via the tray.",
                    flush=True,
                )
                return

            try:
                from pynput import keyboard
            except Exception:
                return

            try:
                listener = keyboard.GlobalHotKeys(
                    {self._hotkey_combo(): self._toggle_requested.emit}
                )
                listener.daemon = True
                listener.start()
                self._hotkey_listener = listener
            except Exception:
                self._hotkey_listener = None

    def _teardown_hotkey(self) -> None:
        with self._hotkey_listener_lock:
            listener = self._hotkey_listener
            self._hotkey_listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
