"""HUD dashboard window: a single full-screen interface hosting the
J.A.R.V.I.S dashboard rendered as a local web page.

The page (``dashboard/index.html``) is driven live by a
:class:`DashboardBridge` exposed over ``QWebChannel`` as ``window.jarvis``:
system stats, voice state (orb colour + status), weather, and the chat
round-trip all flow through it. The host (the tray) wires the chat
submission callable and forwards daemon replies via ``deliver_reply``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from jarvis.debug import debug_log
from desktop_app.dashboard.bridge import DashboardBridge


def _dashboard_index() -> Path:
    return Path(__file__).resolve().parent / "dashboard" / "index.html"


class DashboardWindow(QMainWindow):
    """Main window rendering the HUD dashboard with a live data bridge."""

    def __init__(self, submit_fn: Optional[Callable[[str], None]] = None) -> None:
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S")
        self.setMinimumSize(1100, 680)

        self._view = QWebEngineView(self)
        self.setCentralWidget(self._view)

        # Bridge + channel: registered as ``jarvis`` on the JS side.
        self.bridge = DashboardBridge(submit_fn=submit_fn, parent=self)
        self._channel = QWebChannel(self)
        self._channel.registerObject("jarvis", self.bridge)
        self._view.page().setWebChannel(self._channel)

        index = _dashboard_index()
        if index.exists():
            self._view.load(QUrl.fromLocalFile(str(index)))
        else:
            debug_log(f"dashboard index.html missing at {index}", "desktop")

    @property
    def view(self) -> QWebEngineView:
        return self._view

    # ── host (tray) hooks ──────────────────────────────────────────────
    def set_submit_fn(self, fn: Optional[Callable[[str], None]]) -> None:
        self.bridge.set_submit_fn(fn)

    def deliver_reply(self, text: Optional[str]) -> None:
        self.bridge.deliver_reply(text)

    def deliver_busy(self) -> None:
        self.bridge.deliver_busy()
