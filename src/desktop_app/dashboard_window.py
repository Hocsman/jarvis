"""HUD dashboard window: a single full-screen interface hosting the
YUBA dashboard rendered as a local web page.

The page (``dashboard/index.html``) is driven live by a
:class:`DashboardBridge` exposed over ``QWebChannel`` as ``window.jarvis``:
system stats, voice state (orb colour + status), weather, and the chat
round-trip all flow through it. The host (the tray) wires the chat
submission callable and forwards daemon replies via ``deliver_reply``.
The view shows that page and no other, since any page it shows could reach
the bridge (see :class:`_DashboardPage`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel

from jarvis.debug import debug_log
from desktop_app.dashboard.bridge import DashboardBridge, dispatch_chat_ipc_line


def _dashboard_index() -> Path:
    return Path(__file__).resolve().parent / "dashboard" / "index.html"


def _is_own_page(url: QUrl, index: Path) -> bool:
    """Whether the dashboard view may navigate to ``url``.

    ``about:`` carries Qt's internal page states. Anything else must be the
    index file itself; a query or a fragment leaves the document the same.
    The paths are compared as the platform compares them (without regard
    to case on Windows).
    """
    if (url.scheme() or "").lower() == "about":
        return True
    if not url.isLocalFile():
        return False
    shown = os.path.normcase(os.path.normpath(url.toLocalFile()))
    return shown == os.path.normcase(os.path.normpath(str(index)))


class _DashboardPage(QWebEnginePage):
    """WebEngine page that shows the dashboard's own page and no other.

    The page is handed the bridge over ``QWebChannel``, and Qt gives the
    channel's transport to whatever document the view shows: a foreign
    page reached by a link, a redirect or a dropped file could call
    ``submitQuery`` and drive the daemon and its tools. So every
    navigation, in any frame, is refused unless it lands on ``index.html``,
    and a drop never navigates.
    """

    def __init__(self, index: Path, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self.settings().setAttribute(QWebEngineSettings.WebAttribute.NavigateOnDropEnabled, False)

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        if _is_own_page(url, self._index):
            return super().acceptNavigationRequest(url, navigation_type, is_main_frame)
        debug_log(
            f"dashboard navigation refused: {url.scheme()}://{url.host()}",
            "desktop",
        )
        return False


class DashboardWindow(QMainWindow):
    """Main window rendering the HUD dashboard with a live data bridge."""

    def __init__(self, submit_fn: Optional[Callable[[str], None]] = None) -> None:
        super().__init__()
        self.setWindowTitle("YUBA")
        self.setMinimumSize(1100, 680)

        index = _dashboard_index()
        self._view = QWebEngineView(self)
        self._view.setPage(_DashboardPage(index, self._view))
        self.setCentralWidget(self._view)

        # Bridge + channel: registered as ``jarvis`` on the JS side.
        cfg = None
        try:
            from jarvis.config import load_settings
            cfg = load_settings()
        except Exception:
            pass
        self.bridge = DashboardBridge(
            submit_fn=submit_fn, cfg=cfg, parent=self
        )
        self._channel = QWebChannel(self)
        self._channel.registerObject("jarvis", self.bridge)
        self._view.page().setWebChannel(self._channel)

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

    def process_ipc_line(self, line: str) -> bool:
        """Parse a ``__CHAT__:`` daemon event line and drive the bridge.

        Mirrors ``ChatWindow.process_ipc_line`` so the dashboard can be
        fed the same daemon event stream by the tray's log reader.
        Returns True if the line was a chat event.
        """
        return dispatch_chat_ipc_line(line, self.bridge)
