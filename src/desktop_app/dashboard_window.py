"""HUD dashboard window: a single full-screen interface hosting the
J.A.R.V.I.S dashboard rendered as a local web page.

Milestone 1 ships the visual shell only (static / mock data driven by
the page's own JS). Later milestones add the Python<->JS bridge
(``QWebChannel``) that streams live chat, JarvisState, system stats,
and weather into the page, replacing the mock values.

The page itself lives next to this module under ``dashboard/index.html``
so the markup/CSS/JS can be edited without touching Python.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView

from jarvis.debug import debug_log


def _dashboard_index() -> Path:
    return Path(__file__).resolve().parent / "dashboard" / "index.html"


class DashboardWindow(QMainWindow):
    """Frameless-friendly main window that renders the HUD dashboard.

    Public API mirrors the other windows so the tray can manage it the
    same way: ``show()`` / ``hide()``. The web view loads the local
    ``index.html`` on construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S")
        self.setMinimumSize(1100, 680)

        self._view = QWebEngineView(self)
        self.setCentralWidget(self._view)

        index = _dashboard_index()
        if index.exists():
            self._view.load(QUrl.fromLocalFile(str(index)))
        else:
            debug_log(f"dashboard index.html missing at {index}", "desktop")

    @property
    def view(self) -> QWebEngineView:
        return self._view
