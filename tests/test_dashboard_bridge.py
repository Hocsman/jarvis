"""Tests for the dashboard Python<->JS bridge.

The bridge exposes chat (submitQuery + reply/echo/busy signals), live
system stats, and a JarvisState->orb mapping to the HUD page. These
tests exercise the Python side directly (no web view needed).
"""

from __future__ import annotations

import json
import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def _qapp():
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def _make_bridge(submit_fn=None):
    from desktop_app.dashboard.bridge import DashboardBridge
    return DashboardBridge(submit_fn=submit_fn)


class TestChatRoundTrip:

    @pytest.mark.unit
    def test_submit_forwards_and_echoes(self, _qapp) -> None:
        sent = {}
        b = _make_bridge(submit_fn=lambda t: sent.setdefault("q", t))
        echoes = []
        b.userEcho.connect(echoes.append)

        b.submitQuery("quelle météo à Paris")
        assert sent["q"] == "quelle météo à Paris"
        assert echoes == ["quelle météo à Paris"]
        assert b._chat_pending is True  # orb held at THINKING

    @pytest.mark.unit
    def test_reply_clears_pending(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        replies = []
        b.replyReceived.connect(replies.append)

        b.submitQuery("x")
        assert b._chat_pending is True
        b.deliver_reply("réponse")
        assert replies == ["réponse"]
        assert b._chat_pending is False

    @pytest.mark.unit
    def test_busy_clears_pending(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        busies = []
        b.busy.connect(lambda: busies.append(True))
        b.submitQuery("x")
        b.deliver_busy()
        assert busies == [True]
        assert b._chat_pending is False

    @pytest.mark.unit
    def test_empty_query_ignored(self, _qapp) -> None:
        sent = []
        b = _make_bridge(submit_fn=lambda t: sent.append(t))
        b.submitQuery("   ")
        assert sent == []

    @pytest.mark.unit
    def test_no_submit_fn_preview(self, _qapp) -> None:
        """Standalone (no daemon): submit echoes + returns a preview
        line instead of forwarding."""
        b = _make_bridge(submit_fn=None)
        replies = []
        b.replyReceived.connect(replies.append)
        b.submitQuery("hello")
        assert replies and "aperçu" in replies[0]
        assert b._chat_pending is False  # nothing pending without a daemon


class TestStateMapping:

    @pytest.mark.unit
    def test_pending_forces_thinking(self, _qapp) -> None:
        b = _make_bridge(submit_fn=lambda t: None)
        states = []
        b.stateChanged.connect(states.append)
        b.submitQuery("x")
        # last emitted state during pending must be THINKING (amber)
        last = json.loads(states[-1])
        assert last["status"] == "Réflexion…"

    @pytest.mark.unit
    def test_voice_state_maps_to_accent(self, _qapp) -> None:
        from jarvis.state import get_jarvis_state, JarvisState
        b = _make_bridge()
        states = []
        b.stateChanged.connect(states.append)

        get_jarvis_state().set_state(JarvisState.LISTENING)
        b._emit_state(force=True)
        listening = json.loads(states[-1])
        assert listening["status"] == "Je vous écoute…"
        assert isinstance(listening["accent"], list) and len(listening["accent"]) == 3


class TestStats:

    @pytest.mark.unit
    def test_stats_payload_shape(self, _qapp) -> None:
        b = _make_bridge()
        stats = []
        b.statsUpdated.connect(stats.append)
        b._emit_stats()
        assert stats, "stats should emit"
        o = json.loads(stats[-1])
        for k in ("cpu", "ram", "disk_used", "disk_total", "disk_pct"):
            assert k in o
            assert isinstance(o[k], (int, float))


# Note: DashboardWindow.process_ipc_line is a thin parser that delegates
# to bridge.deliver_reply / deliver_busy (both covered above). It isn't
# unit-tested here: the window builds a QWebEngineView, which is not built
# in the suite's process (Chromium aborts a process whose QApplication has
# no program name, which the suite's shared one may lack); a web view is
# built in a process of its own, as test_memory_viewer_navigation.py does.
# The parse + delegate path is verified manually via the standalone
# round-trip and exercised live in the app.


class TestTheGearOpensTheSettings:
    """The gear in the page's top bar opens the settings window, as the
    tray's action does (``desktop_app.spec.md``, "Dashboard")."""

    @pytest.mark.unit
    def test_the_settings_open_once_the_pages_call_has_returned(self, _qapp) -> None:
        from desktop_app.dashboard.bridge import DashboardBridge

        opened = []
        b = DashboardBridge(open_settings=lambda: opened.append(True))

        b.openSettings()
        assert opened == []      # a modal window is not opened inside the page's call
        _qapp.processEvents()

        assert opened == [True]

    @pytest.mark.unit
    def test_with_no_host_to_open_them_the_gear_does_nothing(self, _qapp) -> None:
        b = _make_bridge()

        b.openSettings()
        _qapp.processEvents()

    @pytest.mark.unit
    def test_the_page_can_reach_it(self) -> None:
        """QWebChannel publishes slots only: a plain method would leave the
        page's call, and the gear, doing nothing."""
        from desktop_app.dashboard.bridge import DashboardBridge

        assert DashboardBridge.staticMetaObject.indexOfSlot("openSettings()") >= 0

    @pytest.mark.unit
    def test_the_gear_in_the_page_asks_for_them(self) -> None:
        from pathlib import Path

        page = (Path(__file__).resolve().parent.parent / "src" / "desktop_app"
                / "dashboard" / "index.html").read_text(encoding="utf-8")

        assert 'id="settings-gear"' in page
        assert "openSettings" in page
