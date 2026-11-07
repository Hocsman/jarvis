"""Tests for the dashboard Python<->JS bridge and the page it drives.

The bridge exposes chat (submitQuery + reply/echo/busy signals), live
system stats, the weather, the settings and a JarvisState->orb mapping to
the HUD page. These tests exercise the Python side directly, and read the
page's markup, stylesheet and script as text to check how its controls
are wired; no web view is built.
"""

from __future__ import annotations

import json
import os
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_PAGE = Path(__file__).resolve().parent.parent / "src" / "desktop_app" / "dashboard" / "index.html"


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
# unit-tested here because instantiating QWebEngineView under headless
# pytest is unstable; the parse + delegate path is verified manually via
# the standalone round-trip and exercised live in the app.


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
        page = _PAGE.read_text(encoding="utf-8")

        assert 'id="settings-gear"' in page
        assert "openSettings" in page


def _page_style() -> str:
    style = re.search(r"<style>(.*?)</style>", _PAGE.read_text(encoding="utf-8"), re.S).group(1)
    return re.sub(r"/\*.*?\*/", "", style, flags=re.S)


def _page_script() -> str:
    """The page's inline script without its comments, so a line left only
    in a comment is not read as code. ``//`` after a colon or a quote is
    kept, as in a URL."""
    script = re.search(r"<script>(.*?)</script>", _PAGE.read_text(encoding="utf-8"), re.S).group(1)
    script = re.sub(r"/\*.*?\*/", "", script, flags=re.S)
    return re.sub(r"(?<![:\"'])//[^\n]*", "", script)


def _page_elements() -> list:
    """Every tag the page's markup opens, as ``(tag, attributes)``."""
    class Collect(HTMLParser):
        def __init__(self):
            super().__init__()
            self.found = []

        def handle_starttag(self, tag, attrs):
            self.found.append((tag, dict(attrs)))

    parser = Collect()
    parser.feed(_PAGE.read_text(encoding="utf-8"))
    return parser.found


def _pointer_classes() -> set:
    """The classes the stylesheet gives a pointing hand: what looks clickable.

    Reads rules whose selector is one bare class (``.ctl``), the only shape
    the page gives ``cursor: pointer``; a compound or descendant selector
    would not be seen."""
    classes = set()
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", _page_style()):
        if re.search(r"cursor:\s*pointer", body):
            for selector in selectors.split(","):
                m = re.fullmatch(r"\s*\.([\w-]+)\s*", selector)
                if m:
                    classes.add(m.group(1))
    return classes


def _clickable_elements() -> list:
    pointer = _pointer_classes()
    return [(tag, attrs) for tag, attrs in _page_elements()
            if pointer & set((attrs.get("class") or "").split())]


def _ids_the_script_writes(member: str) -> set:
    """Ids whose element the script writes ``member`` on, directly or
    through the ``const`` it holds the element in. The page reaches its
    elements with ``getElementById("…")`` in double quotes, and that is
    the only form read here."""
    script = _page_script()
    ids = set(re.findall(r'getElementById\("([\w-]+)"\)' + member, script))
    for var, element_id in re.findall(r'const\s+(\w+)\s*=\s*document\.getElementById\("([\w-]+)"\)', script):
        if re.search(rf"\b{var}" + member, script):
            ids.add(element_id)
    return ids


def _ids_the_script_binds() -> set:
    """Ids whose element the script gives a click handler."""
    return _ids_the_script_writes(r'\s*\.\s*(?:onclick\s*=|addEventListener\(\s*"click")')


def _ids_the_script_can_enable() -> set:
    """Ids whose ``disabled`` the script sets at run time, so a button
    disabled in the markup may still be clicked."""
    return _ids_the_script_writes(r"\s*\.\s*disabled\s*=")


def _js_function_body(script: str, name: str) -> str:
    start = script.index("{", script.index(f"function {name}("))
    depth = 0
    for i in range(start, len(script)):
        depth += {"{": 1, "}": -1}.get(script[i], 0)
        if depth == 0:
            return script[start + 1:i]
    raise AssertionError(f"function {name} never closes")


def _wait_for(condition, app, seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


class TestNothingOnThePageLooksClickableAndDoesNothing:
    """Every control on the page acts or is disabled in the markup
    (``desktop_app.spec.md``, "Dashboard")."""

    @pytest.mark.unit
    def test_everything_that_looks_clickable_is_a_button(self) -> None:
        clickable = _clickable_elements()
        assert clickable, "the stylesheet should give some controls a pointer"

        assert [(tag, attrs) for tag, attrs in clickable if tag != "button"] == []

    @pytest.mark.unit
    def test_every_button_acts_or_is_disabled(self) -> None:
        """Only a button disabled in the markup and never enabled by the
        script may have no handler."""
        buttons = [attrs for tag, attrs in _page_elements() if tag == "button"]
        bound = _ids_the_script_binds()
        can_enable = _ids_the_script_can_enable()
        assert buttons

        dead = [attrs for attrs in buttons
                if attrs.get("id") not in bound
                and ("disabled" not in attrs or attrs.get("id") in can_enable)]

        assert dead == []

    @pytest.mark.unit
    def test_a_disabled_button_does_not_look_clickable(self) -> None:
        assert re.search(r"button:disabled[^{]*\{[^}]*cursor:\s*(?:default|not-allowed)", _page_style())

    @pytest.mark.unit
    def test_every_member_of_the_bridge_the_page_touches_exists(self, _qapp) -> None:
        from PyQt6.QtCore import QMetaMethod

        meta = _make_bridge().metaObject()
        methods = {bytes(meta.method(i).name()).decode(): meta.method(i).methodType()
                   for i in range(meta.methodCount())}
        properties = {meta.property(i).name() for i in range(meta.propertyCount())}
        script = _page_script()
        called = set(re.findall(r"\bjarvis\.(\w+)\s*\(", script))
        touched = set(re.findall(r"\bjarvis\.(\w+)", script))
        assert called and touched

        assert {name for name in called if methods.get(name) != QMetaMethod.MethodType.Slot} == set()
        assert touched - set(methods) - properties == set()


class TestTheRefreshButtons:
    """The ⟳ on the stats and weather cards fetch now rather than at the
    next tick (``desktop_app.spec.md``, "Dashboard")."""

    @pytest.mark.unit
    def test_refreshing_the_stats_pushes_a_reading_at_once(self, _qapp) -> None:
        b = _make_bridge()
        stats = []
        b.statsUpdated.connect(stats.append)

        b.refreshStats()

        assert stats, "a reading should come without waiting for the timer"
        assert "cpu" in json.loads(stats[-1])

    @pytest.mark.unit
    def test_refreshing_the_weather_fetches_it_again(self, _qapp, monkeypatch) -> None:
        from desktop_app.dashboard import bridge as bridge_module

        monkeypatch.setattr(bridge_module, "fetch_weather_summary",
                            lambda city, language="en": {"temp": 12, "loc": city})
        b = bridge_module.DashboardBridge(
            weather_city="Lyon", cfg=SimpleNamespace(location_enabled=True, response_language="fr"))
        weather = []
        b.weatherUpdated.connect(weather.append)

        b.refreshWeather()
        _wait_for(lambda: weather, _qapp)

        assert weather and json.loads(weather[-1])["loc"] == "Lyon"

    @pytest.mark.unit
    def test_a_refresh_while_the_weather_is_on_its_way_fetches_nothing_more(self, _qapp, monkeypatch) -> None:
        import threading

        from desktop_app.dashboard import bridge as bridge_module

        asked, release = [], threading.Event()

        def slow_fetch(city, language="en"):
            asked.append(city)
            release.wait(2)
            return {"temp": 12, "loc": city}

        monkeypatch.setattr(bridge_module, "fetch_weather_summary", slow_fetch)
        b = bridge_module.DashboardBridge(weather_city="Lyon", cfg=SimpleNamespace(location_enabled=True))
        weather = []
        b.weatherUpdated.connect(weather.append)

        b.refreshWeather()
        b.refreshWeather()
        release.set()
        _wait_for(lambda: weather, _qapp)

        assert asked == ["Lyon"]

    @pytest.mark.unit
    def test_once_the_weather_has_come_a_refresh_fetches_it_again(self, _qapp, monkeypatch) -> None:
        from desktop_app.dashboard import bridge as bridge_module

        asked = []
        monkeypatch.setattr(bridge_module, "fetch_weather_summary",
                            lambda city, language="en": asked.append(city) or {"loc": city})
        b = bridge_module.DashboardBridge(weather_city="Lyon", cfg=SimpleNamespace(location_enabled=True))
        weather = []
        b.weatherUpdated.connect(weather.append)

        b.refreshWeather()
        _wait_for(lambda: weather, _qapp)
        b.refreshWeather()
        _wait_for(lambda: len(weather) == 2, _qapp)

        assert asked == ["Lyon", "Lyon"]

    @pytest.mark.unit
    def test_a_failed_fetch_does_not_block_the_next_one(self, _qapp, monkeypatch) -> None:
        from desktop_app.dashboard import bridge as bridge_module

        asked = []

        def failing_fetch(city, language="en"):
            asked.append(city)
            raise OSError("network down")

        monkeypatch.setattr(bridge_module, "fetch_weather_summary", failing_fetch)
        b = bridge_module.DashboardBridge(weather_city="Lyon", cfg=SimpleNamespace(location_enabled=True))

        b.refreshWeather()
        _wait_for(lambda: asked, _qapp)
        # Refresh until one gets through: the failed fetch must not hold the next one back.
        _wait_for(lambda: b.refreshWeather() or len(asked) >= 2, _qapp)

        assert asked[:2] == ["Lyon", "Lyon"]

    @pytest.mark.unit
    def test_with_location_off_refreshing_the_weather_fetches_nothing(self, _qapp, monkeypatch) -> None:
        from desktop_app.dashboard import bridge as bridge_module

        asked = []
        monkeypatch.setattr(bridge_module, "fetch_weather_summary",
                            lambda city, language="en": asked.append(city))
        b = bridge_module.DashboardBridge(
            weather_city="Lyon", cfg=SimpleNamespace(location_enabled=False))

        b.refreshWeather()
        _wait_for(lambda: asked, _qapp, seconds=0.2)

        assert asked == []

    @pytest.mark.unit
    def test_the_page_can_read_whether_there_is_weather_to_fetch(self, _qapp) -> None:
        from desktop_app.dashboard.bridge import DashboardBridge

        on = DashboardBridge(weather_city="Lyon", cfg=SimpleNamespace(location_enabled=True))
        off = DashboardBridge(weather_city="Lyon", cfg=SimpleNamespace(location_enabled=False))

        assert on.property("weatherAvailable") is True
        assert off.property("weatherAvailable") is False

    @pytest.mark.unit
    def test_the_weather_refresh_waits_for_the_bridge_to_say_there_is_weather(self) -> None:
        buttons = {attrs.get("id"): attrs for tag, attrs in _page_elements() if tag == "button"}

        assert "disabled" in buttons["weather-refresh"]
        assert "jarvis.weatherAvailable" in _page_script()


class TestClearingTheTranscript:
    """🗑️ Effacer empties the page and nothing else: the dialogue memory is
    the daemon's and voice shares it (``desktop_app.spec.md``, "Dashboard")."""

    @pytest.mark.unit
    def test_the_clear_button_is_wired_to_clearing_the_transcript(self) -> None:
        assert re.search(r'getElementById\("clear-transcript"\)\.onclick\s*=\s*clearTranscript\b',
                         _page_script())

    @pytest.mark.unit
    def test_clearing_asks_nothing_of_the_bridge(self) -> None:
        assert "jarvis" not in _js_function_body(_page_script(), "clearTranscript")
