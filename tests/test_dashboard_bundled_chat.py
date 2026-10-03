"""The dashboard's chat in the bundled (single process) desktop app.

The packaged build runs the daemon in a thread of the desktop process, so
there is no stdin to write a query to and no stdout to read a reply from.
The dashboard still has to reach the daemon: a message typed on the page
must come back as the stage labels, the streamed tokens and the final
reply, exactly as it does when the daemon is a subprocess.

These tests drive the real tray, the real window and bridge, and the real
``jarvis.daemon.submit_text_query``. Only what cannot exist headless is
replaced: the web view (a stand-in widget), the thread that would run the
daemon's main loop (inert), and the reply engine (scripted). What they
read is what the page would have been shown.

The dashboard is a web view, which a frozen macOS build cannot show; the
last class covers how the app starts there.
"""

from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import MagicMock

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Qt refuses this import once a QApplication exists, and the window module
# the tests build reaches for it.
pytest.importorskip("PyQt6.QtWebEngineWidgets")

pytestmark = pytest.mark.unit


# ── Stand-ins for what cannot exist in a headless run ─────────────────


class _InertThread:
    """Takes the place of the QThread that would run the daemon's main loop.

    The daemon is "running" as far as the tray can tell; the engine behind
    it is initialised by the ``engine`` fixture below.
    """

    class _Signal:
        def connect(self, _slot):
            pass

    def __init__(self):
        self.finished = self._Signal()

    def start(self):
        pass

    def wait(self, _timeout_ms=None):
        return True

    def isFinished(self):
        return True


class _ScriptedEngine:
    """The reply engine, announcing its phases and streaming a fixed reply."""

    def __init__(self):
        self.asked = []
        self.stages = [("routing", None), ("tool", "weatherTool")]
        self.tokens = ["Bon", "jour"]
        self.reply = "Bonjour !"

    def __call__(
        self, *args, text="", on_token=None, on_stage=None,
        dialogue_memory=None, **kwargs,
    ):
        self.asked.append(text)
        for stage_id, detail in self.stages:
            on_stage(stage_id, detail)
        for chunk in self.tokens:
            on_token(chunk)
        # The real engine stores the turn, which is what the voice path and
        # a chat window opened later read back.
        dialogue_memory.add_interaction(text, self.reply)
        return self.reply


class _Page:
    """What the page is shown: one list per bridge signal it listens to."""

    def __init__(self, bridge):
        self.echoes, self.stages, self.tokens = [], [], []
        self.replies, self.busies, self.statuses = [], [], []
        bridge.userEcho.connect(self.echoes.append)
        bridge.stageChanged.connect(self.stages.append)
        bridge.tokenReceived.connect(self.tokens.append)
        bridge.replyReceived.connect(self.replies.append)
        bridge.busy.connect(lambda: self.busies.append(True))
        bridge.stateChanged.connect(
            lambda payload: self.statuses.append(json.loads(payload)["status"])
        )


def _drain(qapp, until, timeout=5.0):
    """Run the Qt event loop until ``until()`` holds: the daemon answers from
    a worker thread, and its events are queued onto this one."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        qapp.processEvents()
        if until():
            return True
        time.sleep(0.01)
    qapp.processEvents()
    return until()


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def engine(monkeypatch, qapp):
    """The daemon, initialised as a running one is, answering with a script."""
    from jarvis import daemon
    from jarvis.memory.conversation import DialogueMemory

    scripted = _ScriptedEngine()
    monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", scripted)
    monkeypatch.setattr(
        daemon, "_global_dialogue_memory",
        DialogueMemory(inactivity_timeout=300, max_interactions=20),
    )
    monkeypatch.setattr(daemon, "_global_cfg", object())
    monkeypatch.setattr(daemon, "_global_db", object())
    monkeypatch.setattr(daemon, "_global_stop_requested", False)
    monkeypatch.setattr(daemon, "_global_skip_shutdown_diary_update", False)
    # start_daemon wires the confirmation callbacks into this table.
    monkeypatch.setattr(daemon, "_confirmation_callbacks", dict(daemon._confirmation_callbacks))
    yield scripted
    # The worker answers before it releases the one-query lock.
    _drain(qapp, lambda: not daemon._chat_query_lock.locked(), timeout=2.0)


@pytest.fixture
def tray(qapp, monkeypatch):
    """A bundled-mode tray with the parts that need a desktop stubbed.

    The chat path itself runs for real: the submit hook, the signals that
    carry the daemon's answers onto the main thread (wired by the tray's own
    method), ``show_dashboard``, ``show_chat``, ``start_daemon`` and
    ``stop_daemon``. The tray's ``__init__`` is the one part not run, since
    it kills stray daemon processes and builds a dozen windows; the call to
    that wiring method from it is therefore not covered here.
    """
    from PyQt6.QtWidgets import QWidget

    import desktop_app.app as app_mod
    from desktop_app.face_widget import JarvisState, get_jarvis_state

    class _StandInPage:
        def setWebChannel(self, _channel):
            pass

    class _StandInView(QWidget):
        """QWebEngineView cannot be constructed in a headless run."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self._page = _StandInPage()

        def page(self):
            return self._page

        def load(self, _url):
            pass

    monkeypatch.setattr("desktop_app.dashboard_window.QWebEngineView", _StandInView)
    monkeypatch.setattr(app_mod, "QThread", _InertThread)
    monkeypatch.setattr("desktop_app.chat_window._decision_writer", None)
    get_jarvis_state().set_state(JarvisState.IDLE)

    t = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
    t.is_bundled = True
    t.is_listening = False
    t.daemon_process = None
    t.daemon_thread = None
    t._daemon_stop_expected = False
    t.chat_window = None
    t.dashboard_window = None
    t._chat_submit_fn = None
    t._chat_cancel_fn = None
    t._chat_control_fn = None
    t._dashboard_submit_fn = None
    t.log_reader_threads = []
    t.log_signals = MagicMock()
    t.tray_icon = MagicMock()
    t.toggle_action = MagicMock()
    t.quick_stop_action = MagicMock()
    t.status_action = MagicMock()
    t._confirm_signals = MagicMock()
    t.update_icon = lambda: None
    t._connect_dictation_history = lambda *args, **kwargs: None
    # The tray's own wiring of the signals that carry the daemon's events
    # onto the main thread, not a copy of it: dropping a connection there
    # silences the page.
    t._wire_chat_signals()
    yield t
    for window in (t.dashboard_window, t.chat_window):
        if window is not None:
            window.close()
    qapp.processEvents()


def _bridge(tray):
    return tray.dashboard_window.bridge


# ── The dashboard reaches the daemon ──────────────────────────────────


class TestDashboardChatInBundledMode:

    def test_a_query_reaches_the_daemon_and_streams_back(self, qapp, tray, engine):
        from desktop_app.dashboard.bridge import _STATE_VIEW, stage_label

        tray.show_dashboard()   # the dashboard opens at launch, before any daemon
        tray.start_daemon()
        page = _Page(_bridge(tray))

        _bridge(tray).submitQuery("what is the weather")
        assert _drain(qapp, lambda: page.replies), "no reply ever reached the page"

        assert engine.asked == ["what is the weather"]
        assert page.echoes == ["what is the weather"]
        assert page.stages == [stage_label("routing"), stage_label("tool", "weatherTool")]
        assert page.tokens == engine.tokens
        assert page.replies == [engine.reply]
        # The orb is released from its thinking state once the reply lands.
        assert page.statuses[-1] != _STATE_VIEW["thinking"][1]

    def test_the_bus_carries_the_redacted_query_and_never_the_raw_one(
        self, qapp, tray, engine,
    ):
        from desktop_app.dashboard.bridge import parse_chat_ipc_line

        lines = []
        tray._dashboard_chat_signals.line_received.connect(lines.append)
        tray.show_dashboard()
        tray.start_daemon()
        page = _Page(_bridge(tray))

        _bridge(tray).submitQuery("write to bob@example.com about the invoice")
        assert _drain(qapp, lambda: page.replies)

        kinds = [parse_chat_ipc_line(line)[0] for line in lines]
        assert kinds[0] == "start" and kinds[-1] == "complete"
        assert {"stage", "token"} <= set(kinds)
        assert not any("bob@example.com" in line for line in lines)

    def test_a_busy_daemon_tells_the_page_instead_of_queueing(self, qapp, tray, engine):
        from jarvis import daemon
        from desktop_app.dashboard.bridge import _STATE_VIEW

        tray.show_dashboard()
        tray.start_daemon()
        page = _Page(_bridge(tray))

        assert daemon._chat_query_lock.acquire(blocking=False)   # a query is running
        try:
            _bridge(tray).submitQuery("a second question")
            assert _drain(qapp, lambda: page.busies), "the page was never told"
        finally:
            daemon._chat_query_lock.release()

        assert engine.asked == []
        assert page.replies == []
        assert page.statuses[-1] != _STATE_VIEW["thinking"][1]

    def test_the_chat_window_still_answers_in_bundled_mode(self, qapp, tray, engine):
        tray.show_dashboard()
        tray.start_daemon()
        tray.show_chat()

        tray.chat_window.input_widget.setPlainText("typed in the chat window")
        tray.chat_window._send()
        assert _drain(qapp, lambda: engine.reply in tray.chat_window.transcript_text())

        assert engine.asked == ["typed in the chat window"]

    def test_a_chat_window_opened_later_shows_a_dashboard_exchange_once(
        self, qapp, tray, engine,
    ):
        """The window seeds itself from the daemon's memory on its first show.
        A reply it had also been handed as an event would appear twice."""
        tray.show_dashboard()
        tray.start_daemon()
        page = _Page(_bridge(tray))
        _bridge(tray).submitQuery("asked on the dashboard")
        assert _drain(qapp, lambda: page.replies)

        tray.show_chat()
        qapp.processEvents()

        transcript = tray.chat_window.transcript_text()
        assert transcript.count(engine.reply) == 1
        assert transcript.count("asked on the dashboard") == 1


# ── ...and says so when there is no daemon to reach ───────────────────


class TestDashboardChatWithoutADaemon:

    def _preview_answer(self, qapp):
        """What a bridge with nothing behind it says."""
        from desktop_app.dashboard.bridge import DashboardBridge

        standalone = DashboardBridge(submit_fn=None)
        said = []
        standalone.replyReceived.connect(said.append)
        standalone.submitQuery("hello")
        return said[0]

    def test_before_the_daemon_starts_the_preview_answer_is_given(
        self, qapp, tray, engine,
    ):
        tray.show_dashboard()
        page = _Page(_bridge(tray))

        _bridge(tray).submitQuery("hello")

        assert page.replies == [self._preview_answer(qapp)]
        assert engine.asked == []

    @pytest.mark.parametrize("ending", ["crashes", "is stopped"])
    def test_once_the_daemon_is_gone_the_preview_answer_is_given_again(
        self, qapp, tray, engine, ending,
    ):
        tray.show_dashboard()
        tray.start_daemon()
        if ending == "crashes":
            tray._on_daemon_finished()
        else:
            tray.stop_daemon(show_diary_dialog=False, skip_diary_update=True)
        page = _Page(_bridge(tray))

        _bridge(tray).submitQuery("hello")

        assert page.replies == [self._preview_answer(qapp)]
        assert engine.asked == []

    def test_a_dashboard_opened_after_the_daemon_started_is_wired(self, qapp, tray, engine):
        tray.start_daemon()
        tray.show_dashboard()
        page = _Page(_bridge(tray))

        _bridge(tray).submitQuery("hello")

        assert _drain(qapp, lambda: page.replies)
        assert engine.asked == ["hello"]


# ── Subprocess mode is untouched ──────────────────────────────────────


class TestDashboardChatInSubprocessMode:

    def test_the_dashboard_writes_its_query_to_the_daemons_stdin(
        self, qapp, tray, monkeypatch,
    ):
        import io

        import desktop_app.app as app_mod
        from jarvis.daemon import CHAT_QUERY_IPC_PREFIX

        class _Eof:
            def readline(self):
                return ""

        stdin = io.StringIO()
        process = MagicMock()
        process.stdin = stdin
        process.stdout = _Eof()
        process.poll.return_value = None
        monkeypatch.setattr(app_mod.subprocess, "Popen", lambda *args, **kwargs: process)
        tray.is_bundled = False

        tray.show_dashboard()
        tray.start_daemon()
        _bridge(tray).submitQuery("over stdin")

        (line,) = stdin.getvalue().splitlines()
        assert line.startswith(CHAT_QUERY_IPC_PREFIX)
        assert json.loads(line[len(CHAT_QUERY_IPC_PREFIX):]) == {"text": "over stdin"}


# ── A frozen macOS build cannot show a web view ───────────────────────


class _Orb:
    """The floating orb's stand-in: all the tray asks of it is to be shown."""

    def __init__(self):
        self.visible = False

    def show_orb(self):
        self.visible = True


# The builds that cannot show a web view are the frozen macOS ones, and only
# those. Each is a (sys.platform, sys.frozen) pair.
_MACOS_BUNDLE = ("darwin", True)
_BUILDS_THAT_CAN_SHOW_ONE = [
    ("win32", True), ("win32", False), ("linux", True), ("darwin", False),
]


@pytest.fixture
def build(monkeypatch):
    """Make the process look like a given build of the app."""

    def _as(platform, frozen):
        monkeypatch.setattr(sys, "platform", platform)
        monkeypatch.setattr(sys, "frozen", frozen, raising=False)

    return _as


class TestFrozenMacOSBuild:
    """QtWebEngine crashes the app when a view is shown inside a frozen
    macOS bundle, which is why the memory viewer opens the system browser
    there. The dashboard is a web view too, and it is shown at launch."""

    def test_no_web_view_is_built_there(self, qapp, tray, build, monkeypatch):
        class _MustNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError("a QWebEngineView was built in a frozen macOS bundle")

        monkeypatch.setattr("desktop_app.dashboard_window.QWebEngineView", _MustNotBeBuilt)
        build(*_MACOS_BUNDLE)

        tray.show_dashboard()

        assert tray.dashboard_window is None

    @pytest.mark.parametrize("platform, frozen", _BUILDS_THAT_CAN_SHOW_ONE)
    def test_everywhere_else_the_dashboard_opens(self, qapp, tray, build, platform, frozen):
        build(platform, frozen)

        tray.show_dashboard()

        assert tray.dashboard_window is not None
        assert tray.dashboard_window.isVisible()

    @pytest.mark.parametrize(
        "platform, frozen, has_webengine, opens",
        [(*b, True, "dashboard") for b in _BUILDS_THAT_CAN_SHOW_ONE]
        + [(*_MACOS_BUNDLE, True, "orb"), ("win32", True, False, "orb")],
    )
    def test_launch_opens_the_dashboard_or_else_the_floating_orb(
        self, qapp, tray, build, monkeypatch, platform, frozen, has_webengine, opens,
    ):
        import desktop_app.app as app_mod

        monkeypatch.setattr(app_mod, "HAS_WEBENGINE", has_webengine)
        build(platform, frozen)
        tray.orb_window = _Orb()

        tray._show_primary_window()

        dashboard_opened = tray.dashboard_window is not None and tray.dashboard_window.isVisible()
        assert (dashboard_opened, tray.orb_window.visible) == (opens == "dashboard", opens == "orb")

    @pytest.mark.parametrize(
        "platform, frozen, offered",
        [(*b, True) for b in _BUILDS_THAT_CAN_SHOW_ONE] + [(*_MACOS_BUNDLE, False)],
    )
    def test_the_tray_menu_offers_the_dashboard_only_where_it_can_open(
        self, qapp, tray, build, monkeypatch, platform, frozen, offered,
    ):
        import desktop_app.app as app_mod

        monkeypatch.setattr(app_mod, "HAS_WEBENGINE", True)
        build(platform, frozen)
        tray.orb_window = None
        tray._pending_confirmation = None

        tray.create_menu()

        assert (getattr(tray, "dashboard_action", None) in tray.menu.actions()) is offered
