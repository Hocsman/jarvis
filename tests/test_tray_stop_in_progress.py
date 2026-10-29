"""Nothing starts or stops her while a stop is in progress.

Stopping waits for the daemon to write its diary: up to a minute for a
subprocess, three for a thread. The tray keeps the Qt event loop turning
while it waits, so the windows stay alive, and so does every way of asking
for another start or stop: the toggle, the quick stop, the setup wizard,
the restart that Settings offers (from the tray and from the dashboard's
gear), the status timer and the thread's own "finished" signal.

Any of those running inside the wait used to leave the app in a state that
lies. A subprocess restarted from Settings was killed by the first stop's
timeout a minute later, and the app read Stopped. In bundled mode the
first stop forgot the thread the restart had just started: a daemon with
the microphone live while the tray said Stopped.

So one stop runs at a time, it alone decides what the tray shows when it
ends, and while it runs the tray says so.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── Stand-ins for what a desktop and a daemon provide ─────────────────


class _Clock:
    """The module's clock, one second per reading and no real sleeping.

    A stop that loses track of its daemon waits out its whole timeout; on
    this clock that takes a blink instead of a minute.
    """

    def __init__(self):
        self._now = 0.0

    def time(self):
        self._now += 1.0
        return self._now

    def sleep(self, _seconds):
        pass


class _Stdin:
    def __init__(self, process):
        self._process = process

    def write(self, _text):
        self._process.stop_requested()

    def flush(self):
        pass

    def close(self):
        self._process.stop_requested()


class _DaemonProcess:
    """A daemon subprocess that exits when asked to, as the real one does.

    ``on_exit`` callables run once, when it exits.
    """

    def __init__(self):
        self.stdin = _Stdin(self)
        self.stdout = None
        self.pid = 4242
        self.returncode = None
        self.asked_to_stop = 0
        self.killed = False
        self.on_exit = []

    def stop_requested(self):
        self.asked_to_stop += 1
        if self.returncode is None:
            self.returncode = 0
            for callback in self.on_exit:
                callback()

    def poll(self):
        return self.returncode

    def send_signal(self, _sig):
        self.stop_requested()

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self):
        for slot in list(self._slots):
            slot()


class _DaemonThread:
    """Takes the place of the QThread the bundled daemon runs in.

    It ends when the daemon is asked to stop, and Qt then delivers its
    ``finished`` signal on the main thread, as a queued call.
    """

    started: list = []

    def __init__(self, *args, **kwargs):
        self.finished = _Signal()
        self.running = False

    def start(self):
        self.running = True
        _DaemonThread.started.append(self)

    def end(self):
        from PyQt6.QtCore import QTimer

        if self.running:
            self.running = False
            QTimer.singleShot(0, self.finished.emit)

    def isFinished(self):
        return not self.running

    def wait(self, _timeout_ms=None):
        return not self.running


class _ChatWindow:
    """Records the daemon lifecycle the chat window is shown."""

    def __init__(self):
        self.statuses = []

    def set_daemon_hooks(self, **_hooks):
        pass

    def set_daemon_status(self, status):
        self.statuses.append(status)


class _SavedSettings:
    """The settings window, closed with Save."""

    def exec(self):
        from PyQt6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def launched(monkeypatch):
    """Every daemon subprocess the tray launches."""
    import desktop_app.app as app_mod

    processes = []

    def popen(*_args, **_kwargs):
        process = _DaemonProcess()
        processes.append(process)
        return process

    monkeypatch.setattr(app_mod.subprocess, "Popen", popen)
    return processes


@pytest.fixture
def answers(monkeypatch):
    """The questions and notes the tray puts to the user, answered Yes."""
    from PyQt6.QtWidgets import QMessageBox

    asked = {"question": [], "shown": []}

    def question(_parent, title, text, *_args, **_kwargs):
        asked["question"].append((title, text))
        return QMessageBox.StandardButton.Yes

    def shown(box):
        asked["shown"].append((box.windowTitle(), box.text()))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "question", question)
    monkeypatch.setattr(QMessageBox, "exec", shown)
    monkeypatch.setattr("desktop_app.settings_window.SettingsWindow", _SavedSettings)
    return asked


@pytest.fixture
def wizards(monkeypatch):
    """Every setup wizard the tray opens; each is cancelled."""
    from PyQt6.QtWidgets import QWizard

    opened = []

    class _Wizard:
        def __init__(self, *args, **kwargs):
            opened.append(self)

        def exec(self):
            return QWizard.DialogCode.Rejected

    monkeypatch.setattr("desktop_app.setup_wizard.SetupWizard", _Wizard)
    return opened


@pytest.fixture
def tray(qapp, monkeypatch):
    """The tray with real menu entries and a real event loop, and stand-ins
    for the daemon, the diary dialog and the notifications.

    ``tray.logs`` is what reached the log viewer.
    """
    from PyQt6.QtGui import QAction

    import desktop_app.app as app_mod
    from jarvis import daemon

    monkeypatch.setattr(app_mod, "time", _Clock())
    monkeypatch.setattr(app_mod, "DiaryUpdateDialog", MagicMock)
    monkeypatch.setattr(app_mod, "QThread", _DaemonThread)
    monkeypatch.setattr(_DaemonThread, "started", [])
    monkeypatch.setattr("desktop_app.chat_window._decision_writer", None)
    monkeypatch.setattr(daemon, "_confirmation_callbacks", dict(daemon._confirmation_callbacks))
    monkeypatch.setattr(daemon, "set_diary_update_callbacks", lambda **_kwargs: None)

    def request_stop(skip_diary_update=False):
        for thread in _DaemonThread.started:
            thread.end()

    monkeypatch.setattr(daemon, "request_stop", request_stop)

    t = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
    t.app = qapp
    t.is_bundled = False
    t.is_listening = False
    t.daemon_process = None
    t.daemon_thread = None
    t._daemon_stopping = False
    t.chat_window = _ChatWindow()
    t.dashboard_window = None
    t._chat_submit_fn = None
    t._chat_cancel_fn = None
    t._chat_control_fn = None
    t._dashboard_submit_fn = None
    t.log_reader_threads = []
    t.logs = []
    t.log_signals = app_mod.LogSignals()
    t.log_signals.new_log.connect(t.logs.append)
    t.tray_icon = MagicMock()
    t._confirm_signals = MagicMock()
    t.update_icon = lambda: None
    t._connect_dictation_history = lambda *args, **kwargs: None
    t.toggle_action = QAction("▶️ Start Listening")
    t.quick_stop_action = QAction("⚡ Stop Now (Skip Diary)")
    t.quick_stop_action.setEnabled(False)
    t.setup_wizard_action = QAction("🔧 Setup Wizard")
    t.status_action = QAction("⚪ Status: Stopped")
    t._wire_chat_signals()
    yield t
    qapp.processEvents()


def _listening(tray, *, bundled=False):
    """Start her the way the tray does."""
    tray.is_bundled = bundled
    tray.start_daemon()
    assert tray.is_listening
    return tray.daemon_thread if bundled else tray.daemon_process


def _during_the_stop(action):
    """Run ``action`` the way a click would arrive while the stop waits:
    from the event loop the stop keeps turning."""
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(0, action)


def _notifications(tray):
    return [call.args[:2] for call in tray.tray_icon.showMessage.call_args_list]


# ── The tray says a stop is in progress ───────────────────────────────


class TestTheTraySaysSheIsStopping:

    def test_while_the_stop_waits_the_menu_shows_it_and_offers_nothing_that_would_fight_it(self, tray, launched):
        _listening(tray)
        seen = {}

        def look():
            seen["toggle"] = tray.toggle_action.text()
            seen["toggle enabled"] = tray.toggle_action.isEnabled()
            seen["quick stop enabled"] = tray.quick_stop_action.isEnabled()
            seen["wizard enabled"] = tray.setup_wizard_action.isEnabled()
            seen["status"] = tray.status_action.text()

        _during_the_stop(look)
        tray.stop_daemon()

        assert "Stopping" in seen["toggle"]
        assert seen["toggle enabled"] is False
        assert seen["quick stop enabled"] is False
        assert seen["wizard enabled"] is False
        assert "Stopping" in seen["status"]

    def test_once_stopped_the_menu_offers_to_start_her_again(self, tray, launched):
        _listening(tray)

        tray.stop_daemon()

        assert tray.toggle_action.text() == "▶️ Start Listening"
        assert tray.toggle_action.isEnabled()
        assert tray.setup_wizard_action.isEnabled()
        assert not tray.quick_stop_action.isEnabled()
        assert tray.status_action.text() == "⚪ Status: Stopped"

    def test_a_stop_that_fails_does_not_leave_the_menu_stuck_on_stopping(self, tray, launched):
        process = _listening(tray)

        def broken_poll():
            raise OSError("the pipe is gone")

        process.poll = broken_poll
        tray.stop_daemon()

        assert "Stopping" not in tray.toggle_action.text()
        assert tray.toggle_action.isEnabled()
        assert tray.setup_wizard_action.isEnabled()


# ── Settings saved while she stops ────────────────────────────────────


class TestSettingsSavedDuringAStop:
    """Settings opens from the tray and from the dashboard's gear, both
    reachable while the stop waits."""

    def test_saving_does_not_start_a_daemon_the_stop_would_then_kill(self, tray, launched, answers):
        _listening(tray)

        _during_the_stop(tray.show_settings)
        tray.stop_daemon()

        assert len(launched) == 1, "a daemon was started inside the stop"
        assert not any(p.killed for p in launched)
        assert tray.is_listening is False
        assert tray.daemon_process is None

    def test_the_user_is_told_the_settings_apply_at_the_next_start(self, tray, launched, answers):
        _listening(tray)

        _during_the_stop(tray.show_settings)
        tray.stop_daemon()

        assert answers["question"] == [], "a restart was offered while she was stopping"
        assert len(answers["shown"]) == 1

    def test_in_bundled_mode_no_daemon_is_left_running_behind_a_stopped_tray(self, tray, answers):
        _listening(tray, bundled=True)

        _during_the_stop(tray.show_settings)
        tray.stop_daemon()

        assert len(_DaemonThread.started) == 1, "a daemon was started inside the stop"
        assert tray.is_listening is False
        orphans = [
            t for t in _DaemonThread.started
            if t.running and t is not tray.daemon_thread
        ]
        assert orphans == []

    def test_with_no_stop_in_progress_saving_still_restarts_her(self, tray, launched, answers):
        first = _listening(tray)

        tray.show_settings()

        assert len(answers["question"]) == 1
        assert first.asked_to_stop >= 1
        assert len(launched) == 2
        assert tray.is_listening is True
        assert tray.daemon_process is launched[1]


# ── The other ways in ─────────────────────────────────────────────────


class TestOtherWaysInDuringAStop:

    def test_the_toggle_neither_stops_her_twice_nor_starts_her(self, tray, launched):
        _listening(tray)

        _during_the_stop(tray.toggle_listening)
        tray.stop_daemon()

        assert len(launched) == 1
        assert not any("Failed to stop" in line for line in tray.logs)
        assert ("Jarvis Stopped", "Voice assistant is no longer listening") in _notifications(tray)
        assert tray.is_listening is False

    def test_a_quick_stop_sends_the_daemon_nothing_more(self, tray, launched):
        process = _listening(tray)
        asked_before_the_stop = process.asked_to_stop

        _during_the_stop(tray.quick_stop_daemon)
        tray.stop_daemon()
        asked_by_this_stop = process.asked_to_stop - asked_before_the_stop

        fresh = _listening(tray)
        fresh_asks = fresh.asked_to_stop
        tray.stop_daemon()

        # The same number of shutdown requests as a stop on its own.
        assert asked_by_this_stop == fresh.asked_to_stop - fresh_asks
        assert not any("Failed to stop" in line for line in tray.logs)

    def test_a_start_asked_for_by_anything_else_is_refused(self, tray, launched):
        _listening(tray)

        _during_the_stop(tray.start_daemon)
        tray.stop_daemon()

        assert len(launched) == 1
        assert not any(p.killed for p in launched)
        assert tray.is_listening is False

    def test_the_setup_wizard_does_not_open(self, tray, launched, wizards):
        _listening(tray)

        _during_the_stop(tray.show_setup_wizard)
        tray.stop_daemon()

        assert wizards == []
        assert len(launched) == 1
        assert tray.is_listening is False

    def test_once_the_stop_is_over_she_can_be_started_again(self, tray, launched):
        _listening(tray)
        tray.stop_daemon()

        tray.toggle_listening()

        assert tray.is_listening is True
        assert tray.daemon_process is launched[-1]
        assert len(launched) == 2


# ── The status check and the thread's own signal ──────────────────────


class TestTheDaemonEndingDuringItsStop:
    """The daemon ending is what the stop waits for. The status timer and
    the thread's ``finished`` signal both notice it too, from the same
    event loop: they must not call it a crash nor pull the daemon out from
    under the stop."""

    def test_the_status_check_does_not_call_a_requested_stop_a_crash(self, tray, launched):
        process = _listening(tray)
        # The status timer fires between the daemon exiting and the stop
        # noticing it.
        process.on_exit.append(lambda: _during_the_stop(tray.check_daemon_status))

        tray.stop_daemon()

        messages = _notifications(tray)
        assert ("Jarvis Stopped", "Voice assistant is no longer listening") in messages
        assert not any("unexpectedly" in text for _title, text in messages)
        assert not any("Failed to stop" in line for line in tray.logs)
        assert "crashed" not in tray.chat_window.statuses
        assert tray.chat_window.statuses[-1] == "stopped"
        assert tray.is_listening is False

    def test_in_bundled_mode_the_thread_ending_completes_the_stop_cleanly(self, tray):
        from jarvis.state import JarvisState, get_jarvis_state

        _listening(tray, bundled=True)
        get_jarvis_state().set_state(JarvisState.LISTENING)

        tray.stop_daemon()
        tray.app.processEvents()

        messages = _notifications(tray)
        assert ("Jarvis Stopped", "Voice assistant is no longer listening") in messages
        assert not any("unexpectedly" in text for _title, text in messages)
        assert not any("Failed to stop" in line for line in tray.logs)
        assert "crashed" not in tray.chat_window.statuses
        assert tray.chat_window.statuses[-1] == "stopped"
        assert tray.is_listening is False
        assert tray.daemon_thread is None
        # Nothing is listening: the face and the orb must not look ready.
        assert get_jarvis_state().state is JarvisState.ASLEEP
