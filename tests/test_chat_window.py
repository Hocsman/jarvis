"""Behaviour tests for the ChatWindow (text chat interface).

These verify the contract in ``src/desktop_app/chat_window.spec.md``:

- The window has a transcript area, an input box, a send button, and a stop
  button (visible only while a query is in flight).
- Sending submits text via ``jarvis.daemon.submit_text_query`` and appends the
  user's message to the transcript.
- Daemon callback signals (start/complete/busy) update the transcript and the
  status indicator on the Qt main thread.
- The stop button calls ``jarvis.daemon.cancel_active_chat_query``.
- Closing hides the window; it does not quit the daemon.
- Styling uses the shared theme stylesheet (no hardcoded colour literals in
  the widget classes).
"""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestPhoneShell:
    def test_window_controls_and_rounded_frame(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt, QPoint
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QSizeGrip
        monkeypatch.setattr('desktop_app.chat_window.get_hot_window_messages', lambda: [])
        win = ChatWindow()
        win.show()
        QTest.qWait(30)
        rendered = win.grab().toImage()
        assert rendered.pixelColor(0, 0).alpha() == 0
        centre = rendered.rect().center()
        assert rendered.pixelColor(centre).alpha() == 255
        assert win.findChild(QSizeGrip).isVisible()
        QTest.mouseDClick(win.title_bar, Qt.MouseButton.LeftButton, pos=QPoint(50, 10))
        assert win.isMaximized()
        QTest.mouseDClick(win.title_bar, Qt.MouseButton.LeftButton, pos=QPoint(50, 10))
        assert not win.isMaximized()
        win.minimise_button.click()
        assert win.isMinimized()
        win.showNormal()
        win.close()

    def test_custom_frame_close_preserves_conversation(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt
        monkeypatch.setattr('desktop_app.chat_window.get_hot_window_messages', lambda: [])
        win = ChatWindow()
        win.show()
        win._append_assistant('Still here')
        assert win.windowFlags() & Qt.WindowType.FramelessWindowHint
        win.close_button.click()
        assert not win.isVisible()
        win.show()
        assert win.transcript_text() == 'Still here'
        win.close()

    def test_empty_state_disappears_without_becoming_a_message(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        monkeypatch.setattr('desktop_app.chat_window.get_hot_window_messages', lambda: [])
        win = ChatWindow()
        win.show()
        assert win.empty_state.isVisible()
        assert win.transcript_text() == ''
        win._append_user('Hello')
        assert not win.empty_state.isVisible()
        win.close()

    @pytest.mark.parametrize('size', [(380, 560), (480, 780), (800, 650)])
    def test_controls_fit_and_messages_are_literal_at_all_sizes(self, qapp, monkeypatch, size):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QLabel
        from PyQt6.QtTest import QTest
        monkeypatch.setattr('desktop_app.chat_window.get_hot_window_messages', lambda: [])
        win = ChatWindow()
        win.resize(*size)
        win.show()
        win._append_assistant('<b>literal message</b> ' * 20)
        win._set_thinking(True)
        QTest.qWait(50)
        for widget in (win.input_widget, win.send_button, win.stop_button, win.close_button):
            assert win.rect().contains(widget.mapTo(win, widget.rect().bottomRight()))
            assert widget.width() >= 36
        bubble = next(w for w in win.findChildren(QLabel) if w.objectName() == 'bubble')
        assert bubble.textFormat() == Qt.TextFormat.PlainText
        assert bubble.width() < win.transcript_widget.viewport().width()
        win.close()


@pytest.mark.unit
class TestChatWindowStructure:
    """The window exposes the UI elements the spec requires."""

    def test_has_transcript_input_send_and_stop(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert win.transcript_widget is not None
        assert win.input_widget is not None
        assert win.send_button is not None
        assert win.stop_button is not None

    def test_stop_button_hidden_at_rest(self, qapp):
        """The stop button is only relevant while a query is running."""
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        assert not win.stop_button.isVisible()

    def test_window_title_mentions_jarvis(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        title = win.windowTitle()
        assert "Jarvis" in title


@pytest.mark.unit
class TestChatWindowSend:
    """Sending a message dispatches to the daemon and echoes the user text."""

    def test_send_calls_submit_text_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("what is the weather")
        win._send()
        assert calls == ["what is the weather"]

    def test_send_appends_user_message_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hello there")
        win._send()
        text = win.transcript_text()
        assert "hello there" in text

    def test_send_clears_input(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("clear me after send")
        win._send()
        assert win.input_widget.toPlainText() == ""

    def test_send_empty_does_nothing(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("   ")
        win._send()
        assert calls == []

    def test_send_when_daemon_unavailable_does_not_submit(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow(daemon_available=False)
        win.input_widget.setPlainText("are you there")
        win._send()

        assert calls == []
        text = win.transcript_text().lower()
        assert "start listening" in text
        assert win.input_widget.toPlainText() == "are you there"


@pytest.mark.unit
class TestChatWindowCallbacks:
    """Daemon callback signals update the UI on the main thread."""

    def test_on_complete_appends_reply_to_transcript(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        win._send()
        # Simulate the daemon completing with a reply.
        win._on_complete("It is sunny today.")
        text = win.transcript_text()
        assert "It is sunny today." in text

    def test_on_complete_hides_stop_button(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi")
        win._send()
        qapp.processEvents()
        # While "thinking" the stop button should be visible.
        assert win.stop_button.isVisible()
        win._on_complete("done")
        qapp.processEvents()
        assert not win.stop_button.isVisible()

    def test_on_busy_appends_busy_notice(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.input_widget.setPlainText("second query")
        win._send()
        # Simulate the daemon rejecting because a query is already running.
        win._on_busy()
        text = win.transcript_text()
        # The notice is language-neutral in shape but must mention the query
        # was not accepted.
        assert "second query" in text  # user echo stays
        assert "busy" in text.lower() or "already" in text.lower()


@pytest.mark.unit
class TestChatWindowStop:
    """The stop button cancels the chat query, not the whole daemon."""

    def test_stop_calls_cancel_active_chat_query(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        called = []
        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: called.append(True)
        )
        # request_stop must NOT be called because it tears down the whole
        # voice assistant.
        request_stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop",
            lambda: request_stop_called.append(True),
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        win._stop()
        qapp.processEvents()
        assert called == [True]
        assert request_stop_called == []

    def test_stop_resets_thinking_indicator(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()
        assert win.stop_button.isVisible()
        win._stop()
        qapp.processEvents()
        assert not win.stop_button.isVisible()


@pytest.mark.unit
class TestChatWindowLifecycle:
    """Closing hides rather than tearing down daemon state."""

    def test_close_event_hides_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtGui import QCloseEvent

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        # The daemon stop function must NOT be called on close.
        stop_called = []
        monkeypatch.setattr(
            "jarvis.daemon.request_stop", lambda: stop_called.append(True)
        )
        win.closeEvent(QCloseEvent())
        assert stop_called == []


@pytest.mark.unit
class TestChatWindowSubmitFn:
    """When a ``submit_fn`` is injected (subprocess mode), sending routes
    through it instead of the daemon's direct call path."""

    def test_submit_fn_receives_text(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        calls = []
        win = ChatWindow(submit_fn=lambda text: calls.append(text))
        # The bundled path must NOT be touched when submit_fn is set.
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use submit_fn")),
        )
        win.input_widget.setPlainText("via stdin")
        win._send()
        assert calls == ["via stdin"]

    def test_daemon_availability_toggles_input_controls(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()

        win.set_daemon_available(True)
        assert win.send_button.isEnabled()
        assert win.input_widget.isEnabled()

        win.set_daemon_available(False)
        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()


@pytest.mark.unit
class TestDesktopAppChatDispatch:
    """The desktop app routes ``__CHAT__:`` IPC lines to the chat window on the
    main thread via ``_on_chat_ipc_line`` + ``ChatWindow.process_ipc_line``."""

    def _make_tray(self):
        import desktop_app.app as app_mod
        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None
        tray.is_listening = True
        return tray

    def test_on_chat_ipc_line_creates_window_lazily(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        assert tray.chat_window is None
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hi"}}')
        assert tray.chat_window is not None

    def test_dispatch_complete_appends_reply(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"hello back"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert "hello back" in tray.chat_window.transcript_text()

    def test_dispatch_start_sets_thinking(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"start","data":"a query"}}')
        tray.chat_window.show()
        qapp.processEvents()
        assert tray.chat_window.stop_button.isVisible()

    def test_dispatch_busy_appends_notice(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"busy","data":null}}')
        tray.chat_window.show()
        qapp.processEvents()
        text = tray.chat_window.transcript_text().lower()
        assert "busy" in text

    def test_dispatch_malformed_line_is_swallowed(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        # Must not raise; window is created lazily but no reply text lands.
        tray._on_chat_ipc_line(f"{CHAT_IPC_PREFIX}not json")
        qapp.processEvents()

    def test_process_ipc_line_returns_false_for_non_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        win = ChatWindow()
        assert win.process_ipc_line("not a chat line") is False

    def test_process_ipc_line_returns_true_for_malformed_chat(self, qapp):
        from desktop_app.chat_window import ChatWindow
        from jarvis.daemon import CHAT_IPC_PREFIX
        win = ChatWindow()
        assert win.process_ipc_line(f"{CHAT_IPC_PREFIX}not json") is True

    def test_late_ipc_line_creates_unavailable_window_when_daemon_stopped(self, qapp):
        from jarvis.daemon import CHAT_IPC_PREFIX
        tray = self._make_tray()
        tray.is_listening = False

        tray._on_chat_ipc_line(f'{CHAT_IPC_PREFIX}{{"type":"complete","data":"late"}}')

        assert tray.chat_window is not None
        assert not tray.chat_window.send_button.isEnabled()

    def test_show_chat_marks_window_unavailable_when_daemon_stopped(self, qapp):
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None
        tray.is_listening = False

        tray.show_chat()

        assert tray.chat_window is not None
        assert not tray.chat_window.send_button.isEnabled()

    def test_show_chat_marks_existing_window_available_when_daemon_started(self, qapp):
        import desktop_app.app as app_mod
        from desktop_app.chat_window import ChatWindow

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = ChatWindow(daemon_available=False)
        tray._chat_submit_fn = lambda text: None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None
        tray.is_listening = True

        tray.show_chat()

        assert tray.chat_window.send_button.isEnabled()
        assert tray.chat_window._submit_fn is tray._chat_submit_fn

    def test_show_chat_wires_all_three_hooks_as_one(self, qapp):
        """Submit, cancel and control share a lifecycle: a window shown after
        the daemon started routes Stop and rewind through the tray's hooks,
        not through the in-process daemon."""
        import desktop_app.app as app_mod
        from desktop_app.chat_window import ChatWindow

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = ChatWindow(daemon_available=False)
        submits, cancels, controls = [], [], []
        tray._chat_submit_fn = submits.append
        tray._chat_cancel_fn = lambda: cancels.append(True)
        tray._chat_control_fn = lambda kind, payload: controls.append((kind, payload))
        tray.is_listening = True

        tray.show_chat()
        win = tray.chat_window
        win.input_widget.setPlainText("hello")
        win._send()
        win._stop()
        win._on_complete(None)
        win._rewind_to_user(1, "hello")

        assert submits == ["hello"]
        assert cancels == [True]
        assert controls == [("rewind", {"user_index": 1, "content": "hello"})]


@pytest.mark.unit
class TestChatWindowDaemonStatus:
    """The chat window shows daemon lifecycle state without requiring logs."""

    def test_initial_unavailable_state_shows_status_banner(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        win.show()
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Start Listening" in win._status_label.text()

    def test_starting_state_disables_submission_and_shows_progress(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()

        win.set_daemon_status("starting")
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Starting" in win._status_label.text()

    def test_stopping_state_disables_submission_and_shows_progress(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()

        win.set_daemon_status("stopping")
        qapp.processEvents()

        assert not win.send_button.isEnabled()
        assert not win.input_widget.isEnabled()
        assert win._status_label.isVisible()
        assert "Stopping" in win._status_label.text()

    def test_running_state_hides_status_banner_and_reenables_submission(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow(daemon_available=False)
        win.show()
        qapp.processEvents()

        win.set_daemon_status("running")
        qapp.processEvents()

        assert win.send_button.isEnabled()
        assert win.input_widget.isEnabled()
        assert not win._status_label.isVisible()

    def test_crashed_state_resets_thinking_and_explains_reconnect(self, qapp):
        from desktop_app.chat_window import ChatWindow

        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win._set_thinking(True)
        qapp.processEvents()

        win.set_daemon_status("crashed")
        qapp.processEvents()

        assert not win.stop_button.isVisible()
        assert not win.send_button.isEnabled()
        assert win._status_label.isVisible()
        label = win._status_label.text().lower()
        assert "unexpectedly" in label
        assert "start listening" in label


@pytest.mark.unit
class TestDesktopAppChatStatus:
    """The tray forwards daemon lifecycle state to an open chat window."""

    def test_set_chat_daemon_status_updates_existing_window(self, qapp):
        import desktop_app.app as app_mod
        from desktop_app.chat_window import ChatWindow

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = ChatWindow()
        tray.chat_window.show()
        tray._chat_submit_fn = lambda text: None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None

        tray._set_chat_daemon_status("crashed")
        qapp.processEvents()

        assert not tray.chat_window.send_button.isEnabled()
        assert "unexpectedly" in tray.chat_window._status_label.text().lower()


@pytest.mark.unit
class TestChatWindowInputKeys:
    """Enter sends; Shift+Enter inserts a newline (does not send)."""

    def test_enter_sends(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.NoModifier,
        )
        win._input_key_press(event)
        assert calls == ["hi"]
        assert win.input_widget.toPlainText() == ""

    def test_numpad_enter_sends(self, qapp, monkeypatch):
        """Numpad Enter (Key_Enter) sends just like the main Return key."""
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("hi")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Enter,
            _Qt.KeyboardModifier.NoModifier,
        )
        win._input_key_press(event)
        assert calls == ["hi"]

    def test_shift_enter_does_not_send(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtCore import Qt as _Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        calls = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: calls.append(text),
        )
        win = ChatWindow()
        win.input_widget.setPlainText("line one")
        event = QKeyEvent(
            QEvent.Type.KeyPress,
            _Qt.Key.Key_Return,
            _Qt.KeyboardModifier.ShiftModifier,
        )
        win._input_key_press(event)
        # Default QPlainTextEdit handling inserts a newline; no send.
        assert calls == []


@pytest.mark.unit
class TestChatWindowTranscriptScroll:
    """New messages keep the latest content visible (auto-scroll to bottom)."""

    def test_append_scrolls_to_bottom_after_many_lines(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtTest import QTest

        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages", lambda: []
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        # Force a tall transcript so the viewport is scrolled past the first
        # lines. Each append must bring the cursor (the view) back to the end.
        for _ in range(80):
            win._append_assistant("line of transcript content " * 4)
        QTest.qWait(100)

        scroll_bar = win.transcript_widget.verticalScrollBar()
        assert scroll_bar.maximum() > 0
        assert scroll_bar.value() == scroll_bar.maximum()

    @pytest.mark.parametrize("kind", ["user", "assistant", "system"])
    def test_new_message_scrolls_to_bottom_from_scrolled_up_position(self, qapp, monkeypatch, kind):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtTest import QTest

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])

        win = ChatWindow()
        win.show()
        for index in range(80):
            win._append_assistant(f"older message {index} " * 4)
        QTest.qWait(100)

        scroll_bar = win.transcript_widget.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.minimum())
        assert scroll_bar.value() < scroll_bar.maximum()

        getattr(win, f"_append_{kind}")("newest message")
        QTest.qWait(100)

        assert scroll_bar.value() == scroll_bar.maximum()


@pytest.mark.unit
class TestChatWindowCloseHidesNotDestroys:
    """Closing the window hides it; the tray re-shows the same instance."""

    def test_close_event_hides_window_without_destroying(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtGui import QCloseEvent

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        assert win.isVisible()

        win.closeEvent(QCloseEvent())
        qapp.processEvents()

        # Hidden, but the same instance is still usable (not destroyed).
        assert not win.isVisible()
        # The transcript and inputs remain intact: closing never resets state.
        assert win.transcript_widget is not None
        assert win.input_widget is not None


@pytest.mark.unit
class TestChatWindowHotWindowReplay:
    """Opening the window for the first time replays the daemon's current hot
    window so the user sees recent voice/text turns instead of a blank
    transcript. Seeded once; re-showing never duplicates."""

    def test_first_show_seeds_transcript_from_hot_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        hot_window = [
            {"role": "user", "content": "what is the weather"},
            {"role": "assistant", "content": "It is sunny."},
        ]
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: hot_window,
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()

        text = win.transcript_text()
        assert "what is the weather" in text
        assert "It is sunny." in text

    def test_re_show_does_not_duplicate_seeded_turns(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: [{"role": "user", "content": "hi"}],
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.hide()
        qapp.processEvents()
        win.show()
        qapp.processEvents()

        text = win.transcript_text()
        assert text.count("hi") == 1

    def test_empty_hot_window_leaves_transcript_blank(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages", lambda: []
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()

        assert win.transcript_text() == ""


@pytest.mark.unit
class TestChatWindowSmsLook:
    """The window reads as an SMS thread with a single contact: no session
    sidebar, one continuous conversation, speech bubbles aligned by sender,
    and a contact header."""

    def _window(self, qapp, **kwargs):
        from desktop_app.chat_window import ChatWindow
        return ChatWindow(**kwargs)

    def test_header_shows_contact_and_presence(self, qapp):
        from PyQt6.QtWidgets import QLabel
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        texts = [label.text() for label in win.findChildren(QLabel)]
        assert "Jarvis" in texts
        assert "Online" in texts

    def test_header_shows_typing_while_query_in_flight(self, qapp, monkeypatch):
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi")
        win._send()
        qapp.processEvents()
        assert win._header_status.text() == "Typing…"

    def test_single_conversation_accumulates_all_turns(self, qapp, monkeypatch):
        """Voice-seeded turns and typed turns live in one transcript; there
        is no way to split the conversation into separate sessions."""
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages",
            lambda: [
                {"role": "user", "content": "voice question"},
                {"role": "assistant", "content": "voice answer"},
            ],
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("typed question")
        win._send()
        win._on_complete("typed answer")

        text = win.transcript_text()
        assert "voice question" in text
        assert "voice answer" in text
        assert "typed question" in text
        assert "typed answer" in text

    def test_user_bubble_right_assistant_left(self, qapp, monkeypatch):
        """SMS layout: the user's bubble sits on the right half of the
        window, Jarvis's reply on the left half."""
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.show()
        qapp.processEvents()
        win.input_widget.setPlainText("hi there")
        win._send()
        win._on_complete("hello back")
        qapp.processEvents()

        bubbles = [
            label
            for label in win.transcript_widget.findChildren(QLabel)
            if label.objectName() == "bubble"
        ]
        assert len(bubbles) == 2
        user_bubble, assistant_bubble = bubbles
        mid = win.width() // 2
        assert user_bubble.mapTo(win, user_bubble.rect().topLeft()).x() > mid
        assert assistant_bubble.mapTo(win, assistant_bubble.rect().topLeft()).x() < mid

    def test_bubbles_show_plain_text_without_role_prefixes(self, qapp, monkeypatch):
        """The bubbles carry the message bodies only; position and colour
        convey the sender, so there is no 'You:' / 'Jarvis:' prefix."""
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.input_widget.setPlainText("no prefix")
        win._send()
        win._on_complete("plain reply")

        texts = [
            label.text()
            for label in win.transcript_widget.findChildren(QLabel)
            if label.objectName() == "bubble"
        ]
        assert texts == ["no prefix", "plain reply"]
        assert all("You:" not in t and "Jarvis:" not in t for t in texts)

    def test_bubbles_carry_timestamps(self, qapp, monkeypatch):
        from PyQt6.QtWidgets import QLabel
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: None
        )
        win = self._window(qapp)
        win.input_widget.setPlainText("timed")
        win._send()

        time_labels = [
            label
            for label in win.transcript_widget.findChildren(QLabel)
            if label.objectName() == "timestamp"
        ]
        assert time_labels, "each bubble should show a muted timestamp"
        assert all(":" in label.text() for label in time_labels)


@pytest.mark.unit
class TestChatRewind:
    """The rewind button under a sent message rolls the conversation back
    to that message and regenerates a fresh reply. The daemon is asked
    first and the transcript follows its verdict, so the view and the
    memory cannot part ways."""

    def _window(self, qapp, monkeypatch, **kwargs):
        from desktop_app.chat_window import ChatWindow
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages", lambda: []
        )
        return ChatWindow(**kwargs)

    def _send(self, win, text):
        win.input_widget.setPlainText(text)
        win._send()

    def _rewind_buttons(self, win):
        from PyQt6.QtWidgets import QPushButton
        return [
            b.objectName() for b in win.transcript_widget.findChildren(QPushButton)
            if b.objectName().startswith("rewind_")
        ]

    def _verdict_line(self, kind, user_index):
        from jarvis.daemon import CHAT_IPC_PREFIX
        return f'{CHAT_IPC_PREFIX}{{"type": "{kind}", "data": {{"user_index": {user_index}}}}}'

    def test_every_sent_message_carries_a_rewind_button(self, qapp, monkeypatch):
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "one")
        win._on_complete("reply one")
        self._send(win, "two")

        assert self._rewind_buttons(win) == ["rewind_1", "rewind_2"]

    def test_rewind_truncates_transcript_and_regenerates(self, qapp, monkeypatch):
        rewinds = []
        submits = []
        monkeypatch.setattr(
            "jarvis.daemon.rewind_chat_to_user",
            lambda idx, text: rewinds.append((idx, text)) or True,
        )
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query",
            lambda text, **kw: submits.append(text),
        )
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._on_complete("first reply")
        self._send(win, "second")
        win._on_complete("second reply")

        win._rewind_to_user(1, "first")

        assert rewinds == [(1, "first")], "the daemon memory is rewound to the message's text"
        assert submits[-1] == "first", "the rewound message must be re-submitted"
        assert "first" in win.transcript_text()
        assert "second" not in win.transcript_text()
        assert "first reply" not in win.transcript_text()

        # The fresh reply lands through the normal complete path.
        win._on_complete("fresh reply")
        assert "fresh reply" in win.transcript_text()

    def test_rewind_keeps_later_user_messages_after_regenerate(self, qapp, monkeypatch):
        """After a rewind + regenerate, the message ordinal stays stable so
        a subsequent send continues the conversation correctly."""
        monkeypatch.setattr("jarvis.daemon.rewind_chat_to_user", lambda idx, text: True)
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "one")
        win._on_complete("a")
        self._send(win, "two")
        win._on_complete("b")
        win._rewind_to_user(2, "two")
        win._on_complete("c")
        self._send(win, "three")

        assert self._rewind_buttons(win) == ["rewind_1", "rewind_2", "rewind_3"]

    def test_a_refused_rewind_leaves_the_transcript_and_says_so(self, qapp, monkeypatch):
        """The daemon can refuse (a query in flight, a turn the memory no
        longer holds): nothing moves, nothing is re-asked, and the user is
        told rather than left with a button that did nothing."""
        submits = []
        monkeypatch.setattr("jarvis.daemon.rewind_chat_to_user", lambda idx, text: False)
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: submits.append(text)
        )
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._on_complete("first reply")
        self._send(win, "second")
        win._on_complete("second reply")
        before = win.transcript_text()

        win._rewind_to_user(1, "first")

        assert submits == ["first", "second"]
        assert win.transcript_text().startswith(before)
        assert "second reply" in win.transcript_text()
        assert "Could not rewind" in win.transcript_text()
        assert not win.stop_button.isVisible()
        assert win.send_button.isEnabled()

    def test_rewind_noops_while_query_in_flight(self, qapp, monkeypatch):
        rewinds = []
        monkeypatch.setattr(
            "jarvis.daemon.rewind_chat_to_user",
            lambda idx, text: rewinds.append(idx) or True,
        )
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "question")  # leaves _query_in_flight True

        win._rewind_to_user(1, "question")

        assert rewinds == [], "rewind must be disabled while a query is in flight"

    def test_subprocess_rewind_waits_for_the_daemon_verdict(self, qapp, monkeypatch):
        commands = []
        submits = []
        win = self._window(
            qapp, monkeypatch,
            submit_fn=submits.append,
            control_fn=lambda kind, payload: commands.append((kind, payload)),
        )
        self._send(win, "question")
        win._on_complete("answer")
        self._send(win, "later")
        win._on_complete("later answer")

        win._rewind_to_user(1, "question")

        assert commands == [("rewind", {"user_index": 1, "content": "question"})]
        # Nothing moves until the daemon answers.
        assert "later answer" in win.transcript_text()
        assert submits == ["question", "later"]
        assert not win.send_button.isEnabled(), "the exchange is in flight from the user's side"

        assert win.process_ipc_line(self._verdict_line("rewound", 1)) is True

        assert submits == ["question", "later", "question"]
        assert "later" not in win.transcript_text()
        assert "answer" not in win.transcript_text()
        assert "question" in win.transcript_text()

    def test_subprocess_rewind_nack_keeps_transcript_and_memory_in_step(self, qapp, monkeypatch):
        submits = []
        win = self._window(
            qapp, monkeypatch,
            submit_fn=submits.append,
            control_fn=lambda kind, payload: None,
        )
        self._send(win, "question")
        win._on_complete("answer")
        before = win.transcript_text()

        win._rewind_to_user(1, "question")
        assert win.process_ipc_line(self._verdict_line("rewind_nack", 1)) is True

        assert submits == ["question"]
        assert win.transcript_text().startswith(before)
        assert "Could not rewind" in win.transcript_text()
        assert not win.stop_button.isVisible()
        assert self._rewind_buttons(win) == ["rewind_1"]

    def test_a_verdict_for_no_pending_rewind_changes_nothing(self, qapp, monkeypatch):
        win = self._window(qapp, monkeypatch, submit_fn=lambda t: None)
        self._send(win, "question")
        win._on_complete("answer")
        before = win.transcript_text()

        assert win.process_ipc_line(self._verdict_line("rewound", 1)) is True
        assert win.process_ipc_line(self._verdict_line("rewind_nack", 7)) is True

        assert win.transcript_text() == before
        assert not win.stop_button.isVisible()

    def test_stop_during_a_pending_rewind_rewinds_without_asking_again(self, qapp, monkeypatch):
        submits = []
        win = self._window(
            qapp, monkeypatch,
            submit_fn=submits.append,
            cancel_fn=lambda: None,
            control_fn=lambda kind, payload: None,
        )
        self._send(win, "question")
        win._on_complete("answer")
        win._rewind_to_user(1, "question")
        win._stop()

        assert win.process_ipc_line(self._verdict_line("rewound", 1)) is True

        assert submits == ["question"], "the abandoned regeneration is not asked"
        assert "answer" not in win.transcript_text(), "the transcript matches the rewound memory"
        assert not win.stop_button.isVisible()

    def test_a_message_the_daemon_refused_loses_its_ordinal(self, qapp, monkeypatch):
        """A busy rejection means the memory never received the turn: the
        bubble stays as what the user typed, but it cannot be rewound to,
        and the turns after it keep counting in step with the memory."""
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._on_complete("reply one")
        self._send(win, "second")
        win._on_busy()
        self._send(win, "third")
        win._on_complete("reply three")

        assert "second" in win.transcript_text()
        assert self._rewind_buttons(win) == ["rewind_1", "rewind_2"]

    def test_a_message_the_engine_dropped_loses_its_ordinal(self, qapp, monkeypatch):
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._on_complete(None)

        assert "first" in win.transcript_text()
        assert self._rewind_buttons(win) == []

        self._send(win, "second")
        win._on_complete("reply")
        assert self._rewind_buttons(win) == ["rewind_1"]

    def test_a_cancelled_message_keeps_its_ordinal(self, qapp, monkeypatch):
        """Stop drops the answer, not the turn: the engine still stores it."""
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        monkeypatch.setattr("jarvis.daemon.cancel_active_chat_query", lambda: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._stop()
        win._on_complete("late reply")

        assert "late reply" not in win.transcript_text()
        assert self._rewind_buttons(win) == ["rewind_1"]

    def test_a_daemon_that_comes_back_retires_the_old_rewind_anchors(self, qapp, monkeypatch):
        """A restarted daemon starts a fresh memory. The rows on screen stay
        as history, but they name turns of a memory that is gone, and the
        new turns count from one again, in step with the new memory."""
        monkeypatch.setattr("jarvis.daemon.submit_text_query", lambda text, **kw: None)
        win = self._window(qapp, monkeypatch)
        self._send(win, "first")
        win._on_complete("reply one")
        self._send(win, "second")
        win._on_complete("reply two")
        assert self._rewind_buttons(win) == ["rewind_1", "rewind_2"]

        win.set_daemon_status("stopped")
        win.set_daemon_status("running")

        assert "first" in win.transcript_text()
        assert self._rewind_buttons(win) == []
        self._send(win, "third")
        win._on_complete("reply three")
        assert self._rewind_buttons(win) == ["rewind_1"]

    def test_a_daemon_that_comes_back_seeds_its_hot_window_on_next_show(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        hot_window = []
        monkeypatch.setattr(
            "desktop_app.chat_window.get_hot_window_messages", lambda: list(hot_window)
        )
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        assert win.transcript_text() == ""

        hot_window.extend([
            {"role": "user", "content": "voice question"},
            {"role": "assistant", "content": "voice answer"},
        ])
        win.set_daemon_status("stopped")
        win.set_daemon_status("running")
        win.hide()
        win.show()
        qapp.processEvents()

        assert win.transcript_text().count("voice question") == 1
        assert "voice answer" in win.transcript_text()

    def test_a_daemon_going_away_abandons_a_pending_rewind(self, qapp, monkeypatch):
        submits = []
        win = self._window(
            qapp, monkeypatch,
            submit_fn=submits.append,
            control_fn=lambda kind, payload: None,
        )
        self._send(win, "question")
        win._on_complete("answer")
        win._rewind_to_user(1, "question")

        win.set_daemon_status("crashed")
        win.set_daemon_status("running")
        assert win.process_ipc_line(self._verdict_line("rewound", 1)) is True

        assert submits == ["question"], "a verdict from a daemon that is gone changes nothing"
        assert "answer" in win.transcript_text()


@pytest.mark.unit
class TestChatWindowHiddenAppend:
    """A message that lands while the window is hidden counts like one that
    lands while it is shown: the introductory panel is gone on the next
    show and the transcript is the only thing on screen."""

    def test_reply_on_a_never_shown_window_hides_the_empty_state_on_show(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from jarvis.daemon import CHAT_IPC_PREFIX

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        win = ChatWindow()
        win.process_ipc_line(f'{CHAT_IPC_PREFIX}{{"type": "complete", "data": "hello"}}')

        win.show()
        qapp.processEvents()

        assert not win.empty_state.isVisible()
        assert win.transcript_widget.isVisible()
        assert "hello" in win.transcript_text()

    def test_notice_after_close_hides_the_empty_state_on_re_show(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        win = ChatWindow()
        win.show()
        qapp.processEvents()
        win.close()
        win._append_system("something settled while you were away")

        win.show()
        qapp.processEvents()

        assert not win.empty_state.isVisible()
        assert win.transcript_widget.isVisible()


@pytest.mark.unit
class TestChatWindowScrollKeepsPlace:
    """The view follows the newest message only while the reader is at the
    end; a resize does not yank a reader who scrolled up."""

    def _tall_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow
        from PyQt6.QtTest import QTest

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        win = ChatWindow()
        win.show()
        for index in range(80):
            win._append_assistant(f"older message {index} " * 4)
        QTest.qWait(100)
        return win

    def test_a_resize_keeps_a_reader_who_scrolled_up_in_place(self, qapp, monkeypatch):
        from PyQt6.QtTest import QTest

        win = self._tall_window(qapp, monkeypatch)
        bar = win.transcript_widget.verticalScrollBar()
        bar.setValue(bar.minimum())
        assert bar.value() == bar.minimum() < bar.maximum()

        win.resize(win.width(), win.height() - 40)
        QTest.qWait(100)
        assert bar.value() == bar.minimum()

        win.resize(win.width() + 60, win.height())
        QTest.qWait(100)
        assert bar.value() == bar.minimum()

    def test_a_resize_keeps_a_reader_at_the_end_at_the_end(self, qapp, monkeypatch):
        from PyQt6.QtTest import QTest

        win = self._tall_window(qapp, monkeypatch)
        bar = win.transcript_widget.verticalScrollBar()
        assert bar.value() == bar.maximum()

        win.resize(win.width(), win.height() - 40)
        QTest.qWait(100)
        assert bar.value() == bar.maximum()


@pytest.mark.unit
class TestChatWindowMinimumSize:
    def test_the_window_cannot_shrink_below_the_usable_minimum(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        win = ChatWindow()
        win.show()
        win.resize(100, 100)
        qapp.processEvents()

        assert win.width() >= 380
        assert win.height() >= 560


@pytest.mark.unit
class TestChatWindowCancelKeepsLateReplyOut:
    """The window's half of cancellation: the answer to an abandoned
    exchange never reaches the transcript, and the next send is unaffected."""

    def test_a_reply_arriving_after_stop_stays_out_and_the_next_send_lands(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        monkeypatch.setattr("jarvis.daemon.cancel_active_chat_query", lambda: None)
        submits = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: submits.append(text)
        )
        win = ChatWindow()
        win.input_widget.setPlainText("question")
        win._send()
        win._stop()
        win._on_complete("late reply")
        assert "late reply" not in win.transcript_text()

        win.input_widget.setPlainText("again")
        win._send()
        win._on_complete("fresh reply")

        assert submits == ["question", "again"]
        assert "fresh reply" in win.transcript_text()

    def test_stop_routes_through_cancel_fn_in_subprocess_mode(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        in_process = []
        monkeypatch.setattr(
            "jarvis.daemon.cancel_active_chat_query", lambda: in_process.append(True)
        )
        routed = []
        win = ChatWindow(submit_fn=lambda text: None, cancel_fn=lambda: routed.append(True))
        win.input_widget.setPlainText("question")
        win._send()

        win._stop()

        assert routed == [True]
        assert in_process == [], "the query lives in the other process"


@pytest.mark.unit
class TestChatWindowGuards:
    def test_a_second_send_while_a_reply_is_pending_is_refused(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        submits = []
        monkeypatch.setattr(
            "jarvis.daemon.submit_text_query", lambda text, **kw: submits.append(text)
        )
        win = ChatWindow()
        win.input_widget.setPlainText("a")
        win._send()
        win.input_widget.setPlainText("b")
        win._send()

        assert submits == ["a"]
        assert win.input_widget.toPlainText() == "b", "the refused text is not lost"

    @pytest.mark.parametrize("payload", ["[1, 2]", '"just a string"', "42", "null"])
    def test_a_chat_line_whose_payload_is_not_an_object_is_ignored(self, qapp, monkeypatch, payload):
        from desktop_app.chat_window import ChatWindow
        from jarvis.daemon import CHAT_IPC_PREFIX

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        win = ChatWindow()

        assert win.process_ipc_line(f"{CHAT_IPC_PREFIX}{payload}") is True

        assert win.transcript_text() == ""
        assert not win.stop_button.isVisible()


@pytest.mark.unit
class TestDesktopAppStdinBus:
    """In subprocess mode the tray's hooks are the only way a query, a
    cancel, a rewind or a decision reaches the daemon: one line each on
    its stdin."""

    def _tray(self):
        import io
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None
        tray.is_listening = True
        sink = io.StringIO()
        tray.daemon_process = type("P", (), {"stdin": sink})()
        return tray, sink

    def _lines(self, sink):
        return sink.getvalue().splitlines()

    def test_submit_writes_the_query_line(self, qapp):
        import json
        from jarvis.daemon import CHAT_QUERY_IPC_PREFIX

        tray, sink = self._tray()
        tray._submit_chat_subprocess("hello over stdin")

        (line,) = self._lines(sink)
        assert line.startswith(CHAT_QUERY_IPC_PREFIX)
        assert json.loads(line[len(CHAT_QUERY_IPC_PREFIX):]) == {"text": "hello over stdin"}

    def test_cancel_writes_the_bare_cancel_line(self, qapp):
        from jarvis.daemon import CHAT_CANCEL_IPC_PREFIX

        tray, sink = self._tray()
        tray._cancel_chat_subprocess()

        assert self._lines(sink) == [CHAT_CANCEL_IPC_PREFIX]

    def test_rewind_writes_the_rewind_line_with_its_anchor(self, qapp):
        import json
        from jarvis.daemon import CHAT_REWIND_IPC_PREFIX

        tray, sink = self._tray()
        tray._control_chat_subprocess("rewind", {"user_index": 2, "content": "again"})

        (line,) = self._lines(sink)
        assert line.startswith(CHAT_REWIND_IPC_PREFIX)
        assert json.loads(line[len(CHAT_REWIND_IPC_PREFIX):]) == {"user_index": 2, "content": "again"}

    def test_an_unknown_control_writes_nothing(self, qapp):
        tray, sink = self._tray()
        tray._control_chat_subprocess("archive", {"anything": 1})

        assert sink.getvalue() == ""

    def test_a_decision_writes_the_decision_line(self, qapp):
        import json
        from jarvis.daemon import CHAT_DECISION_IPC_PREFIX

        tray, sink = self._tray()
        tray._send_decision_subprocess("cf_1", True)

        (line,) = self._lines(sink)
        assert line.startswith(CHAT_DECISION_IPC_PREFIX)
        assert json.loads(line[len(CHAT_DECISION_IPC_PREFIX):]) == {"request_id": "cf_1", "approved": True}

    def test_a_dead_pipe_on_submit_resets_the_window(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        tray, _ = self._tray()
        tray.daemon_process = None
        tray.chat_window = ChatWindow(submit_fn=tray._submit_chat_subprocess)

        tray.chat_window.input_widget.setPlainText("hi")
        tray.chat_window._send()
        qapp.processEvents()

        assert not tray.chat_window.stop_button.isVisible(), "no reply is coming"
        assert not tray.chat_window.send_button.isEnabled(), "the daemon is gone"

    def test_a_dead_pipe_on_rewind_stops_the_window_waiting(self, qapp, monkeypatch):
        from desktop_app.chat_window import ChatWindow

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        tray, _ = self._tray()
        tray.daemon_process = None
        win = ChatWindow(submit_fn=lambda text: None, control_fn=tray._control_chat_subprocess)
        tray.chat_window = win
        win.input_widget.setPlainText("question")
        win._send()
        win._on_complete("answer")

        win._rewind_to_user(1, "question")

        assert not win.stop_button.isVisible(), "no verdict is coming"
        assert not win.send_button.isEnabled(), "the daemon is gone"

    def test_a_dead_pipe_on_a_decision_raises_so_the_card_says_so(self, qapp):
        tray, _ = self._tray()
        tray.daemon_process = None

        with pytest.raises(RuntimeError):
            tray._send_decision_subprocess("cf_1", False)


@pytest.mark.unit
class TestDesktopAppChatBus:
    """What the tray does with chat lines in either mode."""

    def _tray(self):
        import desktop_app.app as app_mod

        tray = app_mod.JarvisSystemTray.__new__(app_mod.JarvisSystemTray)
        tray.chat_window = None
        tray._chat_submit_fn = None
        tray._chat_cancel_fn = None
        tray._chat_control_fn = None
        tray.is_listening = True
        return tray

    def test_chat_lines_stay_out_of_the_log_viewer(self):
        """The complete event carries the whole reply, which can echo what
        the user typed; the log window is outside the redaction invariant."""
        from desktop_app.app import _should_emit_as_log
        from jarvis.daemon import CHAT_IPC_PREFIX

        assert _should_emit_as_log(f'{CHAT_IPC_PREFIX}{{"type": "complete", "data": "secret"}}') is False
        assert _should_emit_as_log('__DIARY__:{"type": "token", "data": "x"}') is True
        assert _should_emit_as_log("🚀 plain log line") is True

    def test_bundled_confirmation_reply_reaches_the_chat_window(self, qapp, monkeypatch):
        """Bundled mode has no stdout bus: the narration of a confirmed
        action is fed through the same parser as a subprocess line."""
        from desktop_app.app import _chat_event_line

        monkeypatch.setattr("desktop_app.chat_window.get_hot_window_messages", lambda: [])
        tray = self._tray()

        tray._on_chat_ipc_line(_chat_event_line("complete", "fait, le fichier est parti"))

        tray.chat_window.show()
        qapp.processEvents()
        assert "fait, le fichier est parti" in tray.chat_window.transcript_text()

    def test_clearing_the_hooks_resets_the_decision_writer(self, qapp, monkeypatch):
        """After the daemon is gone, a decision must not be written to a
        dead pipe; the in-process route is what is left."""
        from desktop_app import chat_window as cw

        written = []
        resolved = []
        monkeypatch.setattr(
            "jarvis.daemon.resolve_confirmation",
            lambda rid, ok: resolved.append((rid, ok)) or "inconnue",
        )
        cw.set_decision_writer(lambda rid, ok: written.append((rid, ok)))
        tray = self._tray()

        tray._clear_chat_hooks()
        cw.send_confirmation_decision("cf_1", True)

        assert written == []
        assert resolved == [("cf_1", True)]
