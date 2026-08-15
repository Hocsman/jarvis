"""The tray, which is the only surface always there.

A destructive action can be approved by a click and by nothing else. If
the user is across the room talking to Yuba with every window closed,
they hear the question and — without this — have no way to answer it.
The action then expires, and the thing they asked for never happens, with
no explanation beyond silence.

So a waiting question puts an entry in the tray menu and raises a
notification, both of which open the card. The entry disappears when the
question settles, because a menu item offering to answer a question
nobody is asking is worse than none.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tray(qapp):
    """The tray app with its heavy parts stubbed."""
    from desktop_app.app import JarvisSystemTray

    app = JarvisSystemTray.__new__(JarvisSystemTray)
    app.tray_icon = MagicMock()
    app.chat_window = None
    app._pending_confirmation = None
    app.menu = None
    app._build_confirmation_action()
    return app


def _question(request_id="cf_abc123", tool="localFiles", risk="destructif"):
    return {
        "request_id": request_id,
        "tool": tool,
        "risk": risk,
        "channel": "geste",
        "origin": "voix",
        "shown": f"{tool} · {risk}\n" + '{"operation": "delete"}',
        "hazards": [],
        "ttl_sec": 180.0,
    }


# ── It appears ────────────────────────────────────────────────────────


def test_no_entry_until_there_is_a_question(tray):
    assert tray.confirmation_action.isVisible() is False


def test_a_question_puts_an_entry_in_the_menu(tray):
    tray.on_confirmation_raised(_question())

    assert tray.confirmation_action.isVisible() is True


def test_the_entry_names_the_tool(tray):
    """"Something needs your permission" tells the user nothing about
    whether to walk over."""
    tray.on_confirmation_raised(_question(tool="localFiles"))

    assert "localFiles" in tray.confirmation_action.text()


def test_a_question_raises_a_notification(tray):
    """The user is not looking at the menu bar. Something has to reach
    them."""
    tray.on_confirmation_raised(_question())

    assert tray.tray_icon.showMessage.called


def test_the_notification_says_what_is_being_asked(tray):
    tray.on_confirmation_raised(_question(tool="localFiles"))

    body = " ".join(str(a) for a in tray.tray_icon.showMessage.call_args.args)
    assert "localFiles" in body


def test_the_notification_carries_no_arguments(tray):
    """A notification lands on a lock screen and in a system log. The
    card is where the path belongs."""
    q = _question()
    q["shown"] = 'localFiles · destructif\n{"path": "/Users/hocine/secret.key"}'
    tray.on_confirmation_raised(q)

    body = " ".join(str(a) for a in tray.tray_icon.showMessage.call_args.args)
    assert "secret.key" not in body


# ── It leads somewhere ────────────────────────────────────────────────


def test_the_entry_opens_the_card(tray, monkeypatch):
    opened = []
    monkeypatch.setattr(type(tray), "show_chat", lambda self: opened.append(True))
    tray.on_confirmation_raised(_question())

    tray.confirmation_action.trigger()

    assert opened == [True]


def test_clicking_the_notification_opens_the_card(tray, monkeypatch):
    opened = []
    monkeypatch.setattr(type(tray), "show_chat", lambda self: opened.append(True))
    tray.on_confirmation_raised(_question())

    tray.on_notification_clicked()

    assert opened == [True]


def test_a_notification_click_with_nothing_pending_does_nothing(tray, monkeypatch):
    opened = []
    monkeypatch.setattr(type(tray), "show_chat", lambda self: opened.append(True))

    tray.on_notification_clicked()

    assert opened == []


# ── It goes away ──────────────────────────────────────────────────────


def test_a_settled_question_removes_the_entry(tray):
    tray.on_confirmation_raised(_question())

    tray.on_confirmation_settled("cf_abc123", "accordé")

    assert tray.confirmation_action.isVisible() is False


def test_a_settlement_for_another_question_leaves_the_entry(tray):
    tray.on_confirmation_raised(_question())

    tray.on_confirmation_settled("cf_autre", "accordé")

    assert tray.confirmation_action.isVisible() is True


def test_an_expired_question_removes_the_entry(tray):
    tray.on_confirmation_raised(_question())

    tray.on_confirmation_settled("cf_abc123", "expiré")

    assert tray.confirmation_action.isVisible() is False


# ── The card is fed too ───────────────────────────────────────────────


def test_the_question_reaches_an_open_chat_window(tray):
    """Bundled mode has no stdout bus between the two halves, so the
    window is fed directly."""
    window = MagicMock()
    tray.chat_window = window

    tray.on_confirmation_raised(_question())

    assert window._show_confirmation.call_args.args[0]["request_id"] == "cf_abc123"


def test_a_settlement_reaches_an_open_chat_window(tray):
    window = MagicMock()
    tray.chat_window = window
    tray.on_confirmation_raised(_question())

    tray.on_confirmation_settled("cf_abc123", "décliné")

    assert window._settle_confirmation.called


def test_a_missing_chat_window_is_not_an_error(tray):
    tray.chat_window = None

    tray.on_confirmation_raised(_question())

    assert tray.confirmation_action.isVisible() is True


def test_a_window_already_showing_it_is_not_fed_twice(tray):
    """In subprocess mode the same line reaches the window through
    `process_ipc_line` first. Re-rendering the card it already shows
    would reset the buttons under a user mid-click."""
    window = MagicMock()
    window._pending_request_id = "cf_abc123"
    tray.chat_window = window

    tray.on_confirmation_raised(_question())

    assert window._show_confirmation.called is False


# ── The subprocess route ──────────────────────────────────────────────


def test_a_confirm_line_reaches_the_tray(tray):
    """Subprocess mode: the tray learns about a question from the same
    stdout line the chat window reads."""
    import json

    from jarvis.daemon import CHAT_IPC_PREFIX

    tray._route_confirmation_line(
        CHAT_IPC_PREFIX + json.dumps({"type": "confirm", "data": _question()})
    )

    assert tray.confirmation_action.isVisible() is True


def test_a_settled_line_clears_the_tray(tray):
    import json

    from jarvis.daemon import CHAT_IPC_PREFIX

    tray.on_confirmation_raised(_question())

    tray._route_confirmation_line(CHAT_IPC_PREFIX + json.dumps({
        "type": "confirm_settled",
        "data": {"request_id": "cf_abc123", "outcome": "accordé"},
    }))

    assert tray.confirmation_action.isVisible() is False


def test_an_unrelated_line_is_ignored(tray):
    tray._route_confirmation_line("🚀 Jarvis daemon started")

    assert tray.confirmation_action.isVisible() is False


def test_a_malformed_line_does_not_crash(tray):
    from jarvis.daemon import CHAT_IPC_PREFIX

    tray._route_confirmation_line(CHAT_IPC_PREFIX + "{pas du json")

    assert tray.confirmation_action.isVisible() is False
