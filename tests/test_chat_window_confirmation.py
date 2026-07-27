"""The card: the only way a destructive action can be approved.

Voice cannot grant one, by design, so if this card does not appear or its
buttons do not work, a destructive action has no route to happening at
all — and the user is left listening to a question with no way to answer
it.

Two properties matter more than the rest. It shows the call exactly as
the gate resolved it, because it is the copy the user decides on and the
arguments are model output. And it refuses by default: escape, closing,
and the default button all decline, so an absent-minded keystroke can
only ever be a no.
"""

from __future__ import annotations

import json

import pytest


def _confirm_line(**over):
    from jarvis.daemon import CHAT_IPC_PREFIX

    data = {
        "request_id": "cf_abc123",
        "tool": "localFiles",
        "risk": "destructif",
        "channel": "geste",
        "origin": "voix",
        "shown": 'localFiles · destructif\n{"operation": "delete", "path": "/a  b"}',
        "hazards": [],
        "ttl_sec": 180.0,
    }
    data.update(over)
    return CHAT_IPC_PREFIX + json.dumps({"type": "confirm", "data": data})


def _settled_line(request_id="cf_abc123", outcome="accordé"):
    from jarvis.daemon import CHAT_IPC_PREFIX

    return CHAT_IPC_PREFIX + json.dumps({
        "type": "confirm_settled",
        "data": {"request_id": request_id, "outcome": outcome},
    })


@pytest.fixture
def win(qapp):
    from desktop_app.chat_window import ChatWindow

    w = ChatWindow()
    yield w
    w.close()


# ── It appears ────────────────────────────────────────────────────────


def test_no_card_until_there_is_a_question(win):
    assert win.confirmation_card.isVisible() is False


def test_a_question_raises_the_card(win):
    win.process_ipc_line(_confirm_line())

    assert win.confirmation_card.isVisibleTo(win)


def test_the_card_shows_the_call_exactly_as_resolved(win):
    """The copy the user decides on. Whitespace is not collapsed and
    nothing is truncated — the ledger's copy is redacted, and a path is
    not a sentence."""
    win.process_ipc_line(_confirm_line())

    assert '"path": "/a  b"' in win.confirmation_detail.text()


def test_the_card_names_the_risk(win):
    win.process_ipc_line(_confirm_line())

    assert "destructif" in win.confirmation_detail.text()


def test_a_hazard_is_shown(win):
    win.process_ipc_line(_confirm_line(hazards=["caractères invisibles"]))

    assert "caractères invisibles" in win.confirmation_hazards.text()
    assert win.confirmation_hazards.isVisibleTo(win)


def test_no_hazard_strip_when_there_is_nothing_to_warn_about(win):
    win.process_ipc_line(_confirm_line())

    assert win.confirmation_hazards.isVisibleTo(win) is False


# ── It refuses by default ─────────────────────────────────────────────


def test_the_declining_button_is_the_default(win):
    """An absent-minded return key must only ever be a no."""
    win.process_ipc_line(_confirm_line())

    assert win.confirmation_decline_button.isDefault() is True
    assert win.confirmation_approve_button.isDefault() is False


def test_escape_declines(win, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent

    sent = []
    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: sent.append((rid, approved)),
    )
    win.process_ipc_line(_confirm_line())

    win.keyPressEvent(QKeyEvent(
        QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier,
    ))

    assert sent == [("cf_abc123", False)]


# ── The buttons send exactly one decision ─────────────────────────────


def test_approving_sends_one_decision(win, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: sent.append((rid, approved)),
    )
    win.process_ipc_line(_confirm_line())

    win.confirmation_approve_button.click()

    assert sent == [("cf_abc123", True)]


def test_a_double_click_still_sends_one_decision(win, monkeypatch):
    """The daemon is atomic about this too, but a UI that fires twice
    makes the second one look like it failed."""
    sent = []
    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: sent.append((rid, approved)),
    )
    win.process_ipc_line(_confirm_line())

    win.confirmation_approve_button.click()
    win.confirmation_approve_button.click()

    assert len(sent) == 1


def test_declining_sends_a_refusal(win, monkeypatch):
    sent = []
    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: sent.append((rid, approved)),
    )
    win.process_ipc_line(_confirm_line())

    win.confirmation_decline_button.click()

    assert sent == [("cf_abc123", False)]


# ── The input never locks ─────────────────────────────────────────────


def test_a_waiting_card_does_not_block_typing(win):
    """A modal question would strand a user who wants to ask something
    else, or to say why they are hesitating."""
    win.process_ipc_line(_confirm_line())

    assert win.input_widget.isEnabled()
    assert win.send_button.isEnabled()


# ── It goes away when settled ─────────────────────────────────────────


def test_a_settled_question_removes_the_card(win):
    win.process_ipc_line(_confirm_line())

    win.process_ipc_line(_settled_line())

    assert win.confirmation_card.isVisibleTo(win) is False


def test_a_settlement_for_another_question_leaves_the_card(win):
    win.process_ipc_line(_confirm_line())

    win.process_ipc_line(_settled_line(request_id="cf_autre"))

    assert win.confirmation_card.isVisibleTo(win)


def test_an_expired_question_says_so_rather_than_vanishing(win):
    """Being told a decision was waiting and never told what became of it
    is worse than being told it lapsed."""
    win.process_ipc_line(_confirm_line())

    win.process_ipc_line(_settled_line(outcome="expiré"))

    assert "expir" in win.transcript_widget.toPlainText().lower()


# ── A click that did not take ─────────────────────────────────────────


def test_a_rejected_click_leaves_the_card_live(win, monkeypatch):
    """`occupée` means a turn is in flight, not that the question is
    settled. The card must stay usable, or a correct decision looks like
    a broken button."""
    from jarvis.daemon import CHAT_IPC_PREFIX

    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: None,
    )
    win.process_ipc_line(_confirm_line())
    win.confirmation_approve_button.click()

    win.process_ipc_line(CHAT_IPC_PREFIX + json.dumps({
        "type": "confirm_nack",
        "data": {"request_id": "cf_abc123", "outcome": "occupée"},
    }))

    assert win.confirmation_card.isVisibleTo(win)
    assert win.confirmation_approve_button.isEnabled()


def test_a_rejected_click_says_why(win, monkeypatch):
    from jarvis.daemon import CHAT_IPC_PREFIX

    monkeypatch.setattr(
        "desktop_app.chat_window.send_confirmation_decision",
        lambda rid, approved: None,
    )
    win.process_ipc_line(_confirm_line())

    win.process_ipc_line(CHAT_IPC_PREFIX + json.dumps({
        "type": "confirm_nack",
        "data": {"request_id": "cf_abc123", "outcome": "occupée"},
    }))

    assert win.confirmation_status.text()


# ── Malformed input changes nothing ───────────────────────────────────


def test_a_confirm_without_an_id_raises_no_card(win):
    win.process_ipc_line(_confirm_line(request_id=None))

    assert win.confirmation_card.isVisibleTo(win) is False


def test_a_malformed_confirm_line_does_not_crash(win):
    from jarvis.daemon import CHAT_IPC_PREFIX

    assert win.process_ipc_line(CHAT_IPC_PREFIX + "{pas du json") is True
