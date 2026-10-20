"""A Claude Code report arrives in the chat window as her own message.

It lands with no turn of its own, minutes after the one that started the
job, so it cannot ride the `complete` event a reply ends with.

See src/jarvis/projets/projets.spec.md, *Delivering the report*.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def win(qapp):
    from desktop_app.chat_window import ChatWindow

    w = ChatWindow()
    yield w
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _line(data):
    from jarvis.daemon import CHAT_IPC_PREFIX

    return CHAT_IPC_PREFIX + json.dumps({"type": "delegation_done", "data": data})


def test_the_report_is_shown_as_her_message(win):
    texte = "Claude Code a fini sur chantipay. J'ai ajouté le bouton."

    assert win.process_ipc_line(_line({"projet": "chantipay", "ok": True, "texte": texte}))

    assert win._messages[-1]["kind"] == "assistant"
    assert win._messages[-1]["text"] == texte


def test_a_report_without_text_shows_nothing(win):
    before = list(win._messages)

    win.process_ipc_line(_line({"projet": "chantipay", "ok": True}))
    win.process_ipc_line(_line("pas un objet"))

    assert win._messages == before


def test_a_window_not_yet_shown_takes_the_report_from_its_first_seeding(qapp, monkeypatch):
    """Bundled mode: the report is already in the dialogue memory the
    window seeds from on first show. Appending it on arrival as well would
    show it twice, the first copy above the turns that led to it."""
    from desktop_app import chat_window as cw

    texte = "Claude Code a fini sur chantipay : « Fait. »"
    monkeypatch.setattr(cw, "get_hot_window_messages", lambda: [
        {"role": "user", "content": "demande à Claude Code d'ajouter un bouton"},
        {"role": "assistant", "content": "C'est parti."},
        {"role": "assistant", "content": texte},
    ])
    w = cw.ChatWindow(daemon_available=True)
    try:
        w.process_ipc_line(_line({"projet": "chantipay", "ok": True, "texte": texte}))
        assert w._messages == []

        w.show()
        qapp.processEvents()

        assert [m["text"] for m in w._messages].count(texte) == 1
        assert w._messages[-1]["text"] == texte
    finally:
        w.close()
        w.deleteLater()
        qapp.processEvents()


def test_a_report_the_seeding_will_not_bring_is_shown_at_once(qapp, monkeypatch):
    """Subprocess mode: the memory lives in the daemon's process, and the
    window's seeding finds nothing there."""
    from desktop_app import chat_window as cw

    monkeypatch.setattr(cw, "get_hot_window_messages", lambda: [])
    w = cw.ChatWindow(daemon_available=True)
    try:
        w.process_ipc_line(_line({"projet": "chantipay", "ok": True, "texte": "Fait."}))
        assert [m["text"] for m in w._messages] == ["Fait."]
    finally:
        w.close()
        w.deleteLater()
        qapp.processEvents()
