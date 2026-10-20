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
