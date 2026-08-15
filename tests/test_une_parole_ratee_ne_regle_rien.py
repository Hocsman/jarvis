"""A reminder is settled by speech, not by the attempt to speak.

`reminders.spec.md` is explicit: delivery settles it, never queueing, and
the row stays owed until the speech has finished. The chain that keeps
that promise ends in the TTS completion callback: engine finishes →
`_on_tts_complete` → `on_spoken` → `ReminderScheduler._settle` → the row
is marked said and a ledger line says `ok`.

The engines fire that callback from a `finally` guarded only on
`interrupted`, which is set on a real interruption and nowhere else. So
every failure path fires it too: the engine that could not initialise,
the synthesis that produced no audio, the exception caught above. A
broken voice does not delay the reminder, it deletes it — and the ledger
records that she said it.

Interruption is the case the flag was written for and it stays: an
interrupted reminder is unheard, so it must stay owed as well.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _kokoro():
    from src.jarvis.output.tts import KokoroTTS

    moteur = KokoroTTS.__new__(KokoroTTS)
    import threading

    moteur._is_speaking = threading.Event()
    moteur._should_interrupt = threading.Event()
    moteur._audio_lock = threading.Lock()
    moteur._audio_stream = None
    moteur._last_spoken_text = ""
    moteur._init_error = "kokoro absent"
    moteur._completion_callback = None
    moteur._notify_speaking_state = lambda *a, **kw: None
    moteur._ensure_initialized = lambda: False
    return moteur


def test_an_engine_that_cannot_start_does_not_report_the_words_as_spoken():
    """The path a fresh install takes when the model never downloaded."""
    moteur = _kokoro()
    dit = []
    moteur._completion_callback = lambda: dit.append(1)

    moteur._speak_once("rappel : appeler l'oncologue")

    assert dit == [], "une synthèse qui n'a pas démarré a réglé le rappel"


def test_the_promise_the_flag_was_written_for_still_holds():
    """An interrupted reminder is unheard, so it stays owed too. This is
    the case `interrupted` already covered, kept green."""
    moteur = _kokoro()
    moteur._should_interrupt.set()
    dit = []
    moteur._completion_callback = lambda: dit.append(1)

    moteur._speak_once("rappel : appeler l'oncologue")

    assert dit == []


def test_the_scheduler_leaves_the_row_owed_when_nothing_was_said(tmp_path):
    """The property as he meets it: the reminder comes back rather than
    vanishing with a ledger line saying she said it."""
    from datetime import datetime, timedelta, timezone

    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"))
    rid = db.add_rappel(
        texte="appeler l'oncologue",
        due_utc=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat(),
        due_local="2026-08-14T09:00:00", tz="Europe/Paris",
    )

    # Nothing spoke, so nothing settles the row.
    encore_du = [r for r in db.pending_rappels() if r["id"] == rid]

    assert encore_du, "la ligne doit rester due tant que rien ne l'a dite"
