"""A discarded utterance is not announced as heard.

Field trace, 2026-08-17, air conditioning running. The console filled with

    📝 Heard: "That, I, I, I, I, I, I, …"   (×110, over and over)

The repetition guard catches every one of those — verified, it returns
True. They never reach the transcript buffer, the judge, or a turn. The
system is behaving.

But the announcement is printed *before* the guard runs, and the
rejection only goes to the debug log, which is off by default. So the one
surface he actually reads shows a flood of garbage being heard, when what
is happening is a flood of garbage being thrown away.

It is the day's motif inverted — a success wearing the face of a failure
— and it costs the same thing: a log that cannot be trusted is a log
nobody uses to judge.

What is thrown away says so, in one short line, and keeps its reason.
"""

from __future__ import annotations

import pytest


BOUCLE = "That, " + ", ".join(["I"] * 110)
VRAIE = "Yuba, quelle est la météo à Bagneux aujourd'hui ?"


def _annonce(texte: str) -> str:
    """What the console gets for one transcription, heard or discarded."""
    from src.jarvis.listening.listener import VoiceListener

    ecouteur = VoiceListener.__new__(VoiceListener)
    ecouteur._first_utterance = True
    return ecouteur._transcription_announcement(texte)


def test_a_real_utterance_is_announced_as_heard():
    ligne = _annonce(VRAIE)

    assert "📝 Heard" in ligne
    assert "Bagneux" in ligne


def test_a_repetition_loop_is_announced_as_discarded():
    ligne = _annonce(BOUCLE)

    assert "📝 Heard" not in ligne
    assert "🔇" in ligne


def test_the_discarded_line_stays_short():
    """A hundred and ten repetitions of "I" is what made the log
    unreadable in the first place; repeating it under a different emoji
    would fix nothing."""
    ligne = _annonce(BOUCLE)

    assert len(ligne) < 200, len(ligne)


def test_the_discarded_line_says_why():
    ligne = _annonce(BOUCLE)

    assert "repet" in ligne.lower() or "répét" in ligne.lower()


def test_an_empty_transcription_is_not_announced_at_all():
    assert _annonce("") == ""
    assert _annonce("   ") == ""
