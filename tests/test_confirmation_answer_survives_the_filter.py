"""The one word the question was asking for.

A spoken confirmation opens a twelve-second window and the user says
"oui". That utterance then has to survive every filter between the
microphone and the approval judge, and one of them was tuned for the
opposite case.

`whisper_min_confidence` is 0.3, computed from `avg_logprob`. A single
short word carries almost no acoustic context, so its logprob is poor by
construction — the observed answer scored 0.22 and was dropped with
`🔇 Low confidence`, before anything about confirmations was consulted.
The shorter the answer the likelier it is discarded, and "oui" and "non"
are as short as answers get. The spoken channel was therefore
unanswerable in practice, however well the judge behind it worked.

Lifting the bar is safe precisely because the judge is behind it. That
filter exists to stop her *acting* on a mumble; here nothing acts on it.
`read_approval` fails closed, so a fragment it cannot read as a clear yes
is not an approval. The hard `no_speech_prob` filter still applies, so
silence and hallucinations are still dropped — "is this speech at all"
and "am I sure enough of the words to act" are different questions, and
only the second one has moved.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.jarvis.listening.listener import VoiceListener
from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.confirmation import CHANNEL_GESTE, CHANNEL_PAROLE, PendingAction


class _Cfg:
    whisper_min_confidence = 0.3
    whisper_no_speech_threshold = 0.5
    voice_debug = False


def _listener(memory):
    """A listener with only what the filter reads."""
    listener = VoiceListener.__new__(VoiceListener)
    listener.cfg = _Cfg()
    listener.dialogue_memory = memory
    return listener


def _pending(channel):
    memory = DialogueMemory()
    memory.raise_pending(PendingAction.create(
        tool="setRoutine", args={}, risk="action", channel=channel,
        origin="voix", query_redacted="tous les matins", raised_at_turn=1,
        ttl_sec=180.0,
    ))
    return memory


def _segment(text, confidence, no_speech=0.1):
    # `avg_logprob + 1` is the confidence the listener computes.
    return SimpleNamespace(
        text=text, avg_logprob=confidence - 1.0, no_speech_prob=no_speech,
    )


def _kept(listener, *segments):
    return [s.text for s in listener._filter_noisy_segments(list(segments))]


# ── The answer gets through ───────────────────────────────────────────


def test_a_quiet_yes_reaches_the_judge_while_a_question_waits():
    """0.22 is what was actually observed, against a 0.3 floor."""
    listener = _listener(_pending(CHANNEL_PAROLE))

    assert _kept(listener, _segment("Oui....", 0.22)) == ["Oui...."]


def test_a_quiet_no_gets_through_as_well():
    """A refusal dropped is worse than an approval dropped: the user
    said no, nothing recorded it, and the question expires as though
    they had ignored it."""
    listener = _listener(_pending(CHANNEL_PAROLE))

    assert _kept(listener, _segment("Non...", 0.19)) == ["Non..."]


# ── And only then ─────────────────────────────────────────────────────


def test_the_same_mumble_is_dropped_when_nothing_is_waiting():
    """The filter is not weakened, it is suspended for the one moment a
    fragment is expected to be an answer."""
    listener = _listener(DialogueMemory())

    assert _kept(listener, _segment("Oui....", 0.22)) == []


def test_a_click_only_question_does_not_relax_anything():
    """Nothing said can settle a destructive action, so letting mumbles
    through would only collect answers that have to be thrown away."""
    listener = _listener(_pending(CHANNEL_GESTE))

    assert _kept(listener, _segment("Oui....", 0.22)) == []


# ── What still gets dropped either way ────────────────────────────────


def test_silence_is_still_silence():
    """Two different questions: "is this speech at all" and "am I sure
    enough of the words to act". Only the second one has moved."""
    listener = _listener(_pending(CHANNEL_PAROLE))

    assert _kept(listener, _segment("Merci d'avoir regardé", 0.9, no_speech=0.9)) == []


def test_a_confident_answer_was_never_the_problem():
    listener = _listener(_pending(CHANNEL_PAROLE))

    assert _kept(listener, _segment("oui vas-y", 0.8)) == ["oui vas-y"]


# ── And the word that wakes her ───────────────────────────────────────


def test_her_own_name_survives_a_poor_transcription():
    """`avg_logprob` is a poor judge of a proper noun the model has never
    seen — which is exactly why "Yuba" comes back as "Youba", "Nuba",
    "Juba". The segment carrying it scores low and is thrown away, the
    sentence that follows arrives with no wake word in it and is ignored,
    and she looks broken. That is the one failure a wake word cannot
    afford."""
    listener = _listener(DialogueMemory())
    listener.cfg.wake_word = "yuba"
    listener.cfg.wake_aliases = []

    assert _kept(listener, _segment("Yuba...", 0.20)) == ["Yuba..."]


def test_a_mangled_spelling_of_it_survives_too():
    listener = _listener(DialogueMemory())
    listener.cfg.wake_word = "yuba"
    listener.cfg.wake_aliases = []

    assert _kept(listener, _segment("Youba !...", 0.24)) == ["Youba !..."]


def test_ambient_speech_without_her_name_is_still_dropped():
    """36 of the 38 segments this filter dropped in one afternoon were a
    television and a conversation in the room. Lifting the bar for all of
    them would send every one of those to the intent judge."""
    listener = _listener(DialogueMemory())
    listener.cfg.wake_word = "yuba"
    listener.cfg.wake_aliases = []

    assert _kept(listener, _segment("C'est trop dur, merci hein...", 0.28)) == []


def test_keeping_it_decides_nothing():
    """The wake detector and the intent judge both still run. This only
    stops the word being deleted before either can see it."""
    listener = _listener(DialogueMemory())
    listener.cfg.wake_word = "yuba"
    listener.cfg.wake_aliases = []

    assert _kept(listener, _segment("Yuba", 0.9)) == ["Yuba"]
