"""Overriding an echo verdict needs proof of speech, not absence of proof.

Field trace (2026-08-16). The judge read an utterance and said, in its own
words, that it was the assistant's echo and not a user query. The code
overrode it and ran a full turn on her own garbled voice. She answered
"Hocine, je crois que la reconnaissance vocale te fait un sale coup" —
to herself.

The override exists for barge-in: the user speaking over her should not
be thrown away. But its test was `not is_pure_echo`, and `is_pure_echo`
rests on `fuzz.partial_ratio`, which looks for a *contiguous* window. The
echo that slipped through was decimated — Whisper dropped four words from
the middle and corrupted two others — so it is no longer a contiguous
slice of what she said and the score fell under the bar. A similarity
measure that failed to find its match was read as evidence of new
content. That is the absence of proof standing in for proof.

Measured per word, no threshold can separate the two: 'même' against
'semaine' scores 36, exactly what 'météo' against 'écoute' scores, and
'oiseaux' against 'août' scores 18 — below several genuinely new words.
Aggregated over the utterance it separates cleanly, and stably: the
fraction of heard words with no close match in the TTS was 0% and 18% on
the two real echoes, 62% and 100% on barge-in, unchanged whether the
per-word bar sits at 70, 80 or 90.

So the question asked is "is there anything here she did not say?", which
is the question that was meant all along.
"""

from __future__ import annotations

import pytest

from src.jarvis.listening.listener import _carries_speech_she_did_not_say


# Her actual reply from the trace.
DITE = ("Ah, pardon, la transcription a dû manger quelques mots — tu veux "
        "dire que tu te mets en mode semaine, ou tu voulais dire autre "
        "chose ? Je t'écoute.")


def test_the_decimated_echo_that_slipped_through_is_not_speech():
    """The case that made her answer herself: words dropped from the
    middle, two corrupted, nothing new."""
    entendu = "te mets en mode même dire autre chose je t'écris"

    assert _carries_speech_she_did_not_say(entendu, DITE) is False


def test_a_near_verbatim_echo_is_not_speech():
    entendu = ("ah pardon, la transcription a dû manger quelques mots. "
               "tu veux dire que")

    assert _carries_speech_she_did_not_say(entendu, DITE) is False


def test_an_interruption_is_speech():
    """The control, and the reason the override exists at all. Losing
    this would be worse than the bug."""
    assert _carries_speech_she_did_not_say("non attends arrête", DITE) is True


def test_speech_mixed_into_echo_is_still_speech():
    """The genuine mixed case the override was written for: he talks over
    the tail of her sentence."""
    entendu = ("tu veux dire que non attends donne-moi plutôt la météo "
               "à Bagneux")

    assert _carries_speech_she_did_not_say(entendu, DITE) is True


def test_with_nothing_to_compare_against_it_stays_out_of_the_way():
    """No TTS text means no grounds to call anything an echo. Failing
    open here loses nothing; failing shut would eat real queries."""
    assert _carries_speech_she_did_not_say("donne-moi la météo", "") is True


def test_an_empty_utterance_carries_nothing():
    assert _carries_speech_she_did_not_say("", DITE) is False
    assert _carries_speech_she_did_not_say("   ", DITE) is False


def test_it_does_not_key_on_a_language():
    """Same shape in English: her line re-heard, versus a real barge-in."""
    dite = ("I could not retrieve the weather for Bagneux, the service "
            "did not answer. Would you like me to try again in a moment?")

    assert _carries_speech_she_did_not_say(
        "could not retrieve the weather for bagneux the service", dite) is False
    assert _carries_speech_she_did_not_say(
        "forget it play some music instead", dite) is True


# ── And at the third site, where it leaked next ────────────────────────


DITE_APPLE = (
    "Pour la semaine prochaine — semaine du 17 août — Apple TV+ sort deux "
    "programmes le vendredi 21 août : la saison 5 de la série animée pour "
    "enfants Eau-Paisible, et le documentaire sportif The Dynasty : UConn "
    "Huskies sur une équipe universitaire de basket. Pas de quoi faire "
    "sauter au plafond, à moins que vous ne soyez fan de basket américain "
    "ou de leçons de vie animées."
)


def test_the_tail_left_by_the_prefix_stripper_is_not_speech():
    """Second trace, same day. The stripper cut the part it recognised and
    kept "de leçons gagnées" — her own "de leçons de vie animées", mangled.
    The judge, handed an already-cleaned fragment, read it as a query and a
    turn ran. Whatever the stripper leaves over is not speech by default."""
    assert _carries_speech_she_did_not_say(
        "de leçons gagnées.", DITE_APPLE) is False


def test_the_whole_unstripped_echo_is_not_speech_either():
    assert _carries_speech_she_did_not_say(
        "faire sauter au plafond des fans de baskets américains ou de "
        "leçons gagnées.", DITE_APPLE) is False


def test_a_real_ask_over_that_same_reply_survives():
    """The control at this site: he cuts in with something of his own."""
    assert _carries_speech_she_did_not_say(
        "non attends, plutôt la météo à Bagneux", DITE_APPLE) is True
