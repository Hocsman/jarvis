"""What is thrown away has to look like her voice, not just the seam.

`salvage_after_echo_tail` exists for a real case: Whisper merges the tail
of her reply and the start of his into one segment, and we want his half.
It scans right-to-left for a five-word window that resembles her last
twenty words, and discards everything before it.

The flaw is that it only ever checked the seam. Nothing checked that the
discarded prefix was her speech at all — and when he corrects her he
necessarily reuses her wording, so the winning window is found *inside
his own sentence* and the correction is what gets thrown away.

Reproduced, on the archetypal case:

    she    : "your next meeting is at four o'clock with the design team…"
    he     : "no my next meeting is at three o'clock can you check again"
    kept   : "can you check again"

The word that mattered — three — is gone. And it goes quietly: the
caller only `debug_log`s, unlike the sibling path that prints `✂️`, and
it overwrites the segment in the transcript buffer, so the original does
not survive anywhere for the intent judge to see.

The trigger upstream is purely temporal — he began speaking within the
echo tolerance of her finishing — so "there was no echo" is precisely
the case this fires on.
"""

from __future__ import annotations

import pytest


ELLE = ("your next meeting is at four o'clock with the design team "
        "in the main conference room on the second floor.")


def _detecteur(dit=ELLE):
    from src.jarvis.listening.echo_detection import EchoDetector
    d = EchoDetector()
    d._last_tts_text = dit
    return d


# ── What must survive ─────────────────────────────────────────────────


def test_a_correction_that_reuses_her_words_is_not_cut(): 
    """The whole point of the sentence is the word he changed. Cutting
    the left half throws away the correction and keeps the polite tail."""
    entendu = "no my next meeting is at three o'clock can you check again"

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert garde is None or "three" in garde, (
        f"the correction was cut: {garde!r}")


def test_a_follow_up_reusing_her_subject_is_not_cut():
    entendu = "and the design team meeting can we move it to friday"

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert garde is None or "friday" in garde, garde


def test_disagreeing_with_her_keeps_the_disagreement():
    entendu = "no not the second floor it is on the third floor"

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert garde is None or "third" in garde, garde


# ── What must still be cut ────────────────────────────────────────────


def test_her_own_words_leaking_into_the_mic_are_still_removed():
    """The case the scan exists for: the segment opens with her sentence,
    verbatim, and his question is stuck to the end of it."""
    entendu = ("in the main conference room on the second floor "
               "who else is coming")

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert garde and "coming" in garde and "conference" not in garde, garde


def test_a_mangled_echo_prefix_is_still_removed():
    """Whisper mis-hears one word of the echo, which is why the exact
    salvage paths miss it and this one exists."""
    entendu = ("in the main conference rooms on the second floor "
               "who else is coming")

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert garde and "coming" in garde and "conference" not in garde, garde


def test_the_cut_still_overshoots_by_one_word_and_that_is_known():
    """A separate, older weakness, pinned rather than left to be
    rediscovered. The scan takes the RIGHTMOST window that looks like
    her, so it eats the first word of his real speech: "who else is
    coming" comes back as "else is coming".

    Left alone deliberately. Losing an interrogative costs a shade of
    meaning the intent judge reads past, where changing which window
    wins would move every salvage in the system. It is measured here so
    that a future change to the scan has something to compare against.
    """
    entendu = ("in the main conference room on the second floor "
               "who else is coming")

    assert _detecteur().salvage_after_echo_tail(entendu) == "else is coming"


def test_speech_with_nothing_of_hers_in_it_is_untouched():
    entendu = "remind me to buy bread on the way home tonight"

    assert _detecteur().salvage_after_echo_tail(entendu) is None


def test_pure_echo_is_not_salvaged_into_a_fragment():
    entendu = "in the main conference room on the second floor"

    garde = _detecteur().salvage_after_echo_tail(entendu)

    assert not garde


# ── The property that decides it ──────────────────────────────────────


def test_the_discarded_half_must_itself_look_like_her():
    """Stated directly, because it is the rule the scan was missing. A
    seam that matches is not enough: everything to its left is about to
    be destroyed, so it has to be hers."""
    from src.jarvis.listening.echo_detection import EchoDetector

    d = _detecteur()
    # Left half is his own words; only the seam resembles hers.
    sien = "no my next meeting is at three o'clock can you check again"
    # Left half is hers, verbatim.
    sien_a_elle = ("in the main conference room on the second floor "
                   "can you check again")

    garde = d.salvage_after_echo_tail(sien_a_elle)
    assert garde and "check again" in garde
    coupe = d.salvage_after_echo_tail(sien)
    assert coupe is None or coupe == sien or "three" in coupe
