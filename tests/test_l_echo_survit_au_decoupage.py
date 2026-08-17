"""The echo reference is the reply, never the chunk.

`EchoDetector.track_tts_start(text)` replaces `_last_tts_text`. Speaking a
reply sentence by sentence through that call would leave it holding only
the last sentence spoken, and everything downstream reads it: the fuzzy
`partial_ratio` net, the intent judge that receives it as context, and
`_carries_speech_she_did_not_say`, which decides whether to override an
echo verdict and whether a stripped remainder is speech.

An echo of the first sentence, reaching the microphone while the third is
being spoken, would then be compared against the third, match nothing,
and be taken for the user. She would answer her own earlier sentences —
the failure fixed on 2026-08-16, reintroduced by the feature that
followed it.

So one reply accumulates, and a new reply starts over.
"""

from __future__ import annotations

import pytest

from src.jarvis.listening.echo_detection import EchoDetector
from src.jarvis.listening.listener import _carries_speech_she_did_not_say


UNE = "Il fait beau à Bagneux aujourd'hui."
DEUX = "La température atteint vingt-quatre degrés."
TROIS = "Prends une veste quand même."


def _detecteur_ayant_dit(*morceaux):
    d = EchoDetector()
    for i, m in enumerate(morceaux):
        d.track_tts_start(m, continues=i > 0)
    return d


def test_the_reference_holds_every_chunk_of_the_reply():
    d = _detecteur_ayant_dit(UNE, DEUX, TROIS)

    for morceau in (UNE, DEUX, TROIS):
        assert morceau.lower().strip(" .") in d._last_tts_text


def test_an_echo_of_the_first_sentence_is_still_recognised():
    """The failure this exists to prevent, stated as a behaviour: the
    guard added on 2026-08-16 must still see the first sentence while the
    third is playing."""
    d = _detecteur_ayant_dit(UNE, DEUX, TROIS)

    assert _carries_speech_she_did_not_say(
        "il fait beau à bagneux aujourd'hui", d._last_tts_text) is False


def test_a_real_interruption_is_still_speech():
    """The control. An accumulating reference must not swallow him."""
    d = _detecteur_ayant_dit(UNE, DEUX, TROIS)

    assert _carries_speech_she_did_not_say(
        "non attends, mets plutôt de la musique", d._last_tts_text) is True


def test_a_new_reply_starts_over():
    """Without the reset the reference grows for the life of the process,
    and eventually everything he says resembles something she once said."""
    d = _detecteur_ayant_dit(UNE, DEUX, TROIS)

    d.track_tts_start("Je ne sais pas.", continues=False)

    assert "bagneux" not in d._last_tts_text
    assert "je ne sais pas" in d._last_tts_text


def test_the_unstreamed_call_behaves_exactly_as_before():
    """One call, no flag: the existing path is untouched."""
    d = EchoDetector()

    d.track_tts_start("Il fait beau.")

    assert d._last_tts_text == "il fait beau."


def test_the_start_time_is_the_first_chunk_not_the_last():
    """Echo windows are timed from when she started speaking. Restamping
    on every chunk would keep pushing the window forward and let a late
    echo look like fresh speech."""
    import time

    d = EchoDetector()
    d.track_tts_start(UNE)
    debut = d._tts_start_time
    time.sleep(0.02)
    d.track_tts_start(DEUX, continues=True)

    assert d._tts_start_time == debut
