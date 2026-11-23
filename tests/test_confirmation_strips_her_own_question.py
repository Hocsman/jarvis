"""Reading a yes out of a passage that contains the question.

The spoken channel opens a twelve-second window, and Whisper hands back
the whole of it as one segment. In production that segment carried four
things: her question echoed back, the user's "Oui, je valide", and two
sentences about something else entirely.

Two filters then get it wrong in sequence. The intent judge extracts a
query from it — its normal job, finding the thing addressed to her in a
flow of speech — and picked "pourquoi", so the approval reader was handed
a word that answers nothing. And whatever it is handed still contains her
own question, which is not an approval and must not read as one.

So the answer candidate is the raw transcript, with her own words taken
back out of it first.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.confirmation import (
    SPOKEN_TEMPLATE, describe_action, strip_own_question,
)
from src.jarvis.tools.policy import RISK_ACTION


QUESTION = "Je m'apprête à changer quelque chose. Tu valides ?"


# ── Her words come back out ───────────────────────────────────────────


def test_the_answer_survives_and_the_question_does_not():
    """The real transcript, verbatim from the log."""
    entendu = (
        "Je m'apprête à changer quelque chose, tu valides ?  Oui, je valide.  "
        "Pourquoi ?  Et depuis tout à l'heure, y'a pas...  Eh, on lance un délire."
    )

    reste = strip_own_question(entendu, QUESTION)

    assert "Oui, je valide" in reste
    assert "m'apprête" not in reste


def test_a_transcript_of_nothing_but_the_echo_comes_back_empty():
    """Empty is right: nothing was said but her own voice, and the judge
    fails closed on it."""
    assert strip_own_question(QUESTION, QUESTION) == ""


def test_a_plain_answer_is_left_alone():
    assert strip_own_question("Oui, je valide.", QUESTION) == "Oui, je valide."


def test_a_refusal_is_left_alone():
    """A no dropped is worse than a yes dropped: the user said no,
    nothing recorded it, and the question expires as though they had
    ignored it."""
    assert "Non" in strip_own_question("Non, surtout pas.", QUESTION)


def test_nothing_is_stripped_when_she_said_nothing():
    assert strip_own_question("Oui", "") == "Oui"


# ── His answer taking up her question's words ─────────────────────────

# His shortest answers take up her verb with a word of his, and score as her
# question coming back. The listener lets them through as his answer; taken
# out here, what is left is empty and read as a no he never gave.
_HIS_ANSWERS_TAKING_UP_HER_VERB = [
    ("archivePhotos", "Ok lance"),
    ("archivePhotos", "Lance pas."),
    ("sortOldScans", "Valide pas."),
    ("archivePhotos", "Tu valides, ouais."),
]


@pytest.mark.parametrize("outil, answer", _HIS_ANSWERS_TAKING_UP_HER_VERB)
def test_his_answer_taking_up_her_verb_is_left_whole(outil, answer):
    from rapidfuzz import fuzz

    from src.jarvis.listening.echo_detection import EchoDetector

    dit = describe_action(outil, {}, RISK_ACTION).spoken
    assert fuzz.partial_ratio(answer.lower(), dit.lower()) >= EchoDetector.PURE_ECHO_THRESHOLD

    assert strip_own_question(answer, dit) == answer


def test_his_answer_in_another_language_is_left_whole():
    question = "I'm about to empty the recycle bin. Do you approve?"

    assert strip_own_question("Sure, approve.", question) == "Sure, approve."


def test_her_question_then_his_answer_keeps_only_his_answer():
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken

    assert strip_own_question(f"{dit} Lance pas.", dit) == "Lance pas."


@pytest.mark.parametrize("entendu", [
    "Lance.",
    "Je m'apprête à lancer archive photos, tu valides ?",
    "je m'apprête à lancer archivePhotos. tu valides ?",
], ids=["her-verb-alone", "as-whisper-writes-it-heard", "lower-case"])
def test_words_that_are_all_hers_still_come_out(entendu):
    """Not one word of his: her question coming back, whole or in part, as
    Whisper writes it heard. A short answer made only of her words is taken
    out with it, the cost the listener keeps too."""
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken

    assert strip_own_question(entendu, dit) == ""


# ── Against the sentence she actually says ────────────────────────────


@pytest.mark.parametrize("outil", ["setRoutine", "webSearch", "localFiles"])
def test_it_works_against_the_real_spoken_question(outil):
    dit = describe_action(outil, {}, RISK_ACTION).spoken
    entendu = f"{dit} Oui vas-y."

    reste = strip_own_question(entendu, dit)

    assert "Oui vas-y" in reste
    assert "valides" not in reste


# ── The whole road, from the microphone to the judge ──────────────────


def test_the_answer_is_read_from_what_was_heard_not_from_the_query():
    """The intent judge's job is finding the thing addressed to her in a
    flow of speech. When a question is waiting there is no request to
    find — there is a reply to read — and in production it extracted
    "pourquoi" from a segment that also contained "Oui, je valide"."""
    from unittest.mock import MagicMock, patch

    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import settle_pending_confirmation
    from src.jarvis.tools.confirmation import GRANTED, PendingAction

    memory = DialogueMemory()
    # Asked on one turn, answered on the next: the claim is only granted
    # to the turn immediately after the question.
    pose = memory.begin_turn("voix")
    memory.raise_pending(PendingAction.create(
        tool="setRoutine", args={}, risk=RISK_ACTION, channel="parole",
        origin="voix", query_redacted="tous les matins", raised_at_turn=pose,
        ttl_sec=180.0,
    ))
    memory.begin_turn("voix")
    lu = {}

    def _judge(cfg, utterance):
        lu["utterance"] = utterance
        return GRANTED

    cfg = MagicMock()
    with patch("src.jarvis.reply.engine.read_approval", side_effect=_judge), \
         patch("src.jarvis.reply.engine.utterance_channel_available",
               return_value=True):
        settled = settle_pending_confirmation(
            cfg=cfg, dialogue_memory=memory, origin="voix",
            utterance=(
                "Je m'apprête à changer quelque chose, tu valides ?  "
                "Oui, je valide.  Pourquoi ?"
            ),
        )

    assert "Oui, je valide" in lu["utterance"]
    assert "m'apprête" not in lu["utterance"]
    assert settled.approval is not None


def test_his_refusal_taking_up_her_verb_reaches_the_reader():
    """Taken out as her echo, his refusal would leave nothing, and nothing is
    read as a no he did not give in those words. It is his words the reader
    must be given."""
    from unittest.mock import MagicMock, patch

    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import settle_pending_confirmation
    from src.jarvis.tools.confirmation import DENIED, PendingAction

    memory = DialogueMemory()
    pose = memory.begin_turn("voix")
    memory.raise_pending(PendingAction.create(
        tool="archivePhotos", args={}, risk=RISK_ACTION, channel="parole",
        origin="voix", query_redacted="range les photos", raised_at_turn=pose,
        ttl_sec=180.0,
    ))
    memory.begin_turn("voix")
    lu = {}

    def _judge(cfg, utterance):
        lu["utterance"] = utterance
        return DENIED

    with patch("src.jarvis.reply.engine.read_approval", side_effect=_judge), \
         patch("src.jarvis.reply.engine.utterance_channel_available",
               return_value=True):
        settle_pending_confirmation(
            cfg=MagicMock(), dialogue_memory=memory, origin="voix",
            utterance="Lance pas.",
        )

    assert lu["utterance"] == "Lance pas."
