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

    reste = strip_own_question(entendu, QUESTION, action="")

    assert "Oui, je valide" in reste
    assert "m'apprête" not in reste


def test_a_transcript_of_nothing_but_the_echo_comes_back_empty():
    """Empty is right: nothing was said but her own voice, and it reads as
    unclear without the judge being asked."""
    assert strip_own_question(QUESTION, QUESTION, action="") == ""


def test_a_plain_answer_is_left_alone():
    assert strip_own_question("Oui, je valide.", QUESTION, action="") == "Oui, je valide."


def test_a_refusal_is_left_alone():
    """A no dropped is worse than a yes dropped: the user said no,
    nothing recorded it, and the question expires as though they had
    ignored it."""
    assert "Non" in strip_own_question("Non, surtout pas.", QUESTION, action="")


def test_nothing_is_stripped_when_she_said_nothing():
    assert strip_own_question("Oui", "", action="") == "Oui"


# ── His answer taking up her question's words ─────────────────────────

_EN_QUESTION = "I'm about to empty the recycle bin. Do you approve?"


def _scores_as_her_question(heard: str, dit: str) -> bool:
    from rapidfuzz import fuzz

    from src.jarvis.listening.echo_detection import EchoDetector

    return fuzz.partial_ratio(heard.lower(), dit.lower()) >= EchoDetector.PURE_ECHO_THRESHOLD


# His shortest answers take up her verb with a word of his, and score as her
# question coming back. Taken out whole, nothing of his would be left to
# read, his "pas" with the rest; kept whole, her own words would be read with
# them. The reader is given his words: from the first word she did not say to
# the last.
_HIS_ANSWERS_TAKING_UP_HER_VERB = [
    ("archivePhotos", "Ok lance", "Ok"),
    ("archivePhotos", "Lance pas.", "pas"),
    ("sortOldScans", "Valide pas.", "pas"),
    ("archivePhotos", "Tu valides, ouais.", "ouais"),
]


@pytest.mark.parametrize("outil, answer, his", _HIS_ANSWERS_TAKING_UP_HER_VERB)
def test_his_answer_taking_up_her_verb_keeps_his_words(outil, answer, his):
    dit = describe_action(outil, {}, RISK_ACTION).spoken
    assert _scores_as_her_question(answer, dit)

    assert strip_own_question(answer, dit, action=outil) == his


def test_his_answer_in_another_language_keeps_his_words():
    assert _scores_as_her_question("Sure, approve.", _EN_QUESTION)

    assert strip_own_question("Sure, approve.", _EN_QUESTION, action="") == "Sure"


def test_her_question_then_his_answer_keeps_only_his_words():
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken

    assert strip_own_question(f"{dit} Lance pas.", dit, action="archivePhotos") == "pas"


@pytest.mark.parametrize("entendu", [
    # Her question and his answer with no sentence mark between them.
    describe_action("archivePhotos", {}, RISK_ACTION).spoken.rstrip(" ?").replace(".", ",")
    + ", ouais.",
    # Her tool name as Whisper writes it heard, then his word.
    "archive photos tu valides ouais",
], ids=["commas-between", "no-marks"])
def test_her_question_and_his_answer_in_one_fragment_keep_his_answer(entendu):
    """Taken out as a sentence of hers, the fragment would leave nothing
    of his to read."""
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken
    assert _scores_as_her_question(entendu, dit)

    assert strip_own_question(entendu, dit, action="archivePhotos") == "ouais"


@pytest.mark.parametrize("entendu, left", [
    ("Lancer archive fotos. Tu valides ? Hmm.", "fotos Hmm."),
    ("Je m'apprête à lancer archive fotos tu valides", "fotos"),
], ids=["her-verb-and-name", "her-whole-question"])
def test_her_question_coming_back_bent_leaves_none_of_her_words(entendu, left):
    """Whisper bends a word of hers into one she did not say ("fotos" for
    "photos"). Her words around it are not his and never reach the reader,
    where her own "lancer …" could read as a go-ahead."""
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken
    assert _scores_as_her_question(entendu.split(".")[0], dit)

    assert strip_own_question(entendu, dit, action="archivePhotos") == left


def test_his_refusal_in_front_of_her_words_keeps_his_refusal():
    """His no before her words, then a word of assent: the no stays, rather
    than leaving the assent alone to be read."""
    dit = describe_action("sortOldScans", {}, RISK_ACTION).spoken
    assert _scores_as_her_question("Pas à lancer sortOldScans.", dit)

    assert strip_own_question("Pas à lancer sortOldScans. Ok.", dit, action="sortOldScans") == "Pas Ok."


@pytest.mark.parametrize("entendu, dit, outil", [
    ("Lance.", describe_action("archivePhotos", {}, RISK_ACTION).spoken, "archivePhotos"),
    ("Je m'apprête à lancer archive photos, tu valides ?",
     describe_action("archivePhotos", {}, RISK_ACTION).spoken, "archivePhotos"),
    (describe_action("archivePhotos", {}, RISK_ACTION).spoken.lower(),
     describe_action("archivePhotos", {}, RISK_ACTION).spoken, "archivePhotos"),
    ("Approve.", _EN_QUESTION, ""),
], ids=["her-verb-alone", "as-whisper-writes-it-heard", "lower-case", "english-verb-alone"])
def test_words_that_are_all_hers_still_come_out(entendu, dit, outil):
    """Not one word of his, in her order: her question coming back, whole or
    in part, as Whisper writes it heard. A short answer made only of her
    words in her order is taken out with it, the cost the listener keeps
    too."""
    from src.jarvis.listening.echo_detection import comes_back_in_her_order

    assert comes_back_in_her_order(entendu, dit)

    assert strip_own_question(entendu, dit, action=outil) == ""


@pytest.mark.parametrize("entendu, dit, outil", [
    ("Je valide.", describe_action("archivePhotos", {}, RISK_ACTION).spoken, "archivePhotos"),
    ("Je lance.", describe_action("sortOldScans", {}, RISK_ACTION).spoken, "sortOldScans"),
    ("I approve.", _EN_QUESTION, ""),
], ids=["je-valide", "je-lance", "english"])
def test_his_answer_of_her_words_out_of_her_order_is_kept_whole(entendu, dit, outil):
    """Every word hers, a few of them picked out of her order, none naming
    what she is about to do: his answer picking her words out of her
    question, which her voice coming back never does. It reaches the reader
    as he said it."""
    from src.jarvis.listening.echo_detection import (
        carries_a_word_she_did_not_say, picks_her_words_out,
    )

    assert _scores_as_her_question(entendu, dit)
    assert not carries_a_word_she_did_not_say(entendu, dit)
    assert picks_her_words_out(entendu, dit, action=outil)

    assert strip_own_question(entendu, dit, action=outil) == entendu


def test_her_question_then_his_answer_of_her_words_keeps_his_answer():
    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken

    assert strip_own_question(f"{dit} Je valide.", dit, action="archivePhotos") == "Je valide."


@pytest.mark.parametrize("entendu", [
    "Je à lancer archive photos.",
    "Je lancer archive photos.",
    "Lancer archive photos lancer.",
    "Je lancer archive.",
], ids=["her-two-word-verb-dropped", "two-words-dropped", "repeated", "short-naming-her-action"])
def test_her_question_coming_back_out_of_her_order_still_comes_out(entendu):
    """Whisper drops or repeats words of her question as it comes back, and
    her order breaks; most of it is still there, the name of what she is
    about to do with it. Kept whole, her own "lancer archive photos" would
    be read as his go-ahead."""
    from src.jarvis.listening.echo_detection import comes_back_in_her_order

    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken
    assert _scores_as_her_question(entendu, dit)
    assert not comes_back_in_her_order(entendu, dit)

    assert strip_own_question(entendu, dit, action="archivePhotos") == ""


# ── Against the sentence she actually says ────────────────────────────


@pytest.mark.parametrize("outil", ["setRoutine", "webSearch", "localFiles"])
def test_it_works_against_the_real_spoken_question(outil):
    dit = describe_action(outil, {}, RISK_ACTION).spoken
    entendu = f"{dit} Oui vas-y."

    reste = strip_own_question(entendu, dit, action=outil)

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
    """Taken out as her echo, his refusal would leave nothing of his for the
    reader, his "pas" with it. It is his words the reader is given."""
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

    assert lu["utterance"] == "pas"


def test_an_answer_left_empty_by_the_cut_is_not_a_refusal():
    """Her verb alone cannot be told from her echo, so the cut leaves nothing.
    Nothing is not a no: the question is over and the turn is answered as an
    ordinary one, rather than spoken back as "understood, I won't" and kept
    as `décliné` in his name."""
    from unittest.mock import MagicMock, patch

    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import DECLINED_REPLY, settle_pending_confirmation
    from src.jarvis.tools.confirmation import PendingAction

    memory = DialogueMemory()
    pose = memory.begin_turn("voix")
    memory.raise_pending(PendingAction.create(
        tool="archivePhotos", args={}, risk=RISK_ACTION, channel="parole",
        origin="voix", query_redacted="range les photos", raised_at_turn=pose,
        ttl_sec=180.0,
    ))
    memory.begin_turn("voix")

    with patch("src.jarvis.tools.confirmation._ask_model") as asked, \
         patch("src.jarvis.reply.engine.utterance_channel_available",
               return_value=True):
        settled = settle_pending_confirmation(
            cfg=MagicMock(), dialogue_memory=memory, origin="voix",
            utterance="Lance.",
        )

    assert asked.called is False
    assert settled.approval is None
    assert settled.declined is False
    assert settled.reply != DECLINED_REPLY
    assert memory.peek_pending() is None


def test_the_log_says_a_fragment_of_her_words_was_kept_for_their_order():
    from unittest.mock import patch

    dit = describe_action("archivePhotos", {}, RISK_ACTION).spoken
    with patch("src.jarvis.debug.debug_log") as logged:
        strip_own_question("Je valide.", dit, action="archivePhotos")

    lines = [str(call.args[0]) for call in logged.call_args_list]
    assert any("kept whole" in line and "out of her order" in line for line in lines), lines
