"""Her question must not delete the answer to it.

The listener drops a hot-window transcript that scores at or above
`EchoDetector.PURE_ECHO_THRESHOLD` against the last thing spoken — that
is how she avoids answering her own voice. The word-count guard beside
it bounds the transcript from *above*, so it never fires for a one-word
reply, and a dropped chunk is discarded before the intent judge with
only a `🔇 Heard (echo)` line in the log.

So the wording of a spoken question is a contract with a fuzzy string
matcher. Phrase it "answer yes or no" and the answers "yes" and "no"
score 100 against it and are deleted on arrival — the user answers, and
nothing happens, and nothing says why. Measured, not assumed: that exact
phrasing does score 100.

This file exists so a copy edit that reintroduces the problem fails the
build instead of silently costing the user their answer.
"""

from __future__ import annotations

import pytest
from rapidfuzz import fuzz

from src.jarvis.listening.echo_detection import EchoDetector
from src.jarvis.tools.confirmation import SPOKEN_TEMPLATE, describe_action
from src.jarvis.tools.policy import RISK_ACTION, RISK_DESTRUCTIVE, RISK_READ

# Things a French speaker plausibly says when answering a request for
# permission. Not a lexicon the code matches on — the judge reads the
# sentence and this list never reaches production. It is the corpus the
# wording is measured against.
LIKELY_ANSWERS = [
    "oui", "non", "ouais", "vas-y", "vas-y fais-le", "ok", "d'accord",
    "oui vas-y", "non merci", "surtout pas", "laisse tomber", "annule",
    "fais-le", "ne fais pas ça", "attends", "plus tard", "non non non",
    "oui bien sûr", "évidemment", "certainement pas",
]


def _score(answer: str, question: str) -> float:
    """The comparison the listener actually performs."""
    return fuzz.partial_ratio(answer.lower(), question.lower())


@pytest.mark.parametrize("answer", LIKELY_ANSWERS)
def test_no_likely_answer_is_mistaken_for_her_own_voice(answer):
    question = describe_action("localFiles", {"operation": "read"}, RISK_ACTION).spoken

    assert _score(answer, question) < EchoDetector.PURE_ECHO_THRESHOLD, (
        f"« {answer} » scores {_score(answer, question):.0f} against « {question} » "
        f"and would be dropped as echo before anything read it"
    )


def test_there_is_headroom_rather_than_a_hair(pytestconfig):
    """Passing at 69 would be passing by accident. Whisper varies the
    transcription; the margin is what absorbs that."""
    question = describe_action("localFiles", {"operation": "read"}, RISK_ACTION).spoken
    worst = max(_score(a, question) for a in LIKELY_ANSWERS)

    assert worst <= EchoDetector.PURE_ECHO_THRESHOLD - 10, (
        f"worst answer scores {worst:.0f}, too close to "
        f"{EchoDetector.PURE_ECHO_THRESHOLD}"
    )


def test_the_phrasing_that_is_known_to_break_it_would_fail_this_test():
    """A guard on the guard: if this comparison ever stops discriminating,
    the test above passes vacuously."""
    broken = "Je veux lancer localFiles. Réponds par oui ou non."

    assert _score("oui", broken) >= EchoDetector.PURE_ECHO_THRESHOLD
    assert _score("non", broken) >= EchoDetector.PURE_ECHO_THRESHOLD


def test_the_question_states_no_decision_of_its_own():
    """Her voice comes back to her. A question containing the token the
    judge grants on could approve itself."""
    from src.jarvis.tools.confirmation import DENIED, GRANTED

    question = describe_action("localFiles", {"operation": "read"}, RISK_ACTION).spoken
    words = {w.strip(".,;:!?'\"").lower() for w in question.split()}

    assert GRANTED not in words
    assert DENIED not in words


def test_the_arguments_are_never_spoken():
    """A path read aloud is a path mis-heard. The card carries the detail;
    the voice carries the question."""
    d = describe_action("localFiles", {"path": "/Users/hocine/notes.txt"},
                        RISK_ACTION)

    assert "/Users/hocine/notes.txt" not in d.spoken


def test_the_tool_name_reaches_the_question():
    """Otherwise she is asking permission for nothing in particular."""
    assert "localFiles" in describe_action(
        "localFiles", {}, RISK_ACTION,
    ).spoken


def test_the_template_is_what_gets_spoken():
    assert describe_action("deleteMeal", {}, RISK_ACTION).spoken == \
        SPOKEN_TEMPLATE.format(tool="deleteMeal")


# ── What she says when the name has to go ─────────────────────────────


@pytest.mark.parametrize("risk", [RISK_READ, RISK_ACTION, RISK_DESTRUCTIVE])
def test_the_risk_sentence_does_not_eat_its_own_answer(risk):
    """Every longer wording measured for this collided with an answer of
    its own: "consulter" carries "on" for "non", "qui agit sur ton
    système" carries "ur" for "oui". A question that deletes the reply to
    it is not a question, so each of these is measured rather than
    trusted — a future rewording that starts eating answers fails here
    instead of doing it silently at 07:00."""
    from src.jarvis.tools.confirmation import (
        SPOKEN_ANONYME_PAR_RISQUE, _would_swallow_the_answer,
    )

    assert not _would_swallow_the_answer(SPOKEN_ANONYME_PAR_RISQUE[risk])


def test_a_name_that_would_eat_the_answer_leaves_the_risk_behind():
    """`setRoutine` carries "ou", and a "Oui." transcribed with a full
    stop scores 75 against the named sentence — over the echo threshold.
    The name goes; what the user is approving must not go with it."""
    from src.jarvis.tools.confirmation import describe_action

    said = describe_action("setRoutine", {"routine": "x"}, RISK_ACTION).spoken

    assert "setRoutine" not in said
    assert "changer" in said


def test_a_harmless_name_is_still_named():
    """Dropping the name is the exception. `webSearch` collides with
    nothing, so there is no reason to be vague about it."""
    from src.jarvis.tools.confirmation import describe_action

    assert "webSearch" in describe_action("webSearch", {}, RISK_READ).spoken


def test_the_risk_survives_a_tool_name_that_is_not_a_tool_name():
    """An MCP server names its own tools, with no character class and no
    length bound. Whatever it sent, the risk is ours."""
    from src.jarvis.tools.confirmation import describe_action

    said = describe_action("evil\n## forgé", {}, RISK_DESTRUCTIVE).spoken

    assert "forgé" not in said
    assert "irréversible" in said
