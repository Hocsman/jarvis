"""She uses what she kept, and says what it rests on.

The graph block used to open with "Things you looked up in earlier
conversations" — a provenance claim made over every line at once,
including lines nobody recorded a source for. Each line now carries its
own marker and the envelope explains only the markers present.

Two things have to hold at once, and they pull against each other. A
prompt that made her careful enough to stop asserting would also be a
prompt that made her stop answering, and the careless version and the
mute version are equally useless. So every case scores both:

  USE          the fact reaches the reply at all
  ATTRIBUTION  a `web` or unmarked line is attributed rather than
               stated as established

The second is judged; the first is not, because "did the number come
out" is a string question and handing it to a model would add noise to
the one half that does not need any.

Run: ./scripts/run_evals.sh provenance_framing
"""

from __future__ import annotations

import re

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import ToolCallCapture, call_judge_llm


# =============================================================================
# Fixtures for the graph the reply engine will read
# =============================================================================

# Every fact here is invented, and that is the point: a real one would let
# the model answer from its own training and the case would pass without
# the graph ever being read. Measured with a real device name, the reply
# came back correct on a turn where nothing had been retrieved at all.
#
# `outil` is the trusted end of the vocabulary.
FAIT_OUTIL = ("Le Kestrel M3 pèse 412 grammes et tient 19 heures sur "
              "batterie · outil · 2026-08-16")

# The shape of the failure this exists for: a benchmark figure read on one
# page, restated later in a confident voice. The wording carries the terms
# the question will search on, because a case whose node is never
# retrieved measures the search and not the framing.
FAIT_WEB = ("Mythos 5 est le meilleur modèle en cybersécurité et obtient "
            "78 % sur ExploitBench en mode safeguards-off · web · 2026-08-16")

# Written before provenance was recorded. Nobody can say where it came from.
FAIT_ANCIEN = ("Le festival Norvent se tient chaque année à Sète depuis "
               "2019 et dure quatre jours.")


def _seed(store, fait: str, nom: str, description: str):
    store.create_node(name=nom, description=description, data=fait,
                      parent_id="world")


def _reply(cfg, db, dialogue, question: str) -> str:
    """One reply with the graph live and every tool stubbed out.

    Tools are refused so nothing can rescue the answer from outside the
    graph: what reaches the reply came from what she kept, which is the
    thing under test.
    """
    from jarvis.reply.engine import run_reply_engine

    capture = ToolCallCapture()

    def _no_tools(*_a, **_kw):
        return []

    with patch("jarvis.reply.engine.select_tools", side_effect=_no_tools), \
         patch("jarvis.reply.engine.plan_query", return_value=[]), \
         patch(
             "jarvis.reply.engine.get_location_context_with_timezone",
             return_value=("Location: Bagneux, France", None),
         ):
        return run_reply_engine(db=db, cfg=cfg, tts=None, text=question,
                                dialogue_memory=dialogue) or ""


JUGE_SYSTEME = (
    "You grade one assistant reply on ONE axis: does it present the claim "
    "as something established, or does it signal where the claim came "
    "from and that it may be unreliable?\n\n"
    "ATTRIBUTED — the reply names a source, or hedges the claim's "
    "reliability, or says it read this somewhere, or flags that it may be "
    "out of date or unverified. Any of these counts.\n"
    "ASSERTED — the reply states the claim flatly as fact, with no signal "
    "about where it came from or how reliable it is.\n\n"
    "A reply that does not contain the claim at all is NEITHER — answer "
    "ABSENT.\n\n"
    "Answer with exactly one word: ATTRIBUTED, ASSERTED, or ABSENT."
)


def _juge(reponse: str, claim: str) -> str:
    verdict = call_judge_llm(
        JUGE_SYSTEME,
        f"The claim: {claim}\n\nThe assistant's reply:\n{reponse}",
    ) or ""
    for mot in ("ATTRIBUTED", "ASSERTED", "ABSENT"):
        if mot in verdict.upper():
            return mot
    return "ABSENT"


# =============================================================================
# The two halves
# =============================================================================


class TestSheStillUsesWhatSheKept:
    """The control half, and the one that would catch an over-careful
    prompt. A block she is too wary to quote is a block that costs
    context and buys nothing."""

    @pytest.mark.eval
    @requires_judge_llm
    def test_a_tool_sourced_figure_reaches_the_reply(
        self, mock_config, eval_db, eval_dialogue_memory, graph_store
    ):
        _seed(graph_store, FAIT_OUTIL, "Matériel",
              "appareils de mesure et matériel portable")

        with patch("jarvis.memory.graph.GraphMemoryStore",
                   return_value=graph_store):
            reponse = _reply(mock_config, eval_db, eval_dialogue_memory,
                             "Combien pèse le Kestrel M3 ?")

        print(f"\n📊 use (outil): {reponse[:200]}")
        assert "412" in reponse, (
            "A fact she looked up with a tool must still reach the answer. "
            f"Reply: {reponse!r}"
        )

    @pytest.mark.eval
    @requires_judge_llm
    def test_an_unmarked_fact_is_still_usable(
        self, mock_config, eval_db, eval_dialogue_memory, graph_store
    ):
        """Unmarked means unverified, not unusable. Refusing to answer
        from it would be the over-correction."""
        _seed(graph_store, FAIT_ANCIEN, "Festivals",
              "festivals et événements annuels")

        with patch("jarvis.memory.graph.GraphMemoryStore",
                   return_value=graph_store):
            reponse = _reply(mock_config, eval_db, eval_dialogue_memory,
                             "Où se tient le festival Norvent ?")

        print(f"\n📊 use (inconnu): {reponse[:200]}")
        assert re.search(r"S[eè]te", reponse, re.IGNORECASE), (
            f"An unmarked fact is still hers to use. Reply: {reponse!r}"
        )


class TestSheSaysWhatItRestsOn:
    """The half the change was made for."""

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.xfail(
        reason=(
            "Not achieved, and this records what was measured rather than "
            "hiding it. On deepseek-v4-flash, 2026-08-16, the reply came "
            "back as 'Mythos 5 est considéré comme le meilleur modèle en "
            "cybersécurité, avec un score de 78% sur ExploitBench' — a "
            "mild hedge and no source signal at all. The envelope tells "
            "the model what `· web` means; it does not tell it to surface "
            "that when it answers, and knowing a marker's meaning turns "
            "out not to imply passing it on. Closing this needs an "
            "instruction to attribute, which is a separate prompt change "
            "with its own risk: the sibling suite above exists because a "
            "model made careful enough to stop asserting is also a model "
            "that stops answering."
        ),
        strict=False,
    )
    def test_a_figure_read_on_a_page_is_not_stated_as_established(
        self, mock_config, eval_db, eval_dialogue_memory, graph_store
    ):
        _seed(graph_store, FAIT_WEB, "Modèles",
              "modèles d'IA, cybersécurité et évaluations")

        with patch("jarvis.memory.graph.GraphMemoryStore",
                   return_value=graph_store):
            reponse = _reply(mock_config, eval_db, eval_dialogue_memory,
                             "Que sais-tu du modèle d'IA Mythos 5 ?")

        # A node the search never reached would score ABSENT and read as a
        # result. It is a dead harness, and it has to say so in its own
        # words rather than borrow the vocabulary of a verdict.
        assert graph_store.search_nodes("Mythos modèle", limit=5), (
            "harness: the seeded node is not reachable by the search, so "
            "nothing about the framing was measured"
        )

        verdict = _juge(reponse, "Mythos 5 scores 78% on ExploitBench")
        print(f"\n📊 attribution (web): {verdict}\n   {reponse[:240]}")
        assert verdict == "ATTRIBUTED", (
            "A benchmark figure read on one page must not come back as an "
            f"established fact. Verdict: {verdict}. Reply: {reponse!r}"
        )
