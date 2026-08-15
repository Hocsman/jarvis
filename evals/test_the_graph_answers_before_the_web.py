"""
End-to-end eval — she reads what she already looked up.

The graph is where a lookup goes to be kept: a film, a place, a piece of
hardware, whatever she searched for once so she does not have to search
for it again. The core holds the user; the graph holds the world.

The failure this catches is the quiet kind. Nothing errors, nothing looks
wrong in the reply, and the answer is even correct — she simply pays a
web search and a page fetch to re-learn a sentence already sitting in her
own database. On a metered provider that is money; on a slow link it is
seconds; and neither shows up anywhere a user would look.

It went unnoticed for weeks because the door in front of the graph asked
the memory extractor for *implicit personal questions about the user*,
and the user had been moved out of the graph into the core. The key kept
asking about the person, the room held only the world, and the branch was
never entered. What addresses a world-fact index is the subject of the
query, so that is what opens it now.

Each case seeds a fact into the graph and asks about it plainly. The
subjects are invented, so no model prior can supply the answer, and every
tool answers "No results found", so the network cannot either. A reply
carrying the fact can only have read it from the graph.

There are two doors between the question and the graph, and this file
tests them apart because only one of them is closed. The gate below the
planner is the one this eval was written for. Above it sits the planner,
which decides whether memory is consulted at all, and which is measured
separately at the bottom of this file — with the plan it produces, she
still goes out to the network for things she already knows.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh the_graph_answers_before_the_web
"""

import re

from dataclasses import dataclass
from typing import Tuple
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    JUDGE_MODEL,
)


# Tools that mean she went out to the network rather than reading what
# she had. Calling any of them is the defect, whatever the reply says.
LOOKUP_TOOLS = {"webSearch", "fetchWebPage"}


@dataclass
class RecallCase:
    text: str
    # (node name, what she noted when she looked it up)
    node: Tuple[str, str]
    # One of these must appear in the reply, compared with `_squash` so a
    # clock written "6 p.m." and one written "18:00" both count. What is
    # under test is which channel the fact came from, not the typography
    # the model reached for.
    expect: Tuple[str, ...]


def _squash(text: str) -> str:
    """Lowercase and drop spacing and punctuation, so formatting cannot
    fail an assertion about content."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


RECALL_CASES = [
    pytest.param(
        RecallCase(
            text="Jarvis, how much memory does the Kestrel M3 have?",
            node=(
                "Kestrel M3",
                "The Kestrel M3 is a compact desktop workstation with 96 GB of "
                "unified memory and a 2 TB soldered drive.",
            ),
            expect=("96",),
        ),
        id="Hardware she looked up once is not looked up twice",
    ),
    pytest.param(
        RecallCase(
            text="Jarvis, tu te souviens qui a réalisé Le Chant des Salines ?",
            node=(
                "Le Chant des Salines",
                "Le Chant des Salines is a 2019 documentary directed by Amara "
                "Belhaj, filmed in the Camargue salt flats.",
            ),
            expect=("belhaj",),
        ),
        id="A film credit survives into another language",
    ),
    pytest.param(
        RecallCase(
            text="Jarvis, remind me what Café Rouvière is known for?",
            node=(
                "Cafe Rouviere",
                "Cafe Rouviere is a coffee roaster in Lyon, known for its "
                "Yirgacheffe filter and for closing on Sundays.",
            ),
            expect=("yirgacheffe",),
        ),
        id="A local detail she noted is recalled rather than re-searched",
    ),
]


def _make_runner(capture: ToolCallCapture):
    """Answers every tool truthfully, so a reply built on a web search is
    indistinguishable from one built on memory — except in the capture."""
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=True, reply_text="No results found.")

    return _runner


def _ask(mock_config, eval_db, eval_dialogue_memory, tmp_path,
         case: "RecallCase", *, planner: bool):
    """Seed the graph with the case's fact and ask its question."""
    from jarvis.memory.graph import GraphMemoryStore
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.db_path = str(tmp_path / "jarvis.db")
    mock_config.location_enabled = False
    mock_config.memory_enrichment_source = "graph"
    mock_config.planner_enabled = planner

    store = GraphMemoryStore(mock_config.db_path)
    try:
        name, data = case.node
        store.create_node(
            name=name,
            description=f"What is known about {name}",
            data=data,
            parent_id=store.get_root().id,
        )
    finally:
        store.close()

    capture = ToolCallCapture()
    with patch(
        "jarvis.reply.engine.run_tool_with_retries",
        side_effect=_make_runner(capture),
    ):
        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None,
            text=case.text,
            dialogue_memory=eval_dialogue_memory,
        )

    print(f"\n  Stored lookup ({JUDGE_MODEL}, planner={planner}): {case.text}")
    print(f"  Graph holds: {case.node[1]}")
    print(f"  Tools called: {capture.tool_names()}")
    print(f"  Response: {(response or '')[:300]}")
    return response, capture


@pytest.mark.eval
@requires_judge_llm
class TestTheGraphAnswersBeforeTheWeb:
    """What she has already looked up reaches the prompt on its own."""

    @pytest.mark.parametrize("case", RECALL_CASES)
    def test_a_stored_lookup_reaches_the_reply(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: RecallCase,
    ):
        """The gate itself, isolated.

        The planner is out of the way here so enrichment always runs and
        what is measured is the gate alone. Every tool returns nothing,
        so the fact can only have arrived through the graph.
        """
        response, capture = _ask(mock_config, eval_db, eval_dialogue_memory,
                                 tmp_path, case, planner=False)

        assert_not_fallback_reply(response, context="graph-gate")

        assert any(_squash(w) in _squash(response) for w in case.expect), (
            f"The reply carries none of {case.expect!r}, and no tool supplied "
            f"it either, so the stored lookup never reached the prompt. Graph "
            f"holds: {case.node[1]!r}. Tools called: {capture.tool_names()}. "
            f"Response: {(response or '')[:400]}"
        )


# The two cases the deterministic drop reaches: a step is dropped only
# when every word it would look up is already in the memory that reached
# the prompt, so a step and a note written in the same language can meet
# and a step written in another cannot.
_COVERED_DETERMINISTICALLY = {
    "Hardware she looked up once is not looked up twice",
    "A local detail she noted is recalled rather than re-searched",
}

_NETWORK_CASES = [
    p if p.id in _COVERED_DETERMINISTICALLY else pytest.param(
        *p.values,
        id=p.id,
        marks=pytest.mark.xfail(
            reason="A step is dropped only when its own argument words are "
                   "present in the memory that reached the prompt. This step "
                   "is written in the user's language and the note in the "
                   "extractor's, so no lexical rule can match them, and "
                   "whether it searches is left to the model.",
            strict=False,
        ),
    )
    for p in RECALL_CASES
]


@pytest.mark.eval
@requires_judge_llm
class TestSheDoesNotPayTwice:
    """The door above the gate.

    Enrichment used to run only when the planner asked for `searchMemory`,
    and the planner reads a factual question as an errand for the network.
    Two things close that. The graph is opened by the concrete arguments
    the planner composed into its own steps, which name the subject and
    exist whether or not memory was planned. And a read-only step whose
    every word is already answered by the memory in front of the model is
    dropped before the plan is injected.

    What is left is the cross-lingual case named in its mark: deciding
    that an English note answers a French question is a semantic
    judgement, and no lexical rule makes it. The fact does reach the
    prompt; whether she searches anyway is the model's call.
    """

    @pytest.mark.parametrize("case", _NETWORK_CASES)
    def test_she_does_not_go_out_for_something_she_has(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: RecallCase,
    ):
        response, capture = _ask(mock_config, eval_db, eval_dialogue_memory,
                                 tmp_path, case, planner=True)

        went_out = LOOKUP_TOOLS.intersection(capture.tool_names())
        assert not went_out, (
            f"She called {sorted(went_out)} for something already in her own "
            f"graph: {case.node[1]!r}. The reply may still be correct, which is "
            f"why this is worth pinning — the cost is a network round trip "
            f"nobody sees. Tools called: {capture.tool_names()}"
        )

    def test_she_still_goes_out_for_the_half_she_does_not_have(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
    ):
        """Coverage has to be total or the rule cancels the wrong search.

        A note about the machine's memory says nothing about its price.
        This is the test that fails first if anyone loosens full coverage
        into partial overlap, and it is worth more than the ones above:
        a saved round trip is money, a suppressed one is a wrong answer.
        """
        case = RecallCase(
            text="Jarvis, what does the Kestrel M3 sell for these days?",
            node=(
                "Kestrel M3",
                "The Kestrel M3 is a compact desktop workstation with 96 GB of "
                "unified memory and a 2 TB soldered drive.",
            ),
            expect=(),
        )

        _, capture = _ask(mock_config, eval_db, eval_dialogue_memory,
                          tmp_path, case, planner=True)

        assert LOOKUP_TOOLS.intersection(capture.tool_names()), (
            "She answered about a price from a note that only records the "
            "machine's memory and storage. Standing in for a lookup is only "
            "safe when the stored text answers the whole question; here it "
            f"answers none of it. Tools called: {capture.tool_names()}"
        )
