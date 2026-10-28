"""
End-to-end eval — she reads the line that answers, not the first lines.

A graph node reaches the prompt within a fixed budget of characters. A
node holds up to `SPLIT_THRESHOLD` tokens, some twenty times that budget,
before it splits into smaller ones, and more when its split keeps
failing (on the small tier, a split that would lose facts is refused).
The search finds such a node for any question its text answers; what
decides whether the answer reaches her is which of its lines are shown.

Each case seeds one large node of invented lookups, the answer deep inside
it, and asks about that answer plainly. The subjects are invented, so no
model prior supplies it, and every tool answers "No results found", so the
network cannot either. A reply carrying the answer read it from the node.

Measured on `gemma4:e2b`, interleaved call by call, 7 runs a case: shown
the node's first 300 characters, no reply carried the answer (0 of 21);
shown its lines that name the search, 13 of 21 did. Most of the misses
quote another part of the right line, so the line reached the prompt and
the small model trusted its empty web search over it, the failure
`test_the_graph_answers_before_the_web.py` measures on its own.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh the_graph_reads_a_large_node
"""

import re
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import ToolCallCapture, assert_not_fallback_reply, JUDGE_MODEL


# Invented lookups, worded apart, none of which answers a case below.
FILLER_LOOKUPS = [
    "The Corvid R7 router supports up to 14 devices on its guest network.",
    "Harrowgate Lido closes for maintenance every second Tuesday.",
    "The Quillon desk lamp has three colour temperatures.",
    "Tamsin Okafor's novel The Salt Ledger was shortlisted for the Brennan Prize.",
    "The Vey Street bakery sells rye loaves only on Fridays.",
    "The Ostrand Arena box office opens two hours before each match.",
    "The Pellin S2 e-reader stores roughly 6,000 books.",
    "Marlow Fen nature reserve has a bird hide on its eastern path.",
    "The Brisk Ferry crosses from Halden Quay every forty minutes.",
    "The Ardent 40 kettle boils a litre in about three minutes.",
    "The Gilded Pike pub serves its last orders at half past ten.",
    "Calder Mills Museum lets visitors operate a working loom.",
    "The Nimbus Hollow campsite takes bookings from March onwards.",
    "Rennick & Daughters repair vintage typewriters by appointment.",
    "The Talvi 3 headphones fold flat into their case.",
    "Ashcombe Library lends telescopes to its members.",
    "The Fairholt Market antiques stalls trade on the first Sunday of the month.",
    "The Lumen Tram line runs every eight minutes at peak times.",
    "Orrin Vale vineyard offers tours on Saturday afternoons.",
    "The Sable Court cinema shows subtitled films on Wednesdays.",
    "The Wexmoor half marathon starts at the old mill.",
    "The Cobalt V9 drill has a two-speed gearbox.",
    "Pennick Bay's lifeguards patrol from May to September.",
    "The Thistledown cafe roasts its beans on site.",
    "Hollin Tor is the highest point of the Brackwater hills.",
    "The Juno Arc bike light charges over USB-C.",
    "The Wren Street theatre seats around 180 people.",
    "Glasshouse Row's gardens close at dusk in winter.",
    "The Marten Q1 projector throws a 100-inch picture from three metres.",
    "Elsworth Pools offer an adults-only session on Thursday evenings.",
]


@dataclass
class LargeNodeCase:
    text: str
    answer_line: str
    # One of these must appear in the reply (compared squashed).
    expect: Tuple[str, ...]


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


CASES = [
    pytest.param(
        LargeNodeCase(
            text="Jarvis, how much memory does the Kestrel M3 have?",
            answer_line="The Kestrel M3 is a compact desktop workstation with 96 GB of unified memory.",
            expect=("96", "ninety-six"),
        ),
        id="A specification deep in a large node",
    ),
    pytest.param(
        LargeNodeCase(
            text="Jarvis, tu te souviens qui a réalisé Le Chant des Salines ?",
            answer_line="Le Chant des Salines is a 2019 documentary directed by Amara Belhaj.",
            expect=("belhaj",),
        ),
        id="A film credit deep in a large node, asked in another language",
    ),
    pytest.param(
        LargeNodeCase(
            text="Jarvis, remind me what Café Rouvière is known for?",
            answer_line="Cafe Rouviere is a coffee roaster in Lyon known for its Yirgacheffe filter.",
            expect=("yirgacheffe",),
        ),
        id="A local detail deep in a large node",
    ),
]


def node_data(answer_line: str) -> str:
    """The large node, the answer two thirds of the way down, every line
    with a provenance suffix as the graph stores it."""
    lines = list(FILLER_LOOKUPS)
    lines.insert(2 * len(lines) // 3, answer_line)
    return "\n".join(f"{line} · web · 2026-09-{1 + i % 28:02d}" for i, line in enumerate(lines))


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=True, reply_text="No results found.")

    return _runner


def ask(mock_config, eval_db, eval_dialogue_memory, tmp_path, case: LargeNodeCase):
    """Seed the large node under the World branch and ask the case's question."""
    from jarvis.memory.graph import BRANCH_WORLD, GraphMemoryStore
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.db_path = str(tmp_path / "jarvis.db")
    mock_config.location_enabled = False
    mock_config.memory_enrichment_source = "graph"
    mock_config.planner_enabled = False

    store = GraphMemoryStore(mock_config.db_path)
    try:
        store.update_node(BRANCH_WORLD, data=node_data(case.answer_line))
        assert store.get_node(BRANCH_WORLD).data.index(case.answer_line) > 1000, (
            "the answer must sit far past the start of the node")
    finally:
        store.close()

    capture = ToolCallCapture()
    with patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_make_runner(capture)):
        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None,
            text=case.text, dialogue_memory=eval_dialogue_memory,
        )

    print(f"\n  Large node ({JUDGE_MODEL}): {case.text}")
    print(f"  Tools called: {capture.tool_names()}")
    print(f"  Response: {(response or '')[:300]}")
    return response, capture


@pytest.mark.eval
@requires_judge_llm
class TestTheGraphReadsALargeNode:
    """The line of a large node that answers the question reaches the reply."""

    @pytest.mark.parametrize("case", CASES)
    def test_the_answer_deep_in_a_large_node_reaches_the_reply(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: LargeNodeCase,
    ):
        response, capture = ask(mock_config, eval_db, eval_dialogue_memory, tmp_path, case)

        assert_not_fallback_reply(response, context="large-node")
        assert any(_squash(w) in _squash(response) for w in case.expect), (
            f"The reply carries none of {case.expect!r}, and no tool supplied it, "
            f"so the line of the node that answers never reached the prompt. "
            f"Tools called: {capture.tool_names()}. Response: {(response or '')[:400]}"
        )
