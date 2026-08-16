"""Nothing was looked up, so nothing was learned.

The `world` branch is fed by pages the assistant did not write and cannot
vouch for, and it had none of the provenance discipline the core files
enforce for facts about him. The extractor was asked to *guess* whether a
sentence came from a lookup:

    Heuristic: would a different assistant on a different day produce
    the same answer? If yes, it's a lookup → extract.

A confident invented statistic passes that perfectly. Observed on
2026-08-16: after one web page about model benchmarks, she restated a
figure two turns later with no tool call at all, added a verdict of her
own, and a claim of the same family reached the graph as a bare fact.

The system already knows which tools ran. `DialogueMemory._tool_turns`
holds them, timestamped, right beside the messages — excluded from the
diary so raw payloads never reach the summariser, which stays true. What
crosses is not the payload but the fact of the call.

So provenance stops being a judgement about prose and becomes a property
of the transcript: no tool in the window, no lookup, no extraction, and
no LLM call to decide otherwise.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory


# ── What the window knows ──────────────────────────────────────────────


def _appel(nom: str) -> list:
    """One reply's worth of tool messages, the shape the engine stores."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": nom, "arguments": {}}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "…"},
    ]


def test_a_window_with_no_tool_reports_none():
    memoire = DialogueMemory()
    memoire.add_message("user", "c'est quel modèle exactement ?")
    memoire.add_message("assistant", "c'est Mythos 5, 78 % sur ExploitBench.")

    assert memoire.tools_in_pending_window() == []


def test_a_window_reports_the_tool_that_ran():
    memoire = DialogueMemory()
    memoire.add_message("user", "la météo à Bagneux ?")
    memoire.record_tool_turn(_appel("getWeather"))
    memoire.add_message("assistant", "il fait 24 degrés.")

    assert memoire.tools_in_pending_window() == ["getWeather"]


def test_a_tool_from_a_window_already_written_does_not_count():
    """The diary is cumulative. A lookup that was already summarised must
    not vouch for the next window's prose."""
    memoire = DialogueMemory()
    memoire.add_message("user", "la météo ?")
    memoire.record_tool_turn(_appel("getWeather"))
    memoire.add_message("assistant", "24 degrés.")
    _, instantane = memoire.get_pending_chunks_with_snapshot()
    memoire.mark_saved_up_to(instantane)

    memoire.add_message("user", "et le modèle dont tu parlais ?")
    memoire.add_message("assistant", "Mythos 5, 78 % sur ExploitBench.")

    assert memoire.tools_in_pending_window() == []


def test_several_tools_are_all_reported_once_each():
    memoire = DialogueMemory()
    memoire.add_message("user", "cherche et donne-moi la météo")
    memoire.record_tool_turn(_appel("webSearch"))
    memoire.record_tool_turn(_appel("getWeather"))
    memoire.record_tool_turn(_appel("webSearch"))

    assert sorted(memoire.tools_in_pending_window()) == ["getWeather", "webSearch"]


# ── The rule that does the work ────────────────────────────────────────


def _extrait(outils):
    from src.jarvis.memory.graph_ops import extract_graph_memories

    cfg = MagicMock()
    with patch("src.jarvis.memory.graph_ops.call_llm_direct") as appel:
        appel.return_value = '["un fait quelconque"]'
        faits = extract_graph_memories(
            summary="Mythos 5 obtient 78 % sur ExploitBench.",
            cfg=cfg, chat_model="m", tools_used=outils,
        )
    return faits, appel.call_count


def test_a_window_without_a_tool_extracts_nothing():
    faits, appels = _extrait([])

    assert faits == []


def test_and_it_does_not_even_ask_the_model():
    """Filtering after the call would still pay for it, and would leave
    the decision to the same judgement that failed."""
    _, appels = _extrait([])

    assert appels == 0


def test_a_window_with_a_tool_still_extracts():
    """The control, and it is the one that matters: a gate that blocked
    everything would satisfy both tests above and destroy the feature."""
    faits, appels = _extrait(["webSearch"])

    assert faits == ["un fait quelconque"]
    assert appels == 1


def test_a_caller_that_cannot_establish_the_tools_still_extracts():
    """The memory viewer imports historical summaries where the tools are
    unrecoverable. Blocking those would delete a working feature to
    punish an unknown; they are marked `inconnu` instead."""
    faits, appels = _extrait(None)

    assert faits == ["un fait quelconque"]
    assert appels == 1


# ── The source vocabulary ──────────────────────────────────────────────


@pytest.mark.parametrize("outils,attendu", [
    (["webSearch"], "web"),
    (["fetchWebPage"], "web"),
    (["getWeather"], "outil"),
    (["chrome-devtools__navigate_page"], "outil"),
    (["getWeather", "webSearch"], "web"),
    (None, "inconnu"),
    ([], "inconnu"),
])
def test_the_source_follows_what_actually_ran(outils, attendu):
    """A window that touched the web is `web` even when another tool ran
    too: the weaker guarantee is the one that has to be reported."""
    from src.jarvis.memory.graph_ops import source_for_tools

    assert source_for_tools(outils) == attendu


def test_there_is_no_word_for_a_fact_the_model_invented():
    """A fact with no tool behind it does not get a weaker label. It does
    not get written, which is what the empty-window rule above enforces."""
    from src.jarvis.memory.graph_ops import SOURCES

    assert "modèle" not in SOURCES
    assert set(SOURCES) == {"web", "outil", "inconnu"}
