"""Running a tool is not the same as having looked something up.

The rule shipped earlier says: a window with no tool holds no lookup, so
it yields no world facts. Correct as far as it went, and then the field
showed what it missed.

Trace (2026-08-17). The only tool of the turn was `toolSearchTool`, which
returns a list of tool *names* — it consults nothing. Under the rule as
written the window counted as tool-bearing, extraction was allowed, and
anything the model had said could have been written to `world` labelled
`outil`. Nothing broke that day only because the extractor refused the
content for an unrelated reason: the conversation was about her own
capabilities.

The same hole covers `remember`, `forget`, `logMeal`, `setReminder`,
`setGoal` — those write. And `fetchMeals`, `listGoals`,
`reviewLearnings` — those read *his* data, which is not the world.

Which is the very error the rule exists to prevent, committed inside the
fix against it: treating "a tool ran" as "something was learned".

The direction of failure is chosen. An unknown name — every MCP server's
tools — counts as a lookup, because an MCP tool usually is one and
excluding them would silently drop real facts. Being wrong there costs a
mislabelled source; being wrong the other way costs his memory.
"""

from __future__ import annotations

import pytest

from src.jarvis.memory.provenance import (
    SOURCE_TOOL,
    SOURCE_UNKNOWN,
    SOURCE_WEB,
    lookup_tools,
    source_for_tools,
)


# ── Ce qui consulte ────────────────────────────────────────────────────


@pytest.mark.parametrize("nom", ["webSearch", "fetchWebPage", "getWeather",
                                 "screenshot", "localFiles"])
def test_a_tool_that_brings_something_in_counts(nom):
    assert lookup_tools([nom]) == [nom]


def test_an_mcp_tool_counts():
    """Unknown names count. An MCP tool usually does look something up,
    and dropping them all would lose real facts to protect against a
    mislabel."""
    assert lookup_tools(["chrome-devtools__navigate_page"]) == [
        "chrome-devtools__navigate_page"]


# ── Ce qui ne consulte pas ─────────────────────────────────────────────


@pytest.mark.parametrize("nom", [
    "toolSearchTool",   # rend des noms d'outils
    "stop",             # sentinelle de fin
    "refreshMCPTools",  # recharge un catalogue
    "remember", "forget",                      # écrivent le noyau
    "logMeal", "deleteMeal",                   # écrivent
    "setReminder", "setRoutine", "cancelRoutine",
    "setGoal", "noteGoal", "closeGoal",
    "fetchMeals", "listGoals", "reviewLearnings",  # lisent ses données
])
def test_a_tool_that_writes_or_reads_his_own_data_does_not_count(nom):
    assert lookup_tools([nom]) == []


def test_the_trace_that_prompted_this():
    """The turn as it happened: one tool, and it consulted nothing."""
    assert lookup_tools(["toolSearchTool"]) == []
    assert source_for_tools(lookup_tools(["toolSearchTool"])) == SOURCE_UNKNOWN


# ── Mélanges ───────────────────────────────────────────────────────────


def test_a_lookup_beside_a_write_still_counts():
    assert lookup_tools(["remember", "webSearch", "stop"]) == ["webSearch"]


def test_the_source_follows_the_lookups_only():
    assert source_for_tools(lookup_tools(["remember", "getWeather"])) == SOURCE_TOOL
    assert source_for_tools(lookup_tools(["stop", "webSearch"])) == SOURCE_WEB


def test_nothing_in_means_nothing_out():
    assert lookup_tools([]) == []
    assert lookup_tools(None) == []


# ── La porte de l'extraction ───────────────────────────────────────────


def _extrait(outils):
    from unittest.mock import MagicMock, patch

    from src.jarvis.memory.graph_ops import extract_graph_memories

    with patch("src.jarvis.memory.graph_ops.call_llm_direct") as appel:
        appel.return_value = '["un fait quelconque"]'
        faits = extract_graph_memories(
            summary="Elle a parlé de ses propres capacités.",
            cfg=MagicMock(), chat_model="m", tools_used=outils)
    return faits, appel.call_count


def test_a_window_whose_only_tool_consulted_nothing_extracts_nothing():
    faits, appels = _extrait(["toolSearchTool", "stop"])

    assert faits == []
    assert appels == 0


def test_a_window_with_a_real_lookup_still_extracts():
    """The control. A gate that blocked everything would satisfy the test
    above and destroy the feature."""
    faits, appels = _extrait(["stop", "webSearch"])

    assert faits == ["un fait quelconque"]
    assert appels == 1
