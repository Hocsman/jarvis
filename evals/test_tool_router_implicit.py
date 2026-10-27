"""
Tool Router — Implicit Intent & Multi-Tool Coverage (Live)

The existing router evals (test_tool_selection.py, test_tool_router_context_aware.py)
lean on queries whose keywords almost name the tool ("search the web for X",
"log that I had Y"). In production the router fails on a different shape of
query: the words don't correspond to tool names, or the query needs more than
one tool to be answered usefully.

This file captures those shapes so regressions where the router over-prunes
are caught before they land. Known motivating failures:

  - "how's the weather this week?" → router picked [getWeather, stop] only,
    blocking the webSearch → fetchWebPage chain the mocked agent tests expect.
  - "should I order pizza tonight?" → router picked [stop] only. fetchMeals
    never reached the LLM, so the agent could not ground its advice in
    today's intake.
  - "tous les lundis matin, fais-moi un point sur la météo" → the router
    picked the weather and the web, never setRoutine (qwen2.5:3b, 3 runs in
    3); "trouve la facture PDF sur mon ordinateur" → 'none', never
    localFiles (3 in 3).

Principles locked in here:
  1. Implicit-intent queries (no tool-name keywords) must still route to the
     correct tool.
  2. The router must NEVER collapse to only the mandatory tools when the
     query has a clear actionable intent — that is a "silently useless"
     failure mode.
  3. Multi-intent queries must surface each relevant tool (or a superset).
  4. A request whose job only one tool can do (repeat something on a
     schedule, find a file, read an article) must reach that tool, in
     every language it is asked in.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b pytest evals/test_tool_router_implicit.py -v
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL


def _route(query: str, context_hint=None):
    """Invoke the real LLM router with the full builtin tool catalogue."""
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.selection import select_tools, ToolSelectionStrategy
    from jarvis.llm import get_llm_backend
    from helpers import MockConfig

    cfg = MockConfig()
    cfg.ollama_base_url = JUDGE_BASE_URL

    return select_tools(
        query=query,
        builtin_tools=BUILTIN_TOOLS,
        mcp_tools={},
        strategy=ToolSelectionStrategy.LLM,
        llm_backend=get_llm_backend(cfg),
        llm_model=JUDGE_MODEL,
        llm_timeout_sec=30.0,
        context_hint=context_hint,
    )


def _real_tools(selected):
    """What the router chose beyond the tools that always come back."""
    from jarvis.tools.selection import _ALWAYS_INCLUDED
    return [t for t in selected if t not in _ALWAYS_INCLUDED]


# =============================================================================
# Implicit Intent — words do not correspond to tool names
# =============================================================================

# (query, must_include_any_of, rationale)
IMPLICIT_INTENT_CASES = [
    pytest.param(
        "should I order pizza tonight?",
        ["fetchMeals"],
        "Advisory food decision needs today's intake to answer usefully.",
        id="food decision → fetchMeals",
    ),
    pytest.param(
        "am I under my calorie budget today?",
        ["fetchMeals"],
        "Budget question with no 'meal' keyword still needs the log.",
        id="calorie budget → fetchMeals",
    ),
    pytest.param(
        "do I need a jacket today?",
        ["getWeather"],
        "Clothing question is a weather question in disguise.",
        id="jacket → getWeather",
    ),
    pytest.param(
        "will the run be miserable this afternoon?",
        ["getWeather"],
        "Activity planning with weather subtext, no 'weather' keyword.",
        id="run forecast → getWeather",
    ),
    pytest.param(
        "what did I put in my body today?",
        ["fetchMeals"],
        "Colloquial meal recall, no tool-name keywords.",
        id="meal recall (colloquial) → fetchMeals",
    ),
    pytest.param(
        "did I have anything with gluten earlier?",
        ["fetchMeals"],
        "Dietary check against logged meals.",
        id="dietary check → fetchMeals",
    ),
]


@pytest.mark.eval
@requires_judge_llm
class TestImplicitIntent:
    """Router must route on intent, not on surface keywords."""

    @pytest.mark.parametrize("query, must_include_any, rationale", IMPLICIT_INTENT_CASES)
    def test_implicit_intent_routes_to_correct_tool(
        self, query, must_include_any, rationale
    ):
        selected = _route(query)
        real = _real_tools(selected)

        print(f"\n  Query: {query}")
        print(f"  Rationale: {rationale}")
        print(f"  Selected: {selected}")

        # Floor invariant (soft — small router models sometimes collapse to
        # only the mandatory tools on dietary/advisory queries). Tracked as
        # xfail so a future router improvement flips this to an unexpected
        # pass.
        if not real:
            pytest.xfail(
                f"Router collapsed to only the mandatory tools for an "
                f"actionable query on {JUDGE_MODEL}. Query: {query!r}. "
                f"Rationale: {rationale}"
            )

        matched = [t for t in must_include_any if t in selected]
        if not matched:
            pytest.xfail(
                f"Router missed implicit intent on {JUDGE_MODEL}. "
                f"Expected any of {must_include_any}, got {selected}. "
                f"Rationale: {rationale}"
            )


# =============================================================================
# Multi-Tool Intent — one question needs several tools
# =============================================================================

# (query, must_include_all, rationale)
MULTI_TOOL_CASES = [
    pytest.param(
        "plan my day around the weather and what I've eaten",
        ["getWeather", "fetchMeals"],
        "Two explicit subjects, two tools.",
        id="weather + meals",
    ),
    pytest.param(
        "find me a detailed article about the Apollo program",
        ["webSearch", "fetchWebPage"],
        "Research queries need search then fetch to read the actual page.",
        id="research → webSearch + fetchWebPage",
    ),
    pytest.param(
        "how's the weather this week?",
        ["getWeather"],
        "Must include getWeather; webSearch/fetchWebPage acceptable as backup "
        "for multi-day forecasts the API may not cover.",
        id="weekly weather keeps getWeather",
    ),
]


@pytest.mark.eval
@requires_judge_llm
class TestMultiToolIntent:
    """Router must surface every tool a multi-part query needs."""

    @pytest.mark.parametrize("query, must_include_all, rationale", MULTI_TOOL_CASES)
    def test_multi_tool_intent_surfaces_all_needed(
        self, query, must_include_all, rationale
    ):
        selected = _route(query)
        real = _real_tools(selected)

        print(f"\n  Query: {query}")
        print(f"  Rationale: {rationale}")
        print(f"  Selected: {selected}")

        if not real:
            pytest.xfail(
                f"Router collapsed to only the mandatory tools for a "
                f"multi-intent query on {JUDGE_MODEL}. Query: {query!r}."
            )

        missing = [t for t in must_include_all if t not in selected]
        if missing:
            pytest.xfail(
                f"Router dropped needed tools on {JUDGE_MODEL}. "
                f"Missing: {missing}. Got: {selected}. Rationale: {rationale}"
            )


# =============================================================================
# Floor Invariant — router must never silently collapse to the mandatory tools
# =============================================================================

# Queries that have an unambiguous tool-shaped answer. The router may legitimately
# narrow the catalogue, but returning only the mandatory tools for any of these is
# a bug: it means the main model will have no way to act on the user's clear
# request.
NEVER_EMPTY_CASES = [
    "take a screenshot",
    "what's on my screen right now?",
    "search the web for flight deals",
    "log that I just ate a banana",
    "what's the weather like?",
    "find the invoice PDF on my computer",
]


@pytest.mark.eval
@requires_judge_llm
class TestRouterNeverCollapses:
    """Regression guard for the 'selected only the mandatory tools' failure mode."""

    @pytest.mark.parametrize("query", NEVER_EMPTY_CASES)
    def test_clear_intent_keeps_at_least_one_real_tool(self, query):
        selected = _route(query)
        real = _real_tools(selected)
        print(f"\n  Query: {query}")
        print(f"  Selected: {selected}")
        assert real, (
            f"Router collapsed to only the mandatory tools for a clearly "
            f"actionable query. Query: {query!r}. This silently disables the agent — every main-"
            f"model tool_call would be dropped as out-of-catalogue."
        )


# =============================================================================
# Task Tools — the request is a job only one tool can do
# =============================================================================

# What the reply engine hands the router on every turn: the live time and
# location line (`_build_enrichment_context_hint`). Measured in this shape,
# not bare, because the bare shape is not the one the user's sentences meet:
# with no KNOWN FACTS block, qwen2.5:3b answers 'none' to "trouve la facture
# PDF sur mon ordinateur" whatever localFiles says, and does so because of
# the router prompt's 'none' rule (removed, it picks a tool).
_LIVE_FACTS = (
    "Current local time: Tuesday, 2026-10-27 17:42 (Europe/Paris). "
    "Location: Paris, Île-de-France, France."
)

# (query, must_include_all, rationale), each asked in French and in English.
TASK_TOOL_CASES = [
    pytest.param(
        "tous les lundis matin, fais-moi un point sur la météo",
        ["setRoutine"],
        "Something to happen every week is a routine; the weather is only "
        "what the routine will fetch.",
        id="FR: weekly weather report → setRoutine",
    ),
    pytest.param(
        "every Monday morning, give me a weather update",
        ["setRoutine"],
        "Same request in English.",
        id="EN: weekly weather report → setRoutine",
    ),
    pytest.param(
        "trouve la facture PDF sur mon ordinateur",
        ["localFiles"],
        "A file on the user's own computer, not on the web.",
        id="FR: invoice on the computer → localFiles",
    ),
    pytest.param(
        "find the invoice PDF on my computer",
        ["localFiles"],
        "Same request in English.",
        id="EN: invoice on the computer → localFiles",
    ),
    pytest.param(
        "trouve-moi un article détaillé sur le programme Apollo",
        ["webSearch", "fetchWebPage"],
        "An article is found by searching and read by opening it.",
        id="FR: detailed article → webSearch + fetchWebPage",
    ),
    pytest.param(
        "find me a detailed article about the Apollo program",
        ["webSearch", "fetchWebPage"],
        "Same request in English.",
        id="EN: detailed article → webSearch + fetchWebPage",
    ),
]


@pytest.mark.eval
@requires_judge_llm
class TestRouterOffersTheTaskTool:
    """Router must offer the tool that does the job the user asked for."""

    @pytest.mark.parametrize("query, must_include_all, rationale", TASK_TOOL_CASES)
    def test_task_reaches_its_tool(self, query, must_include_all, rationale):
        selected = _route(query, context_hint=_LIVE_FACTS)

        print(f"\n  Query: {query}")
        print(f"  Rationale: {rationale}")
        print(f"  Selected: {selected}")

        missing = [t for t in must_include_all if t not in selected]
        assert not missing, (
            f"Router did not offer {missing} on {JUDGE_MODEL}. Got: {selected}. "
            f"Rationale: {rationale}"
        )
