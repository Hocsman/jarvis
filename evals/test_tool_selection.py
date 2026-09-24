"""
Tool Selection Evaluations

Tests that the embedding-based tool selection strategy actually filters tools
meaningfully — a weather query should select weather-related tools, not all tools.

Run: .venv/bin/python -m pytest evals/test_tool_selection.py -v
"""

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL


# =============================================================================
# Test Data
# =============================================================================

# Queries paired with the tools they MUST include and a cap on how many
# content tools the strategy may pick. The cap covers content picks only:
# the strategy adds its own always-included set on top, so the assertion
# derives the total from the product constant rather than pinning a count
# that goes stale whenever a tool becomes mandatory.
TOOL_SELECTION_CASES = [
    pytest.param(
        "what's the weather like tomorrow",
        ["getWeather"],
        4,
        id="weather query selects getWeather and few others",
    ),
    pytest.param(
        "what's the weather in London this weekend",
        ["getWeather"],
        4,
        id="location weather query selects getWeather and few others",
    ),
    pytest.param(
        "log that I had a chicken salad for lunch",
        ["logMeal"],
        4,
        id="meal logging selects logMeal and few others",
    ),
    pytest.param(
        "what did I eat yesterday",
        ["fetchMeals"],
        4,
        id="meal recall selects fetchMeals and few others",
    ),
    pytest.param(
        "search the web for Python tutorials",
        ["webSearch"],
        4,
        id="web search query selects webSearch and few others",
    ),
]


def _max_total_selected(max_content_tools: int) -> int:
    """Total selection cap: content picks plus the always-included set."""
    from jarvis.tools.selection import _ALWAYS_INCLUDED
    return max_content_tools + len(_ALWAYS_INCLUDED)


@pytest.mark.eval
class TestToolSelectionFiltering:
    """Validates that embedding tool selection meaningfully filters tools."""

    @requires_judge_llm
    @pytest.mark.parametrize("query, must_include, max_tools", TOOL_SELECTION_CASES)
    def test_embedding_selects_relevant_tools(
        self,
        mock_config,
        query,
        must_include,
        max_tools,
    ):
        """Embedding strategy should select relevant tools, not all of them.

        Tool selection uses a fixed embed model (nomic-embed-text) regardless of
        the judge model, so we only run this once per eval run (during the
        gemma4 phase) to save time.
        """
        if "gemma4" not in JUDGE_MODEL:
            pytest.skip(f"Tool selection uses fixed embed model; only runs in gemma4 phase (current: {JUDGE_MODEL})")

        from jarvis.tools.selection import select_tools, ToolSelectionStrategy
        from jarvis.tools.registry import BUILTIN_TOOLS
        from jarvis.llm import get_embedding_backend

        selected = select_tools(
            query=query,
            builtin_tools=BUILTIN_TOOLS,
            mcp_tools={},
            strategy=ToolSelectionStrategy.EMBEDDING,
            embedding_backend=get_embedding_backend(mock_config),
            embed_model=mock_config.ollama_embed_model,
            embed_timeout_sec=10.0,
        )

        total_builtin = len(BUILTIN_TOOLS)

        # Must include the expected tools
        for tool in must_include:
            assert tool in selected, (
                f"Expected '{tool}' in selected tools but got: {selected}"
            )

        # Must include 'stop' (always included)
        assert "stop" in selected, f"'stop' should always be included, got: {selected}"

        # Must NOT include everything — that means filtering isn't working
        assert len(selected) <= _max_total_selected(max_tools), (
            f"Expected at most {_max_total_selected(max_tools)} tools but got {len(selected)}/{total_builtin}: {selected}"
        )

        print(f"  ✅ Selected {len(selected)}/{total_builtin} tools: {selected}")


@pytest.mark.eval
class TestToolSelectionFilteringLLM:
    """Validates that LLM-router tool selection meaningfully filters tools.

    Unlike the embedding strategy (pinned to nomic-embed-text), this exercises
    the default `llm` strategy against whichever judge model is active, so the
    same cases run once per supported chat model.
    """

    @requires_judge_llm
    @pytest.mark.parametrize("query, must_include, max_tools", TOOL_SELECTION_CASES)
    def test_llm_selects_relevant_tools(
        self,
        mock_config,
        query,
        must_include,
        max_tools,
    ):
        from jarvis.tools.selection import select_tools, ToolSelectionStrategy
        from jarvis.tools.registry import BUILTIN_TOOLS
        from jarvis.llm import get_llm_backend

        selected = select_tools(
            query=query,
            builtin_tools=BUILTIN_TOOLS,
            mcp_tools={},
            strategy=ToolSelectionStrategy.LLM,
            llm_backend=get_llm_backend(mock_config),
            llm_model=JUDGE_MODEL,
            llm_timeout_sec=15.0,
        )

        total_builtin = len(BUILTIN_TOOLS)

        for tool in must_include:
            assert tool in selected, (
                f"Expected '{tool}' in selected tools but got: {selected}"
            )

        assert "stop" in selected, f"'stop' should always be included, got: {selected}"

        assert len(selected) <= _max_total_selected(max_tools), (
            f"Expected at most {_max_total_selected(max_tools)} tools but got {len(selected)}/{total_builtin}: {selected}"
        )

        print(f"  ✅ [{JUDGE_MODEL}] Selected {len(selected)}/{total_builtin} tools: {selected}")
