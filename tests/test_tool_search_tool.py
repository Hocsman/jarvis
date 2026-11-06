"""Unit tests for the toolSearchTool builtin."""

from unittest.mock import patch

import pytest

from jarvis.tools.builtin.tool_search import ToolSearchTool
from jarvis.tools.base import ToolContext


def _ctx(cfg):
    return ToolContext(
        db=None,
        cfg=cfg,
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
        user_print=lambda _m: None,
        language=None,
    )


class TestToolSearchTool:
    def test_rejects_missing_query(self, mock_config):
        tool = ToolSearchTool()
        result = tool.run({}, _ctx(mock_config))
        assert result.success is False
        assert "query" in (result.error_message or "").lower()

    def test_invokes_select_tools_and_formats_list(self, mock_config):
        tool = ToolSearchTool()
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools",
            return_value=["webSearch", "stop", "toolSearchTool", "getWeather"],
        ) as mock_sel:
            result = tool.run({"query": "look up a fact"}, _ctx(mock_config))
        assert mock_sel.called
        assert result.success is True
        text = result.reply_text or ""
        # Sentinel and self are filtered out; real tools appear as
        # `name: description`.
        assert "webSearch" in text
        assert "getWeather" in text
        assert "stop" not in text.split("\n")[0]
        assert "toolSearchTool" not in text.splitlines()[0]
        # Each line has the colon-joined description format.
        for line in text.splitlines():
            assert ":" in line or line.strip() in ("webSearch", "getWeather")

    def test_empty_result_returns_honest_note(self, mock_config):
        tool = ToolSearchTool()
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools",
            return_value=["stop", "toolSearchTool"],
        ):
            result = tool.run({"query": "do something"}, _ctx(mock_config))
        assert result.success is True
        assert "no additional tools" in (result.reply_text or "").lower()

    def test_a_search_that_matches_nothing_says_so(self, mock_config):
        """A search that comes back with only the tools every turn already
        offers found nothing. Listing `forget` and `remember` as hits would
        hand the model a write to the user's memory as the answer to a
        question about something else."""
        mock_config.tool_selection_strategy = "keyword"
        tool = ToolSearchTool()

        result = tool.run({"query": "zzyzx qwvx"}, _ctx(mock_config))

        assert result.success is True
        assert result.reply_text == "No additional tools found for that description."

    def test_the_tools_every_turn_offers_are_not_listed_as_found(self, mock_config):
        from jarvis.tools.selection import _ALWAYS_INCLUDED

        tool = ToolSearchTool()
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools",
            return_value=["getWeather", "stop", "forget", "remember"],
        ):
            result = tool.run({"query": "weather tomorrow"}, _ctx(mock_config))

        listed = [line.split(":", 1)[0] for line in (result.reply_text or "").splitlines()]
        assert listed == ["getWeather"]
        assert not set(listed) & set(_ALWAYS_INCLUDED)

    def test_a_search_for_a_tools_exact_name_finds_it(self, mock_config, tmp_path):
        """The system prompt tells the chat model a tool's name when the
        router may not have offered it (`claudeCode`, projets.spec.md); the
        search for that name finds it, even from a router that answers
        'none'."""
        from unittest.mock import MagicMock

        (tmp_path / "quillon").mkdir()
        mock_config.claude_code_projects_dir = str(tmp_path)
        mock_config.tool_selection_strategy = "llm"
        router = MagicMock()
        router.direct.return_value = "none"
        tool = ToolSearchTool()
        with patch("jarvis.tools.builtin.tool_search.get_auxiliary_backend",
                   return_value=router):
            result = tool.run({"query": "claudeCode"}, _ctx(mock_config))

        listed = [line.split(":", 1)[0] for line in (result.reply_text or "").splitlines()]
        assert listed == ["claudeCode"]

    def test_a_tool_switched_off_is_not_found_by_its_name(self, mock_config):
        from unittest.mock import MagicMock

        mock_config.claude_code_projects_dir = ""
        mock_config.tool_selection_strategy = "llm"
        router = MagicMock()
        router.direct.return_value = "none"
        tool = ToolSearchTool()
        with patch("jarvis.tools.builtin.tool_search.get_auxiliary_backend",
                   return_value=router):
            result = tool.run({"query": "claudeCode"}, _ctx(mock_config))

        assert result.reply_text == "No additional tools found for that description."

    def test_select_tools_exception_returns_error(self, mock_config):
        tool = ToolSearchTool()
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools",
            side_effect=RuntimeError("router down"),
        ):
            result = tool.run({"query": "x"}, _ctx(mock_config))
        assert result.success is False
        assert "router down" in (result.error_message or "")
