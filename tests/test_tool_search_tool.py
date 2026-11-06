"""Unit tests for the toolSearchTool builtin."""

import threading
import time
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

    def test_select_tools_exception_returns_error(self, mock_config):
        tool = ToolSearchTool()
        with patch(
            "jarvis.tools.builtin.tool_search.select_tools",
            side_effect=RuntimeError("router down"),
        ):
            result = tool.run({"query": "x"}, _ctx(mock_config))
        assert result.success is False
        assert "router down" in (result.error_message or "")


def _listed(result) -> list:
    return [line.split(":", 1)[0] for line in (result.reply_text or "").splitlines()]


class _Router:
    """The router's backend: answers ``answer``, raises ``raises``, or
    holds its answer until ``release`` is set (or five seconds pass)."""

    def __init__(self, answer="none", raises=None, holds=False):
        self.answer = answer
        self.raises = raises
        self.release = threading.Event()
        if not holds:
            self.release.set()

    def direct(self, *args, **kwargs):
        self.release.wait(5.0)
        if self.raises is not None:
            raise self.raises
        return self.answer


class TestASearchForAToolsName:
    """The system prompt can give the chat model a tool's name and tell it
    to search for it when the router left it out (`claudeCode`,
    projets.spec.md). A search for a tool's exact name finds it, whatever
    the router makes of the query (tool_search.spec.md, Name cue)."""

    @pytest.fixture
    def cfg(self, mock_config, tmp_path):
        (tmp_path / "quillon").mkdir()
        mock_config.claude_code_projects_dir = str(tmp_path)
        mock_config.tool_selection_strategy = "llm"
        mock_config.tool_router_model = "router-test"
        return mock_config

    def _search(self, cfg, query, router):
        with patch("jarvis.tools.builtin.tool_search.get_auxiliary_backend",
                   return_value=router):
            return ToolSearchTool().run({"query": query}, _ctx(cfg))

    @pytest.mark.parametrize("query", ["claudeCode", '"claudeCode"', "`claudeCode`"])
    def test_the_bare_name_is_found_from_a_router_that_answers_none(self, cfg, query):
        result = self._search(cfg, query, _Router("none"))

        assert _listed(result) == ["claudeCode"]

    def test_a_tool_switched_off_is_not_found_by_its_name(self, cfg):
        cfg.claude_code_projects_dir = ""

        result = self._search(cfg, "claudeCode", _Router("none"))

        assert result.reply_text == "No additional tools found for that description."

    def test_a_name_among_other_words_comes_with_what_the_router_found(self, cfg):
        result = self._search(
            cfg, "claudeCode corrige le bug des dates dans le projet quillon",
            _Router("webSearch"),
        )

        listed = _listed(result)
        assert listed[0] == "claudeCode"
        assert "webSearch" in listed

    @pytest.mark.parametrize(
        "query", ["Claude Code", "claudecode", "claudeCodes", "myclaudeCode"],
    )
    def test_only_the_exact_name_counts(self, cfg, query):
        """The words a person says for it are the router's business: a name
        is an identifier, and an identifier is spelt one way."""
        result = self._search(cfg, query, _Router("none"))

        assert "claudeCode" not in _listed(result)

    def test_an_mcp_tool_is_found_by_its_name(self, cfg):
        from jarvis.tools.registry import ToolSpec

        cfg.mcps = {"homeassistant": {"command": "invented"}}
        mcp = {"homeassistant__turn_on": ToolSpec(
            "homeassistant__turn_on", "Turn on a smart home device.", {})}
        with patch("jarvis.tools.registry.get_cached_mcp_tools", return_value=mcp):
            result = self._search(cfg, "homeassistant__turn_on", _Router("none"))

        assert _listed(result) == ["homeassistant__turn_on"]

    def test_a_degraded_router_still_lets_a_name_in_a_longer_query_be_found(self, cfg):
        result = self._search(
            cfg, "claudeCode ajoute une page de contact au projet quillon",
            _Router(raises=TimeoutError("router down")),
        )

        assert "claudeCode" in _listed(result)

    def test_a_bare_name_is_answered_without_waiting_for_the_router(self, cfg):
        """The router has nothing to add to a query made only of names, and
        its round trip can take up to its whole deadline."""
        router = _Router("none", holds=True)
        started = time.monotonic()
        try:
            result = self._search(cfg, "claudeCode", router)
            elapsed = time.monotonic() - started
        finally:
            router.release.set()

        assert _listed(result) == ["claudeCode"]
        assert elapsed < 2.0

    def test_several_names_are_all_found_without_the_router(self, cfg):
        router = _Router("none", holds=True)
        started = time.monotonic()
        try:
            result = self._search(cfg, "claudeCode, getWeather", router)
            elapsed = time.monotonic() - started
        finally:
            router.release.set()

        assert set(_listed(result)) == {"claudeCode", "getWeather"}
        assert elapsed < 2.0

    def test_a_router_that_raises_still_lets_a_name_in_a_longer_query_be_found(self, cfg):
        """Building the router's backends, or the router itself, can raise
        rather than fall back; the name the query carries is still found."""
        with patch("jarvis.tools.builtin.tool_search.select_tools",
                   side_effect=RuntimeError("router down")):
            result = ToolSearchTool().run(
                {"query": "claudeCode ajoute une page de contact au projet quillon"},
                _ctx(cfg),
            )

        assert result.success is True
        assert _listed(result) == ["claudeCode"]

    def test_a_query_naming_many_tools_finds_no_more_than_the_router_would(self, cfg):
        """A search is not a way to list the catalogue: the names it finds
        are capped as the router's picks are, the first ones kept."""
        from jarvis.tools.registry import offered_builtin_tools
        from jarvis.tools.selection import _ALWAYS_INCLUDED, _LLM_MAX_SELECTED

        names = [n for n in offered_builtin_tools(cfg)
                 if n not in (*_ALWAYS_INCLUDED, "toolSearchTool")]
        query = " ".join(names)

        result = self._search(cfg, query, _Router("none"))

        assert _listed(result) == names[:_LLM_MAX_SELECTED]


class TestToolsNamedIn:
    """Which tools of the catalogue a query names: each name as a whole
    word, spelt as the catalogue spells it, with the characters a tool
    name may hold (naming.PLAIN_NAME: letters, digits, `_`, `.`, `-`)."""

    @pytest.fixture
    def catalogue(self, mock_config, tmp_path):
        from jarvis.tools.registry import ToolSpec, offered_builtin_tools

        (tmp_path / "quillon").mkdir()
        mock_config.claude_code_projects_dir = str(tmp_path)
        mcp = {
            name: ToolSpec(name, "An invented tool.", {})
            for name in (
                "chrome-devtools__take_snapshot", "devtools__take_snapshot",
                "srv__files", "srv__files.read",
            )
        }
        return offered_builtin_tools(mock_config), mcp

    def _named(self, catalogue, query):
        from jarvis.tools.builtin.tool_search import tools_named_in

        return tools_named_in(query, *catalogue)

    @pytest.mark.parametrize("query", ["", None, "   "])
    def test_an_empty_query_names_nothing(self, catalogue, query):
        assert self._named(catalogue, query) == []

    def test_several_names_in_catalogue_order(self, catalogue):
        named = self._named(catalogue, "getWeather puis claudeCode et webSearch")

        assert named == [n for n in catalogue[0] if n in ("getWeather", "claudeCode", "webSearch")]

    def test_a_hyphenated_name_is_found_and_its_tail_is_not(self, catalogue):
        named = self._named(catalogue, "chrome-devtools__take_snapshot")

        assert named == ["chrome-devtools__take_snapshot"]

    def test_a_dotted_name_is_found_and_its_head_is_not(self, catalogue):
        named = self._named(catalogue, "lis le fichier avec srv__files.read")

        assert named == ["srv__files.read"]

    @pytest.mark.parametrize("query", [
        "claudeCode.", "Utilise claudeCode.", "«claudeCode»", "(claudeCode)",
        "claudeCode...", "`claudeCode`, vite",
    ])
    def test_punctuation_around_a_name_does_not_hide_it(self, catalogue, query):
        assert self._named(catalogue, query) == ["claudeCode"]

    @pytest.mark.parametrize("query", [
        "àclaudeCode", "claudeCodeé", "жclaudeCode", "claudeCode数", "claudeCode_x",
        "x.claudeCode", "claudeCode-x", "claudeCode.x",
    ])
    def test_a_name_glued_to_a_letter_of_any_script_is_not_a_match(self, catalogue, query):
        assert "claudeCode" not in self._named(catalogue, query)
