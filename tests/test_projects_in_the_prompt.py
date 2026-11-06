"""She knows she can work on his projects, whatever the router offered.

Asked whether she can reach his code, a chat model shown only the tools
every turn carries answers from its prior: an assistant has no access to
anyone's files. With handing projects to Claude Code switched on, that is
false, and the router (a small model) does not reliably offer `claudeCode`
for a question about what she can do. So while the tool is offered, the
system prompt says what she can do and how to reach the tool when it is not
among her tools this turn. Off, or on a routine turn, which never reaches
`claudeCode`, it says nothing.

See src/jarvis/projets/projets.spec.md, What she knows she can do.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.projets.prompt import format_projets_block


def _cfg(tmp_path, projects_dir: str) -> MagicMock:
    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.llm_chat_model = "test-large"
    cfg.mcps = {}
    cfg.voice_debug = False
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_enabled = False
    cfg.llm_thinking_enabled = False
    cfg.agentic_max_turns = 8
    cfg.memory_enrichment_source = "diary"
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.tool_search_max_calls = 3
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_tools_timeout_sec = 8.0
    cfg.claude_code_projects_dir = projects_dir
    cfg.tts_engine = "kyutai"
    return cfg


def _system_prompt(cfg, text="tu peux bosser sur mes projets ?", scope=None, origin="voix"):
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}

    def _capture(**kw):
        messages = kw.get("messages") or []
        seen.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(
            db=MagicMock(), cfg=cfg, tts=None, text=text,
            dialogue_memory=DialogueMemory(), origin=origin, scope=scope,
        )
    return seen.get("system", "")


@pytest.fixture
def projects(tmp_path):
    root = tmp_path / "Projets"
    (root / "quillon").mkdir(parents=True)
    return root


# ── What the block says ───────────────────────────────────────────────


def test_it_names_the_tool_and_the_search_that_finds_it():
    """The two names she needs: the tool that does the work, and the
    escape hatch that adds it to a turn whose router left it out. The
    search for the tool's exact name finds it whatever the router says
    (selection.spec.md, Name cue)."""
    block = format_projets_block()

    assert "claudeCode" in block
    assert "toolSearchTool" in block


def test_it_names_no_project():
    """Folder names are the user's, and the block rides in every prompt to
    whichever provider the chat model runs on. The tool lists the projects
    when it needs to, in the turn that asks."""
    block = format_projets_block()

    assert "quillon" not in block


# ── When it reaches the prompt ────────────────────────────────────────


def test_switched_on_the_prompt_says_she_can(tmp_path, projects):
    system = _system_prompt(_cfg(tmp_path, str(projects)))

    assert format_projets_block() in system


def test_switched_off_the_prompt_says_nothing_of_it(tmp_path):
    system = _system_prompt(_cfg(tmp_path, ""))

    assert format_projets_block() not in system
    assert "claudeCode" not in system


def test_a_turn_typed_in_the_chat_window_knows_it_too(tmp_path, projects):
    system = _system_prompt(_cfg(tmp_path, str(projects)), origin="chat")

    assert format_projets_block() in system


def test_a_routine_turn_is_not_told(tmp_path, projects):
    """A routine never reaches `claudeCode` (routines.spec.md), and its
    catalogue is its envelope: no escape hatch to reach it with."""
    from src.jarvis.routines.scope import RoutineScope

    system = _system_prompt(
        _cfg(tmp_path, str(projects)), text="résume les mails", origin="routine",
        scope=RoutineScope(nom="matin", outils=["webSearch"]),
    )

    assert format_projets_block() not in system
