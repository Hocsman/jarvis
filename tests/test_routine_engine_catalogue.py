"""What a routine's turn is even offered.

Three removals, and the first is the one that matters most.

`toolSearchTool` is `lecture` and sits in this user's `## Libre`. Given
it, the model can append **any name in the registry** to its own
allow-list mid-turn and regenerate the schema — `macos__execute_script`
included. The gate would still refuse the call, so nothing executes; but
a routine reaching for arbitrary code execution because a fetched page
suggested it is not a thing that should get as far as being refused,
logged, and reported in a morning write-up as if it were an ordinary
blocked step. It is simply not in the catalogue.

`stop` goes for a duller reason: it ends a conversation, and there is no
conversation to end.

The warm profile block goes because a routine does not need to know who
the user is to summarise their mail, and every line of it is a line of
their private life sent to a remote endpoint while they sleep. A routine
that genuinely needs it says so in its own block.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.routines.scope import RoutineScope


def _cfg(tmp_path):
    cfg = MagicMock()
    cfg.llm_chat_model = "test-large"
    cfg.ollama_chat_model = "test-large"
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    cfg.db_path = str(tmp_path / "jarvis.db")
    return cfg


def _turn(tmp_path, *, scope, text="résume-moi mes mails"):
    """One engine turn, handing back what the model was offered."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}

    def _capture(**kw):
        messages = kw.get("messages") or []
        seen.setdefault("tools", kw.get("tools"))
        seen.setdefault("system", messages[0].get("content", "") if messages else "")
        return {"message": {"content": "rien à signaler"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="rien à signaler"):
        run_reply_engine(
            db=MagicMock(), cfg=_cfg(tmp_path), tts=None, text=text,
            dialogue_memory=DialogueMemory(), origin="routine", scope=scope,
        )
    return seen


def _offered(seen) -> set:
    tools = seen.get("tools") or []
    return {
        (t.get("function") or {}).get("name")
        for t in tools if isinstance(t, dict)
    }


# ── The escalation route is not in the catalogue ──────────────────────


def test_a_routine_is_not_offered_the_tool_that_widens_its_own_catalogue(tmp_path):
    """`toolSearchTool` appends any registry name to the turn's
    allow-list. The gate would still refuse the resulting call, but a
    routine reaching for arbitrary code execution because a fetched page
    suggested it should not get as far as being refused and written up."""
    seen = _turn(tmp_path, scope=RoutineScope(nom="matin", outils=["webSearch"]))

    assert "toolSearchTool" not in _offered(seen)


def test_nor_the_tool_that_changes_which_tools_exist(tmp_path):
    """`refreshMCPTools` is `## Libre` for this user and rediscovers
    servers mid-run. A catalogue that can change under the envelope is
    not an envelope."""
    seen = _turn(tmp_path, scope=RoutineScope(
        nom="matin", outils=["webSearch", "refreshMCPTools"],
    ))

    assert "refreshMCPTools" not in _offered(seen)


def test_nor_stop(tmp_path):
    """It ends a conversation, and there is no conversation to end."""
    seen = _turn(tmp_path, scope=RoutineScope(nom="matin", outils=["webSearch"]))

    assert "stop" not in _offered(seen)


# ── Only what the envelope names ──────────────────────────────────────


def test_only_the_envelope_is_offered(tmp_path):
    seen = _turn(tmp_path, scope=RoutineScope(
        nom="matin", outils=["webSearch", "getWeather"],
    ))

    assert _offered(seen) <= {"webSearch", "getWeather"}


def test_an_empty_envelope_offers_nothing(tmp_path):
    seen = _turn(tmp_path, scope=RoutineScope(nom="matin", outils=[]))

    assert _offered(seen) == set()


def test_a_tool_named_in_the_envelope_but_absent_from_the_catalogue_is_skipped(tmp_path):
    """A user typing a name by hand, or an MCP server that went away."""
    seen = _turn(tmp_path, scope=RoutineScope(
        nom="matin", outils=["webSearch", "outilQuiNExistePas"],
    ))

    assert "outilQuiNExistePas" not in _offered(seen)


# ── An empty list is empty, not everything ────────────────────────────


def test_an_empty_allow_list_produces_an_empty_schema():
    """Two callers deliberately narrow to nothing: a routine with an
    empty envelope, and the resume turn that runs one approved call and
    narrates it. Reading `[]` as "the whole catalogue" hands both of them
    every builtin there is."""
    from src.jarvis.tools.registry import (
        generate_tools_description,
        generate_tools_json_schema,
    )

    assert generate_tools_json_schema([], None) == []
    assert "logMeal" not in generate_tools_description([], None)


def test_asking_for_nothing_in_particular_still_means_everything():
    """`None` is the other half of the contract, and callers rely on it."""
    from src.jarvis.tools.registry import BUILTIN_TOOLS, generate_tools_json_schema

    assert len(generate_tools_json_schema(None, None)) == len(BUILTIN_TOOLS)


# ── The user's private life stays home ────────────────────────────────


def test_the_profile_is_not_sent_by_default(tmp_path):
    """Every line of it is the user's private life going to a remote
    endpoint while they sleep, and a routine does not need to know who
    they are to summarise their mail."""
    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

    cfg = _cfg(tmp_path)
    MemoryCore.for_config(cfg).remember(SECTION_PROFILE, "Il vit à Lyon.")

    seen = _turn(tmp_path, scope=RoutineScope(nom="matin", outils=["webSearch"]))

    assert "Lyon" not in seen.get("system", "")


def test_a_routine_that_asks_for_it_gets_it(tmp_path):
    """Some genuinely need it. It is opt-in, per routine, in the block
    the user can read."""
    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

    cfg = _cfg(tmp_path)
    MemoryCore.for_config(cfg).remember(SECTION_PROFILE, "Il vit à Lyon.")

    seen = _turn(tmp_path, scope=RoutineScope(
        nom="matin", outils=["webSearch"], memoire=True,
    ))

    assert "Lyon" in seen.get("system", "")


# ── An attended turn is untouched ─────────────────────────────────────


def test_voice_still_gets_the_whole_catalogue(tmp_path):
    """Every removal above is conditional on running unattended."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}

    def _capture(**kw):
        seen.setdefault("tools", kw.get("tools"))
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(
            db=MagicMock(), cfg=_cfg(tmp_path), tts=None, text="bonjour",
            dialogue_memory=DialogueMemory(), origin="voix",
        )

    offered = _offered(seen)
    assert "stop" in offered
    assert "toolSearchTool" in offered
