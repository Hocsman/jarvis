"""
The spoken-form rule: units, signs and dates written as the words a person
says, for a reply the voice will read aloud.

It reaches a turn from the voice and nothing else. A reply typed in the chat
window is read, not heard, and keeps "11 °C"; a routine's reply is written to
its journal. The evals that measure what the rule does to the replies live in
``evals/test_spoken_units_as_words.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.prompts import ModelSize, SPOKEN_FORM, get_system_prompts


@pytest.mark.unit
class TestThePromptComponents:

    @pytest.mark.parametrize("size", list(ModelSize))
    def test_a_spoken_reply_is_given_the_rule(self, size):
        assert SPOKEN_FORM in get_system_prompts(size, spoken=True).to_list()

    @pytest.mark.parametrize("size", list(ModelSize))
    def test_a_written_reply_is_not(self, size):
        assert SPOKEN_FORM not in get_system_prompts(size, spoken=False).to_list()
        assert SPOKEN_FORM not in get_system_prompts(size).to_list()

    @pytest.mark.parametrize("size", list(ModelSize))
    def test_the_rule_is_all_that_tells_them_apart(self, size):
        spoken = get_system_prompts(size, spoken=True).to_list()
        written = get_system_prompts(size, spoken=False).to_list()

        assert [c for c in spoken if c != SPOKEN_FORM] == written


def _cfg(tmp_path):
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
    cfg.response_language = ""
    cfg.tts_engine = "kyutai"
    return cfg


@pytest.fixture
def db():
    from src.jarvis.memory.db import Database

    d = Database(":memory:", sqlite_vss_path=None)
    yield d
    d.close()


def _system_message(db, cfg, **kw) -> str:
    """The system message the chat model is handed for one turn."""
    from src.jarvis.reply.engine import run_reply_engine

    seen = {}

    def _capture(**call):
        seen.setdefault("messages", list(call.get("messages") or []))
        return {"message": {"content": "ok"}}

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               return_value="ok"):
        run_reply_engine(
            db=db, cfg=cfg, tts=None, text="il fait combien dehors ?",
            dialogue_memory=DialogueMemory(inactivity_timeout=300.0), **kw,
        )
    return seen["messages"][0]["content"]


@pytest.mark.unit
class TestWhichTurnsAreGivenTheRule:

    def test_a_turn_from_the_voice_is(self, db, tmp_path):
        system = _system_message(db, _cfg(tmp_path), origin="voix", language="fr")

        assert SPOKEN_FORM in system

    def test_a_turn_from_the_text_chat_is_not(self, db, tmp_path):
        system = _system_message(db, _cfg(tmp_path), origin="chat")

        assert SPOKEN_FORM not in system

    def test_a_routine_turn_is_not(self, db, tmp_path):
        from src.jarvis.routines.scope import RoutineScope

        system = _system_message(
            db, _cfg(tmp_path), origin="routine",
            scope=RoutineScope(nom="matin", outils=["getWeather"], memoire=False),
        )

        assert SPOKEN_FORM not in system

    def test_a_turn_that_names_no_origin_is_not(self, db, tmp_path):
        system = _system_message(db, _cfg(tmp_path))

        assert SPOKEN_FORM not in system


def _digest_system_prompt(db, cfg, **kw) -> str:
    """The system prompt the max-turn digest is handed when a turn's loop
    runs out of turns: the model only ever calls a tool, so the digest
    writes the reply."""
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.tools.types import ToolExecutionResult

    cfg.agentic_max_turns = 2
    cfg.intent_judge_model = ""
    cfg.llm_digest_timeout_sec = 8.0
    seen = {}

    def _always_a_tool_call(**_call):
        return {"message": {"role": "assistant", "content": "", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "getWeather", "arguments": {"location": "Valmorin"}},
        }]}}

    def _digest(**call):
        seen["system_prompt"] = call["system_prompt"]
        return "Je n'ai pas pu finir. Il fait 11,2 degrés à Valmorin."

    with patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory",
               return_value={}), \
         patch("src.jarvis.reply.engine.select_tools",
               return_value=["getWeather", "stop"]), \
         patch("src.jarvis.reply.engine.chat_with_messages",
               side_effect=_always_a_tool_call), \
         patch("src.jarvis.reply.engine.run_tool_with_retries",
               return_value=ToolExecutionResult(
                   success=True, reply_text="Temperature: 11.2°C")), \
         patch("src.jarvis.reply.enrichment.call_llm_direct", side_effect=_digest):
        reply = run_reply_engine(
            db=db, cfg=cfg, tts=None, text="il fait combien dehors ?",
            dialogue_memory=DialogueMemory(inactivity_timeout=300.0), **kw,
        )
    assert reply == "Je n'ai pas pu finir. Il fait 11,2 degrés à Valmorin."
    return seen["system_prompt"]


@pytest.mark.unit
class TestTheReplyWrittenWhenTheLoopRunsOut:
    """When the loop runs out of turns, the max-turn digest writes the reply,
    and on a turn from the voice that reply is spoken too."""

    def test_on_a_turn_from_the_voice_it_is_given_the_rule(self, db, tmp_path):
        system = _digest_system_prompt(db, _cfg(tmp_path), origin="voix", language="fr")

        assert SPOKEN_FORM in system

    def test_on_a_turn_from_the_text_chat_it_is_not(self, db, tmp_path):
        system = _digest_system_prompt(db, _cfg(tmp_path), origin="chat")

        assert SPOKEN_FORM not in system

    def test_its_own_instructions_are_kept_either_way(self, db, tmp_path):
        from src.jarvis.reply.enrichment import _LOOP_DIGEST_SYSTEM_PROMPT

        spoken = _digest_system_prompt(db, _cfg(tmp_path), origin="voix")
        written = _digest_system_prompt(db, _cfg(tmp_path), origin="chat")

        assert written == _LOOP_DIGEST_SYSTEM_PROMPT
        assert spoken.startswith(_LOOP_DIGEST_SYSTEM_PROMPT)
