"""The anti-loop guard reads this reply, not the conversation.

A call identical to one already run in this reply is not run again. Dialogue
carryover replays earlier replies' tool calls and results into the same
message list, carrying the same `tool_name` and sometimes the same
arguments. None of them is a loop inside this reply: ask about the weather
in London twice and then about Tokyo, and the Tokyo call runs; ask about
London again in a later reply, and it runs again, because the weather is
read afresh each time it is asked for.

Distinct calls of one tool inside one reply are covered by
`tests/test_distinct_calls_of_one_tool.py`.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.tools.types import ToolExecutionResult

from tests.test_engine_tool_carryover_guard import _mock_cfg


def _tour_reussi(nom: str, ville: str, call_id: str) -> list[dict]:
    """A completed tool turn, in the shape carryover replays."""
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": nom, "arguments": {"location": ville}},
        }]},
        {"role": "tool", "tool_call_id": call_id, "tool_name": nom,
         "content": f"{ville}: 15°C and partly cloudy.",
         "tool_failed": False},
    ]


def _appel(nom: str, args: dict, call_id: str) -> dict:
    return {"message": {"content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": nom, "arguments": args},
    }]}}


def _memoire_avec_deux_tours() -> DialogueMemory:
    dm = DialogueMemory()
    dm.add_message("user", "quel temps à Londres ?")
    dm.record_tool_turn(_tour_reussi("getWeather", "London", "c1"))
    dm.add_message("assistant", "15°C et nuageux.")
    dm.add_message("user", "et demain ?")
    dm.record_tool_turn(_tour_reussi("getWeather", "London tomorrow", "c2"))
    dm.add_message("assistant", "16°C demain.")
    return dm


@pytest.mark.unit
@patch("src.jarvis.memory.core.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.core.build_core_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_the_first_call_of_a_turn_survives_earlier_turns(
    mock_chat, mock_extract, mock_run, _mem, _plan, _graph, _warm, _fmt,
):
    """Two prior turns used getWeather. He now asks about a different
    city. Nothing has repeated inside this reply, so nothing is a loop."""
    mock_chat.side_effect = [
        _appel("getWeather", {"location": "Tokyo"}, "c3"),
        {"message": {"content": "22°C et dégagé à Tokyo."}},
    ]
    mock_extract.side_effect = ["", "22°C et dégagé à Tokyo."]
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="Tokyo: 22°C and clear.")

    with patch("src.jarvis.reply.engine.select_tools", return_value=["getWeather"]):
        run_reply_engine(db=Mock(), cfg=_mock_cfg(), tts=None,
                         text="et à Tokyo ?", dialogue_memory=_memoire_avec_deux_tours())

    appels = [c.kwargs.get("tool_name") for c in mock_run.call_args_list]
    assert "getWeather" in appels


@pytest.mark.unit
@patch("src.jarvis.memory.core.format_warm_profile_block", return_value="")
@patch("src.jarvis.memory.core.build_core_profile",
       return_value={"user": "", "directives": ""})
@patch("src.jarvis.memory.graph.GraphMemoryStore")
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_the_same_question_in_a_later_reply_is_looked_up_again(
    mock_chat, mock_extract, mock_run, _mem, _plan, _graph, _warm, _fmt,
):
    """The carryover holds a getWeather call for London with these exact
    arguments. A new reply asking for London again is not a repeat of it:
    the weather may have changed, and nothing in this reply has run yet."""
    mock_chat.side_effect = [
        _appel("getWeather", {"location": "London"}, "c3"),
        {"message": {"content": "Toujours 15°C."}},
    ]
    mock_extract.side_effect = ["", "Toujours 15°C."]
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="London: 15°C and partly cloudy.")

    with patch("src.jarvis.reply.engine.select_tools", return_value=["getWeather"]):
        run_reply_engine(db=Mock(), cfg=_mock_cfg(), tts=None,
                         text="et à Londres, maintenant ?",
                         dialogue_memory=_memoire_avec_deux_tours())

    assert [c.kwargs.get("tool_args") for c in mock_run.call_args_list] == [
        {"location": "London"},
    ]
