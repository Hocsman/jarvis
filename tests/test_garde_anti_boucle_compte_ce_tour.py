"""The anti-loop guard counts this reply, not the conversation.

A tool is refused after two results already came back for it. The count
walked the last ten messages, and dialogue carryover replays earlier
turns' tool results into that same list carrying the same `tool_name`.
Ask about the weather in London twice and then ask about Tokyo, and the
first call of the new turn is refused before it runs — the model is told
to use results it does not have, and answers from nothing.

The engine already knows where the current reply starts: `user_msg_index`
is the split it uses for tool carry-over recording and for the end-of-loop
digest. The guard becomes the third user of that same line.
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
def test_a_real_repeat_inside_one_reply_is_still_refused_and_said(
    mock_chat, mock_extract, mock_run, _mem, _plan, _graph, _warm, _fmt, capsys,
):
    """The guard stays alive, and stops being mute. Three calls in one
    reply, different arguments each time so the exact-signature guard
    does not catch them first."""
    mock_chat.side_effect = [
        _appel("getWeather", {"location": "Tokyo"}, "a1"),
        _appel("getWeather", {"location": "Osaka"}, "a2"),
        _appel("getWeather", {"location": "Kyoto"}, "a3"),
        {"message": {"content": "Voilà."}},
    ]
    mock_extract.side_effect = ["", "", "", "Voilà."]
    mock_run.return_value = ToolExecutionResult(
        success=True, reply_text="22°C and clear.")

    with patch("src.jarvis.reply.engine.select_tools", return_value=["getWeather"]):
        run_reply_engine(db=Mock(), cfg=_mock_cfg(), tts=None,
                         text="le temps au Japon ?", dialogue_memory=DialogueMemory())

    assert len(mock_run.call_args_list) == 2
    assert "⚠️" in capsys.readouterr().out
