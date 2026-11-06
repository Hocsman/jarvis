"""
End-to-end eval: a turn whose tool router gave up, and whether it still
reaches the tool it needs.

When the router gives up (it timed out, or answered nothing it could use),
the keyword strategy picks in its place. A sentence that shares no word with
any tool description, which is most sentences in a language other than the
descriptions', gets the always-included tools alone (`stop`, `forget`,
`remember`), and the engine adds `toolSearchTool` beside them
(`src/jarvis/tools/selection.spec.md`, keyword step 5 and LLM step 7). The
turn then depends on the chat model looking for the tool it needs, or on the
planner naming it.

Two arms over the same sentences, interleaved call by call:

- `bounded`: what the engine offers on such a turn, the degraded keyword
  pick for the sentence, through the real `select_tools`, with a router
  that gave up;
- `catalogue`: every offered tool, the other answer a no-match could give.

A run reaches the tool when `getWeather` is called. `toolSearchTool` answers
as it does when its own search works (it names `getWeather`), so what is
measured is whether the chat model looks, then uses what it found. Tool
implementations are canned; the chat model and the planner run for real.
`EVAL_DEGRADED_TURN_MCP_TOOLS` adds invented MCP tools to the catalogue.

Measured on 2026-11-06, 22 runs an arm (11 rounds over two French
sentences), arms interleaved, Ollama 0.40.14 and OpenRouter:

| Chat model | Catalogue | Arm | `getWeather` called | through `toolSearchTool` |
|---|---|---|---|---|
| `gemma4:e2b` (local, text tools) | 22 builtins | bounded | 0/22 | 0/22 |
| `gemma4:e2b` | 22 builtins | catalogue | 13/22 | 0/22 |
| `gemma4:e2b` | 112 tools | bounded | 0/22 | 0/22 |
| `gemma4:e2b` | 112 tools | catalogue | 22/22 | 0/22 |
| `deepseek/deepseek-v4-flash` (native tools) | 22 builtins | bounded | 18/22 | 18/22 |
| `deepseek/deepseek-v4-flash` | 22 builtins | catalogue | 22/22 | 0/22 |
| `deepseek/deepseek-v4-flash` | 112 tools | bounded | 20/22 | 20/22 |
| `deepseek/deepseek-v4-flash` | 112 tools | catalogue | 22/22 | 0/22 |

`gemma4:e2b` never called `toolSearchTool`: on the bounded arm it answered
without a tool or ran `webSearch`, which the planner named. The large model
looked through `toolSearchTool` whenever it reached the weather, and its
misses ran `webSearch` instead.

Run:
    ./scripts/run_evals.sh a_degraded_turn_finds_its_tool
    EVAL_DEGRADED_TURN_ROUNDS=11 ./scripts/run_evals.sh a_degraded_turn_finds_its_tool
    EVAL_DEGRADED_TURN_ROUNDS=11 EVAL_DEGRADED_TURN_MCP_TOOLS=90 \
        ./scripts/run_evals.sh a_degraded_turn_finds_its_tool
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=deepseek/deepseek-v4-flash \
    EVAL_DEGRADED_TURN_ROUNDS=11 ./scripts/run_evals.sh a_degraded_turn_finds_its_tool
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, ToolCallCapture
from jarvis.reply.prompts import ModelSize, detect_model_size

# Two ways of asking for tomorrow's weather that share no word with any tool
# description: the keyword strategy matches nothing in either, so the
# degraded pick is the always-included tools alone (asserted below, so a
# change to a description that makes one of them match is seen here).
SENTENCES = [
    "Quel temps fait-il à Lyon demain matin ?",
    "Est-ce qu'il fera beau demain à Lyon ?",
]

ROUNDS = int(os.environ.get("EVAL_DEGRADED_TURN_ROUNDS", "0") or 0)

# Invented MCP tools added to the catalogue, for the size a few enabled
# servers give (112 tools with 90 of them). None of
# their words is one of the sentences', so the degraded pick stays the same.
MCP_TOOLS = int(os.environ.get("EVAL_DEGRADED_TURN_MCP_TOOLS", "0") or 0)

_INVENTED_SERVERS = (
    "notes", "lights", "music", "calendar", "mail", "files", "browser", "tasks",
    "contacts", "photos", "printer", "thermostat", "vacuum", "garage", "podcasts",
)
_INVENTED_ACTIONS = ("list", "get", "create", "update", "delete", "search")


def _invented_mcp_tools(count: int) -> dict:
    from jarvis.tools.registry import ToolSpec

    tools = {}
    for server in _INVENTED_SERVERS:
        for action in _INVENTED_ACTIONS:
            if len(tools) >= count:
                return tools
            name = f"{server}__{action}_items"
            tools[name] = ToolSpec(
                name,
                f"{action.capitalize()} {server} items through the {server} server.",
                {"type": "object", "properties": {"item": {"type": "string"}}},
            )
    return tools

MOCK_WEATHER_LYON = (
    "Weather for Lyon, Auvergne-Rhône-Alpes, France, tomorrow:\n"
    "Morning: light rain, 9C\n"
    "Afternoon: cloudy, 13C\n"
    "Wind: 15 km/h from the north\n"
)

MOCK_SEARCH_FOUND_WEATHER = (
    "getWeather: Current weather conditions and the forecast for a place."
)


class _RouterThatGaveUp:
    """What a router backend returns when its deadline expires."""

    def direct(self, *args, **kwargs):
        return None


def _bounded_selection(**kwargs):
    """The engine's own degraded pick: the real `select_tools`, with a router
    that gave up."""
    from jarvis.tools.selection import select_tools

    kwargs["llm_backend"] = _RouterThatGaveUp()
    return select_tools(**kwargs)


def _whole_catalogue(**kwargs):
    """The other answer a no-match could give: every offered tool."""
    from jarvis.tools.selection import ToolSelection

    names = list(kwargs["builtin_tools"]) + list(kwargs.get("mcp_tools") or {})
    return ToolSelection.fallback(names)


ARMS = {"bounded": _bounded_selection, "catalogue": _whole_catalogue}


def _respond(name, args):
    if name == "toolSearchTool":
        return MOCK_SEARCH_FOUND_WEATHER
    if name == "getWeather":
        return MOCK_WEATHER_LYON
    if name == "webSearch":
        # A substitute that answers, worse than the right tool would.
        return "Web search results for 'météo Lyon': forecast pages for Lyon.\n"
    return "OK"


def _run_turn(arm: str, sentence: str, mock_config, eval_db, eval_dialogue_memory):
    """One reply to ``sentence`` with the ``arm``'s allow-list; returns the
    names of the tools the turn called, in order."""
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.tool_selection_strategy = "llm"
    mock_config.tool_search_max_calls = 3
    mcp = _invented_mcp_tools(MCP_TOOLS)
    mock_config.mcps = {name.split("__")[0]: {"command": "invented"} for name in mcp}
    capture = ToolCallCapture()

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=True, reply_text=_respond(tool_name, tool_args or {}))

    with patch("jarvis.reply.engine.select_tools", side_effect=ARMS[arm]), \
         patch("jarvis.tools.registry.get_cached_mcp_tools", return_value=mcp), \
         patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         patch("jarvis.reply.engine.get_location_context_with_timezone",
               return_value=("Location: Lyon, France", None)):
        run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None, text=sentence,
            dialogue_memory=eval_dialogue_memory,
        )
    return capture.tool_names()


def _through_the_hatch(names: list[str]) -> bool:
    return (
        "toolSearchTool" in names and "getWeather" in names
        and names.index("toolSearchTool") < names.index("getWeather")
    )


@pytest.mark.unit
@pytest.mark.parametrize("mcp_count", [0, 90])
@pytest.mark.parametrize("sentence", SENTENCES)
def test_the_sentence_reaches_the_degraded_pick_with_nothing_matched(sentence, mcp_count, mock_config):
    """The scenario holds only while no tool description shares a word with
    the sentence."""
    from jarvis.tools.registry import offered_builtin_tools
    from jarvis.tools.selection import _ALWAYS_INCLUDED, ToolSelectionStrategy

    catalogue = offered_builtin_tools(mock_config)
    pick = _bounded_selection(
        query=sentence, builtin_tools=catalogue, mcp_tools=_invented_mcp_tools(mcp_count),
        strategy=ToolSelectionStrategy.LLM, llm_model="router-model",
    )

    assert pick.degraded is True
    assert list(pick) == [t for t in _ALWAYS_INCLUDED if t in catalogue]


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.xfail(
    detect_model_size(JUDGE_MODEL) == ModelSize.SMALL,
    reason="a small chat model does not look through toolSearchTool when "
           "the turn offers it nothing else: never, in the runs measured "
           "above (selection.spec.md, LLM step 7)",
    strict=False,
)
@pytest.mark.parametrize("sentence", SENTENCES)
def test_a_turn_whose_router_gave_up_reaches_the_weather_tool(
    sentence, mock_config, eval_db, eval_dialogue_memory
):
    names = _run_turn("bounded", sentence, mock_config, eval_db, eval_dialogue_memory)
    print(f"\n📊 degraded turn: {names}")

    assert "getWeather" in names, (
        "A turn offered only the always-included tools and toolSearchTool "
        f"must look for the weather tool and use it. Tools called: {names}"
    )


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.skipif(ROUNDS <= 0, reason="set EVAL_DEGRADED_TURN_ROUNDS to compare the arms")
def test_the_bounded_turn_against_the_whole_catalogue(mock_config):
    """Arms interleaved call by call, so both share the provider's hour.
    Each run starts from an empty conversation and an empty database, so no
    run reads another's."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database

    tallies = {arm: {"runs": 0, "weather": 0, "hatch": 0} for arm in ARMS}
    for round_no in range(ROUNDS):
        for sentence in SENTENCES:
            for arm in ARMS:
                db = Database(":memory:", sqlite_vss_path=None)
                try:
                    names = _run_turn(
                        arm, sentence, mock_config, db, DialogueMemory(inactivity_timeout=300),
                    )
                finally:
                    db.close()
                tally = tallies[arm]
                tally["runs"] += 1
                tally["weather"] += "getWeather" in names
                tally["hatch"] += _through_the_hatch(names)
                print(f"   r{round_no} {arm:9} {names}", flush=True)

    print(f"\n📊 degraded turn, {JUDGE_MODEL}, {MCP_TOOLS} invented MCP tools:")
    for arm, tally in tallies.items():
        print(
            f"   {arm:9} getWeather {tally['weather']}/{tally['runs']}, "
            f"through toolSearchTool {tally['hatch']}/{tally['runs']}"
        )
