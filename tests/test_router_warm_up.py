"""The tool router is warmed on its own prompt before she says she is listening.

The router's prompt is a system prompt and a catalogue of every tool, a few
thousand tokens that are the same from one call to the next. A warm-up that
only loads the model leaves all of it to the first real call, which then pays
the model's first computation and the whole prompt at once, at the moment the
rest of the start-up (speech, the face) is busy with the same GPU. Measured on
a local qwen2.5:3b: 512 of 3,387 prompt tokens read in the 8 s the router is
given, twice, both on the first exchange after a start.

So the warm-up sends what the first router call will send, through the backend
it will use, and the call that follows only has to read what differs: the
facts, the dialogue and the sentence.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class _Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class _Spec:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


def _catalogue():
    return {
        "getWeather": _Tool("getWeather", "Get current weather conditions for a place."),
        "fetchMeals": _Tool("fetchMeals", "Retrieve the meals the user logged."),
        "webSearch": _Tool("webSearch", "Search the web for current information."),
        "stop": _Tool("stop", "End the current conversation."),
        "forget": _Tool("forget", "Retire a stored fact."),
        "remember": _Tool("remember", "Save something to long-term memory."),
    }


def _mcp():
    return {"lights__turn_on": _Spec("lights__turn_on", "Turn on a smart home light.")}


class _RecordingBackend:
    """A backend that remembers every warm-up and every router call."""

    def __init__(self, answer: str = "getWeather", warm_ok: bool = True):
        self.warmed: list[dict] = []
        self.asked: list[dict] = []
        self._answer = answer
        self._warm_ok = warm_ok

    def warm_up(self, model, timeout_sec=60.0, keep_alive="30m", prompt=None):
        self.warmed.append(
            {"model": model, "timeout_sec": timeout_sec, "keep_alive": keep_alive, "prompt": prompt}
        )
        return self._warm_ok

    def direct(self, chat_model, system_prompt, user_content, timeout_sec=10.0, **_):
        self.asked.append({"model": chat_model, "system": system_prompt, "user": user_content})
        return self._answer


def _shared_opening(warmed: dict, asked: dict) -> str:
    """What the router call repeats of the warm-up, from the first character."""
    warm_system, warm_user = warmed["prompt"]
    assert asked["system"] == warm_system, "the system prompt differs, nothing is reused"
    return os.path.commonprefix([warm_user, asked["user"]])


# ── The router's own prompt ───────────────────────────────────────────


@pytest.mark.unit
def test_the_first_router_call_repeats_the_whole_catalogue_the_warm_up_sent():
    from jarvis.tools.selection import ToolSelectionStrategy, select_tools, warm_up_router

    backend = _RecordingBackend()
    catalogue, mcp = _catalogue(), _mcp()

    assert warm_up_router(
        catalogue, mcp, llm_backend=backend, llm_model="router-model",
        timeout_sec=5.0, keep_alive="30m",
    ) is True
    select_tools(
        "what's the weather in Paris tomorrow", catalogue, mcp,
        strategy=ToolSelectionStrategy.LLM,
        llm_backend=backend, llm_model="router-model", llm_timeout_sec=5.0,
        context_hint=(
            "Current time: Friday 6 November 2026, 09:12\n"
            "Recent dialogue (short-term memory):\nuser: hello\nassistant: hello"
        ),
    )

    assert len(backend.warmed) == 1 and len(backend.asked) == 1
    shared = _shared_opening(backend.warmed[0], backend.asked[0])
    routed = [n for n in list(catalogue) + list(mcp) if n not in ("stop", "forget", "remember")]
    for name in routed:
        assert f"- {name}:" in shared, f"{name} is read again by the first call"


@pytest.mark.unit
def test_the_warm_up_goes_to_the_routers_model_with_the_residency_it_was_given():
    from jarvis.tools.selection import warm_up_router

    backend = _RecordingBackend()

    warm_up_router(
        _catalogue(), {}, llm_backend=backend, llm_model="router-model",
        timeout_sec=42.0, keep_alive="1m",
    )

    assert backend.warmed[0]["model"] == "router-model"
    assert backend.warmed[0]["keep_alive"] == "1m"
    assert backend.warmed[0]["timeout_sec"] == 42.0


@pytest.mark.unit
def test_a_warm_up_without_a_model_asks_nothing():
    from jarvis.tools.selection import warm_up_router

    backend = _RecordingBackend()

    assert warm_up_router(
        _catalogue(), {}, llm_backend=backend, llm_model="  ",
        timeout_sec=5.0, keep_alive="30m",
    ) is False
    assert backend.warmed == []


@pytest.mark.unit
def test_a_warm_up_that_fails_says_so():
    from jarvis.tools.selection import warm_up_router

    backend = _RecordingBackend(warm_ok=False)

    assert warm_up_router(
        _catalogue(), {}, llm_backend=backend, llm_model="router-model",
        timeout_sec=5.0, keep_alive="30m",
    ) is False


@pytest.mark.unit
def test_a_warm_up_that_raises_says_so_and_goes_no_further():
    """Start-up never stops on a warm-up: a backend that raises is a warm-up
    that failed."""
    from jarvis.tools.selection import warm_up_router

    class _Broken:
        def warm_up(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    assert warm_up_router(
        _catalogue(), {}, llm_backend=_Broken(), llm_model="router-model",
        timeout_sec=5.0, keep_alive="30m",
    ) is False


# ── The warm-up the listener runs, against the engine's own call ──────


@pytest.mark.unit
def test_the_start_up_warm_up_covers_what_the_reply_engines_router_call_sends(
    mock_config, db, dialogue_memory
):
    """The warm-up resolves its model, backend and catalogue on its own, apart
    from the reply engine. The two must still send the same opening, or the
    warm-up computes a prompt nobody asks for."""
    from jarvis.reply.engine import run_reply_engine, warm_up_tool_router
    from jarvis.tools.registry import offered_builtin_tools
    from jarvis.tools.selection import _ALWAYS_INCLUDED

    backend = _RecordingBackend(answer="getWeather")
    mock_config.tool_selection_strategy = "llm"
    mock_config.tool_router_model = "router-model"

    with patch("jarvis.reply.engine.get_auxiliary_backend", return_value=backend):
        assert warm_up_tool_router(mock_config, timeout_sec=5.0, keep_alive="30m") is True
        with patch("jarvis.reply.engine.plan_query", return_value=[]), \
             patch("jarvis.reply.engine.chat_with_messages",
                   return_value={"message": {"content": "Il fait beau.", "role": "assistant"}}), \
             patch("jarvis.reply.engine.extract_search_params_for_memory",
                   return_value={"keywords": []}):
            run_reply_engine(
                db=db, cfg=mock_config, tts=None, text="what is the weather in Paris",
                dialogue_memory=dialogue_memory,
            )

    warmed = backend.warmed[0]
    assert warmed["model"] == "router-model"
    router_calls = [a for a in backend.asked if a["system"] == warmed["prompt"][0]]
    assert router_calls, "the reply engine's router call never came"
    shared = _shared_opening(warmed, router_calls[0])
    for name in offered_builtin_tools(mock_config):
        if name not in _ALWAYS_INCLUDED:
            assert f"- {name}:" in shared, f"{name} is read again by the first call"


@pytest.mark.unit
def test_there_is_nothing_to_warm_without_a_router_model(mock_config):
    from jarvis.reply.engine import warm_up_tool_router

    backend = _RecordingBackend()
    mock_config.tool_router_model = ""
    mock_config.intent_judge_model = ""
    mock_config.llm_chat_model = ""

    with patch("jarvis.reply.engine.get_auxiliary_backend", return_value=backend):
        assert warm_up_tool_router(mock_config, timeout_sec=5.0, keep_alive="30m") is False

    assert backend.warmed == []
