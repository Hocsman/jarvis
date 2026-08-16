"""The way out is named in the message that says you are stuck.

Field trace (2026-08-16, deepseek-v4-flash, his machine). He asks for the
weather in Bagneux; the microphone is weak and Whisper hears "me donner
Théo sur Banyo". The router reads only that text and picks
``webSearch, fetchWebPage, remember, stop, forget``, to which the engine
appends ``toolSearchTool``. The chat model understands anyway and calls
``getWeather`` with the right argument. It is refused, and gets back:

    Error: Tool 'getWeather' is not available.
    Available tools: webSearch, fetchWebPage, remember, stop, forget...

``toolSearchTool`` is appended last, always, so a bare ``[:5]`` eats it —
and it is the one tool built for exactly this moment. Its own description
tells the model never to say "I can't do that" without calling it first.
Hiding it inside the refusal is hiding the exit inside the locked door.

That trace recovered, because ``webSearch`` substitutes for a weather
lookup. Nothing substitutes for a Chrome navigation or ``setReminder``:
there the model reads five tools, none of them fit, and it apologises for
a limit it does not have.

``stop`` is appended just before it and gets eaten the same way, which
costs the model its termination sentinel.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine

from tests.test_engine_tool_carryover_guard import _mock_cfg


# ── The message itself ─────────────────────────────────────────────────


def _message(refuse: str, allowed: list[str]) -> str:
    from src.jarvis.reply.engine import _unavailable_tool_message

    return _unavailable_tool_message(refuse, allowed)


LONGUE = ["webSearch", "fetchWebPage", "remember", "logMeal", "forget",
          "stop", "toolSearchTool"]


def test_the_escape_hatch_is_named_however_long_the_list_is():
    texte = _message("getWeather", LONGUE)

    assert "toolSearchTool" in texte


def test_the_termination_sentinel_survives_too():
    """Without `stop` the model has no way to say it is done, so it keeps
    calling tools until the turn cap."""
    texte = _message("getWeather", LONGUE)

    assert "stop" in texte


def test_the_tools_it_can_actually_use_are_still_there():
    """The control. A message that only advertised the escape hatch would
    pass the two tests above and be worse than what it replaced."""
    texte = _message("getWeather", LONGUE)

    assert "webSearch" in texte
    assert "fetchWebPage" in texte


def test_what_is_hidden_is_counted_rather_than_implied():
    """A bare "..." tells the model something is missing but not how much,
    so it cannot tell a trimmed list from a complete one. Thirty lines
    above, the console preview already does this properly."""
    huit = [f"outil{i}" for i in range(8)] + ["stop", "toolSearchTool"]

    texte = _message("getWeather", huit)

    assert "..." not in texte
    assert "3 more" in texte, texte


def test_a_trimmed_list_still_ends_on_the_escape_hatch():
    """The count must not push the hatch back out: trimming is what broke
    it in the first place."""
    huit = [f"outil{i}" for i in range(8)] + ["stop", "toolSearchTool"]

    texte = _message("getWeather", huit)

    assert "toolSearchTool" in texte
    assert "stop" in texte


def test_a_short_list_is_shown_whole_and_says_nothing_about_more():
    courte = ["webSearch", "stop", "toolSearchTool"]

    texte = _message("getWeather", courte)

    assert "webSearch" in texte
    assert "more" not in texte


def test_the_refused_name_is_the_one_it_asked_for():
    texte = _message("getWeather", LONGUE)

    assert "getWeather" in texte


def test_a_routine_that_was_denied_the_hatch_is_not_offered_it():
    """Under a routine envelope `scope.allows` strips both control tools
    on purpose: the whole point is that a routine cannot widen its own
    allow-list. Advertising a tool the gate will refuse teaches the model
    to spend a turn on a closed door."""
    sous_cadre = ["webSearch", "fetchWebPage"]

    texte = _message("getWeather", sous_cadre)

    assert "toolSearchTool" not in texte
    assert "webSearch" in texte


# ── And the same thing through the engine ──────────────────────────────


def _tour_ou_le_modele_demande_un_outil_hors_liste():
    """His trace: six tools routed, the sixth being the escape hatch, and
    a model that calls a seventh."""
    vu: dict = {}

    def _capture(**kw):
        for m in kw.get("messages") or []:
            if m.get("role") == "tool":
                vu.setdefault("outils", []).append(m)
        if "demandé" not in vu:
            vu["demandé"] = True
            return {"message": {"content": "", "tool_calls": [{
                "id": "a1", "type": "function",
                "function": {"name": "getWeather",
                             "arguments": {"location": "Bagneux, France"}},
            }]}}
        return {"message": {"content": "il fait beau."}}

    routes = ["webSearch", "fetchWebPage", "remember", "stop", "forget"]

    with patch("src.jarvis.memory.core.format_warm_profile_block", return_value=""), \
         patch("src.jarvis.memory.core.build_core_profile",
               return_value={"user": "", "directives": ""}), \
         patch("src.jarvis.memory.graph.GraphMemoryStore"), \
         patch("src.jarvis.reply.engine.plan_query", return_value=[]), \
         patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={}), \
         patch("src.jarvis.reply.engine.extract_text_from_response",
               side_effect=["", "il fait beau."]), \
         patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_capture), \
         patch("src.jarvis.reply.engine.select_tools", return_value=routes):
        run_reply_engine(db=MagicMock(), cfg=_mock_cfg(), tts=None,
                         text="la météo à Bagneux",
                         dialogue_memory=DialogueMemory())

    return "\n".join(m.get("content", "") for m in vu.get("outils", []))


def test_the_engine_hands_the_model_the_exit_it_just_needed():
    texte = _tour_ou_le_modele_demande_un_outil_hors_liste()

    assert "not available" in texte, "le refus doit atteindre le modèle"
    assert "toolSearchTool" in texte
