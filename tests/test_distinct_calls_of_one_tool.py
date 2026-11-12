"""One tool, several arguments: a list is work, the same call again is a loop.

Field trace (2026-11-05, evening routine, native tools): the plan named
three `filesystem__list_directory` steps, one per folder. The first two
ran; the third, on a different folder, was refused because the tool had
"already returned 2 results this turn", and the routine told the user it
could not read one of his projects. A count by tool name cannot tell three
folders from one folder asked three times.

What a reply may run:

- every call whose arguments differ from the calls already run in this
  reply, until the turn budget (`agentic_max_turns`) is spent;
- never the same call twice: the name and arguments of every call run in
  this reply are kept, whatever came back (a result, an error, a refusal),
  and an identical call is answered from what it already returned, with a
  console line saying so.

Folder names, paths and cities are invented.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.routines.scope import RoutineScope
from src.jarvis.tools.registry import ToolSpec
from src.jarvis.tools.types import ToolExecutionResult

from tests.test_engine_tool_carryover_guard import _mock_cfg


_LIST_DIR = "filesystem__list_directory"
_ROOT = r"C:\Users\test\Projets"
_FOLDERS = {
    _ROOT: "[DIR] Lanterne\n[DIR] Boussole",
    _ROOT + r"\Lanterne": "[FILE] lanterne_marker.toml\n[DIR] src",
    _ROOT + r"\Boussole": "[FILE] boussole_marker.json\n[DIR] docs",
}


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"message": {"content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": args},
    }]}}


def _answer(text: str) -> dict:
    return {"message": {"content": text}}


def _engine_patches(select=None):
    """Everything around the loop that would otherwise call a model."""
    patches = [
        patch("src.jarvis.memory.core.format_warm_profile_block", return_value=""),
        patch("src.jarvis.memory.core.build_core_profile",
              return_value={"user": "", "directives": ""}),
        patch("src.jarvis.memory.graph.GraphMemoryStore"),
        patch("src.jarvis.reply.engine.extract_search_params_for_memory",
              return_value={}),
        patch("src.jarvis.reply.engine.digest_loop_for_max_turns",
              return_value="Not everything could be done."),
    ]
    if select is not None:
        patches.append(
            patch("src.jarvis.reply.engine.select_tools", return_value=select))
    return patches


def _run(*, cfg, text, chat, tools, plan=(), memory=None, scope=None,
         origin=None, outcome=None):
    """Run one reply. ``chat`` is the model's turns in order; ``outcome``
    maps a call's arguments to what the tool returns. Returns the reply,
    the arguments of every call that actually ran, and what the model was
    sent on each turn."""
    ran: list[dict] = []
    sent: list[list[dict]] = []
    turns = iter(chat)

    def _chat(**kw):
        sent.append([dict(m) for m in kw.get("messages") or []])
        return next(turns)

    def _tool(**kw):
        ran.append(dict(kw.get("tool_args") or {}))
        if outcome is not None:
            return outcome(kw.get("tool_args") or {})
        return ToolExecutionResult(success=True, reply_text="ok")

    patches = _engine_patches(select=None if scope is not None else tools)
    patches += [
        patch("src.jarvis.reply.engine.plan_query", return_value=list(plan)),
        patch("src.jarvis.reply.engine.chat_with_messages", side_effect=_chat),
        patch("src.jarvis.reply.engine.run_tool_with_retries", side_effect=_tool),
    ]
    for p in patches:
        p.start()
    try:
        kwargs = {}
        if scope is not None:
            kwargs["scope"] = scope
        if origin is not None:
            kwargs["origin"] = origin
        reply = run_reply_engine(
            db=Mock(), cfg=cfg, tts=None, text=text,
            dialogue_memory=memory if memory is not None else DialogueMemory(),
            **kwargs,
        )
    finally:
        for p in reversed(patches):
            p.stop()
    return reply, ran, sent


def _tool_results(messages: list[dict]) -> list[str]:
    return [m.get("content") or "" for m in messages if m.get("tool_name")]


# ── The field case ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_an_evening_routine_reads_each_of_the_three_folders_it_planned(capsys):
    """Three folders, one tool, three different paths. All three are read,
    and the model writing the summary has all three listings."""
    cfg = _mock_cfg()
    cfg.mcps = {"filesystem": {"command": "invented-fs-server"}}
    spec = ToolSpec(
        name=_LIST_DIR,
        description="List the entries of a directory.",
        inputSchema={"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]},
        annotations={"readOnlyHint": True},
    )
    paths = list(_FOLDERS)
    plan = [f"{_LIST_DIR} path='{p}'" for p in paths] + ["Reply to the user."]
    chat = [_call(_LIST_DIR, {"path": p}, f"c{i}") for i, p in enumerate(paths)]
    chat.append(_answer("Lanterne and Boussole are both in progress."))

    with patch("src.jarvis.tools.registry.get_cached_mcp_tools",
               return_value={_LIST_DIR: spec}):
        reply, ran, sent = _run(
            cfg=cfg, text="fais le bilan de mes projets", chat=chat,
            tools=None, plan=plan, origin="routine",
            scope=RoutineScope(nom="bilanDuSoir", outils=[_LIST_DIR]),
            outcome=lambda a: ToolExecutionResult(
                success=True, reply_text=_FOLDERS[a["path"]]),
        )

    assert [a["path"] for a in ran] == paths
    last_turn_results = _tool_results(sent[-1])
    for listing in _FOLDERS.values():
        assert any(listing in r for r in last_turn_results)
    assert "Refused" not in capsys.readouterr().out
    assert reply


# ── Distinct arguments are work, on both calling protocols ────────────


@pytest.mark.unit
@pytest.mark.parametrize("model", ["test-large", "test-3b"])
def test_three_cities_in_one_reply_are_all_looked_up(model):
    """`test-3b` is classified SMALL, so it runs the text-tools protocol;
    `test-large` runs native tool calls. Neither refuses the third city."""
    cfg = _mock_cfg()
    cfg.llm_chat_model = model
    cities = ["Ambrel", "Corvane", "Tessaly"]
    chat = [_call("getWeather", {"location": c}, f"w{i}")
            for i, c in enumerate(cities)]
    chat.append(_answer("Mild everywhere."))

    _, ran, _ = _run(
        cfg=cfg, text="compare the weather in Ambrel, Corvane and Tessaly",
        chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text=f"{a['location']}: 15°C."),
    )

    assert [a["location"] for a in ran] == cities


@pytest.mark.unit
def test_a_small_model_s_plan_runs_each_city_it_names():
    """The plan drives a small model: each concrete step is dispatched
    without asking the chat model, which only writes the answer."""
    cfg = _mock_cfg()
    cfg.llm_chat_model = "test-3b"
    cities = ["Ambrel", "Corvane", "Tessaly"]
    plan = [f"getWeather location='{c}'" for c in cities] + ["Reply to the user."]

    _, ran, sent = _run(
        cfg=cfg, text="compare the weather in Ambrel, Corvane and Tessaly",
        chat=[_answer("Mild everywhere.")], tools=["getWeather"], plan=plan,
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text=f"{a['location']}: 15°C."),
    )

    assert [a["location"] for a in ran] == cities
    assert len(sent) == 1


@pytest.mark.unit
def test_a_plan_step_identical_to_one_already_run_is_not_dispatched():
    cfg = _mock_cfg()
    cfg.llm_chat_model = "test-3b"
    plan = ["getWeather location='Ambrel'", "getWeather location='Ambrel'",
            "Reply to the user."]

    _, ran, _ = _run(
        cfg=cfg, text="weather in Ambrel, and in Ambrel?",
        chat=[_answer("15°C in Ambrel."), _answer("15°C in Ambrel.")],
        tools=["getWeather"], plan=plan,
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text="Ambrel: 15°C."),
    )

    assert len(ran) == 1


# ── The same call again is a loop ─────────────────────────────────────


@pytest.mark.unit
def test_the_same_call_again_is_answered_from_what_it_returned_and_said(capsys):
    cfg = _mock_cfg()
    chat = [
        _call("getWeather", {"location": "Ambrel"}, "a1"),
        _call("getWeather", {"location": "Ambrel"}, "a2"),
        _answer("15°C in Ambrel."),
    ]

    _, ran, sent = _run(
        cfg=cfg, text="weather in Ambrel?", chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text="Ambrel: 15°C."),
    )

    assert len(ran) == 1
    # The model still gets an answer to its second call, pointing it at
    # the first result rather than leaving the call unanswered.
    answered = [m for m in sent[-1]
                if m.get("role") == "tool" and m.get("tool_call_id") == "a2"]
    assert answered and answered[0].get("content")
    out = capsys.readouterr().out
    assert any("⚠️" in line and "getWeather" in line for line in out.splitlines())


def _answer_to(call_id: str, messages: list[dict]) -> str:
    found = [m.get("content") or "" for m in messages
             if m.get("role") == "tool" and m.get("tool_call_id") == call_id]
    assert len(found) == 1, f"call {call_id} answered {len(found)} times"
    return found[0]


def _repeat_line(out: str, tool: str) -> str:
    lines = [ln for ln in out.splitlines() if "⚠️" in ln and tool in ln]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.unit
def test_the_same_call_after_a_failure_is_not_run_again(capsys):
    """A call that failed fails the same way when it is repeated word for
    word inside the same reply: retrying it is the loop, not the work.
    The model's second call is still answered, the error it got the first
    time is still in front of it, and the console does not claim the call
    produced anything."""
    cfg = _mock_cfg()
    chat = [
        _call("getWeather", {"location": "Ambrel"}, "f1"),
        _call("getWeather", {"location": "Ambrel"}, "f2"),
        _answer("I could not get the weather for Ambrel."),
    ]

    _, ran, sent = _run(
        cfg=cfg, text="weather in Ambrel?", chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=False, reply_text=None,
            error_message="weather service unreachable"),
    )

    assert len(ran) == 1
    assert "getWeather" in _answer_to("f2", sent[-1])
    assert "weather service unreachable" in _answer_to("f1", sent[-1])
    line = _repeat_line(capsys.readouterr().out, "getWeather")
    assert "called" in line and " ran " not in line


@pytest.mark.unit
def test_the_same_call_after_a_refusal_is_not_run_again(capsys):
    """A call the gate refused is refused again if it reaches the gate
    again; the model is pointed at the refusal instead, and the console
    does not say that a call the gate refused ran."""
    cfg = _mock_cfg()
    refusal = "getWeather is not allowed for this assistant."
    chat = [
        _call("getWeather", {"location": "Ambrel"}, "n1"),
        _call("getWeather", {"location": "Ambrel"}, "n2"),
        _answer("You have not allowed me to check the weather."),
    ]

    _, ran, sent = _run(
        cfg=cfg, text="weather in Ambrel?", chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=False, refused=True, reply_text=refusal),
    )

    assert len(ran) == 1
    assert "getWeather" in _answer_to("n2", sent[-1])
    assert refusal in _answer_to("n1", sent[-1])
    line = _repeat_line(capsys.readouterr().out, "getWeather")
    assert "called" in line and " ran " not in line


@pytest.mark.unit
def test_the_same_call_is_caught_however_many_calls_ran_in_between():
    """The calls of the whole reply are kept, not the last few."""
    cfg = _mock_cfg()
    cities = ["Ambrel", "Corvane", "Tessaly", "Morrow", "Pellin", "Quarry"]
    assert len(cities) + 2 <= cfg.agentic_max_turns
    chat = [_call("getWeather", {"location": c}, f"m{i}")
            for i, c in enumerate(cities)]
    chat.append(_call("getWeather", {"location": cities[0]}, "again"))
    chat.append(_answer("Done."))

    _, ran, _ = _run(
        cfg=cfg, text="weather in six towns", chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text=f"{a['location']}: 15°C."),
    )

    assert [a["location"] for a in ran] == cities


# ── What stops a model that never stops ───────────────────────────────


@pytest.mark.unit
def test_a_model_that_keeps_changing_its_arguments_stops_at_the_turn_budget():
    """Every call spends one turn of `agentic_max_turns`. A model that
    never answers runs at most that many calls, and the reply says the
    request was not fully completed."""
    cfg = _mock_cfg()
    cfg.agentic_max_turns = 4
    chat = [_call("getWeather", {"location": f"Town {n}"}, f"r{n}")
            for n in range(cfg.agentic_max_turns + 3)]

    reply, ran, _ = _run(
        cfg=cfg, text="weather everywhere", chat=chat, tools=["getWeather"],
        outcome=lambda a: ToolExecutionResult(
            success=True, reply_text=f"{a['location']}: 15°C."),
    )

    assert len(ran) == cfg.agentic_max_turns
    assert reply == "Not everything could be done."
