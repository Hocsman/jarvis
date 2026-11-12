"""A routine that planned to read three folders reads all three.

Field trace (2026-11-05, an evening routine, native tool calls): the plan
named three `filesystem__list_directory` steps, the projects folder and two
projects in it. The third call, on its own folder, was refused because the
tool had already returned two results in that reply, and the routine told
the user it could not read one of his projects.

The engine half is pinned by `tests/test_distinct_calls_of_one_tool.py`:
whether a planned folder can be read at all is the engine's decision, not
the model's. What only a live model shows is the other half:

- every planned folder reaches the tool, on that folder;
- it stops on its own answer: no reply is the max-turn digest. The calls
  the model emits are counted, including the identical ones the engine
  answers without running them, so a model that circles is seen even
  when the engine keeps the tool from running twice;
- its reply speaks of each project from its listing and never says one
  could not be read (judged by the same model, in any language).

Two setups, one scenario, an unattended routine with a one-tool envelope:

- `test_a_routine_reads_every_folder_it_planned_and_stops` runs on the
  user's own settings (`real_model_config`: their provider and chat model,
  an empty memory, no location). Only the invented paths and listings
  below and the prompt built around them reach the provider.
- `test_the_suite_small_model_reads_every_folder_it_planned_and_stops`
  runs on the suite's small model (`mock_config`, `gemma4:e2b` on local
  Ollama unless `EVAL_JUDGE_MODEL` names another), which takes the
  text-tools path: the engine dispatches each planned step itself.

The other side of the same rule: with no count per tool name, the turn
budget is all that stops a model that keeps calling one tool on new
arguments. The small model is the one most likely to, and on the
text-tools path every result without a plan ends by inviting another call.
`test_the_suite_small_model_stops_when_a_tool_keeps_finding_nothing` asks
it questions whose tool never finds the answer (a search that matches no
page and suggests other keywords, a place the geocoder does not know),
with the planner on and off, and counts the calls of one tool in a reply
and the replies that end on the max-turn digest.

`EVAL_DISTINCT_CALLS_ROUNDS` sets the runs (3 by default, per question and
planner setting in the last test). A count under 20 runs is reported
without a rate.

    python -m pytest evals/test_distinct_calls_of_one_tool.py -s
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import rate

ROUNDS = int(os.environ.get("EVAL_DISTINCT_CALLS_ROUNDS", "3") or 3)

_TOOL = "filesystem__list_directory"
_MCPS = {"filesystem": {"command": "invented-fs-server"}}
_ROOT = r"C:\Users\test\Projets"
_LISTINGS = {
    _ROOT: "[DIR] Lanterne\n[DIR] Boussole",
    _ROOT + r"\Lanterne":
        "[FILE] Cargo.toml\n[FILE] README.md\n[DIR] src\n[DIR] tests",
    _ROOT + r"\Boussole":
        "[FILE] package.json\n[FILE] vite.config.ts\n[DIR] src\n[DIR] public",
}
_ROUTINE_PHRASE = (
    "Fais-moi le bilan du soir de mes projets : regarde le dossier "
    r"C:\Users\test\Projets et chacun des projets qu'il contient, "
    "puis dis-moi ce qu'il y a dans chacun."
)
_PLAN = [f"{_TOOL} path='{p}'" for p in _LISTINGS] + [
    "Summarise each project for the user.",
]
_JUDGE = (
    "You check a voice assistant's reply. Output only PASS or FAIL. "
    "PASS when the reply says something about what each of the two "
    "projects, Lanterne and Boussole, contains, consistent with the "
    "listings given, and nowhere says that a folder or project could not "
    "be read, listed or accessed. Judge meaning in any language."
)

# What the engine prints, read to count what the model did rather than what
# reached the tool: a call the model emits, a plan step the engine
# dispatches itself, and an identical call answered without running.
_EMITTED = "🛠️ Agent →"
_DISPATCHED = "direct-exec"
_NOT_RUN_AGAIN = "Not run again"


def _norm(path: str) -> str:
    return str(path or "").replace("/", "\\").rstrip("\\").casefold()


def _one_run(cfg, db, memory) -> dict:
    """One routine turn: the reply, the path of every call that reached
    the tool, and what the engine printed about the calls it was asked
    for."""
    from jarvis.reply import engine
    from jarvis.routines.scope import RoutineScope
    from jarvis.tools.registry import ToolSpec
    from jarvis.tools.types import ToolExecutionResult

    spec = ToolSpec(
        name=_TOOL,
        description="List the files and folders directly inside a directory.",
        inputSchema={"type": "object",
                     "properties": {"path": {"type": "string"}},
                     "required": ["path"]},
        annotations={"readOnlyHint": True},
    )
    listings = {_norm(p): text for p, text in _LISTINGS.items()}
    paths: list[str] = []
    max_turn_ending = {"hit": False}
    real_digest = engine.digest_loop_for_max_turns

    def _tool(db, cfg, tool_name, tool_args, **kwargs):
        path = (tool_args or {}).get("path", "")
        paths.append(path)
        listing = listings.get(_norm(path))
        if listing is None:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Access denied - path outside allowed directories: {path}")
        return ToolExecutionResult(success=True, reply_text=listing)

    def _digest(**kwargs):
        max_turn_ending["hit"] = True
        return real_digest(**kwargs)

    console = io.StringIO()
    with patch("jarvis.tools.registry.get_cached_mcp_tools",
               return_value={_TOOL: spec}), \
         patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_tool), \
         patch("jarvis.reply.engine.plan_query", return_value=list(_PLAN)), \
         patch("jarvis.reply.engine.digest_loop_for_max_turns", side_effect=_digest), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         contextlib.redirect_stdout(console):
        reply = engine.run_reply_engine(
            db=db, cfg=cfg, tts=None, text=_ROUTINE_PHRASE,
            dialogue_memory=memory, origin="routine",
            scope=RoutineScope(nom="bilanDuSoir", outils=[_TOOL]),
        )
    lines = console.getvalue().splitlines()
    return {
        "reply": reply,
        "paths": paths,
        "emitted": sum(1 for ln in lines if _EMITTED in ln),
        "dispatched": sum(1 for ln in lines if _DISPATCHED in ln),
        "not_run_again": sum(1 for ln in lines if _NOT_RUN_AGAIN in ln),
        "max_turns": max_turn_ending["hit"],
    }


def _judged(cfg, reply: str) -> bool | None:
    from jarvis.llm import get_llm_backend

    listings = "\n\n".join(f"{p}:\n{t}" for p, t in _LISTINGS.items())
    verdict = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, _JUDGE,
        f"Listings the assistant read:\n{listings}\n\nReply:\n{reply}",
        timeout_sec=60.0,
    )
    if not verdict:
        return None
    return verdict.strip().upper().startswith("PASS")


def _measure(cfg) -> None:
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database

    budget = int(cfg.agentic_max_turns)
    planned = [_norm(p) for p in _LISTINGS]

    read_all = grounded = unjudged = max_turn_endings = 0
    emitted = dispatched = not_run_again = 0
    most_calls = 0
    for run in range(1, ROUNDS + 1):
        db = Database(":memory:", sqlite_vss_path=None)
        try:
            r = _one_run(cfg, db, DialogueMemory(inactivity_timeout=300))
        finally:
            db.close()
        reply = r["reply"]
        seen = [_norm(p) for p in r["paths"]]
        all_read = all(p in seen for p in planned)
        read_all += int(all_read)
        max_turn_endings += int(r["max_turns"])
        emitted += r["emitted"]
        dispatched += r["dispatched"]
        not_run_again += r["not_run_again"]
        calls = r["emitted"] + r["dispatched"]
        most_calls = max(most_calls, calls)
        verdict = _judged(cfg, reply or "") if reply else False
        unjudged += int(verdict is None)
        grounded += int(bool(verdict))
        print(f"   🧪 run {run}: {'✅' if all_read else '❌'} {len(seen)} run, "
              f"{r['emitted']} emitted, {r['dispatched']} dispatched, "
              f"{r['not_run_again']} not run again"
              f"{', max-turn digest' if r['max_turns'] else ''}, "
              f"judge {verdict} — {(reply or '')[:90]!r}")
        if verdict is False:
            # The whole reply, so a judge's mistake can be told from the
            # model's: the listings and the reply are invented.
            print("      " + (reply or "(no reply)").replace("\n", "\n      "))

    print(f"\n📊 {cfg.llm_chat_model}, {ROUNDS} run(s):"
          f"\n   📂 read all three planned folders: {rate(read_all, ROUNDS)}"
          f"\n   🗣️ reply grounded in each listing: {rate(grounded, ROUNDS)}"
          f" (no verdict {unjudged})"
          f"\n   🧮 calls: {emitted} emitted by the model, {dispatched} dispatched from the plan,"
          f" {not_run_again} identical ones answered without running;"
          f" most in one run {most_calls} (turn budget {budget})"
          f"\n   🛑 replies ending on the max-turn digest: {max_turn_endings}/{ROUNDS}")

    assert read_all == ROUNDS, "a planned folder was never read"
    assert max_turn_endings == 0, "a reply ran out of turns instead of answering"
    assert grounded + unjudged == ROUNDS, "a reply said a folder could not be read, or invented its contents"


@pytest.mark.eval
def test_a_routine_reads_every_folder_it_planned_and_stops(real_model_config):
    _measure(dataclasses.replace(real_model_config, mcps=dict(_MCPS)))


@pytest.mark.eval
@requires_judge_llm
def test_the_suite_small_model_reads_every_folder_it_planned_and_stops(mock_config):
    mock_config.mcps = dict(_MCPS)
    _measure(mock_config)


# ── A tool that keeps finding nothing ─────────────────────────────────


def _no_page_matched(tool_args):
    from jarvis.tools.types import ToolExecutionResult

    query = " ".join(str(v) for v in (tool_args or {}).values())
    return ToolExecutionResult(success=True, reply_text=(
        f"Web search results for '{query}':\nNo page matched this query. "
        "Try different or more specific keywords.\n"))


def _unknown_place(tool_args):
    from jarvis.tools.types import ToolExecutionResult

    place = (tool_args or {}).get("location", "")
    return ToolExecutionResult(
        success=False,
        reply_text=f"No place named '{place}' was found by the geocoder.")


_NOTHING_FOUND = [
    ("What time does the Ambrel municipal library open on Sundays?",
     "webSearch", _no_page_matched),
    ("What's the weather like in Ambrel today?",
     "getWeather", _unknown_place),
]


def _one_reply(cfg, db, question, tool, outcome) -> dict:
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.reply import engine

    calls: list[str] = []
    max_turn_ending = {"hit": False}
    real_digest = engine.digest_loop_for_max_turns

    def _tool(db, cfg, tool_name, tool_args, **kwargs):
        calls.append(tool_name)
        return outcome(tool_args)

    def _digest(**kwargs):
        max_turn_ending["hit"] = True
        return real_digest(**kwargs)

    console = io.StringIO()
    with patch("jarvis.reply.engine.select_tools", return_value=[tool, "stop"]), \
         patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_tool), \
         patch("jarvis.reply.engine.digest_loop_for_max_turns", side_effect=_digest), \
         patch("jarvis.reply.engine.get_location_context_with_timezone",
               return_value=("Location: Corvane", None)), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}), \
         contextlib.redirect_stdout(console):
        reply = engine.run_reply_engine(
            db=db, cfg=cfg, tts=None, text=question,
            dialogue_memory=DialogueMemory(inactivity_timeout=300),
        )
    lines = console.getvalue().splitlines()
    return {
        "reply": reply,
        "runs": calls.count(tool),
        "emitted": sum(1 for ln in lines if _EMITTED in ln),
        "not_run_again": sum(1 for ln in lines if _NOT_RUN_AGAIN in ln),
        "max_turns": max_turn_ending["hit"],
    }


@pytest.mark.eval
@requires_judge_llm
def test_the_suite_small_model_stops_when_a_tool_keeps_finding_nothing(mock_config):
    from jarvis.memory.db import Database

    budget = int(mock_config.agentic_max_turns)
    total = max_turn_endings = 0
    by_runs: dict[int, int] = {}
    for question, tool, outcome in _NOTHING_FOUND:
        for planner in (True, False):
            mock_config.planner_enabled = planner
            label = f"{tool}, planner {'on' if planner else 'off'}"
            for run in range(1, ROUNDS + 1):
                db = Database(":memory:", sqlite_vss_path=None)
                try:
                    r = _one_reply(mock_config, db, question, tool, outcome)
                finally:
                    db.close()
                total += 1
                max_turn_endings += int(r["max_turns"])
                by_runs[r["runs"]] = by_runs.get(r["runs"], 0) + 1
                print(f"   🧪 {label}, run {run}: {r['runs']} run, "
                      f"{r['emitted']} emitted, {r['not_run_again']} not run again"
                      f"{', max-turn digest' if r['max_turns'] else ''}"
                      f" — {(r['reply'] or '')[:80]!r}")

    spread = ", ".join(f"{k} with {n} call(s)" for n, k in sorted(by_runs.items()))
    print(f"\n📊 {mock_config.llm_chat_model}, {total} replies:"
          f"\n   🔁 calls of the one tool per reply: {spread} (turn budget {budget})"
          f"\n   🛑 replies ending on the max-turn digest: {rate(max_turn_endings, total)}")

    assert max_turn_endings == 0, "a reply ran out of turns calling a tool that finds nothing"
