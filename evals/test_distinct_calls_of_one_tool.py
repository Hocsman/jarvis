"""A routine that planned to read three folders reads all three.

Field trace (2026-11-05, an evening routine, native tool calls): the plan
named three `filesystem__list_directory` steps, the projects folder and two
projects in it. The third call, on its own folder, was refused because the
tool had already returned two results in that reply, and the routine told
the user it could not read one of his projects.

The engine half is pinned by `tests/test_distinct_calls_of_one_tool.py`:
with a count by tool name, the third folder is never read whatever the
model does, so there is no "before" arm to sample. What only a live model
shows is the other half:

- it calls the tool once per planned folder, on that folder;
- it stops: no identical call, never more calls than the turn budget;
- its reply speaks of each project from its listing and never says one
  could not be read (judged by the same model, in any language).

It runs on the user's own settings (`real_model_config`: their provider
and chat model, an empty memory, no location), as an unattended routine
with a one-tool envelope. Only the invented paths and listings below and
the prompt built around them reach the provider.

`EVAL_DISTINCT_CALLS_ROUNDS` sets the runs (3 by default). A count under
20 runs is reported without a rate.

    python -m pytest evals/test_distinct_calls_of_one_tool.py -s
"""

from __future__ import annotations

import dataclasses
import os
from unittest.mock import patch

import pytest

ROUNDS = int(os.environ.get("EVAL_DISTINCT_CALLS_ROUNDS", "3") or 3)

_TOOL = "filesystem__list_directory"
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


def _norm(path: str) -> str:
    return str(path or "").replace("/", "\\").rstrip("\\").casefold()


def _rate(count: int, runs: int) -> str:
    return f" ({count / runs:.0%})" if runs >= 20 else ""


def _one_run(cfg, db, memory):
    """One routine turn. Returns the reply and the path of every call that
    reached the tool, in order."""
    from jarvis.reply.engine import run_reply_engine
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

    def _tool(db, cfg, tool_name, tool_args, **kwargs):
        path = (tool_args or {}).get("path", "")
        paths.append(path)
        listing = listings.get(_norm(path))
        if listing is None:
            return ToolExecutionResult(
                success=False, reply_text=None,
                error_message=f"Access denied - path outside allowed directories: {path}")
        return ToolExecutionResult(success=True, reply_text=listing)

    with patch("jarvis.tools.registry.get_cached_mcp_tools",
               return_value={_TOOL: spec}), \
         patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_tool), \
         patch("jarvis.reply.engine.plan_query", return_value=list(_PLAN)), \
         patch("jarvis.reply.engine.extract_search_params_for_memory",
               return_value={"keywords": []}):
        reply = run_reply_engine(
            db=db, cfg=cfg, tts=None, text=_ROUTINE_PHRASE,
            dialogue_memory=memory, origin="routine",
            scope=RoutineScope(nom="bilanDuSoir", outils=[_TOOL]),
        )
    return reply, paths


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


@pytest.mark.eval
def test_a_routine_reads_every_folder_it_planned_and_stops(real_model_config):
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database

    cfg = dataclasses.replace(
        real_model_config, mcps={"filesystem": {"command": "invented-fs-server"}},
    )
    budget = int(cfg.agentic_max_turns)
    planned = [_norm(p) for p in _LISTINGS]

    read_all = grounded = unjudged = repeated = 0
    most_calls = 0
    for run in range(1, ROUNDS + 1):
        db = Database(":memory:", sqlite_vss_path=None)
        try:
            reply, paths = _one_run(cfg, db, DialogueMemory(inactivity_timeout=300))
        finally:
            db.close()
        seen = [_norm(p) for p in paths]
        all_read = all(p in seen for p in planned)
        read_all += int(all_read)
        repeated += int(len(seen) != len(set(seen)))
        most_calls = max(most_calls, len(seen))
        verdict = _judged(cfg, reply or "") if reply else False
        unjudged += int(verdict is None)
        grounded += int(bool(verdict))
        print(f"   🧪 run {run}: {'✅' if all_read else '❌'} {len(seen)} call(s), "
              f"judge {verdict} — {(reply or '')[:90]!r}")

    print(f"\n📊 {cfg.llm_chat_model}, {ROUNDS} run(s):"
          f"\n   📂 read all three planned folders: {read_all}/{ROUNDS}{_rate(read_all, ROUNDS)}"
          f"\n   🗣️ reply grounded in each listing: {grounded}/{ROUNDS}{_rate(grounded, ROUNDS)}"
          f" (no verdict {unjudged})"
          f"\n   🔁 runs with an identical call reaching the tool: {repeated}"
          f"\n   🧮 most calls in one run: {most_calls} (turn budget {budget})")

    assert repeated == 0, "an identical call reached the tool twice in one reply"
    assert most_calls <= budget
    assert read_all == ROUNDS, "a planned folder was never read"
    assert grounded + unjudged == ROUNDS, "a reply said a folder could not be read, or invented its contents"
