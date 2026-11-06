"""
End-to-end eval: asked whether she can work on his projects, she knows she
can, and a request for work on one reaches Claude Code.

The tool router is a small model. To a question about what she can do
("can you get into my code?") it offers no tool, and on a large catalogue it
often offers no tool for a request on a project either. The chat model is
then shown only what every turn carries (`stop`, `forget`, `remember`,
`toolSearchTool`) and answers from its prior: an assistant has no access to
anyone's files. With handing projects to Claude Code switched on, that is
false. So while `claudeCode` is offered, the system prompt says what she can
do and how to reach the tool when it is not among her tools this turn
(`src/jarvis/projets/projets.spec.md`, What she knows she can do).

Every turn here meets that worst case: the router's model answers 'none'
(through the real `select_tools`, so the URL cue still applies).
`toolSearchTool` runs for real, its router on the user's own router model,
over the builtins, `EVAL_PROJECTS_MCP_TOOLS` invented MCP tools (89 by
default, the size of a catalogue with four servers) and one invented write
tool shaped like a code host's (`repo__create_or_update_file`). `claudeCode`
gets the tool's own decision (`vet_call`) and starts nothing; every other
tool answers "OK".

Three kinds of sentence, all invented:

- a question about what she can do: passes when a judge reads the reply as
  saying she can work on the projects, and no job was started;
- a request for work on a named project: passes when a job would start on
  that project;
- a sentence that mentions code, a project or the product without asking
  for work on a project: passes when no job would start.

`TestSheKnowsSheCan` runs the engine as it ships. `TestBeforeAndAfter`
compares the prompt without the block and with it, the two arms alternating
call by call in one loop (CLAUDE.md: a provider drifts within a session).
`TestAWriteToolOffered` runs the engine as it ships on the requests alone,
the pre-loop router answering the write tool instead of 'none': a request
passes when a job starts on its project and the write tool was not called.
Set `EVAL_PROJECTS_ROUNDS` to run those two; each round asks every sentence
once on each arm, so five rounds give 20 calls an arm for each kind.

They take ``real_model_config``: the user's provider and models (the chat
model, the planner and the judge on it; the router's model on local Ollama),
sandboxed memory, location off. What leaves the machine is the invented
sentences and the prompt built around an empty memory.

Run:
    ./scripts/run_evals.sh --single projects_question_finds_claude_code
    EVAL_PROJECTS_ROUNDS=5 ./scripts/run_evals.sh --single "projects_question_finds_claude_code and before"
    EVAL_PROJECTS_ROUNDS=5 ./scripts/run_evals.sh --single "projects_question_finds_claude_code and write_tool"
"""

from __future__ import annotations

import dataclasses
import os
import sys
from contextlib import nullcontext
from typing import Dict, List, Tuple
from unittest.mock import patch

import pytest

ROUNDS = int(os.environ.get("EVAL_PROJECTS_ROUNDS", "0") or 0)
MCP_TOOLS = int(os.environ.get("EVAL_PROJECTS_MCP_TOOLS", "89") or 0)

# Below this many calls an arm's rate says nothing (CLAUDE.md, sample sizes).
_MIN_CALLS_FOR_A_RATE = 20

PROJECTS = ("quillon", "baladeur", "orchidee")

CAPABILITY = [
    "Tu peux mettre les mains dans mes projets de code pour les modifier ?",
    "Est-ce que tu as la main sur mes dépôts de code, toi ?",
    "Can you get into my code projects and change things?",
    "¿Puedes acceder a mis proyectos de código para hacer cambios?",
]

# (sentence, the project a job must start on)
REQUESTS = [
    ("Corrige le bug d'affichage des dates dans le projet quillon", "quillon"),
    ("Ajoute une page de contact au projet baladeur", "baladeur"),
    ("Fix the failing login test in the orchidee project", "orchidee"),
    ("Füge im Projekt quillon eine Suchfunktion hinzu", "quillon"),
]

NOT_REQUESTS = [
    "C'est quoi la différence entre Claude Code et Cursor ?",
    "Comment on inverse une liste en Python ?",
    "Remind me tomorrow morning to work on baladeur",
    "What would be a good name for a new coding project?",
]

KINDS = ("capability", "request", "not a request")

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


# A write tool shaped like a code host's: on a real catalogue with such a
# server, a search worded as the task can go to it rather than to Claude
# Code. Invented, like the rest of the catalogue.
WRITE_TOOL = "repo__create_or_update_file"


def _write_tool():
    from jarvis.tools.registry import ToolSpec

    return ToolSpec(
        WRITE_TOOL,
        "Create or update a single file in a remote Git repository.",
        {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "path": {"type": "string", "description": "Path of the file to create or update"},
                "content": {"type": "string", "description": "Content of the file"},
                "message": {"type": "string", "description": "Commit message"},
                "branch": {"type": "string", "description": "Branch to commit to"},
            },
            "required": ["owner", "repo", "path", "content", "message", "branch"],
        },
    )


class _RouterThatAnswers:
    """The pre-loop router's model, answering one fixed thing."""

    def __init__(self, answer: str):
        self.answer = answer

    def direct(self, *args, **kwargs):
        return self.answer


def _router_that_answers_none(**kwargs):
    """As the router answered the question in the field."""
    from jarvis.tools.selection import select_tools

    kwargs["llm_backend"] = _RouterThatAnswers("none")
    return select_tools(**kwargs)


def _router_that_answers_the_write_tool(**kwargs):
    """As a router answers a request on a project when a code host's write
    tool sits in the catalogue: that tool, and not `claudeCode`."""
    from jarvis.tools.selection import select_tools

    kwargs["llm_backend"] = _RouterThatAnswers(WRITE_TOOL)
    return select_tools(**kwargs)


def _cfg(real_model_config, tmp_path):
    root = tmp_path / "Projets"
    for name in PROJECTS:
        (root / name).mkdir(parents=True, exist_ok=True)
    mcp = _invented_mcp_tools(MCP_TOOLS)
    mcp[WRITE_TOOL] = _write_tool()
    return dataclasses.replace(
        real_model_config,
        claude_code_projects_dir=str(root),
        tool_selection_strategy="llm",
        mcps={name.split("__")[0]: {"command": "invented"} for name in mcp},
    ), mcp


def _turn(
    turn_cfg, mcp, sentence: str, *, block: bool, pre_loop=_router_that_answers_none,
) -> Tuple[str, List[Tuple[str, dict]], List[str]]:
    """One voice turn, the pre-loop selection made by ``pre_loop``; returns
    the reply, the jobs that would have started (project, args) and the
    names of the tools called, in order."""
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.claude_code import ClaudeCodeTool, vet_call
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    started: List[Tuple[str, dict]] = []
    called: List[str] = []

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        args = tool_args or {}
        called.append(tool_name)
        if tool_name == "toolSearchTool":
            return BUILTIN_TOOLS["toolSearchTool"].run(args, ToolContext(
                db=db, cfg=cfg, system_prompt="", original_prompt="",
                redacted_text="", max_retries=1, user_print=lambda _m: None,
            ))
        if tool_name == "claudeCode":
            if not str(args.get("tache") or "").strip():
                return ToolExecutionResult(
                    success=True,
                    reply_text="Claude Code has not been given any task since Yuba started.",
                )
            refusal, path = vet_call(args, cfg)
            if refusal:
                return ClaudeCodeTool._refuse(refusal)
            started.append((path.name, args))
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Claude Code has started on the project {path.name}. Its "
                    f"report will be delivered when it finishes. Tell the user "
                    f"it has started; do not claim it is done."
                ),
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    without_block = (
        patch("jarvis.projets.prompt.format_projets_block", return_value="")
        if not block else nullcontext()
    )
    db = Database(":memory:", sqlite_vss_path=None)
    try:
        with without_block, \
             patch("jarvis.reply.engine.select_tools", side_effect=pre_loop), \
             patch("jarvis.tools.registry.get_cached_mcp_tools", return_value=mcp), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
            reply = run_reply_engine(
                db=db, cfg=turn_cfg, tts=None, origin="voix", text=sentence,
                dialogue_memory=DialogueMemory(inactivity_timeout=300),
            )
    finally:
        db.close()
    return reply or "", started, called


_JUDGE_SYSTEM = (
    "You read one reply a voice assistant gave to a user who asked whether "
    "the assistant can work on their code projects. Answer YES if the reply "
    "tells the user the assistant can reach, read or change their code "
    "projects (for example by handing the work to Claude Code, or by asking "
    "which project and what to do). Answer NO if it says or implies it cannot "
    "access them, has no access to their files or machine, or asks them to "
    "paste or share the code instead. Answer with the single word YES or NO."
)


def _says_she_can(cfg, question: str, reply: str) -> bool:
    from jarvis.llm import get_llm_backend

    verdict = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, _JUDGE_SYSTEM,
        f"User: {question}\nAssistant: {reply}\n\nYES or NO:",
        timeout_sec=60.0,
    ) or ""
    return verdict.strip().upper().startswith("YES")


def _passes(cfg, kind: str, case, reply: str, started) -> bool:
    if kind == "capability":
        return not started and _says_she_can(cfg, case, reply)
    if kind == "request":
        return bool(started) and started[-1][0] == case[1]
    return not started


def _sentence(kind: str, case) -> str:
    return case[0] if kind == "request" else case


def _cases():
    for case in CAPABILITY:
        yield "capability", case
    for case in REQUESTS:
        yield "request", case
    for case in NOT_REQUESTS:
        yield "not a request", case


def _to_terminal(capsys, lines: List[str]) -> None:
    from jarvis.utils.console import force_utf8_stream

    with capsys.disabled():
        force_utf8_stream(sys.stdout)
        print("\n".join(lines), flush=True)


@pytest.mark.eval
class TestSheKnowsSheCan:
    """The engine as it ships, on the worst router answer."""

    @pytest.mark.parametrize("question", CAPABILITY)
    def test_asked_whether_she_can_she_says_she_can(self, real_model_config, tmp_path, question):
        cfg, mcp = _cfg(real_model_config, tmp_path)
        reply, started, called = _turn(cfg, mcp, question, block=True)
        print(f"\n  🗣️ {question}\n  🔧 {called}\n  🤖 {reply[:300]!r}")

        assert not started, f"A question about what she can do started a job: {started!r}"
        assert _says_she_can(cfg, question, reply), f"She says she cannot: {reply[:400]!r}"

    @pytest.mark.parametrize("sentence, projet", REQUESTS)
    def test_a_request_on_a_project_reaches_claude_code(
        self, real_model_config, tmp_path, sentence, projet,
    ):
        cfg, mcp = _cfg(real_model_config, tmp_path)
        reply, started, called = _turn(cfg, mcp, sentence, block=True)
        print(f"\n  🗣️ {sentence}\n  🔧 {called}\n  🤖 {reply[:300]!r}")

        assert started, f"No job would have started. Tools: {called}. Reply: {reply[:300]!r}"
        assert started[-1][0] == projet

    @pytest.mark.parametrize("sentence", NOT_REQUESTS)
    def test_the_rest_starts_nothing(self, real_model_config, tmp_path, sentence):
        cfg, mcp = _cfg(real_model_config, tmp_path)
        reply, started, called = _turn(cfg, mcp, sentence, block=True)
        print(f"\n  🗣️ {sentence}\n  🔧 {called}\n  🤖 {reply[:300]!r}")

        assert not started, f"{sentence!r} started a job: {started!r}"


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_PROJECTS_ROUNDS to compare")
class TestBeforeAndAfter:
    """The prompt without the block against the prompt with it, call by call."""

    def test_the_block_against_no_block(self, real_model_config, tmp_path, capsys):
        cfg, mcp = _cfg(real_model_config, tmp_path)
        arms = {"no block": False, "block": True}
        names = list(arms)
        cases = list(_cases())
        hits: Dict[Tuple[str, str], int] = {(a, k): 0 for a in names for k in KINDS}
        calls: Dict[Tuple[str, str], int] = {(a, k): 0 for a in names for k in KINDS}
        searched: Dict[str, int] = {a: 0 for a in names}
        misses: List[str] = []

        def rate(got: int, n: int) -> str:
            return f"{got}/{n} ({100 * got / n:.0f} %)" if n >= _MIN_CALLS_FOR_A_RATE \
                else f"{got}/{n}   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"

        try:
            for rnd in range(ROUNDS):
                for i, (kind, case) in enumerate(cases):
                    turn = (i + rnd) % len(names)
                    for name in names[turn:] + names[:turn]:
                        sentence = _sentence(kind, case)
                        reply, started, called = _turn(cfg, mcp, sentence, block=arms[name])
                        calls[(name, kind)] += 1
                        if "toolSearchTool" in called:
                            searched[name] += 1
                        if _passes(cfg, kind, case, reply, started):
                            hits[(name, kind)] += 1
                        else:
                            misses.append(
                                f"{name} / {kind} / {sentence[:50]!r}: tools {called}, "
                                f"jobs {[p for p, _ in started]}, reply {reply[:120]!r}"
                            )
                _to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUNDS} done"])
        finally:
            lines = [f"📊 Projects question: {len(cases)} sentences x {ROUNDS} rounds, "
                     f"arms interleaved call by call, {len(mcp)} invented MCP tools"]
            for name in names:
                lines.append(f"   🛠️ {name}")
                for kind in KINDS:
                    lines.append(f"      {kind:<14} {rate(hits[(name, kind)], calls[(name, kind)])}")
                lines.append(f"      toolSearchTool called in {searched[name]} turns")
            for miss in misses:
                lines.append(f"      ❌ {miss}")
            _to_terminal(capsys, [""] + lines)

        assert hits[("block", "capability")] > hits[("no block", "capability")], (
            "The block made her say she can no more often than without it."
        )


@pytest.mark.eval
@pytest.mark.skipif(ROUNDS < 1, reason="set EVAL_PROJECTS_ROUNDS to run it")
class TestAWriteToolOffered:
    """The pre-loop router offers a code host's write tool for a request on
    a project, and not `claudeCode`: the tool a task-worded search went to
    on a real catalogue. The engine as it ships, block on. A request passes
    when a job starts on its project and the write tool was not called."""

    def test_a_request_goes_to_claude_code_not_the_write_tool(
        self, real_model_config, tmp_path, capsys,
    ):
        cfg, mcp = _cfg(real_model_config, tmp_path)
        hits = calls = wrote = searched = 0
        misses: List[str] = []

        def rate(got: int, n: int) -> str:
            return f"{got}/{n} ({100 * got / n:.0f} %)" if n >= _MIN_CALLS_FOR_A_RATE \
                else f"{got}/{n}   (under {_MIN_CALLS_FOR_A_RATE} calls: counts only)"

        try:
            for rnd in range(ROUNDS):
                for sentence, projet in REQUESTS:
                    reply, started, called = _turn(
                        cfg, mcp, sentence, block=True,
                        pre_loop=_router_that_answers_the_write_tool,
                    )
                    calls += 1
                    if WRITE_TOOL in called:
                        wrote += 1
                    if "toolSearchTool" in called:
                        searched += 1
                    if started and started[-1][0] == projet and WRITE_TOOL not in called:
                        hits += 1
                    else:
                        misses.append(
                            f"{sentence[:50]!r}: tools {called}, "
                            f"jobs {[p for p, _ in started]}, reply {reply[:120]!r}"
                        )
                _to_terminal(capsys, [f"⏱️ round {rnd + 1}/{ROUNDS} done"])
        finally:
            lines = [
                f"📊 A write tool offered: {len(REQUESTS)} requests x {ROUNDS} rounds, "
                f"the pre-loop router answering {WRITE_TOOL}, {len(mcp)} MCP tools",
                f"   🛠️ job on the right project, write tool not called  {rate(hits, calls)}",
                f"   ✍️ write tool called in {wrote} turns",
                f"   🔎 toolSearchTool called in {searched} turns",
            ]
            for miss in misses:
                lines.append(f"      ❌ {miss}")
            _to_terminal(capsys, [""] + lines)

        assert wrote == 0, f"She called {WRITE_TOOL} in {wrote} of {calls} turns."
        assert hits == calls, f"{calls - hits} of {calls} requests did not reach Claude Code."
