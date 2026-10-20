"""
End-to-end eval — work on a project goes to Claude Code, and only that.

`claudeCode` is the one tool that sends the user's code to a cloud and,
for a change, edits it. Both directions matter. A request for work on a
project must reach it, with the project named and `modifier` telling a
change from a question (a change filed as a question runs read-only, a
question filed as a change costs a click: both safe, both a wasted turn).
And the sentences that merely mention a project or the product must not
reach it: a reminder about a project is a reminder, a question about what
Claude Code is answers from knowledge or the web.

Two layers: the router (the small model that narrows the catalogue) must
offer the tool, then the reply engine must call it so that the tool's own
decision (`vet_call`) starts it on the right project. A change named only
inside the task, or on a near miss, is sent back with the exact name, and
the model may call again within the loop, as it would in production.
Nothing is started: what runs is measured in
tests/test_claude_code_delegation.py, with a real child process.
`remember` is not stubbed: it writes into a core under `tmp_path`, so a
task the turn filed as a memory of the user shows up in the files.

See src/jarvis/projets/projets.spec.md.

Run:
    ./scripts/run_evals.sh claude_code_reaches_the_right_tool
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL, ToolCallCapture, assert_not_fallback_reply


@dataclass
class Case:
    text: str
    projet: Optional[str] = None
    # True: a change. False: a question. None: asking how it is going.
    modifier: Optional[bool] = None


CHANGES = [
    pytest.param(Case("Jarvis, demande à Claude Code d'ajouter un bouton de déconnexion dans chantipay",
                      "chantipay", True), id="FR: add a button"),
    pytest.param(Case("Jarvis, fais corriger par Claude Code le bug d'affichage des dates dans le projet nikahscore",
                      "nikahscore", True), id="FR: fix a bug"),
    pytest.param(Case("Jarvis, ask Claude Code to fix the failing tests in relayer",
                      "relayer", True), id="EN: fix the tests"),
]

QUESTIONS = [
    pytest.param(Case("Jarvis, demande à Claude Code de m'expliquer comment marche l'authentification dans chantipay",
                      "chantipay", False), id="FR: explain the auth"),
    pytest.param(Case("Jarvis, have Claude Code review the database layer of mentaltracker and tell me what is wrong with it",
                      "mentaltracker", False), id="EN: review"),
]

PROGRESS = [
    pytest.param(Case("Jarvis, où en est Claude Code ?"), id="FR: how is it going"),
]

NOT_DELEGATIONS = [
    pytest.param(Case("Jarvis, rappelle-moi demain matin de bosser sur chantipay"),
                 id="A reminder about a project is a reminder"),
    pytest.param(Case("Jarvis, c'est quoi la différence entre Claude Code et Cursor ?"),
                 id="A question about the product is not a task"),
    pytest.param(Case("Jarvis, quel temps fait-il à Lyon ?"),
                 id="Nothing to do with code"),
]


def _route(text: str):
    from helpers import MockConfig
    from jarvis.llm import get_llm_backend
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.selection import ToolSelectionStrategy, select_tools

    cfg = MockConfig()
    cfg.ollama_base_url = JUDGE_BASE_URL
    return select_tools(
        query=text, builtin_tools=BUILTIN_TOOLS, mcp_tools={},
        strategy=ToolSelectionStrategy.LLM, llm_backend=get_llm_backend(cfg),
        llm_model=JUDGE_MODEL, llm_timeout_sec=30.0,
    )


PROJECTS = ("chantipay", "nikahscore", "relayer", "mentaltracker", "lynx")


def _projects_root(tmp_path):
    root = tmp_path / "Projets"
    for name in PROJECTS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _run(case, mock_config, eval_db, eval_dialogue_memory, tmp_path):
    """Run the real reply engine. A claudeCode call gets the tool's own
    decision (`vet_call`): a refusal goes back to the model, which may
    call again, as it would in production; nothing is started."""
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.builtin.claude_code import ClaudeCodeTool, vet_call
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_base_url = JUDGE_BASE_URL
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.claude_code_projects_dir = str(_projects_root(tmp_path))
    # The core lives beside the database: a fresh one per case, so what
    # it holds afterwards is what this turn wrote.
    mock_config.db_path = str(tmp_path / "jarvis.db")
    capture = ToolCallCapture()
    started = []

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        args = tool_args or {}
        capture.record(tool_name, args)
        if tool_name == "claudeCode" and str(args.get("tache") or "").strip():
            refusal, path = vet_call(args, cfg)
            if refusal:
                return ClaudeCodeTool._refuse(refusal)
            started.append((path.name, args))
            return ToolExecutionResult(
                success=True,
                reply_text=(
                    f"Claude Code has started on the project {path.name}. Its "
                    f"report will be delivered when it finishes. Tell the user "
                    f"it has started."
                ),
            )
        if tool_name == "remember":
            # The real tool, so a write shows up in the files rather
            # than as a call a stub answered "OK".
            return BUILTIN_TOOLS["remember"].execute(
                db=db, cfg=cfg, tool_args=tool_args, system_prompt="",
                original_prompt="", redacted_text=kwargs.get("redacted_text", ""),
                max_retries=1, user_print=lambda _m: None,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    with patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None, text=case.text,
            dialogue_memory=eval_dialogue_memory,
        )
    return capture, response, started


def _report(case, capture, response):
    print(f"\n  Query ({JUDGE_MODEL}): {case.text}")
    print(f"  Tools called: {capture.tool_names()}")
    for c in capture.calls:
        print(f"    - {c['name']}({c['args']})")
    print(f"  Response: {(response or '')[:200]}")


def _written(mock_config):
    """Everything the core believes, both sections."""
    from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore

    core = MemoryCore.for_config(mock_config)
    return [e.text for e in core.active(SECTION_PROFILE)] + [
        e.text for e in core.active(SECTION_RULES)
    ]


@pytest.mark.eval
@requires_judge_llm
class TestTheRouterOffersIt:

    @pytest.mark.parametrize("case", CHANGES + QUESTIONS + PROGRESS)
    def test_work_on_a_project_is_offered_claude_code(self, case):
        selected = _route(case.text)
        print(f"\n  Query: {case.text}\n  Selected: {selected}")
        assert "claudeCode" in selected

    @pytest.mark.parametrize("case", NOT_DELEGATIONS)
    def test_the_rest_is_not(self, case):
        selected = _route(case.text)
        print(f"\n  Query: {case.text}\n  Selected: {selected}")
        assert "claudeCode" not in selected


@pytest.mark.eval
@requires_judge_llm
class TestTheReplyCallsIt:

    @pytest.mark.parametrize("case", CHANGES + QUESTIONS)
    def test_a_task_starts_on_its_project(
        self, mock_config, eval_db, eval_dialogue_memory, case, tmp_path,
    ):
        capture, response, started = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path)
        _report(case, capture, response)

        assert_not_fallback_reply(response, context="claudeCode")
        assert capture.has_tool("claudeCode"), "the task never reached Claude Code"
        assert started, "every call was sent back: nothing would have started"
        projet, args = started[-1]
        assert projet == case.projet
        assert str(args.get("tache") or "").strip(), "the task arrived empty"

    @pytest.mark.parametrize("case", CHANGES + QUESTIONS)
    def test_a_task_writes_nothing_into_the_core(
        self, mock_config, eval_db, eval_dialogue_memory, case, tmp_path,
    ):
        """A task handed over is neither a fact about the user nor a rule
        for her. Filed in the core as "the user wants a bug fixed", it is
        injected into every later prompt until he finds it in a file.

        The write comes from the planner, which restates the delegation as
        a `remember that the user wants ...` step that direct execution
        runs. Nothing in `remember`'s description past its first 120
        characters reaches the planner, so the fix is not there: it is
        `claudeCode` opening with what the user wants done, and `remember`
        read last among the mandatory tools in an order that does not
        change with the process (projets.spec.md, selection.spec.md).

        Measured on gemma4:e2b, end to end, the three changes, arms
        interleaved call by call, 21 calls an arm against a description
        opening "Hand a task on one of the user's code projects to Claude
        Code" with the order drawn at random per call, as a set's hash
        seed draws it per process: 2 writes with that opening and order
        against 17 (Fisher p = 5.8e-6), and the four requests to remember
        of test_remember_tool_is_called.py reached the tool 14 times in 28
        on both sides (p = 1.0). The two left were on the English
        sentence, both planned before the delegation: "remember that
        Jarvis needs to use the Claude Code tool", "remember that Jarvis
        is an assistant". In a separate run, adding "done" to
        `remember`'s list of things wanted right now wrote 15 times in 21
        against 14 (p = 1.0): that clause is past what the planner reads.

        So this can still fail on the small canary, about once in ten
        delegations and mostly on the English one (2 in 7 there), and is
        left failing rather than marked expected: what it catches is a
        false standing entry about the user."""
        capture, response, _started = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path)
        written = _written(mock_config)
        _report(case, capture, response)
        print(f"  Core now believes: {written}")

        assert written == [], (
            f"A task for Claude Code was stored as a durable entry about the "
            f"user: {written!r}"
        )

    @pytest.mark.parametrize("case", CHANGES + QUESTIONS)
    def test_a_change_and_a_question_are_told_apart(
        self, mock_config, eval_db, eval_dialogue_memory, case, tmp_path,
    ):
        from jarvis.tools.builtin.claude_code import wants_changes

        capture, response, _started = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path)
        _report(case, capture, response)

        args = capture.get_args("claudeCode")
        assert args is not None, "the task never reached Claude Code"
        if case.modifier is False and wants_changes(args):
            # Measured on gemma4:e2b, 20 calls per schema: no wording of
            # the flag (a boolean `modifier`, an enum, a predicate name)
            # got "explain" filed as a question. The direction is the
            # safe one, a card where none was needed, so it is tracked
            # rather than failed: a model that gets it right flips this.
            pytest.xfail(
                f"{JUDGE_MODEL} filed a question as a change: {args!r}")
        assert wants_changes(args) is case.modifier

    @pytest.mark.parametrize("case", NOT_DELEGATIONS)
    def test_the_rest_never_reaches_it(
        self, mock_config, eval_db, eval_dialogue_memory, case, tmp_path,
    ):
        capture, response, _started = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path)
        _report(case, capture, response)

        assert not capture.has_tool("claudeCode"), (
            f"{case.text!r} handed a project to Claude Code: "
            f"{capture.get_args('claudeCode')!r}"
        )
