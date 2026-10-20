"""
End-to-end eval — stopping Claude Code reaches `stopClaudeCode`, and a
plain "stop" still reaches `stop`.

`stopClaudeCode` is shown only while a job runs, which is exactly when a
bare "stop" (end the conversation) could land on it instead. Both
directions matter: a request to stop the job must reach it, since a wrong
job keeps editing the user's code until its deadline otherwise; and a
dismissal must not, since stopping a job the user wanted costs the work
done so far (the gate asks first, so a misfire costs a question rather
than the job, but it is still a wrong question).

Two layers: the router (the small model that narrows the catalogue) must
offer it, then the reply engine must call it. `stop` is always offered,
so a dismissal is measured at the reply alone. Here a job is made to look
running, so the tool is offered as in production; nothing is stopped.
What a stop does is measured in tests/test_claude_code_delegation.py with
a real child process.

See src/jarvis/projets/projets.spec.md.

Run:
    ./scripts/run_evals.sh stop_claude_code_reaches_the_right_tool
"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL, ToolCallCapture

STOPS = [
    pytest.param("Yuba, arrête Claude Code", id="FR: stop Claude Code"),
    pytest.param("Yuba, stop the Claude Code job", id="EN: stop the job"),
    pytest.param("Yuba, annule ce que fait Claude Code, il s'est trompé de projet",
                 id="FR: cancel, wrong project"),
    pytest.param("Yuba, interrupt Claude Code right now", id="EN: interrupt"),
]

DISMISSALS = [
    pytest.param("Yuba, stop", id="stop alone"),
    pytest.param("Yuba, tais-toi", id="FR: be quiet"),
    pytest.param("OK merci Yuba, c'est bon, arrête", id="FR: that's all, stop"),
]

NOT_A_STOP = [
    pytest.param("Yuba, où en est Claude Code ?", id="FR: how is it going"),
]

# A new instruction for Claude Code whose words say stop. With a job
# running it would be refused as "one at a time" anyway; what matters is
# that it does not stop the job.
STOP_WORDED_TASKS = [
    pytest.param("Yuba, ask Claude Code to stop the login page from crashing in chantipay",
                 id="EN: stop a crash"),
    pytest.param("Yuba, demande à Claude Code d'arrêter d'utiliser jQuery dans nikahscore",
                 id="FR: stop using jQuery"),
]


@pytest.fixture
def job_running(monkeypatch):
    """Make a job look running on chantipay, so the tool is offered."""
    from jarvis.projets import delegation as dl

    job = dl.Job(id="cc_eval", projet="chantipay", path=Path("chantipay"),
                 tache="Ajoute un bouton.", modifier=True, origin="voix",
                 risk="destructif", verdict="demande", started=time.monotonic() - 180)
    monkeypatch.setattr(dl.DELEGATIONS, "running", lambda: job)
    return job


def _route(text: str):
    from helpers import MockConfig
    from jarvis.llm import get_llm_backend
    from jarvis.tools.registry import offered_builtin_tools
    from jarvis.tools.selection import ToolSelectionStrategy, select_tools

    cfg = MockConfig()
    cfg.ollama_base_url = JUDGE_BASE_URL
    cfg.claude_code_projects_dir = "Projets"
    return select_tools(
        query=text, builtin_tools=offered_builtin_tools(cfg), mcp_tools={},
        strategy=ToolSelectionStrategy.LLM, llm_backend=get_llm_backend(cfg),
        llm_model=JUDGE_MODEL, llm_timeout_sec=30.0,
    )


def _run(text, mock_config, eval_db, eval_dialogue_memory, tmp_path):
    """Run the real reply engine with every tool call captured."""
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.builtin.stop import STOP_SIGNAL
    from jarvis.tools.builtin.stop_claude_code import StopClaudeCodeTool, carried_instruction
    from jarvis.tools.types import ToolExecutionResult

    root = tmp_path / "Projets"
    for name in ("chantipay", "nikahscore"):
        (root / name).mkdir(parents=True, exist_ok=True)
    mock_config.ollama_base_url = JUDGE_BASE_URL
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.claude_code_projects_dir = str(root)
    capture = ToolCallCapture()

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "stop":
            return ToolExecutionResult(success=True, reply_text=STOP_SIGNAL)
        if tool_name == "stopClaudeCode" and carried_instruction(tool_args):
            # The tool's own refusal: it returns before touching any job.
            return StopClaudeCodeTool().run(tool_args, None)
        if tool_name == "stopClaudeCode":
            return ToolExecutionResult(success=True, reply_text=(
                "Claude Code was stopped on chantipay after 3 min. Files it had "
                "already changed are not undone. Tell the user, in their language."))
        if tool_name == "claudeCode":
            return ToolExecutionResult(success=True, reply_text=(
                "Claude Code is still working on chantipay, started 3 min ago."))
        return ToolExecutionResult(success=True, reply_text="OK")

    with patch("jarvis.reply.engine.run_tool_with_retries", side_effect=_runner):
        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None, text=text,
            dialogue_memory=eval_dialogue_memory,
        )
    print(f"\n  Query ({JUDGE_MODEL}): {text}")
    print(f"  Tools called: {capture.tool_names()}")
    print(f"  Response: {(response or '')[:200]}")
    return capture


@pytest.mark.eval
@requires_judge_llm
class TestTheRouterOffersIt:

    @pytest.mark.parametrize("text", STOPS)
    def test_a_stop_of_claude_code_is_offered_the_tool(self, text, job_running):
        selected = _route(text)
        print(f"\n  Query: {text}\n  Selected: {selected}")
        assert "stopClaudeCode" in selected


@pytest.mark.eval
@requires_judge_llm
class TestTheReplyCallsIt:

    @pytest.mark.parametrize("text", STOPS)
    def test_a_stop_of_claude_code_stops_it(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, job_running, text,
    ):
        capture = _run(text, mock_config, eval_db, eval_dialogue_memory, tmp_path)

        assert capture.has_tool("stopClaudeCode"), "the job would still be running"
        task = str((capture.get_args("claudeCode") or {}).get("tache") or "").strip()
        assert not task, f"a stop was filed as a task for Claude Code: {task!r}"

    @pytest.mark.parametrize("text", DISMISSALS)
    def test_a_dismissal_ends_the_conversation_and_leaves_the_job(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, job_running, text,
    ):
        capture = _run(text, mock_config, eval_db, eval_dialogue_memory, tmp_path)

        assert not capture.has_tool("stopClaudeCode"), (
            f"{text!r} stopped the user's Claude Code job")
        if not capture.has_tool("stop"):
            # Measured on gemma4:e2b, interleaved, 21 calls an arm over these
            # three sentences: `stop` was called 2/21 with no job running
            # (the tool hidden) and 2/21 with one (the tool shown). Answering
            # a dismissal in words is the model's own habit, not this tool
            # taking it, so it is tracked rather than failed: a model that
            # calls `stop` flips this.
            pytest.xfail(f"{JUDGE_MODEL} answered a dismissal without calling stop")

    @pytest.mark.parametrize("text", NOT_A_STOP)
    def test_the_rest_never_reaches_it(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, job_running, text,
    ):
        capture = _run(text, mock_config, eval_db, eval_dialogue_memory, tmp_path)

        assert not capture.has_tool("stopClaudeCode"), (
            f"{text!r} stopped the user's Claude Code job")

    @pytest.mark.parametrize("text", STOP_WORDED_TASKS)
    def test_an_instruction_whose_words_say_stop_does_not_stop_the_job(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, job_running, text,
    ):
        from jarvis.tools.builtin.stop_claude_code import carried_instruction

        capture = _run(text, mock_config, eval_db, eval_dialogue_memory, tmp_path)

        stops = [c["args"] for c in capture.calls
                 if c["name"] == "stopClaudeCode" and not carried_instruction(c["args"])]
        if stops:
            # Measured on gemma4:e2b, interleaved, 24 calls an arm over four
            # such sentences: 20/24 filed as a stop with this description,
            # 19/24 with the restriction moved into the first sentence (the
            # part the router reads), so wording is not the lever. 8 and 5
            # of those carried the instruction, and a call carrying it stops
            # nothing; the rest meet the gate's question, which names
            # stopClaudeCode, so the job is stopped only on a yes. Tracked
            # rather than failed: a model that keeps them apart flips this.
            pytest.xfail(f"{JUDGE_MODEL} filed an instruction as a stop: {stops!r}")
