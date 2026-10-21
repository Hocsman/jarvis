"""The `stopClaudeCode` tool: stopping a running job without stopping Yuba.

The stop itself (the process tree killed, the slot freed, the bracket
closed) is tested in test_claude_code_delegation.py with a real child
process. Here: what it costs at the gate, when it is shown at all, and
what it says back.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import pytest

from claude_code_fakes import (
    install_fake_claude, make_cfg, make_projects, wait_for_calls, wait_for_file,
)
from jarvis.projets import delegation as dl
from jarvis.tools.builtin.claude_code import ClaudeCodeTool, edits_kept, status_text
from jarvis.tools.builtin.stop_claude_code import StopClaudeCodeTool, unconfirmed_stop_text
from jarvis.tools.policy import RISK_ACTION, RISK_READ


@pytest.fixture
def projects(tmp_path):
    return make_projects(tmp_path, ("chantipay", "nikahscore"))


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    return install_fake_claude(tmp_path, monkeypatch, dl)


@pytest.fixture
def jobs(fake_claude, monkeypatch):
    """A fresh job keeper standing in for the process-wide one."""
    keeper = dl.Delegations()
    delivered = []
    keeper.open(delivered.append)
    keeper.delivered = delivered
    monkeypatch.setattr(dl, "DELEGATIONS", keeper)
    yield keeper
    keeper.close()


def _context(cfg, origin="voix"):
    from jarvis.tools.base import ToolContext

    return ToolContext(
        db=None, cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1, user_print=lambda _m: None,
        origin=origin,
    )


def _stop(cfg, **args):
    return StopClaudeCodeTool().run(args, _context(cfg))


def _delegate(cfg, **args):
    return ClaudeCodeTool().run(args, _context(cfg))


# ── What it costs, and who sees it ────────────────────────────────────


@pytest.mark.parametrize("args", [None, {}, {"query": ""}, {"x": 1}, {"tache": "  "}])
def test_stopping_a_job_is_an_action(args):
    """It changes something outside Yuba's own store (a process editing
    the user's project) and it can be undone by asking again."""
    assert StopClaudeCodeTool().risk_for(args) == RISK_ACTION


@pytest.mark.parametrize("args", [
    {"task": "stop the login page from crashing in chantipay"},
    {"projet": "chantipay", "tache": "arrête d'utiliser jQuery"},
], ids=["an instruction", "a project and a task"])
def test_a_call_carrying_an_instruction_only_reads(args):
    """It has no properties. Text in it is an instruction the model is
    passing on ("ask Claude Code to stop the page from crashing"), and such
    a call stops nothing, so it costs no question."""
    assert StopClaudeCodeTool().risk_for(args) == RISK_READ


def test_it_is_registered():
    from jarvis.tools.registry import BUILTIN_TOOLS

    assert isinstance(BUILTIN_TOOLS.get("stopClaudeCode"), StopClaudeCodeTool)


def test_no_routine_can_reach_it():
    from unittest.mock import MagicMock

    from jarvis.routines.eligibility import refuse_reason

    assert refuse_reason(MagicMock(), "stopClaudeCode")


def test_it_is_shown_only_while_a_job_runs(
        tmp_path, projects, jobs, monkeypatch):
    """The rest of the time there is nothing to stop, and a tool shown to
    the router is one more place for a bare "stop" to land."""
    from jarvis.tools.registry import offered_builtin_tools

    cfg = make_cfg(tmp_path, projects)
    assert "stopClaudeCode" not in offered_builtin_tools(cfg)

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert "stopClaudeCode" in offered_builtin_tools(cfg)
    assert "stopClaudeCode" not in offered_builtin_tools(
        make_cfg(tmp_path, projects, claude_code_projects_dir=""))

    jobs.cancel()
    assert "stopClaudeCode" not in offered_builtin_tools(cfg)


def test_its_job_fits_inside_the_routers_slice():
    """The router reads only the leading sentences that fit its budget,
    and "stop" alone must find nothing there that claims it."""
    from jarvis.tools.selection import _ROUTER_SUMMARY_CHARS, _router_summary

    summary = _router_summary(StopClaudeCodeTool().description)

    assert len(summary) <= _ROUTER_SUMMARY_CHARS
    assert "Claude Code" in summary


# ── What it says ──────────────────────────────────────────────────────


def test_a_stop_kills_the_job_and_says_which_one(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.", modifier=True)
    wait_for_calls(fake_claude)

    result = _stop(cfg)

    assert result.success is True
    assert "chantipay" in result.reply_text
    assert jobs.running() is None
    assert jobs.last().cancelled is True


def test_stopping_a_change_says_its_edits_stay(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    """Stopping is not undoing. The user must not believe the project is
    back as it was."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.", modifier=True)
    wait_for_calls(fake_claude)
    change = _stop(cfg).reply_text

    stopped_change = jobs.last().job

    _delegate(cfg, projet="chantipay", tache="Explique l'auth.", modifier=False)
    wait_for_calls(fake_claude, 2)
    question = _stop(cfg).reply_text
    stopped_question = jobs.last().job

    assert edits_kept(stopped_change) in change
    assert edits_kept(stopped_question) in question
    assert edits_kept(stopped_change) not in question


def test_a_call_carrying_an_instruction_stops_nothing_and_points_at_claude_code(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    """Measured on gemma4:e2b with a job running: "ask Claude Code to stop
    the login page from crashing" is filed here, and a third of those
    calls carry the instruction. A real stop never did."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)

    instruction = "stop the login page from crashing in chantipay"

    result = _stop(cfg, task=instruction)

    assert result.success is True, "nothing failed: the call was not a stop"
    assert ClaudeCodeTool().name in result.reply_text
    assert instruction not in result.reply_text, "the model's own text is not relayed"
    assert jobs.running() is not None
    assert jobs.last() is None


def test_with_nothing_running_it_stops_nothing_and_says_how_the_last_went(
        tmp_path, projects, fake_claude, jobs):
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert jobs.wait(20.0)

    result = _stop(cfg)

    assert result.success is True
    assert status_text() in result.reply_text
    assert jobs.last().report in result.reply_text


def test_a_job_that_had_already_finished_reports_its_work(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_DELAY", "1")
    monkeypatch.setattr(dl, "_kill_tree", lambda proc: None)
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)

    result = _stop(cfg)

    assert jobs.last().cancelled is False
    assert jobs.last().report in result.reply_text


def test_a_job_that_had_already_failed_says_why(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    """It ended on its own before the stop reached it, so its reason is
    what the user needs, not a stop that never happened."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)
    job = jobs.running()
    failed = dl.Outcome(job=job, ok=False, report="Le build a échoué.",
                        session_id=None, duration_s=12.0)
    real_cancel = jobs.cancel
    monkeypatch.setattr(jobs, "cancel", lambda timeout=None: (job, failed))

    result = _stop(cfg)

    assert result.success is True
    assert failed.report in result.reply_text
    assert edits_kept(job) not in result.reply_text
    real_cancel()


def test_a_stop_it_could_not_confirm_is_not_claimed(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    real_cancel = jobs.cancel
    monkeypatch.setattr(jobs, "cancel", lambda timeout=None: (jobs.running(), None))
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)

    result = _stop(cfg)

    assert result.success is True, "nothing failed: the stop is on its way"
    assert unconfirmed_stop_text(jobs.running()) in result.reply_text
    assert edits_kept(jobs.running()) not in result.reply_text
    real_cancel()


def test_a_stop_that_cut_short_a_job_already_reported_says_both(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    """Its work was done and reported, and the stop ended what was still
    exiting (hooks, a slow exit): the report stands, and for a change the
    user must not believe nothing was touched."""
    from pathlib import Path

    monkeypatch.setenv("FAKE_CLAUDE_MODE", "linger")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.", modifier=True)
    wait_for_file(Path(str(fake_claude.log) + ".printed"))

    result = _stop(cfg)

    last = jobs.last()
    assert last.interrupted is True and last.cancelled is False
    assert last.report in result.reply_text
    assert edits_kept(last.job) in result.reply_text


def test_asking_how_it_went_after_a_stop_says_it_was_stopped(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    """Nothing Claude Code said is quoted: it said nothing."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)
    _stop(cfg)

    result = _delegate(cfg)

    assert "chantipay" in result.reply_text
    assert edits_kept(jobs.last().job) in result.reply_text
    assert "«" not in result.reply_text


def test_a_new_task_runs_once_the_wrong_one_is_stopped(
        tmp_path, projects, fake_claude, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _delegate(cfg, projet="chantipay", tache="Ajoute un bouton.")
    wait_for_calls(fake_claude)
    _stop(cfg)
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "ok")

    result = _delegate(cfg, projet="nikahscore", tache="Ajoute un bouton.")

    assert result.success is True
    assert jobs.wait(20.0)
    assert [o.job.projet for o in jobs.delivered] == ["nikahscore"]


def test_its_replies_are_never_digested():
    """Written by code and short, carrying a project name and, when the
    job had finished, Claude Code's report quoted as its words."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from jarvis.reply.engine import _maybe_digest_tool_result

    cfg = SimpleNamespace(tool_result_digest_enabled=True,
                          ollama_chat_model="gemma4:e2b", llm_chat_model="gemma4:e2b")
    raw = "Nothing is running. The last task, on chantipay, finished: « " + (
        "Phrase du rapport. " * 40) + "»"
    with patch("jarvis.reply.enrichment.call_llm_direct") as llm:
        out = _maybe_digest_tool_result(cfg=cfg, query="q", tool_name="stopClaudeCode",
                                        raw_tool_result=raw)
        llm.assert_not_called()
    assert out == raw
