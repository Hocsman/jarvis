"""The `claudeCode` tool: what it costs to call, and what it says back.

The work itself is tested in test_claude_code_delegation.py. Here: the
risk each call carries at the gate, the honest refusals (off, not
installed, unknown or ambiguous project, Yuba's own store inside the
project, a job already running), and the reply that says a job has
started rather than claiming it is done.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from jarvis.projets import claude_code as cc
from jarvis.tools.builtin.claude_code import ClaudeCodeTool
from jarvis.tools.policy import RISK_DESTRUCTIVE, RISK_READ


_FAKE_CLAUDE = r'''
import json, sys, time, os
sys.stdin.read()
if os.environ.get("FAKE_CLAUDE_MODE") == "sleep":
    time.sleep(60)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": "Fait.", "session_id": "aaaaaaaa-0000-4000-8000-000000000002"}))
'''


@pytest.fixture
def projects(tmp_path):
    root = tmp_path / "Projets"
    for name in ("chantipay", "nikahscore", "snake", "snack"):
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """A fresh job keeper standing in for the process-wide one, running
    a stand-in for Claude Code."""
    script = tmp_path / "fake_claude.py"
    script.write_text(_FAKE_CLAUDE, encoding="utf-8")
    monkeypatch.setattr(cc, "resolve_claude_command",
                        lambda command: [sys.executable, str(script)])
    keeper = cc.Delegations()
    delivered = []
    keeper.set_delivery(delivered.append)
    keeper.delivered = delivered
    monkeypatch.setattr(cc, "DELEGATIONS", keeper)
    yield keeper
    keeper.shutdown()


def _cfg(tmp_path, projects, **over):
    values = dict(
        claude_code_projects_dir=str(projects),
        claude_code_command="claude",
        claude_code_timeout_sec=30.0,
        db_path=str(tmp_path / "data" / "jarvis.db"),
    )
    values.update(over)
    return SimpleNamespace(**values)


def _run(cfg, origin="voix", **args):
    from jarvis.tools.base import ToolContext

    context = ToolContext(
        db=None, cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=1, user_print=lambda _m: None,
        origin=origin,
    )
    return ClaudeCodeTool().run(args, context)


# ── What it costs at the gate ─────────────────────────────────────────


@pytest.mark.parametrize("args", [
    {"projet": "chantipay", "tache": "Ajoute un bouton.", "modifier": True},
    {"projet": "chantipay", "tache": "Ajoute un bouton."},
    {"projet": "chantipay", "tache": "Ajoute un bouton.", "modifier": "true"},
    {"projet": "chantipay", "tache": "Ajoute un bouton.", "modifier": "peut-être"},
    {"tache": "Ajoute un bouton dans chantipay.", "modifier": True},
], ids=["a change", "modifier absent", "a string true", "anything unreadable",
        "the project only inside the task"])
def test_a_task_that_may_change_files_is_destructive(args):
    assert ClaudeCodeTool().risk_for(args) == RISK_DESTRUCTIVE


@pytest.mark.parametrize("args", [
    {"projet": "chantipay", "tache": "Explique l'authentification.", "modifier": False},
    {"projet": "chantipay", "tache": "Explique l'authentification.", "modifier": "false"},
    {"projet": "chantipay"},
    {"tache": ""},
    {},
], ids=["a question", "a string false", "no task", "an empty task", "nothing"])
def test_a_question_or_a_progress_check_only_reads(args):
    assert ClaudeCodeTool().risk_for(args) == RISK_READ


def test_it_is_registered():
    from jarvis.tools.registry import BUILTIN_TOOLS

    assert isinstance(BUILTIN_TOOLS.get("claudeCode"), ClaudeCodeTool)


def test_no_routine_can_reach_it():
    """The report is spoken to someone who has to be in the room."""
    from jarvis.routines.scope import JAMAIS_EN_ROUTINE

    assert "claudeCode" in JAMAIS_EN_ROUTINE


# ── Honest refusals ───────────────────────────────────────────────────


def test_off_until_a_folder_is_named(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects, claude_code_projects_dir=""),
                  projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert "claude_code_projects_dir" in result.reply_text
    assert jobs.running() is None


def test_not_installed_says_so_and_runs_nothing(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setattr(cc, "resolve_claude_command", lambda command: None)

    result = _run(_cfg(tmp_path, projects), projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert "Claude Code" in result.reply_text
    assert jobs.running() is None


def test_an_unknown_project_lists_those_that_exist(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects), projet="zzzzzz", tache="Ajoute un bouton.")

    assert result.success is False
    for name in ("chantipay", "nikahscore"):
        assert name in result.reply_text
    assert jobs.running() is None


def test_an_ambiguous_project_names_the_candidates(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects), projet="snak", tache="Ajoute un bouton.")

    assert result.success is False
    assert "snake" in result.reply_text and "snack" in result.reply_text
    assert jobs.running() is None


def test_a_task_without_a_project_lists_the_projects_and_runs_nothing(
        tmp_path, projects, jobs):
    """So she can ask which one, and the next answer finds the tool
    still offered (a failed call carries over to the next turn)."""
    result = _run(_cfg(tmp_path, projects), tache="Ajoute un bouton.")

    assert result.success is False
    for name in ("chantipay", "nikahscore"):
        assert name in result.reply_text
    assert jobs.running() is None


def test_a_task_without_a_project_also_says_how_the_last_one_went(
        tmp_path, projects, jobs):
    """A small model files "how is Claude Code doing?" as a task with no
    project. The answer it needs is the state of the job."""
    cfg = _cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert jobs.wait(20.0)

    result = _run(cfg, tache="où en est Claude Code ?", modifier=False)

    assert "Fait." in result.reply_text
    assert jobs.wait(5.0) and len(jobs.delivered) == 1


def test_a_project_named_inside_the_task_is_the_one_used(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects), tache="Ajoute un bouton dans chantipay.")

    assert result.success is True
    assert jobs.wait(20.0)
    (outcome,) = jobs.delivered
    assert outcome.job.projet == "chantipay"


def test_a_projet_that_names_nothing_falls_back_to_the_task(tmp_path, projects, jobs):
    """Observed: `projet` holding "date display bug" and the project in
    another field. What the card shows is what is searched."""
    result = _run(_cfg(tmp_path, projects), projet="date display bug",
                  tache="fix the date display bug in nikahscore")

    assert result.success is True
    assert jobs.wait(20.0)
    assert jobs.delivered[0].job.projet == "nikahscore"


def test_a_project_holding_yubas_own_store_is_refused(tmp_path, projects, jobs):
    """The core lists her permissions. An agent editing the folder that
    holds it would make them advisory."""
    cfg = _cfg(tmp_path, projects,
               db_path=str(projects / "chantipay" / "data" / "jarvis.db"))

    result = _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert jobs.running() is None


# ── Launching, and asking how it is going ─────────────────────────────


def test_a_launch_says_it_has_started_not_that_it_is_done(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects), projet="chantipay",
                  tache="Ajoute un bouton.", modifier=True)

    assert result.success is True
    assert "chantipay" in result.reply_text
    assert jobs.wait(20.0)
    (outcome,) = jobs.delivered
    assert outcome.job.origin == "voix"
    assert outcome.job.modifier is True


def test_a_question_runs_read_only(tmp_path, projects, jobs):
    _run(_cfg(tmp_path, projects), projet="chantipay",
         tache="Explique l'authentification.", modifier=False)

    assert jobs.wait(20.0)
    (outcome,) = jobs.delivered
    assert outcome.job.modifier is False


def test_a_second_task_names_the_one_running(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = _cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    result = _run(cfg, projet="nikahscore", tache="Corrige le build.")

    assert result.success is False
    assert "chantipay" in result.reply_text


def test_asking_with_nothing_running_says_there_is_nothing(tmp_path, projects, jobs):
    result = _run(_cfg(tmp_path, projects))

    assert result.success is True
    assert jobs.running() is None
    assert result.reply_text


def test_asking_while_it_runs_names_the_project(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = _cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    result = _run(cfg)

    assert result.success is True
    assert "chantipay" in result.reply_text


def test_asking_after_it_finished_gives_the_report(tmp_path, projects, jobs):
    cfg = _cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert jobs.wait(20.0)

    result = _run(cfg, projet="chantipay")

    assert result.success is True
    assert "Fait." in result.reply_text
