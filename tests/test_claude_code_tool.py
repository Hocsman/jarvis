"""The `claudeCode` tool: what it costs to call, and what it says back.

The work itself is tested in test_claude_code_delegation.py. Here: the
risk each call carries at the gate, the honest refusals (off, not
installed, unknown or ambiguous project, a change whose project the card
does not name exactly, Yuba's own store, a job already running), and the
reply that says a job has started rather than claiming it is done.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import pytest

from claude_code_fakes import install_fake_claude, make_cfg, make_projects
from jarvis.projets import delegation as dl
from jarvis.tools.builtin.claude_code import ClaudeCodeTool, vet_call
from jarvis.tools.policy import RISK_DESTRUCTIVE, RISK_READ


@pytest.fixture
def projects(tmp_path):
    return make_projects(tmp_path, ("chantipay", "nikahscore", "snake", "snack"))


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """A fresh job keeper standing in for the process-wide one."""
    install_fake_claude(tmp_path, monkeypatch, dl)
    keeper = dl.Delegations()
    delivered = []
    keeper.open(delivered.append)
    keeper.delivered = delivered
    monkeypatch.setattr(dl, "DELEGATIONS", keeper)
    yield keeper
    keeper.close()


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


def test_it_is_offered_only_once_a_folder_is_named(tmp_path, projects):
    """Off, it would cost every router call its description and could
    only ever refuse."""
    from jarvis.tools.registry import offered_builtin_tools

    off = make_cfg(tmp_path, projects, claude_code_projects_dir="")
    on = make_cfg(tmp_path, projects)

    assert "claudeCode" not in offered_builtin_tools(off)
    assert "claudeCode" in offered_builtin_tools(on)
    assert "getTime" in offered_builtin_tools(off)


# ── Honest refusals ───────────────────────────────────────────────────


def test_off_until_a_folder_is_named(tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects, claude_code_projects_dir=""),
                  projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert "claude_code_projects_dir" in result.reply_text
    assert jobs.running() is None


def test_not_installed_says_so_and_runs_nothing(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setattr(dl, "resolve_claude_command", lambda command: None)

    result = _run(make_cfg(tmp_path, projects), projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert "Claude Code" in result.reply_text
    assert jobs.running() is None


def test_an_unknown_project_lists_those_that_exist(tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects), projet="zzzzzz", tache="Ajoute un bouton.")

    assert result.success is False
    for name in ("chantipay", "nikahscore"):
        assert name in result.reply_text
    assert jobs.running() is None


def test_an_ambiguous_project_names_the_candidates(tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects), projet="snak", tache="Ajoute un bouton.")

    assert result.success is False
    assert "snake" in result.reply_text and "snack" in result.reply_text
    assert jobs.running() is None


def test_a_task_without_a_project_lists_the_projects_and_runs_nothing(
        tmp_path, projects, jobs):
    """So she can ask which one, and the next answer finds the tool
    still offered (a failed call carries over to the next turn)."""
    result = _run(make_cfg(tmp_path, projects), tache="Ajoute un bouton.")

    assert result.success is False
    for name in ("chantipay", "nikahscore"):
        assert name in result.reply_text
    assert jobs.running() is None


@pytest.mark.parametrize("projet", ["", "zzzzzz"], ids=["no project", "an unknown one"])
def test_when_no_project_is_found_the_reply_says_how_the_last_job_went(
        tmp_path, projects, jobs, projet):
    """A small model files "how is Claude Code doing?" as a task with no
    project, or with a project that is not one."""
    cfg = make_cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert jobs.wait(20.0)

    result = _run(cfg, projet=projet, tache="où en est Claude Code ?", modifier=False)

    assert "J'ai ajouté le bouton de déconnexion." in result.reply_text
    assert len(jobs.delivered) == 1


def test_a_project_holding_yubas_own_store_is_refused(tmp_path, projects, jobs):
    """The core lists her permissions. An agent editing the folder that
    holds it would make them advisory."""
    cfg = make_cfg(tmp_path, projects,
                   db_path=str(projects / "chantipay" / "data" / "jarvis.db"))

    result = _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert jobs.running() is None


def test_a_project_inside_yubas_own_store_is_refused(tmp_path, jobs):
    """The other direction: a projects folder pointed at the core."""
    core = tmp_path / "data" / "yuba"
    (core / "journal").mkdir(parents=True)
    cfg = make_cfg(tmp_path, core)

    result = _run(cfg, projet="journal", tache="Résume le journal.", modifier=False)

    assert result.success is False
    assert jobs.running() is None


# ── A change runs only on the project the card names ──────────────────


def test_a_change_on_a_near_miss_asks_for_the_exact_name(tmp_path, projects, jobs):
    """The card would show "chantipai" while chantipay is edited."""
    result = _run(make_cfg(tmp_path, projects), projet="chantipai",
                  tache="Ajoute un bouton.", modifier=True)

    assert result.success is False
    assert "projet='chantipay'" in result.reply_text
    assert jobs.running() is None


def test_a_change_named_only_inside_the_task_asks_for_the_exact_name(
        tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects), tache="Ajoute un bouton dans chantipay.")

    assert result.success is False
    assert "projet='chantipay'" in result.reply_text
    assert jobs.running() is None


def test_a_question_on_a_near_miss_or_inside_the_task_runs(tmp_path, projects, jobs):
    cfg = make_cfg(tmp_path, projects)

    first = _run(cfg, projet="chantipai", tache="Explique l'auth.", modifier=False)
    assert first.success is True
    assert jobs.wait(20.0)
    second = _run(cfg, tache="Explique l'auth de nikahscore.", modifier=False)
    assert second.success is True
    assert jobs.wait(20.0)

    assert [o.job.projet for o in jobs.delivered] == ["chantipay", "nikahscore"]


def test_vet_call_is_the_tools_own_decision(tmp_path, projects):
    """What the eval measures is what the tool does."""
    cfg = make_cfg(tmp_path, projects)

    refusal, path = vet_call({"projet": "chantipay", "tache": "x"}, cfg)
    assert refusal is None and path == projects / "chantipay"

    refusal, path = vet_call({"tache": "x dans chantipay"}, cfg)
    assert path is None and "projet='chantipay'" in refusal


# ── Launching, and asking how it is going ─────────────────────────────


def test_a_launch_says_it_has_started_not_that_it_is_done(tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects), projet="chantipay",
                  tache="Ajoute un bouton.", modifier=True)

    assert result.success is True
    assert "chantipay" in result.reply_text
    assert jobs.wait(20.0)
    (outcome,) = jobs.delivered
    assert outcome.job.origin == "voix"
    assert outcome.job.modifier is True


def test_a_question_runs_read_only(tmp_path, projects, jobs):
    _run(make_cfg(tmp_path, projects), projet="chantipay",
         tache="Explique l'authentification.", modifier=False)

    assert jobs.wait(20.0)
    (outcome,) = jobs.delivered
    assert outcome.job.modifier is False


def test_a_second_task_names_the_one_running(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    result = _run(cfg, projet="nikahscore", tache="Corrige le build.")

    assert result.success is False
    assert "chantipay" in result.reply_text


def test_a_task_after_yuba_stopped_its_jobs_runs_nothing(tmp_path, projects, jobs):
    jobs.close()

    result = _run(make_cfg(tmp_path, projects), projet="chantipay", tache="Ajoute un bouton.")

    assert result.success is False
    assert jobs.running() is None


def test_asking_with_nothing_running_says_there_is_nothing(tmp_path, projects, jobs):
    result = _run(make_cfg(tmp_path, projects))

    assert result.success is True
    assert jobs.running() is None
    assert result.reply_text


def test_asking_while_it_runs_names_the_project(tmp_path, projects, jobs, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")

    result = _run(cfg)

    assert result.success is True
    assert "chantipay" in result.reply_text


def test_asking_after_it_finished_gives_the_report(tmp_path, projects, jobs):
    cfg = make_cfg(tmp_path, projects)
    _run(cfg, projet="chantipay", tache="Ajoute un bouton.")
    assert jobs.wait(20.0)

    result = _run(cfg, projet="chantipay")

    assert result.success is True
    assert "J'ai ajouté le bouton de déconnexion." in result.reply_text


def test_its_replies_are_never_digested():
    """Written by code, already bounded, and carrying project names the
    user must repeat word for word. A summary would paraphrase exactly
    those, and cost a model call with the query lock held."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from jarvis.reply.engine import _maybe_digest_tool_result

    cfg = SimpleNamespace(tool_result_digest_enabled=True,
                          ollama_chat_model="gemma4:e2b", llm_chat_model="gemma4:e2b")
    raw = "Claude Code was not started: the projects are: " + ", ".join(
        f"projet-{i}" for i in range(40))
    with patch("jarvis.reply.enrichment.call_llm_direct") as llm:
        out = _maybe_digest_tool_result(cfg=cfg, query="q", tool_name="claudeCode",
                                        raw_tool_result=raw)
        llm.assert_not_called()
    assert out == raw
