"""Handing a project to Claude Code: what runs, where, and what comes back.

These run a real child process standing in for Claude Code, so "the task
went in on standard input" and "the job was killed" are facts about a
process rather than about a mock's call count. The stand-in logs what it
was given (its arguments, its standard input, its working directory) and
answers the way `claude -p --output-format json` does.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from jarvis.projets import claude_code as cc


_FAKE_CLAUDE = r'''
import json, os, sys, time
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
task = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "stdin": task, "cwd": os.getcwd()}) + "\n")
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
if mode == "sleep":
    time.sleep(60)
if mode == "garbage":
    sys.stderr.write("boom: not logged in\n")
    sys.exit(1)
session = os.environ.get("FAKE_CLAUDE_SESSION", "aaaaaaaa-0000-4000-8000-000000000001")
if mode == "error":
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "result": "Le build a échoué.",
                      "session_id": session}))
    sys.exit(1)
report = os.environ.get("FAKE_CLAUDE_REPORT", "J'ai ajouté le bouton de déconnexion.")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": report, "session_id": session}))
'''


class _Ledger:
    """A database that only keeps the ledger rows, as dicts."""

    def __init__(self):
        self.rows = []

    def record_action(self, **row):
        self.rows.append(row)


@pytest.fixture
def projects(tmp_path):
    root = tmp_path / "Projets"
    for name in ("chantipay", "nikahscore", "france-carte-grise", "Relayer"):
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(_FAKE_CLAUDE, encoding="utf-8")
    log = tmp_path / "claude.log"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setattr(cc, "resolve_claude_command",
                        lambda command: [sys.executable, str(script)])

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    return calls


def _cfg(tmp_path, projects, **over):
    values = dict(
        claude_code_projects_dir=str(projects),
        claude_code_command="claude",
        claude_code_timeout_sec=30.0,
        db_path=str(tmp_path / "data" / "jarvis.db"),
    )
    values.update(over)
    return SimpleNamespace(**values)


@pytest.fixture
def delegations():
    d = cc.Delegations()
    delivered = []
    d.set_delivery(delivered.append)
    d.delivered = delivered
    yield d
    d.shutdown()


def _launch(d, cfg, projects, name="chantipay", tache="Ajoute un bouton de déconnexion.",
            modifier=True, origin="voix", db=None):
    started, job = d.start(
        cfg=cfg, db=db, projet=name, path=projects / name, tache=tache,
        modifier=modifier, origin=origin, risk="destructif" if modifier else "lecture",
    )
    return started, job


def _wait(d, timeout=20.0):
    assert d.wait(timeout), "the job did not finish in time"


# ── Finding the project ───────────────────────────────────────────────


def test_a_spoken_name_finds_its_folder(projects):
    """Names arrive transcribed: spaces where the folder has none, a
    hyphenated folder said as words, any case."""
    assert cc.resolve_project(projects, "Nikah Score").path == projects / "nikahscore"
    assert cc.resolve_project(projects, "France carte grise").path == projects / "france-carte-grise"
    assert cc.resolve_project(projects, "relayer").path == projects / "Relayer"


def test_a_near_miss_finds_its_folder_when_only_one_is_near(projects):
    assert cc.resolve_project(projects, "chantipai").path == projects / "chantipay"


def test_two_candidates_run_nothing_and_name_both(tmp_path):
    root = tmp_path / "Projets"
    for name in ("snake", "snack"):
        (root / name).mkdir(parents=True)

    found = cc.resolve_project(root, "snak")

    assert found.path is None
    assert found.reason == "ambiguous"
    assert sorted(found.candidates) == ["snack", "snake"]


def test_an_unknown_name_lists_what_exists(projects):
    found = cc.resolve_project(projects, "zzzzzz")

    assert found.path is None
    assert found.reason == "unknown"
    assert sorted(found.candidates) == sorted(
        ["chantipay", "nikahscore", "france-carte-grise", "Relayer"])


def test_a_project_named_only_inside_the_task_is_found(projects):
    """Small models fold the project into the task and leave `projet`
    empty. The names are folder names, not words of any language."""
    found = cc.mentioned_project(projects, "add a logout button to chantipay")
    assert found.path == projects / "chantipay"


def test_a_project_said_in_several_words_is_found_in_the_task(projects):
    found = cc.mentioned_project(projects, "corrige le bug dans France carte grise")
    assert found.path == projects / "france-carte-grise"


def test_the_longer_name_wins_over_the_one_it_contains(tmp_path):
    root = tmp_path / "Projets"
    for name in ("Relayer", "Relayer-main"):
        (root / name).mkdir(parents=True)

    assert cc.mentioned_project(root, "fix the tests in relayer main").path == root / "Relayer-main"
    assert cc.mentioned_project(root, "fix the tests in relayer").path == root / "Relayer"


def test_two_projects_in_the_task_are_a_question(projects):
    found = cc.mentioned_project(projects, "compare chantipay and nikahscore")

    assert found.path is None
    assert found.reason == "ambiguous"
    assert sorted(found.candidates) == ["chantipay", "nikahscore"]


def test_a_task_naming_no_project_finds_none(projects):
    found = cc.mentioned_project(projects, "how is it going?")

    assert found.path is None
    assert found.reason == "unknown"


def test_hidden_folders_and_files_are_not_projects(projects):
    (projects / ".cache").mkdir()
    (projects / "notes.txt").write_text("x", encoding="utf-8")

    names = [p.name for p in cc.list_projects(projects)]

    assert ".cache" not in names
    assert "notes.txt" not in names
    assert cc.resolve_project(projects, "notes txt").path is None


def test_a_missing_root_is_not_a_place_to_look(tmp_path):
    found = cc.resolve_project(tmp_path / "absent", "chantipay")

    assert found.path is None
    assert found.reason == "no_root"


def test_a_link_leading_out_of_the_root_is_not_a_project(tmp_path):
    root = tmp_path / "Projets"
    root.mkdir()
    outside = tmp_path / "ailleurs"
    outside.mkdir()
    try:
        (root / "lien").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create a directory link")

    assert cc.resolve_project(root, "lien").path is None


# ── What runs ─────────────────────────────────────────────────────────


def test_the_task_goes_in_on_stdin_and_never_on_the_command_line(
        tmp_path, projects, fake_claude, delegations):
    """On Windows an npm install of Claude Code is a .cmd file, parsed
    by cmd.exe: a task carrying `&` on the command line would run as a
    second command."""
    task = 'Corrige le login & del /q C:\\ | echo "pwned" %PATH% ^ !x!'
    _launch(delegations, _cfg(tmp_path, projects), projects, tache=task)
    _wait(delegations)

    (call,) = fake_claude()
    assert call["stdin"] == task
    assert not any("pwned" in arg or "del /q" in arg for arg in call["argv"])
    assert Path(call["cwd"]).resolve() == (projects / "chantipay").resolve()


def test_the_report_rule_carries_nothing_cmd_exe_would_parse():
    for char in '"&|<>^%!\r\n':
        assert char not in cc._REPORT_RULE


def test_a_change_may_edit_and_a_question_only_reads(
        tmp_path, projects, fake_claude, delegations):
    cfg = _cfg(tmp_path, projects)
    _launch(delegations, cfg, projects, modifier=True)
    _wait(delegations)
    _launch(delegations, cfg, projects, name="nikahscore", modifier=False)
    _wait(delegations)

    change, question = fake_claude()
    mode = lambda argv: argv[argv.index("--permission-mode") + 1]
    assert mode(change["argv"]) == "acceptEdits"
    assert mode(question["argv"]) == "default"
    for call in (change, question):
        assert "-p" in call["argv"]
        assert call["argv"][call["argv"].index("--output-format") + 1] == "json"


# ── What comes back ───────────────────────────────────────────────────


def test_the_report_reaches_the_delivery(tmp_path, projects, fake_claude, delegations):
    _launch(delegations, _cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is True
    assert outcome.report == "J'ai ajouté le bouton de déconnexion."
    said = cc.compose_announcement(outcome)
    assert "chantipay" in said
    assert "J'ai ajouté le bouton de déconnexion." in said


def test_a_failure_says_so_with_its_reason(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    _launch(delegations, _cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert "Le build a échoué." in cc.compose_announcement(outcome)


def test_output_that_is_not_a_report_is_a_failure_with_stderr_as_reason(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    _launch(delegations, _cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert "not logged in" in outcome.report


def test_secrets_in_the_report_are_not_read_aloud(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    secret = "sk-" + "a1B2c3D4" * 5
    monkeypatch.setenv("FAKE_CLAUDE_REPORT", f"J'ai mis la clé {secret} dans le .env.")
    _launch(delegations, _cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert secret not in cc.compose_announcement(outcome)


def test_a_job_past_its_deadline_is_killed_and_reported(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    began = time.monotonic()
    _launch(delegations, _cfg(tmp_path, projects, claude_code_timeout_sec=1.5), projects)
    _wait(delegations, timeout=20.0)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert time.monotonic() - began < 15.0
    assert delegations.running() is None


# ── One at a time, and what the next one remembers ────────────────────


def test_a_second_task_while_one_runs_is_refused(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = _cfg(tmp_path, projects)
    started, first = _launch(delegations, cfg, projects)
    assert started is True

    started, running = _launch(delegations, cfg, projects, name="nikahscore")

    assert started is False
    assert running.projet == "chantipay"


def test_shutdown_kills_a_running_job(tmp_path, projects, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    d = cc.Delegations()
    _launch(d, _cfg(tmp_path, projects), projects)
    deadline = time.monotonic() + 10
    while not fake_claude() and time.monotonic() < deadline:
        time.sleep(0.05)

    d.shutdown()

    assert d.running() is None
    assert d.wait(10.0)


def test_a_follow_up_on_the_same_project_resumes_its_session(
        tmp_path, projects, fake_claude, delegations):
    cfg = _cfg(tmp_path, projects)
    _launch(delegations, cfg, projects)
    _wait(delegations)
    _launch(delegations, cfg, projects, tache="Ajoute un test pour ce bouton.")
    _wait(delegations)
    _launch(delegations, cfg, projects, name="nikahscore")
    _wait(delegations)

    first, follow_up, other = fake_claude()
    assert "--resume" not in first["argv"]
    assert follow_up["argv"][follow_up["argv"].index("--resume") + 1] == \
        "aaaaaaaa-0000-4000-8000-000000000001"
    assert "--resume" not in other["argv"]


def test_a_session_id_that_is_not_one_never_reaches_the_command_line(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_SESSION", "x & calc.exe")
    cfg = _cfg(tmp_path, projects)
    _launch(delegations, cfg, projects)
    _wait(delegations)
    _launch(delegations, cfg, projects)
    _wait(delegations)

    _, follow_up = fake_claude()
    assert "--resume" not in follow_up["argv"]
    assert not any("calc" in arg for arg in follow_up["argv"])


# ── The ledger ────────────────────────────────────────────────────────


def test_the_job_is_bracketed_in_the_ledger(tmp_path, projects, fake_claude, delegations):
    db = _Ledger()
    _launch(delegations, _cfg(tmp_path, projects), projects, db=db)
    _wait(delegations)

    opened, closed = db.rows
    assert opened["tool"] == closed["tool"] == "claudeCode:chantipay"
    assert opened["outcome"] == "démarré"
    assert closed["outcome"] == "ok"
    assert opened["request_id"] == closed["request_id"]
    assert closed["duration_ms"] is not None
    assert "Ajoute" not in json.dumps(db.rows, default=str), (
        "the ledger records what was done, never the task's text twice "
        "nor anything Claude Code said")


def test_a_failed_job_closes_its_bracket_as_a_failure(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    db = _Ledger()
    _launch(delegations, _cfg(tmp_path, projects), projects, db=db)
    _wait(delegations)

    assert [row["outcome"] for row in db.rows] == ["démarré", "échec"]
