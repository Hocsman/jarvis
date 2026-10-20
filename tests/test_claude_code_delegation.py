"""Handing a project to Claude Code: what runs, where, and what comes back.

These run a real child process standing in for Claude Code (see
claude_code_fakes.py), so "the task went in on standard input" and "the
job was killed" are facts about a process rather than about a mock's
call count.

See src/jarvis/projets/projets.spec.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import psutil
import pytest

from claude_code_fakes import Ledger, install_fake_claude, make_cfg, make_projects
from jarvis.projets import delegation as dl

NAMES = ("chantipay", "nikahscore", "france-carte-grise", "Relayer")


@pytest.fixture
def projects(tmp_path):
    return make_projects(tmp_path, NAMES)


@pytest.fixture
def fake_claude(tmp_path, monkeypatch):
    return install_fake_claude(tmp_path, monkeypatch, dl)


@pytest.fixture
def delegations():
    d = dl.Delegations()
    delivered = []
    d.open(delivered.append)
    d.delivered = delivered
    yield d
    d.close()


def _launch(d, cfg, projects, name="chantipay", tache="Ajoute un bouton de déconnexion.",
            modifier=True, origin="voix", db=None):
    return d.start(
        cfg=cfg, db=db, projet=name, path=projects / name, tache=tache,
        modifier=modifier, origin=origin, risk="destructif" if modifier else "lecture",
    )


def _wait(d, timeout=20.0):
    assert d.wait(timeout), "the job did not finish in time"


# ── Finding the project from its name ─────────────────────────────────


def test_a_spoken_name_finds_its_folder(projects):
    """Names arrive transcribed: spaces where the folder has none, a
    hyphenated folder said as words, any case. Those are exact."""
    for spoken, folder in (("Nikah Score", "nikahscore"),
                           ("France carte grise", "france-carte-grise"),
                           ("relayer", "Relayer")):
        found = dl.resolve_project(projects, spoken)
        assert found.path == projects / folder
        assert found.exact is True


def test_a_near_miss_finds_its_folder_but_is_not_exact(projects):
    found = dl.resolve_project(projects, "chantipai")

    assert found.path == projects / "chantipay"
    assert found.exact is False


def test_two_candidates_run_nothing_and_name_both(tmp_path):
    root = make_projects(tmp_path, ("snake", "snack"))

    found = dl.resolve_project(root, "snak")

    assert found.path is None
    assert found.reason == dl.AMBIGUOUS
    assert sorted(found.candidates) == ["snack", "snake"]


def test_an_unknown_name_lists_what_exists(projects):
    found = dl.resolve_project(projects, "zzzzzz")

    assert found.path is None
    assert found.reason == dl.UNKNOWN
    assert sorted(found.candidates) == sorted(NAMES)


def test_hidden_folders_and_files_are_not_projects(projects):
    (projects / ".cache").mkdir()
    (projects / "notes.txt").write_text("x", encoding="utf-8")

    names = [p.name for p in dl.list_projects(projects)]

    assert ".cache" not in names
    assert "notes.txt" not in names


def test_a_missing_root_is_not_a_place_to_look(tmp_path):
    assert dl.resolve_project(tmp_path / "absent", "chantipay").reason == dl.NO_ROOT


def test_a_link_leading_out_of_the_root_is_not_a_project(tmp_path):
    root = tmp_path / "Projets"
    root.mkdir()
    outside = tmp_path / "ailleurs"
    outside.mkdir()
    try:
        (root / "lien").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine cannot create a directory link")

    assert dl.resolve_project(root, "lien").path is None


# ── Finding the project named inside free text ────────────────────────


def test_a_project_named_inside_the_task_is_found(projects):
    """Small models fold the project into the task and leave `projet`
    empty. The names are folder names, not words of any language."""
    found = dl.mentioned_project(projects, "add a logout button to chantipay")

    assert found.path == projects / "chantipay"
    assert found.exact is False


def test_a_project_said_in_several_words_is_found_in_the_task(projects):
    found = dl.mentioned_project(projects, "corrige le bug dans France carte grise")
    assert found.path == projects / "france-carte-grise"


def test_a_name_inside_another_word_is_not_a_mention(tmp_path):
    """'core' is inside 'score'. Only whole words count."""
    root = make_projects(tmp_path, ("core", "jeu-quiz"))

    assert dl.mentioned_project(root, "améliore le calcul du score").path is None


def test_a_name_is_never_assembled_across_two_arguments(projects):
    assert dl.mentioned_project(projects, "chanti", "pay la facture").path is None


def test_the_longer_name_wins_over_the_one_it_contains(tmp_path):
    root = make_projects(tmp_path, ("Relayer", "Relayer-main"))

    assert dl.mentioned_project(root, "fix the tests in relayer main").path == root / "Relayer-main"
    assert dl.mentioned_project(root, "fix the tests in relayer").path == root / "Relayer"


def test_two_projects_in_the_task_are_a_question(projects):
    found = dl.mentioned_project(projects, "compare chantipay and nikahscore")

    assert found.reason == dl.AMBIGUOUS
    assert sorted(found.candidates) == ["chantipay", "nikahscore"]


# ── Choosing between the two ──────────────────────────────────────────


def test_a_named_project_is_never_replaced_by_one_the_task_mentions(tmp_path):
    """The card shows `projet`. A folder the task happens to mention must
    not silently take its place."""
    root = make_projects(tmp_path, ("nikahscore", "finances"))

    found = dl.choose_project(root, "mon appli RH", ["ajoute la page finances"])

    assert found.path is None
    assert found.reason == dl.UNKNOWN


def test_a_project_field_carrying_extra_words_still_names_it(projects):
    found = dl.choose_project(projects, "mentaltracker database layer", [])
    assert found.path is None  # not one of these projects

    found = dl.choose_project(projects, "chantipay database layer", [])
    assert found.path == projects / "chantipay"
    assert found.exact is False


def test_with_no_project_field_the_task_is_searched(projects):
    found = dl.choose_project(projects, "", ["add a logout button to chantipay"])

    assert found.path == projects / "chantipay"
    assert found.exact is False


# ── What runs ─────────────────────────────────────────────────────────


def test_the_task_goes_in_on_stdin_and_never_on_the_command_line(
        tmp_path, projects, fake_claude, delegations):
    """On Windows an npm install of Claude Code is a .cmd file, parsed
    by cmd.exe: a task carrying `&` on the command line would run as a
    second command."""
    task = 'Corrige le login & del /q C:\\ | echo "pwned" %PATH% ^ !x!'
    _launch(delegations, make_cfg(tmp_path, projects), projects, tache=task)
    _wait(delegations)

    (call,) = fake_claude()
    assert call["stdin"] == task
    assert not any("pwned" in arg or "del /q" in arg for arg in call["argv"])
    assert Path(call["cwd"]).resolve() == (projects / "chantipay").resolve()


def test_the_command_line_carries_nothing_cmd_exe_would_parse():
    argv = dl.build_command(["claude"], modifier=False,
                            resume="aaaaaaaa-0000-4000-8000-000000000001")
    for arg in argv:
        for char in '"&|<>^%!\r\n':
            assert char not in arg, (arg, char)


def _disallowed(argv):
    i = argv.index("--disallowedTools")
    out = []
    for arg in argv[i + 1:]:
        if arg.startswith("-"):
            break
        out.append(arg)
    return out


def test_a_question_cannot_edit_run_commands_or_fetch(
        tmp_path, projects, fake_claude, delegations):
    """Headless `default` mode alone is not read-only: Claude Code still
    honours allow rules from the user's and the project's settings. Deny
    rules beat allow rules."""
    _launch(delegations, make_cfg(tmp_path, projects), projects, modifier=False)
    _wait(delegations)

    (call,) = fake_claude()
    argv = call["argv"]
    assert argv[argv.index("--permission-mode") + 1] == "default"
    for tool in ("Bash", "Edit", "Write", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch"):
        assert tool in _disallowed(argv)
    assert "--strict-mcp-config" in argv


def test_a_change_may_edit_but_runs_no_command(
        tmp_path, projects, fake_claude, delegations):
    _launch(delegations, make_cfg(tmp_path, projects), projects, modifier=True)
    _wait(delegations)

    (call,) = fake_claude()
    argv = call["argv"]
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
    assert "Bash" in _disallowed(argv)
    assert "Edit" not in _disallowed(argv)
    assert "--strict-mcp-config" in argv
    assert "-p" in argv
    assert argv[argv.index("--output-format") + 1] == "json"


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe's lookup rule")
def test_the_project_folder_is_never_searched_for_executables(
        tmp_path, projects, fake_claude, delegations):
    """An npm shim runs `node` by bare name, and cmd.exe looks in the
    current directory first: a repository's own node.exe would run."""
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (call,) = fake_claude()
    assert call["no_cwd_exe"] == "1"


# ── What comes back ───────────────────────────────────────────────────


def test_the_report_reaches_the_delivery(tmp_path, projects, fake_claude, delegations):
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is True
    assert outcome.report == "J'ai ajouté le bouton de déconnexion."
    said = dl.compose_announcement(outcome)
    assert "chantipay" in said
    assert "« J'ai ajouté le bouton de déconnexion. »" in said


def test_a_failure_says_so_with_its_reason(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert "Le build a échoué." in dl.compose_announcement(outcome)


def test_output_that_is_not_a_report_is_a_failure_with_stderr_as_reason(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "garbage")
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert "not logged in" in outcome.report


def test_secrets_in_the_report_are_not_read_aloud(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    secret = "sk-" + "a1B2c3D4" * 5
    monkeypatch.setenv("FAKE_CLAUDE_REPORT", f"J'ai mis la clé {secret} dans le .env.")
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert secret not in outcome.report
    assert secret not in dl.compose_announcement(outcome)


def test_a_long_report_is_cut_at_a_sentence(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    """It is spoken, kept in every prompt for minutes and quoted back:
    the rule asking for three sentences is a request, the cap is a fact."""
    monkeypatch.setenv("FAKE_CLAUDE_REPORT", " ".join(
        f"Phrase numéro {i} du rapport, assez longue pour compter." for i in range(60)))
    _launch(delegations, make_cfg(tmp_path, projects), projects)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert len(outcome.report) <= dl._REPORT_MAX_CHARS + 10
    assert outcome.report.startswith("Phrase numéro 0")
    assert "compter." in outcome.report


def test_a_job_past_its_deadline_is_killed_and_reported(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    began = time.monotonic()
    _launch(delegations, make_cfg(tmp_path, projects, claude_code_timeout_sec=1.5), projects)
    _wait(delegations, timeout=30.0)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert time.monotonic() - began < 25.0
    assert delegations.running() is None


def test_what_the_agent_started_dies_with_it(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    """Killing only the top process leaves the agent running: an npm
    install runs node under cmd.exe."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "grandchild")
    _launch(delegations, make_cfg(tmp_path, projects, claude_code_timeout_sec=3.0), projects)
    _wait(delegations, timeout=30.0)

    marker = Path(str(fake_claude.log) + ".grandchild")
    pid = int(marker.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.skipif(os.name != "nt", reason="tied to the parent through a Windows job object")
def test_a_job_does_not_outlive_a_killed_yuba(tmp_path, projects, fake_claude, monkeypatch):
    """A daemon killed rather than stopped never runs its shutdown: the
    agent must not keep editing with nobody left to report to."""
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    src = Path(__file__).resolve().parent.parent / "src"
    script = tmp_path / "host.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(src)!r})
        from types import SimpleNamespace
        from pathlib import Path
        from jarvis.projets import delegation as dl
        dl.resolve_claude_command = lambda c: [sys.executable, {str(tmp_path / 'fake_claude.py')!r}]
        cfg = SimpleNamespace(claude_code_projects_dir={str(projects)!r}, claude_code_command="claude",
                              claude_code_timeout_sec=120.0, db_path={str(tmp_path / 'd' / 'j.db')!r})
        d = dl.Delegations()
        d.open(lambda o: None)
        d.start(cfg=cfg, db=None, projet="chantipay", path=Path({str(projects / 'chantipay')!r}),
                tache="x", modifier=True, origin="voix", risk="destructif")
        print("started", flush=True)
        time.sleep(120)
    """), encoding="utf-8")
    host = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    try:
        assert host.stdout.readline().strip() == "started"
        deadline = time.monotonic() + 15
        while not fake_claude() and time.monotonic() < deadline:
            time.sleep(0.1)
        (call,) = fake_claude()
        host.kill()
        host.wait(10)
        deadline = time.monotonic() + 10
        while psutil.pid_exists(call["pid"]) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not psutil.pid_exists(call["pid"])
    finally:
        if host.poll() is None:
            host.kill()


# ── One at a time, open and closed ────────────────────────────────────


def test_a_second_task_while_one_runs_is_refused(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    cfg = make_cfg(tmp_path, projects)
    started, first = _launch(delegations, cfg, projects)
    assert started is True

    started, running = _launch(delegations, cfg, projects, name="nikahscore")

    assert started is False
    assert running.projet == "chantipay"


def test_closing_kills_a_running_job_and_delivers_nothing(
        tmp_path, projects, fake_claude, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "sleep")
    d = dl.Delegations()
    delivered = []
    d.open(delivered.append)
    _launch(d, make_cfg(tmp_path, projects), projects)
    deadline = time.monotonic() + 10
    while not fake_claude() and time.monotonic() < deadline:
        time.sleep(0.05)

    d.close()

    assert d.running() is None
    assert d.wait(10.0)
    assert delivered == []


def test_nothing_starts_once_closed(tmp_path, projects, fake_claude, delegations):
    """A turn still in flight when the daemon stops must not launch a job
    that outlives it."""
    delegations.close()

    with pytest.raises(dl.DelegationsClosed):
        _launch(delegations, make_cfg(tmp_path, projects), projects)
    assert fake_claude() == []


def test_reopening_starts_fresh(tmp_path, projects, fake_claude, delegations):
    """A daemon restarted inside the same process keeps no session and no
    last report from the previous one."""
    cfg = make_cfg(tmp_path, projects)
    _launch(delegations, cfg, projects)
    _wait(delegations)
    delegations.close()

    delegations.open(delegations.delivered.append)
    assert delegations.last() is None
    _launch(delegations, cfg, projects)
    _wait(delegations)

    first, second = fake_claude()
    assert "--resume" not in second["argv"]


# ── What the next one remembers ───────────────────────────────────────


def test_a_follow_up_change_on_the_same_project_resumes_its_session(
        tmp_path, projects, fake_claude, delegations):
    cfg = make_cfg(tmp_path, projects)
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


def test_a_change_never_resumes_a_session_no_one_approved(
        tmp_path, projects, fake_claude, delegations):
    """A question runs without a card. Its conversation must not ride
    into a change the user approved without seeing it."""
    cfg = make_cfg(tmp_path, projects)
    _launch(delegations, cfg, projects, modifier=False)
    _wait(delegations)
    _launch(delegations, cfg, projects, modifier=True)
    _wait(delegations)
    _launch(delegations, cfg, projects, modifier=False)
    _wait(delegations)

    question, change, question_after = fake_claude()
    assert "--resume" not in change["argv"]
    assert "--resume" in question_after["argv"]


@pytest.mark.parametrize("session", ["x & calc.exe", "--dangerously-skip-permissions",
                                     "not-a-uuid-at-all"])
def test_a_session_id_that_is_not_one_never_reaches_the_command_line(
        tmp_path, projects, fake_claude, delegations, monkeypatch, session):
    monkeypatch.setenv("FAKE_CLAUDE_SESSION", session)
    cfg = make_cfg(tmp_path, projects)
    _launch(delegations, cfg, projects)
    _wait(delegations)
    _launch(delegations, cfg, projects)
    _wait(delegations)

    _, follow_up = fake_claude()
    assert "--resume" not in follow_up["argv"]
    assert session not in follow_up["argv"]


# ── Failing to start ──────────────────────────────────────────────────


def test_a_project_gone_before_launch_is_reported_and_frees_the_slot(
        tmp_path, projects, fake_claude, delegations):
    db = Ledger()
    (projects / "chantipay").rmdir()

    _launch(delegations, make_cfg(tmp_path, projects), projects, db=db)
    _wait(delegations)

    (outcome,) = delegations.delivered
    assert outcome.ok is False
    assert "chantipay" in outcome.report
    assert [row["outcome"] for row in db.rows] == ["démarré", "échec"]
    assert delegations.running() is None


def test_a_thread_that_cannot_start_frees_the_slot(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    class _NoThread(threading.Thread):
        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(dl.threading, "Thread", _NoThread)
    db = Ledger()

    with pytest.raises(RuntimeError):
        _launch(delegations, make_cfg(tmp_path, projects), projects, db=db)

    assert delegations.running() is None
    assert delegations.wait(1.0)
    assert [row["outcome"] for row in db.rows] == ["démarré", "échec"]


# ── The ledger ────────────────────────────────────────────────────────


def test_the_job_is_bracketed_in_the_ledger(tmp_path, projects, fake_claude, delegations):
    db = Ledger()
    _launch(delegations, make_cfg(tmp_path, projects), projects, db=db)
    _wait(delegations)

    opened, closed = db.rows
    assert opened["tool"] == closed["tool"] == "claudeCode:chantipay"
    assert opened["outcome"] == "démarré"
    assert closed["outcome"] == "ok"
    assert opened["request_id"] == closed["request_id"]
    assert opened["verdict"] == closed["verdict"]
    assert closed["duration_ms"] is not None
    assert "Ajoute" not in json.dumps(db.rows, default=str), (
        "the ledger records what was done, never the task's text nor "
        "anything Claude Code said")


def test_a_failed_job_closes_its_bracket_as_a_failure(
        tmp_path, projects, fake_claude, delegations, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "error")
    db = Ledger()
    _launch(delegations, make_cfg(tmp_path, projects), projects, db=db)
    _wait(delegations)

    assert [row["outcome"] for row in db.rows] == ["démarré", "échec"]
