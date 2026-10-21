"""A stand-in for the Claude Code CLI, shared by the delegation tests.

It is a real child process: it logs what it was given (arguments,
standard input, working directory, one environment variable) and answers
the way `claude -p --output-format json` does. Its behaviour is chosen
through environment variables, which the child inherits.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

FAKE_CLAUDE = r'''
import json, os, subprocess, sys, time
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
task = sys.stdin.read()
with open(os.environ["FAKE_CLAUDE_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({
        "argv": sys.argv[1:], "stdin": task, "cwd": os.getcwd(), "pid": os.getpid(),
        "no_cwd_exe": os.environ.get("NoDefaultCurrentDirectoryInExePath"),
    }) + "\n")
mode = os.environ.get("FAKE_CLAUDE_MODE", "ok")
time.sleep(float(os.environ.get("FAKE_CLAUDE_DELAY", "0")))
if mode == "grandchild":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    with open(os.environ["FAKE_CLAUDE_LOG"] + ".grandchild", "w", encoding="utf-8") as f:
        f.write(str(child.pid))
    time.sleep(120)
if mode == "sleep":
    time.sleep(120)
if mode == "orphan":
    # A child that starts a grandchild and exits at once: the grandchild's
    # parent is gone, so walking parents from the top never reaches it,
    # and it holds the job's output open, so the job never ends by itself.
    subprocess.run([sys.executable, "-c",
                    "import subprocess, sys; "
                    "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], "
                    "stdin=subprocess.DEVNULL); "
                    "open(sys.argv[1], 'w').write(str(p.pid))",
                    os.environ["FAKE_CLAUDE_LOG"] + ".orphan"])
    time.sleep(120)
if mode == "garbage":
    sys.stderr.write("boom: not logged in\n")
    sys.exit(1)
session = os.environ.get("FAKE_CLAUDE_SESSION", "aaaaaaaa-0000-4000-8000-000000000001")
if mode in ("error", "linger_error"):
    print(json.dumps({"type": "result", "subtype": "error_during_execution",
                      "is_error": True, "result": "Le build a échoué.",
                      "session_id": session}), flush=True)
    if mode == "linger_error":
        with open(os.environ["FAKE_CLAUDE_LOG"] + ".printed", "w", encoding="utf-8") as f:
            f.write("1")
        time.sleep(120)
    sys.exit(1)
report = os.environ.get("FAKE_CLAUDE_REPORT", "J'ai ajouté le bouton de déconnexion.")
print(json.dumps({"type": "result", "subtype": "success", "is_error": False,
                  "result": report, "session_id": session}), flush=True)
if mode == "printed_then_fail":
    # Its result is out, then something after it fails on its own.
    sys.exit(1)
if mode == "linger":
    # Its result is printed, and it is still running: hooks, a slow exit.
    with open(os.environ["FAKE_CLAUDE_LOG"] + ".printed", "w", encoding="utf-8") as f:
        f.write("1")
    time.sleep(120)
'''


def install_fake_claude(tmp_path: Path, monkeypatch, delegation) -> "callable":
    """Make ``delegation`` run the stand-in. Returns a function giving the
    calls it logged, oldest first."""
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    log = tmp_path / "claude.log"
    monkeypatch.setenv("FAKE_CLAUDE_LOG", str(log))
    monkeypatch.setattr(delegation, "resolve_claude_command",
                        lambda command: [sys.executable, str(script)])

    def calls():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    calls.log = log
    return calls


def wait_for_calls(calls, count: int = 1, timeout: float = 10.0) -> list:
    """Wait until the stand-in has been started ``count`` times, and
    return what it logged."""
    deadline = time.monotonic() + timeout
    while len(calls()) < count and time.monotonic() < deadline:
        time.sleep(0.05)
    logged = calls()
    assert len(logged) >= count, f"the stand-in started {len(logged)} times, not {count}"
    return logged


def wait_for_file(path: Path, timeout: float = 10.0) -> Path:
    """Wait until the stand-in has written ``path``, a marker it leaves
    at a known point of its run."""
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert path.exists(), f"the stand-in never wrote {path.name}"
    return path


def make_projects(tmp_path: Path, names) -> Path:
    root = tmp_path / "Projets"
    for name in names:
        (root / name).mkdir(parents=True)
    return root


def make_cfg(tmp_path: Path, projects: Path, **over) -> SimpleNamespace:
    values = dict(
        claude_code_projects_dir=str(projects),
        claude_code_command="claude",
        claude_code_timeout_sec=30.0,
        db_path=str(tmp_path / "data" / "jarvis.db"),
    )
    values.update(over)
    return SimpleNamespace(**values)


class Ledger:
    """A database that only keeps the ledger rows, as dicts."""

    def __init__(self):
        self.rows = []

    def record_action(self, **row):
        self.rows.append(row)
