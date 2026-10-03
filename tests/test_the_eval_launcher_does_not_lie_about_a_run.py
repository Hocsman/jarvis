"""The eval launchers say what happened to the run, not what is convenient.

``scripts/run_evals.sh`` tells a suite that failed (pytest exit 1) apart from
a suite that never ran (5: nothing collected, 4: bad arguments, anything else:
pytest or the interpreter did not get as far as reporting). Reporting the
second kind as "some evaluations failed" is a run that measured nothing wearing
the face of one that did, and it sends the reader to the evals when the fault
is in the command. ``scripts/run_evals.bat`` is the launcher this project's
Windows developers actually use, so it keeps the same contract: the exit code
is pytest's own, the message says which of the two it was, and a default run
measures the small model that evals/helpers.py names as the canary.

Both launchers are run here, side by side, against a stand-in interpreter that
prints what it was asked and exits with a code the test chooses. No model and
no eval is involved, and the Ollama the launcher probes is either absent or a
stub on a loopback port of this test's own.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import pytest

from jarvis.config import DEFAULT_CHAT_MODEL, SUPPORTED_CHAT_MODELS

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# The tier above the default, the one the launcher compares it with.
LARGE_MODEL = list(SUPPORTED_CHAT_MODELS)[-1]

# An address curl refuses before it opens a socket: the launcher sees no
# Ollama, at once and without touching a network.
NO_OLLAMA = "unsupported://no-ollama"


def test_the_windows_launcher_is_plain_ascii():
    """cmd.exe does not render Unicode, so a .bat never carries any."""
    raw = (SCRIPTS / "run_evals.bat").read_bytes()
    assert all(byte < 128 for byte in raw), "run_evals.bat holds a non-ASCII byte"


def _git_bash() -> Optional[str]:
    """The bash that runs a shell script on this machine, if there is one.

    On Windows the one on PATH may be the WSL launcher, which cannot run a
    script from a Windows path, so only Git for Windows' own is used."""
    if os.name != "nt":
        return shutil.which("bash")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    candidate = Path(program_files) / "Git" / "usr" / "bin" / "bash.exe"
    return str(candidate) if candidate.exists() else None


@dataclass(frozen=True)
class Launcher:
    name: str
    script: Path
    # What the launcher exits with when the interpreter cannot be run.
    cannot_run: int

    def available(self) -> Optional[str]:
        """None when this launcher can run here, else why it cannot."""
        if self.script.suffix == ".bat":
            return None if os.name == "nt" else "cmd.exe runs this script"
        return None if _git_bash() else "no bash to run this script"

    def stand_in(self, folder: Path, exit_codes: dict[str, int]) -> Path:
        """An interpreter that prints the model it was run for and its
        arguments, then exits with the code chosen for that model (0 when
        none is). Asked for its version, it answers at once with success."""
        if self.script.suffix == ".bat":
            lines = ["@echo off", 'if "%1"=="--version" exit /b 0',
                     "echo STAND-IN model=%EVAL_JUDGE_MODEL% args=%*"]
            lines += [f'if "%EVAL_JUDGE_MODEL%"=="{model}" exit /b {code}'
                      for model, code in exit_codes.items()]
            lines.append("exit /b 0")
            path = folder / "stand_in_python.cmd"
            path.write_text("\r\n".join(lines) + "\r\n", encoding="ascii", newline="")
            return path
        lines = ["#!/bin/sh", 'if [ "$1" = "--version" ]; then exit 0; fi',
                 'echo "STAND-IN model=$EVAL_JUDGE_MODEL args=$*"']
        lines += [f'if [ "$EVAL_JUDGE_MODEL" = "{model}" ]; then exit {code}; fi'
                  for model, code in exit_codes.items()]
        lines.append("exit 0")
        path = folder / "stand_in_python.sh"
        path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="")
        path.chmod(0o755)
        return path

    def command(self, project: Path, arguments: tuple[str, ...]) -> list[str]:
        script = project / "scripts" / self.script.name
        if self.script.suffix == ".bat":
            return ["cmd", "/c", str(script), *arguments]
        return [_git_bash(), script.as_posix(), *arguments]

    def interpreter_value(self, path: Path) -> str:
        return str(path) if self.script.suffix == ".bat" else path.as_posix()

    @property
    def output_encoding(self) -> str:
        """cmd.exe writes in the console's code page; bash writes UTF-8."""
        return "oem" if self.script.suffix == ".bat" else "utf-8"


BAT = Launcher("run_evals.bat", SCRIPTS / "run_evals.bat", cannot_run=9009)
SH = Launcher("run_evals.sh", SCRIPTS / "run_evals.sh", cannot_run=127)


@pytest.fixture(params=[BAT, SH], ids=lambda launcher: launcher.name)
def launcher(request):
    unavailable = request.param.available()
    if unavailable:
        pytest.skip(unavailable)
    return request.param


@pytest.fixture
def project(tmp_path, launcher):
    """A project skeleton holding a copy of the launcher, so what it writes
    (a report, a temp folder) lands in the sandbox and not in the checkout."""
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(launcher.script, root / "scripts" / launcher.script.name)
    return root


def _run(launcher, project: Path, *arguments: str, exit_codes=None, ollama=NO_OLLAMA, **env_overrides):
    interpreter = launcher.stand_in(project, exit_codes or {})
    env = {k: v for k, v in os.environ.items() if not k.startswith("EVAL_")}
    env.update(PYTHON=launcher.interpreter_value(interpreter), EVAL_JUDGE_BASE_URL=ollama)
    env.update(env_overrides)
    result = subprocess.run(
        launcher.command(project, ("--no-report", *arguments)),
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        encoding=launcher.output_encoding,
        errors="replace",
        stdin=subprocess.DEVNULL,  # an error path ends in ``pause``: EOF lets it return
        timeout=120,
    )
    return result, result.stdout + result.stderr


def _models_run(output: str) -> list[str]:
    return [
        line.split("model=", 1)[1].split(" args=", 1)[0]
        for line in output.splitlines()
        if line.startswith("STAND-IN model=")
    ]


class TestTheExitCodeIsPytests:
    def test_a_run_that_passed_says_so_and_exits_zero(self, launcher, project):
        result, output = _run(launcher, project)

        assert result.returncode == 0, output
        assert "All evaluations passed" in output

    def test_a_run_with_failing_evals_says_some_failed(self, launcher, project):
        result, output = _run(launcher, project, exit_codes={DEFAULT_CHAT_MODEL: 1})

        assert result.returncode == 1, output
        assert "Some evaluations failed" in output
        assert "did not run" not in output

    @pytest.mark.parametrize(
        "code, reason",
        [
            (5, "collected no tests"),
            (4, "usage error"),
            (2, "exited 2 before reporting"),
        ],
        ids=["nothing-collected", "usage-error", "anything-else"],
    )
    def test_a_run_that_never_ran_is_not_reported_as_failing_evals(self, launcher, project, code, reason):
        result, output = _run(launcher, project, exit_codes={DEFAULT_CHAT_MODEL: code})

        assert result.returncode == code, output
        assert "Some evaluations failed" not in output
        assert "All evaluations passed" not in output
        assert "The suite did not run" in output
        assert reason in output
        assert "Nothing was measured" in output

    def test_an_interpreter_that_cannot_be_executed_is_not_a_failing_eval_either(self, launcher, project):
        missing = launcher.interpreter_value(project / "no_such_python")

        result, output = _run(launcher, project, PYTHON=missing)

        assert result.returncode == launcher.cannot_run, output
        assert "could not be executed" in output
        assert "Some evaluations failed" not in output
        assert _models_run(output) == []


class TestWhatAPlainRunMeasures:
    def test_a_single_run_measures_the_default_small_model(self, launcher, project):
        _, output = _run(launcher, project, "--single")

        assert _models_run(output) == [DEFAULT_CHAT_MODEL], output

    def test_a_run_with_no_ollama_to_compare_with_measures_the_same_model(self, launcher, project):
        _, output = _run(launcher, project)

        assert _models_run(output) == [DEFAULT_CHAT_MODEL], output

    def test_a_model_chosen_by_the_user_is_the_one_measured(self, launcher, project):
        _, output = _run(launcher, project, "--single", EVAL_JUDGE_MODEL="some-other:model")

        assert _models_run(output) == ["some-other:model"], output

    def test_a_run_repeats_each_eval_once_unless_asked_for_more(self, launcher, project):
        _, output = _run(launcher, project, "--single")
        assert "--count=1" in output

        _, output = _run(launcher, project, "--single", EVAL_REPEAT_COUNT="3")
        assert "--count=3" in output


@pytest.fixture
def stub_ollama():
    """A server on a loopback port that answers the two requests the launcher
    makes (is it up, unload the model) with success."""

    class Handler(BaseHTTPRequestHandler):
        def _ok(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_GET(self):
            self._ok()

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self._ok()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@pytest.fixture
def bat_project(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(BAT.script, root / "scripts" / BAT.script.name)
    return root


@pytest.mark.skipif(os.name != "nt", reason="cmd.exe runs this script")
@pytest.mark.skipif(shutil.which("curl") is None, reason="the launcher probes Ollama with curl")
class TestComparingTheTwoTiers:
    """Run through the .bat alone: the .sh waits two real seconds between tiers."""

    def test_both_tiers_are_measured_small_first(self, bat_project, stub_ollama):
        result, output = _run(BAT, bat_project, ollama=stub_ollama)

        assert result.returncode == 0, output
        assert _models_run(output) == [DEFAULT_CHAT_MODEL, LARGE_MODEL], output

    def test_a_single_run_measures_one_tier_even_when_ollama_is_there(self, bat_project, stub_ollama):
        _, output = _run(BAT, bat_project, "--single", ollama=stub_ollama)

        assert _models_run(output) == [DEFAULT_CHAT_MODEL], output

    def test_a_tier_that_did_not_run_is_not_forgotten_by_a_clean_one_after_it(self, bat_project, stub_ollama):
        result, output = _run(BAT, bat_project, ollama=stub_ollama, exit_codes={DEFAULT_CHAT_MODEL: 5})

        assert _models_run(output) == [DEFAULT_CHAT_MODEL, LARGE_MODEL], output
        assert result.returncode == 5, output
        assert "The suite did not run" in output
