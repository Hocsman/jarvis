"""A pytest session run apart, in a child process, for a guard's end-to-end test.

What a guard does to a whole session (fail a test at teardown, name it, spare
the one after) only shows from outside it. So a guard's end-to-end test writes
a small session (a ``pytest.ini``, a ``conftest.py`` that arms the guard, the
modules its tests import and a ``test_inner.py``), runs it with ``run()`` and
reads each test's outcome from the junit report the session writes.

The child gets a home directory of its own and none of the variables that
name a config file, so every config file it resolves lies under that home:
no inner session reaches the user's own ``config.json``, whatever its
conftest arms. A test that needs something in that home (a stand-in for the
user's file) makes the directory and passes it.

``tests/test_inner_session.py`` holds the runner to all of this.
"""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Optional

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"

# The variables that name a config file. The child resolves its own from its
# home instead.
CONFIG_VARIABLES = ("XDG_CONFIG_HOME", "JARVIS_CONFIG_PATH")

TIMEOUT_SEC = 180.0


class Outcomes(dict):
    """Each test's failure text by test name, empty for a test that passed.

    Asking for a test the session did not report fails the caller, naming
    the ones it did."""

    def __missing__(self, name: str) -> str:
        pytest.fail(f"the inner session reported {sorted(self)}, not {name}", pytrace=False)


def run(
    directory: Path,
    *,
    conftest: str,
    tests: str,
    modules: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
    timeout: float = TIMEOUT_SEC,
) -> Outcomes:
    """Run ``tests`` as ``test_inner.py`` under ``conftest``, inside ``directory``.

    ``modules`` maps a file name to the source of a module the tests import.
    ``home`` is the child's home directory, an empty one in ``directory``
    unless given. Sources are dedented before they are written.
    """
    session_dir = directory / "session"
    session_dir.mkdir()
    files = {"pytest.ini": "[pytest]\n", "conftest.py": conftest, **(modules or {})}
    files["test_inner.py"] = tests
    for name, source in files.items():
        (session_dir / name).write_text(textwrap.dedent(source), encoding="utf-8")
    if home is None:
        home = directory / "home"
        home.mkdir()
    report = directory / "report.xml"

    try:
        session = subprocess.run(
            [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
             f"--junitxml={report}", "test_inner.py"],
            cwd=session_dir, env=_environment(home),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as stopped:
        pytest.fail(
            f"the inner pytest session ran past {timeout} s and was stopped\n"
            + _what_it_said(stopped.stdout, stopped.stderr),
            pytrace=False,
        )
    # A session that could not start (a conftest that fails to import, a
    # crash while loading) writes no report. Say what the process said
    # instead of failing later on a missing file.
    if not report.exists():
        pytest.fail(
            f"the inner pytest session wrote no report (exit code {session.returncode})\n"
            + _what_it_said(session.stdout, session.stderr),
            pytrace=False,
        )
    return _outcomes(report)


def _environment(home: Path) -> Dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name not in CONFIG_VARIABLES}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONPATH=os.pathsep.join([str(TESTS_DIR), str(SRC_DIR)]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _what_it_said(stdout, stderr) -> str:
    def text(output) -> str:
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output or ""

    return f"--- stdout ---\n{text(stdout)}\n--- stderr ---\n{text(stderr)}"


def _outcomes(report: Path) -> Outcomes:
    outcomes = Outcomes()
    for case in ET.parse(report).getroot().iter("testcase"):
        failures = [child for child in case if child.tag in ("error", "failure")]
        text = "\n".join(
            (failure.get("message") or "") + (failure.text or "") for failure in failures
        )
        # A test that fails and then breaks on teardown is reported twice
        # under one name.
        name = case.get("name")
        outcomes[name] = "\n".join(part for part in (outcomes.get(name), text) if part)
    return outcomes
