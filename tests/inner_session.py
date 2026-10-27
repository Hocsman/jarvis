"""A pytest session run apart, in a child process, for a guard's end-to-end test.

What a guard does to a whole session (fail a test at teardown, name it, spare
the one after) only shows from outside it. So a guard's end-to-end test hands
``run()`` the sources of a small session (a ``conftest.py`` that arms the
guard, the modules its tests import, the tests), and ``run()`` writes them
with a ``pytest.ini``, runs the session in a child process and reads each
test's outcome from the junit report it writes.

The child gets a home directory of its own, and the variables that name a
config file (``XDG_CONFIG_HOME``, ``JARVIS_CONFIG_PATH``) point inside it, so
every config file it resolves lies under that home: no inner session reaches
the user's own ``config.json``, whatever its conftest arms. The variables are
set rather than removed because ``load_settings()`` loads a ``.env`` found
above the app's code without overriding a variable already set, and a
developer's ``.env`` may name a config file. A test that needs something in
that home (a stand-in for the user's file) makes the directory and passes it.

``tests/test_inner_session.py`` holds the runner to all of this.
"""

import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"

TIMEOUT_SEC = 180.0


class Outcomes(dict):
    """Each test's failure text by test name, empty for a test that passed.

    A test that skipped reads as skipped, with its reason, so it never
    passes for one that ran. Asking for a test the session did not report
    fails the caller, naming the ones it did and showing what failed (a
    test file that could not be collected is reported under its own name)."""

    def __missing__(self, name: str) -> str:
        failed = "".join(f"\n--- {reported} ---\n{text}" for reported, text in self.items() if text)
        pytest.fail(
            f"the inner session reported {sorted(self)}, not {name}{failed}", pytrace=False
        )


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

    ``directory`` is a fresh, empty directory. ``modules`` maps a file name
    to the source of a module the tests import. ``home`` is the child's home
    directory, an empty one in ``directory`` unless given. Sources are
    dedented before they are written.
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
    config_home = home / ".config"
    env = dict(os.environ)
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(config_home),
        JARVIS_CONFIG_PATH=str(config_home / "jarvis" / "config.json"),
        PYTHONPATH=os.pathsep.join([str(TESTS_DIR), str(SRC_DIR)]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def _what_it_said(stdout: Optional[Union[str, bytes]], stderr: Optional[Union[str, bytes]]) -> str:
    # On POSIX a TimeoutExpired carries the raw bytes read so far, text=True
    # or not.
    def text(output: Optional[Union[str, bytes]]) -> str:
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output or ""

    return f"--- stdout ---\n{text(stdout)}\n--- stderr ---\n{text(stderr)}"


def _outcomes(report: Path) -> Outcomes:
    outcomes = Outcomes()
    for case in ET.parse(report).getroot().iter("testcase"):
        parts = []
        for child in case:
            if child.tag in ("error", "failure"):
                parts.append((child.get("message") or "") + (child.text or ""))
            elif child.tag == "skipped":
                parts.append(f"skipped: {child.get('message') or ''}\n{child.text or ''}")
        # A test that fails and then breaks on teardown is reported twice
        # under one name.
        name = case.get("name")
        outcomes[name] = "\n".join(part for part in (outcomes.get(name), *parts) if part)
    return outcomes
