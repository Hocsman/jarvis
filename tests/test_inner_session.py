"""A guard's end-to-end test reads a pytest session it ran apart.

``inner_session.py`` writes the session, runs it in a child process and reads
its junit report. These tests hold it to what its callers rely on: each test's
outcome by name, a clear failure when the session said nothing usable, and a
child that resolves no config file of the user's, whatever its conftest arms.
The child is told only the paths of the user's files, never what is in them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config_guard
import inner_session

REPORTED = """
import pytest


@pytest.fixture
def breaks_on_teardown():
    yield
    raise RuntimeError("the teardown broke")


def test_passes():
    pass


def test_fails():
    assert False, "the call failed"


def test_fails_then_breaks_on_teardown(breaks_on_teardown):
    assert False, "the call failed first"
"""

# The child's config loader records the file it would open rather than open
# it, so a child that did reach the user's home reads nothing of theirs.
RESOLVES = """
import os
from pathlib import Path

import config_guard
from outside import USERS_FILES, USERS_HOME


def test_the_loader_asks_only_for_files_under_a_home_of_its_own(monkeypatch):
    import jarvis.config

    asked = []
    monkeypatch.setattr(jarvis.config, "_load_json", lambda path: asked.append(path) or {})
    jarvis.config.load_config()

    home = config_guard.key(Path.home())
    assert home != USERS_HOME, "the child runs in the user's home"
    assert asked, "the loader asked for no file"
    for path in asked:
        assert config_guard.key(path).startswith(home + os.sep), path
        assert config_guard.key(path) not in USERS_FILES, path
"""


class TestWhatASessionReports:
    @pytest.fixture(scope="class")
    def inner(self, tmp_path_factory, _real_user_config):
        outside = (
            f"USERS_FILES = {sorted(_real_user_config)!r}\n"
            f"USERS_HOME = {config_guard.key(Path.home())!r}\n"
        )
        return inner_session.run(
            tmp_path_factory.mktemp("inner_session"),
            conftest="",
            tests=REPORTED + RESOLVES,
            modules={"outside.py": outside},
        )

    def test_a_test_that_passed_reads_empty(self, inner):
        assert inner["test_passes"] == ""

    def test_a_test_that_failed_reads_its_failure(self, inner):
        assert "the call failed" in inner["test_fails"]

    def test_a_test_that_failed_and_then_broke_on_teardown_reads_both(self, inner):
        outcome = inner["test_fails_then_breaks_on_teardown"]

        assert "the call failed first" in outcome, outcome
        assert "the teardown broke" in outcome, outcome

    def test_a_test_the_session_did_not_report_fails_naming_those_it_did(self, inner):
        with pytest.raises(pytest.fail.Exception, match="test_passes"):
            inner["test_it_never_had"]

    def test_the_child_resolves_no_config_file_of_the_users(self, inner):
        outcome = inner["test_the_loader_asks_only_for_files_under_a_home_of_its_own"]

        assert outcome == "", outcome


def test_a_session_that_wrote_no_report_fails_saying_what_it_printed(tmp_path):
    with pytest.raises(pytest.fail.Exception, match="the conftest broke on import"):
        inner_session.run(
            tmp_path,
            conftest='raise RuntimeError("the conftest broke on import")\n',
            tests="def test_never_runs():\n    pass\n",
        )


def test_a_session_that_runs_past_its_timeout_is_stopped_and_fails(tmp_path):
    sleeps = "import time\n\n\ndef test_sleeps():\n    time.sleep(600)\n"

    with pytest.raises(pytest.fail.Exception, match="ran past"):
        inner_session.run(tmp_path, conftest="", tests=sleeps, timeout=2)
