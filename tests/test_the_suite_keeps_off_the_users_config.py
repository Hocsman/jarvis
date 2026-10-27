"""The suite keeps off the user's own config file.

``config.json`` holds the user's choices and, for a remote provider, their API
keys. A test that reads it puts their settings into whatever the test prints;
a test that saves over it changes the assistant they run every day.

The sandboxes in ``conftest.py`` give the run, and then every test, a config
file of its own. They have to reach the modules that did ``from jarvis.config
import default_config_path`` at import (the settings window, the setup
wizard, the app), which hold the function rather than the name, and the code
that runs outside a test's own fixtures: collection, wider fixtures, and the
pytest sessions those start in a child process. The guard in
``config_guard.py`` refuses the user's file to ``_load_json`` and
``_save_json`` and fails the test that asked. These tests hold both to that,
and never put the user's real file at stake: a test checks its path is not
one of the user's before it saves there, the guard's own tests use a stand-in
file, and its end-to-end tests run in a session whose home directory is a
temporary one.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import config_guard
from jarvis.config import _load_json as load_json_bound_at_import
from jarvis.config import _save_json as save_json_bound_at_import
from jarvis.config import default_config_path as default_config_path_bound_at_import

TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"


def test_the_guard_is_armed_for_every_test_in_this_suite(request):
    assert "_no_real_user_config" in request.fixturenames


class TestTheSandbox:
    """What a test gets when it resolves the config path."""

    def test_a_path_function_bound_at_import_resolves_outside_the_users_files(
        self, _real_user_config
    ):
        assert config_guard.key(default_config_path_bound_at_import()) not in _real_user_config

    def test_the_name_and_a_binding_made_at_import_name_one_file(self):
        import jarvis.config

        assert jarvis.config.default_config_path() == default_config_path_bound_at_import()

    def test_what_is_saved_through_a_binding_made_at_import_is_what_the_loader_reads(self):
        from jarvis.config import load_config

        path = default_config_path_bound_at_import()
        assert not config_guard.is_protected(path), path
        named = "a-model-only-this-test-names"
        assert save_json_bound_at_import(path, {"llm_chat_model": named})

        assert load_config()["llm_chat_model"] == named

    def test_a_tests_file_lies_in_the_sandbox_and_starts_absent(self, _bac_a_sable):
        path = default_config_path_bound_at_import()

        assert _bac_a_sable in path.parents
        assert not path.exists()


# What a child process started outside a test resolves, as the native guard's
# end-to-end session would: the path, then the file JARVIS_CONFIG_PATH names.
CHILD_RESOLVES = """
import os
from jarvis.config import default_config_path
print(default_config_path())
print(os.environ.get("JARVIS_CONFIG_PATH", ""))
"""


class TestOutsideATestsOwnFixtures:
    """A class-scoped fixture is set up before the test's own sandbox, so
    what it resolves, and what a child process it starts resolves, is the
    run's config home."""

    @pytest.fixture(scope="class")
    def resolved_by_a_class_fixture(self):
        import jarvis.config

        return jarvis.config.default_config_path()

    @pytest.fixture(scope="class")
    def resolved_by_its_child(self):
        env = {**os.environ, "PYTHONPATH": str(SRC_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
        child = subprocess.run(
            [sys.executable, "-c", CHILD_RESOLVES],
            env=env, capture_output=True, text=True, timeout=120,
        )
        assert child.returncode == 0, child.stderr[-2000:]
        path, named = child.stdout.splitlines()
        return path, named

    def test_a_class_fixture_resolves_outside_the_users_files(
        self, resolved_by_a_class_fixture, _real_user_config
    ):
        assert config_guard.key(resolved_by_a_class_fixture) not in _real_user_config

    def test_a_child_it_starts_resolves_outside_them_too(
        self, resolved_by_its_child, _real_user_config
    ):
        path, named = resolved_by_its_child

        assert config_guard.key(path) not in _real_user_config
        assert named, "the child would fall back to default_config_path()"
        assert config_guard.key(named) not in _real_user_config


class TestTheGuard:
    """The guard against a stand-in for the user's file, so a broken guard
    reads a temporary file rather than theirs."""

    @pytest.fixture
    def protected(self, tmp_path, monkeypatch):
        path = tmp_path / "home" / ".config" / "jarvis" / "config.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"llm_api_key": "a key the test must not see"}', encoding="utf-8")
        monkeypatch.setattr(
            config_guard, "_protected", config_guard._protected | {config_guard.key(path)}
        )
        return path

    def test_a_read_through_a_binding_made_at_import_is_refused(self, protected):
        with pytest.raises(config_guard.RealUserConfigRefused):
            load_json_bound_at_import(protected)

        assert config_guard.take_requests() == [f"_load_json({protected})"]

    def test_a_save_through_a_binding_made_at_import_writes_nothing(self, protected):
        before = protected.read_bytes()

        with pytest.raises(config_guard.RealUserConfigRefused):
            save_json_bound_at_import(protected, {"llm_api_key": "written by a test"})

        assert config_guard.take_requests() == [f"_save_json({protected})"]
        assert protected.read_bytes() == before

    def test_the_loader_is_refused_through_the_config_modules_own_names(
        self, protected, monkeypatch
    ):
        from jarvis.config import load_config

        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(protected))

        with pytest.raises(config_guard.RealUserConfigRefused):
            load_config()

        assert config_guard.take_requests() == [f"_load_json({protected})"]

    def test_the_config_module_imported_under_its_other_name_is_guarded_too(self, protected):
        twin = importlib.import_module("src.jarvis.config")

        with pytest.raises(config_guard.RealUserConfigRefused):
            twin._load_json(protected)

        assert config_guard.take_requests() == [f"_load_json({protected})"]

    def test_the_same_file_spelt_another_way_is_refused(self, protected):
        detour = protected.parent / ".." / "jarvis" / protected.name

        with pytest.raises(config_guard.RealUserConfigRefused):
            load_json_bound_at_import(detour)

        assert len(config_guard.take_requests()) == 1

    def test_a_file_of_the_tests_own_is_left_alone(self, protected, tmp_path):
        own = tmp_path / "own.json"

        assert save_json_bound_at_import(own, {"tts_enabled": False})
        assert load_json_bound_at_import(own) == {"tts_enabled": False}
        assert config_guard.take_requests() == []


class TestTheFilesItProtects:
    def test_the_file_the_app_would_open_from_this_home(self, _real_user_config):
        expected = Path.home() / ".config" / "jarvis" / "config.json"

        assert config_guard.key(expected) in _real_user_config

    def test_the_file_under_the_config_home_the_environment_names(self, tmp_path, monkeypatch):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config-home"))

        expected = tmp_path / "config-home" / "jarvis" / "config.json"
        assert config_guard.key(expected) in config_guard.real_user_config_paths()

    def test_the_file_jarvis_config_path_names(self, tmp_path, monkeypatch):
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "named.json"))

        expected = tmp_path / "named.json"
        assert config_guard.key(expected) in config_guard.real_user_config_paths()


# A module of the app's kind: it bound the loader and the path at import.
INNER_MODULE = """
from jarvis.config import _load_json, _save_json, default_config_path


def read():
    return _load_json(default_config_path())


def save(values):
    return _save_json(default_config_path(), values)
"""

INNER_CONFTEST = """
from config_guard import _no_real_user_config, _real_user_config  # noqa: F401
"""

INNER_TESTS = """
import bound_at_import


def test_a_read_whose_refusal_the_caller_swallows():
    try:
        seen = bound_at_import.read()
    except Exception:
        seen = None
    assert seen is None


def test_a_save_whose_refusal_the_caller_swallows():
    try:
        bound_at_import.save({"llm_api_key": "written by a test"})
    except Exception:
        pass


def test_a_file_of_its_own_is_left_alone(tmp_path):
    from jarvis.config import _load_json, _save_json

    _save_json(tmp_path / "own.json", {"tts_enabled": False})
    assert _load_json(tmp_path / "own.json") == {"tts_enabled": False}
"""

USERS_OWN = '{"llm_api_key": "a key the inner session must not see"}'


def _run_inner_session(tmp_path: Path):
    home = tmp_path / "home"
    users_file = home / ".config" / "jarvis" / "config.json"
    users_file.parent.mkdir(parents=True)
    users_file.write_text(USERS_OWN, encoding="utf-8")

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (session_dir / "conftest.py").write_text(textwrap.dedent(INNER_CONFTEST), encoding="utf-8")
    (session_dir / "bound_at_import.py").write_text(textwrap.dedent(INNER_MODULE), encoding="utf-8")
    (session_dir / "test_inner.py").write_text(textwrap.dedent(INNER_TESTS), encoding="utf-8")
    report = tmp_path / "report.xml"

    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("XDG_CONFIG_HOME", "JARVIS_CONFIG_PATH")
    }
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONPATH=os.pathsep.join([str(TESTS_DIR), str(SRC_DIR)]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    session = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         f"--junitxml={report}", "test_inner.py"],
        cwd=session_dir, env=env, capture_output=True, text=True, timeout=180,
    )
    # A session that could not start writes no report. Say what the process
    # said instead of failing later on a missing file.
    assert report.exists(), (
        f"the inner pytest session wrote no report (exit code {session.returncode})\n"
        f"--- stdout ---\n{session.stdout}\n--- stderr ---\n{session.stderr}"
    )
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        failures = [child for child in case if child.tag in ("error", "failure")]
        outcomes[case.get("name")] = "\n".join(
            (failure.get("message") or "") + (failure.text or "") for failure in failures
        )
    return outcomes, users_file


class TestTheGuardEndToEnd:
    """An inner session with the guard and no sandbox, whose home directory
    holds a stand-in for the user's file."""

    @pytest.fixture(scope="class")
    def inner(self, tmp_path_factory):
        return _run_inner_session(tmp_path_factory.mktemp("inner_session"))

    @staticmethod
    def _outcome(inner, name: str) -> str:
        outcomes, _ = inner
        assert name in outcomes, f"the inner session reported {sorted(outcomes)}, not {name}"
        return outcomes[name]

    def test_a_read_the_caller_swallows_still_fails_the_test(self, inner):
        outcome = self._outcome(inner, "test_a_read_whose_refusal_the_caller_swallows")

        assert "asked for the user's own config file" in outcome, outcome
        assert "_load_json(" in outcome, outcome

    def test_nothing_of_the_file_reaches_the_test_that_asked(self, inner):
        outcome = self._outcome(inner, "test_a_read_whose_refusal_the_caller_swallows")

        assert "must not see" not in outcome, outcome

    def test_a_save_the_caller_swallows_still_fails_the_test(self, inner):
        outcome = self._outcome(inner, "test_a_save_whose_refusal_the_caller_swallows")

        assert "_save_json(" in outcome, outcome

    def test_the_users_file_is_as_it_was(self, inner):
        _, users_file = inner

        assert users_file.read_text(encoding="utf-8") == USERS_OWN
        assert sorted(path.name for path in users_file.parent.iterdir()) == [users_file.name]

    def test_a_file_of_its_own_passes(self, inner):
        assert self._outcome(inner, "test_a_file_of_its_own_is_left_alone") == ""
