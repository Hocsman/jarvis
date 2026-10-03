"""A test leaves no thread running behind it.

A thread a test starts and forgets keeps running through every test that
follows: it takes locks, patches nothing back, calls a process-wide
``time.sleep`` or ``requests.get`` that a later test is counting, and prints
into someone else's captured output. The guard in ``thread_guard.py`` snapshots
the live threads before each test and, once every fixture of that test has
been torn down, fails the test that owns any thread that is still alive after
a short grace. These tests hold the guard to that: it fails a leak, it spares a
thread that winds down on its own, it names the test and the thread, and it
leaves alone what is not the test's to stop.
"""

import _thread
import os
import subprocess
import sys
import textwrap
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import thread_guard


TESTS_DIR = Path(__file__).resolve().parent
GRACE = 5.0


def _start_blocked(name: str):
    """A daemon thread that runs until its ``release`` is set."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, name=name, daemon=True)
    thread.start()
    return thread, release


class TestWhatTheGuardReports:
    def test_a_thread_started_after_the_snapshot_and_still_running_is_reported(self):
        before = thread_guard.snapshot()
        thread, release = _start_blocked("leaky-worker")
        try:
            survivors = thread_guard.new_survivors(before, grace=0.05)
            assert survivors == [thread]
        finally:
            release.set()
            thread.join(GRACE)

    def test_a_thread_that_winds_down_inside_the_grace_is_not_reported(self):
        before = thread_guard.snapshot()
        thread = threading.Thread(target=time.sleep, args=(0.05,), daemon=True)
        thread.start()
        try:
            assert thread_guard.new_survivors(before, grace=GRACE) == []
        finally:
            thread.join(GRACE)

    def test_a_thread_already_running_at_the_snapshot_is_not_reported(self):
        thread, release = _start_blocked("older-than-the-snapshot")
        try:
            before = thread_guard.snapshot()
            assert thread_guard.new_survivors(before, grace=0.05) == []
        finally:
            release.set()
            thread.join(GRACE)

    def test_every_survivor_is_reported_under_one_shared_grace(self):
        before = thread_guard.snapshot()
        started = [_start_blocked(f"leaky-{n}") for n in range(3)]
        try:
            survivors = thread_guard.new_survivors(before, grace=0.05)
            assert set(survivors) == {thread for thread, _ in started}
        finally:
            for thread, release in started:
                release.set()
                thread.join(GRACE)

    def test_the_report_names_the_test_and_each_thread(self):
        before = thread_guard.snapshot()
        thread, release = _start_blocked("leaky-worker")
        timer = threading.Timer(60, lambda: None)
        timer.name = "leaky-timer"
        timer.daemon = True
        timer.start()
        try:
            survivors = thread_guard.new_survivors(before, grace=0.05)
            text = thread_guard.report("tests/test_x.py::test_y", survivors)
        finally:
            release.set()
            timer.cancel()
            thread.join(GRACE)
            timer.join(GRACE)

        assert "tests/test_x.py::test_y" in text
        assert "leaky-worker" in text
        assert "leaky-timer" in text
        # A Timer has no target; what it will call is what identifies it.
        assert "<lambda>" in text
        assert "daemon" in text
        # What is already exempt is listed, so a thread is not added twice.
        for resident in thread_guard.RESIDENTS:
            assert resident.label in text


class TestWaitingForAWorker:
    """For code under test that starts a one-shot worker and keeps no handle
    to it, a test waits for the worker inside whatever patches it relies on."""

    def test_it_returns_once_the_worker_has_finished(self):
        before = thread_guard.snapshot()
        threading.Thread(target=time.sleep, args=(0.05,), daemon=True).start()
        thread_guard.wait_for_workers(before, timeout=GRACE)
        assert thread_guard.new_survivors(before, grace=0) == []

    def test_it_fails_naming_a_worker_that_never_finishes(self):
        before = thread_guard.snapshot()
        thread, release = _start_blocked("stuck-worker")
        try:
            with pytest.raises(pytest.fail.Exception, match="stuck-worker"):
                thread_guard.wait_for_workers(before, timeout=0.05)
        finally:
            release.set()
            thread.join(GRACE)


class TestWhatTheGuardLeavesAlone:
    def test_a_thread_python_registers_for_a_foreign_caller_is_a_resident(self):
        # Qt and PortAudio call into Python from threads Python did not
        # start; the first threading.current_thread() in one registers a
        # stand-in for it.
        registered = threading.Event()
        hold = threading.Event()

        def foreign():
            threading.current_thread()
            registered.set()
            hold.wait(GRACE)

        before = thread_guard.snapshot()
        _thread.start_new_thread(foreign, ())
        try:
            assert registered.wait(GRACE)
            stand_ins = [
                t for t in threading.enumerate()
                if t not in before and isinstance(t, threading._DummyThread)
            ]
            assert stand_ins, "the foreign thread did not register a stand-in"
            assert thread_guard.new_survivors(before, grace=0.05) == []
        finally:
            hold.set()

    def test_the_mcp_runtime_loop_is_resident_only_for_a_test_that_owns_its_shutdown(self):
        before = thread_guard.snapshot()
        thread, release = _start_blocked("JarvisMCPRuntime")
        try:
            owns = thread_guard.new_survivors(
                before, fixtures=frozenset({"shutdown_persistent_runtime"}), grace=0.05
            )
            does_not_own = thread_guard.new_survivors(
                before, fixtures=frozenset(), grace=0.05
            )
        finally:
            release.set()
            thread.join(GRACE)

        assert owns == []
        assert does_not_own == [thread]

    def test_a_thread_whose_entry_point_is_pytests_own_is_a_resident(self):
        release = threading.Event()

        def entry_point():
            release.wait(GRACE)

        # What identifies a thread pytest or one of its plugins started is
        # where its entry point is defined.
        entry_point.__module__ = "_pytest.some_helper"

        before = thread_guard.snapshot()
        thread = threading.Thread(target=entry_point, daemon=True)
        thread.start()
        try:
            assert thread_guard.new_survivors(before, grace=0.05) == []
        finally:
            release.set()
            thread.join(GRACE)


INNER_CONFTEST = """
import thread_guard

# A short grace keeps the inner session quick; the value is not what is under test.
thread_guard.JOIN_GRACE_SEC = 0.3

from thread_guard import _no_thread_outlives_its_test  # noqa: F401
"""

INNER_TESTS = """
import threading
import time

RELEASE = threading.Event()


def test_leaves_a_thread_running():
    threading.Thread(target=RELEASE.wait, name="leaky-worker", daemon=True).start()


def test_runs_right_after_the_leak_and_is_not_blamed_for_it():
    pass


def test_joins_its_own_thread():
    thread = threading.Thread(target=time.sleep, args=(0.01,), name="tidy-worker")
    thread.start()
    thread.join()


def test_leaves_a_thread_that_winds_down_inside_the_grace():
    threading.Thread(target=time.sleep, args=(0.05,), name="brief-worker", daemon=True).start()
"""


def _run_inner_session(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text(textwrap.dedent(INNER_CONFTEST), encoding="utf-8")
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(INNER_TESTS), encoding="utf-8")
    report = tmp_path / "report.xml"
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
         f"--junitxml={report}", "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    outcomes = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        failures = [child for child in case if child.tag in ("error", "failure")]
        outcomes[case.get("name")] = failures
    return outcomes


class TestTheGuardEndToEnd:
    @pytest.fixture(scope="class")
    def inner(self, tmp_path_factory):
        return _run_inner_session(tmp_path_factory.mktemp("inner_session"))

    def test_the_test_that_leaks_a_thread_fails(self, inner):
        assert inner["test_leaves_a_thread_running"], (
            "the session passed a test that left a thread running"
        )

    def test_the_failure_names_the_test_and_the_thread(self, inner):
        failure = inner["test_leaves_a_thread_running"][0]
        text = (failure.get("message") or "") + (failure.text or "")
        assert "test_inner.py::test_leaves_a_thread_running" in text
        assert "leaky-worker" in text

    def test_the_next_test_is_not_blamed_for_a_thread_it_did_not_start(self, inner):
        assert inner["test_runs_right_after_the_leak_and_is_not_blamed_for_it"] == []

    def test_a_test_that_joins_its_own_thread_passes(self, inner):
        assert inner["test_joins_its_own_thread"] == []

    def test_a_thread_that_winds_down_inside_the_grace_passes(self, inner):
        assert inner["test_leaves_a_thread_that_winds_down_inside_the_grace"] == []


def test_the_guard_is_armed_for_every_test_in_this_suite(request):
    assert "_no_thread_outlives_its_test" in request.fixturenames
