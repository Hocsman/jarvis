"""The Qt pin check says which Qt module failed to load, and why.

On Windows, requirements.txt holds Qt at a release the suite is green on,
because later ones fail inside the loader (``Qt6Core.dll`` and
STATUS_ENTRYPOINT_NOT_FOUND, 0xc0000139). The pin can only be lifted by trying
a candidate set, and ``scripts/check_qt_pin.py`` is how: it imports each Qt
module the app uses in a child process of its own, because a loader failure
can end the process that triggers it, and one module failing must not hide the
verdict on the others.

The verdict is classified from what the child leaves behind: its exit status,
and its stderr, read only for the fixed phrases Python itself writes (the
operating system's own wording changes with the user's language). These tests
play the child with a fake that fails in each way, and run the real check once
against the Qt this suite runs on.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from scripts import check_qt_pin
from scripts.check_qt_pin import LOADER_STATUS, QT_MODULES, check_module, classify, main


def _child(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(["python"], returncode, stdout, stderr)


def _healthy(qt="6.9.2", pyqt="6.9.1"):
    return _child(0, f"qt={qt} pyqt={pyqt}\n")


class _FakeChildren:
    """Stands in for ``subprocess.run``: answers each module's child by name
    and records how it was started."""

    def __init__(self, answers=None, default=None):
        self.answers = answers or {}
        self.default = default or _healthy()
        self.started = []

    def __call__(self, argv, **kwargs):
        self.started.append((argv, kwargs))
        outcome = self.answers.get(argv[-1], self.default)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class TestOneChildPerModule:
    def test_every_module_is_imported_in_a_child_process_of_its_own(self):
        children = _FakeChildren()

        main(["--python", "scratch-python"], run=children)

        assert [argv[-1] for argv, _ in children.started] == list(QT_MODULES)
        assert {argv[0] for argv, _ in children.started} == {"scratch-python"}

    def test_the_modules_the_app_depends_on_are_the_ones_checked(self):
        assert set(QT_MODULES) == {"QtCore", "QtWidgets", "QtSvg", "QtWebEngineWidgets"}

    def test_a_failure_in_one_does_not_hide_the_verdict_on_the_others(self, capsys):
        children = _FakeChildren({"QtWidgets": _child(LOADER_STATUS)})

        code = main(["--python", "p"], run=children)
        out = capsys.readouterr().out

        assert code != 0
        assert len(children.started) == len(QT_MODULES)
        for module in QT_MODULES:
            assert module in out

    def test_a_child_is_given_a_limit_so_a_hung_import_cannot_hang_the_check(self):
        children = _FakeChildren()

        main(["--python", "p", "--timeout", "7"], run=children)

        assert all(kwargs.get("timeout") == 7 for _, kwargs in children.started)


class TestAHealthyQt:
    def test_every_module_loading_exits_zero_and_prints_both_versions(self, capsys):
        code = main(["--python", "p"], run=_FakeChildren(default=_healthy("6.9.2", "6.9.1")))
        out = capsys.readouterr().out

        assert code == 0
        assert "6.9.2" in out
        assert "6.9.1" in out

    def test_a_child_that_exits_zero_without_its_versions_is_not_a_pass(self):
        verdict = classify("QtCore", 0, "", "")

        assert not verdict.ok
        assert verdict.detail


class TestTheWindowsLoaderFailure:
    @pytest.mark.parametrize(
        "status",
        [LOADER_STATUS, LOADER_STATUS - 2**32],
        ids=["as-unsigned", "as-signed"],
    )
    def test_the_exit_status_is_recognised_however_the_platform_reports_it(self, status):
        verdict = classify("QtCore", status, "", "")

        assert verdict.kind == check_qt_pin.LOADER
        assert "0xc0000139" in verdict.detail.lower()

    def test_an_import_error_from_the_loader_is_recognised_in_any_language_of_the_os(self):
        stderr = (
            "Traceback (most recent call last):\n"
            '  File "<string>", line 1, in <module>\n'
            "ImportError: DLL load failed while importing QtCore: "
            "Le point d'entree de procedure est introuvable.\n"
        )

        verdict = classify("QtCore", 1, "", stderr)

        assert verdict.kind == check_qt_pin.LOADER

    def test_it_says_what_to_try_next(self, capsys):
        children = _FakeChildren({"QtCore": _child(LOADER_STATUS)})

        main(["--python", "p"], run=children)
        out = capsys.readouterr().out

        verdict = classify("QtCore", LOADER_STATUS, "", "")
        assert len(verdict.hints) >= 2
        for hint in verdict.hints:
            assert hint in out

    def test_a_crash_is_not_mistaken_for_it(self):
        windows_access_violation = classify("QtCore", 0xC0000005, "", "")
        posix_segfault = classify("QtCore", -11, "", "")

        assert windows_access_violation.kind == check_qt_pin.CRASHED
        assert posix_segfault.kind == check_qt_pin.CRASHED


class TestOtherFailures:
    def test_pyqt6_not_installed_at_all(self):
        stderr = "ModuleNotFoundError: No module named 'PyQt6'\n"

        verdict = classify("QtCore", 1, "", stderr)

        assert verdict.kind == check_qt_pin.NOT_INSTALLED
        assert verdict.hints

    def test_one_part_of_it_missing_is_told_apart_from_all_of_it(self):
        stderr = "ModuleNotFoundError: No module named 'PyQt6.QtWebEngineWidgets'\n"

        verdict = classify("QtWebEngineWidgets", 1, "", stderr)

        assert verdict.kind == check_qt_pin.MODULE_MISSING
        assert "QtWebEngineWidgets" in verdict.detail

    def test_a_child_that_never_answers_is_reported_as_timed_out(self):
        children = _FakeChildren({"QtSvg": subprocess.TimeoutExpired(["python"], 5)})

        verdict = check_module("QtSvg", python="p", timeout=5, run=children)

        assert verdict.kind == check_qt_pin.TIMED_OUT
        assert not verdict.ok

    def test_any_other_failure_shows_the_last_line_of_its_evidence(self):
        stderr = "Traceback (most recent call last):\n  ...\nRuntimeError: it broke\n"

        verdict = classify("QtCore", 1, "", stderr)

        assert verdict.kind == check_qt_pin.FAILED
        assert "RuntimeError: it broke" in verdict.detail

    def test_an_interpreter_that_cannot_be_started_is_a_failure_of_the_check_not_of_qt(self, capsys):
        def no_such_interpreter(argv, **kwargs):
            raise FileNotFoundError(argv[0])

        code = main(["--python", "no-such-python"], run=no_such_interpreter)
        out = capsys.readouterr().out

        assert code != 0
        assert "no-such-python" in out


try:
    import PyQt6.QtCore as _qt_core
except ImportError:  # pragma: no cover - the suite's own requirements include it
    _qt_core = None

# Another suite stands a mock in for a Qt module it cannot import, so a Qt
# that is not really there can still "import" here; only a real one has a
# version string.
_REAL_QT = _qt_core is not None and isinstance(getattr(_qt_core, "QT_VERSION_STR", None), str)


@pytest.mark.skipif(not _REAL_QT, reason="PyQt6 is not installed")
class TestAgainstTheRealQt:
    """One real child process per module. QtWebEngineWidgets is left out: it
    needs system libraries a headless runner may not have, and this suite
    does not own that."""

    @pytest.mark.parametrize("module", ["QtCore", "QtWidgets", "QtSvg"])
    def test_a_module_loads_and_reports_the_runtime_qt_and_the_bindings(self, module):
        verdict = check_module(module, python=sys.executable)

        assert verdict.ok, verdict.detail
        # The runtime Qt is what the loader resolved; QT_VERSION_STR is only
        # the Qt the bindings were built against, and need not be the same.
        assert verdict.qt_version == _qt_core.qVersion()
        assert verdict.pyqt_version == _qt_core.PYQT_VERSION_STR

    def test_a_module_the_install_does_not_have_is_named(self):
        verdict = check_module("QtNoSuchModule", python=sys.executable)

        assert verdict.kind == check_qt_pin.MODULE_MISSING
        assert "QtNoSuchModule" in verdict.detail
