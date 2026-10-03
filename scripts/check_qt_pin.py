#!/usr/bin/env python3
"""Check that an install of Qt loads, one Qt module per child process.

On Windows, requirements.txt holds Qt at a release the suite is green on,
because later ones fail inside the DLL loader. This is the check that decides
whether a candidate set of Qt wheels may replace the pin (the procedure sits
beside the pin in requirements.txt). It imports the Qt modules the app uses,
prints the Qt and PyQt versions each one reports, and exits non-zero if any of
them fails.

Each module is imported in a child process of its own. A loader failure can
end the process that triggers it, and a check that died with it would say
nothing about the modules after it. The child's exit status and stderr are what
the verdict is read from: the Windows loader failure (status 0xc0000139,
STATUS_ENTRYPOINT_NOT_FOUND) is recognised by name and comes with what to try
next. Only the fixed phrases Python itself writes are matched, never the
operating system's wording, which follows the user's language.

Usage:
    python scripts/check_qt_pin.py
    python scripts/check_qt_pin.py --python C:\\scratch\\Scripts\\python.exe

Exit status: 0 when every module loads, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

# The Qt modules the app uses. QtSvg is in the list because the pin was
# chosen with it in mind; QtWebEngineWidgets because it ships in a wheel of
# its own that has to match the runtime wheels.
QT_MODULES = ("QtCore", "QtWidgets", "QtSvg", "QtWebEngineWidgets")

# STATUS_ENTRYPOINT_NOT_FOUND: the loader could not resolve a function a DLL
# imports. It is what Qt6Core.dll fails with on Windows past the pinned release.
LOADER_STATUS = 0xC0000139

OK = "ok"
LOADER = "loader"
NOT_INSTALLED = "not-installed"
MODULE_MISSING = "module-missing"
CRASHED = "crashed"
TIMED_OUT = "timed-out"
NOT_RUN = "not-run"
FAILED = "failed"

# What the child runs: import the module, then ask Qt for its own versions.
# qVersion() is the runtime Qt, the one the loader had to resolve; the
# QT_VERSION_STR constant is the Qt the bindings were built against, which is
# not what the pin is about.
CHILD_CODE = (
    "import importlib, sys\n"
    "importlib.import_module('PyQt6.' + sys.argv[1])\n"
    "from PyQt6.QtCore import PYQT_VERSION_STR, qVersion\n"
    "print('qt=' + qVersion() + ' pyqt=' + PYQT_VERSION_STR)\n"
)

_VERSIONS = re.compile(r"^qt=(\S+) pyqt=(\S+)$", re.MULTILINE)
_MISSING = re.compile(r"No module named '([^']+)'")
# The prefix of the ImportError Python raises when the loader refuses a DLL.
_DLL_LOAD_FAILED = "DLL load failed"

_LOADER_HINTS = (
    "Repeat the check in a scratch environment that holds only the Qt wheels, so "
    "no other DLL on the machine can be picked ahead of Qt's own. If it passes "
    "there, the fault is the environment and not the wheels.",
    "Install the newest Microsoft Visual C++ Redistributable (x64), and look for an "
    "older msvcp140.dll or vcruntime140.dll beside the interpreter or earlier on "
    "PATH: the loader may resolve Qt's imports against that copy instead.",
    "Try the previous patch release of the Qt runtime wheels (PyQt6-Qt6 and "
    "PyQt6-WebEngine-Qt6) under the same bindings. Keep the pin in requirements.txt "
    "until a set passes this check and the test suite.",
)


@dataclass(frozen=True)
class Verdict:
    """What one module's child showed."""

    module: str
    kind: str
    qt_version: str = ""
    pyqt_version: str = ""
    detail: str = ""
    hints: Tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.kind == OK


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def classify(module: str, returncode: int, stdout: str, stderr: str) -> Verdict:
    """The verdict on one child, from its exit status and what it printed."""
    evidence = _last_line(stderr)
    if returncode == 0:
        found = _VERSIONS.search(stdout)
        if found:
            return Verdict(module, OK, qt_version=found.group(1), pyqt_version=found.group(2))
        return Verdict(
            module, FAILED,
            detail="exited 0 without printing its versions, so nothing shows the import ran",
        )

    # Windows reports an NT status as an unsigned number, and some callers as
    # a signed one; masking makes them the same value. On POSIX a negative
    # status is a signal and masks to something no NT status is.
    status = returncode & 0xFFFFFFFF
    if status == LOADER_STATUS:
        return Verdict(
            module, LOADER,
            detail=(
                f"the Windows loader could not resolve an entry point in a Qt DLL "
                f"(status 0x{LOADER_STATUS:08x}, STATUS_ENTRYPOINT_NOT_FOUND)"
            ),
            hints=_LOADER_HINTS,
        )
    if _DLL_LOAD_FAILED in stderr:
        return Verdict(
            module, LOADER,
            detail=f"the Windows loader refused a Qt DLL: {evidence}",
            hints=_LOADER_HINTS,
        )

    missing = _MISSING.search(stderr)
    if missing and missing.group(1) == "PyQt6":
        return Verdict(
            module, NOT_INSTALLED,
            detail="PyQt6 is not installed for this interpreter",
            hints=(
                "Install the candidate set into this interpreter first (PyQt6, "
                "PyQt6-Qt6, PyQt6-WebEngine and PyQt6-WebEngine-Qt6, at the versions "
                "under test), or pass --python with the interpreter of the scratch "
                "environment that has it.",
            ),
        )
    if missing and missing.group(1).startswith("PyQt6."):
        return Verdict(
            module, MODULE_MISSING,
            detail=f"PyQt6 is installed but has no {module}: {evidence}",
            hints=(
                "A wheel that provides this module is missing. QtWebEngineWidgets "
                "comes from PyQt6-WebEngine (its runtime is PyQt6-WebEngine-Qt6); the "
                "others come from PyQt6 (runtime PyQt6-Qt6). Install them at "
                "versions that match each other.",
            ),
        )

    if returncode < 0 or status & 0xC0000000 == 0xC0000000:
        how = f"signal {-returncode}" if returncode < 0 else f"status 0x{status:08x}"
        return Verdict(
            module, CRASHED,
            detail=f"the import ended the child process instead of raising ({how})",
            hints=(
                f'Run python -c "import PyQt6.{module}" by hand: a crash dialog or '
                "a message on the console usually names the DLL.",
            ),
        )

    return Verdict(module, FAILED, detail=evidence or f"exited {returncode}")


def check_module(
    module: str,
    python: str = sys.executable,
    timeout: float = 60.0,
    run: Callable = subprocess.run,
) -> Verdict:
    """Import one Qt module in a child process of its own and judge the result."""
    argv = [python, "-c", CHILD_CODE, module]
    try:
        done = run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return Verdict(
            module, TIMED_OUT,
            detail=f"the import did not finish within {timeout:g}s",
            hints=(
                "A hung import is a failed one. Rerun with a larger --timeout first, "
                "to rule out a cold disk cache.",
            ),
        )
    except OSError as error:
        return Verdict(
            module, NOT_RUN,
            detail=f"could not start {python}: {error}",
            hints=("Check the path given to --python.",),
        )
    return classify(module, done.returncode, done.stdout or "", done.stderr or "")


def check_all(
    python: str = sys.executable,
    timeout: float = 60.0,
    run: Callable = subprocess.run,
) -> List[Verdict]:
    return [check_module(module, python, timeout, run) for module in QT_MODULES]


def render(verdicts: Sequence[Verdict], python: str) -> List[str]:
    """The report, one line per fact, hints indented under the failure."""
    lines = ["🔎 Qt pin check", f"   🐍 Interpreter: {python}", ""]
    width = max(len(v.module) for v in verdicts)
    for verdict in verdicts:
        if verdict.ok:
            lines.append(
                f"   ✅ {verdict.module:<{width}}  "
                f"Qt {verdict.qt_version}, PyQt {verdict.pyqt_version}"
            )
            continue
        lines.append(f"   ❌ {verdict.module:<{width}}  {verdict.detail}")
        lines.extend(f"      💡 {hint}" for hint in verdict.hints)
    failed = [v for v in verdicts if not v.ok]
    lines.append("")
    if failed:
        lines.append(
            f"❌ {len(failed)} of {len(verdicts)} Qt modules did not load. "
            "Keep the pin in requirements.txt until this passes."
        )
    else:
        lines.append(f"✅ All {len(verdicts)} Qt modules load.")
    return lines


def _utf8_console() -> None:
    """Emoji need a console that can carry them; the Windows default cannot."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Optional[Sequence[str]] = None, run: Callable = subprocess.run) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter whose Qt is checked (default: the one running this script)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="seconds each module's import may take (default: 60)",
    )
    args = parser.parse_args(argv)

    _utf8_console()
    verdicts = check_all(args.python, args.timeout, run)
    for line in render(verdicts, args.python):
        print(line)
    return 0 if all(v.ok for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
