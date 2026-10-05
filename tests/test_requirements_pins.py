"""One ONNX Runtime on every machine the project builds or develops on.

``onnxruntime`` arrives through ``faster-whisper`` and ``piper-tts``, whose
ranges are wide, so left free it resolves to a different release on the CI
runner, in the frozen build and on a developer machine. ``requirements.txt``
holds it with exact pins scoped by environment markers instead.

These tests read the file the way pip does (``packaging`` parses each line and
evaluates its marker) against explicit environments, so they can say what
every platform resolves to without installing anything. They pin the
mechanism, not the release: that exactly one line applies wherever the file is
installed, that it is an exact pin, and that it sits inside the range the
packages that need it declare. Which release the pin holds is the file's
business and moves without touching these tests.

What they cannot show is whether a wheel for the pinned release exists on a
given platform; that is a fact of the package index, read when the pin moves.
"""

from __future__ import annotations

import importlib.metadata as metadata
import itertools
import re
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = ROOT / "requirements.txt"

RUNTIME = "onnxruntime"

# The packages whose own metadata says which onnxruntime they accept.
NEEDS_THE_RUNTIME = ("faster-whisper", "piper-tts")

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Environments, spelled out the way the interpreter reports them
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "windows": {"os_name": "nt", "sys_platform": "win32", "platform_system": "Windows"},
    "linux": {"os_name": "posix", "sys_platform": "linux", "platform_system": "Linux"},
    "macos": {"os_name": "posix", "sys_platform": "darwin", "platform_system": "Darwin"},
    "freebsd": {"os_name": "posix", "sys_platform": "freebsd14", "platform_system": "FreeBSD"},
    "cygwin": {"os_name": "posix", "sys_platform": "cygwin", "platform_system": "CYGWIN_NT-10.0"},
}

_PYTHONS = {"3.11": "3.11.9", "3.12": "3.12.4"}


def _environment(system: str, machine: str, python: str) -> dict:
    full = _PYTHONS[python]
    return {
        **_SYSTEMS[system],
        "platform_machine": machine,
        "python_version": python,
        "python_full_version": full,
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
        "implementation_version": full,
        "platform_release": "",
        "platform_version": "",
        "extra": "",
    }


# Where the project builds or is developed, with the machine string Python
# reports there. These are the platforms the pins must answer for.
PUBLISHED = [
    ("Windows amd64", "windows", "AMD64"),
    ("Windows arm64", "windows", "ARM64"),
    ("Linux x86_64", "linux", "x86_64"),
    ("Linux aarch64", "linux", "aarch64"),
    ("macOS arm64", "macos", "arm64"),
    ("macOS x86_64", "macos", "x86_64"),
]

_PUBLISHED_CASES = [
    pytest.param(system, machine, python, id=f"{label}-py{python}")
    for (label, system, machine), python in itertools.product(PUBLISHED, _PYTHONS)
]

# Every system against every machine string, including ones the project does
# not build for: a marker pair that leaves a gap or overlaps shows up here.
_OTHER_MACHINES = ("x86", "i386", "i686", "armv7l", "ppc64le", "s390x", "riscv64", "loongarch64")
_GRID = [
    (system, machine, python)
    for system in _SYSTEMS
    for machine in sorted({m for _, _, m in PUBLISHED} | set(_OTHER_MACHINES))
    for python in _PYTHONS
]


# ---------------------------------------------------------------------------
# Reading requirements.txt
# ---------------------------------------------------------------------------

_COMMENT = re.compile(r"(^|\s)#.*$")


def _lines() -> list[str]:
    """The requirement lines of the file: comments and blank lines dropped."""
    out = []
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = _COMMENT.sub("", raw).strip()
        if line and not line.startswith("-"):
            out.append(line)
    return out


def _requirements() -> list[Requirement]:
    return [Requirement(line) for line in _lines()]


def _runtime_requirements(environment: dict) -> list[Requirement]:
    """The onnxruntime lines of requirements.txt that apply in ``environment``."""
    return [
        req
        for req in _requirements()
        if canonicalize_name(req.name) == RUNTIME
        and (req.marker is None or req.marker.evaluate(environment))
    ]


def _describe(requirements: list[Requirement]) -> str:
    return ", ".join(str(r) for r in requirements) or "none"


def _the_pin(system: str, machine: str, python: str) -> Requirement:
    """The one onnxruntime line that applies, failing plainly when it is not exactly one."""
    applying = _runtime_requirements(_environment(system, machine, python))
    assert len(applying) == 1, (
        f"{RUNTIME} lines applying on {system}/{machine}/Python {python}: {_describe(applying)}"
    )
    return applying[0]


# ---------------------------------------------------------------------------
# The file is something pip can read
# ---------------------------------------------------------------------------

class TestTheFileParses:
    def test_every_requirement_line_is_one_pip_can_read(self):
        unreadable = []
        for line in _lines():
            try:
                Requirement(line)
            except InvalidRequirement as exc:
                unreadable.append(f"{line!r}: {exc}")
        assert not unreadable

    def test_the_runtime_is_pinned_in_the_file_at_all(self):
        names = {canonicalize_name(r.name) for r in _requirements()}
        assert RUNTIME in names, f"requirements.txt leaves {RUNTIME} to whatever the other packages pull in"


# ---------------------------------------------------------------------------
# One line applies wherever the file is installed
# ---------------------------------------------------------------------------

class TestExactlyOnePinApplies:
    @pytest.mark.parametrize("system, machine, python", _PUBLISHED_CASES)
    def test_on_every_platform_the_project_builds_for(self, system, machine, python):
        _the_pin(system, machine, python)

    def test_on_any_platform_whatever_the_machine_string(self):
        wrong = {}
        for system, machine, python in _GRID:
            applying = _runtime_requirements(_environment(system, machine, python))
            if len(applying) != 1:
                wrong[f"{system}/{machine}/py{python}"] = _describe(applying)
        assert not wrong, f"the markers leave a gap or overlap: {wrong}"


# ---------------------------------------------------------------------------
# What applies is an exact pin
# ---------------------------------------------------------------------------

class TestThePinIsExact:
    @pytest.mark.parametrize("system, machine, python", _PUBLISHED_CASES)
    def test_it_names_one_release_and_nothing_wider(self, system, machine, python):
        req = _the_pin(system, machine, python)
        specifiers = list(req.specifier)
        assert len(specifiers) == 1, f"{req} is a range, not a pin"
        (spec,) = specifiers
        assert spec.operator == "==", f"{req} does not use =="
        assert "*" not in spec.version, f"{req} is a wildcard, not a release"
        Version(spec.version)
        assert not req.extras and req.url is None, f"{req} asks for more than the release"


# ---------------------------------------------------------------------------
# The pin is a release the packages that need it accept
# ---------------------------------------------------------------------------

def _declared_ranges(package: str, environment: dict) -> list[Requirement]:
    """What the installed ``package`` declares for onnxruntime in ``environment``."""
    declared = []
    for line in metadata.requires(package) or []:
        try:
            req = Requirement(line)
        except InvalidRequirement:
            continue
        if canonicalize_name(req.name) != RUNTIME:
            continue
        if req.marker is not None and not req.marker.evaluate(environment):
            continue
        declared.append(req)
    return declared


class TestThePinFitsWhatTheDependentsDeclare:
    @pytest.mark.parametrize("package", NEEDS_THE_RUNTIME)
    @pytest.mark.parametrize("system, machine, python", _PUBLISHED_CASES)
    def test_it_is_inside_the_range_the_package_declares(self, package, system, machine, python):
        try:
            metadata.distribution(package)
        except metadata.PackageNotFoundError:
            pytest.skip(f"{package} is not installed here, so its declared {RUNTIME} range cannot be read")

        environment = _environment(system, machine, python)
        declared = _declared_ranges(package, environment)
        if not declared:
            pytest.skip(f"the installed {package} declares no {RUNTIME} requirement for this environment")

        pinned = _the_pin(system, machine, python)
        (spec,) = list(pinned.specifier)
        version = Version(spec.version)
        refused = [str(d) for d in declared if not d.specifier.contains(version, prereleases=True)]
        assert not refused, (
            f"{pinned} on {system}/{machine}/Python {python} is outside what {package} "
            f"{metadata.version(package)} declares: {refused}"
        )
