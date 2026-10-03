"""The PyInstaller spec ships every file the app reads at runtime.

PyInstaller follows imports and nothing else: a data file opened through
``Path(__file__)`` is absent from a frozen build unless ``jarvis_desktop.spec``
lists it. The dashboard page was missing from every packaged build that way,
and the primary window opened empty.

The spec is a Python script that PyInstaller executes, so it is executed here
too, with PyInstaller's own names stubbed, once per platform. The files that
must ship are derived from the source tree, so a file added tomorrow is held
to the same rule without anyone editing this test.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import sys
import types
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SPEC = ROOT / "jarvis_desktop.spec"
INSTALLER = ROOT / "installer" / "windows" / "jarvis_setup.iss"
PLATFORMS = ["win32", "darwin", "linux"]


def _load_module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their annotations through sys.modules
    spec.loader.exec_module(module)
    return module


layout = _load_module("check_bundle_layout")


class _Analysis:
    """Stands in for PyInstaller's ``Analysis``: keeps what the spec passed it."""

    def __init__(self, scripts, **kwargs):
        self.scripts = scripts
        self.kwargs = kwargs
        self.datas = list(kwargs.get("datas", []))
        self.binaries = list(kwargs.get("binaries", []))
        self.pure = []
        self.zipped_data = []
        self.zipfiles = []


class _Stage:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def load_spec(platform: str, scratch: Path) -> dict:
    """Run the spec as PyInstaller would from the project root, on ``platform``.

    The packages the spec probes for (piper, PyQt6) are replaced with empty
    stand-ins rooted in ``scratch``, so the result does not depend on what is
    installed and nothing is written outside ``scratch``.
    """
    namespace = {
        "__file__": str(SPEC),
        "Analysis": _Analysis,
        "PYZ": _Stage,
        "EXE": _Stage,
        "COLLECT": _Stage,
        "BUNDLE": _Stage,
    }
    stand_ins = {
        "PyInstaller": _module("PyInstaller"),
        "PyInstaller.utils": _module("PyInstaller.utils"),
        "PyInstaller.utils.hooks": _module(
            "PyInstaller.utils.hooks",
            collect_data_files=lambda *a, **k: [],
            collect_submodules=lambda *a, **k: [],
        ),
        "piper": _module("piper", __file__=str(scratch / "piper" / "__init__.py")),
        "PyQt6": _module("PyQt6", __file__=str(scratch / "PyQt6" / "__init__.py")),
    }
    source = compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec")
    with pytest.MonkeyPatch.context() as patch:
        for name, module in stand_ins.items():
            patch.setitem(sys.modules, name, module)
        patch.setattr(sys, "platform", platform)
        # The macOS branch writes qt.conf next to the spec and then edits the
        # finished bundle; the test must touch neither.
        patch.setattr(Path, "write_text", lambda self, data, *a, **k: len(data))
        patch.setattr(Path, "unlink", lambda self, *a, **k: None)
        patch.setattr(shutil, "copy2", lambda src, dst, *a, **k: dst)
        patch.chdir(ROOT)
        exec(source, namespace)
    return namespace


def resolve(entries, workdir: Path = ROOT):
    """Expand ``datas`` entries the way PyInstaller does.

    Returns ``(files, unresolved)``: ``files`` maps each destination path inside
    the bundle to the source file it comes from, ``unresolved`` lists the
    entries whose source matches nothing (PyInstaller aborts the build on those).
    """
    files: dict[str, Path] = {}
    unresolved: list[tuple[str, str]] = []
    for source, dest in entries:
        source_path = Path(os.path.normpath(Path(workdir) / source))
        if source_path.is_file():
            roots = [source_path]
            was_glob = False
        elif source_path.is_dir():
            roots = [source_path]
            was_glob = False
        else:
            roots = sorted(source_path.parent.glob(source_path.name))
            was_glob = True
        if not roots:
            unresolved.append((source, dest))
            continue
        for root in roots:
            if root.is_file():
                files[PurePosixPath(dest, root.name).as_posix()] = root
                continue
            base = root.parent if was_glob else root
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    relative = PurePosixPath(path.relative_to(base).as_posix())
                    files[PurePosixPath(dest, relative).as_posix()] = path
    return files, unresolved


def _in_repo_source(entries):
    """The entries that point at this repository's own files (not at site-packages)."""
    return [(s, d) for s, d in entries if Path(os.path.normpath(Path(ROOT) / s)).is_relative_to(ROOT)]


@pytest.fixture(params=PLATFORMS)
def datas(request, tmp_path):
    return load_spec(request.param, tmp_path)["datas"]


class TestRuntimeDataFilesShip:
    def test_every_data_file_the_app_reads_is_in_the_bundle(self, datas):
        shipped, _ = resolve(datas)
        missing = [
            f"src/{rel}"
            for rel in layout.runtime_data_files(SRC)
            if rel.as_posix() not in shipped
        ]
        assert not missing, f"jarvis_desktop.spec does not bundle: {missing}"

    def test_each_file_lands_where_its_package_looks_for_it(self, datas):
        """The app resolves data from ``Path(__file__).parent``, so the bundle mirrors ``src/``."""
        shipped, _ = resolve(datas)
        for rel in layout.runtime_data_files(SRC):
            assert shipped[rel.as_posix()] == SRC / rel

    def test_no_entry_pointing_into_the_repository_is_unresolvable(self, datas):
        """PyInstaller exits when a source matches nothing, which stops the build."""
        generated = {"qt.conf"}
        _, unresolved = resolve(datas)
        broken = [
            (s, d)
            for s, d in _in_repo_source(unresolved)
            if Path(s).name not in generated
        ]
        assert not broken

    def test_code_and_developer_documentation_stay_out_of_the_data(self, datas):
        """Listing a whole package folder as data would copy its modules and specs again."""
        shipped, _ = resolve(datas)
        for dest in shipped:
            name = PurePosixPath(dest).name
            assert not name.endswith((".py", ".spec.md"))
            assert name != "CLAUDE.md"

    def test_every_file_under_src_is_classified(self):
        """A file of an unknown kind is reported, so it cannot be skipped by accident."""
        assert layout.unclassified_files(SRC) == []


class TestLicenceTextsShip:
    """The project licence and the third-party inventory travel with every binary."""

    @pytest.mark.parametrize("name", layout.LICENCE_FILES)
    def test_the_file_exists_in_the_repository(self, name):
        assert (ROOT / name).is_file()

    @pytest.mark.parametrize("name", layout.LICENCE_FILES)
    def test_the_bundle_carries_it(self, datas, name):
        shipped, _ = resolve(datas)
        assert shipped.get(name) == ROOT / name

    def test_the_inventory_the_generator_writes_is_one_of_them(self):
        generator = _load_module("generate_third_party_notices")
        assert generator.NOTICES_NAME in layout.LICENCE_FILES


class TestInstallerShipsTheLicenceTexts:
    """Inno Setup copies ``dist\\Jarvis`` wholesale; the licence texts are also placed where a user looks."""

    @staticmethod
    def _section(header: str) -> list[str]:
        lines, inside = [], False
        for line in INSTALLER.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                inside = stripped.lower() == f"[{header.lower()}]"
                continue
            if inside and stripped and not stripped.startswith(";"):
                lines.append(stripped)
        return lines

    def _sources(self) -> list[Path]:
        found = []
        for line in self._section("Files"):
            match = re.match(r'Source:\s*"([^"]+)"', line)
            if match:
                found.append((INSTALLER.parent / match.group(1).replace("\\", "/")).resolve())
        return found

    @pytest.mark.parametrize("name", layout.LICENCE_FILES)
    def test_the_installer_copies_it_next_to_the_executable(self, name):
        assert (ROOT / name).resolve() in self._sources()
        destinations = [
            line for line in self._section("Files") if name in line and 'DestDir: "{app}"' in line
        ]
        assert destinations, f"{name} is not copied into {{app}}"

    def test_the_installer_shows_the_licence_before_installing(self):
        match = None
        for line in self._section("Setup"):
            match = match or re.match(r"LicenseFile=(.+)$", line)
        assert match, "the [Setup] section names no LicenseFile"
        shown = (INSTALLER.parent / match.group(1).strip().replace("\\", "/")).resolve()
        assert shown == (ROOT / "LICENSE").resolve()


class TestTheResolverMatchesPyInstaller:
    """The expansion above is only worth trusting if PyInstaller expands the same way."""

    def test_same_destinations_as_pyinstallers_own_expansion(self, tmp_path):
        utils = pytest.importorskip("PyInstaller.building.utils")
        entries = _in_repo_source(load_spec("win32", tmp_path)["datas"])
        entries = [(s, d) for s, d in entries if Path(s).name != "qt.conf"]
        ours, _ = resolve(entries)
        theirs = {
            Path(target).as_posix(): Path(source)
            for target, source in utils.format_binaries_and_datas(entries, workingdir=str(ROOT))
        }
        assert {k: Path(v) for k, v in ours.items()} == theirs
