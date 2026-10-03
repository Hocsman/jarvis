"""The bundle layout check says what a PyInstaller build is missing.

``scripts/check_bundle_layout.py`` is what ``test_bundled_app.bat`` and
``test_bundled_app.sh`` run on a fresh build. It derives the data files
the app reads at runtime from the source tree (nothing is listed by
hand), looks for each of them where the frozen app will look, and names
every one that is absent. These tests build small fake projects and fake
builds, so they pin the mechanism and not today's file list.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_bundle_layout", ROOT / "scripts" / "check_bundle_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _project(tmp_path: Path) -> Path:
    """A small project: code, docs, build-only icons and three runtime files."""
    root = tmp_path / "project"
    src = root / "src"
    _touch(src / "pkg" / "__init__.py")
    _touch(src / "pkg" / "module.py")
    _touch(src / "pkg" / "pkg.spec.md")
    _touch(src / "pkg" / "CLAUDE.md")
    _touch(src / "pkg" / "__pycache__" / "module.cpython-311.pyc")
    _touch(src / "pkg" / "assets" / "icon.ico")
    _touch(src / "pkg" / ".DS_Store")
    _touch(src / "pkg" / "page" / "index.html", "<html></html>")
    _touch(src / "pkg" / "page" / "style.css", "body {}")
    _touch(src / "pkg" / "assets" / "logo.png", "png")
    return root


RUNTIME = {
    "pkg/page/index.html",
    "pkg/page/style.css",
    "pkg/assets/logo.png",
}


def _fake_build(root: Path, platform: str, *, data_dir: str | None = None) -> Path:
    """Lay out a build the way PyInstaller does for ``platform``."""
    dist = root / "dist"
    wanted = sorted(RUNTIME)
    if platform == "darwin":
        bundle = dist / "Jarvis.app" / "Contents"
        _touch(bundle / "MacOS" / "Jarvis")
        data = bundle / (data_dir or "Resources")
    else:
        exe = "Jarvis.exe" if platform == "win32" else "Jarvis"
        _touch(dist / "Jarvis" / exe)
        data = dist / "Jarvis" / (data_dir or "_internal")
    for rel in wanted:
        _touch(data / rel)
    return dist


class TestRuntimeDataFiles:
    def test_the_list_is_derived_from_the_tree(self, tmp_path):
        root = _project(tmp_path)
        found = {p.as_posix() for p in checker.runtime_data_files(root / "src")}
        assert found == RUNTIME

    def test_paths_are_relative_to_the_source_root_with_forward_slashes(self, tmp_path):
        root = _project(tmp_path)
        for path in checker.runtime_data_files(root / "src"):
            assert isinstance(path, PurePosixPath)
            assert not path.is_absolute()

    def test_code_docs_build_only_icons_and_caches_are_not_data(self, tmp_path):
        root = _project(tmp_path)
        found = {p.as_posix() for p in checker.runtime_data_files(root / "src")}
        for name in (
            "pkg/module.py",
            "pkg/pkg.spec.md",
            "pkg/CLAUDE.md",
            "pkg/assets/icon.ico",
            "pkg/.DS_Store",
        ):
            assert name not in found
        assert not any("__pycache__" in name for name in found)

    def test_a_file_of_an_unknown_kind_is_reported_instead_of_guessed(self, tmp_path):
        root = _project(tmp_path)
        _touch(root / "src" / "pkg" / "page" / "model.weights")
        assert [p.as_posix() for p in checker.unclassified_files(root / "src")] == [
            "pkg/page/model.weights"
        ]
        assert "pkg/page/model.weights" not in {
            p.as_posix() for p in checker.runtime_data_files(root / "src")
        }

    def test_a_clean_tree_has_nothing_unclassified(self, tmp_path):
        root = _project(tmp_path)
        assert checker.unclassified_files(root / "src") == []


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
class TestCheckBundle:
    def test_a_complete_build_has_no_problems(self, tmp_path, platform):
        root = _project(tmp_path)
        dist = _fake_build(root, platform)
        assert checker.check_bundle(dist, root, platform) == []

    @pytest.mark.parametrize("missing", sorted(RUNTIME))
    def test_a_missing_data_file_is_named(self, tmp_path, platform, missing):
        root = _project(tmp_path)
        dist = _fake_build(root, platform)
        for candidate in dist.rglob("*"):
            if candidate.is_file() and candidate.as_posix().endswith("/" + missing):
                candidate.unlink()
        problems = checker.check_bundle(dist, root, platform)
        assert len(problems) == 1
        assert missing in problems[0]

    def test_a_missing_executable_is_reported(self, tmp_path, platform):
        root = _project(tmp_path)
        dist = _fake_build(root, platform)
        for candidate in dist.rglob("Jarvis*"):
            if candidate.is_file():
                candidate.unlink()
        problems = checker.check_bundle(dist, root, platform)
        assert any("executable" in p.lower() for p in problems)

    def test_an_absent_dist_is_one_clear_problem(self, tmp_path, platform):
        root = _project(tmp_path)
        problems = checker.check_bundle(root / "dist", root, platform)
        assert len(problems) == 1
        assert "dist" in problems[0]

    def test_an_unclassified_source_file_is_a_problem(self, tmp_path, platform):
        root = _project(tmp_path)
        dist = _fake_build(root, platform)
        _touch(root / "src" / "pkg" / "page" / "model.weights")
        problems = checker.check_bundle(dist, root, platform)
        assert any("model.weights" in p for p in problems)


class TestWhereTheFrozenAppLooks:
    def test_windows_and_linux_data_lives_under_the_internal_folder(self, tmp_path):
        """A file next to the executable instead of under ``_internal`` is not found by the app."""
        for platform in ("win32", "linux"):
            root = _project(tmp_path / platform)
            dist = _fake_build(root, platform, data_dir=".")
            problems = checker.check_bundle(dist, root, platform)
            assert any("pkg/page/index.html" in p for p in problems)

    def test_macos_accepts_the_frameworks_folder_too(self, tmp_path):
        root = _project(tmp_path)
        dist = _fake_build(root, "darwin", data_dir="Frameworks")
        assert checker.check_bundle(dist, root, "darwin") == []


class TestCommandLine:
    def _args(self, root: Path, dist: Path, platform: str, *extra: str) -> list[str]:
        return [
            "--dist", str(dist),
            "--project-root", str(root),
            "--platform", platform,
            *extra,
        ]

    def test_exit_status_follows_the_verdict(self, tmp_path, capsys):
        root = _project(tmp_path)
        dist = _fake_build(root, "win32")
        assert checker.main(self._args(root, dist, "win32")) == 0
        next(dist.rglob("index.html")).unlink()
        assert checker.main(self._args(root, dist, "win32")) == 1
        assert "pkg/page/index.html" in capsys.readouterr().out

    def test_plain_output_is_ascii_for_a_batch_file_to_show(self, tmp_path, capsys):
        root = _project(tmp_path)
        dist = _fake_build(root, "win32")
        next(dist.rglob("index.html")).unlink()
        checker.main(self._args(root, dist, "win32", "--plain"))
        out = capsys.readouterr().out
        assert out and out.isascii()

    def test_default_output_leads_with_emoji(self, tmp_path, capsys):
        root = _project(tmp_path)
        dist = _fake_build(root, "win32")
        checker.main(self._args(root, dist, "win32"))
        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert lines and not lines[0].isascii()


class TestTheLocalBuildScripts:
    """``test_bundled_app`` builds, checks the layout, then launches: it must look where the build puts things."""

    BAT = ROOT / "scripts" / "test_bundled_app.bat"
    SH = ROOT / "scripts" / "test_bundled_app.sh"

    def test_the_batch_file_is_plain_ascii(self):
        """cmd.exe does not render Unicode, so a .bat carries no emoji."""
        assert self.BAT.read_bytes().isascii()

    def test_the_batch_file_checks_the_executable_the_windows_build_produces(self):
        exe = checker.executable_path(Path("dist"), "win32")
        assert str(exe).replace("/", "\\") in self.BAT.read_text(encoding="ascii")

    def test_the_shell_script_checks_the_executables_the_other_builds_produce(self):
        text = self.SH.read_text(encoding="utf-8")
        for platform in ("darwin", "linux"):
            assert checker.executable_path(Path("dist"), platform).as_posix() in text

    @pytest.mark.parametrize("script", ["BAT", "SH"])
    def test_both_run_the_checker_on_the_build(self, script):
        text = getattr(self, script).read_text(encoding="utf-8")
        assert "check_bundle_layout.py" in text
        assert (ROOT / "scripts" / "check_bundle_layout.py").is_file()

    def test_the_batch_file_asks_for_plain_output(self):
        assert "--plain" in self.BAT.read_text(encoding="ascii")
