#!/usr/bin/env python3
"""Check that a PyInstaller build carries what the app reads at runtime.

PyInstaller bundles Python code on its own and nothing else: a data file the
app opens through ``Path(__file__)`` is absent from the build unless the spec
lists it, and the app then fails quietly (an empty window, a missing icon).
This script derives the runtime data files from the source tree, looks for each
one where the frozen app will look, and names every one that is absent.

Usage:
    python scripts/check_bundle_layout.py [--dist DIR] [--plain]

Exit status is 0 when the build is complete and 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Files that ship so the licence texts travel with the binaries. They sit at
# the project root and land at the top of the bundle's data tree.
LICENCE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.txt")

# Kinds of file the app opens at runtime. A file under src/ with a suffix in
# this set must reach the build.
RUNTIME_SUFFIXES = frozenset({
    ".html", ".htm", ".css", ".js", ".json", ".txt", ".csv",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp",
    ".ttf", ".otf", ".woff", ".woff2",
    ".wav", ".mp3", ".ogg",
})

# Kinds of file that never need to reach the build. Code is collected by
# PyInstaller's import analysis, ``.ico`` files are embedded in the executable
# by ``icon=`` at build time and never opened afterwards, and ``.spec.md`` /
# ``CLAUDE.md`` are documentation for developers.
NOT_DATA_SUFFIXES = frozenset({".py", ".pyc", ".pyo", ".ico"})
NOT_DATA_NAMES = frozenset({"CLAUDE.md"})
NOT_DATA_ENDINGS = (".spec.md",)


def _skipped(relative: PurePosixPath) -> bool:
    """Caches and hidden files are neither code nor data."""
    return any(part == "__pycache__" or part.startswith(".") for part in relative.parts)


def _is_runtime(relative: PurePosixPath) -> Optional[bool]:
    """True for a runtime data file, False for a file that never ships, None if unsure."""
    name = relative.name
    suffix = relative.suffix.lower()
    if name in NOT_DATA_NAMES or name.endswith(NOT_DATA_ENDINGS) or suffix in NOT_DATA_SUFFIXES:
        return False
    if suffix in RUNTIME_SUFFIXES:
        return True
    return None


def _walk(src_root: Path) -> Iterable[PurePosixPath]:
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(src_root).as_posix())
        if not _skipped(relative):
            yield relative


def runtime_data_files(src_root: Path) -> List[PurePosixPath]:
    """Data files under ``src_root`` the frozen app must find, relative to it."""
    return [rel for rel in _walk(src_root) if _is_runtime(rel) is True]


def unclassified_files(src_root: Path) -> List[PurePosixPath]:
    """Files under ``src_root`` that are neither known code/docs nor known data.

    Guessing either way is how a file goes missing from a build, so a file of
    an unknown kind is reported until its suffix is classified in this module.
    """
    return [rel for rel in _walk(src_root) if _is_runtime(rel) is None]


def executable_path(dist_dir: Path, platform: str) -> Path:
    """Where the build puts the program a user launches."""
    if platform == "darwin":
        return dist_dir / "Jarvis.app" / "Contents" / "MacOS" / "Jarvis"
    return dist_dir / "Jarvis" / ("Jarvis.exe" if platform == "win32" else "Jarvis")


def _data_roots(dist_dir: Path, platform: str) -> List[Path]:
    """Folders the frozen app resolves its data files from."""
    if platform == "darwin":
        contents = dist_dir / "Jarvis.app" / "Contents"
        return [contents / "Resources", contents / "Frameworks"]
    return [dist_dir / "Jarvis" / "_internal"]


def expected_bundle_files(project_root: Path) -> List[PurePosixPath]:
    """Every file the build must carry, as a path inside its data tree."""
    files = list(runtime_data_files(project_root / "src"))
    files.extend(PurePosixPath(name) for name in LICENCE_FILES)
    return files


def check_bundle(dist_dir: Path, project_root: Path, platform: Optional[str] = None) -> List[str]:
    """Return one line per thing missing from the build at ``dist_dir``."""
    platform = platform or sys.platform
    problems: List[str] = []

    if not dist_dir.is_dir():
        return [f"no build at {dist_dir} (dist folder not found)"]

    for rel in unclassified_files(project_root / "src"):
        problems.append(
            f"cannot tell whether src/{rel} ships: classify its suffix in scripts/check_bundle_layout.py"
        )

    exe = executable_path(dist_dir, platform)
    if not exe.is_file():
        problems.append(f"executable not found at {exe}")

    roots = _data_roots(dist_dir, platform)
    for rel in expected_bundle_files(project_root):
        if not any((root / rel).is_file() for root in roots):
            where = " or ".join(str(root) for root in roots)
            problems.append(f"missing {rel} under {where}")
    return problems


def _marks(plain: bool) -> dict:
    if plain:
        return {"check": "[..]", "ok": "[ok]", "bad": "[missing]", "indent": "    "}
    return {"check": "\U0001F50D", "ok": "✅", "bad": "❌", "indent": "    "}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check a PyInstaller build for missing data files.")
    parser.add_argument("--dist", default=None, help="PyInstaller output folder (default: <project>/dist)")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="project folder holding src/")
    parser.add_argument("--platform", default=sys.platform, help="platform layout to expect (default: this one)")
    parser.add_argument("--plain", action="store_true", help="ASCII output, for batch files")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root)
    dist_dir = Path(args.dist) if args.dist else project_root / "dist"
    marks = _marks(args.plain)

    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

    print(f"{marks['check']} Checking the build layout at {dist_dir}")
    problems = check_bundle(dist_dir, project_root, args.platform)
    if problems:
        for problem in problems:
            print(f"{marks['indent']}{marks['bad']} {problem}")
        return 1
    count = len(expected_bundle_files(project_root))
    print(f"{marks['indent']}{marks['ok']} Executable and {count} data files are in place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
