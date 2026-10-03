"""The spec registry lists the specs that exist, in both instruction files.

`CLAUDE.md` tells every agent "always search for related spec files before
starting any work" and hands it a registry to search. A spec that is not in
the registry is a spec nobody is told to read, and a registry row that points
nowhere is a promise the repository does not keep. `AGENTS.md` carries the
same registry for the agents that read that file instead, so the two must say
the same thing or one of them is wrong.

These tests pin the mechanism rather than today's list: the specs are found by
walking the repository, so the next spec added without a row fails here.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

INSTRUCTION_FILES = ("CLAUDE.md", "AGENTS.md")

# Directories that hold someone else's files or a second checkout of this one.
_NOT_OURS = {"node_modules", "worktrees", "dist", "build", "__pycache__"}

# A registry row starts with the spec's repo-relative path in backticks.
_ROW = re.compile(r"^\|\s*`(?P<path>[^`]+\.spec\.md)`\s*\|")


def _specs_on_disk() -> set[str]:
    """Every `*.spec.md` in the repository, as forward-slash relative paths."""
    found: set[str] = set()
    for folder, subfolders, files in os.walk(REPO_ROOT):
        # Hidden folders cover `.git`, `.mamba_env` and agent worktrees under
        # `.claude`; pruning them keeps the walk off the interpreter's tree.
        subfolders[:] = [
            name for name in subfolders
            if not name.startswith(".") and name not in _NOT_OURS
        ]
        for name in files:
            if name.endswith(".spec.md"):
                relative = Path(folder, name).relative_to(REPO_ROOT)
                found.add(relative.as_posix())
    return found


def _registry(instruction_file: str) -> dict[str, str]:
    """Registry rows of an instruction file: spec path -> the row's full text."""
    text = (REPO_ROOT / instruction_file).read_text(encoding="utf-8")
    rows: dict[str, str] = {}
    for line in text.splitlines():
        match = _ROW.match(line)
        if match:
            path = match.group("path")
            assert path not in rows, (
                f"{instruction_file} lists {path} twice; one row per spec.")
            rows[path] = line.strip()
    return rows


def test_the_walk_finds_the_specs():
    """The scan itself works, so the guards below cannot pass on an empty set."""
    on_disk = _specs_on_disk()

    assert on_disk, "no *.spec.md found: the repository walk is broken"
    assert all(not path.startswith(".") for path in on_disk)


@pytest.mark.parametrize("instruction_file", INSTRUCTION_FILES)
def test_every_spec_on_disk_has_a_row(instruction_file):
    registered = set(_registry(instruction_file))

    missing = sorted(_specs_on_disk() - registered)

    assert not missing, (
        f"{instruction_file} has no registry row for: {', '.join(missing)}. "
        f"Add one row per spec to its Spec File Registry.")


@pytest.mark.parametrize("instruction_file", INSTRUCTION_FILES)
def test_every_row_points_at_a_spec_that_exists(instruction_file):
    dangling = sorted(
        path for path in _registry(instruction_file)
        if not (REPO_ROOT / path).is_file()
    )

    assert not dangling, (
        f"{instruction_file} lists specs that do not exist: "
        f"{', '.join(dangling)}. Remove the row or fix the path.")


def test_the_two_registries_agree_row_for_row():
    claude = _registry("CLAUDE.md")
    agents = _registry("AGENTS.md")

    only_claude = sorted(set(claude) - set(agents))
    only_agents = sorted(set(agents) - set(claude))
    reworded = sorted(
        path for path in set(claude) & set(agents)
        if claude[path] != agents[path]
    )

    assert not (only_claude or only_agents or reworded), (
        "CLAUDE.md and AGENTS.md carry different registries. "
        f"Only in CLAUDE.md: {only_claude}. Only in AGENTS.md: {only_agents}. "
        f"Worded differently: {reworded}.")
