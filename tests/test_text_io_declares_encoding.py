"""The codebase never lets the locale pick a text encoding for it.

Python resolves a missing ``encoding=`` against the locale, which is UTF-8 on
macOS and cp1252 on a French Windows. The same line then reads one file on one
machine and raises ``UnicodeDecodeError`` on another, and the failure surfaces
far from the call that caused it.

Emoji are everywhere here by design (``CLAUDE.md`` requires them in command
line output), and user paths carry accents, so the gap is reached in normal
use rather than in some corner case. This test walks the source and asks every
text-mode call to say which encoding it means.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Modes that carry text. Binary modes decode nothing, so they take no encoding.
TEXT_MODES = {"r", "w", "a", "r+", "w+", "a+", "rt", "wt", "at"}


def _mode(call: ast.Call, position: int):
    """The mode argument, whether it was passed by name or by position."""
    for keyword in call.keywords:
        if keyword.arg == "mode":
            return keyword.value.value if isinstance(keyword.value, ast.Constant) else None
    if len(call.args) > position and isinstance(call.args[position], ast.Constant):
        return call.args[position].value
    return None


def _offenders(source: str) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if any(keyword.arg == "encoding" for keyword in node.keywords):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            found.append((node.lineno, func.attr))
        elif isinstance(func, ast.Attribute) and func.attr in ("NamedTemporaryFile", "TemporaryFile"):
            # These default to ``w+b``, the opposite of ``open``: no mode means
            # binary, and binary takes no encoding.
            mode = _mode(node, 0)
            if mode in TEXT_MODES:
                found.append((node.lineno, func.attr))
        elif isinstance(func, ast.Name) and func.id == "open":
            # A bare ``open`` only. ``wave.open`` and friends are binary APIs
            # that take no encoding at all.
            mode = _mode(node, 1)
            if mode is None or mode in TEXT_MODES:
                found.append((node.lineno, "open"))
    return found


@pytest.mark.unit
@pytest.mark.parametrize("tree", ["src", "tests"])
def test_every_text_io_call_names_its_encoding(tree):
    reported = []
    for path in sorted((ROOT / tree).rglob("*.py")):
        for lineno, kind in _offenders(path.read_text(encoding="utf-8")):
            reported.append(f"{path.relative_to(ROOT)}:{lineno} {kind}()")

    assert not reported, (
        "these calls let the locale choose the encoding, so they behave "
        "differently on Windows than on macOS:\n  " + "\n  ".join(reported)
    )
