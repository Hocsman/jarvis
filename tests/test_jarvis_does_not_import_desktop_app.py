"""The core knows nothing of the desktop app.

``desktop_app.spec.md`` makes the dependency one-way: ``desktop_app`` imports
``jarvis``, never the reverse. That is what lets the assistant run headless
(no Qt in the process) and lets another UI sit on the same core.

A line in a specification does not hold a codebase to it, and an import
tucked inside a function, wrapped in ``try/except``, never fails at import
time: it fails silently, on the one path that needed it. So this test reads
the source of every module under ``src/jarvis`` and refuses any import of the
desktop package, wherever in the file it sits.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from conftest import ROOT

JARVIS_SRC = ROOT / "src" / "jarvis"

# The desktop package is reachable under two names in this repository: the
# runtime one, and the ``src.``-prefixed one the older tests use.
FORBIDDEN_ROOTS = ("desktop_app", "src.desktop_app")


def _is_desktop_module(dotted_name: str | None) -> bool:
    if not dotted_name:
        return False
    return any(
        dotted_name == root or dotted_name.startswith(root + ".")
        for root in FORBIDDEN_ROOTS
    )


def _dynamic_import_target(node: ast.Call) -> str | None:
    """The literal module name of ``importlib.import_module("x")`` or
    ``__import__("x")``, or None when the call is neither."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name not in ("import_module", "__import__") or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def desktop_imports(source: str) -> list[tuple[int, str]]:
    """Every (line, module) in ``source`` that reaches the desktop package.

    Walks the whole tree rather than the module's top level, so an import
    inside a function, a method, a ``try`` block or a conditional counts.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend(
                (node.lineno, alias.name)
                for alias in node.names
                if _is_desktop_module(alias.name)
            )
        elif isinstance(node, ast.ImportFrom):
            # A relative import (level > 0) stays inside its own package.
            if node.level == 0 and node.module:
                # ``from desktop_app.x import y`` names the package in the
                # module; ``from src import desktop_app`` names it in the
                # imported name.
                candidates = [node.module] + [
                    f"{node.module}.{alias.name}" for alias in node.names
                ]
                hit = next((c for c in candidates if _is_desktop_module(c)), None)
                if hit:
                    found.append((node.lineno, hit))
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if _is_desktop_module(target):
                found.append((node.lineno, target))
    return sorted(found)


# ---------------------------------------------------------------------------
# The scanner itself: it must see what it claims to see.
# ---------------------------------------------------------------------------


class TestTheScannerSeesEveryShapeOfImport:
    @pytest.mark.parametrize(
        "source",
        [
            "import desktop_app",
            "import desktop_app.face_widget as fw",
            "from desktop_app.face_widget import JarvisState",
            "from desktop_app import face_widget",
            "import src.desktop_app.app",
            "from src.desktop_app.app import main",
            "from src import desktop_app",
            "def f():\n    from desktop_app.face_widget import x\n",
            "class C:\n    def m(self):\n        try:\n            import desktop_app\n        except ImportError:\n            pass\n",
            "import importlib\nimportlib.import_module('desktop_app.app')\n",
            "__import__('desktop_app')\n",
        ],
    )
    def test_it_is_flagged(self, source):
        assert desktop_imports(source), f"missed an import of the desktop package in:\n{source}"

    @pytest.mark.parametrize(
        "source",
        [
            "import os\nfrom pathlib import Path\n",
            "from . import desktop_app_like\n",
            "from .. import desktop_app\n",
            "from jarvis.state import JarvisState\n",
            "import desktop_application_notes\n",
            "name = 'desktop_app'\n",
            "# from desktop_app import x\n",
            "import importlib\nimportlib.import_module('jarvis.state')\n",
        ],
    )
    def test_it_is_not_flagged(self, source):
        assert desktop_imports(source) == []


# ---------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------


def _jarvis_modules() -> list[Path]:
    return sorted(JARVIS_SRC.rglob("*.py"))


def test_the_scan_covers_the_whole_core():
    """A scan that finds nothing to read would pass for the wrong reason."""
    modules = _jarvis_modules()
    names = {path.name for path in modules}
    assert len(modules) > 50
    assert {"daemon.py", "listener.py", "tts.py", "engine.py"} <= names


def test_jarvis_does_not_import_desktop_app():
    offenders = []
    for path in _jarvis_modules():
        source = path.read_text(encoding="utf-8")
        for line, module in desktop_imports(source):
            offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line} imports {module}")

    assert not offenders, (
        "src/jarvis must not know src/desktop_app (desktop_app.spec.md). "
        "Whatever both sides need belongs to jarvis, and desktop_app imports it from there:\n  "
        + "\n  ".join(offenders)
    )
