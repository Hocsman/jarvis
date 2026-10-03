"""The viewer's script has to actually parse.

The whole UI is one Python string, served to the browser as one script
from the viewer's own origin. A single bad character anywhere in it, an
apostrophe that closes a JS string early or a stray brace, stops the
browser parsing the script at all. Not the statement: the script. Every
handler in the page silently ceases to exist, the tabs stop responding,
and nothing is logged where anyone looks.

This catches that before the page reaches a person, which no amount of
API-level testing can: the endpoints answer perfectly while the page
that calls them is dead.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


def _scripts():
    """Every script the served page runs, fetched the way a browser does."""
    from conftest import viewer_scripts
    from src.desktop_app.memory_viewer import app

    app.config["TESTING"] = True
    return viewer_scripts(app.test_client())


def test_the_page_has_a_script():
    """Guards the extraction itself: if the markup ever changes shape,
    this test failing is better than the syntax check quietly passing
    over an empty list."""
    scripts = _scripts()

    assert scripts
    assert all(script.strip() for script in scripts)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_script_parses():
    for i, script in enumerate(_scripts()):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(script)
            path = f.name
        try:
            result = subprocess.run(
                ["node", "--check", path], capture_output=True, text=True,
            )
        finally:
            Path(path).unlink(missing_ok=True)

        assert result.returncode == 0, (
            f"le script #{i} de la page ne parse pas :\n{result.stderr}"
        )


def test_no_python_escape_leaks_into_a_js_string():
    """The specific way it broke, kept as its own line of defence.

    The script is a plain triple-quoted string, so `\\'` written inside
    it reaches the browser as a bare `'` and closes whichever JS literal
    it sits in. Use double quotes on the JS side instead of escaping."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src" / "desktop_app" / "memory_viewer.py"
    ).read_text(encoding="utf-8")

    assert "\\'" not in source
