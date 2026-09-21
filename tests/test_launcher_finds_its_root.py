"""The Windows launcher finds the project from wherever it is started.

A desktop shortcut, a terminal in the user's home and a terminal in the
repository start the same script from different working directories, with
or without ``--voice-debug``. Its own location is the one thing all of them
share, so that is what the script has to locate the project from.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LANCEUR = ROOT / "scripts" / "run_desktop_app.bat"

pytestmark = pytest.mark.skipif(os.name != "nt", reason="cmd.exe runs this script")


@pytest.fixture
def projet_factice(tmp_path):
    """A project skeleton whose interpreter answers at once and does nothing.

    ``where.exe`` stands in for Python: it rejects every argument the script
    hands it, immediately and without side effects, so the script runs from
    start to finish without installing anything or opening a window.
    """
    projet = tmp_path / "projet"
    (projet / "scripts").mkdir(parents=True)
    shutil.copy(LANCEUR, projet / "scripts" / LANCEUR.name)
    env = projet / ".mamba_env"
    env.mkdir()
    shutil.copy(Path(os.environ["SystemRoot"]) / "System32" / "where.exe", env / "python.exe")
    return projet


@pytest.mark.integration
@pytest.mark.parametrize("arguments", [[], ["--voice-debug"]], ids=["sans-argument", "voice-debug"])
def test_the_launcher_finds_the_project_from_a_foreign_directory(projet_factice, tmp_path, arguments):
    ailleurs = tmp_path / "ailleurs"
    ailleurs.mkdir()

    resultat = subprocess.run(
        ["cmd", "/c", str(projet_factice / "scripts" / LANCEUR.name), *arguments],
        cwd=ailleurs,
        capture_output=True,
        text=True,
        encoding="oem",
        errors="replace",
        stdin=subprocess.DEVNULL,  # an error path ends in ``pause``: EOF lets it return
        timeout=120,
    )
    sortie = resultat.stdout + resultat.stderr

    assert "Mamba environment not found" not in sortie, sortie
    assert "Checking Python version" in sortie, sortie
