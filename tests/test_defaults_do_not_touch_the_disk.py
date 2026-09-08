"""Asking where something lives must not create it.

``_default_db_path`` answers a question: which file would hold the database
if the user pinned none. Reading that answer happens in a dozen places that
want nothing more than the string, the settings parser and a menu item among
them, and the test suite calls it on every run.

Creating the directory as a side effect of the answer put a folder in the
user's real profile whenever any of that ran. The directory is made where it
is actually needed, by the code that opens the database and by the menu item
that reveals it in the file manager.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_resolving_the_default_db_path_creates_nothing(tmp_path, monkeypatch):
    """The path comes back; the disk stays as it was."""
    faux_home = tmp_path / "profil"
    faux_home.mkdir()
    # ``Path.home()`` reads USERPROFILE on Windows and HOME elsewhere.
    monkeypatch.setenv("USERPROFILE", str(faux_home))
    monkeypatch.setenv("HOME", str(faux_home))

    from jarvis.config import _default_db_path

    chemin = Path(_default_db_path())

    assert chemin.name == "jarvis.db"
    assert not chemin.parent.exists(), (
        f"resolving the default created {chemin.parent}"
    )
    assert list(faux_home.iterdir()) == [], "the home directory was touched"
