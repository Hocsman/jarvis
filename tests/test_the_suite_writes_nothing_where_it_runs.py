"""Running the suite leaves the directory it was started from as it found it,
and the user's own data as it found it.

The test configuration's database is ``:memory:``, which has no file, and
``MemoryCore.for_config`` places the core beside the database. The parent of
``:memory:`` is ``.``, so the core would land in whichever directory the suite
runs from, usually the repository root, one ``git add -A`` away from
publishing a profile a test wrote.

A configuration that takes the database's default path names the user's
data home, and the core beside it is the user's own ``profil.md``. A test
handed such a configuration gets a sandboxed core instead, so one missed
patch cannot read the user's life into a prompt or write a test's into it.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

# The suite imports modules under both names, and they are distinct objects.
MODULES = ("jarvis.memory.core", "src.jarvis.memory.core")

DEPOT = Path(__file__).resolve().parent.parent


def _hors_du_depot(chemin: Path) -> bool:
    """Outside the repository, which is what ``git add -A`` would publish.

    Compared with the repository rather than the working directory: the
    sandbox sits under the system temp folder, inside the user profile, so a
    run started from the profile would see the sandbox beneath its working
    directory while nothing reached the repository at all.
    """
    cible = chemin.resolve()
    return cible != DEPOT and DEPOT not in cible.parents


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_the_test_config_keeps_its_core_out_of_the_repository(module, mock_config):
    core = importlib.import_module(module).MemoryCore.for_config(mock_config)

    assert core.directory.is_absolute(), core.directory
    assert _hors_du_depot(core.directory), core.directory


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_what_a_test_remembers_is_written_outside_the_repository(module, mock_config):
    """The directory is not enough: the file actually written is what leaks."""
    noyau = importlib.import_module(module)
    core = noyau.MemoryCore.for_config(mock_config)

    core.remember(noyau.SECTION_PROFILE, "écrit par un test")
    fichier = core.path_for(noyau.SECTION_PROFILE)

    assert fichier.exists(), fichier
    assert _hors_du_depot(fichier), fichier


def _foyer() -> Path:
    """Where the user's own database lives when nothing pins a path."""
    from jarvis.config import _default_db_path

    return Path(_default_db_path()).expanduser().resolve().parent


def _hors_du_foyer(chemin: Path) -> bool:
    cible = chemin.resolve()
    foyer = _foyer()
    return cible != foyer and foyer not in cible.parents


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_a_config_pinned_to_the_users_data_home_is_kept_out_of_it(module):
    from jarvis.config import _default_db_path

    core = importlib.import_module(module).MemoryCore.for_config(
        SimpleNamespace(db_path=_default_db_path())
    )

    assert core.directory.is_absolute(), core.directory
    assert _hors_du_foyer(core.directory), core.directory
    assert _hors_du_depot(core.directory), core.directory


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_an_absolute_path_elsewhere_keeps_the_real_resolution(module, tmp_path):
    noyau = importlib.import_module(module)

    core = noyau.MemoryCore.for_config(SimpleNamespace(db_path=str(tmp_path / "jarvis.db")))

    assert core.directory == tmp_path / noyau.CORE_DIRNAME
