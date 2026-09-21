"""Running the suite leaves the directory it was started from as it found it.

The test configuration's database is ``:memory:``, which has no file, and
``MemoryCore.for_config`` places the core beside the database. The parent of
``:memory:`` is ``.``, so the core would land in whichever directory the suite
runs from, usually the repository root, one ``git add -A`` away from
publishing a profile a test wrote.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# The suite imports modules under both names, and they are distinct objects.
MODULES = ("jarvis.memory.core", "src.jarvis.memory.core")


def _hors_du_repertoire_courant(chemin: Path) -> bool:
    ici = Path.cwd().resolve()
    cible = chemin.resolve()
    return cible != ici and ici not in cible.parents


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_the_test_config_keeps_its_core_out_of_the_working_directory(module, mock_config):
    core = importlib.import_module(module).MemoryCore.for_config(mock_config)

    assert core.directory.is_absolute(), core.directory
    assert _hors_du_repertoire_courant(core.directory), core.directory


@pytest.mark.unit
@pytest.mark.parametrize("module", MODULES)
def test_what_a_test_remembers_is_written_outside_the_working_directory(module, mock_config):
    """The directory is not enough: the file actually written is what leaks."""
    noyau = importlib.import_module(module)
    core = noyau.MemoryCore.for_config(mock_config)

    core.remember(noyau.SECTION_PROFILE, "écrit par un test")
    fichier = core.path_for(noyau.SECTION_PROFILE)

    assert fichier.exists(), fichier
    assert _hors_du_repertoire_courant(fichier), fichier
