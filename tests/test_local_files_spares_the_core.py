"""The files that decide what she may do are not files she may touch.

`yuba/` holds the six artefacts the user owns outright: what she believes
about him, the standing rules he gave her, what she is allowed to run
without asking, his goals, and what runs while he sleeps. Every one of
them is authority — read by the prompt, or by the gate, or by the
scheduler — and every one of them sits under `$HOME`, which `localFiles`
otherwise treats as its whole world.

The gate already classifies any non-read `localFiles` operation as
`destructif`, so writing there costs a click on a card that shows the
exact path. That is a real defence and it is not the one being added
here. This is the second: a path in a card is only a defence if the
person reading it recognises what it means, and
`~/.local/share/jarvis/yuba/appris.md` reads as an ordinary file unless
you happen to know that a line in it becomes something she believes.

So the directory is refused outright, whatever the risk vocabulary says
and whoever clicked. A tool that can rewrite the file listing its own
permissions is a tool whose permissions are advisory.

Reading is refused too, and for a different reason: those files are
already in her prompt when they belong there. A `read` that pulls them
into a tool result puts them somewhere the routine envelope and the
digest were never designed to reason about.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _tool():
    from src.jarvis.tools.builtin.local_files import LocalFilesTool
    return LocalFilesTool()


def _ctx(cfg):
    from src.jarvis.tools.base import ToolContext
    return ToolContext(
        db=MagicMock(), cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=0, user_print=lambda *a, **k: None,
    )


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """A config whose core really does sit under the home `localFiles`
    is allowed to roam.

    Without the `expanduser` patch every call below would be refused for
    being outside `$HOME`, and the tests would pass green while testing
    nothing — the guard could be absent entirely.
    """
    import os

    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path) if p == "~" else p)
    c = MagicMock()
    c.db_path = str(tmp_path / "jarvis.db")
    return c


def test_the_fixture_actually_reaches_into_the_home(cfg, tmp_path):
    """The control. If this ever fails, every refusal below is a refusal
    about the home boundary and proves nothing about the core."""
    import os

    voisin = tmp_path / "voisin.txt"
    voisin.write_text("lisible\n", encoding="utf-8")
    assert str(_core_dir(cfg)).startswith(os.path.expanduser("~"))

    r = _tool().run({"operation": "read", "path": str(voisin)}, _ctx(cfg))
    assert r.success, "the fixture is not inside the permitted home"


def _core_dir(cfg):
    from src.jarvis.memory.core import MemoryCore
    d = MemoryCore.for_config(cfg).directory
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Every door into the directory ─────────────────────────────────────


@pytest.mark.parametrize("fichier", ["outils.md", "profil.md", "regles.md",
                                     "objectifs.md", "routines.md", "appris.md"])
def test_no_operation_reaches_a_core_file(cfg, fichier):
    """Named one by one rather than looped over a constant, so adding a
    seventh artefact without adding it here is a visible omission."""
    cible = _core_dir(cfg) / fichier
    cible.write_text("- 2026-08-04 · dit : quelque chose\n", encoding="utf-8")

    for operation in ("read", "write", "append", "delete"):
        r = _tool().run({"operation": operation, "path": str(cible),
                         "content": "x"}, _ctx(cfg))

        assert not r.success, f"{operation} reached {fichier}"

    assert cible.read_text(encoding="utf-8").startswith("- 2026-08-04")


def test_the_directory_itself_cannot_be_listed(cfg):
    """Listing is how you learn the names to attack."""
    _core_dir(cfg)

    r = _tool().run({"operation": "list", "path": str(_core_dir(cfg))}, _ctx(cfg))

    assert not r.success


def test_a_subdirectory_is_refused_too(cfg):
    """`yuba/journal/` holds what the routines wrote overnight."""
    sous = _core_dir(cfg) / "journal"
    sous.mkdir(parents=True, exist_ok=True)
    (sous / "2026-08-04.md").write_text("le résumé du matin\n", encoding="utf-8")

    r = _tool().run({"operation": "read", "path": str(sous / "2026-08-04.md")},
                    _ctx(cfg))

    assert not r.success


def test_a_relative_path_that_climbs_into_it_is_refused(cfg):
    """The guard resolves before it decides, so `..` buys nothing."""
    cible = _core_dir(cfg)
    detour = str(cible / "sous" / ".." / "outils.md")

    r = _tool().run({"operation": "read", "path": detour}, _ctx(cfg))

    assert not r.success


def test_the_refusal_names_the_reason_not_just_the_path(cfg):
    """A refusal that only echoes the path is indistinguishable from the
    home-boundary one, so the model retries it and the user cannot tell
    which rule stopped it."""
    r = _tool().run({"operation": "read", "path": str(_core_dir(cfg) / "profil.md")},
                    _ctx(cfg))

    message = (r.reply_text or "").lower()
    assert "yuba/" in message
    assert "possèdes" in message


# ── And the rest of the home is untouched ─────────────────────────────


def test_an_ordinary_file_still_works(cfg, tmp_path):
    """The guard is a hole in the permission, not a new permission."""
    ordinaire = tmp_path / "notes.txt"
    ordinaire.write_text("bonjour\n", encoding="utf-8")

    r = _tool().run({"operation": "read", "path": str(ordinaire)}, _ctx(cfg))

    assert r.success
    assert "bonjour" in (r.reply_text or "")


def test_a_lookalike_directory_is_not_refused(cfg, tmp_path):
    """`yuba-sauvegarde` is his, not hers. Matching on the name rather
    than on the resolved directory would eat it."""
    voisin = tmp_path / "yuba-sauvegarde"
    voisin.mkdir()
    (voisin / "profil.md").write_text("une copie\n", encoding="utf-8")

    r = _tool().run({"operation": "read", "path": str(voisin / "profil.md")},
                    _ctx(cfg))

    assert r.success
