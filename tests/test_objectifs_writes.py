"""Adding a line to a file that belongs to somebody else.

Two operations only: a dated point, and the line that closes a goal.
Both are inserts or single-line replacements inside one block's span, and
both inherit `rewrite_quand`'s discipline word for word, because the
hazard is identical — the user may have the file open in an editor, and
the value passing through arrives from a model reading a Whisper
transcription.

Never inside a comment: the file's own header explains the grammar using
the same words the fields use, and a parse-based check cannot see that
edit because a comment does not become a block. Compared byte for byte
rather than parse-alike, because two files can parse the same and differ
in a paragraph the user wrote.
"""

from __future__ import annotations

import os

import pytest

from src.jarvis.objectifs.page import (
    Point,
    append_point,
    close_objectif,
    objectifs_path,
    parse_objectifs,
)


FICHIER = """# Objectifs

<!--
  Un objectif est ce vers quoi tu travailles.
  phrase: ceci est une explication, pas un objectif
  clos: ceci non plus
-->

## entretien
phrase: préparer l'entretien
fini quand: l'entretien est passé
points:
- 2026-08-02 · dit · premier call

## appart
phrase: trouver un appartement
fini quand: le bail est signé
"""


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


@pytest.fixture
def cfg(tmp_path):
    config = _Cfg(tmp_path)
    path = objectifs_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FICHIER, encoding="utf-8")
    return config


def _texte(cfg):
    return objectifs_path(cfg).read_text(encoding="utf-8")


def _points(cfg, nom):
    return [p.texte for p in parse_objectifs(_texte(cfg))[nom].points]


# ── A dated line ──────────────────────────────────────────────────────


def test_a_point_lands_in_its_own_goal(cfg):
    assert append_point(cfg, "entretien",
                        Point("2026-08-04", "dit", "exercice rendu")) is True

    assert _points(cfg, "entretien") == ["premier call", "exercice rendu"]


def test_a_goal_with_no_points_yet_gains_the_list(cfg):
    assert append_point(cfg, "appart",
                        Point("2026-08-04", "dit", "visite jeudi")) is True

    assert _points(cfg, "appart") == ["visite jeudi"]


def test_nothing_else_in_the_file_moves(cfg):
    avant = _texte(cfg).splitlines(keepends=True)

    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    apres = _texte(cfg).splitlines(keepends=True)
    assert len(apres) == len(avant) + 1
    idx = next(i for i, (a, b) in enumerate(zip(avant, apres)) if a != b)
    assert avant[idx:] == apres[idx + 1:]


def test_another_goal_is_untouched(cfg):
    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    assert _points(cfg, "appart") == []


def test_a_goal_that_is_not_there_changes_nothing(cfg):
    avant = _texte(cfg)

    assert append_point(cfg, "jamais-écrit",
                        Point("2026-08-04", "dit", "x")) is False
    assert _texte(cfg) == avant


def test_a_note_that_could_forge_a_block_is_refused(cfg):
    avant = _texte(cfg)

    assert append_point(cfg, "entretien",
                        Point("2026-08-04", "dit", "x\n## forgé\nphrase: y")) is False
    assert _texte(cfg) == avant


def test_the_header_comment_is_not_a_goal(cfg):
    """It explains the grammar using the same words the fields use, and a
    parse-based check cannot see an edit there."""
    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    assert "ceci est une explication" in _texte(cfg)


# ── Closing one ───────────────────────────────────────────────────────


def test_closing_writes_the_line(cfg):
    assert close_objectif(cfg, "entretien", "2026-08-06 · atteint") is True

    o = parse_objectifs(_texte(cfg))["entretien"]
    assert o.clos == "2026-08-06 · atteint"
    assert o.est_ouvert is False


def test_closing_keeps_everything_it_recorded(cfg):
    close_objectif(cfg, "entretien", "2026-08-06 · atteint")

    assert _points(cfg, "entretien") == ["premier call"]


def test_closing_one_that_is_already_closed_changes_nothing(cfg):
    close_objectif(cfg, "entretien", "2026-08-06 · atteint")
    avant = _texte(cfg)

    assert close_objectif(cfg, "entretien", "2026-08-07 · autre") is False
    assert _texte(cfg) == avant


def test_a_closing_line_that_could_forge_a_block_is_refused(cfg):
    avant = _texte(cfg)

    assert close_objectif(cfg, "entretien", "x\n## forgé") is False
    assert _texte(cfg) == avant


# ── The file moved under it ───────────────────────────────────────────


def test_a_file_that_changed_since_it_was_read_is_left_alone(cfg, monkeypatch):
    """The user has objectifs.md open in an editor and it saves between
    the read and the write."""
    from src.jarvis.objectifs import page as module

    original = module._compose_point

    def _sabotage(*args, **kwargs):
        out = original(*args, **kwargs)
        chemin = objectifs_path(cfg)
        os.utime(chemin, (0, 0))
        chemin.write_text(_texte(cfg) + "\n## ajouté-à-la-main\n", encoding="utf-8")
        return out

    monkeypatch.setattr(module, "_compose_point", _sabotage)

    assert append_point(cfg, "entretien", Point("2026-08-04", "dit", "x")) is False
    assert "ajouté-à-la-main" in _texte(cfg)
    assert _points(cfg, "entretien") == ["premier call"]


def test_a_missing_file_is_a_refusal_not_an_exception(tmp_path):
    cfg = _Cfg(tmp_path)

    assert append_point(cfg, "x", Point("2026-08-04", "dit", "y")) is False


def test_the_next_read_sees_it(cfg):
    from src.jarvis.objectifs.page import load_objectifs

    load_objectifs(cfg)
    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    assert len(load_objectifs(cfg)["entretien"].points) == 2


def test_the_file_keeps_its_permissions(cfg):
    chemin = objectifs_path(cfg)
    os.chmod(chemin, 0o600)
    avant = os.stat(chemin).st_mode

    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    assert os.stat(chemin).st_mode == avant


def test_no_temporary_file_is_left_behind(cfg):
    append_point(cfg, "entretien", Point("2026-08-04", "dit", "x"))

    restes = [p.name for p in objectifs_path(cfg).parent.iterdir()
              if p.name != "objectifs.md"]
    assert restes == []
