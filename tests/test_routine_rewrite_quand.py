"""Changing one line of a file that belongs to somebody else.

`setRoutine` has only ever added bytes to `routines.md`. This is the one
exception, and it exists because the alternative is worse: a re-armed
routine carrying a newly spoken hour would leave `quand:` saying the old
one, and a file that names an hour nothing fires at is worse than a file
that was never written.

So the exception is drawn as narrowly as it can be. Only inside the
target block's own span, never inside a comment, every schedule line in
that span and not merely the last, and the result is compared to the
original **byte for byte** — every line identical except the ones
deliberately rewritten. Parsing the result and finding it equivalent is
not the same claim: two files can parse alike and differ in a paragraph
the user wrote.

And it does not run at all when the line already says the right thing,
which is the common case. Zero bytes, unchanged mtime, nothing to lose.
"""

from __future__ import annotations

import os

import pytest

from src.jarvis.routines.scope import (
    parse_routines,
    rewrite_quand,
    routines_path,
)


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


FICHIER = """# Routines

<!--
  Un commentaire d'en-tête.
  quand: ceci n'est pas un horaire, c'est une explication
-->

## matin
phrase: résume mes mails
quand: tous les jours à 07:00
outils:
- webSearch

## revue
phrase: fais le point
quand: chaque semaine, jour 0, à 09:00
outils:
- webSearch
"""


@pytest.fixture
def cfg(tmp_path):
    config = _Cfg(tmp_path)
    path = routines_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FICHIER, encoding="utf-8")
    return config


def _texte(cfg):
    return routines_path(cfg).read_text(encoding="utf-8")


# ── What it changes ───────────────────────────────────────────────────


def test_the_schedule_line_of_the_named_block_is_replaced(cfg):
    assert rewrite_quand(cfg, "matin", "tous les jours à 08:30") is True

    assert parse_routines(_texte(cfg))["matin"].quand == "tous les jours à 08:30"


def test_nothing_else_in_the_file_moves(cfg):
    avant = _texte(cfg).splitlines(keepends=True)

    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    apres = _texte(cfg).splitlines(keepends=True)
    assert len(avant) == len(apres)
    differentes = [i for i, (a, b) in enumerate(zip(avant, apres)) if a != b]
    assert len(differentes) == 1


def test_another_block_keeps_its_own_schedule(cfg):
    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    assert parse_routines(_texte(cfg))["revue"].quand == "chaque semaine, jour 0, à 09:00"


# ── What it refuses to touch ──────────────────────────────────────────


def test_a_schedule_line_inside_a_comment_is_not_a_schedule_line(cfg):
    """The header comment carries the word. Rewriting there would edit
    the file's own explanation of itself, and the dry run could not see
    it because a comment does not parse into a block."""
    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    assert "ceci n'est pas un horaire" in _texte(cfg)


def test_a_name_with_no_block_changes_nothing(cfg):
    avant = _texte(cfg)

    assert rewrite_quand(cfg, "jamais-écrite", "tous les jours à 08:30") is False
    assert _texte(cfg) == avant


def test_a_value_that_could_forge_a_line_is_refused(cfg):
    """The value is composed by code today, but a newline in it would
    append a field, a heading, or a whole block."""
    avant = _texte(cfg)

    assert rewrite_quand(cfg, "matin", "08:30\noutils:\n- localFiles") is False
    assert _texte(cfg) == avant


# ── Every schedule line in the span, not just one ─────────────────────


def test_a_block_carrying_two_schedule_lines_keeps_neither_stale(cfg, tmp_path):
    """`parse_routines` is last-wins, so replacing only the last would
    leave an earlier contradictory line above it — and that is the one a
    human reads first."""
    routines_path(cfg).write_text(
        "# Routines\n\n## matin\nquand: tous les jours à 07:00\n"
        "phrase: x\nquand: tous les jours à 07:00\noutils:\n- webSearch\n",
        encoding="utf-8",
    )

    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    assert "07:00" not in _texte(cfg)


def test_a_block_with_no_schedule_line_gains_one(cfg):
    routines_path(cfg).write_text(
        "# Routines\n\n## matin\nphrase: x\noutils:\n- webSearch\n",
        encoding="utf-8",
    )

    assert rewrite_quand(cfg, "matin", "tous les jours à 08:30") is True
    assert parse_routines(_texte(cfg))["matin"].quand == "tous les jours à 08:30"


# ── The file moved under it ───────────────────────────────────────────


def test_a_file_that_changed_since_it_was_read_is_left_alone(cfg, monkeypatch):
    """The user has routines.md open in an editor and it saves between
    the read and the write. Writing the composed text would erase
    whatever they had just typed."""
    from src.jarvis.routines import scope as module

    original = module._compose_quand

    def _sabotage(*args, **kwargs):
        out = original(*args, **kwargs)
        chemin = routines_path(cfg)
        os.utime(chemin, (0, 0))
        chemin.write_text(_texte(cfg) + "\n## ajoutée-à-la-main\n", encoding="utf-8")
        return out

    monkeypatch.setattr(module, "_compose_quand", _sabotage)

    assert rewrite_quand(cfg, "matin", "tous les jours à 08:30") is False
    assert "ajoutée-à-la-main" in _texte(cfg)
    assert parse_routines(_texte(cfg))["matin"].quand == "tous les jours à 07:00"


def test_an_unreadable_file_is_a_refusal_not_an_exception(tmp_path):
    cfg = _Cfg(tmp_path)

    assert rewrite_quand(cfg, "matin", "tous les jours à 08:30") is False


def test_the_next_read_sees_the_change(cfg):
    """The mtime cache cannot tell two writes inside one filesystem tick
    apart, so a re-read straight afterwards would see the file as it was
    a moment before."""
    from src.jarvis.routines.scope import load_routines

    load_routines(cfg)
    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    assert load_routines(cfg)["matin"].quand == "tous les jours à 08:30"


def test_the_file_keeps_its_permissions(cfg):
    chemin = routines_path(cfg)
    os.chmod(chemin, 0o600)
    avant = os.stat(chemin).st_mode

    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    assert os.stat(chemin).st_mode == avant


def test_no_temporary_file_is_left_behind(cfg):
    rewrite_quand(cfg, "matin", "tous les jours à 08:30")

    restes = [p.name for p in routines_path(cfg).parent.iterdir()
              if p.name != "routines.md"]
    assert restes == []
