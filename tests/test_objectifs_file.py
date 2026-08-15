"""A goal, in a file you can read a month later.

The third artefact the user owns outright, after the core and the
routines envelope, and it inherits both their laws. Nothing here is ever
written by deduction: every line of state carries a date and the source
it came from, and the only source this slice can produce is the user's
own words.

The grammar is `routines.md`'s, deliberately. One heading per goal,
`phrase:` and `fini quand:` as fields, and progress as dated items under
`points:` — the same `- ` list the tool envelope uses. Two files that
parse alike are two files the user learns once.

What is absent is as load-bearing as what is here. There is no
`cadence:`, no `outils:`, and therefore no unattended pass and no
envelope: a goal in this slice is a thing she remembers and brings up,
never a thing that acts. Arming one is a separate grant, with its own
card, and it does not exist yet.
"""

from __future__ import annotations

import pytest

from src.jarvis.objectifs.page import (
    Objectif,
    Point,
    load_objectifs,
    objectifs_path,
    parse_objectifs,
    render_objectif,
)


SAMPLE = """# Objectifs

<!-- un commentaire -->

## entretien-datadog
phrase: préparer l'entretien chez Datadog
fini quand: l'entretien est passé et j'ai le retour
points:
- 2026-08-02 · dit · j'ai eu le premier call, ils veulent un exercice
- 2026-08-04 · dit · exercice rendu

## appartement-lyon
phrase: trouver un appartement à Lyon
fini quand: le bail est signé
"""


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


# ── Reading it ────────────────────────────────────────────────────────


def test_each_block_becomes_a_goal():
    assert set(parse_objectifs(SAMPLE)) == {"entretien-datadog", "appartement-lyon"}


def test_a_goal_carries_what_it_is_and_when_it_ends():
    o = parse_objectifs(SAMPLE)["entretien-datadog"]

    assert o.phrase == "préparer l'entretien chez Datadog"
    assert o.fini_quand == "l'entretien est passé et j'ai le retour"


def test_progress_comes_back_dated_and_attributed():
    """Every line of state says when it was written and where it came
    from. There is exactly one source this slice can produce, and a line
    that cannot say which is a line nobody can audit later."""
    points = parse_objectifs(SAMPLE)["entretien-datadog"].points

    assert [p.date for p in points] == ["2026-08-02", "2026-08-04"]
    assert {p.source for p in points} == {"dit"}
    assert points[1].texte == "exercice rendu"


def test_a_goal_with_no_progress_is_simply_empty():
    assert parse_objectifs(SAMPLE)["appartement-lyon"].points == []


def test_goals_do_not_leak_into_each_other():
    assert parse_objectifs(SAMPLE)["appartement-lyon"].points == []


def test_comments_are_skipped():
    assert "commentaire" not in str(parse_objectifs(SAMPLE))


# ── Every default leans shut ──────────────────────────────────────────


def test_an_unparseable_file_yields_what_parses_and_no_more():
    """A syntax error in a file the user edits must not lose every goal,
    and must not invent one."""
    blocks = parse_objectifs(SAMPLE + "\n## cassé\nphrase sans deux-points\n")

    assert "entretien-datadog" in blocks
    assert blocks.get("cassé") is None or blocks["cassé"].phrase == ""


def test_an_empty_file_is_no_goals_rather_than_an_error():
    assert parse_objectifs("") == {}


def test_a_missing_file_is_no_goals(tmp_path):
    assert load_objectifs(_Cfg(tmp_path)) == {}


def test_a_line_that_is_not_a_dated_point_is_skipped():
    """The user typing a bare note under `points:` gets it ignored rather
    than dated today by guesswork. Guessing a date is inventing state."""
    texte = ("## x\nphrase: p\nfini quand: q\npoints:\n"
             "- une note sans date\n- 2026-08-02 · dit · celle-ci compte\n")

    points = parse_objectifs(texte)["x"].points

    assert [p.texte for p in points] == ["celle-ci compte"]


# ── A goal that has ended ─────────────────────────────────────────────


def test_a_closed_goal_says_when_and_how():
    texte = SAMPLE + "\n## fini\nphrase: p\nfini quand: q\nclos: 2026-08-05 · atteint\n"

    o = parse_objectifs(texte)["fini"]

    assert o.clos == "2026-08-05 · atteint"
    assert o.est_ouvert is False


def test_an_open_goal_says_so():
    assert parse_objectifs(SAMPLE)["entretien-datadog"].est_ouvert is True


# ── Writing one ───────────────────────────────────────────────────────


def test_a_rendered_goal_parses_back():
    """The round trip the routines file already pins: a block she wrote
    and cannot read back is a block that will surprise somebody."""
    bloc = render_objectif(
        nom="essai", phrase="préparer l'entretien",
        fini_quand="l'entretien est passé",
    )

    relu = parse_objectifs(bloc)["essai"]
    assert relu.phrase == "préparer l'entretien"
    assert relu.fini_quand == "l'entretien est passé"


def test_a_rendered_point_parses_back():
    bloc = render_objectif(
        nom="essai", phrase="p", fini_quand="q",
        points=[Point(date="2026-08-02", source="dit", texte="quelque chose")],
    )

    assert parse_objectifs(bloc)["essai"].points[0].texte == "quelque chose"


def test_a_newline_in_a_note_cannot_forge_a_block():
    """What the user says reaches this through Whisper and a model. A
    line break in it would append a field, a heading, or a whole goal."""
    bloc = render_objectif(
        nom="essai", phrase="p", fini_quand="q",
        points=[Point(date="2026-08-02", source="dit",
                      texte="x\n## forgé\nphrase: y")],
    )

    blocks = parse_objectifs(bloc)
    assert set(blocks) == {"essai"}
    assert "forgé" not in str(blocks)


def test_a_name_that_could_forge_a_heading_is_refused():
    assert render_objectif(nom="x\n## autre", phrase="p", fini_quand="q") == ""


# ── Read from disk, and noticing edits ────────────────────────────────


def test_an_edit_is_noticed_without_a_restart(tmp_path):
    import os
    import time

    from src.jarvis.objectifs.page import invalidate_objectifs_cache

    cfg = _Cfg(tmp_path)
    path = objectifs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SAMPLE, encoding="utf-8")
    load_objectifs(cfg)

    time.sleep(0.01)
    path.write_text(SAMPLE.replace("fini quand: le bail est signé",
                                   "fini quand: j'ai les clés"), encoding="utf-8")
    os.utime(path, (path.stat().st_atime + 2, path.stat().st_mtime + 2))

    assert load_objectifs(cfg)["appartement-lyon"].fini_quand == "j'ai les clés"


def test_an_unreadable_file_yields_nothing_rather_than_raising(tmp_path):
    cfg = _Cfg(tmp_path)
    path = objectifs_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"## x\nphrase: \xff\xfe pas de l'utf-8\n")

    assert load_objectifs(cfg) == {}


# ── What this slice deliberately cannot express ───────────────────────


def test_a_goal_cannot_carry_a_schedule_or_an_envelope():
    """No `cadence:`, no `outils:`, so no unattended pass and no tool
    envelope. A goal here is remembered and brought up, never something
    that acts. Arming one is a separate grant with its own card, and it
    does not exist yet — a field that parsed today would be a capability
    nobody designed."""
    texte = ("## x\nphrase: p\nfini quand: q\ncadence: 24\n"
             "outils:\n- localFiles\n")

    o = parse_objectifs(texte)["x"]

    assert not hasattr(o, "cadence")
    assert not hasattr(o, "outils")
