"""The page of things she thinks she noticed.

A fourth artefact the user owns outright, beside his profile, his rules,
his tool policy, his goals and his routines, and it parses like them. She
writes candidates into it; he resolves each one with a single character.

What makes it safe is what it is not. Nothing in this file is a belief.
No prompt reads it, so a proposal sitting here for six months changes
nothing she says or does. It becomes a belief the moment he ticks it and
not one moment earlier — there is no timeout that accepts, no sweeper
that tidies, no "unless you object". The quiet state is already the safe
one.

The three states are the ones his other files already use: an untouched
box waits, a struck line is refused, and refusal is as durable as
acceptance because a proposal he struck out is never offered again. What
a tick harvests is the line *as it currently reads*, so he can fix a
clumsy sentence, or one that arrived in the wrong language, before
agreeing to be described by it.
"""

from __future__ import annotations

import pytest


SAMPLE = """# Appris

<!-- un commentaire -->

## Profil
- [ ] 2026-08-04 · journal : Il court le mardi matin avant le travail.
  > « the user mentioned running on Tuesday mornings before work »
- [x] 2026-08-02 · journal : Il a un chat qui s'appelle Miso.
  > « the user's cat Miso »
- ~~2026-07-30 · journal : Il déteste le café.~~
  > « the user said the coffee was bad »

## Règles
- [ ] 2026-08-03 · journal : Toujours répondre en français.
  > « the user asked the assistant to reply in French »
"""


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


def _page(cfg):
    from src.jarvis.appris.page import appris_path
    p = appris_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ── Reading it ────────────────────────────────────────────────────────


def test_each_proposal_is_read_with_its_state():
    from src.jarvis.appris.page import ETAT_ATTENTE, ETAT_COCHEE, ETAT_RAYEE, parse_appris

    etats = [p.etat for p in parse_appris(SAMPLE)]

    assert etats == [ETAT_ATTENTE, ETAT_COCHEE, ETAT_RAYEE, ETAT_ATTENTE]


def test_a_proposal_carries_its_section_date_and_text():
    from src.jarvis.appris.page import SECTION_PROFIL, parse_appris

    p = parse_appris(SAMPLE)[0]

    assert p.section == SECTION_PROFIL
    assert p.date == "2026-08-04"
    assert p.texte == "Il court le mardi matin avant le travail."


def test_the_quote_from_the_journal_comes_with_it():
    """He is being asked to agree to a sentence about himself. The line
    of his own diary it came from is what lets him check it rather than
    trust it."""
    assert "Tuesday mornings" in parse_appris_first().citation


def parse_appris_first():
    from src.jarvis.appris.page import parse_appris
    return parse_appris(SAMPLE)[0]


def test_the_rules_section_is_read_as_rules():
    from src.jarvis.appris.page import SECTION_REGLES, parse_appris

    assert parse_appris(SAMPLE)[3].section == SECTION_REGLES


def test_the_heading_is_matched_whatever_its_accents_and_case():
    """He types the heading himself when he reorganises the file."""
    from src.jarvis.appris.page import SECTION_REGLES, parse_appris

    for titre in ("## Règles", "## regles", "## REGLES", "## Regles"):
        texte = f"{titre}\n- [ ] 2026-08-03 · journal : Répondre en français.\n"
        assert parse_appris(texte)[0].section == SECTION_REGLES


def test_an_item_under_an_unknown_heading_has_no_section():
    """He is free to add `## Divers`. Nothing there can be harvested,
    because nothing says which of his files it would land in."""
    from src.jarvis.appris.page import parse_appris

    texte = "## Divers\n- [x] 2026-08-04 · journal : Quelque chose.\n"

    assert parse_appris(texte)[0].section is None


# ── Every default leans shut ──────────────────────────────────────────


def test_a_struck_line_is_refused_whatever_its_box_says():
    """He strikes a line out by hand and may well leave the `x` in it.
    Struck wins, exactly as the core reads its own files."""
    from src.jarvis.appris.page import ETAT_RAYEE, parse_appris

    texte = "## Profil\n- [x] ~~2026-08-04 · journal : Non.~~\n"

    assert parse_appris(texte)[0].etat == ETAT_RAYEE


def test_a_box_holding_something_else_is_not_a_tick():
    """`[?]`, `[-]`, `[o]` are him hesitating. Reading any of them as
    yes would be reading hesitation as consent."""
    from src.jarvis.appris.page import ETAT_COCHEE, parse_appris

    for marque in ("?", "-", "o", "X ", " "):
        texte = f"## Profil\n- [{marque}] 2026-08-04 · journal : Peut-être.\n"
        etat = parse_appris(texte)[0].etat
        assert etat != ETAT_COCHEE or marque.strip().lower() == "x"


def test_a_capital_X_is_a_tick():
    from src.jarvis.appris.page import ETAT_COCHEE, parse_appris

    texte = "## Profil\n- [X] 2026-08-04 · journal : Oui.\n"

    assert parse_appris(texte)[0].etat == ETAT_COCHEE


def test_an_empty_file_is_no_proposals():
    from src.jarvis.appris.page import parse_appris

    assert parse_appris("") == []


def test_a_missing_file_is_no_proposals(tmp_path):
    from src.jarvis.appris.page import load_appris

    assert load_appris(_Cfg(tmp_path)) == []


def test_an_unreadable_file_yields_nothing_rather_than_raising(tmp_path):
    from src.jarvis.appris.page import load_appris

    cfg = _Cfg(tmp_path)
    _page(cfg).write_bytes(b"## Profil\n- [ ] \xff\xfe pas de l'utf-8\n")

    assert load_appris(cfg) == []


def test_comments_are_skipped():
    from src.jarvis.appris.page import parse_appris

    assert "commentaire" not in str(parse_appris(SAMPLE))


def test_a_line_the_grammar_does_not_recognise_is_left_alone():
    """The file is his. A note he typed between two proposals is not a
    proposal, and must not become one."""
    from src.jarvis.appris.page import parse_appris

    texte = ("## Profil\n- une note à moi-même\n"
             "- [ ] 2026-08-04 · journal : Celle-ci compte.\n")

    props = parse_appris(texte)
    assert [p.texte for p in props] == ["Celle-ci compte."]


# ── Writing one ───────────────────────────────────────────────────────


def test_a_rendered_proposal_parses_back():
    from src.jarvis.appris.page import ETAT_ATTENTE, parse_appris, render_proposition

    bloc = render_proposition(date="2026-08-04",
                              texte="Il court le mardi matin.",
                              citation="the user mentioned running")

    p = parse_appris("## Profil\n" + bloc)[0]
    assert p.texte == "Il court le mardi matin."
    assert "running" in p.citation
    assert p.etat == ETAT_ATTENTE


def test_a_newline_in_a_proposal_cannot_forge_a_line():
    """What reaches this came out of a model reading his diary. A line
    break in it would forge a second proposal, a heading, or a tick."""
    from src.jarvis.appris.page import render_proposition

    assert render_proposition(date="2026-08-04",
                              texte="x\n- [x] 2026-08-04 · journal : forgé",
                              citation="c") == ""


def test_a_strikethrough_in_a_proposal_is_refused():
    """`~~` is the refusal mark. A proposal containing one would render
    as already refused, or worse, as half-refused."""
    from src.jarvis.appris.page import render_proposition

    assert render_proposition(date="2026-08-04", texte="~~x~~",
                              citation="c") == ""


def test_an_overlong_proposal_is_refused():
    """The profile block has a character budget and stops at the first
    oversized entry, so one runaway line would truncate everything he
    actually said."""
    from src.jarvis.appris.page import TEXTE_MAX, render_proposition

    assert render_proposition(date="2026-08-04", texte="x" * (TEXTE_MAX + 1),
                              citation="c") == ""


def test_a_proposal_with_no_date_is_refused():
    from src.jarvis.appris.page import render_proposition

    assert render_proposition(date="", texte="x", citation="c") == ""


# ── The guarded write ─────────────────────────────────────────────────


def test_appending_a_proposal_leaves_every_other_line_alone(tmp_path):
    """His own lines survive in order. A guarded write that reformatted
    the file would be a write that eats the paragraph he typed in it."""
    from src.jarvis.appris.page import ajouter_propositions, parse_appris

    cfg = _Cfg(tmp_path)
    depart = SAMPLE + "\nune note de fin, à moi\n"
    _page(cfg).write_text(depart, encoding="utf-8")

    assert ajouter_propositions(cfg, [("profil", "2026-08-05",
                                       "Il aime le thé.", "the user likes tea")])

    avant = depart.splitlines()
    apres = _page(cfg).read_text(encoding="utf-8").splitlines()

    # Every original line, still there and still in order.
    reste = iter(apres)
    assert all(any(a == b for b in reste) for a in avant), (
        f"a line of his was lost or reordered.\n{avant}\n{apres}")
    assert any(p.texte == "Il aime le thé." for p in parse_appris("\n".join(apres)))


def test_a_file_that_moved_underneath_is_not_written(tmp_path):
    """He may have it open in an editor. Composing against what we read
    and then writing over what he has since saved is how you lose his
    work, so the file is re-stat'd immediately before the replace.

    The sabotage happens inside `compose`, which is the only window that
    exists between the read and the write."""
    from src.jarvis.appris.page import _write_guarded

    cfg = _Cfg(tmp_path)
    _page(cfg).write_text(SAMPLE, encoding="utf-8")

    def compose_qui_sabote(lignes):
        _page(cfg).write_text(SAMPLE + "\nil a écrit pendant ce temps\n",
                              encoding="utf-8")
        return lignes + ["- [ ] 2026-08-05 · journal : Trop tard.\n"]

    assert not _write_guarded(cfg, compose_qui_sabote)
    assert "Trop tard" not in _page(cfg).read_text(encoding="utf-8")
    assert "il a écrit pendant ce temps" in _page(cfg).read_text(encoding="utf-8")


def test_the_file_is_created_with_its_header(tmp_path):
    from src.jarvis.appris.page import ajouter_propositions

    cfg = _Cfg(tmp_path)

    assert ajouter_propositions(cfg, [("profil", "2026-08-05", "Il aime le thé.", "tea")])

    texte = _page(cfg).read_text(encoding="utf-8")
    assert texte.startswith("# Appris")
    assert "## Profil" in texte and "## Règles" in texte
    assert "elle ne la sait pas" in texte


# ── Marking one as harvested ──────────────────────────────────────────


def test_a_harvested_proposal_is_struck_and_stamped(tmp_path):
    from src.jarvis.appris.page import ETAT_RAYEE, load_appris, marquer_retenue

    cfg = _Cfg(tmp_path)
    _page(cfg).write_text(SAMPLE, encoding="utf-8")
    cochee = [p for p in load_appris(cfg) if p.texte.startswith("Il a un chat")][0]

    assert marquer_retenue(cfg, cochee.ligne, "2026-08-11")

    apres = _page(cfg).read_text(encoding="utf-8")
    assert "retenu le 2026-08-11" in apres
    assert [p.etat for p in load_appris(cfg) if p.texte.startswith("Il a un chat")] == [ETAT_RAYEE]


def test_marking_touches_exactly_one_line(tmp_path):
    from src.jarvis.appris.page import load_appris, marquer_retenue

    cfg = _Cfg(tmp_path)
    _page(cfg).write_text(SAMPLE, encoding="utf-8")
    avant = _page(cfg).read_text(encoding="utf-8").splitlines()
    cochee = [p for p in load_appris(cfg) if p.texte.startswith("Il a un chat")][0]

    marquer_retenue(cfg, cochee.ligne, "2026-08-11")

    apres = _page(cfg).read_text(encoding="utf-8").splitlines()
    assert len(avant) == len(apres)
    assert sum(1 for a, b in zip(avant, apres) if a != b) == 1


def test_marking_a_line_that_is_gone_changes_nothing(tmp_path):
    """He may have deleted it by hand between the read and the write."""
    from src.jarvis.appris.page import marquer_retenue

    cfg = _Cfg(tmp_path)
    _page(cfg).write_text(SAMPLE, encoding="utf-8")

    assert not marquer_retenue(cfg, "- [x] une ligne qui n'existe pas", "2026-08-11")
    assert _page(cfg).read_text(encoding="utf-8") == SAMPLE


# ── Noticing his edits ────────────────────────────────────────────────


def test_an_edit_is_noticed_without_a_restart(tmp_path):
    import os
    import time

    from src.jarvis.appris.page import invalidate_appris_cache, load_appris

    cfg = _Cfg(tmp_path)
    p = _page(cfg)
    p.write_text(SAMPLE, encoding="utf-8")
    load_appris(cfg)

    time.sleep(0.01)
    p.write_text(SAMPLE.replace("Il a un chat qui s'appelle Miso.",
                                "Il a une chatte qui s'appelle Miso."), encoding="utf-8")
    os.utime(p, (p.stat().st_atime + 2, p.stat().st_mtime + 2))

    assert any("chatte" in x.texte for x in load_appris(cfg))


def test_what_he_rewrote_is_what_is_read_back(tmp_path):
    """The point of the file. He fixes a clumsy sentence, or one that
    arrived in English, and it is his version that will be harvested."""
    from src.jarvis.appris.page import load_appris

    cfg = _Cfg(tmp_path)
    _page(cfg).write_text(
        "## Profil\n- [x] 2026-08-04 · journal : Il court le mardi, tôt.\n",
        encoding="utf-8")

    assert load_appris(cfg)[0].texte == "Il court le mardi, tôt."
