"""A window is read once, so it has to be finished with before it counts.

`journal_lu` has no expiry: a row recorded as read is never offered
again. That makes every premature record a permanent, silent loss of
things she could have proposed and he will never learn existed — the
asymmetry this whole module is built around, since a duplicate costs him
one character to strike and a lost proposal costs him the proposal.

Three ways the shipped code retired a window it had not finished with,
all found after the first real run:

  the cap truncated the model's list and the rest were dropped, not
  deferred;

  the window was recorded before the proposals were written, so a write
  that failed — which the page's own docstring calls ordinary, "he may
  have this open in an editor" — retired the days anyway;

  every item failed on shape or grounding, which on a small model is the
  ordinary outcome, and the days were retired having learnt nothing.

Deferral has to be bounded or the same window is re-read for ever. It is
bounded by progress: what was kept is on the page, and the page
suppresses it next time. That advance is bounded in turn by a ceiling on
how many unanswered proposals may pile up before she stops reading at
all and says so.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")
        self.llm_chat_model = "test-model"
        self.appris_jours = 14
        self.appris_max_propositions = 3
        self.appris_seuil_doublon = 90
        self.appris_timeout_sec = 30.0


def _ctx(cfg):
    from src.jarvis.tools.base import ToolContext
    return ToolContext(db=MagicMock(), cfg=cfg, system_prompt="", original_prompt="",
                       redacted_text="", max_retries=0, user_print=lambda *a, **k: None)


def _tool():
    from src.jarvis.tools.registry import BUILTIN_TOOLS
    return BUILTIN_TOOLS["reviewLearnings"]


def _page(cfg, texte):
    from src.jarvis.appris.page import appris_path
    p = appris_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texte, encoding="utf-8")
    return p


def _lecture(**kw):
    from src.jarvis.appris.propose import Lecture
    return Lecture(**kw)


def _cand(texte="Il court le mardi.", citation="runs on Tuesdays"):
    from src.jarvis.appris.propose import Candidat
    return Candidat("fait", texte, citation, "2026-08-04")


FENETRE = [("2026-08-04", "d4")]


# ── The window is only retired when the pass finished with it ─────────


def test_a_truncated_reading_does_not_retire_the_window(tmp_path):
    """More candidates than the cap allows. What is left over is
    deferred to the next ask, not deleted — and it can only be deferred
    if the days it came from stay readable."""
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE,
                                     gardes=[_cand()], debordee=True)):
        _tool().run({}, ctx)

    assert not ctx.db.marquer_journal_lu.called


def test_a_failed_write_does_not_retire_the_window(tmp_path):
    """The page's own docstring calls a concurrent edit ordinary. If the
    write loses, the days must survive so the proposals can be made
    again."""
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE, gardes=[_cand()])), \
         patch("src.jarvis.tools.builtin.review_learnings.ajouter_propositions",
               return_value=False):
        resultat = _tool().run({}, ctx)

    assert not ctx.db.marquer_journal_lu.called
    assert "NOT" in resultat.reply_text or "could not" in resultat.reply_text.lower()


def test_a_reading_whose_every_item_was_malformed_does_not_retire_it(tmp_path):
    """On a small model this is the ordinary outcome, not an edge case.
    Retiring the days here means the day he stops pinning a weak model,
    the backlog is already gone."""
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE, bruts=4,
                                     mal_formes=2, infondes=2)):
        _tool().run({}, ctx)

    assert not ctx.db.marquer_journal_lu.called


def test_a_reading_whose_every_item_was_already_known_does_retire_it(tmp_path):
    """This one IS finished. Those suppressions are correct and permanent,
    and re-reading the same days would only repeat them for ever."""
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE, bruts=3,
                                     connus=2, refuses=1)):
        _tool().run({}, ctx)

    ctx.db.marquer_journal_lu.assert_called_once_with(FENETRE)


def test_a_clean_reading_retires_it_after_the_write_not_before(tmp_path):
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)
    ordre = []

    def _ecrit(*a, **k):
        ordre.append("écrit")
        return True

    ctx.db.marquer_journal_lu.side_effect = lambda *a, **k: ordre.append("marqué")

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE, gardes=[_cand()])), \
         patch("src.jarvis.tools.builtin.review_learnings.ajouter_propositions",
               side_effect=_ecrit):
        _tool().run({}, ctx)

    assert ordre == ["écrit", "marqué"]


def test_the_truncation_is_announced(tmp_path):
    """A deferral he cannot see is indistinguishable from a loss."""
    cfg = _Cfg(tmp_path)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE,
                                     gardes=[_cand()], debordee=True)):
        texte = _tool().run({}, _ctx(cfg)).reply_text

    assert "more" in texte.lower() or "again" in texte.lower()


# ── Deferral has to be bounded ────────────────────────────────────────


def test_she_stops_reading_when_the_page_is_already_full(tmp_path):
    """Otherwise a window that keeps overflowing is re-read on every ask
    for ever, and the page grows into a list nobody resolves."""
    cfg = _Cfg(tmp_path)
    lignes = "".join(
        f"- [ ] 2026-08-0{i % 9 + 1} · journal : Une proposition numéro {i}.\n"
        for i in range(cfg.appris_max_propositions * 3)
    )
    _page(cfg, "## Profil\n" + lignes + "\n## Règles\n")
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions") as lecture:
        resultat = _tool().run({}, ctx)

    assert not lecture.called
    assert not ctx.db.marquer_journal_lu.called
    assert resultat.reply_text


def test_answering_some_of_them_lets_her_read_again(tmp_path):
    """The ceiling counts what is unanswered, so striking or ticking
    lines is what re-opens the journal."""
    cfg = _Cfg(tmp_path)
    lignes = "".join(f"- ~~2026-08-04 · journal : Refusée {i}.~~\n"
                     for i in range(cfg.appris_max_propositions * 3))
    _page(cfg, "## Profil\n" + lignes + "\n## Règles\n")

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=FENETRE)) as lecture:
        _tool().run({}, _ctx(cfg))

    assert lecture.called


# ── She must not mine her own voice ───────────────────────────────────


def test_a_citation_quoting_her_own_proposal_is_dropped(tmp_path):
    """She reads the page aloud, the diary records her reading, and the
    next pass finds its own words in his journal and proposes them back.
    The prompt already forbids it and cannot be relied on to win."""
    from src.jarvis.appris.page import ETAT_ATTENTE, Proposition
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    note = ("[2026-08-05] The assistant said it had noted that the user is a "
            "senior developer with several SaaS projects.")
    db = MagicMock()
    db.get_recent_conversation_summaries.return_value = [
        {"date_utc": "2026-08-05", "summary": note}]
    db.journal_deja_lu.return_value = {}

    deja = [Proposition(section="profil", date="2026-08-03",
                        texte="They are a senior developer with several SaaS projects.",
                        citation="they are a senior developer with several SaaS projects",
                        etat=ETAT_ATTENTE, ligne="- [ ] x")]

    import json
    reponse = json.dumps([{
        "genre": "fait",
        "texte": "He is a senior developer with several SaaS projects.",
        "citation": "the user is a senior developer with several SaaS projects",
    }])

    from src.jarvis.memory.core import MemoryCore
    with patch("src.jarvis.appris.propose._appeler_modele", return_value=reponse):
        lecture = propositions(cfg, db, core=MemoryCore(tmp_path / "yuba"), deja=deja)

    assert lecture.gardes == []
