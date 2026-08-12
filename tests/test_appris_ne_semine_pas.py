"""She does not mine the days she spoke on.

Observed on the real machine, not imagined. She read three proposals
aloud; the summariser recorded the reading; the next pass found those
sentences in his journal and proposed all three back — and he had struck
every one of them an hour earlier.

Two promises broke at once. The self-feeding loop, and the one that
matters more: **a refusal is as durable as an acceptance**. A proposal
that returns after being struck is consent by attrition, which is what
that rule exists to prevent.

The lexical guard could not save it. The struck lines were English and
the returning ones French, so nothing matched — and it was the language
change itself, shipped an hour before, that put them in different
languages.

So the fix is not a better comparison. It is the mechanism: a day she
spoke proposals on is a day whose summary carries her own voice, and she
does not read those days at all. Deterministic, and it names no language
because it never looks at a word.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest


def _db(tmp_path):
    from src.jarvis.memory.db import Database
    return Database(str(tmp_path / "t.db"), sqlite_vss_path=None)


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")
        self.llm_chat_model = "test-model"
        self.appris_jours = 14
        self.appris_max_propositions = 3
        self.appris_seuil_doublon = 90
        self.appris_timeout_sec = 30.0
        self.response_language = "français"


# ── The ledger of days she spoke on ───────────────────────────────────


def test_no_day_is_hers_to_begin_with(tmp_path):
    assert _db(tmp_path).jours_ou_elle_a_parle() == set()


def test_a_day_she_spoke_on_is_remembered(tmp_path):
    db = _db(tmp_path)

    db.marquer_jour_de_parole("2026-08-12")

    assert db.jours_ou_elle_a_parle() == {"2026-08-12"}


def test_recording_the_same_day_twice_is_harmless(tmp_path):
    db = _db(tmp_path)
    db.marquer_jour_de_parole("2026-08-12")
    db.marquer_jour_de_parole("2026-08-12")

    assert db.jours_ou_elle_a_parle() == {"2026-08-12"}


def test_it_survives_a_reopen(tmp_path):
    _db(tmp_path).marquer_jour_de_parole("2026-08-12")

    assert "2026-08-12" in _db(tmp_path).jours_ou_elle_a_parle()


def test_bookkeeping_never_raises_into_the_caller(tmp_path):
    _db(tmp_path).marquer_jour_de_parole(None)  # type: ignore[arg-type]


# ── And the window skips them ─────────────────────────────────────────


def _fenetre_db(rows, parle=()):
    db = MagicMock()
    db.get_recent_conversation_summaries.return_value = [
        {"date_utc": d, "summary": s} for d, s in rows
    ]
    db.journal_deja_lu.return_value = {}
    db.jours_ou_elle_a_parle.return_value = set(parle)
    return db


def test_a_day_she_spoke_on_is_never_read(tmp_path):
    """The exact observed case: her own reading, summarised, mined back."""
    from src.jarvis.appris.propose import _fenetre

    note = ("Hocine a demandé ce que l'assistant avait appris de lui. "
            "L'assistant a listé trois informations en attente de validation : "
            "Hocine est développeur senior avec plusieurs projets SaaS.")

    fenetre, _ = _fenetre(_Cfg(tmp_path),
                          _fenetre_db([("2026-08-12", note)], parle=["2026-08-12"]))

    assert fenetre == []


def test_the_other_days_are_still_read(tmp_path):
    from src.jarvis.appris.propose import _fenetre

    fenetre, _ = _fenetre(
        _Cfg(tmp_path),
        _fenetre_db([("2026-08-12", "sa voix"), ("2026-08-11", "sa vie à lui")],
                    parle=["2026-08-12"]))

    assert [d for d, _ in fenetre] == ["2026-08-11"]


def test_a_day_she_never_spoke_on_is_read(tmp_path):
    from src.jarvis.appris.propose import _fenetre

    fenetre, _ = _fenetre(_Cfg(tmp_path),
                          _fenetre_db([("2026-08-11", "sa vie à lui")]))

    assert [d for d, _ in fenetre] == ["2026-08-11"]


# ── The tool records the day ──────────────────────────────────────────


def _ctx(cfg):
    from src.jarvis.tools.base import ToolContext
    return ToolContext(db=MagicMock(), cfg=cfg, system_prompt="", original_prompt="",
                       redacted_text="", max_retries=0, user_print=lambda *a, **k: None)


def test_reading_the_journal_marks_today_as_hers(tmp_path):
    """Anything she says about what she found lands in today's summary,
    so today is contaminated whatever the reading returned."""
    from src.jarvis.appris.propose import Lecture
    from src.jarvis.tools.registry import BUILTIN_TOOLS

    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=Lecture(appelee=True, lues=[("2026-08-11", "d")])):
        BUILTIN_TOOLS["reviewLearnings"].run({}, ctx)

    assert ctx.db.marquer_jour_de_parole.called


def test_a_reading_that_did_not_happen_still_marks_the_day(tmp_path):
    """She still tells him she could not look, and that sentence is what
    the summariser writes down."""
    from src.jarvis.appris.propose import Lecture
    from src.jarvis.tools.registry import BUILTIN_TOOLS

    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=Lecture(appelee=False)):
        BUILTIN_TOOLS["reviewLearnings"].run({}, ctx)

    assert ctx.db.marquer_jour_de_parole.called
