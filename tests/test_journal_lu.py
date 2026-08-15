"""What she has already been asked about.

A pass over the journal must not re-propose the same lines every time he
asks, and must not go blind to a day because it looked at it once.

The second half is the trap. A diary row is rewritten in place all day
long — `INSERT OR REPLACE` keyed on the date — so remembering "the 4th
was read" from the first pass onwards means everything he says after
10am on the 4th is never read at all, for ever, and nothing anywhere
says so. What is recorded is therefore the text as it read when it was
read, and the row re-opens the moment its content moves.

Nothing is recorded for a reading that did not happen. A model that
timed out has not looked at anything, and marking its window read would
skip those days permanently on the strength of a failure.
"""

from __future__ import annotations

import hashlib

import pytest


def _db(tmp_path):
    from src.jarvis.memory.db import Database
    return Database(str(tmp_path / "t.db"), sqlite_vss_path=None)


def _digest(texte: str) -> str:
    return hashlib.sha256(texte.encode("utf-8")).hexdigest()


# ── Remembering what was read ─────────────────────────────────────────


def test_nothing_is_read_to_begin_with(tmp_path):
    assert _db(tmp_path).journal_deja_lu() == {}


def test_a_row_recorded_comes_back_with_what_it_said(tmp_path):
    db = _db(tmp_path)

    db.marquer_journal_lu([("2026-08-04", _digest("il court le mardi"))])

    assert db.journal_deja_lu() == {"2026-08-04": _digest("il court le mardi")}


def test_a_rewritten_row_no_longer_matches(tmp_path):
    """The whole point. The date is unchanged, the text is not, so the
    digest differs and the caller reads it again."""
    db = _db(tmp_path)
    db.marquer_journal_lu([("2026-08-04", _digest("il court le mardi"))])

    deja = db.journal_deja_lu()

    assert deja.get("2026-08-04") != _digest("il court le mardi, et il a un chat")


def test_reading_the_same_row_twice_records_it_once(tmp_path):
    db = _db(tmp_path)

    db.marquer_journal_lu([("2026-08-04", _digest("a"))])
    db.marquer_journal_lu([("2026-08-04", _digest("b"))])

    deja = db.journal_deja_lu()
    assert list(deja) == ["2026-08-04"]
    assert deja["2026-08-04"] == _digest("b")


def test_several_rows_land_together(tmp_path):
    db = _db(tmp_path)

    db.marquer_journal_lu([("2026-08-03", _digest("a")), ("2026-08-04", _digest("b"))])

    assert set(db.journal_deja_lu()) == {"2026-08-03", "2026-08-04"}


def test_an_empty_batch_is_not_an_error(tmp_path):
    """A reading that happened and proposed nothing still has a window,
    but a caller handing over nothing must not raise."""
    db = _db(tmp_path)

    db.marquer_journal_lu([])

    assert db.journal_deja_lu() == {}


def test_bookkeeping_never_raises_into_the_caller(tmp_path):
    """Same contract as the action ledger: losing a mark costs one
    re-read, and raising costs the user their answer."""
    db = _db(tmp_path)

    db.marquer_journal_lu([("pas une date", None)])  # type: ignore[list-item]


def test_it_survives_a_reopen(tmp_path):
    db = _db(tmp_path)
    db.marquer_journal_lu([("2026-08-04", _digest("a"))])

    assert _db(tmp_path).journal_deja_lu()["2026-08-04"] == _digest("a")


def test_an_existing_database_gains_the_table(tmp_path):
    """The migration is additive and idempotent: start-up runs it every
    time, and an install that predates the table must not need a wipe."""
    import sqlite3

    chemin = str(tmp_path / "vieille.db")
    vieille = sqlite3.connect(chemin)
    vieille.execute("CREATE TABLE meals (id INTEGER PRIMARY KEY)")
    vieille.execute("PRAGMA user_version = 3")
    vieille.commit()
    vieille.close()

    from src.jarvis.memory.db import Database

    db = Database(chemin, sqlite_vss_path=None)
    db.marquer_journal_lu([("2026-08-04", _digest("a"))])

    assert db.journal_deja_lu()["2026-08-04"] == _digest("a")


# ── The fourth source word ────────────────────────────────────────────


def test_a_confirmed_entry_round_trips(tmp_path):
    """`confirmé` says she noticed it and he agreed, which is neither
    `dit` (he stated it) nor `corrigé` (he corrected her). A reader of
    his own file a month later can tell the three apart."""
    from src.jarvis.memory.core import (
        SECTION_PROFILE, SOURCE_CONFIRMED, MemoryCore,
    )

    core = MemoryCore(tmp_path / "yuba")
    core.remember(SECTION_PROFILE, "Il court le mardi matin.",
                  source=SOURCE_CONFIRMED)

    entree = core.active(SECTION_PROFILE)[0]
    assert entree.source == "confirmé"
    assert entree.text == "Il court le mardi matin."


def test_retiring_a_confirmed_entry_keeps_the_word(tmp_path):
    """The word says where the belief came from. Losing it on retirement
    would rewrite how he came to agree to it."""
    from src.jarvis.memory.core import (
        SECTION_PROFILE, SOURCE_CONFIRMED, MemoryCore,
    )

    core = MemoryCore(tmp_path / "yuba")
    core.remember(SECTION_PROFILE, "Il court le mardi matin.",
                  source=SOURCE_CONFIRMED)
    core.retire(SECTION_PROFILE, "Il court le mardi matin.")

    assert "confirmé" in (core.path_for(SECTION_PROFILE)).read_text(encoding="utf-8")


def test_the_model_never_sees_the_word(tmp_path):
    """The attribution is for him. The model needs the fact and nothing
    else, and a source word in the prompt is one more thing for it to
    reason about."""
    from src.jarvis.memory.core import (
        SECTION_PROFILE, SOURCE_CONFIRMED, MemoryCore, build_core_profile,
    )

    core = MemoryCore(tmp_path / "yuba")
    core.remember(SECTION_PROFILE, "Il court le mardi matin.",
                  source=SOURCE_CONFIRMED)

    bloc = str(build_core_profile(core))
    assert "court le mardi" in bloc
    assert "confirmé" not in bloc
