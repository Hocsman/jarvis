"""A day can be found by what it was about, not only by its words.

The diary's search is meant to be hybrid: sixty per cent semantic, forty
per cent keyword. On a default install the sqlite-vss extension is not
loaded, so a Python vector store is built instead — and searched on every
query. It was never written to: all three write sites were gated on
sqlite-vss being enabled, so the store the reader consults stayed empty
for the life of the install.

The half that makes it a silent failure rather than a missing feature is
the cost. The query is embedded on every enrichment turn regardless, so
the round-trip is paid, the store is consulted, and the ranking collapses
to keyword matching — which usually returns something, so nothing ever
looks broken.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _vecteur(marque: float) -> list[float]:
    """A 768-wide vector pointing in one of a few distinct directions."""
    v = [0.0] * 768
    v[int(marque)] = 1.0
    return v


def _cfg(tmp_path):
    return SimpleNamespace(
        db_path=str(tmp_path / "journal.db"),
        embedding_model="nomic-embed-text",
        llm_chat_model="m",
        ollama_chat_model="m",
        ollama_base_url="http://x",
        llm_provider="ollama",
        summary_max_chars=2000,
    )


def _aujourdhui() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).date().isoformat()


def _flush(db, cfg, texte: str, direction: float):
    """One real diary flush, with the embedding boundary under our
    control. The date is today's — the flush derives it itself."""
    from src.jarvis.memory import conversation

    with patch.object(conversation, "generate_conversation_summary",
                      return_value=(texte, "")), \
         patch.object(conversation, "_embed_text",
                      return_value=_vecteur(direction)):
        conversation.update_daily_conversation_summary(
            db=db, cfg=cfg, new_chunks=[texte],
        )


def _autre_jour(db, date: str, texte: str, direction: float):
    """A background day, written through the same pair the flush uses."""
    rid = db.upsert_conversation_summary(date_utc=date, summary=texte, topics="")
    db.upsert_summary_embedding(rid, _vecteur(direction))


def test_a_day_is_findable_by_meaning_and_not_only_by_its_words(tmp_path):
    """The property the hybrid search exists for. The query shares no word
    with either day; only the vectors separate them."""
    from src.jarvis.memory.db import Database

    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)

    _flush(db, cfg, "il a parlé de bicyclettes toute la matinée", 7)
    _autre_jour(db, "2026-01-05", "il a parlé de comptabilité toute la matinée", 300)

    resultats = db.search_hybrid("recherche", json.dumps(_vecteur(7)), top_k=5)

    assert resultats, "un magasin de vecteurs alimenté doit répondre"
    assert f"[{_aujourdhui()}]" in resultats[0]["text"]


def test_the_other_direction_returns_the_other_day(tmp_path):
    """The control: the ranking follows the query's meaning rather than
    the order the days were written in."""
    from src.jarvis.memory.db import Database

    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)

    _flush(db, cfg, "il a parlé de bicyclettes toute la matinée", 7)
    _autre_jour(db, "2026-01-05", "il a parlé de comptabilité toute la matinée", 300)

    resultats = db.search_hybrid("recherche", json.dumps(_vecteur(300)), top_k=5)

    assert resultats
    assert "[2026-01-05]" in resultats[0]["text"]


def test_rewriting_a_day_replaces_its_vector_rather_than_adding_one(tmp_path):
    """A day is rewritten on every flush. Its vector has to follow, or the
    search answers about a version of the day that no longer exists."""
    from src.jarvis.memory.db import Database

    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)

    _flush(db, cfg, "il a parlé de bicyclettes", 7)
    _flush(db, cfg, "finalement il a parlé de comptabilité", 300)

    proches = db.search_hybrid("recherche", json.dumps(_vecteur(7)), top_k=5)

    assert len(proches) == 1, f"une journée, un vecteur — vu {[r['text'] for r in proches]}"


def test_nothing_is_embedded_when_there_is_nowhere_to_put_it(tmp_path):
    """The other half of the cost. A store that cannot hold anything must
    not cost a round-trip per flush."""
    from src.jarvis.memory import conversation
    from src.jarvis.memory.db import Database

    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    db._python_vector_store = None
    db.is_vss_enabled = False

    with patch.object(conversation, "generate_conversation_summary",
                      return_value=("il a parlé de vélos", "")), \
         patch.object(conversation, "_embed_text") as embed:
        conversation.update_daily_conversation_summary(
            db=db, cfg=cfg, new_chunks=["il a parlé de vélos"],
        )

    embed.assert_not_called()


def test_no_query_is_embedded_when_there_is_nothing_to_compare_it_to(tmp_path):
    """The same rule on the read side. A query vector with nothing to
    compare it against costs a hop per turn and changes no result."""
    from src.jarvis.memory import conversation
    from src.jarvis.memory.db import Database

    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    db._python_vector_store = None
    db.is_vss_enabled = False
    db.upsert_conversation_summary(
        date_utc="2026-01-05", summary="il a parlé de vélos", topics="vélos")

    with patch.object(conversation, "_embed_text") as embed:
        conversation.search_conversation_memory_by_keywords(
            db=db, keywords=["vélos"], cfg=cfg,
        )

    embed.assert_not_called()
