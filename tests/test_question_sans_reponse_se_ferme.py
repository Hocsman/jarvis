"""A question nobody answered must stop looking like a question still waiting.

The gate writes one `demandé` row when it raises a card, and one settling
row when the episode ends: `ok`, `échec`, `refusé`, `décliné`, `expiré`.
Two ways out leave no settling row at all.

The card sits past its TTL and the next question displaces it —
`raise_pending` overwrites a stale card and the old one vanishes with the
object. And the process dies with a card in hand: a pending confirmation
deliberately never reaches disk, so nothing on restart knows the question
existed except its own orphan row.

Both leave the Activity tab claiming she is still waiting on an answer
she can no longer accept. The ledger's whole job is to say what happened,
and "still asking" is not what happened.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _iso(delta_sec: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_sec)).isoformat()


def _db(tmp_path):
    from src.jarvis.memory.db import Database

    return Database(str(tmp_path / "t.db"))


def _lignes(db, request_id: str) -> list[str]:
    return [r["outcome"] for r in db.recent_actions(200)
            if r.get("request_id") == request_id]


# ── A card displaced by the next one ───────────────────────────────────


def _action(nom: str, ttl: float = 180.0):
    from src.jarvis.tools.confirmation import PendingAction

    return PendingAction.create(
        tool=nom, args={"x": 1}, risk="action", origin="chat",
        query_redacted="fais quelque chose", raised_at_turn=1,
        ttl_sec=ttl, channel="geste",
    )


def test_a_card_left_to_expire_is_settled_when_the_next_one_arrives(tmp_path):
    """`raise_pending` overwrites a stale card. The episode it drops is
    one the user was really asked about, and it has to close."""
    from src.jarvis.memory.conversation import DialogueMemory

    memoire = DialogueMemory()
    vieille = _action("setGoal", ttl=0.0)
    memoire.raise_pending(vieille)

    perdue = memoire.take_expired_pending()

    assert perdue is not None
    assert perdue.request_id == vieille.request_id
    assert memoire.peek_pending() is None


def test_a_card_still_within_its_deadline_is_left_alone(tmp_path):
    """The control. Sweeping a live question would answer it for him."""
    from src.jarvis.memory.conversation import DialogueMemory

    memoire = DialogueMemory()
    memoire.raise_pending(_action("setGoal", ttl=180.0))

    assert memoire.take_expired_pending() is None
    assert memoire.peek_pending() is not None


def test_the_gate_closes_the_episode_it_displaces(tmp_path):
    """End to end through the gate: a stale card leaves an `expiré` row
    before the new question is raised."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.tools.confirmation import Confirmation
    from src.jarvis.tools.registry import run_tool_with_retries

    db = _db(tmp_path)
    memoire = DialogueMemory()
    vieille = _action("setGoal", ttl=0.0)
    memoire.raise_pending(vieille)

    confirmation = Confirmation(
        store=memoire, publish=lambda a: None, ttl_sec=180.0, approval=None,
    )
    run_tool_with_retries(
        db=db, cfg=_cfg(tmp_path), tool_name="setRoutine",
        tool_args={"routine": "matin"}, system_prompt="",
        original_prompt="fais", redacted_text="fais",
        confirmation=confirmation, origin="chat",
    )

    assert "expiré" in _lignes(db, vieille.request_id)


def _cfg(tmp_path):
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.db_path = str(tmp_path / "t.db")
    cfg.mcps = {}
    cfg.voice_debug = False
    return cfg


# ── A card the process died holding ────────────────────────────────────


def test_a_question_from_a_dead_process_is_closed_at_start_up(tmp_path, capsys):
    """A pending confirmation never reaches disk, on purpose. So a row
    still open when the daemon starts belongs to a question whose card
    died with the process that raised it: nothing can ever answer it."""
    from src.jarvis.memory.db import close_orphan_questions

    db = _db(tmp_path)
    db.record_action(tool="setGoal", args={"x": 1}, risk="action",
                     verdict="demande", outcome="demandé",
                     request_id="q-morte", origin="chat")

    fermees = close_orphan_questions(db)

    assert fermees == 1
    assert "expiré" in _lignes(db, "q-morte")
    assert "⏳" in capsys.readouterr().out


def test_a_question_that_was_answered_is_left_alone(tmp_path):
    """Every settled shape stays settled: closing one twice would put a
    second ending on an episode that already had one."""
    from src.jarvis.memory.db import close_orphan_questions

    db = _db(tmp_path)
    for i, issue in enumerate(("ok", "décliné", "refusé", "échec", "expiré")):
        rid = f"q-{i}"
        db.record_action(tool="setGoal", args=None, risk="action",
                         verdict="demande", outcome="demandé",
                         request_id=rid, origin="chat")
        db.record_action(tool="setGoal", args=None, risk="action",
                         verdict="demande", outcome=issue,
                         request_id=rid, origin="chat")

    assert close_orphan_questions(db) == 0


def test_a_free_call_is_not_a_question(tmp_path):
    """Only an episode that actually asked something can be left hanging.
    A free call writes one row and is complete."""
    from src.jarvis.memory.db import close_orphan_questions

    db = _db(tmp_path)
    db.record_action(tool="webSearch", args=None, risk="lecture",
                     verdict="libre", outcome="ok", origin="chat")

    assert close_orphan_questions(db) == 0


def test_nothing_is_announced_when_nothing_was_hanging(tmp_path, capsys):
    """A line printed on every start-up is a line he learns to skip."""
    from src.jarvis.memory.db import close_orphan_questions

    db = _db(tmp_path)
    capsys.readouterr()

    close_orphan_questions(db)

    assert capsys.readouterr().out == ""
