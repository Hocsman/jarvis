"""A question claimed and then dropped still has to close its episode.

The gate writes one `demandé` row when it raises a card. Every ending
writes a second row under the same id — `ok`, `échec`, `refusé`,
`décliné`, `expiré` — and nothing but that id says the two belong
together.

`settle_pending_confirmation` claims the card *before* asking the judge
what the sentence meant. On an unreadable answer it hands the turn back
with the card already gone and writes nothing, so the episode stays open
for the rest of the session: the sweep that closes orphans only runs at
the next daemon start, and by then it labels it `expiré`, a word the spec
defines as never answered and no longer answerable. He did answer. She
could not read it.

`expiré` is wrong for the same reason `décliné` would be: it puts his
name on a decision he did not make. The word for this is its own.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _db(tmp_path):
    from src.jarvis.memory.db import Database

    return Database(str(tmp_path / "t.db"))


def _action(nom="setGoal"):
    from src.jarvis.tools.confirmation import PendingAction

    return PendingAction.create(
        tool=nom, args={"x": 1}, risk="action", channel="parole",
        origin="voix", query_redacted="pose-moi un objectif",
        raised_at_turn=1, ttl_sec=180.0,
    )


def _lignes(db, request_id):
    return [r["outcome"] for r in db.recent_actions(200)
            if r.get("request_id") == request_id]


def _regle(db, settled, origin="voix"):
    from src.jarvis.reply.engine import _record_settled_action

    _record_settled_action(db, MagicMock(), settled, origin=origin,
                           redacted="euh attends")


def test_an_answer_she_could_not_read_closes_the_episode(tmp_path):
    from src.jarvis.reply.engine import _Settled

    db = _db(tmp_path)
    action = _action()
    db.record_action(tool=action.tool, args=action.args, risk=action.risk,
                     verdict="demande", outcome="demandé",
                     request_id=action.request_id, origin="voix")

    _regle(db, _Settled(action=action))

    issues = _lignes(db, action.request_id)
    assert len(issues) == 2, f"un épisode ouvert : {issues}"


def test_it_is_not_recorded_as_a_refusal(tmp_path):
    """`décliné` means he was asked and said no. Reading an unreadable
    answer as a no puts his name on a decision he never made — the
    reasoning the judge already applies when it answers UNCLEAR on a
    timeout rather than DENIED."""
    from src.jarvis.reply.engine import _Settled

    db = _db(tmp_path)
    action = _action()

    _regle(db, _Settled(action=action))

    assert "décliné" not in _lignes(db, action.request_id)


def test_a_decline_is_still_a_decline(tmp_path):
    """The control. A change that gave every ending the same word would
    pass the two tests above and lose the distinction the ledger exists
    to keep."""
    from src.jarvis.reply.engine import _Settled

    db = _db(tmp_path)
    action = _action()

    _regle(db, _Settled(action=action, declined=True))

    assert "décliné" in _lignes(db, action.request_id)


def test_nothing_is_written_when_no_card_was_waiting(tmp_path):
    """An ordinary turn writes no settlement, because no question was
    claimed."""
    from src.jarvis.reply.engine import _Settled

    db = _db(tmp_path)

    _regle(db, _Settled())

    assert db.recent_actions(50) == []
