"""Exactly one question waits at a time, and it never reaches disk.

The store is a slot on `DialogueMemory` because that is the object voice
and text already share — the same conversation seen two ways. Not the
hot cache, which evicts under load and would let a busy router discard a
destructive action mid-question. Not a module global, which unattended
routines running on their own threads would trample. Not the database:
a deletion proposed before a crash and approved after it is an approval
given without the context that produced it, so a restart erases every
pending request rather than resurrecting it.

The invariants that matter are all about *not* granting: one slot so a
second question cannot quietly displace the first, an answer that must
belong to the turn immediately after the question, an origin that must
match so a click cannot answer for a voice question, and a pop that
returns the record exactly once.
"""

from __future__ import annotations

import threading

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.confirmation import (
    CHANNEL_GESTE,
    CHANNEL_PAROLE,
    PendingAction,
)
from src.jarvis.tools.policy import RISK_ACTION, RISK_DESTRUCTIVE


@pytest.fixture
def dm():
    return DialogueMemory()


def _action(dm, *, tool="localFiles", args=None, channel=CHANNEL_PAROLE,
            origin="voix", risk=RISK_ACTION, ttl_sec=180.0, turn=None):
    return PendingAction.create(
        tool=tool, args=args if args is not None else {"operation": "read"},
        risk=risk, channel=channel, origin=origin, query_redacted="fais-le",
        raised_at_turn=dm.current_turn() if turn is None else turn,
        ttl_sec=ttl_sec,
    )


# ── One slot ──────────────────────────────────────────────────────────


def test_a_raised_question_is_held(dm):
    dm.begin_turn()
    p = _action(dm)

    assert dm.raise_pending(p) is p
    assert dm.peek_pending() is p


def test_a_second_distinct_question_is_refused(dm):
    """One card at a time. A three-step plan asks about the first step
    and never reaches the second, rather than stacking questions the user
    has to disentangle."""
    dm.begin_turn()
    first = _action(dm, tool="localFiles")
    dm.raise_pending(first)

    assert dm.raise_pending(_action(dm, tool="deleteMeal")) is None
    assert dm.peek_pending() is first


def test_the_same_call_proposed_twice_is_one_question(dm):
    """A re-ask, not a duplicate: the model re-proposing the same tool
    should not mint a second id and a second ledger row."""
    dm.begin_turn()
    first = dm.raise_pending(_action(dm, args={"operation": "read", "path": "/a"}))

    dm.begin_turn()
    again = dm.raise_pending(_action(dm, args={"path": "/a", "operation": "read"}))

    assert again is not None
    assert again.request_id == first.request_id


def test_a_re_ask_moves_the_question_into_the_current_turn(dm):
    """Otherwise the answer to the re-asked question arrives one turn too
    late to be accepted."""
    dm.begin_turn()
    dm.raise_pending(_action(dm))
    dm.begin_turn()
    again = dm.raise_pending(_action(dm))

    assert again.raised_at_turn == dm.current_turn()


def test_a_re_ask_does_not_restart_the_clock(dm):
    """A model that re-proposes the same tool every turn must not be able
    to hold a card open indefinitely. The deadline belongs to the moment
    the user was first asked."""
    dm.begin_turn()
    first = dm.raise_pending(_action(dm))
    dm.begin_turn()
    again = dm.raise_pending(_action(dm))

    assert again.created_monotonic == first.created_monotonic


# ── The spoken door ───────────────────────────────────────────────────


def test_the_next_utterance_may_answer(dm):
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, origin="voix"))
    seq = dm.begin_turn()

    assert dm.take_pending_for_utterance("voix", seq) is p


def test_an_answer_arriving_a_turn_late_is_not_accepted(dm):
    """The window is the very next turn. A grant that survives an
    intervening exchange is a grant answering a question the user has
    stopped thinking about."""
    dm.begin_turn()
    dm.raise_pending(_action(dm, origin="voix"))
    dm.begin_turn()
    seq = dm.begin_turn()

    assert dm.take_pending_for_utterance("voix", seq) is None


def test_a_stale_question_is_not_destroyed_by_a_late_utterance(dm):
    """It leaves the click door open: the card is still on screen, and
    the user pressing the button after talking about something else is a
    perfectly good decision."""
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, origin="voix"))
    dm.begin_turn()
    seq = dm.begin_turn()
    dm.take_pending_for_utterance("voix", seq)

    assert dm.peek_pending() is p


def test_a_typed_answer_cannot_settle_a_spoken_question(dm):
    """Different rooms. The person at the keyboard is not necessarily the
    person who was asked."""
    dm.begin_turn()
    dm.raise_pending(_action(dm, origin="voix"))
    seq = dm.begin_turn()

    assert dm.take_pending_for_utterance("chat", seq) is None


def test_a_gesture_only_question_is_never_taken_by_an_utterance(dm):
    """The whole trust boundary in one assertion: no sentence, however
    confidently transcribed, reaches a destructive action."""
    dm.begin_turn()
    dm.raise_pending(_action(dm, channel=CHANNEL_GESTE, risk=RISK_DESTRUCTIVE))
    seq = dm.begin_turn()

    assert dm.take_pending_for_utterance("voix", seq) is None


def test_an_expired_question_is_not_taken(dm):
    dm.begin_turn()
    dm.raise_pending(_action(dm, ttl_sec=0.0))
    seq = dm.begin_turn()

    assert dm.take_pending_for_utterance("voix", seq) is None


def test_taking_it_consumes_it(dm):
    """Consume before deciding: the record is gone before the judge runs,
    before the digest is checked and before anything executes."""
    dm.begin_turn()
    dm.raise_pending(_action(dm, origin="voix"))
    seq = dm.begin_turn()
    dm.take_pending_for_utterance("voix", seq)

    assert dm.peek_pending() is None


# ── The gesture door ──────────────────────────────────────────────────


def test_a_click_settles_the_question_by_id(dm):
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, channel=CHANNEL_GESTE))

    assert dm.take_pending_by_id(p.request_id) is p


def test_a_click_cannot_settle_it_twice(dm):
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, channel=CHANNEL_GESTE))
    dm.take_pending_by_id(p.request_id)

    assert dm.take_pending_by_id(p.request_id) is None


def test_an_unknown_id_settles_nothing(dm):
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, channel=CHANNEL_GESTE))

    assert dm.take_pending_by_id("cf_inventé") is None
    assert dm.peek_pending() is p


def test_an_expired_question_cannot_be_clicked(dm):
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, channel=CHANNEL_GESTE, ttl_sec=0.0))

    assert dm.take_pending_by_id(p.request_id) is None


def test_two_threads_racing_on_one_id_get_one_between_them(dm):
    """A double-click, or a click racing the expiry sweep. Exactly one
    execution, or the gate is not a gate."""
    dm.begin_turn()
    p = dm.raise_pending(_action(dm, channel=CHANNEL_GESTE))

    got = []
    barrier = threading.Barrier(8)

    def _claim():
        barrier.wait()
        got.append(dm.take_pending_by_id(p.request_id))

    threads = [threading.Thread(target=_claim) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len([g for g in got if g is not None]) == 1


def test_two_threads_racing_to_raise_get_one_question(dm):
    dm.begin_turn()
    raised = []
    barrier = threading.Barrier(8)

    def _raise(i):
        p = PendingAction.create(
            tool=f"tool{i}", args={}, risk=RISK_ACTION, channel=CHANNEL_PAROLE,
            origin="voix", query_redacted="", raised_at_turn=dm.current_turn(),
            ttl_sec=180.0,
        )
        barrier.wait()
        raised.append(dm.raise_pending(p))

    threads = [threading.Thread(target=_raise, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len([r for r in raised if r is not None]) == 1


# ── Clearing ──────────────────────────────────────────────────────────


def test_a_new_conversation_drops_the_question(dm):
    """A question raised before a gap does not survive it: whatever the
    user was doing then, they are doing something else now."""
    dm.begin_turn()
    dm.raise_pending(_action(dm))

    dm.clear_pending()

    assert dm.peek_pending() is None


def test_there_is_no_pending_question_on_a_fresh_memory(dm):
    assert dm.peek_pending() is None


def test_the_turn_counter_advances(dm):
    a = dm.begin_turn()
    b = dm.begin_turn()

    assert b == a + 1


# ── Never on disk ─────────────────────────────────────────────────────


def test_nothing_pending_is_written_to_the_database(dm, tmp_path):
    """Structural: a pending request holds tool arguments the model wrote
    from the user's words, and an approval that outlives the process is
    an approval given without the context that produced it."""
    from src.jarvis.memory.db import Database

    db = Database(str(tmp_path / "t.db"), sqlite_vss_path=None)
    try:
        dm.begin_turn()
        dm.raise_pending(_action(dm, args={"path": "/secret"}))

        tables = [
            r[0] for r in db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        for table in tables:
            if table.startswith("sqlite_"):
                continue
            rows = db.conn.execute(f'SELECT * FROM "{table}"').fetchall()
            assert not any("/secret" in str(tuple(r)) for r in rows)
    finally:
        db.close()
