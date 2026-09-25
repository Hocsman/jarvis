"""The click, and the line that carries it.

`__CHAT_DECISION__` is the one message on the stdin bus that authorises
an irreversible action, so it is validated harder than the query line
beside it: a real string id, a real boolean, and an id that names the
question actually waiting. Anything else changes nothing and does not
take the monitor down with it.

The click itself never queues and never waits. It takes the shared query
lock without blocking, and on failure says so and leaves the card up
with its deadline still running — a third semantic on a lock that
already has two would be worse than telling the user to try again, and
silently discarding a decision they correctly made is worse than both.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis import daemon
from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.tools.confirmation import CHANNEL_GESTE, PendingAction


@pytest.fixture(autouse=True)
def _clean_daemon():
    """The daemon's globals are module state; leave them as found."""
    saved = (
        daemon._global_dialogue_memory, daemon._global_cfg, daemon._global_db,
        daemon._global_stop_requested,
    )
    yield
    (
        daemon._global_dialogue_memory, daemon._global_cfg, daemon._global_db,
        daemon._global_stop_requested,
    ) = saved
    if daemon._chat_query_lock.locked():
        daemon._chat_query_lock.release()


@pytest.fixture
def waiting():
    """A daemon with one destructive question on the table."""
    dm = DialogueMemory()
    dm.begin_turn()
    action = PendingAction.create(
        tool="localFiles", args={"operation": "delete", "path": "/a"},
        risk="destructif", channel=CHANNEL_GESTE, origin="voix",
        query_redacted="supprime", raised_at_turn=dm.current_turn(),
        ttl_sec=180.0,
    )
    dm.raise_pending(action)

    daemon._global_dialogue_memory = dm
    daemon._global_cfg = MagicMock()
    daemon._global_db = MagicMock()
    daemon._global_stop_requested = False
    return dm, action


# ── The decision line ─────────────────────────────────────────────────


def _line(payload):
    return daemon.CHAT_DECISION_IPC_PREFIX + json.dumps(payload)


def test_a_well_formed_decision_is_handled(waiting):
    _, action = waiting

    with patch.object(daemon, "resolve_confirmation", return_value="ok") as resolved:
        handled = daemon.handle_chat_decision_stdin_line(
            _line({"request_id": action.request_id, "approved": True})
        )

    assert handled is True
    assert resolved.call_args.args == (action.request_id, True)


def test_a_line_that_is_not_a_decision_is_left_alone():
    assert daemon.handle_chat_decision_stdin_line("SHUTDOWN") is False


@pytest.mark.parametrize("payload", [
    {"request_id": 42, "approved": True},
    {"request_id": None, "approved": True},
    {"request_id": ["cf_x"], "approved": True},
    {"approved": True},
    {"request_id": "cf_x"},
    {"request_id": "cf_x", "approved": "true"},
    {"request_id": "cf_x", "approved": 1},
    {"request_id": "cf_x", "approved": None},
])
def test_a_malformed_decision_authorises_nothing(waiting, payload):
    """`approved` must be a real boolean. A truthy 1 or "true" from a
    buggy writer must not run a deletion."""
    with patch.object(daemon, "resolve_confirmation") as resolved:
        handled = daemon.handle_chat_decision_stdin_line(_line(payload))

    assert handled is True
    assert resolved.called is False


def test_unparseable_json_does_not_crash_the_monitor(waiting):
    with patch.object(daemon, "resolve_confirmation") as resolved:
        handled = daemon.handle_chat_decision_stdin_line(
            daemon.CHAT_DECISION_IPC_PREFIX + "{pas du json"
        )

    assert handled is True
    assert resolved.called is False


def test_an_explicit_refusal_is_carried_through(waiting):
    _, action = waiting

    with patch.object(daemon, "resolve_confirmation") as resolved:
        daemon.handle_chat_decision_stdin_line(
            _line({"request_id": action.request_id, "approved": False})
        )

    assert resolved.call_args.args == (action.request_id, False)


# ── Resolving ─────────────────────────────────────────────────────────


def test_an_unknown_id_settles_nothing(waiting):
    dm, action = waiting

    assert daemon.resolve_confirmation("cf_inventé", True) == "inconnue"
    assert dm.peek_pending() is action


def test_a_click_claims_the_question(waiting):
    dm, action = waiting

    with patch.object(daemon, "_resume_after_confirmation"):
        daemon.resolve_confirmation(action.request_id, True)

    assert dm.peek_pending() is None


def test_a_second_click_finds_nothing_left(waiting):
    """A double-click is one execution. The resume double releases the
    lock as the real one does in its `finally`, so the second click is
    judged on the question being gone rather than on the lock."""
    _, action = waiting

    def _resume_like_production(_action):
        daemon._chat_query_lock.release()

    with patch.object(daemon, "_resume_after_confirmation",
                      side_effect=_resume_like_production):
        first = daemon.resolve_confirmation(action.request_id, True)
    second = daemon.resolve_confirmation(action.request_id, True)

    assert first == "ok"
    assert second == "inconnue"


def test_a_resume_that_cannot_start_does_not_wedge_the_daemon(waiting):
    """The worker owns the lock's release. If it never starts, nothing
    releases it and every later query is rejected as busy for the life of
    the process."""
    _, action = waiting

    with patch("threading.Thread", side_effect=RuntimeError("plus de fils")):
        daemon.resolve_confirmation(action.request_id, True)

    assert daemon._chat_query_lock.locked() is False


def test_a_refusal_runs_nothing_and_records_it(waiting):
    dm, action = waiting

    with patch.object(daemon, "_resume_after_confirmation") as resumed:
        outcome = daemon.resolve_confirmation(action.request_id, False)

    assert outcome == "décliné"
    assert resumed.called is False
    assert dm.peek_pending() is None
    assert daemon._global_db.record_action.call_args.kwargs["outcome"] == "décliné"


def test_a_click_while_a_turn_is_running_is_reported_busy(waiting):
    """Not queued, not dropped. The card stays up with its deadline
    running, because a decision the user correctly made must not vanish
    without a word."""
    dm, action = waiting
    daemon._chat_query_lock.acquire()
    try:
        outcome = daemon.resolve_confirmation(action.request_id, True)
    finally:
        daemon._chat_query_lock.release()

    assert outcome == "occupée"
    assert dm.peek_pending() is action


def test_a_click_during_shutdown_runs_nothing(waiting):
    dm, action = waiting
    daemon._global_stop_requested = True

    with patch.object(daemon, "_resume_after_confirmation") as resumed:
        outcome = daemon.resolve_confirmation(action.request_id, True)

    assert outcome == "arrêt"
    assert resumed.called is False
    assert dm.peek_pending() is action


def test_resolving_with_no_daemon_does_nothing(waiting):
    daemon._global_dialogue_memory = None

    assert daemon.resolve_confirmation("cf_x", True) == "inconnue"


# ── Announcing ────────────────────────────────────────────────────────


def test_a_question_is_announced_on_the_chat_bus(waiting, capsys):
    _, action = waiting

    daemon.announce_confirmation(action, origin="voix")

    lines = [
        l for l in capsys.readouterr().out.splitlines()
        if l.startswith(daemon.CHAT_IPC_PREFIX)
    ]
    event = json.loads(lines[-1][len(daemon.CHAT_IPC_PREFIX):])
    assert event["type"] == "confirm"
    assert event["data"]["request_id"] == action.request_id


def test_a_voice_question_reaches_the_chat_bus_too(waiting, capsys):
    """The chat window is the surface with buttons. A spoken question
    that never reached it would be a destructive action nobody can
    approve."""
    _, action = waiting

    daemon.announce_confirmation(action, origin="voix")

    assert daemon.CHAT_IPC_PREFIX in capsys.readouterr().out


def test_the_announcement_carries_what_the_card_must_show(waiting, capsys):
    _, action = waiting

    daemon.announce_confirmation(action, origin="voix")

    out = capsys.readouterr().out.splitlines()
    event = json.loads(
        [l for l in out if l.startswith(daemon.CHAT_IPC_PREFIX)][-1]
        [len(daemon.CHAT_IPC_PREFIX):]
    )
    data = event["data"]
    assert data["tool"] == "localFiles"
    assert data["risk"] == "destructif"
    assert "/a" in data["shown"]
    assert data["channel"] == CHANNEL_GESTE


def test_the_announcement_never_carries_unredacted_user_text(waiting, capsys):
    """`__CHAT__:` lines are logged. The query rides along redacted, as
    everywhere else on this bus."""
    dm = daemon._global_dialogue_memory
    dm.clear_pending()
    dm.begin_turn()
    action = PendingAction.create(
        tool="localFiles", args={"path": "/a"}, risk="destructif",
        channel=CHANNEL_GESTE, origin="chat",
        query_redacted="écris à hocsman92@gmail.com",
        raised_at_turn=dm.current_turn(), ttl_sec=180.0,
    )

    daemon.announce_confirmation(action, origin="chat")

    assert "hocsman92@gmail.com" not in capsys.readouterr().out


def test_a_settled_question_is_announced_as_settled(waiting, capsys):
    _, action = waiting
    capsys.readouterr()

    with patch.object(daemon, "_resume_after_confirmation"):
        daemon.resolve_confirmation(action.request_id, True)

    events = [
        json.loads(l[len(daemon.CHAT_IPC_PREFIX):])
        for l in capsys.readouterr().out.splitlines()
        if l.startswith(daemon.CHAT_IPC_PREFIX)
    ]
    assert any(e["type"] == "confirm_settled" for e in events)


# ── Shutdown ──────────────────────────────────────────────────────────


def test_shutdown_revokes_a_waiting_question(waiting):
    """A question nobody answered before the machine went down does not
    come back when it comes up."""
    dm, action = waiting

    daemon.revoke_pending_confirmation()

    assert dm.peek_pending() is None
    assert daemon._global_db.record_action.call_args.kwargs["outcome"] == "expiré"


# ── The narration of a confirmed action ───────────────────────────────


def _resume_with(monkeypatch, origin, reply="fait"):
    """Run the resume path for a question of the given origin, with the
    engine stubbed, and report what was spoken and what reached the bus."""
    dm = DialogueMemory()
    dm.begin_turn()
    action = PendingAction.create(
        tool="localFiles", args={"operation": "delete", "path": "/a"},
        risk="destructif", channel=CHANNEL_GESTE, origin=origin,
        query_redacted="supprime", raised_at_turn=dm.current_turn(),
        ttl_sec=180.0,
    )
    daemon._global_dialogue_memory = dm
    daemon._global_cfg = MagicMock()
    daemon._global_db = MagicMock()
    daemon._global_stop_requested = False
    monkeypatch.setattr("src.jarvis.reply.engine.run_reply_engine", lambda **kw: reply)
    spoken = []
    monkeypatch.setattr(
        daemon, "_speak_from_worker",
        lambda text, on_spoken=None: spoken.append(text) or True,
    )
    # The resume expects to own the lock resolve_confirmation took.
    assert daemon._chat_query_lock.acquire(blocking=False)
    daemon._resume_after_confirmation(action)
    return spoken


def _complete_events(capsys):
    return [
        json.loads(l[len(daemon.CHAT_IPC_PREFIX):])
        for l in capsys.readouterr().out.splitlines()
        if l.startswith(daemon.CHAT_IPC_PREFIX)
        and json.loads(l[len(daemon.CHAT_IPC_PREFIX):]).get("type") == "complete"
    ]


def test_a_confirmed_chat_action_is_narrated_without_speaking(monkeypatch, capsys):
    """Text chat never speaks, not even for the action it just approved."""
    capsys.readouterr()

    spoken = _resume_with(monkeypatch, origin="chat")

    assert spoken == []
    assert [e["data"] for e in _complete_events(capsys)] == ["fait"]
    assert not daemon._chat_query_lock.locked()


def test_a_confirmed_voice_action_is_spoken_once(monkeypatch, capsys):
    capsys.readouterr()

    spoken = _resume_with(monkeypatch, origin="voix")

    assert spoken == ["fait"]
    assert [e["data"] for e in _complete_events(capsys)] == ["fait"]


def test_in_bundled_mode_the_narration_reaches_the_wired_callback(monkeypatch):
    """Bundled mode has no stdout bus; the desktop wires a callback so the
    window learns what became of the action it approved."""
    heard = []
    daemon.set_confirmation_callbacks(on_confirm_reply=heard.append)
    try:
        _resume_with(monkeypatch, origin="chat")
    finally:
        daemon.set_confirmation_callbacks()

    assert heard == ["fait"]
