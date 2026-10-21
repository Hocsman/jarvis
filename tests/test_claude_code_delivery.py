"""The report of a Claude Code job reaches the user, in both builds.

A job ends minutes after the turn that started it, so its report arrives
with no turn of its own. It has to land in the conversation (so a
follow-up can refer to it), on both buses (a surface wired to one of them
exists in exactly half the builds), and in the user's ear unless the task
was typed. And it must not cut across a reply the user is waiting for.

See src/jarvis/projets/projets.spec.md, *Delivering the report*.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from src.jarvis import daemon
from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.projets.delegation import Job, Outcome


@pytest.fixture(autouse=True)
def wired(monkeypatch):
    daemon.set_confirmation_callbacks()
    dm = DialogueMemory()
    spoken = []
    monkeypatch.setattr(daemon, "_global_dialogue_memory", dm)
    monkeypatch.setattr(daemon, "_speak_from_worker",
                        lambda text, on_spoken=None: spoken.append(text) or True)
    yield dm, spoken
    daemon.set_confirmation_callbacks()


def _outcome(origin="voix", ok=True, report="J'ai ajouté le bouton."):
    job = Job(
        id="job1", projet="chantipay", path=Path("/p/chantipay"),
        tache="Ajoute un bouton.", modifier=True, origin=origin,
        risk="destructif", verdict="demande", started=time.monotonic(),
    )
    return Outcome(job=job, ok=ok, report=report, session_id=None, duration_s=12.0)


def _events(capsys, kind):
    out = capsys.readouterr().out
    return [
        json.loads(line[len(daemon.CHAT_IPC_PREFIX):])
        for line in out.splitlines()
        if line.startswith(daemon.CHAT_IPC_PREFIX)
        and json.loads(line[len(daemon.CHAT_IPC_PREFIX):]).get("type") == kind
    ]


def test_the_report_joins_the_conversation(wired):
    dm, _ = wired

    daemon.deliver_delegation(_outcome())

    (message,) = dm.get_recent_messages()
    assert message["role"] == "assistant"
    assert "chantipay" in message["content"]
    assert "J'ai ajouté le bouton." in message["content"]


def test_the_report_reaches_the_stdout_bus(wired, capsys):
    daemon.deliver_delegation(_outcome())

    (event,) = _events(capsys, "delegation_done")
    assert event["data"]["projet"] == "chantipay"
    assert event["data"]["ok"] is True
    assert "J'ai ajouté le bouton." in event["data"]["texte"]


def test_the_report_reaches_a_bundled_build_too(wired):
    told = []
    daemon.set_confirmation_callbacks(on_delegation_done=told.append)

    daemon.deliver_delegation(_outcome())

    (data,) = told
    assert "J'ai ajouté le bouton." in data["texte"]


def test_a_spoken_task_gets_a_spoken_report(wired):
    _, spoken = wired

    daemon.deliver_delegation(_outcome(origin="voix"))

    (said,) = spoken
    assert "J'ai ajouté le bouton." in said


def test_a_typed_task_is_never_answered_aloud(wired):
    _, spoken = wired

    daemon.deliver_delegation(_outcome(origin="chat"))

    assert spoken == []


def test_it_waits_for_the_reply_in_flight(wired, monkeypatch):
    """Cutting across a reply the user is waiting for is worse than a
    few seconds late."""
    dm, spoken = wired
    lock = threading.Lock()
    monkeypatch.setattr(daemon, "_chat_query_lock", lock)
    lock.acquire()
    done = threading.Event()
    threading.Thread(
        target=lambda: (daemon.deliver_delegation(_outcome()), done.set()),
        daemon=True,
    ).start()

    time.sleep(0.6)
    assert spoken == [] and dm.get_recent_messages() == []

    lock.release()
    assert done.wait(5.0)
    assert len(spoken) == 1


def test_nothing_is_delivered_once_yuba_is_stopping(wired, monkeypatch):
    """Nothing is left to hear it, and the shutdown diary pass is about to
    read the memory it would land in."""
    dm, spoken = wired
    monkeypatch.setattr(daemon, "_global_stop_requested", True)

    daemon.deliver_delegation(_outcome())

    assert spoken == [] and dm.get_recent_messages() == []


def test_the_bundled_build_wires_the_report_to_the_chat_window():
    """Both routes, or the report exists in exactly half the builds."""
    import inspect

    from src.desktop_app import app as desktop

    source = inspect.getsource(desktop)
    assert "on_delegation_done=" in source
    assert '_chat_event_line("delegation_done"' in source


# ── A stop question outlives nothing ──────────────────────────────────


def _question(dm, tool, args):
    from src.jarvis.tools.confirmation import CHANNEL_GESTE, PendingAction

    dm.begin_turn()
    action = PendingAction.create(
        tool=tool, args=args, risk="action", channel=CHANNEL_GESTE,
        origin="voix", query_redacted="arrête", raised_at_turn=dm.current_turn(),
        ttl_sec=180.0,
    )
    dm.raise_pending(action)
    return action


def test_a_stop_question_still_waiting_when_the_job_ends_is_withdrawn(
        wired, monkeypatch, capsys):
    """Approved later, it would stop whatever runs then: a job the user
    never saw a question about."""
    from unittest.mock import MagicMock

    from src.jarvis.projets.delegation import STOP_TOOL_NAME

    dm, _ = wired
    db = MagicMock()
    monkeypatch.setattr(daemon, "_global_db", db)
    action = _question(dm, STOP_TOOL_NAME, {})

    daemon.deliver_delegation(_outcome())

    assert dm.peek_pending() is None
    assert db.record_action.call_args.kwargs["outcome"] == "expiré"
    assert db.record_action.call_args.kwargs["request_id"] == action.request_id
    (settled,) = _events(capsys, "confirm_settled")
    assert settled["data"] == {"request_id": action.request_id, "outcome": "expiré"}


def test_a_withdrawn_stop_question_reaches_a_bundled_build_too(wired, monkeypatch):
    from unittest.mock import MagicMock

    from src.jarvis.projets.delegation import STOP_TOOL_NAME

    dm, _ = wired
    monkeypatch.setattr(daemon, "_global_db", MagicMock())
    told = []
    daemon.set_confirmation_callbacks(
        on_confirm_settled=lambda request_id, outcome: told.append((request_id, outcome)))
    action = _question(dm, STOP_TOOL_NAME, {})

    daemon.delegation_changed(_outcome().job)

    assert told == [(action.request_id, "expiré")]


def test_a_stop_question_waiting_when_the_running_job_changes_is_withdrawn(
        wired, monkeypatch):
    """A job starting or ending: any stop question still waiting was about
    another, or about nothing, and approved later it would stop the wrong
    one."""
    from unittest.mock import MagicMock

    from src.jarvis.projets.delegation import STOP_TOOL_NAME

    dm, _ = wired
    db = MagicMock()
    monkeypatch.setattr(daemon, "_global_db", db)
    _question(dm, STOP_TOOL_NAME, {})

    daemon.delegation_changed(_outcome().job)

    assert dm.peek_pending() is None
    assert db.record_action.call_args.kwargs["outcome"] == "expiré"


def test_a_stop_question_raised_while_the_report_waits_is_withdrawn(
        wired, monkeypatch):
    """The reply in flight when the job ended was offered the stop, and may
    ask about a job that no longer runs."""
    from unittest.mock import MagicMock

    from src.jarvis.projets.delegation import STOP_TOOL_NAME

    dm, _ = wired
    monkeypatch.setattr(daemon, "_global_db", MagicMock())
    lock = threading.Lock()
    monkeypatch.setattr(daemon, "_chat_query_lock", lock)
    lock.acquire()
    done = threading.Event()
    threading.Thread(
        target=lambda: (daemon.deliver_delegation(_outcome()), done.set()),
        daemon=True,
    ).start()
    time.sleep(0.3)
    _question(dm, STOP_TOOL_NAME, {})

    lock.release()

    assert done.wait(5.0)
    assert dm.peek_pending() is None


def test_any_other_question_is_left_waiting(wired, monkeypatch):
    from unittest.mock import MagicMock

    dm, _ = wired
    monkeypatch.setattr(daemon, "_global_db", MagicMock())
    action = _question(dm, "localFiles", {"operation": "delete", "path": "/a"})

    daemon.deliver_delegation(_outcome())

    assert dm.peek_pending() == action
