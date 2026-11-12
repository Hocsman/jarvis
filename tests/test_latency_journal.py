"""The latency journal: how long a spoken exchange takes, stage by stage.

It keeps durations and enumerations and nothing else, so it can stay on by
default. These tests pin that property and the behaviour around it:

1. the table has no free-text column, and no text a caller hands the journal
   reaches it;
2. a streamed turn, a turn spoken after the engine returns, and a silent turn
   each come out as one row with the right numbers;
3. marks without a turn, a journal that is off and a database that fails cost
   the row and never the pipeline;
4. the settings, the retention and the report.

The clock is injected, so every number is exact.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class _Clock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _journal(db, clock, *, enabled: bool = True):
    from jarvis.latency import LatencyJournal

    cfg = SimpleNamespace(latency_journal_enabled=enabled)
    journal = LatencyJournal(clock=clock)
    journal.configure(db, cfg)
    return journal


def _only_row(db) -> dict:
    rows = db.recent_latency()
    assert len(rows) == 1, f"expected one row, found {len(rows)}"
    return rows[0]


def _a_streamed_turn(journal, clock):
    """Speech ends at 1000.0. Everything below is offsets from that."""
    clock.now = 1000.4
    journal.note_heard(1000.0)                    # transcript ready at +400 ms
    journal.note_judge(1.2, "directed")           # judge call took 1200 ms
    clock.now = 1006.1                            # query dispatched at +6100 ms
    journal.begin_turn({"judge": "judge-m", "router": "router-m", "chat": "chat-m"})
    journal.on_stage("routing")
    clock.now = 1006.9
    journal.on_stage("planning")                  # routing took 800 ms
    clock.now = 1009.3
    journal.on_stage("generating")                # planning took 2400 ms
    clock.now = 1010.3
    journal.mark_first_sound()                    # generating ran 1000 ms, sound at +10300 ms
    clock.now = 1011.0
    journal.on_stage("tool", "getWeather")        # after the sound: counted, not timed
    clock.now = 1012.0
    journal.end_turn()                            # reply complete at +12000 ms


# ── 1. Durations and enumerations, never text ─────────────────────────


@pytest.mark.unit
def test_the_table_has_no_free_text_column(db):
    """The property that lets the journal stay on by default. Text columns
    hold an enumerated outcome or a model identifier; everything else is a
    number."""
    columns = {
        row["name"]: row["type"].upper()
        for row in db.conn.execute("PRAGMA table_info(latency_log)").fetchall()
    }

    text_columns = {name for name, kind in columns.items() if kind == "TEXT"}
    assert text_columns == {
        "ts_utc", "origin", "judge_outcome", "judge_model", "router_model", "chat_model",
    }
    assert all(kind in ("TEXT", "INTEGER") for kind in columns.values())


@pytest.mark.unit
def test_no_text_a_caller_hands_the_journal_reaches_the_table(db):
    clock = _Clock()
    journal = _journal(db, clock)
    secret = "my private diagnosis"

    clock.now = 1000.2
    journal.note_heard(1000.0)
    journal.note_judge(0.5, secret)               # not an enumerated outcome
    clock.now = 1001.0
    journal.begin_turn({"chat": "chat-m"})
    journal.on_stage("tool", secret)              # a detail that carries text
    clock.now = 1002.0
    journal.end_turn()

    dump = json.dumps(db.recent_latency(), default=str)
    assert secret not in dump
    assert _only_row(db)["judge_outcome"] is None


# ── 2. One row per turn, with the right numbers ───────────────────────


@pytest.mark.unit
def test_a_streamed_turn_is_one_row_with_the_stages_it_waited_through(db):
    clock = _Clock()
    journal = _journal(db, clock)

    _a_streamed_turn(journal, clock)

    row = _only_row(db)
    assert row["origin"] == "voice"
    assert row["transcribed_ms"] == 400
    assert row["judge_ms"] == 1200
    assert row["judge_outcome"] == "directed"
    assert row["dispatch_ms"] == 6100
    assert (row["route_ms"], row["plan_ms"], row["memory_ms"]) == (800, 2400, 0)
    assert row["generate_ms"] == 1000
    assert row["tools_ms"] == 0                    # the tool ran after the first sound
    assert row["tool_calls"] == 1
    assert row["first_sound_ms"] == 10300
    assert row["total_ms"] == 12000
    assert row["router_degraded"] == 0
    assert (row["judge_model"], row["router_model"], row["chat_model"]) == (
        "judge-m", "router-m", "chat-m",
    )
    assert row["ts_utc"].startswith(
        datetime.fromtimestamp(1000.0, timezone.utc).isoformat()[:19]
    )


@pytest.mark.unit
def test_a_turn_spoken_after_the_engine_returns_gets_its_first_sound_later(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1002.0
    journal.begin_turn({})
    journal.on_stage("generating")
    clock.now = 1005.0
    journal.end_turn()                             # nothing spoken yet

    assert _only_row(db)["first_sound_ms"] is None
    assert _only_row(db)["generate_ms"] == 3000    # all of it preceded the sound

    clock.now = 1006.5
    journal.mark_first_sound()
    assert _only_row(db)["first_sound_ms"] == 6500


@pytest.mark.unit
def test_a_sound_long_after_the_turn_is_not_this_turns(db):
    """A reminder or a routine speaking later must not be read as the reply."""
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1002.0
    journal.end_turn()

    clock.now = 1002.0 + 31.0
    journal.mark_first_sound()

    assert _only_row(db)["first_sound_ms"] is None


@pytest.mark.unit
def test_only_the_first_sound_of_a_turn_counts(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1002.0
    journal.mark_first_sound()
    clock.now = 1003.0
    journal.mark_first_sound()                     # the next sentence of the same reply
    clock.now = 1004.0
    journal.end_turn()
    clock.now = 1005.0
    journal.mark_first_sound()                     # and a late one

    assert _only_row(db)["first_sound_ms"] == 2000


@pytest.mark.unit
def test_a_silent_turn_keeps_all_its_stage_time(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    journal.on_stage("planning")
    clock.now = 1003.0
    journal.on_stage("memory")
    clock.now = 1004.5
    journal.on_stage("tool", "webSearch")
    clock.now = 1006.5
    journal.on_stage("tool", "webSearch")
    clock.now = 1007.5
    journal.end_turn()

    row = _only_row(db)
    assert row["first_sound_ms"] is None
    assert (row["plan_ms"], row["memory_ms"]) == (2000, 1500)
    assert row["tools_ms"] == 3000                 # two tool stages, added up
    assert row["tool_calls"] == 2
    assert row["total_ms"] == 7500


@pytest.mark.unit
def test_the_judge_that_could_not_answer_is_recorded_as_such(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    journal.note_judge(5.0, "no_verdict")
    clock.now = 1006.0
    journal.begin_turn({})
    journal.end_turn()

    row = _only_row(db)
    assert row["judge_outcome"] == "no_verdict"
    assert row["judge_ms"] == 5000


@pytest.mark.unit
def test_a_turn_with_no_judge_call_leaves_the_judge_columns_empty(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    journal.end_turn()

    row = _only_row(db)
    assert row["judge_ms"] is None and row["judge_outcome"] is None


@pytest.mark.unit
def test_a_router_that_gave_up_is_flagged(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    journal.note_router_degraded()
    journal.end_turn()

    assert _only_row(db)["router_degraded"] == 1


@pytest.mark.unit
def test_the_last_utterance_heard_is_the_one_measured(db):
    """Two utterances inside the collection window: the silence after the
    second is what the user waited through."""
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1003.3
    journal.note_heard(1003.0)
    clock.now = 1008.0
    journal.begin_turn({})
    clock.now = 1009.0
    journal.end_turn()

    row = _only_row(db)
    assert row["transcribed_ms"] == 300
    assert row["dispatch_ms"] == 5000
    assert row["total_ms"] == 6000


@pytest.mark.unit
def test_a_turn_does_not_inherit_the_notes_of_the_one_before(db):
    clock = _Clock()
    journal = _journal(db, clock)
    _a_streamed_turn(journal, clock)

    clock.now = 2000.0
    journal.begin_turn({})                         # dispatched with no new speech heard
    clock.now = 2001.0
    journal.end_turn()

    newest = db.recent_latency()[0]
    assert newest["transcribed_ms"] is None
    assert newest["judge_ms"] is None
    assert newest["first_sound_ms"] is None


# ── 3. Never the pipeline's problem ───────────────────────────────────


@pytest.mark.unit
def test_marks_without_an_open_turn_do_nothing(db):
    clock = _Clock()
    journal = _journal(db, clock)

    journal.on_stage("generating")
    journal.note_router_degraded()
    journal.mark_first_sound()
    journal.end_turn()

    assert db.recent_latency() == []


@pytest.mark.unit
def test_a_journal_that_is_off_writes_nothing(db):
    clock = _Clock()
    journal = _journal(db, clock, enabled=False)

    _a_streamed_turn(journal, clock)

    assert db.recent_latency() == []


@pytest.mark.unit
def test_a_journal_nobody_configured_writes_nothing_and_does_not_raise():
    from jarvis.latency import LatencyJournal

    journal = LatencyJournal(clock=_Clock())
    journal.note_heard(1.0)
    journal.begin_turn({})
    journal.on_stage("generating")
    journal.mark_first_sound()
    journal.end_turn()


@pytest.mark.unit
def test_a_database_that_fails_costs_the_row_and_not_the_turn():
    clock = _Clock()
    broken = MagicMock()
    broken.record_latency.side_effect = RuntimeError("disk full")
    broken.set_latency_first_sound.side_effect = RuntimeError("disk full")
    journal = _journal(broken, clock)

    _a_streamed_turn(journal, clock)               # must not raise


@pytest.mark.unit
def test_a_turn_left_open_is_replaced_by_the_next(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})                         # the engine never returned
    clock.now = 1100.3
    journal.note_heard(1100.0)
    clock.now = 1101.0
    journal.begin_turn({})
    clock.now = 1102.0
    journal.end_turn()

    assert _only_row(db)["ts_utc"].startswith(
        datetime.fromtimestamp(1100.0, timezone.utc).isoformat()[:19]
    )


# ── 4. Retention, settings, report ────────────────────────────────────


@pytest.mark.unit
def test_old_rows_are_dropped_and_recent_ones_kept(db):
    now = datetime.now(timezone.utc)
    row = {"ts_utc": (now - timedelta(days=40)).isoformat(), "origin": "voice"}
    db.record_latency(dict(row))
    db.record_latency({**row, "ts_utc": (now - timedelta(days=2)).isoformat()})

    dropped = db.prune_latency(30)

    assert dropped == 1
    assert len(db.recent_latency()) == 1


@pytest.mark.unit
def test_the_journal_can_be_erased(db):
    db.record_latency({"ts_utc": datetime.now(timezone.utc).isoformat(), "origin": "voice"})

    db.clear_latency()

    assert db.recent_latency() == []


def _settings_from(tmp_path, monkeypatch, values: dict):
    from jarvis.config import load_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


@pytest.mark.unit
def test_the_journal_is_on_by_default_and_keeps_a_month(tmp_path, monkeypatch):
    settings = _settings_from(tmp_path, monkeypatch, {})

    assert settings.latency_journal_enabled is True
    assert 1 <= settings.latency_journal_days <= 365


@pytest.mark.unit
def test_the_loader_reads_the_journal_settings(tmp_path, monkeypatch):
    settings = _settings_from(
        tmp_path, monkeypatch, {"latency_journal_enabled": False, "latency_journal_days": 7},
    )

    assert settings.latency_journal_enabled is False
    assert settings.latency_journal_days == 7


@pytest.mark.unit
@pytest.mark.parametrize("asked", [0, -5, 100000])
def test_the_retention_stays_inside_the_window_the_settings_offer(tmp_path, monkeypatch, asked):
    from desktop_app.settings_window import FIELD_METADATA

    field = next(fm for fm in FIELD_METADATA if fm.key == "latency_journal_days")
    settings = _settings_from(tmp_path, monkeypatch, {"latency_journal_days": asked})

    assert field.min_val <= settings.latency_journal_days <= field.max_val


@pytest.mark.unit
def test_the_settings_window_offers_both_journal_settings():
    from desktop_app.settings_window import FIELD_METADATA

    keys = {fm.key for fm in FIELD_METADATA}
    assert {"latency_journal_enabled", "latency_journal_days"} <= keys


# ── 5. The speech engines mark the sound, not the synthesis ──────────


def _journal_with_an_open_turn(db, clock):
    from jarvis.latency import get_latency_journal

    journal = get_latency_journal()
    journal.reset(clock=clock)
    journal.configure(db, SimpleNamespace(latency_journal_enabled=True))
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    return journal


@pytest.mark.unit
def test_the_audio_stream_starting_marks_the_first_sound(db):
    from jarvis.output import tts as tts_module

    clock = _Clock()
    journal = _journal_with_an_open_turn(db, clock)
    try:
        clock.now = 1003.0
        tts_module._sound_started()
        journal.end_turn()

        assert _only_row(db)["first_sound_ms"] == 3000
    finally:
        journal.reset()


@pytest.mark.unit
@pytest.mark.parametrize("engine", ["ChatterboxTTS", "PiperTTS", "KokoroTTS", "KyutaiTTS"])
def test_publishing_the_speaking_state_is_not_the_first_sound(db, engine):
    """`SPEAKING` is published when a chunk begins to be synthesised, before
    anything is audible; marking there would understate the wait."""
    from jarvis.output import tts as tts_module

    clock = _Clock()
    journal = _journal_with_an_open_turn(db, clock)
    try:
        clock.now = 1002.0
        getattr(tts_module, engine)._notify_speaking_state(SimpleNamespace(), True)
        journal.end_turn()

        assert _only_row(db)["first_sound_ms"] is None
    finally:
        journal.reset()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("engine", "playback_starts", "outside_the_locks"),
    [
        ("ChatterboxTTS", "pygame.mixer.music.play()", False),   # no audio lock in that engine
        ("PiperTTS", "self._audio_stream.start()", True),
        ("KokoroTTS", "self._audio_stream.start()", True),
        ("KyutaiTTS", "self._audio_stream.start()", True),
    ],
)
def test_each_speech_engine_marks_the_sound_once_playback_has_started(
    engine, playback_starts, outside_the_locks
):
    """The call follows the moment the stream starts, and where the stream is
    started under audio locks it sits outside them (a journal write must never
    hold playback up). Indentation is what tells the two apart."""
    import inspect

    from jarvis.output import tts as tts_module

    lines = inspect.getsource(getattr(tts_module, engine)).splitlines()
    start_at = next(i for i, l in enumerate(lines) if playback_starts in l)
    call_at = next(i for i, l in enumerate(lines) if l.strip() == "_sound_started()")

    def indent(line):
        return len(line) - len(line.lstrip())

    assert start_at < call_at
    if outside_the_locks:
        assert indent(lines[call_at]) < indent(lines[start_at])
    else:
        assert indent(lines[call_at]) <= indent(lines[start_at])


# ── 6. Time that belongs to no engine stage ───────────────────────────


@pytest.mark.unit
def test_time_before_the_engines_first_stage_is_booked_as_preparation(db):
    """Redaction, the MCP refresh, the approval judge for a spoken answer and
    the wait for the query lock all come before the first announcement."""
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1004.0
    journal.on_stage("routing")                    # 3 s went to preparation
    clock.now = 1004.5
    journal.end_turn()

    row = _only_row(db)
    assert row["prepare_ms"] == 3000
    assert row["route_ms"] == 500


@pytest.mark.unit
def test_a_turn_that_never_reached_a_stage_is_all_preparation(db):
    """A spoken refusal returns before the engine announces anything."""
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1003.5
    journal.end_turn()

    assert _only_row(db)["prepare_ms"] == 2500


# ── 7. Which sound belongs to which turn ──────────────────────────────


def _a_turn(journal, clock, *, start, sound_at=None, will_speak=True):
    clock.now = start + 0.3
    journal.note_heard(start)
    clock.now = start + 1.0
    journal.begin_turn({})
    if sound_at is not None:
        clock.now = sound_at
        journal.mark_first_sound()
    clock.now = (sound_at or start + 1.0) + 1.0
    journal.end_turn(will_speak=will_speak)


@pytest.mark.unit
def test_a_turn_whose_reply_will_not_be_spoken_takes_no_late_sound(db):
    """A dismissal returns nothing to say; a reminder a few seconds later is
    not its first sound."""
    clock = _Clock()
    journal = _journal(db, clock)
    _a_turn(journal, clock, start=1000.0, will_speak=False)

    clock.now = 1008.0
    journal.mark_first_sound()

    assert _only_row(db)["first_sound_ms"] is None


@pytest.mark.unit
def test_a_sound_of_a_later_reply_does_not_fill_an_earlier_silent_row(db):
    """Turn A got no sound; turn B streams and its tail is spoken after B's row
    is written. That tail must not be written into A's row."""
    clock = _Clock()
    journal = _journal(db, clock)
    _a_turn(journal, clock, start=1000.0)                          # silent, armed
    _a_turn(journal, clock, start=1008.0, sound_at=1010.0)         # streamed

    clock.now = 1014.0                                             # B's tail
    journal.mark_first_sound()

    rows = {r["ts_utc"][:19]: r for r in db.recent_latency()}
    first, second = sorted(rows)
    assert rows[first]["first_sound_ms"] is None
    assert rows[second]["first_sound_ms"] == 2000


# ── 8. Writing after the reply is queued ──────────────────────────────


@pytest.mark.unit
def test_finishing_a_turn_does_no_io_and_flushing_writes_it(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1002.0
    journal.finish_turn(will_speak=True)

    assert db.recent_latency() == []               # nothing written yet

    journal.flush()
    assert len(db.recent_latency()) == 1
    journal.flush()                                # idempotent
    assert len(db.recent_latency()) == 1


@pytest.mark.unit
def test_a_sound_between_finishing_and_flushing_lands_in_the_row(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1002.0
    journal.finish_turn(will_speak=True)
    clock.now = 1002.5
    journal.mark_first_sound()
    journal.flush()

    assert _only_row(db)["first_sound_ms"] == 2500


@pytest.mark.unit
def test_the_next_turn_flushes_a_row_nobody_flushed(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 1002.0
    journal.finish_turn(will_speak=False)

    clock.now = 1010.3
    journal.note_heard(1010.0)
    clock.now = 1011.0
    journal.begin_turn({})

    assert len(db.recent_latency()) == 1


# ── 9. Marks never raise, durations never go negative ─────────────────


@pytest.mark.unit
def test_no_mark_raises_even_when_the_clock_does(db):
    def broken_clock():
        raise RuntimeError("clock failed")

    from jarvis.latency import LatencyJournal

    journal = LatencyJournal(clock=broken_clock)
    journal.configure(db, SimpleNamespace(latency_journal_enabled=True))

    journal.note_heard(1000.0)
    journal.note_judge(1.0, "directed")
    journal.begin_turn({})
    journal.on_stage("routing")
    journal.note_router_degraded()
    journal.mark_first_sound()
    journal.end_turn()
    journal.flush()


@pytest.mark.unit
def test_a_clock_that_steps_back_never_gives_a_negative_duration(db):
    clock = _Clock()
    journal = _journal(db, clock)
    clock.now = 1000.3
    journal.note_heard(1000.0)
    clock.now = 1001.0
    journal.begin_turn({})
    clock.now = 990.0                              # the clock was set back mid-turn
    journal.mark_first_sound()
    journal.end_turn()

    row = _only_row(db)
    assert row["first_sound_ms"] >= 0 and row["total_ms"] >= 0


# ── 10. The table refuses text, whoever writes it ─────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("column,value", [
    ("judge_outcome", "my private diagnosis"),
    ("origin", "my private diagnosis"),
    ("chat_model", "x" * 500),
])
def test_the_table_itself_refuses_text_it_was_not_built_for(db, column, value):
    row = {"ts_utc": datetime.now(timezone.utc).isoformat(), "origin": "voice", column: value}

    stored = db.record_latency(row)

    assert stored is None
    assert db.recent_latency() == []


@pytest.mark.unit
def test_a_database_written_before_the_journal_gains_the_table_and_keeps_its_rows(tmp_path):
    import sqlite3

    from jarvis.memory.db import Database

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE action_log (id INTEGER PRIMARY KEY, ts_utc TEXT NOT NULL, origin TEXT, "
                "tool TEXT NOT NULL, args TEXT, risk TEXT NOT NULL, verdict TEXT NOT NULL, "
                "outcome TEXT NOT NULL, duration_ms INTEGER, query TEXT, request_id TEXT)")
    con.execute("INSERT INTO action_log (ts_utc, tool, risk, verdict, outcome) "
                "VALUES ('2026-10-01T10:00:00+00:00', 'getWeather', 'lecture', 'libre', 'ok')")
    con.execute("PRAGMA user_version = 5")
    con.commit()
    con.close()

    database = Database(str(path))
    try:
        assert database.record_latency({"ts_utc": datetime.now(timezone.utc).isoformat(),
                                        "origin": "voice"}) is not None
        assert [a["tool"] for a in database.recent_actions()] == ["getWeather"]
    finally:
        database.close()


@pytest.mark.unit
def test_a_hand_written_false_in_quotes_really_turns_the_journal_off(tmp_path, monkeypatch):
    for spelling in ("false", "False", "0", "off", "no"):
        settings = _settings_from(tmp_path, monkeypatch, {"latency_journal_enabled": spelling})
        assert settings.latency_journal_enabled is False, spelling


@pytest.mark.unit
def test_turning_the_journal_off_erases_what_it_kept_at_the_next_start(db):
    from jarvis.daemon import _apply_latency_retention

    db.record_latency({"ts_utc": datetime.now(timezone.utc).isoformat(), "origin": "voice"})

    _apply_latency_retention(
        db, SimpleNamespace(latency_journal_enabled=False, latency_journal_days=30))

    assert db.recent_latency() == []
