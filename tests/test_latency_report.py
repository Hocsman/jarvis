"""`scripts/latency_report.py` reads the latency journal and prints durations.

The report is the reason the journal exists: it turns rows into "what did a
spoken exchange cost, and where". It prints durations and counts only, opens
the database read-only so a running assistant is never disturbed and nothing
of the rest of the database is copied, says plainly when there is nothing to
report, and can erase the journal.
"""

from __future__ import annotations

import importlib.util
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT = REPO_ROOT / "scripts" / "latency_report.py"


def _report():
    spec = importlib.util.spec_from_file_location("latency_report", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(first_sound=8000, *, day_offset=0, **overrides):
    when = datetime.now(timezone.utc) - timedelta(days=day_offset)
    row = {
        "ts_utc": when.isoformat(), "origin": "voice",
        "transcribed_ms": 400, "judge_ms": 1200, "judge_outcome": "directed",
        "dispatch_ms": 6000, "prepare_ms": 100, "route_ms": 800, "plan_ms": 2000,
        "memory_ms": 0, "tools_ms": 0, "generate_ms": 1000, "first_sound_ms": first_sound,
        "total_ms": (first_sound or 5000) + 3000, "router_degraded": 0, "tool_calls": 0,
    }
    row.update(overrides)
    return row


@pytest.mark.unit
def test_the_summary_gives_median_p90_and_max_of_each_stage():
    report = _report()
    rows = [_row(first_sound=s) for s in (4000, 6000, 8000, 10000, 20000)]

    summary = report.summarise(rows)

    assert summary["count"] == 5
    first = summary["stages"]["first_sound_ms"]
    assert (first["n"], first["median"], first["max"]) == (5, 8000, 20000)
    assert first["p90"] == 20000                    # nearest rank over five turns


@pytest.mark.unit
def test_the_wait_for_the_user_is_what_dispatch_has_that_recognition_and_judge_do_not():
    report = _report()

    summary = report.summarise([_row(dispatch_ms=6000, transcribed_ms=400, judge_ms=1200)])

    assert summary["stages"]["wait_ms"]["median"] == 4400


@pytest.mark.unit
def test_preparation_is_a_stage_of_its_own():
    report = _report()

    summary = report.summarise([_row(prepare_ms=2500)])

    assert summary["stages"]["prepare_ms"]["median"] == 2500


@pytest.mark.unit
def test_a_reply_spoken_after_the_engine_returned_shows_when_speech_started():
    """The first sound of a late row comes after the reply text is complete;
    the difference is the time speech took to start (synthesis, queue)."""
    report = _report()
    late = _row(first_sound=9000, total_ms=7500)           # sound 1.5 s after the text
    streamed = _row(first_sound=4000, total_ms=7000)       # sound before the text was done

    summary = report.summarise([late, streamed])

    start = summary["stages"]["speech_start_ms"]
    assert (start["n"], start["median"]) == (1, 1500)


@pytest.mark.unit
def test_an_empty_stage_is_left_out_rather_than_counted_as_zero():
    report = _report()
    rows = [_row(first_sound=None), _row(first_sound=6000)]

    summary = report.summarise(rows)

    assert summary["count"] == 2
    assert summary["stages"]["first_sound_ms"]["n"] == 1
    assert summary["stages"]["first_sound_ms"]["median"] == 6000


@pytest.mark.unit
def test_it_counts_how_often_the_router_gave_up_and_what_the_judge_said():
    report = _report()
    rows = [
        _row(router_degraded=1, judge_outcome="directed"),
        _row(router_degraded=0, judge_outcome="no_verdict"),
        _row(router_degraded=0, judge_outcome="not_directed"),
        _row(router_degraded=1, judge_outcome="directed"),
    ]

    summary = report.summarise(rows)

    assert summary["router_degraded"] == 2
    assert summary["judge_outcomes"] == {"directed": 2, "no_verdict": 1, "not_directed": 1}


@pytest.mark.unit
def test_it_gives_one_line_per_day():
    report = _report()
    rows = [_row(first_sound=6000, day_offset=0), _row(first_sound=8000, day_offset=0),
            _row(first_sound=10000, day_offset=2)]

    summary = report.summarise(rows)

    assert [d["count"] for d in summary["days"]] == [1, 2]      # oldest first
    assert summary["days"][1]["first_sound_median"] == 7000


@pytest.mark.unit
def test_the_rendering_names_the_numbers_and_nothing_else():
    report = _report()
    text = report.render(report.summarise([_row(), _row(first_sound=12000)]), days=7)

    assert "2 spoken exchanges" in text
    assert "first sound" in text.lower()
    assert "p90" in text.lower()
    assert "preparation" in text.lower()


@pytest.mark.unit
def test_an_empty_journal_says_so_and_how_to_fill_it():
    report = _report()

    text = report.render(report.summarise([]), days=7)

    assert "no spoken exchange" in text.lower()
    assert "latency_journal_enabled" in text


def _journal_database(tmp_path, rows):
    from jarvis.memory.db import Database

    path = tmp_path / "jarvis.db"
    database = Database(str(path))
    for row in rows:
        assert database.record_latency(row) is not None
    return path, database


@pytest.mark.unit
def test_main_reads_the_live_database_read_only_and_copies_nothing(tmp_path, capsys, monkeypatch):
    """The assistant is running: its writer connection is open and the newest
    rows sit in the write-ahead log. The report must read them in place,
    without copying a database that also holds the diary and the memory."""
    import shutil
    import tempfile

    report = _report()
    path, database = _journal_database(tmp_path, [
        _row(first_sound=5000), _row(first_sound=9000), _row(first_sound=99000, day_offset=40),
    ])
    try:
        before = path.read_bytes()

        def forbidden(*args, **kwargs):
            raise AssertionError("the report must not copy the database")

        monkeypatch.setattr(shutil, "copy2", forbidden)
        monkeypatch.setattr(shutil, "copy", forbidden)
        monkeypatch.setattr(shutil, "copyfile", forbidden)
        monkeypatch.setattr(tempfile, "TemporaryDirectory", forbidden)

        code = report.main(["--db", str(path), "--days", "7"])

        out = capsys.readouterr().out
        assert code == 0
        assert "2 spoken exchanges" in out                 # the 40-day-old row is outside the window
        assert path.read_bytes() == before
    finally:
        database.close()


@pytest.mark.unit
def test_main_refuses_to_write_through_the_reading_connection(tmp_path):
    report = _report()
    path, database = _journal_database(tmp_path, [_row()])
    try:
        con = report._open_read_only(path)
        with pytest.raises(sqlite3.OperationalError):
            con.execute("DELETE FROM latency_log")
        con.close()
    finally:
        database.close()


@pytest.mark.unit
def test_main_without_a_journal_table_does_not_crash(tmp_path, capsys):
    report = _report()
    path = tmp_path / "old.db"
    sqlite3.connect(path).close()

    code = report.main(["--db", str(path)])

    assert code == 0
    assert "no spoken exchange" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_main_with_a_missing_database_says_so(tmp_path, capsys):
    report = _report()

    code = report.main(["--db", str(tmp_path / "nope.db")])

    assert code == 1
    assert "not found" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_main_with_something_that_is_not_a_database_says_so_without_a_traceback(tmp_path, capsys):
    report = _report()
    path = tmp_path / "notes.db"
    path.write_text("this is not a database " * 50, encoding="utf-8")

    code = report.main(["--db", str(path)])

    assert code == 1
    assert "cannot read" in capsys.readouterr().out.lower()


@pytest.mark.unit
def test_clear_erases_the_journal_and_nothing_else(tmp_path, capsys):
    report = _report()
    path, database = _journal_database(tmp_path, [_row(), _row(first_sound=9000)])
    database.conn.execute(
        "INSERT INTO action_log (ts_utc, tool, risk, verdict, outcome) "
        "VALUES ('2026-10-01T10:00:00+00:00', 'getWeather', 'lecture', 'libre', 'ok')")
    database.conn.commit()
    database.close()

    code = report.main(["--db", str(path), "--clear"])

    out = capsys.readouterr().out
    assert code == 0
    assert "2" in out
    con = sqlite3.connect(path)
    assert con.execute("SELECT COUNT(*) FROM latency_log").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM action_log").fetchone()[0] == 1
    con.close()
