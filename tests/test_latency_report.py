"""`scripts/latency_report.py` reads the latency journal and prints durations.

The report is the reason the journal exists: it turns rows into "what did a
spoken exchange cost, and where". It prints durations and counts only, reads
a copy of the database so a running assistant is never disturbed, and says
plainly when there is nothing to report.
"""

from __future__ import annotations

import importlib.util
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
        "dispatch_ms": 6000, "route_ms": 800, "plan_ms": 2000, "memory_ms": 0,
        "tools_ms": 0, "generate_ms": 1000, "first_sound_ms": first_sound,
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


@pytest.mark.unit
def test_an_empty_journal_says_so_and_how_to_fill_it():
    report = _report()

    text = report.render(report.summarise([]), days=7)

    assert "no spoken exchange" in text.lower()
    assert "latency_journal_enabled" in text


@pytest.mark.unit
def test_main_reads_a_copy_and_leaves_the_database_alone(tmp_path, capsys):
    from jarvis.memory.db import Database

    report = _report()
    path = tmp_path / "jarvis.db"
    database = Database(str(path))
    for seconds in (5000, 9000):
        database.record_latency(_row(first_sound=seconds))
    old = _row(first_sound=99000, day_offset=40)
    database.record_latency(old)
    database.close()
    before = path.read_bytes()

    code = report.main(["--db", str(path), "--days", "7"])

    out = capsys.readouterr().out
    assert code == 0
    assert "2 spoken exchanges" in out                # the 40-day-old row is outside the window
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.copy*"))


@pytest.mark.unit
def test_main_without_a_journal_table_does_not_crash(tmp_path, capsys):
    import sqlite3

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
