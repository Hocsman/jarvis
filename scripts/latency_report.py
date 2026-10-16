#!/usr/bin/env python3
"""Print how long spoken exchanges took, from the latency journal.

Usage:
    python scripts/latency_report.py [--days N] [--db PATH]
    python scripts/latency_report.py --clear [--db PATH]

Prints, for the voice turns of the last N days (default 7), the count, median,
90th percentile and maximum of each stage and of the first sound, one line per
day, and how often the tool router gave up and what the intent judge said. It
prints durations and counts only: the journal holds nothing else (see
src/jarvis/latency.spec.md).

The database is opened read-only, in place, so a running assistant is never
disturbed and nothing of the rest of it (the diary, the memory) is copied.
`--clear` erases the journal and nothing else.

Run it from a source checkout with the project's environment active, or pass
`--db` with the path of the database.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import statistics
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# (column or derived name, label). Two are derived:
#   wait_ms        what a query spent between the end of speech and the dispatch
#                  beyond recognition and the judge, chiefly the time given to
#                  the user to carry on;
#   speech_start_ms  for a reply spoken after the engine returned, how long
#                  speech took to start once the text was complete (synthesis,
#                  queue).
STAGES = (
    ("transcribed_ms", "recognition"),
    ("judge_ms", "intent judge"),
    ("wait_ms", "wait to carry on"),
    ("dispatch_ms", "to dispatch (all)"),
    ("prepare_ms", "preparation"),
    ("route_ms", "tool router"),
    ("plan_ms", "planner"),
    ("memory_ms", "memory"),
    ("tools_ms", "tools"),
    ("generate_ms", "reply (to 1st sound)"),
    ("speech_start_ms", "speech start (late)"),
    ("first_sound_ms", "FIRST SOUND"),
    ("total_ms", "reply complete"),
)

# What the report reads. Nothing else is selected.
COLUMNS = (
    "ts_utc", "transcribed_ms", "judge_ms", "judge_outcome", "dispatch_ms",
    "prepare_ms", "route_ms", "plan_ms", "memory_ms", "tools_ms", "generate_ms",
    "first_sound_ms", "total_ms", "router_degraded",
)


def _percentile(values: List[int], fraction: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _derived(key: str, row: Dict[str, Any]) -> Optional[int]:
    if key == "wait_ms":
        dispatch, heard = row.get("dispatch_ms"), row.get("transcribed_ms")
        if dispatch is None or heard is None:
            return None
        return max(0, dispatch - heard - (row.get("judge_ms") or 0))
    if key == "speech_start_ms":
        first, total = row.get("first_sound_ms"), row.get("total_ms")
        if first is None or total is None or first <= total:
            return None
        return first - total
    return row.get(key)


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Medians, percentiles, counts: numbers about the rows, never their text."""
    stages: Dict[str, Dict[str, int]] = {}
    for key, _label in STAGES:
        values = [v for v in (_derived(key, r) for r in rows) if v is not None]
        if values:
            stages[key] = {
                "n": len(values),
                "median": int(statistics.median(values)),
                "p90": _percentile(values, 0.9),
                "max": max(values),
            }

    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        day = datetime.fromisoformat(row["ts_utc"]).astimezone().date().isoformat()
        by_day.setdefault(day, []).append(row)
    days = []
    for day in sorted(by_day):
        firsts = [r["first_sound_ms"] for r in by_day[day] if r.get("first_sound_ms") is not None]
        days.append({
            "date": day,
            "count": len(by_day[day]),
            "first_sound_median": int(statistics.median(firsts)) if firsts else None,
        })

    return {
        "count": len(rows),
        "stages": stages,
        "days": days,
        "router_degraded": sum(1 for r in rows if r.get("router_degraded") == 1),
        "judge_outcomes": dict(Counter(r["judge_outcome"] for r in rows if r.get("judge_outcome"))),
    }


def _seconds(ms: Optional[int]) -> str:
    return "   -  " if ms is None else f"{ms / 1000.0:6.1f}s"


def render(summary: Dict[str, Any], days: int) -> str:
    if summary["count"] == 0:
        return (
            f"📭 No spoken exchange recorded in the last {days} days.\n"
            "   Is the journal on? Check `latency_journal_enabled` in the settings (Advanced),\n"
            "   and that the assistant has been used by voice since it was updated."
        )
    n = summary["count"]
    lines = [
        f"📊 Latency journal, last {days} days",
        f"   {n} spoken exchange{'s' if n != 1 else ''}",
        "",
        "⏱️  Where the time goes (seconds)",
        f"   {'stage':<22}{'n':>5}{'median':>9}{'p90':>9}{'max':>9}",
    ]
    for key, label in STAGES:
        stats = summary["stages"].get(key)
        if stats is None:
            continue
        lines.append(
            f"   {label:<22}{stats['n']:>5}{_seconds(stats['median']):>9}"
            f"{_seconds(stats['p90']):>9}{_seconds(stats['max']):>9}"
        )
    lines += ["", "📅 Per day (median time to first sound)"]
    for day in summary["days"]:
        lines.append(
            f"   {day['date']}   {day['count']:>3} exchange{'s' if day['count'] != 1 else ' '}"
            f"   {_seconds(day['first_sound_median'])}"
        )
    lines += ["", "🔎 Counts"]
    lines.append(f"   tool router gave up: {summary['router_degraded']} of {n}")
    outcomes = summary["judge_outcomes"]
    if outcomes:
        said = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(outcomes.items()))
        lines.append(f"   intent judge: {said}")
    return "\n".join(lines)


def _open_read_only(db_path: Path) -> sqlite3.Connection:
    """The live database, in place, with no way to write through the connection.

    A read-only URI connection reads a write-ahead-log database that another
    process is writing: it sees the committed rows, including those not yet
    checkpointed into the main file, and never blocks the writer.
    """
    uri = db_path.resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    return con


def _rows(db_path: Path, cutoff_iso: str) -> List[Dict[str, Any]]:
    con = _open_read_only(db_path)
    try:
        present = {r[1] for r in con.execute("PRAGMA table_info(latency_log)").fetchall()}
        if not present:
            return []
        columns = [c for c in COLUMNS if c in present]
        cursor = con.execute(
            f"SELECT {', '.join(columns)} FROM latency_log WHERE ts_utc >= ? ORDER BY ts_utc",
            (cutoff_iso,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        con.close()


def _clear(db_path: Path) -> int:
    """Erase the journal and nothing else. Returns how many rows went."""
    con = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        has_table = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='latency_log'"
        ).fetchone()
        if not has_table:
            return 0
        cursor = con.execute("DELETE FROM latency_log")
        con.commit()
        return cursor.rowcount
    finally:
        con.close()


def main(argv: Optional[List[str]] = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Report on the latency journal.")
    parser.add_argument("--days", type=int, default=7, help="how many days back (default 7)")
    parser.add_argument("--db", help="path to the assistant's database (default: from settings)")
    parser.add_argument("--clear", action="store_true", help="erase the latency journal and exit")
    args = parser.parse_args(argv)

    if args.db:
        db_path = Path(args.db).expanduser()
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from jarvis.config import load_settings

        db_path = Path(load_settings().db_path).expanduser()
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return 1

    try:
        if args.clear:
            print(f"🧹 Erased {_clear(db_path)} latency journal rows. Nothing else was touched.")
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat()
        print(render(summarise(_rows(db_path, cutoff)), args.days))
        return 0
    except sqlite3.Error as e:
        print(f"❌ Cannot read the journal in {db_path}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
