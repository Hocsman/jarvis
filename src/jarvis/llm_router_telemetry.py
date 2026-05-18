"""Local-only telemetry for the hybrid LLM router.

Records per-call metrics (provider, model, intent label, token counts,
estimated cost, latency) to a SQLite table named ``llm_router_stats``
in the same database file the rest of the project uses
(``cfg.db_path``). Privacy contract: only metrics are stored, never
prompt or response text. This module is deliberately separate from
``memory/`` so the router can ship without touching the memory
package's schema.

The table is created on first call. Inserts are best-effort: every
exception is logged at debug level (gated by ``cfg.voice_debug`` via
the project's ``debug_log``) and swallowed so a broken stats table
never blocks a reply.

Pricing model
-------------
Cost estimates ride ``providers.pricing.get_pricing()``. Local Ollama
calls and unknown models are billed at 0 (best-effort). Cache reads
and writes apply the standard 0.1x / 1.25x multipliers on the input
side when ``usage`` includes them.
"""

from __future__ import annotations

import sqlite3
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

from .debug import debug_log
from .providers.pricing import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    get_pricing,
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_router_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    intent TEXT,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0,
    cost_estimate_usd REAL NOT NULL DEFAULT 0.0,
    latency_ms INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_llm_router_stats_ts ON llm_router_stats(ts_utc);
CREATE INDEX IF NOT EXISTS idx_llm_router_stats_provider ON llm_router_stats(provider);
"""


_lock = threading.Lock()
_initialised_paths: set[str] = set()


@contextmanager
def _connect(db_path: str):
    """Open a short-lived connection. The router owns this file handle
    rather than reusing ``memory.db.Database`` to keep the package
    boundary clean — see the module docstring."""
    conn = sqlite3.connect(db_path, timeout=2.0)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _bootstrap_schema(db_path: str) -> None:
    """Create the table on first use. Cached per ``db_path``."""
    with _lock:
        if db_path in _initialised_paths:
            return
        try:
            with _connect(db_path) as conn:
                conn.executescript(_SCHEMA_SQL)
            _initialised_paths.add(db_path)
        except Exception as exc:
            # Gated debug logging with traceback (per spec adjustment D).
            debug_log(
                f"telemetry schema bootstrap failed: {exc!r}\n{traceback.format_exc()}",
                "router",
            )


def estimate_cost_usd(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> float:
    """Return an estimated cost in USD for one LLM call.

    Routes through ``providers.pricing.get_pricing``. Unknown models
    (and local Ollama) bill at zero — best-effort, never crash. Cache
    write / read multipliers apply on the input side per Anthropic's
    public pricing.
    """
    pricing = get_pricing(model)
    if pricing is None:
        return 0.0
    in_rate = pricing["input"]
    out_rate = pricing["output"]

    cost = (tokens_in or 0) * in_rate / 1_000_000.0
    cost += (cache_creation_input_tokens or 0) * in_rate * CACHE_WRITE_MULTIPLIER / 1_000_000.0
    cost += (cache_read_input_tokens or 0) * in_rate * CACHE_READ_MULTIPLIER / 1_000_000.0
    cost += (tokens_out or 0) * out_rate / 1_000_000.0
    return round(cost, 6)


def record(
    db_path: Optional[str],
    *,
    provider: str,
    model: str,
    intent: Optional[str],
    tokens_in: int,
    tokens_out: int,
    cost_estimate_usd: float,
    latency_ms: int,
) -> None:
    """Insert one telemetry row. Best-effort.

    Every failure is logged at debug level (with traceback) and
    swallowed so the reply path is never blocked by a broken stats
    table. A ``None`` or empty ``db_path`` short-circuits — useful for
    tests that don't want the on-disk side effect.
    """
    if not db_path:
        return
    _bootstrap_schema(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with _connect(db_path) as conn:
            conn.execute(
                "INSERT INTO llm_router_stats "
                "(ts_utc, provider, model, intent, tokens_in, tokens_out, cost_estimate_usd, latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    provider,
                    model,
                    intent,
                    int(tokens_in or 0),
                    int(tokens_out or 0),
                    float(cost_estimate_usd or 0.0),
                    int(latency_ms or 0),
                ),
            )
    except Exception as exc:
        debug_log(
            f"telemetry record failed: {exc!r}\n{traceback.format_exc()}",
            "router",
        )


def fetch_recent(db_path: str, limit: int = 100) -> list[dict]:
    """Read recent rows. Intended for debugging / CLI inspection;
    no UI hook is wired in this PR (per spec)."""
    if not db_path:
        return []
    _bootstrap_schema(db_path)
    try:
        with _connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM llm_router_stats ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        debug_log(
            f"telemetry fetch failed: {exc!r}\n{traceback.format_exc()}",
            "router",
        )
        return []


def clear(db_path: str) -> None:
    """Wipe every row. Exposed so users can purge the table from a
    shell without needing a UI hook (per docs/HYBRID_LLM.md)."""
    if not db_path:
        return
    _bootstrap_schema(db_path)
    try:
        with _connect(db_path) as conn:
            conn.execute("DELETE FROM llm_router_stats")
    except Exception as exc:
        debug_log(
            f"telemetry clear failed: {exc!r}\n{traceback.format_exc()}",
            "router",
        )


def _reset_initialised_paths() -> None:
    """Test helper: clear the bootstrap cache so a fresh tmp_path
    re-runs the CREATE TABLE statement."""
    with _lock:
        _initialised_paths.clear()
