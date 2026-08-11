from __future__ import annotations
import sqlite3
import re
from typing import Sequence, Optional
from pathlib import Path
import threading
from datetime import datetime, timedelta, timezone

from ..debug import debug_log

# Bumped whenever the shape changes, so a later version can migrate.
_SCHEMA_VERSION = 4

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

-- Structured meals log (optional feature)
CREATE TABLE IF NOT EXISTS meals (
  id            INTEGER PRIMARY KEY,
  ts_utc        TEXT NOT NULL,
  source_app    TEXT NOT NULL,
  description   TEXT NOT NULL,
  calories_kcal REAL,
  protein_g     REAL,
  carbs_g       REAL,
  fat_g         REAL,
  fiber_g       REAL,
  sugar_g       REAL,
  sodium_mg     REAL,
  potassium_mg  REAL,
  micros_json   TEXT,
  confidence    REAL
);

-- What the assistant DID, one row per tool call. Never what it SAW:
-- there is deliberately no column for tool output. A ledger that kept
-- results would accumulate the contents of every page fetched and every
-- file read, which is a different and far larger thing than a list of
-- actions, and it would put that content somewhere the user reads
-- casually.
CREATE TABLE IF NOT EXISTS action_log (
  id          INTEGER PRIMARY KEY,
  ts_utc      TEXT NOT NULL,
  origin      TEXT,           -- chat, voice, or an unattended runner later
  tool        TEXT NOT NULL,
  args        TEXT,           -- redacted JSON
  risk        TEXT NOT NULL,  -- lecture / action / destructif
  verdict     TEXT NOT NULL,  -- libre / demande / jamais
  outcome     TEXT NOT NULL,  -- ok / échec / refusé / demandé / décliné / expiré
  duration_ms INTEGER,
  query       TEXT,           -- redacted, so a row can be placed in context
  request_id  TEXT            -- ties a question to whatever settled it
);

CREATE INDEX IF NOT EXISTS idx_action_log_ts ON action_log(ts_utc DESC);

-- Promises: things to say at a time, kept across restarts.
--
-- The one part of the assistant that MUST reach disk. A pending
-- confirmation deliberately never does, because an approval that
-- outlives the process was given without the context that produced it.
-- A reminder is the opposite: if a restart loses it, it was never a
-- reminder.
--
-- `kind` and `payload` carry defaults and exist so that recurring
-- routines — the next thing built on this — need no ALTER. `due_utc` is
-- what fires it; `due_local` and `tz` are what she reads back, kept
-- alongside so a timezone change cannot silently rewrite either.
CREATE TABLE IF NOT EXISTS rappels (
  id           TEXT PRIMARY KEY,   -- also the action_log request_id
  created_utc  TEXT NOT NULL,
  origin       TEXT,               -- voix / chat / fichier
  kind         TEXT NOT NULL DEFAULT 'rappel',
  texte        TEXT NOT NULL,      -- said aloud, so kept in the user's words
  payload      TEXT NOT NULL DEFAULT '{}',
  due_utc      TEXT NOT NULL,
  due_local    TEXT NOT NULL,
  tz           TEXT NOT NULL,
  query        TEXT,               -- redacted, like the ledger's
  etat         TEXT NOT NULL,      -- prévu / fini / annulé
  attempts     INTEGER NOT NULL DEFAULT 0,
  last_try_utc TEXT,
  said_utc     TEXT                -- non-NULL once actually spoken
);

CREATE INDEX IF NOT EXISTS idx_rappels_due ON rappels(etat, due_utc);

-- Conversation summaries for diary/memory system
CREATE TABLE IF NOT EXISTS conversation_summaries (
  id         INTEGER PRIMARY KEY,
  date_utc   TEXT NOT NULL,  -- YYYY-MM-DD format
  ts_utc     TEXT NOT NULL,  -- When summary was created
  summary    TEXT NOT NULL,  -- Concise summary of the day's conversations
  topics     TEXT,           -- Comma-separated list of main topics discussed
  source_app TEXT NOT NULL,  -- Source app that generated the conversation
  UNIQUE(date_utc, source_app)
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
  summary,
  topics,
  content='conversation_summaries',
  content_rowid='id',
  tokenize='porter'
);

-- Triggers for conversation summaries FTS
CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON conversation_summaries BEGIN
  INSERT INTO summaries_fts(summaries_fts, rowid, summary, topics) VALUES('delete', old.id, old.summary, old.topics);
  INSERT INTO summaries_fts(rowid, summary, topics) VALUES (new.id, new.summary, new.topics);
END;
"""

_VSS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vss0(
  id INTEGER PRIMARY KEY,
  vec FLOAT[768]
);

CREATE TABLE IF NOT EXISTS journal_lu (
  date_utc TEXT PRIMARY KEY,
  digest   TEXT NOT NULL,
  ts_utc   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_vec (
  summary_id INTEGER PRIMARY KEY REFERENCES conversation_summaries(id) ON DELETE CASCADE,
  emb_id     INTEGER NOT NULL REFERENCES embeddings(id)
);
"""


def _normalize_fts_query(raw: str) -> str:
    # Use improved fuzzy search query generation
    try:
        from .fuzzy_search import generate_flexible_fts_query
        flexible_query = generate_flexible_fts_query(raw)
        if flexible_query:
            return flexible_query
    except ImportError:
        pass
    
    # Fallback: Extract alphanumeric tokens and join them with spaces (logical AND)
    tokens = re.findall(r"[A-Za-z0-9_]+", raw)
    return " ".join(tokens)


class Database:
    def __init__(self, db_path: str, sqlite_vss_path: Optional[str] = None) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self.is_vss_enabled = False
        self._python_vector_store = None
        
        if sqlite_vss_path:
            try:
                self.conn.enable_load_extension(True)
                self.conn.load_extension(sqlite_vss_path)
                self.is_vss_enabled = True
            except Exception:
                self.is_vss_enabled = False
        
        # If sqlite-vss is not available, use best available vector store (FAISS or Python fallback)
        if not self.is_vss_enabled:
            from ..utils.vector_store import get_best_vector_store
            self._python_vector_store = get_best_vector_store(db_path, dimension=768)
            
            # Log which vector store implementation is being used
            import sys
            store_type = type(self._python_vector_store).__name__
            if store_type == "FAISSVectorStore":
                debug_log("Using FAISS vector store for fast search", "jarvis")
            else:
                debug_log("Using Python fallback vector store", "jarvis")
        
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.executescript(_SCHEMA_SQL)
            if self.is_vss_enabled:
                cur.executescript(_VSS_SCHEMA_SQL)
            self._migrate(cur)
            # Stamp the shape so a later change can migrate rather than
            # guess. The only migration precedent in this project wipes
            # its table when the shape surprises it, which is not an
            # acceptable inheritance for tables that will hold things the
            # user asked to keep.
            if cur.execute("PRAGMA user_version").fetchone()[0] < _SCHEMA_VERSION:
                cur.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self.conn.commit()

    def _migrate(self, cur) -> None:
        """Bring an existing database up to the current shape.

        Additive only, and idempotent: startup runs this every time, not
        once. `CREATE TABLE IF NOT EXISTS` leaves a table that already
        exists untouched, so a column added to the DDL above reaches an
        existing install only through here.

        Columns are added, never dropped, and no row is ever deleted. The
        Activity tab presents this table as a record of what Yuba did; a
        migration that discarded rows when the shape surprised it would
        make that claim false the first time the shape changed.
        """
        try:
            existing = {
                row[1] for row in cur.execute("PRAGMA table_info(action_log)").fetchall()
            }
        except Exception as e:  # table absent on a database we cannot read
            debug_log(f"schema inspection skipped: {e}", "jarvis")
            return

        if existing and "request_id" not in existing:
            cur.execute("ALTER TABLE action_log ADD COLUMN request_id TEXT")
            debug_log("action_log migrated: request_id added", "jarvis")

        # An install that predates the learning step has no `journal_lu`.
        # The DDL above only runs on a fresh database, so an existing one
        # reaches it here.
        cur.execute(
            "CREATE TABLE IF NOT EXISTS journal_lu ("
            "  date_utc TEXT PRIMARY KEY,"
            "  digest   TEXT NOT NULL,"
            "  ts_utc   TEXT NOT NULL"
            ")"
        )

    # ── What she has already been asked about ─────────────────────────

    def journal_deja_lu(self) -> dict:
        """Which diary rows have been read, and what they said then.

        Keyed on the date, valued on a digest of the summary as it read
        at the time. The digest is what makes this safe: a diary row is
        rewritten in place all day (`INSERT OR REPLACE` on its date), so
        remembering the date alone would mark today covered from the
        first pass onwards and everything said after it would never be
        read, permanently and silently. Content moves, the digest stops
        matching, and the row is offered again.

        Never raises: bookkeeping that cannot be read costs one re-read.
        """
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT date_utc, digest FROM journal_lu"
                ).fetchall()
            return {r["date_utc"]: r["digest"] for r in rows}
        except Exception as e:
            debug_log(f"journal_lu read skipped: {e}", "memory")
            return {}

    def marquer_journal_lu(self, rows) -> None:
        """Record ``(date_utc, digest)`` pairs as read.

        Called only after a reading that actually happened. A pass whose
        model timed out, answered nothing, or answered unparseably has
        not looked at these days, and recording them would skip them for
        ever on the strength of a failure.

        Never raises: a lost mark costs one re-read, and raising costs
        the user their answer.
        """
        try:
            stamp = datetime.now(timezone.utc).isoformat()
            payload = [
                (str(date), str(digest), stamp)
                for date, digest in (rows or [])
                if date and digest
            ]
            if not payload:
                return
            with self._lock:
                self.conn.executemany(
                    "INSERT OR REPLACE INTO journal_lu (date_utc, digest, ts_utc) "
                    "VALUES (?, ?, ?)",
                    payload,
                )
                self.conn.commit()
        except Exception as e:
            debug_log(f"journal_lu write skipped: {e}", "memory")

    # ── Action ledger ─────────────────────────────────────────────────

    def record_action(
        self,
        *,
        tool: str,
        args: Optional[dict],
        risk: str,
        verdict: str,
        outcome: str,
        duration_ms: Optional[int] = None,
        origin: Optional[str] = None,
        query: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Record one tool call. Bookkeeping: never raises into the caller.

        Arguments and the originating query are redacted on the way in,
        because a tool call carries whatever the user just said.

        ``request_id`` ties a question to whatever settled it: the gate
        writes one row when it asks and one when the answer arrives, and
        nothing but this id says they are the same episode.
        """
        import json
        from ..utils.redact import redact

        try:
            try:
                args_text = redact(json.dumps(args or {}, ensure_ascii=False, default=str))
            except Exception:
                args_text = "<non sérialisable>"
            with self._lock:
                self.conn.execute(
                    "INSERT INTO action_log "
                    "(ts_utc, origin, tool, args, risk, verdict, outcome, "
                    "duration_ms, query, request_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        origin, tool, args_text, risk, verdict, outcome,
                        duration_ms, redact(query) if query else None,
                        request_id,
                    ),
                )
                self.conn.commit()
        except Exception as e:
            debug_log(f"action log write failed (non-fatal): {e}", "tools")

    def recent_actions(self, limit: int = 200) -> list:
        """The most recent calls, newest first."""
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM action_log ORDER BY ts_utc DESC, id DESC LIMIT ?",
                (int(limit),),
            )
            return cur.fetchall()

    def prune_actions(self, max_age_days: int = 90) -> int:
        """Drop entries older than the retention window."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            cur = self.conn.execute("DELETE FROM action_log WHERE ts_utc < ?", (cutoff,))
            self.conn.commit()
            return cur.rowcount

    def clear_actions(self) -> None:
        """Erase the ledger at the user's request."""
        with self._lock:
            self.conn.execute("DELETE FROM action_log")
            self.conn.commit()

    # ── Reminders ─────────────────────────────────────────────────────
    #
    # Two processes open this file — the daemon and the memory viewer —
    # on a WAL database opened without a busy timeout, so a write can
    # lose a race it would win a moment later. Each one retries once.

    ETAT_PENDING = "prévu"
    ETAT_DONE = "fini"
    ETAT_CANCELLED = "annulé"

    def _write(self, sql: str, params: tuple) -> int:
        """Run one write, retrying once if the file was momentarily busy."""
        import time as _time

        for attempt in (0, 1):
            try:
                with self._lock:
                    cur = self.conn.execute(sql, params)
                    self.conn.commit()
                    return cur.rowcount
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or attempt:
                    raise
                debug_log("database busy, retrying once", "jarvis")
                _time.sleep(0.1)
        return 0

    def add_rappel(
        self, *, texte: str, due_utc: str, due_local: str, tz: str,
        origin: Optional[str] = None, query: Optional[str] = None,
        kind: str = "rappel", payload: Optional[dict] = None,
    ) -> str:
        """Record one promise. Returns its id, which is also its ledger key.

        ``texte`` is kept as the user said it, because it is read back
        aloud — redacting it would have her say a placeholder to their
        face. ``query`` is redacted, like the ledger's, because it is
        bookkeeping rather than speech.
        """
        import json
        import uuid

        from ..utils.redact import redact

        rid = uuid.uuid4().hex
        self._write(
            "INSERT INTO rappels "
            "(id, created_utc, origin, kind, texte, payload, due_utc, "
            " due_local, tz, query, etat) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                rid, datetime.now(timezone.utc).isoformat(), origin, kind,
                texte, json.dumps(payload or {}, ensure_ascii=False),
                due_utc, due_local, tz,
                redact(query) if query else None, self.ETAT_PENDING,
            ),
        )
        return rid

    def due_rappels(self, now_utc: str, limit: int = 20,
                    kind: Optional[str] = None) -> list:
        """Rows owed at ``now_utc``, oldest first.

        ``kind`` separates the two things this table holds. A routine
        read out by the reminder scheduler would be spoken as a sentence
        — which is not what a routine is — and then settled, so it would
        never run again.
        """
        clause = "etat = ? AND due_utc <= ?"
        params: list = [self.ETAT_PENDING, now_utc]
        if kind is not None:
            clause += " AND kind = ?"
            params.append(kind)
        params.append(int(limit))
        with self._lock:
            cur = self.conn.execute(
                f"SELECT * FROM rappels WHERE {clause} ORDER BY due_utc ASC LIMIT ?",
                tuple(params),
            )
            return [dict(row) for row in cur.fetchall()]

    def pending_rappels(self, kind: Optional[str] = None) -> list:
        """Everything still owed, soonest first — what the user is shown."""
        clause = "etat = ?"
        params: list = [self.ETAT_PENDING]
        if kind is not None:
            clause += " AND kind = ?"
            params.append(kind)
        with self._lock:
            cur = self.conn.execute(
                f"SELECT * FROM rappels WHERE {clause} ORDER BY due_utc ASC",
                tuple(params),
            )
            return [dict(row) for row in cur.fetchall()]

    def all_rappels(self, limit: int = 200) -> list:
        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM rappels ORDER BY due_utc DESC LIMIT ?", (int(limit),),
            )
            return [dict(row) for row in cur.fetchall()]

    def last_cancelled_rappel(self, kind: str, nom: str) -> Optional[dict]:
        """The most recently created cancelled row of a given name.

        Read only. `annulé` stays final for every write — nothing here
        revives it — but a routine the user is restarting still has its
        old recurrence recorded, and taking the hour from there beats
        inventing one.
        """
        import json

        with self._lock:
            cur = self.conn.execute(
                "SELECT * FROM rappels WHERE kind = ? AND etat = ? "
                "ORDER BY created_utc DESC",
                (kind, self.ETAT_CANCELLED),
            )
            for row in cur.fetchall():
                row = dict(row)
                try:
                    if json.loads(row.get("payload") or "{}").get("nom") == nom:
                        return row
                except Exception:
                    continue
        return None

    def mark_rappel_tried(self, rappel_id: str, now_utc: str) -> None:
        """One more attempt at delivering it. Bounds a failing loop."""
        self._write(
            "UPDATE rappels SET attempts = attempts + 1, last_try_utc = ? "
            "WHERE id = ?",
            (now_utc, rappel_id),
        )

    def settle_rappel(self, rappel_id: str, *, said_utc: str) -> None:
        """It was actually said. Only delivery settles a promise."""
        self._write(
            "UPDATE rappels SET etat = ?, said_utc = ? WHERE id = ?",
            (self.ETAT_DONE, said_utc, rappel_id),
        )

    def cancel_rappel(self, rappel_id: str) -> bool:
        """The user does not want it any more. Returns whether one matched."""
        return self._write(
            "UPDATE rappels SET etat = ? WHERE id = ? AND etat = ?",
            (self.ETAT_CANCELLED, rappel_id, self.ETAT_PENDING),
        ) > 0

    def advance_rappel(self, rappel_id: str, *, due_utc: str,
                       due_local: str, tz: str) -> None:
        """Move a recurring row to its next occurrence.

        Attempts reset, because they bound a *current* failure rather
        than a lifetime: a routine that failed twice in March must not
        arrive in December one failure from being switched off.

        Guarded on `prévu`, so advancing something cancelled cannot
        revive it — cancelling has to be final or the user cannot stop a
        routine.
        """
        self._write(
            "UPDATE rappels SET due_utc = ?, due_local = ?, tz = ?, "
            "attempts = 0, last_try_utc = NULL WHERE id = ? AND etat = ?",
            (due_utc, due_local, tz, rappel_id, self.ETAT_PENDING),
        )

    def set_rappel_payload(self, rappel_id: str, payload: dict) -> None:
        """Replace the row's payload.

        Guarded on `prévu` like `advance_rappel`, so bookkeeping written
        after a cancellation cannot touch a row the user has stopped.
        """
        import json

        self._write(
            "UPDATE rappels SET payload = ? WHERE id = ? AND etat = ?",
            (json.dumps(payload, ensure_ascii=False), rappel_id, self.ETAT_PENDING),
        )

    def prune_rappels(self, max_age_days: int = 90) -> int:
        """Drop old settled and cancelled ones.

        Never anything still owed, however old: pruning a promise is
        losing it, and a reminder set for next year is still a promise.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        return self._write(
            "DELETE FROM rappels WHERE etat != ? AND created_utc < ?",
            (self.ETAT_PENDING, cutoff),
        )

    

    def search_hybrid(self, fts_query: str, query_vec_json: Optional[str], top_k: int = 8) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            safe_q = _normalize_fts_query(fts_query)

            # Use Python vector store if sqlite-vss is not available
            if not self.is_vss_enabled and self._python_vector_store and query_vec_json is not None and safe_q:
                # Parse query vector
                import json as _json
                query_vec = _json.loads(query_vec_json)
                
                # Get vector search results (use max of top_k*3 and 50 for good hybrid scoring)
                vector_search_limit = max(top_k * 3, 50)
                vector_results = self._python_vector_store.search(query_vec, top_k=vector_search_limit)
                
                # Get FTS results (use max of top_k*3 and 50 for good hybrid scoring)
                fts_search_limit = max(top_k * 3, 50)
                fts_sql = f"""
                SELECT s.id, bm25(summaries_fts) AS bm
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY bm
                LIMIT {fts_search_limit}
                """
                fts_rows = cur.execute(fts_sql, (safe_q,)).fetchall()
                fts_scores = {row['id']: row['bm'] for row in fts_rows}
                
                # Combine scores
                combined_scores = {}
                
                # Add vector scores (60% weight)
                for summary_id, distance in vector_results:
                    combined_scores[summary_id] = (1.0 / (1.0 + distance)) * 0.6
                
                # Add FTS scores (40% weight)
                for summary_id, bm_score in fts_scores.items():
                    if summary_id in combined_scores:
                        combined_scores[summary_id] += (1.0 / (1.0 + bm_score)) * 0.4
                    else:
                        combined_scores[summary_id] = (1.0 / (1.0 + bm_score)) * 0.4
                
                # Sort by combined score and fetch summaries
                sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                if sorted_ids:
                    # Fetch summaries for top results
                    placeholders = ','.join('?' * len(sorted_ids))
                    summary_sql = f"""
                    SELECT s.id, 
                           '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                           'summary' AS result_type
                    FROM conversation_summaries s
                    WHERE s.id IN ({placeholders})
                    """
                    rows = cur.execute(summary_sql, [sid for sid, _ in sorted_ids]).fetchall()
                    
                    # Create result rows with scores
                    results = []
                    id_to_score = {sid: score for sid, score in sorted_ids}
                    for row in rows:
                        # Create a new row dict with score
                        result = dict(row)
                        result['score'] = id_to_score.get(row['id'], 0.0)
                        results.append(result)
                    
                    # Sort by score again (in case DB returned in different order)
                    results.sort(key=lambda x: x['score'], reverse=True)
                    return results
                else:
                    return []
                    
            elif self.is_vss_enabled and query_vec_json is not None and safe_q:
                # Hybrid search: 60% vector similarity (semantic) + 40% FTS (exact terms)
                # This balances finding semantically related content with keyword matches
                # Use dynamic limits for efficiency on large datasets
                search_limit = max(top_k * 3, 50)
                summary_sql = f"""
                WITH fts_sum AS (
                  SELECT s.id, bm25(summaries_fts) AS bm
                  FROM summaries_fts
                  JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                  WHERE summaries_fts MATCH ?
                  ORDER BY bm LIMIT {search_limit}
                ),
                v_sum AS (
                  SELECT sv.summary_id AS id, distance
                  FROM vss_search(embeddings, 'vec', ?)
                  JOIN summary_vec sv ON sv.emb_id = rowid
                  LIMIT {search_limit}
                )
                SELECT s.id, (
                    (1.0/(1.0+COALESCE(v_sum.distance, 1))) * 0.6 +
                    (1.0/(1.0+COALESCE(fts_sum.bm, 10))) * 0.4
                  ) AS score,
                  '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                  'summary' AS result_type
                FROM conversation_summaries s
                LEFT JOIN v_sum     ON v_sum.id = s.id
                LEFT JOIN fts_sum   ON fts_sum.id = s.id
                WHERE v_sum.id IS NOT NULL OR fts_sum.id IS NOT NULL
                ORDER BY score DESC
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql, (safe_q, query_vec_json)).fetchall()

            elif safe_q:
                # FTS-only search over conversation summaries
                summary_sql = f"""
                SELECT s.id, bm25(summaries_fts) AS score,
                       '[' || s.date_utc || '] ' || s.summary || ' (Topics: ' || COALESCE(s.topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM summaries_fts
                JOIN conversation_summaries s ON s.id = summaries_fts.rowid
                WHERE summaries_fts MATCH ?
                ORDER BY score
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql, (safe_q,)).fetchall()

            else:
                # Fallback: latest conversation summaries
                summary_sql = f"""
                SELECT id, 0.0 AS score,
                       '[' || date_utc || '] ' || summary || ' (Topics: ' || COALESCE(topics, '') || ')' AS text,
                       'summary' AS result_type
                FROM conversation_summaries
                ORDER BY date_utc DESC
                LIMIT {int(top_k)};
                """
                rows = cur.execute(summary_sql).fetchall()

            return rows

    @staticmethod
    def _pack_vector(vec: Sequence[float]) -> bytes:
        # SQLite-vss expects a float array; packing via array('f') ensures binary blob layout.
        import array
        arr = array.array('f', [float(x) for x in vec])
        return arr.tobytes()

    # --- Meals API ---
    def insert_meal(
        self,
        ts_utc: str,
        source_app: str,
        description: str,
        calories_kcal: Optional[float] = None,
        protein_g: Optional[float] = None,
        carbs_g: Optional[float] = None,
        fat_g: Optional[float] = None,
        fiber_g: Optional[float] = None,
        sugar_g: Optional[float] = None,
        sodium_mg: Optional[float] = None,
        potassium_mg: Optional[float] = None,
        micros_json: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO meals(ts_utc, source_app, description, calories_kcal, protein_g, carbs_g, fat_g, fiber_g, sugar_g, sodium_mg, potassium_mg, micros_json, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_utc,
                    source_app,
                    description,
                    calories_kcal,
                    protein_g,
                    carbs_g,
                    fat_g,
                    fiber_g,
                    sugar_g,
                    sodium_mg,
                    potassium_mg,
                    micros_json,
                    confidence,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_meals_between(self, ts_utc_min: str, ts_utc_max: str) -> list[sqlite3.Row]:
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM meals
                WHERE ts_utc >= ? AND ts_utc <= ?
                ORDER BY ts_utc ASC
                """,
                (ts_utc_min, ts_utc_max),
            ).fetchall()
            return rows

    def delete_meal(self, meal_id: int) -> bool:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM meals WHERE id = ?", (meal_id,))
            self.conn.commit()
            return cur.rowcount > 0

    # --- Conversation Summaries API ---
    def upsert_conversation_summary(
        self,
        date_utc: str,  # YYYY-MM-DD format
        summary: str,
        topics: Optional[str] = None,
        source_app: str = "jarvis",
        ts_utc: Optional[str] = None,
    ) -> int:
        """Insert or update a conversation summary for a given date.

        ``ts_utc`` defaults to "now". Maintenance ops that rewrite an
        existing row's content without changing what it represents (e.g.
        the deflection scrub bulk sweep) should pass through the row's
        original ``ts_utc`` so the audit trail is preserved.
        """
        if ts_utc is None:
            ts_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO conversation_summaries(date_utc, ts_utc, summary, topics, source_app)
                VALUES (?, ?, ?, ?, ?)
                """,
                (date_utc, ts_utc, summary, topics, source_app),
            )
            self.conn.commit()
            return int(cur.lastrowid)

    def get_conversation_summary(self, date_utc: str, source_app: str = "jarvis") -> Optional[sqlite3.Row]:
        """Get conversation summary for a specific date."""
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc = ? AND source_app = ?
                """,
                (date_utc, source_app),
            ).fetchone()
            return row

    def get_recent_conversation_summaries(self, days: int = 7) -> list[sqlite3.Row]:
        """Get conversation summaries from the last N days."""
        from datetime import datetime, timedelta, timezone
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()

        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                WHERE date_utc >= ?
                ORDER BY date_utc DESC
                """,
                (cutoff_date,),
            ).fetchall()
            return rows

    def get_all_conversation_summaries(self) -> list[sqlite3.Row]:
        """Get all conversation summaries, ordered by date ascending (oldest first).

        Used for bulk import into graph memory — processes diary entries
        chronologically so the graph builds up naturally.
        """
        with self._lock:
            cur = self.conn.cursor()
            rows = cur.execute(
                """
                SELECT * FROM conversation_summaries
                ORDER BY date_utc ASC
                """,
            ).fetchall()
            return rows

    def upsert_summary_embedding(self, summary_id: int, vec: Sequence[float]) -> Optional[int]:
        """Store or update embedding for a conversation summary."""
        if self.is_vss_enabled:
            # Use sqlite-vss
            with self._lock:
                cur = self.conn.cursor()
                cur.execute("INSERT INTO embeddings(vec) VALUES (?)", (sqlite3.Binary(self._pack_vector(vec)),))
                emb_id = cur.lastrowid
                cur.execute(
                    "INSERT OR REPLACE INTO summary_vec(summary_id, emb_id) VALUES (?, ?)",
                    (summary_id, emb_id),
                )
                self.conn.commit()
                return int(emb_id)
        elif self._python_vector_store:
            # Use Python vector store
            self._python_vector_store.add_vector(summary_id, list(vec))
            return summary_id  # Return summary_id as a placeholder for emb_id
        else:
            return None

    def close(self) -> None:
        try:
            with self._lock:
                self.conn.close()
        except Exception:
            pass
