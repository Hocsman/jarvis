"""The viewer's reads read, and its renderers render text, not markup.

The gate from the local-only hardening leaves GETs token-free, which is
right for pure reads: no response grants CORS, so a foreign page can
ask but never hear the answer. Three classes of endpoint break that
premise:

- a GET with a side effect (viewing a node increments its access
  score, as the graph spec asks of UI views; a drive-by ``<img>`` from
  any website could inflate the preset nodes' scores and churn the
  ledger's prune);
- integer query parameters with no floor and no guard (``?limit=-1``
  is "no limit" to SQLite, ``?limit=abc`` raises out of the route as a
  500);
- the sweep progress renderers interpolating server-provided strings
  (node names are LLM-extracted from diary content) into ``innerHTML``
  unescaped, where a poisoned name becomes script in the page that
  holds the launch token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


@pytest.fixture()
def viewer(tmp_path, monkeypatch):
    """The viewer app on an isolated database, graph and ledger."""
    from desktop_app import memory_viewer
    from jarvis.memory.db import Database
    from jarvis.memory.graph import GraphMemoryStore

    db_path = str(tmp_path / "viewer.db")
    seed = Database(db_path)
    seed.upsert_conversation_summary(
        date_utc="2026-09-01", summary="one", topics=None, source_app="jarvis",
    )
    seed.upsert_conversation_summary(
        date_utc="2026-09-02", summary="two", topics=None, source_app="jarvis",
    )
    seed.upsert_conversation_summary(
        date_utc="2026-09-03", summary="three", topics=None, source_app="jarvis",
    )
    monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: db_path)
    # The file-backed routes (journal, routines) resolve their directory
    # from the settings; point them at the temporary tree so the test
    # never reads or writes the real user's config or yuba files.
    monkeypatch.setattr(
        memory_viewer, "load_settings",
        lambda: SimpleNamespace(db_path=db_path),
    )

    store = GraphMemoryStore(str(tmp_path / "graph.db"))
    monkeypatch.setattr(memory_viewer, "_graph_store", store)

    ledger = Database(str(tmp_path / "ledger.db"), sqlite_vss_path=None)
    monkeypatch.setattr(memory_viewer, "_activity_db", ledger)

    monkeypatch.setattr(memory_viewer, "_db_conn", None)
    memory_viewer.app.config["TESTING"] = True

    yield SimpleNamespace(
        mv=memory_viewer,
        client=memory_viewer.app.test_client(),
        token=memory_viewer.get_launch_token(),
        auth={"X-Jarvis-Token": memory_viewer.get_launch_token()},
        seed_db=seed,
        store=store,
        ledger=ledger,
        db_path=db_path,
    )

    conn = memory_viewer._db_conn
    if conn is not None:
        conn.close()
    store.close()
    ledger.close()
    seed.close()


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestReadsThatWriteCarryTheToken:
    """Viewing a node writes: the graph spec makes the access score
    rise "when a node is viewed in the UI", and the UI is the served
    page, which holds the launch token. A drive-by subresource GET
    from a foreign site holds no token and must not move the score."""

    def test_a_tokenless_node_view_is_refused(self, viewer):
        node = viewer.store.create_node(
            name="Test", description="d", parent_id="root",
        )

        resp = viewer.client.get(f"/api/graph/node/{node.id}")

        assert resp.status_code == 403

    def test_a_drive_by_view_does_not_move_the_access_score(self, viewer):
        node = viewer.store.create_node(
            name="Test", description="d", parent_id="root",
        )
        before = viewer.store.get_node(node.id).to_dict()["access_count"]

        viewer.client.get(f"/api/graph/node/{node.id}")

        after = viewer.store.get_node(node.id).to_dict()["access_count"]
        assert after == before

    def test_the_served_page_can_still_view_a_node(self, viewer):
        node = viewer.store.create_node(
            name="Test", description="d", parent_id="root",
        )

        resp = viewer.client.get(
            f"/api/graph/node/{node.id}", headers=viewer.auth,
        )

        assert resp.status_code == 200
        assert resp.get_json()["node"]["id"] == node.id
        after = viewer.store.get_node(node.id).to_dict()["access_count"]
        assert after >= 1

    def test_a_side_effect_free_read_stays_token_free(self, viewer):
        resp = viewer.client.get("/api/graph/nodes")

        assert resp.status_code == 200


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestTheLedgerReadDoesNotPrune:
    """The Activity tab shows the ledger; it does not curate it. The
    90-day prune belongs to the reminder scheduler's tick, so a GET
    reachable from any website performs no write transaction."""

    def test_an_old_row_survives_the_read(self, viewer):
        viewer.ledger.conn.execute(
            """INSERT INTO action_log
               (ts_utc, origin, tool, args, risk, verdict, outcome,
                duration_ms, query, request_id)
               VALUES (?, 'chat', 'webSearch', '{}', 'lecture', 'libre',
                       'ok', 5, 'vieille action', 'r1')""",
            ("2020-01-01T00:00:00Z",),
        )
        viewer.ledger.conn.commit()

        resp = viewer.client.get("/api/activity")

        assert resp.status_code == 200
        queries = [a["query"] for a in resp.get_json()["actions"]]
        assert "vieille action" in queries
        remaining = viewer.ledger.recent_actions(limit=500)
        assert any(r["query"] == "vieille action" for r in remaining)


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestIntegerParamsAreBounded:
    """A negative LIMIT is "no limit" to SQLite, and an unparsable one
    raises out of the route as a 500. Every integer query parameter is
    clamped into its range and falls back to its default on garbage."""

    def test_a_negative_limit_clamps_to_the_floor(self, viewer):
        resp = viewer.client.get("/api/memories?limit=-1")

        assert resp.status_code == 200
        assert len(resp.get_json()["memories"]) == 1

    def test_garbage_limit_falls_back_to_the_default(self, viewer):
        resp = viewer.client.get("/api/memories?limit=abc")

        assert resp.status_code == 200
        assert len(resp.get_json()["memories"]) == 3

    def test_a_negative_limit_on_meals_clamps(self, viewer):
        resp = viewer.client.get("/api/meals?limit=-5")

        assert resp.status_code == 200

    def test_garbage_params_do_not_raise_out_of_the_routes(self, viewer):
        for url in (
            "/api/graph/recent?limit=abc",
            "/api/graph/top?limit=-3",
            "/api/graph/nodes?max_depth=abc",
            "/api/graph/tree?max_depth=-2",
            "/api/journal?days=abc",
            "/api/routines?days=abc",
        ):
            resp = viewer.client.get(url)
            assert resp.status_code == 200, url


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestSweepRenderersEscapeServerStrings:
    """Node names are LLM-extracted from diary content and meal
    descriptions from spoken text: a poisoned string reaching
    ``innerHTML`` unescaped is script running in the page that holds
    the launch token. The renderers escape every server-provided
    string they interpolate, in text position and in attributes (the
    surviving ``escapeHtml`` escapes quotes, so an attribute value
    cannot be broken out of)."""

    def test_the_consolidate_log_escapes_the_node_name(self, viewer):
        html = viewer.client.get("/").get_data(as_text=True)

        assert "${escapeHtml(msg.node)}" in html

    def test_the_import_log_escapes_the_date_and_detail(self, viewer):
        html = viewer.client.get("/").get_data(as_text=True)

        assert "${escapeHtml(msg.date)}" in html
        assert "escapeHtml(detail)" in html

    def test_the_meal_card_escapes_the_description(self, viewer):
        html = viewer.client.get("/").get_data(as_text=True)

        assert "${escapeHtml(meal.description)}" in html
        assert "<h3>${meal.description}</h3>" not in html

    def test_the_page_declares_one_quote_escaping_helper(self, viewer):
        """One ``escapeHtml``, the quote-aware one: a second, weaker
        declaration in the same script would shadow it for every call
        site and silently un-defuse the attribute sinks."""
        html = viewer.client.get("/").get_data(as_text=True)

        assert html.count("function escapeHtml(") == 1
        assert ".replace(/\"/g, '&quot;')" in html

    def test_no_sweep_log_line_interpolates_raw_server_strings(self, viewer):
        html = viewer.client.get("/").get_data(as_text=True)

        assert "`<div>${arrow} ${msg.node}" not in html
        assert "`<div>${icon} ${msg.date}" not in html
