"""Tests for the memory viewer's core HTTP API.

The core is the one memory artefact designed to be read and corrected by
the person it describes, and until now it was the only one with no way
to see it short of opening a terminal. These cover what the tab needs:
read both files, show what is believed against what was retired, and
save an edit back without the app taking ownership of the text.
"""

from __future__ import annotations

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False

from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestCoreApi:

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        from src.desktop_app import memory_viewer

        self.core = MemoryCore(tmp_path / "yuba")
        memory_viewer._core = self.core

        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        yield
        memory_viewer._core = None

    # ── Reading ──────────────────────────────────────────────────────

    def test_an_empty_core_reads_as_empty_sections(self):
        data = self.client.get("/api/core").get_json()

        assert data["profile"]["entries"] == []
        assert data["rules"]["entries"] == []

    def test_the_path_is_shown_so_it_can_be_found_outside_the_app(self):
        data = self.client.get("/api/core").get_json()

        assert data["profile"]["path"].endswith("profil.md")
        assert data["rules"]["path"].endswith("regles.md")

    def test_entries_come_back_with_what_the_user_needs_to_judge_them(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")

        entry = self.client.get("/api/core").get_json()["profile"]["entries"][0]

        assert entry["text"] == "Il vit à Lyon."
        assert entry["date"] == "2026-07-20"
        assert entry["source"] == "dit"
        assert entry["retired"] is False

    def test_a_retired_entry_is_returned_and_marked_as_such(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
        self.core.retire(
            SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-20", reason="corrigé",
        )

        entry = self.client.get("/api/core").get_json()["profile"]["entries"][0]

        assert entry["retired"] is True
        assert entry["retired_on"] == "2026-07-20"

    def test_the_raw_text_is_returned_so_it_can_be_edited_as_a_file(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")

        raw = self.client.get("/api/core").get_json()["profile"]["raw"]

        assert "Il vit à Lyon." in raw
        assert raw.lstrip().startswith("#")

    def test_rules_and_profile_stay_separate(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Lyon.")
        self.core.remember(SECTION_RULES, "Toujours répondre en français.")

        data = self.client.get("/api/core").get_json()

        assert [e["text"] for e in data["profile"]["entries"]] == ["Il vit à Lyon."]
        assert [e["text"] for e in data["rules"]["entries"]] == [
            "Toujours répondre en français.",
        ]

    # ── Saving ───────────────────────────────────────────────────────

    def test_an_edit_is_written_through_to_the_file(self):
        body = "# Profil\n\n- 2026-07-20 · dit : Il vit à Lyon.\n"

        resp = self.client.put("/api/core/profile", json={"raw": body})

        assert resp.status_code == 200
        assert [e.text for e in self.core.active(SECTION_PROFILE)] == ["Il vit à Lyon."]

    def test_the_text_is_saved_exactly_as_written(self):
        """The file belongs to the user. Reformatting their prose, or
        dropping a line the parser does not recognise, would make the app
        an unreliable place to edit it."""
        body = "# Mon profil\n\nUne note en prose.\n\n- Il déteste le lundi.\n"

        self.client.put("/api/core/profile", json={"raw": body})

        assert self.core.path_for(SECTION_PROFILE).read_text(encoding="utf-8") == body

    def test_an_edit_can_retire_an_entry_by_striking_it_through(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Paris.", on_date="2026-07-18")
        raw = self.client.get("/api/core").get_json()["profile"]["raw"]

        self.client.put(
            "/api/core/profile",
            json={"raw": raw.replace(
                "- 2026-07-18 · dit : Il vit à Paris.",
                "- ~~2026-07-18 · dit : Il vit à Paris.~~",
            )},
        )

        assert self.core.active(SECTION_PROFILE) == []

    def test_an_unknown_section_is_refused(self):
        resp = self.client.put("/api/core/banana", json={"raw": "- x\n"})

        assert resp.status_code == 404

    def test_a_save_with_no_body_is_refused(self):
        resp = self.client.put("/api/core/profile", json={})

        assert resp.status_code == 400

    def test_a_refused_save_leaves_the_file_untouched(self):
        self.core.remember(SECTION_PROFILE, "Il vit à Lyon.", on_date="2026-07-20")
        before = self.core.path_for(SECTION_PROFILE).read_text(encoding="utf-8")

        self.client.put("/api/core/profile", json={})

        assert self.core.path_for(SECTION_PROFILE).read_text(encoding="utf-8") == before

    def test_saving_reports_back_what_is_now_believed(self):
        body = "# Profil\n\n- 2026-07-20 · dit : Il vit à Lyon.\n"

        data = self.client.put("/api/core/profile", json={"raw": body}).get_json()

        assert [e["text"] for e in data["entries"]] == ["Il vit à Lyon."]


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestKnowledgeTabHidesInertBranches:
    """The graph's user and directives branches were handed over to the
    core and are no longer written or read. Showing them in a tab that
    says its contents reach every reply invites the user to correct a
    line that nothing will ever consult."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        from src.desktop_app import memory_viewer
        from src.jarvis.memory.graph import GraphMemoryStore

        self.store = GraphMemoryStore(str(tmp_path / "test.db"))
        memory_viewer._graph_store = self.store
        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        yield
        self.store.close()
        memory_viewer._graph_store = None

    def _branch_names(self):
        tree = self.client.get("/api/graph/tree").get_json()
        return [child["node"]["id"] for child in tree.get("children", [])]

    def test_an_emptied_user_branch_is_not_shown(self):
        assert "user" not in self._branch_names()
        assert "directives" not in self._branch_names()

    def test_the_world_branch_is_always_shown(self):
        assert "world" in self._branch_names()

    def test_a_branch_still_holding_something_stays_visible(self):
        """Until the daemon next starts and hands it over, that content is
        the only copy. Hiding it would be losing it from view."""
        self.store.update_node("user", data="Le nom de l'utilisateur est Hocine.")

        assert "user" in self._branch_names()

    def test_content_nested_under_the_branch_also_keeps_it_visible(self):
        self.store.create_node(
            name="Identity", description="Who", data="Il vit à Lyon.", parent_id="user",
        )

        assert "user" in self._branch_names()


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestActivityApi:
    """The ledger has to be readable, or it is just a table nobody opens.

    What it must never expose is the same thing it never stores: what a
    tool returned. The tab shows actions, not their contents."""

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path):
        from src.desktop_app import memory_viewer
        from src.jarvis.memory.db import Database

        self.db = Database(str(tmp_path / "test.db"), sqlite_vss_path=None)
        memory_viewer._activity_db = self.db
        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        yield
        self.db.close()
        memory_viewer._activity_db = None

    def _record(self, **kw):
        payload = dict(
            tool="webSearch", args={"query": "météo"}, risk="lecture",
            verdict="libre", outcome="ok", duration_ms=12, origin="chat",
            query="quel temps fait-il",
        )
        payload.update(kw)
        self.db.record_action(**payload)

    def test_an_empty_ledger_reads_as_no_actions(self):
        assert self.client.get("/api/activity").get_json()["actions"] == []

    def test_a_recorded_call_comes_back(self):
        self._record()

        actions = self.client.get("/api/activity").get_json()["actions"]

        assert actions[0]["tool"] == "webSearch"
        assert actions[0]["verdict"] == "libre"

    def test_a_refusal_is_visible(self):
        self._record(tool="localFiles", verdict="demande", outcome="refusé")

        actions = self.client.get("/api/activity").get_json()["actions"]

        assert actions[0]["outcome"] == "refusé"

    def test_the_user_can_clear_the_ledger(self):
        self._record()

        resp = self.client.delete("/api/activity")

        assert resp.status_code == 200
        assert self.db.recent_actions() == []
