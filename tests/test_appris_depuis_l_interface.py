"""Accepting and refusing a proposal from the tab.

The file remains the artefact. The tab is a second door onto it, not a
copy of it: it re-reads the page on every request and writes back
through the same guarded write the tool uses, so his editor and this
window can be open at once without either losing the other's work.

What the tab buys is timing. A tick in the file lands on his next ask,
because nothing watches the file and a watcher is the schedule his
framing excludes. A click here lands now.

It buys nothing else, and must not. Clicking accept is the same act as
ticking the box — his — and it goes through the same harvest, with the
same guards and the same refusal to write a line carrying a redaction
placeholder. Refusing is the strike, and it is as durable here as there.
"""

from __future__ import annotations

import pytest

try:
    import flask  # noqa: F401

    _HAS_FLASK = True
except ImportError:
    _HAS_FLASK = False


PAGE = """# Appris

## Profil
- [ ] 2026-08-04 · journal : Il court le mardi matin.
  > « the user mentioned running on Tuesday mornings »
- [ ] 2026-08-02 · journal : Il a un chat qui s'appelle Miso.
  > « the user's cat Miso »
- ~~2026-07-30 · journal : Il déteste le café.~~
  > « the user said the coffee was bad »

## Règles
- [ ] 2026-08-03 · journal : Toujours répondre en français.
  > « the user asked for French »
"""


@pytest.mark.unit
@pytest.mark.skipif(not _HAS_FLASK, reason="Flask not available")
class TestApprisApi:

    @pytest.fixture(autouse=True)
    def setup_app(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from src.desktop_app import memory_viewer
        from src.jarvis.appris.page import appris_path, invalidate_appris_cache

        cfg = SimpleNamespace(db_path=str(tmp_path / "jarvis.db"))
        monkeypatch.setattr(memory_viewer, "load_settings", lambda: cfg)
        self.cfg = cfg
        self.page = appris_path(cfg)
        self.page.parent.mkdir(parents=True, exist_ok=True)
        self.page.write_text(PAGE, encoding="utf-8")
        invalidate_appris_cache()

        memory_viewer.app.config["TESTING"] = True
        self.client = memory_viewer.app.test_client()
        yield

    def _lignes(self):
        return self.client.get("/api/appris").get_json()["propositions"]

    def _core(self, section):
        from src.jarvis.memory.core import MemoryCore
        return [e.text for e in MemoryCore.for_config(self.cfg).active(section)]

    # ── Reading ──────────────────────────────────────────────────────

    def test_the_tab_shows_every_proposal_with_its_state(self):
        etats = [p["etat"] for p in self._lignes()]

        assert etats.count("attente") == 3
        assert etats.count("rayée") == 1

    def test_a_proposal_carries_what_he_needs_to_judge_it(self):
        """The sentence, where it came from in his own journal, and which
        of his files it would land in. Without the quote he is agreeing
        to a claim about himself on trust."""
        p = self._lignes()[0]

        assert p["texte"] == "Il court le mardi matin."
        assert "Tuesday" in p["citation"]
        assert p["section"] == "profil"
        assert p["date"] == "2026-08-04"

    def test_the_path_is_shown_so_it_can_be_edited_outside_the_app(self):
        assert "appris.md" in self.client.get("/api/appris").get_json()["chemin"]

    def test_a_missing_file_is_no_proposals_rather_than_an_error(self, tmp_path):
        self.page.unlink()

        reponse = self.client.get("/api/appris")

        assert reponse.status_code == 200
        assert reponse.get_json()["propositions"] == []

    # ── Accepting ────────────────────────────────────────────────────

    def test_accepting_writes_it_to_the_core_now(self):
        """The whole reason the tab exists. A tick in the file waits for
        his next ask; a click here does not."""
        from src.jarvis.memory.core import SECTION_PROFILE

        ligne = self._lignes()[0]["ligne"]

        r = self.client.post("/api/appris/retenir", json={"ligne": ligne})

        assert r.status_code == 200 and r.get_json()["retenue"] is True
        assert "Il court le mardi matin." in self._core(SECTION_PROFILE)

    def test_it_lands_with_the_confirmed_source_and_the_journal_date(self):
        from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

        self.client.post("/api/appris/retenir",
                         json={"ligne": self._lignes()[0]["ligne"]})

        e = MemoryCore.for_config(self.cfg).active(SECTION_PROFILE)[0]
        assert e.source == "confirmé"
        assert e.date == "2026-08-04"

    def test_a_rules_proposal_lands_in_the_rules(self):
        from src.jarvis.memory.core import SECTION_RULES

        ligne = [p for p in self._lignes() if p["section"] == "regles"][0]["ligne"]

        self.client.post("/api/appris/retenir", json={"ligne": ligne})

        assert "Toujours répondre en français." in self._core(SECTION_RULES)

    def test_the_accepted_line_is_struck_and_stamped(self):
        ligne = self._lignes()[0]["ligne"]

        self.client.post("/api/appris/retenir", json={"ligne": ligne})

        texte = self.page.read_text(encoding="utf-8")
        assert "~~2026-08-04 · journal : Il court le mardi matin.~~" in texte
        assert "retenu le" in texte

    def test_accepting_touches_only_that_proposal(self):
        from src.jarvis.memory.core import SECTION_PROFILE

        self.client.post("/api/appris/retenir",
                         json={"ligne": self._lignes()[0]["ligne"]})

        assert self._core(SECTION_PROFILE) == ["Il court le mardi matin."]
        assert [p["etat"] for p in self._lignes()].count("attente") == 2

    # ── Refusing ─────────────────────────────────────────────────────

    def test_refusing_strikes_it_and_writes_nothing(self):
        from src.jarvis.memory.core import SECTION_PROFILE

        ligne = self._lignes()[0]["ligne"]

        r = self.client.post("/api/appris/refuser", json={"ligne": ligne})

        assert r.status_code == 200 and r.get_json()["refusee"] is True
        assert self._core(SECTION_PROFILE) == []
        assert [p["etat"] for p in self._lignes()][0] == "rayée"

    def test_a_refusal_carries_no_stamp(self):
        """A struck line with no stamp is what he writes by hand. The tab
        writes the same thing, so the file reads the same whichever door
        the refusal came through."""
        self.client.post("/api/appris/refuser",
                         json={"ligne": self._lignes()[0]["ligne"]})

        ligne = [l for l in self.page.read_text(encoding="utf-8").splitlines()
                 if "court le mardi" in l][0]
        assert "retenu le" not in ligne

    # ── Every default leans shut ─────────────────────────────────────

    def test_a_line_that_is_gone_changes_nothing(self):
        """He may have deleted it in his editor between the page loading
        and the click."""
        r = self.client.post("/api/appris/retenir",
                             json={"ligne": "- [ ] 2026-01-01 · journal : jamais vue."})

        assert r.get_json()["retenue"] is False

    def test_an_empty_request_is_refused(self):
        assert self.client.post("/api/appris/retenir", json={}).status_code == 400
        assert self.client.post("/api/appris/refuser", json={}).status_code == 400

    def test_an_already_struck_proposal_cannot_be_accepted(self):
        """Refusal is as durable here as in the file. A click that could
        resurrect one is a way for a refused belief to come back."""
        from src.jarvis.memory.core import SECTION_PROFILE

        rayee = [p for p in self._lignes() if p["etat"] == "rayée"][0]

        r = self.client.post("/api/appris/retenir", json={"ligne": rayee["ligne"]})

        assert r.get_json()["retenue"] is False
        assert self._core(SECTION_PROFILE) == []

    def test_a_proposal_carrying_a_redaction_placeholder_is_not_written(self):
        """Same guard as the harvest: a placeholder stored as a belief is
        a belief about nothing, read back in every future prompt."""
        from src.jarvis.memory.core import SECTION_PROFILE
        from src.jarvis.appris.page import invalidate_appris_cache

        self.page.write_text(
            "## Profil\n- [ ] 2026-08-04 · journal : Son adresse est [REDACTED_EMAIL].\n",
            encoding="utf-8")
        invalidate_appris_cache()

        r = self.client.post("/api/appris/retenir",
                             json={"ligne": self._lignes()[0]["ligne"]})

        assert r.get_json()["retenue"] is False
        assert self._core(SECTION_PROFILE) == []

    def test_a_proposal_under_an_unknown_heading_is_not_written(self):
        from src.jarvis.appris.page import invalidate_appris_cache

        self.page.write_text(
            "## Divers\n- [ ] 2026-08-04 · journal : Quelque part.\n", encoding="utf-8")
        invalidate_appris_cache()

        r = self.client.post("/api/appris/retenir",
                             json={"ligne": self._lignes()[0]["ligne"]})

        assert r.get_json()["retenue"] is False

    def test_the_tab_tells_a_refusal_from_an_acceptance(self):
        """Both are struck. Only one carries a stamp, and telling him he
        refused something he agreed to is the kind of small lie that
        makes a record useless."""
        ligne = self._lignes()[0]["ligne"]
        self.client.post("/api/appris/retenir", json={"ligne": ligne})
        autre = [p for p in self._lignes() if p["etat"] == "attente"][0]["ligne"]
        self.client.post("/api/appris/refuser", json={"ligne": autre})

        par_texte = {p["texte"]: p for p in self._lignes()}
        assert "retenu le" in par_texte["Il court le mardi matin."]["tampon"]
        assert par_texte["Il a un chat qui s'appelle Miso."]["tampon"] == ""
