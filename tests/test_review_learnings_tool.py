"""One tool, and it only ever runs when he asks.

`reviewLearnings` does two things in one call, in this order: it harvests
whatever he has ticked since last time, then it reads the journal rows he
has not been asked about and writes new proposals into the page.

The order is what makes the file feel alive rather than administrative.
He ticks a line, forgets about it, asks again a week later, and what he
agreed to lands before she offers him anything new.

There is no schedule, no background pass and no start-up sweep. A routine
running while he sleeps is refused outright, and the refusal names the
real reason: not that it writes her memory, but that it reads his life
and forms an opinion about him, which is only defensible because he is in
the room to say no.

The other property tested here is loudness. "I found nothing in your
journal" and "I considered four things and set all four aside" are
different sentences. A suppression nobody can see is how a component
starts failing quietly, and this file has seven counters precisely so
that never happens.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")
        self.llm_chat_model = "test-model"
        self.appris_jours = 14
        self.appris_max_propositions = 3
        self.appris_seuil_doublon = 90
        self.appris_timeout_sec = 30.0


def _ctx(cfg):
    from src.jarvis.tools.base import ToolContext
    return ToolContext(
        db=MagicMock(), cfg=cfg, system_prompt="", original_prompt="",
        redacted_text="", max_retries=0, user_print=lambda *a, **k: None,
    )


def _tool():
    from src.jarvis.tools.registry import BUILTIN_TOOLS
    return BUILTIN_TOOLS["reviewLearnings"]


def _page(cfg, texte):
    from src.jarvis.appris.page import appris_path
    p = appris_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texte, encoding="utf-8")
    return p


def _lecture(**kw):
    from src.jarvis.appris.propose import Lecture
    return Lecture(**kw)


# ── The first law, pinned ─────────────────────────────────────────────


def test_asking_twice_with_no_tick_writes_nothing(tmp_path):
    """Time is not an actor. This is the test that would catch anything
    that started deciding on his behalf, and it is deliberately the
    dullest one in the file."""
    from src.jarvis.appris.propose import Candidat
    from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore

    cfg = _Cfg(tmp_path)
    lecture = _lecture(appelee=True, lues=[("2026-08-04", "d")],
                       gardes=[Candidat("fait", "Il court le mardi.", "runs", "2026-08-04")])

    with patch("src.jarvis.appris.propose.propositions", return_value=lecture):
        _tool().run({}, _ctx(cfg))
        _tool().run({}, _ctx(cfg))

    core = MemoryCore.for_config(cfg)
    assert core.active(SECTION_PROFILE) == []
    assert core.active(SECTION_RULES) == []


def test_a_pending_proposal_reaches_no_prompt(tmp_path):
    """`appris.md` is in nothing the model reads. A proposal is not a
    belief in any operational sense, which is the property that makes
    being wrong here cost nothing."""
    from src.jarvis.memory.core import MemoryCore, build_core_profile, format_warm_profile_block

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n- [ ] 2026-08-04 · journal : Il court le mardi.\n")

    bloc = format_warm_profile_block(build_core_profile(MemoryCore.for_config(cfg)))

    assert "court le mardi" not in (bloc or "")


# ── Harvest first, then read ──────────────────────────────────────────


def test_a_ticked_line_lands_before_anything_new_is_offered(tmp_path):
    from src.jarvis.memory.core import SECTION_PROFILE, MemoryCore

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n- [x] 2026-08-04 · journal : Il court le mardi.\n\n## Règles\n")

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True)):
        resultat = _tool().run({}, _ctx(cfg))

    assert resultat.success
    assert [e.text for e in MemoryCore.for_config(cfg).active(SECTION_PROFILE)] == [
        "Il court le mardi."]


def test_new_proposals_are_written_to_the_page(tmp_path):
    from src.jarvis.appris.page import appris_path
    from src.jarvis.appris.propose import Candidat

    cfg = _Cfg(tmp_path)
    lecture = _lecture(appelee=True, lues=[("2026-08-04", "d")],
                       gardes=[Candidat("fait", "Il a un chat.", "the cat", "2026-08-04")])

    with patch("src.jarvis.appris.propose.propositions", return_value=lecture):
        _tool().run({}, _ctx(cfg))

    assert "Il a un chat." in appris_path(cfg).read_text(encoding="utf-8")


def test_a_rule_lands_under_the_rules_heading(tmp_path):
    from src.jarvis.appris.page import SECTION_REGLES, load_appris
    from src.jarvis.appris.propose import Candidat

    cfg = _Cfg(tmp_path)
    lecture = _lecture(appelee=True, lues=[("2026-08-03", "d")],
                       gardes=[Candidat("regle", "Répondre en français.", "in French",
                                        "2026-08-03")])

    with patch("src.jarvis.appris.propose.propositions", return_value=lecture):
        _tool().run({}, _ctx(cfg))

    assert [p.section for p in load_appris(cfg)] == [SECTION_REGLES]


def test_the_window_is_recorded_only_when_it_was_read(tmp_path):
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=False)):
        _tool().run({}, ctx)

    assert not ctx.db.marquer_journal_lu.called


def test_a_reading_that_happened_records_its_window(tmp_path):
    cfg = _Cfg(tmp_path)
    ctx = _ctx(cfg)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=[("2026-08-04", "d")])):
        _tool().run({}, ctx)

    ctx.db.marquer_journal_lu.assert_called_once_with([("2026-08-04", "d")])


# ── Suppression is loud ───────────────────────────────────────────────


def test_nothing_found_and_everything_set_aside_are_different_sentences(tmp_path):
    cfg = _Cfg(tmp_path)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=[("2026-08-04", "d")])):
        rien = _tool().run({}, _ctx(cfg)).reply_text

    ecarte = _lecture(appelee=True, lues=[("2026-08-04", "d")], bruts=4,
                      connus=1, refuses=1, infondes=1, masques=1)
    with patch("src.jarvis.appris.propose.propositions", return_value=ecarte):
        beaucoup = _tool().run({}, _ctx(cfg)).reply_text

    assert rien != beaucoup
    assert "4" in beaucoup


def test_a_failed_reading_says_so_rather_than_claiming_nothing_was_new(tmp_path):
    """The difference this whole module is built around. Saying "nothing
    new" when the model timed out is a lie she would repeat every time
    the endpoint is down."""
    cfg = _Cfg(tmp_path)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=False)):
        texte = _tool().run({}, _ctx(cfg)).reply_text.lower()

    assert "journal" in texte
    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=[("2026-08-04", "d")])):
        rien = _tool().run({}, _ctx(cfg)).reply_text.lower()
    assert texte != rien


def test_a_truncated_window_is_announced(tmp_path):
    cfg = _Cfg(tmp_path)

    with patch("src.jarvis.appris.propose.propositions",
               return_value=_lecture(appelee=True, lues=[("2026-08-04", "d")],
                                     tronquee=True)):
        texte = _tool().run({}, _ctx(cfg)).reply_text

    assert texte


# ── Never while he sleeps ─────────────────────────────────────────────


def test_a_routine_cannot_reach_it(tmp_path):
    from src.jarvis.routines.scope import RoutineScope
    from src.jarvis.tools.registry import _out_of_scope

    scope = RoutineScope(nom="matin", outils=["reviewLearnings"], memoire=False)
    refus = _out_of_scope("reviewLearnings", scope, "lecture", "libre")

    assert refus is not None and refus.refused


def test_the_refusal_names_reading_his_life_not_writing_her_memory(tmp_path):
    """A refusal that states the wrong reason is the same defect class
    as a false success: it teaches the next reader something untrue."""
    from src.jarvis.routines.scope import RoutineScope
    from src.jarvis.tools.registry import _out_of_scope

    scope = RoutineScope(nom="matin", outils=["reviewLearnings"], memoire=False)
    texte = _out_of_scope("reviewLearnings", scope, "lecture", "libre").reply_text

    assert "mémoire de Yuba" not in texte


def test_a_routine_cannot_be_armed_with_it():
    from unittest.mock import MagicMock

    from src.jarvis.routines.eligibility import refuse_reason

    assert refuse_reason(MagicMock(), "reviewLearnings")


# ── The router can see it ─────────────────────────────────────────────


def test_its_job_fits_inside_the_routers_slice():
    """The router reads only the leading sentences that fit in its
    budget. Two tools have already shipped with their discriminating
    sentence past the cut."""
    from src.jarvis.tools.selection import _ROUTER_SUMMARY_CHARS, _router_summary

    resume = _router_summary(_tool().description)

    assert len(resume) <= _ROUTER_SUMMARY_CHARS
    assert "journal" in resume.lower()


# ── What it costs to ask ──────────────────────────────────────────────


def test_asking_what_she_noticed_costs_nothing(tmp_path):
    """`lecture`, so free by default.

    A tool that declares no risk falls to the gate's default, which is
    destructive, and destructive is settled by a click on a card and
    never by a spoken word. By voice that makes the whole feature
    unreachable — and unreachable in the quietest possible way, because
    the router simply never offers it and nothing anywhere says why.

    Its own writes go through the harvest, whose input is a checkbox he
    ticked by hand in his own editor. There is no sentence a model can
    emit that reaches `profil.md` through this call, so the read pricing
    is the honest one.
    """
    from src.jarvis.tools.policy import RISK_READ, resolve_risk

    assert resolve_risk("reviewLearnings", _tool(), {}) == RISK_READ


def test_it_is_free_under_a_policy_that_never_heard_of_it(tmp_path):
    """His `outils.md` was generated before this tool existed and the
    file is his — nothing rewrites it. A `lecture` tool absent from it
    must still be free, or every tool shipped after his file was written
    would silently need a card."""
    from unittest.mock import patch

    from src.jarvis.tools.policy import FREE, RISK_READ
    from src.jarvis.tools.registry import load_tool_policy

    cfg = _Cfg(tmp_path)
    with patch("src.jarvis.tools.registry.ensure_policy_file"):
        politique = load_tool_policy(cfg)

    assert politique.verdict("reviewLearnings", RISK_READ) == FREE
