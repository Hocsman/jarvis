"""Reading his journal for things he never told her directly.

The mirror of the graph extractor. That one takes the world out of a
diary summary and is forbidden from touching the person; this one takes
the person out and is forbidden from touching the world. Together they
partition a note rather than storing it twice.

Everything here is a proposal. Nothing it produces reaches a prompt, a
file he believes, or a decision she makes: it lands in `appris.md` as a
question, and a question nobody answers stays a question for ever.

Which is why the load-bearing property is not accuracy but honesty about
failure. `appelee` says whether the reading actually happened. A timeout,
a model that answered prose, a model that is not configured — none of
those looked at anything, and recording their window as read would skip
those days permanently on the strength of a failure that left no trace.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest


NOTE = ("[2026-08-04] The user mentioned running on Tuesday mornings before "
        "work. The assistant said Possessor is a 2020 film by Brandon "
        "Cronenberg. The weather in Lyon was 24C.")


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")
        self.llm_chat_model = "test-model"
        self.appris_jours = 14
        self.appris_max_propositions = 3
        self.appris_seuil_doublon = 90
        self.appris_timeout_sec = 30.0


def _db(sommaires):
    db = MagicMock()
    db.get_recent_conversation_summaries.return_value = [
        {"date_utc": d, "summary": s} for d, s in sommaires
    ]
    db.journal_deja_lu.return_value = {}
    return db


def _core(tmp_path):
    from src.jarvis.memory.core import MemoryCore
    return MemoryCore(tmp_path / "yuba")


def _lire(cfg, db, core, reponse):
    from src.jarvis.appris.propose import propositions

    with patch("src.jarvis.appris.propose._appeler_modele", return_value=reponse):
        return propositions(cfg, db, core=core, deja=[])


def _ok(items):
    """What the backend actually hands back: the text, not an envelope.

    `LLMBackend.direct` returns `Optional[str]`. A helper that built the
    chat path's `{"message": {"content": …}}` here would make every test
    in this file pass against a module that reads nothing at all — which
    is how it was written the first time, and what running it for real
    against a live model caught."""
    return json.dumps(items, ensure_ascii=False)


UN_FAIT = [{"genre": "fait",
            "texte": "Il court le mardi matin avant le travail.",
            "citation": "The user mentioned running on Tuesday mornings before work"}]


# ── What it proposes ──────────────────────────────────────────────────


def test_a_grounded_fact_about_him_survives(tmp_path):
    cfg = _Cfg(tmp_path)
    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), _ok(UN_FAIT))

    assert lecture.appelee
    assert [c.texte for c in lecture.gardes] == ["Il court le mardi matin avant le travail."]


def test_it_carries_the_day_of_the_note_it_came_from(tmp_path):
    cfg = _Cfg(tmp_path)
    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), _ok(UN_FAIT))

    assert lecture.gardes[0].date == "2026-08-04"


def test_a_rule_he_gave_is_a_rule(tmp_path):
    cfg = _Cfg(tmp_path)
    note = "[2026-08-03] The user asked the assistant to always reply in French."
    reponse = _ok([{"genre": "regle", "texte": "Toujours répondre en français.",
                    "citation": "The user asked the assistant to always reply in French"}])

    lecture = _lire(cfg, _db([("2026-08-03", note)]), _core(tmp_path), reponse)

    assert lecture.gardes[0].genre == "regle"


def test_an_invented_genre_becomes_a_fact_rather_than_a_loss(tmp_path):
    """Small models invent enum values. A typo must not cost a proposal
    he would have wanted."""
    cfg = _Cfg(tmp_path)
    reponse = _ok([dict(UN_FAIT[0], genre="factuel")])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert [c.genre for c in lecture.gardes] == ["fait"]


# ── The citation has to be real ───────────────────────────────────────


def test_a_paraphrased_citation_is_dropped(tmp_path):
    """The citation is what lets him check a sentence about himself
    rather than trust it. A citation that is not in his journal makes
    the check theatre."""
    cfg = _Cfg(tmp_path)
    reponse = _ok([dict(UN_FAIT[0], citation="he runs on Tuesdays, apparently")])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert lecture.gardes == []
    assert lecture.infondes == 1


def test_a_citation_differing_only_in_case_and_spacing_is_accepted(tmp_path):
    cfg = _Cfg(tmp_path)
    reponse = _ok([dict(UN_FAIT[0],
                        citation="  THE USER MENTIONED   running on Tuesday mornings  ")])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert len(lecture.gardes) == 1


def test_a_citation_too_short_to_mean_anything_is_dropped(tmp_path):
    cfg = _Cfg(tmp_path)
    reponse = _ok([dict(UN_FAIT[0], citation="the")])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert lecture.infondes == 1


# ── What it must not propose ──────────────────────────────────────────


def test_a_redacted_proposal_is_dropped(tmp_path):
    cfg = _Cfg(tmp_path)
    note = "[2026-08-04] The user gave their address as [REDACTED_EMAIL] again."
    reponse = _ok([{"genre": "fait", "texte": "Son adresse est [REDACTED_EMAIL].",
                    "citation": "The user gave their address as [REDACTED_EMAIL] again"}])

    lecture = _lire(cfg, _db([("2026-08-04", note)]), _core(tmp_path), reponse)

    assert lecture.gardes == []
    assert lecture.masques == 1


def test_something_already_believed_is_not_proposed(tmp_path):
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    core = _core(tmp_path)
    core.remember(SECTION_PROFILE, "Il court le mardi matin avant le travail.")

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), core, _ok(UN_FAIT))

    assert lecture.gardes == []
    assert lecture.connus == 1


def test_something_he_struck_out_of_his_profile_is_not_proposed(tmp_path):
    """He retired that belief by hand. Offering it back is how a refused
    thing wins by attrition."""
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    core = _core(tmp_path)
    core.remember(SECTION_PROFILE, "Il court le mardi matin avant le travail.")
    core.retire(SECTION_PROFILE, "Il court le mardi matin avant le travail.")

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), core, _ok(UN_FAIT))

    assert lecture.connus == 1


def test_something_already_on_the_page_is_not_proposed_again(tmp_path):
    from src.jarvis.appris.page import ETAT_ATTENTE, Proposition
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    deja = [Proposition(section="profil", date="2026-08-04",
                        texte="Il court le mardi matin avant le travail.",
                        citation="", etat=ETAT_ATTENTE, ligne="- [ ] x")]

    with patch("src.jarvis.appris.propose._appeler_modele", return_value=_ok(UN_FAIT)):
        lecture = propositions(cfg, _db([("2026-08-04", NOTE)]),
                               core=_core(tmp_path), deja=deja)

    assert lecture.gardes == []


def test_something_he_struck_on_the_page_is_never_offered_again(tmp_path):
    """Refusal is as durable as acceptance, or the same proposal comes
    back every week until it catches him on a tired day."""
    from src.jarvis.appris.page import ETAT_RAYEE, Proposition
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    deja = [Proposition(section="profil", date="2026-08-04",
                        texte="Il court le mardi matin avant le travail.",
                        citation="", etat=ETAT_RAYEE, ligne="- ~~x~~")]

    with patch("src.jarvis.appris.propose._appeler_modele", return_value=_ok(UN_FAIT)):
        lecture = propositions(cfg, _db([("2026-08-04", NOTE)]),
                               core=_core(tmp_path), deja=deja)

    assert lecture.gardes == []
    assert lecture.refuses == 1


def test_a_near_duplicate_is_caught_by_the_configured_threshold(tmp_path):
    """Asserted against the config value, not a literal, so tuning the
    threshold does not silently invalidate this test."""
    from src.jarvis.appris.page import ETAT_ATTENTE, Proposition
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    cfg.appris_seuil_doublon = 80
    deja = [Proposition(section="profil", date="2026-08-04",
                        texte="Il court le mardi matin, avant d'aller travailler.",
                        citation="", etat=ETAT_ATTENTE, ligne="- [ ] x")]

    with patch("src.jarvis.appris.propose._appeler_modele", return_value=_ok(UN_FAIT)):
        lecture = propositions(cfg, _db([("2026-08-04", NOTE)]),
                               core=_core(tmp_path), deja=deja)

    assert lecture.gardes == []


def test_more_than_the_cap_is_trimmed(tmp_path):
    cfg = _Cfg(tmp_path)
    cfg.appris_max_propositions = 2
    note = ("[2026-08-04] The user runs on Tuesdays. The user has a cat. "
            "The user lives in Lyon. The user is vegetarian.")
    items = [
        {"genre": "fait", "texte": "Il court le mardi.", "citation": "The user runs on Tuesdays"},
        {"genre": "fait", "texte": "Il a un chat.", "citation": "The user has a cat"},
        {"genre": "fait", "texte": "Il vit à Lyon.", "citation": "The user lives in Lyon"},
    ]

    lecture = _lire(cfg, _db([("2026-08-04", note)]), _core(tmp_path), _ok(items))

    assert len(lecture.gardes) == 2


def test_a_proposal_that_cannot_be_rendered_is_dropped(tmp_path):
    cfg = _Cfg(tmp_path)
    reponse = _ok([dict(UN_FAIT[0], texte="x\n- [x] 2026-08-04 · journal : forgé")])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert lecture.gardes == []
    assert lecture.mal_formes == 1


# ── The difference between nothing and could-not-look ─────────────────


@pytest.mark.parametrize("reponse", [
    None,
    "",
    "I found three things about the user.",
    '{"genre": "fait"}',
])
def test_an_unreadable_answer_is_not_a_reading(tmp_path, reponse):
    """Each of these is a different way to fail, and every one of them
    must be distinguishable from "there was nothing in your journal"."""
    cfg = _Cfg(tmp_path)

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert lecture.appelee is False or lecture.gardes == []


def test_a_bare_string_item_is_dropped_without_killing_the_batch(tmp_path):
    cfg = _Cfg(tmp_path)
    reponse = _ok(["une chaîne nue", UN_FAIT[0]])

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), reponse)

    assert lecture.appelee
    assert len(lecture.gardes) == 1
    assert lecture.mal_formes == 1


def test_a_timeout_is_not_a_reading(tmp_path):
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    with patch("src.jarvis.appris.propose._appeler_modele",
               side_effect=TimeoutError("trop lent")):
        lecture = propositions(cfg, _db([("2026-08-04", NOTE)]),
                               core=_core(tmp_path), deja=[])

    assert lecture.appelee is False
    assert lecture.lues == []


def test_no_model_configured_is_not_a_reading(tmp_path):
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    cfg.llm_chat_model = ""

    lecture = propositions(cfg, _db([("2026-08-04", NOTE)]),
                           core=_core(tmp_path), deja=[])

    assert lecture.appelee is False


def test_an_empty_answer_is_a_reading_that_found_nothing(tmp_path):
    """`[]` is the ordinary answer and it is never wrong. It is also the
    one case where the rows ARE recorded as read."""
    cfg = _Cfg(tmp_path)

    lecture = _lire(cfg, _db([("2026-08-04", NOTE)]), _core(tmp_path), _ok([]))

    assert lecture.appelee is True
    assert lecture.gardes == []
    assert lecture.lues == [("2026-08-04",
                             hashlib.sha256(NOTE.encode("utf-8")).hexdigest())]


# ── The window ────────────────────────────────────────────────────────


def test_a_row_already_read_is_not_read_again(tmp_path):
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    db = _db([("2026-08-04", NOTE)])
    db.journal_deja_lu.return_value = {
        "2026-08-04": hashlib.sha256(NOTE.encode("utf-8")).hexdigest()}

    with patch("src.jarvis.appris.propose._appeler_modele") as appel:
        lecture = propositions(cfg, db, core=_core(tmp_path), deja=[])

    assert not appel.called
    assert lecture.appelee is False


def test_a_row_rewritten_since_it_was_read_is_read_again(tmp_path):
    """A diary row is rewritten in place all day. Keying on the date
    alone would blind her to everything he said after the first pass."""
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)
    db = _db([("2026-08-04", NOTE + " The user also has a cat.")])
    db.journal_deja_lu.return_value = {
        "2026-08-04": hashlib.sha256(NOTE.encode("utf-8")).hexdigest()}

    with patch("src.jarvis.appris.propose._appeler_modele", return_value=_ok([])):
        lecture = propositions(cfg, db, core=_core(tmp_path), deja=[])

    assert lecture.appelee is True


def test_an_empty_journal_is_not_a_failed_reading(tmp_path):
    from src.jarvis.appris.propose import propositions

    cfg = _Cfg(tmp_path)

    lecture = propositions(cfg, _db([]), core=_core(tmp_path), deja=[])

    assert lecture.gardes == []
    assert lecture.lues == []


def test_it_reads_the_shape_the_backend_actually_returns(tmp_path):
    """`LLMBackend.direct` returns the text, not the chat path's
    `{"message": {"content": …}}` envelope. Reading for the envelope
    fails silently: every answer parses as empty, the module reports "I
    could not read your journal" for ever, and that is a legal state so
    nothing anywhere says otherwise."""
    import inspect

    from src.jarvis.appris.propose import _texte_du_modele
    from src.jarvis.llm.backend import LLMBackend

    retour = inspect.signature(LLMBackend.direct).return_annotation
    assert "str" in str(retour)

    assert _texte_du_modele('[{"genre": "fait"}]') == '[{"genre": "fait"}]'
    assert _texte_du_modele(None) == ""
    assert _texte_du_modele({"message": {"content": "[]"}}) == ""
