"""The tick is the write, and nothing else is.

There is no model on this path. A regular expression reads a checkbox and
`MemoryCore.remember` writes a line. There is no sentence any model can
emit, no page a web tool can return, and no mis-transcription that puts a
character between two brackets — which is the property that makes the
whole step defensible.

Two things are load-bearing here and both are about failure.

The order: the core is written first and the page marked second. If the
mark fails after a successful write, the next pass re-remembers and the
core's own duplicate scan makes that a no-op. Marked first, a failed core
write would delete the line from the page and tell him it worked, which
is the exact defect class this project keeps finding.

And the stop: the loop halts on the first write that raises. Partial work
is reported as partial. A harvest that swallowed three failures and
announced four successes would be the same lie in a quieter voice.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class _Cfg:
    def __init__(self, tmp_path):
        self.db_path = str(tmp_path / "jarvis.db")


def _core(cfg):
    from src.jarvis.memory.core import MemoryCore
    return MemoryCore.for_config(cfg)


def _page(cfg, texte):
    from src.jarvis.appris.page import appris_path
    p = appris_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(texte, encoding="utf-8")
    return p


UNE_COCHEE = """## Profil
- [x] 2026-08-04 · journal : Il court le mardi matin.
  > « the user mentioned running on Tuesday mornings »

## Règles
"""


# ── What a tick does ──────────────────────────────────────────────────


def test_a_ticked_proposal_lands_in_the_profile(tmp_path):
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    resultat = recolter(cfg)

    assert resultat.retenues == ["Il court le mardi matin."]
    assert [e.text for e in _core(cfg).active(SECTION_PROFILE)] == [
        "Il court le mardi matin."]


def test_it_carries_the_confirmed_source(tmp_path):
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    recolter(cfg)

    assert _core(cfg).active(SECTION_PROFILE)[0].source == "confirmé"


def test_it_carries_the_day_of_the_journal_not_today(tmp_path):
    """A belief belongs where the thing happened in his life. The day he
    ticked is stamped on the struck line in `appris.md`, which is the
    file whose job is provenance."""
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    recolter(cfg)

    assert _core(cfg).active(SECTION_PROFILE)[0].date == "2026-08-04"


def test_a_rules_tick_lands_in_the_rules(tmp_path):
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_RULES

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n\n## Règles\n- [x] 2026-08-03 · journal : Répondre en français.\n")

    recolter(cfg)

    assert [e.text for e in _core(cfg).active(SECTION_RULES)] == ["Répondre en français."]


def test_the_harvested_line_is_struck_and_stamped(tmp_path):
    from src.jarvis.appris.page import appris_path
    from src.jarvis.appris.recolte import recolter

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    recolter(cfg)

    texte = appris_path(cfg).read_text(encoding="utf-8")
    assert "~~2026-08-04 · journal : Il court le mardi matin.~~" in texte
    assert "retenu le " in texte


def test_what_he_rewrote_is_what_lands(tmp_path):
    """The reason the file exists. He fixes the sentence, and it is his
    version that describes him from then on."""
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n- [x] 2026-08-04 · journal : Il court le mardi, tôt le matin.\n")

    recolter(cfg)

    assert _core(cfg).active(SECTION_PROFILE)[0].text == "Il court le mardi, tôt le matin."


# ── What a tick does not do ───────────────────────────────────────────


def test_an_untouched_box_lands_nothing(tmp_path):
    """Time is not an actor. This is the first law, and it is the test
    that would catch anything that started deciding on his behalf."""
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE.replace("[x]", "[ ]"))

    recolter(cfg)
    recolter(cfg)

    assert _core(cfg).active(SECTION_PROFILE) == []


def test_a_struck_proposal_lands_nothing(tmp_path):
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n- ~~2026-08-04 · journal : Non.~~\n")

    recolter(cfg)

    assert _core(cfg).active(SECTION_PROFILE) == []


def test_a_tick_under_an_unknown_heading_lands_nothing(tmp_path):
    """Nothing says which of his files `## Divers` would go to, and
    guessing is how you write into the wrong one."""
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE, SECTION_RULES

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Divers\n- [x] 2026-08-04 · journal : Quelque part.\n")

    resultat = recolter(cfg)

    assert resultat.hors_section == 1
    assert _core(cfg).active(SECTION_PROFILE) == []
    assert _core(cfg).active(SECTION_RULES) == []


def test_a_redacted_line_is_refused_and_the_tick_kept(tmp_path):
    """A placeholder stored as a belief is a belief about nothing, and
    it would be read back in every future prompt. The tick stays so he
    can fix the line rather than having to notice it vanished."""
    from src.jarvis.appris.page import appris_path
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n- [x] 2026-08-04 · journal : Son adresse est [REDACTED_EMAIL].\n")

    resultat = recolter(cfg)

    assert resultat.masquees == 1
    assert _core(cfg).active(SECTION_PROFILE) == []
    assert "[x]" in appris_path(cfg).read_text(encoding="utf-8")


def test_something_already_believed_consumes_the_tick_without_a_duplicate(tmp_path):
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE, SOURCE_SAID

    cfg = _Cfg(tmp_path)
    _core(cfg).remember(SECTION_PROFILE, "Il court le mardi matin.", source=SOURCE_SAID)
    _page(cfg, UNE_COCHEE)

    resultat = recolter(cfg)

    assert resultat.deja == 1
    assert len(_core(cfg).active(SECTION_PROFILE)) == 1


# ── When the write fails ──────────────────────────────────────────────


def test_a_failed_core_write_leaves_the_tick_in_place(tmp_path):
    """Marked first, this would delete his line and tell him it worked."""
    from src.jarvis.appris.page import appris_path
    from src.jarvis.appris.recolte import recolter

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    with patch("src.jarvis.memory.core.MemoryCore.remember",
               side_effect=OSError("disque plein")):
        resultat = recolter(cfg)

    assert resultat.echouees == 1
    assert resultat.retenues == []
    assert "[x]" in appris_path(cfg).read_text(encoding="utf-8")


def test_the_loop_stops_at_the_first_failure(tmp_path):
    """Partial work is reported as partial. Carrying on past a disk
    error and announcing three successes is the same lie, quieter."""
    from src.jarvis.appris.recolte import recolter

    cfg = _Cfg(tmp_path)
    _page(cfg, "## Profil\n"
               "- [x] 2026-08-04 · journal : Une.\n"
               "- [x] 2026-08-04 · journal : Deux.\n"
               "- [x] 2026-08-04 · journal : Trois.\n")

    with patch("src.jarvis.memory.core.MemoryCore.remember",
               side_effect=OSError("disque plein")):
        resultat = recolter(cfg)

    assert resultat.echouees == 1
    assert resultat.retenues == []


def test_a_failed_mark_is_recovered_by_the_next_pass(tmp_path):
    """Core first, mark second. The line is written but not struck, so
    the next harvest re-remembers it and the core's duplicate scan makes
    that a no-op instead of a second line."""
    from src.jarvis.appris.recolte import recolter
    from src.jarvis.memory.core import SECTION_PROFILE

    cfg = _Cfg(tmp_path)
    _page(cfg, UNE_COCHEE)

    with patch("src.jarvis.appris.recolte.marquer_retenue", return_value=False):
        recolter(cfg)

    resultat = recolter(cfg)

    assert resultat.deja == 1
    assert len(_core(cfg).active(SECTION_PROFILE)) == 1


def test_nothing_to_harvest_is_not_an_error(tmp_path):
    from src.jarvis.appris.recolte import recolter

    cfg = _Cfg(tmp_path)

    resultat = recolter(cfg)

    assert resultat.retenues == []
    assert resultat.echouees == 0
