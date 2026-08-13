"""A heading he annotated must not hand its tools to the section above.

`outils.md` is generated once and then belongs to him: the file says so,
and the point of it is that he opens it and edits it. Writing
`## Jamais (jamais, vraiment)` beside a section is exactly the kind of
note a person makes in their own file.

The parser required a heading to be one bare word on its own line, so an
annotated one matched nothing at all — and a line that is not a heading
leaves the current verdict in force. Everything he filed under `## Jamais`
therefore inherited `## Libre`, and the tools he most wanted stopped ran
without asking.

Two things close it. A heading may carry a note after its name. And any
line that opens with `##` and is not a heading this file understands
clears the section rather than continuing the previous one — because the
inheritance, not the annotation, is what turned a typo into a permission.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.policy import ASK, FREE, NEVER, RISK_DESTRUCTIVE, RISK_READ, ToolPolicy


def _p(texte):
    return ToolPolicy.parse(texte)


# ── A heading he annotated ────────────────────────────────────────────


@pytest.mark.parametrize("titre", [
    "## Jamais (jamais, vraiment)",
    "## Jamais — rien ici ne tourne",
    "## Jamais  # attention",
    "## Libre (elle le fait toute seule)",
])
def test_a_note_after_the_name_does_not_lose_the_section(titre):
    verdict_attendu = NEVER if "Jamais" in titre else FREE

    p = _p(f"## Libre\n- webSearch\n{titre}\n- localFiles\n")

    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == verdict_attendu


def test_the_case_that_made_this_a_permission_hole():
    """Filed under Jamais, inherited Libre, ran unattended."""
    p = _p("## Libre\n- webSearch\n## Jamais (jamais)\n- localFiles\n")

    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == NEVER


# ── And a heading nobody understands ──────────────────────────────────


@pytest.mark.parametrize("titre", ["## Ne jamais", "## Divers", "## Mes trucs",
                                   "##", "## "])
def test_a_heading_this_file_does_not_understand_stops_the_section(titre):
    """The inheritance is the defect, not the wording. Whatever he wrote,
    the tools under it must fall back to their risk default rather than
    quietly keeping the verdict above."""
    p = _p(f"## Libre\n- webSearch\n{titre}\n- localFiles\n")

    assert p.verdict("localFiles", RISK_DESTRUCTIVE) != FREE
    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == ASK


def test_what_came_before_the_broken_heading_is_untouched():
    p = _p("## Libre\n- webSearch\n## Divers\n- localFiles\n")

    assert p.verdict("webSearch", RISK_READ) == FREE


# ── What must keep working ────────────────────────────────────────────


def test_a_plain_file_is_unchanged():
    p = _p("## Libre\n- webSearch\n- getWeather\n\n## Demande\n- setRoutine\n\n"
           "## Jamais\n- localFiles\n")

    assert p.verdict("webSearch", RISK_READ) == FREE
    assert p.verdict("setRoutine", RISK_DESTRUCTIVE) == ASK
    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == NEVER


def test_headings_are_still_matched_whatever_their_case():
    p = _p("## JAMAIS\n- localFiles\n")

    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == NEVER


def test_a_wildcard_still_works_under_an_annotated_heading():
    p = _p("## Jamais (tout le serveur)\n- chrome__*\n")

    assert p.verdict("chrome__navigate", RISK_DESTRUCTIVE) == NEVER


def test_a_comment_is_still_skipped_and_does_not_break_a_section():
    p = _p("## Jamais\n<!-- une note -->\n- localFiles\n")

    assert p.verdict("localFiles", RISK_DESTRUCTIVE) == NEVER
