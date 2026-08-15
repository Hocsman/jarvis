"""A heading the parser does not understand ends the section above it.

`outils.md` is his, and its own header invites him to move lines between
three sections. So he reorganises: he writes `### Jamais` meaning a
sub-heading, or `##Jamais` with the space missed. Neither matches the
heading pattern, and neither used to end the section above — so
everything he filed under his own `## Jamais` inherited `## Libre`.

The guard against that already exists for a heading carrying a note. It
covered `## ` and stopped there, which left every other shape of `#`
falling through to the section above. Inheritance is what turns a typo
into a permission, so the rule is the complement of the heading pattern
rather than a second pattern that has to keep up with it.
"""

from __future__ import annotations

import pytest

from src.jarvis.tools.policy import ASK, FREE, NEVER, RISK_DESTRUCTIVE, ToolPolicy


FAUX_TITRES = ("### Demande", "#### Demande", "##Demande", "#Demande",
               "# Demande", "###Demande", "##  ", "#")


def test_no_shape_of_heading_lets_the_section_above_carry_on():
    for faux in FAUX_TITRES:
        texte = f"## Libre\n- webSearch\n{faux}\n- localFiles\n"

        verdict = ToolPolicy.parse(texte).verdict("localFiles", RISK_DESTRUCTIVE)

        assert verdict == ASK, f"{faux!r} a laissé « ## Libre » en vigueur"


def test_his_own_never_list_cannot_be_freed_by_a_typo():
    """The direction that costs the most. A mistyped heading above his
    forbidden list would hand every one of them back."""
    texte = "## Libre\n- webSearch\n### Jamais\n- deleteMeal\n- localFiles\n"

    politique = ToolPolicy.parse(texte)

    assert politique.verdict("deleteMeal", RISK_DESTRUCTIVE) == ASK
    assert politique.verdict("localFiles", RISK_DESTRUCTIVE) == ASK


def test_a_real_heading_still_opens_its_section():
    """The control. A guard that ended every section would make the file
    inert, which reads as safe and is not: everything would fall back to
    its risk default, and a `lecture` default is free."""
    texte = "## Libre\n- webSearch\n\n## Jamais\n- deleteMeal\n"
    politique = ToolPolicy.parse(texte)

    assert politique.verdict("webSearch", RISK_DESTRUCTIVE) == FREE
    assert politique.verdict("deleteMeal", RISK_DESTRUCTIVE) == NEVER


def test_a_heading_he_annotated_still_opens_its_section():
    """The case the first guard was written for, kept green."""
    texte = "## Jamais (jamais, vraiment)\n- deleteMeal\n"

    assert ToolPolicy.parse(texte).verdict("deleteMeal", RISK_DESTRUCTIVE) == NEVER
