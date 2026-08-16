"""The rewrite must not be where provenance goes to die.

Observed on his machine, 2026-08-16, on the first fact written under the
new code:

    'As of August 2026, a DDR5 RAM stick cost 450 €, … driven by AI
     demand.'

No `· web`, no date. The merge step is why: when the destination node
already holds data — which `World` always does — an LLM rewrites the
whole node, and it does not reproduce a suffix it was never asked to
keep. That was written up as provenance "thinning over a node's life",
which understated it. The merge is the normal path, not the edge: a
suffix would almost never survive.

Asking the rewrite to preserve an exact format would be asking an LLM for
the guarantee this whole area exists because LLMs do not give. So the
repair is deterministic and happens after it: the claims a node carried
before the merge, and the claims added by this flush, are indexed by
their text; any line that comes back matching one gets its suffix again.

A line the merge reworded matches nothing and stays bare, which reads as
`inconnu` — the truthful answer, since after a rewrite nobody can say
what that sentence rested on.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.jarvis.memory.graph_ops import _reattach_sources, _source_index
from src.jarvis.memory.provenance import (
    SOURCE_TOOL,
    SOURCE_UNKNOWN,
    SOURCE_WEB,
    fact_line,
    fact_source,
)


ANCIEN = fact_line("Le DGX Spark a 128 Go", SOURCE_TOOL, "2026-01-02")
NOUVEAU = fact_line("Une barrette DDR5 coûte 450 €", SOURCE_WEB, "2026-08-16")


def test_a_claim_the_merge_returned_bare_gets_its_source_back():
    index = _source_index([ANCIEN, NOUVEAU])

    rendu = _reattach_sources(["Une barrette DDR5 coûte 450 €"], index)

    assert fact_source(rendu[0]) == SOURCE_WEB


def test_a_line_the_node_already_carried_keeps_its_own_source():
    """The merge hands back every line, old and new. An existing fact must
    not lose what it had just because the rewrite touched the node."""
    index = _source_index([ANCIEN, NOUVEAU])

    rendu = _reattach_sources(["Le DGX Spark a 128 Go"], index)

    assert fact_source(rendu[0]) == SOURCE_TOOL
    assert "2026-01-02" in rendu[0]


def test_a_line_the_merge_reworded_stays_unmarked():
    """Honest by construction: after a rewrite nobody can say what that
    sentence rested on, and `inconnu` is exactly that statement."""
    index = _source_index([ANCIEN, NOUVEAU])

    rendu = _reattach_sources(
        ["Le DGX Spark et le Mac Studio diffèrent sur la mémoire"], index)

    assert fact_source(rendu[0]) == SOURCE_UNKNOWN


def test_a_line_that_still_has_its_suffix_is_left_alone():
    """The rewrite sometimes does keep it. Re-attaching a second one would
    make the line unreadable."""
    index = _source_index([ANCIEN])

    rendu = _reattach_sources([ANCIEN], index)

    assert rendu == [ANCIEN]
    assert rendu[0].count("·") == 2


def test_the_wording_is_never_touched():
    """The control. A repair that rewrote content would be far worse than
    the problem it fixes."""
    index = _source_index([NOUVEAU])
    lignes = ["Une barrette DDR5 coûte 450 €",
              "Un fait sans rapport, jamais vu ailleurs"]

    rendu = _reattach_sources(lignes, index)

    from src.jarvis.memory.provenance import fact_text
    assert [fact_text(l) for l in rendu] == lignes


def test_matching_survives_case_and_spacing():
    index = _source_index([NOUVEAU])

    rendu = _reattach_sources(["une   barrette DDR5 coûte 450 €"], index)

    assert fact_source(rendu[0]) == SOURCE_WEB


def test_an_empty_line_is_passed_through():
    rendu = _reattach_sources(["", "  "], _source_index([NOUVEAU]))

    assert rendu == ["", "  "]


# ── And the whole way through, which is where it actually failed ───────


@patch("src.jarvis.memory.graph_ops.call_llm_direct")
def test_a_fact_written_into_a_populated_node_keeps_its_source(mock_llm, tmp_path):
    """The field failure, reproduced end to end.

    A populated node always takes the merge path, and the rewrite hands
    every line back bare — which is how the first fact written under the
    new code landed in his graph with no marker at all. The unit tests
    above would all have passed while this happened.
    """
    from src.jarvis.memory.graph import GraphMemoryStore
    from src.jarvis.memory.graph_ops import update_graph_from_dialogue

    store = GraphMemoryStore(str(tmp_path / "g.db"))
    store.update_node("world", data="Un fait déjà là, sans marqueur.")

    # Extraction, then the merge answering with bare lines, as it does.
    mock_llm.side_effect = [
        '["Une barrette DDR5 coûte 450 €"]',
        '{"facts": ["Un fait déjà là, sans marqueur.", '
        '"Une barrette DDR5 coûte 450 €"]}',
    ]

    update_graph_from_dialogue(
        store=store, summary="Les prix de la DDR5 ont grimpé.",
        cfg=None, chat_model="model", tools_used=["webSearch"],
        date_utc="2026-08-16",
    )

    data = store.get_node("world").data
    store.close()

    ligne = [l for l in data.split("\n") if "DDR5" in l]
    assert ligne, f"le fait doit être là : {data!r}"
    assert fact_source(ligne[0]) == SOURCE_WEB, (
        f"la fusion a mangé la source : {ligne[0]!r}"
    )


# ── Ce que la vraie fusion fait, mesuré sur sa base ────────────────────


# Verbatim, before and after one real merge on his machine, 2026-08-16.
# The rewrite moved the currency symbol, inserted narrow spaces, swapped
# a hyphen for U+2011 and changed one word. An exact key — even folded
# for case and whitespace — matches none of that, which is why the first
# repair wrote nothing.
AVANT = ("As of August 2026, a DDR5 RAM stick cost 450 €, a 171% increase "
         "from 80 € two years prior, driven by AI demand.")
APRES = ("As of August 2026, a DDR5 RAM stick cost €450, a 171 % increase "
         "from €80 two years earlier, driven by AI demand.")


def test_a_line_the_merge_retypeset_and_reworded_keeps_its_source():
    """The measured failure, turned into the test that had to exist.

    Same line, 94.2 similarity across the rewrite; the closest unrelated
    fact in the same node scores 33.9. The gap is what makes this safe.
    """
    index = _source_index([fact_line(AVANT, SOURCE_WEB, "2026-08-16")])

    rendu = _reattach_sources([APRES], index)

    assert fact_source(rendu[0]) == SOURCE_WEB


def test_a_different_fact_never_borrows_a_source():
    """The control, and the risk this introduces: a loose match would
    stamp `web` on a line that was never looked up."""
    index = _source_index([fact_line(AVANT, SOURCE_WEB, "2026-08-16")])

    rendu = _reattach_sources(
        ["Grand Theft Auto VI is scheduled for release on November 19, 2026."],
        index)

    assert fact_source(rendu[0]) == SOURCE_UNKNOWN


def test_a_fact_consolidated_beyond_recognition_falls_to_unknown():
    """When the rewrite folds several lines into a pattern, nothing
    matches and `inconnu` is the truthful answer."""
    index = _source_index([fact_line(AVANT, SOURCE_WEB, "2026-08-16")])

    rendu = _reattach_sources(["Les prix de la mémoire ont beaucoup monté."],
                              index)

    assert fact_source(rendu[0]) == SOURCE_UNKNOWN
