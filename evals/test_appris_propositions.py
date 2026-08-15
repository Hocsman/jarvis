"""
The mirror of the graph extractor, on the same summaries.

`evals/test_graph_user_exclusion.py` pins one half of a contract: the
graph takes the *world* out of a diary summary and must drop everything
about the person, because the person belongs in the core. This file pins
the other half on the same fixtures with the assertions inverted, and one
test runs both readers over the same notes to assert their outputs are
disjoint.

Two suites green in one run is the only evidence that the two readers
partition a summary rather than double-storing it. Either alone can pass
while the pair quietly overlaps.

The assertion message is the exclusion eval's, inverted, and it is the
point of the file: dropping a real statement of his is as wrong as
inventing one. A reader tuned only to be safe proposes nothing, and a
feature that proposes nothing is a feature he stops asking.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh appris_propositions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from unittest.mock import MagicMock

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, MockConfig


@dataclass
class MirrorCase:
    summary: str
    date_utc: str = "2026-04-20"
    # Substrings that must appear in at least one proposal. A tuple means
    # any one of its members satisfies the match, for when the model
    # rephrases — or translates, which is tested separately below rather
    # than conflated with whether the substance survived at all.
    expected: List[Union[str, Tuple[str, ...]]] = field(default_factory=list)
    # Substrings that must appear in NO proposal.
    forbidden: List[str] = field(default_factory=list)


# The exclusion eval's fixtures, with `expected` and `forbidden` swapped.
CASES = [
    pytest.param(
        MirrorCase(
            summary=(
                "The user mentioned they live in Brighton and have two "
                "cats, Miso and Kuma. They've been vegetarian for five "
                "years and work as a backend engineer."
            ),
            expected=["Brighton", ("Miso", "Kuma"),
                      ("vegetarian", "végétarien")],
        ),
        id="A pure summary of him is entirely proposals",
    ),
    pytest.param(
        MirrorCase(
            summary=(
                "The user told me to always answer in British English, "
                "to keep replies under three sentences, and to never "
                "apologise or say sorry. They also asked me to address "
                "them as Boss going forward."
            ),
            expected=[("British English", "three sentences", "Boss")],
        ),
        id="Standing instructions are proposals for the rules file",
    ),
    pytest.param(
        MirrorCase(
            summary=(
                "The user asked about Trenches Boxing Club. I found that "
                "it's on Mare Street in Hackney, offers evening classes "
                "on weekdays from 6-8pm at 15 pounds per session. I also "
                "confirmed that Possessor is a 2020 sci-fi horror film "
                "directed by Brandon Cronenberg."
            ),
            forbidden=["Mare Street", "Possessor", "Cronenberg", "Hackney"],
        ),
        id="A summary of lookups yields nothing about him",
    ),
    pytest.param(
        MirrorCase(
            summary=(
                "The user said they prefer Thai food over Italian when "
                "eating out. They also told me to keep all food "
                "recommendations under five options. I looked up Som Saa "
                "in Spitalfields: it's a Thai restaurant that takes "
                "bookings up to two months ahead."
            ),
            expected=[("Thai", "thaï", "five options", "cinq options")],
            forbidden=["Som Saa", "Spitalfields"],
        ),
        id="Interleaved: the preference and the rule survive, the lookup does not",
    ),
]


def _cfg(tmp_path):
    cfg = MockConfig()
    cfg.db_path = str(tmp_path / "jarvis.db")
    cfg.llm_chat_model = JUDGE_MODEL
    cfg.appris_model = ""
    cfg.appris_jours = 400
    cfg.appris_max_propositions = 6
    cfg.appris_seuil_doublon = 90
    cfg.appris_timeout_sec = 60.0
    return cfg


def _db(case: MirrorCase):
    db = MagicMock()
    db.get_recent_conversation_summaries.return_value = [
        {"date_utc": case.date_utc, "summary": case.summary}
    ]
    db.journal_deja_lu.return_value = {}
    return db


def _core(tmp_path):
    from jarvis.memory.core import MemoryCore
    return MemoryCore(tmp_path / "yuba")


def _propose(tmp_path, case: MirrorCase):
    from jarvis.appris.propose import propositions

    lecture = propositions(_cfg(tmp_path), _db(case),
                           core=_core(tmp_path), deja=[])
    return lecture


@pytest.mark.eval
@requires_judge_llm
class TestSheProposesWhatTheGraphRefuses:

    @pytest.mark.parametrize("case", CASES)
    def test_the_person_comes_through_and_the_world_does_not(self, tmp_path, case):
        lecture = _propose(tmp_path, case)
        textes = " || ".join(c.texte for c in lecture.gardes)

        print(f"\n  Summary: {case.summary[:90]}…")
        print(f"  Proposals: {textes or '(none)'}")
        print(f"  Set aside: connus={lecture.connus} infondes={lecture.infondes} "
              f"masques={lecture.masques} mal_formes={lecture.mal_formes}")

        assert lecture.appelee, "the journal was not read at all"

        bas = textes.lower()
        for attendu in case.expected:
            formes = (attendu,) if isinstance(attendu, str) else attendu
            assert any(f.lower() in bas for f in formes), (
                f"None of {formes!r} was proposed. Dropping a real statement "
                f"of his is as wrong as inventing one: a reader tuned only to "
                f"be safe proposes nothing, and a feature that proposes "
                f"nothing is a feature he stops asking. Proposals: {textes!r}"
            )
        for interdit in case.forbidden:
            assert interdit.lower() not in bas, (
                f"{interdit!r} was proposed. That is the world, not him, and "
                f"the graph already keeps it — proposing it here stores the "
                f"same thing twice, in the file he reads as being about "
                f"himself. Proposals: {textes!r}"
            )


@pytest.mark.eval
@requires_judge_llm
class TestTheTwoReadersPartitionTheSameNote:
    """Neither suite alone can prove this. Both green in one run can."""

    def test_no_fact_is_stored_by_both(self, tmp_path):
        from jarvis.appris.propose import propositions
        from jarvis.memory.graph import normalise_fact
        from jarvis.memory.graph_ops import extract_graph_memories

        case = CASES[3].values[0]   # the interleaved one, the hard case
        cfg = _cfg(tmp_path)

        mien = {normalise_fact(c.texte)
                for c in propositions(cfg, _db(case), core=_core(tmp_path),
                                      deja=[]).gardes}
        sien = {normalise_fact(f)
                for f in (extract_graph_memories(case.summary, cfg, JUDGE_MODEL) or [])}

        print(f"\n  Proposals about him: {sorted(mien)}")
        print(f"  Facts about the world: {sorted(sien)}")

        assert not (mien & sien), (
            f"The same sentence is stored in both his profile and the world "
            f"graph: {sorted(mien & sien)!r}. The two readers are supposed to "
            f"partition a note, and a fact in both places is a fact that can "
            f"disagree with itself later."
        )


@pytest.mark.eval
@requires_judge_llm
class TestWhatIsNeverAProposal:
    """Two shapes the exclusion eval has no counterpart for."""

    def test_her_own_conclusion_about_him_is_not_a_proposal(self, tmp_path):
        """She noticed he seemed tired. He never said it. Writing that
        into his profile is the deduction this whole fork refuses."""
        case = MirrorCase(summary=(
            "The user asked about deadlines. The assistant noted they seemed "
            "stressed and suggested taking a break."
        ))

        textes = " ".join(c.texte for c in _propose(tmp_path, case).gardes).lower()

        assert "stress" not in textes and "break" not in textes, textes

    def test_a_one_off_want_is_not_a_fact_about_him(self, tmp_path):
        """He needed a plumber on Tuesday. That is not who he is."""
        case = MirrorCase(summary=(
            "The user asked the assistant to find a plumber near them."
        ))

        textes = " ".join(c.texte for c in _propose(tmp_path, case).gardes).lower()

        assert "plumb" not in textes, textes

    def test_small_talk_yields_the_ordinary_empty_answer(self, tmp_path):
        case = MirrorCase(summary=(
            "The user greeted the assistant and asked what time it was. The "
            "assistant said it was 14:20."
        ))

        lecture = _propose(tmp_path, case)

        assert lecture.appelee
        assert lecture.gardes == [], [c.texte for c in lecture.gardes]


@pytest.mark.eval
@requires_judge_llm
class TestTheCitationIsReal:

    @pytest.mark.parametrize("case", CASES)
    def test_every_citation_is_a_substring_of_the_note(self, tmp_path, case):
        """The citation is what lets him check a sentence about himself
        rather than trust it. The guard drops ungrounded ones, so this
        can only fail if the guard itself broke."""
        from jarvis.memory.graph import normalise_fact

        note = normalise_fact(case.summary)
        for c in _propose(tmp_path, case).gardes:
            assert normalise_fact(c.citation) in note, (
                f"{c.citation!r} is not in the note it claims to come from.")


@pytest.mark.eval
@requires_judge_llm
class TestItDoesNotTranslateHim:
    """The proposal keeps the language of the note it came from.

    The rule exists because he corrects these by hand: a sentence
    translated on the way out is a sentence he has to translate back
    before he can judge whether it is even true. It is also the half of
    the contract the summariser now holds up on its own side, so a
    translation here would undo it one layer down.

    Measured rather than assumed: the model does translate sometimes,
    which is why this is its own test with its own name in the output,
    instead of a content assertion that goes red for the wrong reason.
    """

    def test_an_english_note_yields_english_proposals(self, tmp_path):
        case = MirrorCase(summary=(
            "The user mentioned they live in Brighton and have two cats, "
            "Miso and Kuma. They've been vegetarian for five years."
        ))

        textes = " ".join(c.texte for c in _propose(tmp_path, case).gardes).lower()

        print(f"\n  Proposals: {textes}")
        francais = [m for m in ("l'utilisateur", "il a ", "il vit", "végétarien",
                                "habite") if m in textes]
        assert not francais, (
            f"An English note produced French proposals ({francais!r}). He "
            f"corrects these by hand, and a translated sentence is one he has "
            f"to translate back before he can judge whether it is true. "
            f"Proposals: {textes!r}"
        )


@pytest.mark.eval
@requires_judge_llm
class TestItWritesInTheLanguageHeAsksFor:
    """The proposal is written in his language, not the note's.

    The note is not his words. It is LLM #9's paraphrase of what he said,
    and until today that context wrote English for French conversations —
    measured, two of his ten most recent rows were French. So "never
    translate, keep his words" protects nothing one layer down: there is
    nothing of his left to keep.

    What is downstream is his file, which he reads and corrects by hand,
    and which the suppression guards compare against. A proposal arriving
    in another language is one he must translate before he can judge
    whether it is even true, and one that no lexical guard can match
    against what he already believes.

    The code names no language. `response_language` is his own setting;
    when it is empty the note's language is kept, which is the old
    behaviour and the right default for someone who never set it.
    """

    def _cfg_langue(self, tmp_path, langue):
        cfg = _cfg(tmp_path)
        cfg.response_language = langue
        return cfg

    def _textes(self, tmp_path, langue):
        from jarvis.appris.propose import propositions

        case = MirrorCase(summary=(
            "The user mentioned they live in Brighton and have two cats, "
            "Miso and Kuma. They've been vegetarian for five years."
        ))
        lecture = propositions(self._cfg_langue(tmp_path, langue), _db(case),
                               core=_core(tmp_path), deja=[])
        return " ".join(c.texte for c in lecture.gardes).lower()

    def test_an_english_note_yields_french_proposals_when_he_asks_for_french(self, tmp_path):
        textes = self._textes(tmp_path, "français")

        print(f"\n  [français] {textes}")
        assert textes, "nothing was proposed at all"
        assert any(m in textes for m in ("il ", "brighton", "chats", "végétarien")), textes
        assert "they " not in textes, (
            f"The proposal is still in the note's language. He reads this in a "
            f"file of his own and corrects it by hand. Proposals: {textes!r}")

    def test_it_names_no_language_of_its_own(self, tmp_path):
        """Set to something the codebase has never heard of. If it works,
        nothing was hardcoded; if only French works, something was."""
        textes = self._textes(tmp_path, "español")

        print(f"\n  [español] {textes}")
        assert textes
        assert any(m in textes for m in ("el ", "vive", "gatos", "vegetariano")), textes

    def test_with_no_setting_the_notes_language_is_kept(self, tmp_path):
        """The old behaviour, and the right default: somebody who never
        set a language has not asked to be translated."""
        textes = self._textes(tmp_path, "")

        print(f"\n  [aucun réglage] {textes}")
        assert textes
        assert "brighton" in textes
