"""
Knowledge Graph User-Exclusion Evaluations

The extractor feeds the World branch and nothing else: facts about the
user and standing instructions they issued are the core's business, and
the core is written only when the user explicitly asks or corrects (see
``src/jarvis/memory/core.spec.md``).

This eval pins the half of that contract a prompt change can silently
break. A model that starts extracting "the user is vegetarian" from a
summary puts a guessed belief into every future system prompt with no
date, no attribution, and nowhere for the user to see it came from. The
failure is invisible until the assistant confidently acts on something
nobody told it.

The cases are adversarial in the direction that matters: summaries where
user facts and lookups are interleaved, so the model has to keep
extracting the lookups while dropping everything about the person.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh graph_user_exclusion
    EVAL_JUDGE_MODEL=gpt-oss:20b ./scripts/run_evals.sh graph_user_exclusion
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import pytest

from conftest import requires_judge_llm
from helpers import MockConfig

from jarvis.memory.graph_ops import extract_graph_memories


# =============================================================================
# Test Data
# =============================================================================


@dataclass
class ExclusionCase:
    """A summary, what must survive extraction, and what must not."""

    summary: str
    date_utc: Optional[str] = None
    # Keywords that must appear in at least one extracted fact. A tuple
    # means any one of its strings satisfies the match, for when the
    # model paraphrases.
    expected: List[Union[str, Tuple[str, ...]]] = field(default_factory=list)
    # Keywords that must appear in NO extracted fact.
    forbidden: List[str] = field(default_factory=list)


EXCLUSION_CASES = [
    pytest.param(
        ExclusionCase(
            summary=(
                "The user mentioned they live in Brighton and have two "
                "cats, Miso and Kuma. They've been vegetarian for five "
                "years and work as a backend engineer."
            ),
            date_utc="2026-04-20",
            forbidden=["Brighton", "Miso", "Kuma", "vegetarian", "engineer"],
        ),
        id="Pure user summary yields nothing",
    ),
    pytest.param(
        ExclusionCase(
            summary=(
                "The user told me to always answer in British English, "
                "to keep replies under three sentences, and to never "
                "apologise or say sorry. They also asked me to address "
                "them as Boss going forward."
            ),
            date_utc="2026-04-20",
            forbidden=["British English", "three sentences", "apologise", "Boss"],
        ),
        id="Standing instructions are not knowledge",
    ),
    pytest.param(
        ExclusionCase(
            summary=(
                "The user asked about Trenches Boxing Club. I found that "
                "it's on Mare Street in Hackney, offers evening classes "
                "on weekdays from 6-8pm at 15 pounds per session. I also "
                "confirmed that Possessor is a 2020 sci-fi horror film "
                "directed by Brandon Cronenberg."
            ),
            date_utc="2026-04-20",
            expected=["Trenches", "Mare Street", "Possessor", "Cronenberg"],
        ),
        id="Lookups still come through",
    ),
    pytest.param(
        ExclusionCase(
            summary=(
                "The user said they prefer Thai food over Italian when "
                "eating out. They also told me to keep all food "
                "recommendations under five options. I looked up Som Saa "
                "in Spitalfields: it's a Thai restaurant that takes "
                "bookings up to two months ahead."
            ),
            date_utc="2026-04-20",
            expected=[("Som Saa", "Spitalfields")],
            forbidden=["prefer Thai", "five options"],
        ),
        id="Interleaved: the lookup survives, the preference and the rule do not",
    ),
    pytest.param(
        ExclusionCase(
            summary=(
                "The user has been vegetarian for three years and lives "
                "in central London. They told me to stop suggesting fish "
                "dishes when they ask about food. I confirmed that "
                "Mildreds in Covent Garden is a fully vegetarian "
                "restaurant with a Michelin Bib Gourmand rating."
            ),
            date_utc="2026-04-20",
            expected=["Mildreds"],
            forbidden=["vegetarian for three years", "central London", "fish dishes"],
        ),
        id="Interleaved: a restaurant fact beside a diet and a rule",
    ),
]


# =============================================================================
# Helpers
# =============================================================================


def _run_extraction(case: ExclusionCase, config: MockConfig) -> list[str]:
    return extract_graph_memories(
        summary=case.summary,
        cfg=config,
        chat_model=config.llm_chat_model,
        timeout_sec=config.llm_chat_timeout_sec,
        thinking=False,
        date_utc=case.date_utc,
    )


def _matches(facts: list[str], keyword: Union[str, Tuple[str, ...]]) -> bool:
    alternatives = (keyword,) if isinstance(keyword, str) else keyword
    lowered = [k.lower() for k in alternatives]
    return any(any(k in fact.lower() for k in lowered) for fact in facts)


# =============================================================================
# Tests
# =============================================================================


class TestGraphUserExclusion:
    """The extractor must never put the user into the knowledge graph."""

    @requires_judge_llm
    @pytest.mark.parametrize("case", EXCLUSION_CASES)
    def test_only_lookups_are_extracted(self, mock_config, case: ExclusionCase):
        facts = _run_extraction(case, mock_config)

        # Print for report visibility
        print(f"Extracted {len(facts)} facts:")
        for fact in facts:
            print(f"  {fact}")

        for keyword in case.forbidden:
            assert not _matches(facts, keyword), (
                f"{keyword!r} describes the user or an instruction they "
                f"gave, and must not be extracted. Facts: {facts}"
            )

        for keyword in case.expected:
            assert _matches(facts, keyword), (
                f"Expected a fact containing {keyword!r}, but no extracted "
                f"fact matched. Dropping lookups is as wrong as keeping "
                f"user facts. Facts: {facts}"
            )
