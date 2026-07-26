"""
End-to-end eval — a correction in the core beats a stale diary entry.

`core.spec.md` calls the core "the sole authority for what the assistant
believes about the user". Two other channels also carry the user into the
prompt: dated diary summaries, and query-driven graph recall. Only the
core is written on purpose, and only the core can be corrected.

The failure this catches is the one that makes correcting feel pointless.
The user says "no, I moved to Lyon", the core is updated, and the next
question is answered "Paris" from a diary entry written weeks ago. From
where they sit, the correction did nothing, and there is nowhere to look
to find out why.

Seeded so the two channels disagree on purpose, with the stale value in
the more voluminous one, because that is the shape a real memory takes:
months of diary against one corrected line.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh core_outranks_the_diary
"""

from dataclasses import dataclass, field
from typing import List
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    seed_diary_summaries,
    JUDGE_MODEL,
)


@dataclass
class ConflictCase:
    text: str
    # What the core holds, and therefore what must win.
    core_facts: List[str] = field(default_factory=list)
    core_rules: List[str] = field(default_factory=list)
    # Older diary entries carrying the superseded value.
    diary: List[tuple] = field(default_factory=list)
    # One of these must appear in the reply. Several spellings because
    # the core is written in the user's language and the model answers in
    # whichever the conversation is in: what is under test is which
    # channel won, not which language it was phrased in.
    expect: tuple = ()
    # Must not appear in the reply.
    reject: str = ""


CONFLICT_CASES = [
    pytest.param(
        ConflictCase(
            text="Jarvis, where do I live?",
            core_facts=["Il vit à Lyon."],
            diary=[
                ("2026-05-02", "The user mentioned they live in Paris, in the 11th."),
                ("2026-05-14", "The user talked about their commute across Paris."),
                ("2026-06-03", "The user recommended a bakery near their flat in Paris."),
            ],
            expect=("lyon",),
            reject="paris",
        ),
        id="A corrected city beats three older diary mentions",
    ),
    pytest.param(
        ConflictCase(
            text="Jarvis, am I still eating meat?",
            core_facts=["Il est végétarien."],
            diary=[
                ("2026-04-11", "The user described the steak they cooked at the weekend."),
                ("2026-04-20", "The user asked for a recipe for roast chicken."),
            ],
            expect=("végétarien", "vegetarian"),
            reject="steak",
        ),
        id="A corrected diet beats older diary evidence",
    ),
    pytest.param(
        ConflictCase(
            text="Jarvis, which city am I in these days?",
            core_facts=["Il vit à Lyon."],
            core_rules=["Toujours répondre en français."],
            diary=[
                ("2026-05-02", "The user asked for the weather in Paris several times."),
                ("2026-05-19", "The user described their walk home through Paris."),
            ],
            expect=("lyon",),
            reject="paris",
        ),
        id="A corrected city holds while a standing rule is also in play",
    ),
]


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "getWeather":
            location = ((tool_args or {}).get("location") or "").strip()
            return ToolExecutionResult(
                success=True,
                reply_text=f"Weather for {location or 'unknown'}: 14°C, light rain.",
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestCoreOutranksTheDiary:
    """What the user corrected wins over what the diary remembers."""

    @pytest.mark.parametrize("case", CONFLICT_CASES)
    def test_the_corrected_value_is_the_one_used(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: ConflictCase,
    ):
        from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.db_path = str(tmp_path / "jarvis.db")
        mock_config.location_enabled = False
        mock_config.memory_enrichment_source = "diary"

        core = MemoryCore.for_config(mock_config)
        for fact in case.core_facts:
            core.remember(SECTION_PROFILE, fact)
        for rule in case.core_rules:
            core.remember(SECTION_RULES, rule)
        seed_diary_summaries(eval_db, case.diary)

        capture = ToolCallCapture()
        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_runner(capture),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=case.text,
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Core vs diary ({JUDGE_MODEL}): {case.text}")
        print(f"  Core holds: {case.core_facts + case.core_rules}")
        print(f"  Diary says: {[s for _d, s in case.diary]}")
        print(f"  Tools called: {capture.tool_names()}")
        print(f"  Response: {(response or '')[:300]}")

        assert_not_fallback_reply(response, context="core-vs-diary")

        lowered = (response or "").lower()
        assert any(w.lower() in lowered for w in case.expect), (
            f"The reply carries none of {case.expect!r}, which is what the core "
            f"holds after a correction. Answering from the diary instead makes "
            f"correcting the assistant pointless from the user's side. "
            f"Response: {(response or '')[:400]}"
        )
        assert case.reject not in lowered, (
            f"The reply carries {case.reject!r}, the superseded value the diary "
            f"still remembers. The core is the authority for what is true about "
            f"the user. Response: {(response or '')[:400]}"
        )
