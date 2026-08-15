"""
End-to-end eval — the assistant speaks of herself in the feminine.

The persona is a woman. In English that costs nothing, so it is invisible
until the assistant answers in a language that marks grammatical gender,
where every self-referential adjective and participle either lands or
grates. French is where this user hears it, on every reply.

The last case is the one that matters. A model will happily mirror
whatever gender the user just used, so being addressed in the masculine
is exactly when the persona has to hold. If it only holds when the user
sets it up correctly, it is not a persona, it is an echo.

Run:
    EVAL_JUDGE_BASE_URL=https://openrouter.ai/api \
    EVAL_JUDGE_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=deepseek/deepseek-v4-flash \
        ./scripts/run_evals.sh persona_feminine_agreement
"""

import re
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import ToolCallCapture, assert_not_fallback_reply, JUDGE_MODEL


@dataclass
class GenderCase:
    text: str
    # Masculine forms that must not appear. This is the assertion that
    # carries the eval: it fails on exactly the output the persona is
    # meant to prevent. Matched on word boundaries, because every
    # masculine form here is a prefix of its feminine counterpart
    # ("prêt" inside "prête") and a naive check would pass on the very
    # text it exists to catch.
    masculine: Tuple[str, ...]
    # Feminine forms, any one of which satisfies the case. Only set where
    # the question forces one specific adjective. Enumerating them
    # elsewhere pins vocabulary rather than gender: she answered one case
    # with "une majordome numérique" and "coincée", both feminine, both
    # absent from a perfectly reasonable word list.
    feminine: Tuple[str, ...] = ()


CASES = [
    pytest.param(
        GenderCase(
            text="Yuba, tu es prête à commencer ?",
            masculine=("prêt",),
            feminine=("prête",),
        ),
        id="prête, not prêt",
    ),
    pytest.param(
        GenderCase(
            text="Yuba, tu es contente de travailler avec moi ?",
            masculine=("content", "ravi", "heureux", "enchanté", "sûr", "certain"),
        ),
        id="contente, not content",
    ),
    pytest.param(
        GenderCase(
            text="Yuba, tu es sûr de toi ?",
            masculine=("sûr", "certain", "convaincu"),
        ),
        id="holds even when the user addresses her in the masculine",
    ),
]


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestPersonaFeminineAgreement:

    @pytest.mark.parametrize("case", CASES)
    def test_she_speaks_of_herself_in_the_feminine(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: GenderCase,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.db_path = str(tmp_path / "jarvis.db")
        mock_config.location_enabled = False
        mock_config.response_language = "français"

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

        print(f"\n  Persona ({JUDGE_MODEL}): {case.text}")
        print(f"  Response: {(response or '')[:250]}")

        assert_not_fallback_reply(response, context="persona")

        lowered = (response or "").lower()
        hit_masculine = [
            w for w in case.masculine
            if re.search(rf"\b{re.escape(w)}\b", lowered)
        ]
        hit_feminine = [
            w for w in case.feminine
            if re.search(rf"\b{re.escape(w)}\b", lowered)
        ]

        assert not hit_masculine, (
            f"She referred to herself in the masculine ({hit_masculine}). The "
            f"persona is a woman, and in French that shows on every "
            f"self-referential adjective. Response: {(response or '')[:300]}"
        )
        if case.feminine:
            assert hit_feminine, (
                f"No feminine self-reference found among {case.feminine}. "
                f"Either she dodged the agreement entirely or answered in "
                f"another language. Response: {(response or '')[:300]}"
            )
