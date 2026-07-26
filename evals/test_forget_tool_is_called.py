"""
End-to-end eval — "forget that" must reach the tool and drop the entry.

Sibling of ``test_remember_tool_is_called``. The asymmetry matters: a
missed `remember` loses something the user wanted kept, while a missed
`forget` keeps something the user asked to be rid of and answers "done"
anyway. The second is the worse failure, and it is the one nobody
notices, because the entry goes on quietly shaping every later reply.

The core is seeded per case so the model sees real entries in its
context and has something to name. The negative case guards the other
direction: being asked what the assistant knows is not permission to
start deleting it.

Run:
    EVAL_JUDGE_BASE_URL=https://openrouter.ai/api \
    EVAL_JUDGE_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh forget_tool_is_called
"""

from dataclasses import dataclass, field
from typing import List, Optional
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    JUDGE_MODEL,
)


SEEDED_FACTS = [
    "Il vit à Lyon.",
    "Il est végétarien.",
]
SEEDED_RULES = [
    "Toujours répondre en français.",
]


@dataclass
class ForgetCase:
    text: str
    should_call: bool
    # A word that must appear in the entry the tool was asked to drop.
    expect_target: Optional[str] = None
    # Entries that must still be active afterwards.
    expect_kept: List[str] = field(default_factory=list)


FORGET_CASES = [
    pytest.param(
        ForgetCase(
            text="Jarvis, forget that I live in Lyon.",
            should_call=True,
            expect_target="lyon",
            expect_kept=["Il est végétarien."],
        ),
        id="EN: forget a fact",
    ),
    pytest.param(
        ForgetCase(
            text="Jarvis, oublie que je vis à Lyon.",
            should_call=True,
            expect_target="lyon",
            expect_kept=["Il est végétarien."],
        ),
        id="FR: oublie a fact",
    ),
    pytest.param(
        ForgetCase(
            text="Jarvis, the rule about answering in French no longer applies.",
            should_call=True,
            expect_target="français",
            expect_kept=["Il vit à Lyon.", "Il est végétarien."],
        ),
        id="EN: a standing rule is dropped",
    ),
]


DO_NOT_FORGET_CASES = [
    pytest.param(
        ForgetCase(text="Jarvis, what do you know about me?", should_call=False),
        id="Being asked what it knows is not permission to delete it",
    ),
    pytest.param(
        ForgetCase(text="Jarvis, I forgot my umbrella this morning.", should_call=False),
        id="The user forgetting something is not an instruction",
    ),
]


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult
    from jarvis.tools.registry import BUILTIN_TOOLS

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        # forget runs for real against the seeded core: the point is
        # whether the right entry actually stops being believed, which a
        # stub result could not show.
        if tool_name == "forget":
            return BUILTIN_TOOLS["forget"].execute(
                db=db, cfg=cfg, tool_args=tool_args, system_prompt="",
                original_prompt="", redacted_text="", max_retries=1,
                user_print=lambda _m: None,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


def _run(case: ForgetCase, mock_config, eval_db, eval_dialogue_memory, tmp_path):
    from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.db_path = str(tmp_path / "jarvis.db")

    core = MemoryCore.for_config(mock_config)
    for fact in SEEDED_FACTS:
        core.remember(SECTION_PROFILE, fact)
    for rule in SEEDED_RULES:
        core.remember(SECTION_RULES, rule)

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
    return capture, response, core


@pytest.mark.eval
@requires_judge_llm
class TestForgetToolIsCalled:
    """An explicit instruction to forget must reach the tool and land."""

    @pytest.mark.parametrize("case", FORGET_CASES)
    def test_explicit_instruction_drops_the_entry(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: ForgetCase,
    ):
        from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES

        capture, response, core = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        active = [e.text for e in core.active(SECTION_PROFILE)]
        active += [e.text for e in core.active(SECTION_RULES)]

        print(f"\n  Forget tool ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Still believed: {active}")
        print(f"  Response: {(response or '')[:200]}")

        assert_not_fallback_reply(response, context="forget")

        assert capture.has_tool("forget"), (
            "The user asked for something to be forgotten and no call "
            "reached the core. Answering 'done' while the entry keeps "
            "shaping every later reply is the failure this eval exists to "
            f"catch. Tools called: {capture.tool_names()}. "
            f"Response: {(response or '')[:300]}"
        )

        if case.expect_target:
            target = ((capture.get_args("forget") or {}).get("target") or "").lower()
            assert case.expect_target in target, (
                f"forget was called for the wrong entry: expected something "
                f"about {case.expect_target!r}, got {target!r}"
            )

        for kept in case.expect_kept:
            assert kept in active, (
                f"{kept!r} should still be believed: the user asked for one "
                f"thing to go, not for the core to be cleared. Still "
                f"active: {active}"
            )

    @pytest.mark.parametrize("case", DO_NOT_FORGET_CASES)
    def test_ordinary_talk_does_not_delete_anything(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: ForgetCase,
    ):
        from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES

        capture, response, core = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        active = [e.text for e in core.active(SECTION_PROFILE)]
        active += [e.text for e in core.active(SECTION_RULES)]

        print(f"\n  Forget restraint ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        print(f"  Still believed: {active}")

        assert len(active) == len(SEEDED_FACTS) + len(SEEDED_RULES), (
            "Something was dropped from the core on an utterance that asked "
            "for no such thing. Deleting a memory the user never asked to "
            f"lose is unrecoverable from their side. Still active: {active}. "
            f"Call made: {capture.get_args('forget')!r}"
        )
