"""
End-to-end eval — an explicit "remember that..." must reach the core.

The core (``src/jarvis/memory/core.spec.md``) fills through exactly one
path: the model deciding to call ``remember``. If the model chats back
"noted!" and calls nothing, the user believes they taught the assistant
something and nothing was written. That failure is silent from every
direction, and it is model-dependent, so it belongs in an eval rather
than a unit test.

The negative cases carry as much weight as the positive ones. A prompt
tuned to push the model towards ``remember`` will happily start writing
on every turn, and a fact the user never asked to store is precisely the
harm the core's "explicit and corrective only" rule exists to prevent.
An eval that only rewarded calling the tool would drive the fix the
wrong way.

Cases are given in several languages because the assistant is not
English-only and the instruction to remember carries no keyword the code
could match on.

**The positive cases are red on `gemma4:e2b` and this is expected.** The
engine forces text-based tool calling for SMALL models (`use_text_tools`
in `reply/engine.py`), so that model never receives a native tool schema
and has to emit a markdown fence instead. Measured directly against the
native API with the same system prompt, it calls `remember` correctly
5/5 for the French case and 3/5 for the rule case; through the engine's
fence path it emits prose. So this suite measures the fence path on a
2B model, which is the weakest combination the project supports. Run it
against a model large enough for native tools before drawing conclusions
about the prompt or the tool description.

Run:
    EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh remember_tool_is_called
    EVAL_JUDGE_MODEL=gpt-oss:20b ./scripts/run_evals.sh remember_tool_is_called
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    JUDGE_MODEL,
)


@dataclass
class RememberCase:
    """An utterance and whether it should reach the core."""

    text: str
    should_call: bool
    # For positive cases: the saved text must contain one of these, so a
    # call carrying the wrong content still fails. Several alternatives
    # because the tool asks for the user's own language and small models
    # translate anyway: what this eval is here to catch is a missing call
    # or the wrong fact, not a translation habit.
    expect_in_text: tuple = ()
    # For positive cases: "rule" when the utterance is a standing
    # instruction rather than a fact about the user.
    expect_kind: Optional[str] = None


REMEMBER_CASES = [
    pytest.param(
        RememberCase(
            text="Jarvis, remember that I am vegetarian.",
            should_call=True,
            expect_in_text=("vegetarian", "végétarien"),
        ),
        id="EN: explicit remember-that",
    ),
    pytest.param(
        RememberCase(
            text="Jarvis, souviens-toi que je suis végétarien.",
            should_call=True,
            expect_in_text=("végétarien", "vegetarian"),
        ),
        id="FR: explicit souviens-toi",
    ),
    pytest.param(
        RememberCase(
            text="Jarvis, from now on always answer me in French.",
            should_call=True,
            expect_in_text=("french", "français"),
            expect_kind="rule",
        ),
        id="EN: standing instruction becomes a rule",
    ),
    pytest.param(
        RememberCase(
            text="Jarvis, note that my sister is called Amal.",
            should_call=True,
            expect_in_text=("Amal",),
        ),
        id="EN: note-that phrasing",
    ),
]


DO_NOT_REMEMBER_CASES = [
    pytest.param(
        RememberCase(text="Jarvis, what time is it?", should_call=False),
        id="A plain question is not a memory instruction",
    ),
    pytest.param(
        RememberCase(
            text="Jarvis, I had pasta for lunch and it was pretty good.",
            should_call=False,
        ),
        id="Passing small talk is not a memory instruction",
    ),
    pytest.param(
        RememberCase(
            text="Jarvis, can you remind me to call the dentist tomorrow?",
            should_call=False,
        ),
        id="A reminder request is not a request to store a fact",
    ),
]


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "remember":
            text = ((tool_args or {}).get("text") or "").strip()
            if not text:
                return ToolExecutionResult(
                    success=False, reply_text="Nothing was saved: no text was given.",
                )
            return ToolExecutionResult(
                success=True,
                reply_text=f'Saved to long-term memory: "{text}".',
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


def _run(case: RememberCase, mock_config, eval_db, eval_dialogue_memory, tmp_path):
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    # A fresh core per case: nothing pre-loaded, so a call is the model
    # acting on the utterance rather than echoing an existing entry.
    mock_config.db_path = str(tmp_path / "jarvis.db")

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
    return capture, response


@pytest.mark.eval
@requires_judge_llm
class TestRememberToolIsCalled:
    """An explicit instruction to remember must reach the tool."""

    @pytest.mark.parametrize("case", REMEMBER_CASES)
    def test_explicit_instruction_calls_remember(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: RememberCase,
    ):
        capture, response = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        print(f"\n  Remember tool ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Response: {(response or '')[:200]}")

        assert_not_fallback_reply(response, context="remember")

        assert capture.has_tool("remember"), (
            "The user explicitly asked for something to be remembered and "
            "no call reached the core, so nothing was saved. Replying "
            "'noted' without calling the tool is the failure this eval "
            f"exists to catch. Tools called: {capture.tool_names()}. "
            f"Response: {(response or '')[:300]}"
        )

        args = capture.get_args("remember") or {}
        saved = (args.get("text") or "").lower()
        if case.expect_in_text:
            assert any(w.lower() in saved for w in case.expect_in_text), (
                f"remember was called but the saved text carries none of "
                f"{case.expect_in_text!r}: {args!r}"
            )
        if case.expect_kind:
            assert (args.get("kind") or "") == case.expect_kind, (
                f"Expected kind={case.expect_kind!r} for a standing "
                f"instruction, got {args!r}"
            )

    @pytest.mark.parametrize("case", DO_NOT_REMEMBER_CASES)
    def test_ordinary_talk_does_not_write_to_the_core(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case: RememberCase,
    ):
        capture, response = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        print(f"\n  Remember restraint ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")

        assert not capture.has_tool("remember"), (
            "The user did not ask for anything to be remembered, and a "
            "fact stored without being asked for is exactly what the core "
            "forbids: it is invisible, it rides in every later prompt, and "
            f"the user cannot trace it. Call made: {capture.get_args('remember')!r}. "
            f"Utterance: {case.text!r}"
        )
