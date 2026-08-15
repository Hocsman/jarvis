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

Measured, 2026-07-25:

| Model | Result |
|-------|--------|
| `openai/gpt-oss-120b` | 6 passed, 1 xfail |
| `deepseek/deepseek-v4-flash` | 6 passed, 1 xfail |
| `gemma4:e2b` (local default) | positives red |

**The positive cases being red on `gemma4:e2b` is expected, not a
regression.** The engine forces text-based tool calling for SMALL models
(`use_text_tools` in `reply/engine.py`), so that model never receives a
native tool schema and has to emit a markdown fence instead. Measured
directly against the native API with the same system prompt, it calls
`remember` 5/5 on the French case and 3/5 on the rule case; through the
engine's fence path it emits prose. This suite therefore measures the
fence path on a 2B model, the weakest combination the project supports.
Draw conclusions about the prompt or the tool description from a tier
that gets native tools.

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
    # This one was an open defect for a long time, and how it closed is
    # worth keeping. With no reminder feature, "remind me to call the
    # dentist tomorrow" made the model reach for the nearest tool and
    # file the task as a permanent fact about the user. Three prompt
    # attempts were measured and none moved it — a restraint clause in
    # the system prompt, a reminder-is-not-a-memory clause in the tool
    # description, and both together.
    #
    # Neither of the two things that did move it was rhetorical. First,
    # the router had only ever seen the first 120 characters of each
    # description, cut mid-word, so `remember`'s restriction was
    # invisible to the layer deciding whether to offer it at all: whole
    # sentences took it from 0/6 to 2/6. Then `setReminder` gave the
    # model somewhere else to put it: 6/6.
    #
    # Arguing with a model's priors is expensive. Giving it a better
    # option is cheap.
    pytest.param(
        RememberCase(
            text="Jarvis, can you remind me to call the dentist tomorrow?",
            should_call=False,
        ),
        id="A reminder request is not a request to store a fact",
    ),
]


# Observed in production on 2026-07-27. The user asked her, by voice, to
# write the capital of France in the chat; she said it aloud instead;
# he complained that she had not written it. She read "write" as "write
# it down", took HER OWN previous answer, and filed it as a durable fact
# about him.
#
# It is the worst shape this subsystem has: the entry was never his
# statement, it is invisible until he opens a file, and until then it is
# injected into every prompt she ever builds.
#
# `prior` is what makes it reproducible — she cannot copy her own
# sentence unless she has said one.
ECHO_CASES = [
    pytest.param(
        RememberCase(
            text="Tu peux pas m'écrire dans la conversation ?",
            should_call=False,
        ),
        ("Allo Yuba, écris-moi la capitale de la France dans la conversation.",
         "La capitale de la France est Paris."),
        id="FR: the production sequence, verbatim",
    ),
    pytest.param(
        RememberCase(
            text="Can't you write it out for me?",
            should_call=False,
        ),
        ("Write me the capital of France.",
         "The capital of France is Paris."),
        id="EN: the same complaint",
    ),
    pytest.param(
        RememberCase(
            text="Répète-le moi, s'il te plaît.",
            should_call=False,
        ),
        ("Yuba, quelle est la capitale de la France ?",
         "La capitale de la France est Paris."),
        id="Asking her to repeat something is not asking her to keep it",
    ),
]


def _make_runner(capture: ToolCallCapture):
    """Run the real tool against the real core.

    Stubbing it would measure only whether the model emitted a call —
    which is not the question. What matters is whether anything ends up
    in the files the user reads, and a stub can neither show a bad write
    landing nor show the tool refusing one. `capture` still records every
    call, so "the model asked and the tool said no" stays visible as a
    distinct outcome from "the model never asked".
    """
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "remember":
            return BUILTIN_TOOLS["remember"].execute(
                db=db, cfg=cfg, tool_args=tool_args, system_prompt="",
                original_prompt="", redacted_text=kwargs.get("redacted_text", ""),
                max_retries=1, user_print=lambda _m: None,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


def _run(case: RememberCase, mock_config, eval_db, eval_dialogue_memory, tmp_path,
         prior=None):
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    # A fresh core per case: nothing pre-loaded, so a call is the model
    # acting on the utterance rather than echoing an existing entry.
    mock_config.db_path = str(tmp_path / "jarvis.db")

    # Some failures only exist mid-conversation: she cannot copy her own
    # sentence into the core unless she has said one. `prior` replays the
    # turn before, so the answer she is about to steal is in context.
    if prior is not None:
        eval_dialogue_memory.add_message("user", prior[0])
        eval_dialogue_memory.add_message("assistant", prior[1])

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


def _written(mock_config):
    """Everything the core actually believes, both sections."""
    from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore

    core = MemoryCore.for_config(mock_config)
    return [e.text for e in core.active(SECTION_PROFILE)] + [
        e.text for e in core.active(SECTION_RULES)
    ]


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

    @pytest.mark.parametrize("case,prior", ECHO_CASES)
    def test_she_does_not_file_her_own_answer_as_a_fact(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        case: RememberCase, prior,
    ):
        """The assertion is on the FILES, not on the call.

        Whether the model emits a call is its business; whether anything
        lands in the two files the user reads is the contract. A tool that
        is asked and refuses is a pass here, and that distinction is the
        whole reason this eval runs the real tool instead of a stub.
        """
        capture, response = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path, prior=prior,
        )
        written = _written(mock_config)

        print(f"\n  Self-echo ({JUDGE_MODEL})")
        print(f"  She had just said: {prior[1]!r}")
        print(f"  User then said:    {case.text!r}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Core now believes: {written}")

        assert written == [], (
            "She filed something the user never said. The entry is his "
            "memory now: invisible until he opens a file, and injected "
            "into every prompt she builds until he finds it.\n"
            f"  Written: {written!r}\n"
            f"  Her own prior words: {prior[1]!r}\n"
            f"  What he actually said: {case.text!r}"
        )
