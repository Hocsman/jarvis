"""
End-to-end eval — a reminder goes to the reminder tool, and lands.

The defect this closes was not subtle: with nowhere to put "remind me to
call the dentist tomorrow", the model filed the task as a permanent fact
about the user, in the two files injected into every later prompt. The
fix was not a better prompt. Three prompt attempts were measured and none
moved it; giving the model somewhere else to put it moved it to 6/6.

So the assertions here are on the two directions that can each ruin the
feature: a reminder must reach `setReminder` and actually land in the
database, and the things that merely *sound* like reminders must not.
"Remind me what I said yesterday" is a question about the past. "N'oublie
pas que je suis végétarien" is a memory. Both use the same verb, in every
language this has been checked in.

Run:
    EVAL_LLM_PROVIDER=openai_compatible \
    EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
    EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
    EVAL_JUDGE_MODEL=openai/gpt-oss-120b \
      ./scripts/run_evals.sh reminder_reaches_the_right_tool
"""

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_MODEL, ToolCallCapture, assert_not_fallback_reply


@dataclass
class ReminderCase:
    text: str
    should_schedule: bool


SCHEDULES = [
    pytest.param(
        ReminderCase("Jarvis, rappelle-moi dans vingt minutes de sortir le plat du four", True),
        id="FR: a relative time",
    ),
    pytest.param(
        ReminderCase("Jarvis, jeudi, rappelle-moi d'appeler le comptable", True),
        id="FR: a named day",
    ),
    pytest.param(
        ReminderCase("Jarvis, remind me in 45 minutes to check the oven", True),
        id="EN: a relative time",
    ),
    pytest.param(
        ReminderCase("Jarvis, bana yarın sabah Karim'i aramamı hatırlat", True),
        id="TR: tomorrow morning",
    ),
    pytest.param(
        ReminderCase("Jarvis, 30分後にオーブンを確認するよう知らせて", True),
        id="JA: in thirty minutes",
    ),
]

# Each of these uses a verb that looks like a reminder and is not one.
NOT_REMINDERS = [
    pytest.param(
        ReminderCase("Jarvis, rappelle-moi ce que j'ai dit hier", False),
        id="A question about the past is not a reminder",
    ),
    pytest.param(
        ReminderCase("Jarvis, n'oublie pas que je suis végétarien", False),
        id="A standing fact is a memory, not a reminder",
    ),
    pytest.param(
        ReminderCase("Jarvis, quelle heure est-il ?", False),
        id="Asking the time is not scheduling anything",
    ),
]


def _make_runner(capture: ToolCallCapture, reminders_db):
    """Run the real reminder tool. A stub could not show a reminder
    failing to land, which is half of what this measures."""
    from jarvis.tools.registry import BUILTIN_TOOLS
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name in ("setReminder", "remember"):
            return BUILTIN_TOOLS[tool_name].execute(
                db=reminders_db, cfg=cfg, tool_args=tool_args, system_prompt="",
                original_prompt="", redacted_text=kwargs.get("redacted_text", ""),
                max_retries=1, user_print=lambda _m: None,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


def _run(case, mock_config, eval_db, eval_dialogue_memory, tmp_path):
    from jarvis.memory.db import Database
    from jarvis.reply.engine import run_reply_engine

    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.db_path = str(tmp_path / "jarvis.db")
    # Pinned, or the chain resolves to a local model that is not running
    # here and every extraction fails for the wrong reason.
    mock_config.reminder_model = JUDGE_MODEL
    mock_config.reminder_timeout_sec = 20.0
    mock_config.reminder_default_hour = 9
    mock_config.reminders_enabled = True

    db = Database(mock_config.db_path, sqlite_vss_path=None)
    capture = ToolCallCapture()
    try:
        with patch("jarvis.reply.engine.run_tool_with_retries",
                   side_effect=_make_runner(capture, db)):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None, text=case.text,
                dialogue_memory=eval_dialogue_memory,
            )
        scheduled = db.pending_rappels()
    finally:
        db.close()
    return capture, response, scheduled


@pytest.mark.eval
@requires_judge_llm
class TestReminderReachesTheRightTool:

    @pytest.mark.parametrize("case", SCHEDULES)
    def test_a_reminder_lands(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case,
    ):
        capture, response, scheduled = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        print(f"\n  Reminder ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Scheduled: {[(r['due_local'], r['texte']) for r in scheduled]}")
        print(f"  Response: {(response or '')[:200]}")

        assert_not_fallback_reply(response, context="reminder")
        assert len(scheduled) == 1, (
            "Nothing was scheduled. A reminder the user asked for and never "
            "got is the failure this whole subsystem exists to prevent."
        )

    @pytest.mark.parametrize("case", SCHEDULES)
    def test_a_reminder_does_not_reach_the_memory(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case,
    ):
        """The original defect. A task filed as a fact about the user is
        injected into every later prompt until they find it in a file."""
        from jarvis.memory.core import SECTION_PROFILE, SECTION_RULES, MemoryCore

        capture, _, _ = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )
        core = MemoryCore.for_config(mock_config)
        written = [e.text for e in core.active(SECTION_PROFILE)] + [
            e.text for e in core.active(SECTION_RULES)
        ]

        print(f"\n  Tools called: {capture.tool_names()}")
        print(f"  Core now believes: {written}")

        assert written == [], (
            f"A reminder was stored as a durable fact about the user: {written!r}"
        )

    @pytest.mark.parametrize("case", NOT_REMINDERS)
    def test_something_that_only_sounds_like_one_schedules_nothing(
        self, mock_config, eval_db, eval_dialogue_memory, tmp_path, case,
    ):
        """Every one of these uses a verb that looks like a reminder.
        Scheduling on the verb rather than the meaning would fire at the
        user for asking a question."""
        capture, response, scheduled = _run(
            case, mock_config, eval_db, eval_dialogue_memory, tmp_path,
        )

        print(f"\n  Not a reminder ({JUDGE_MODEL}): {case.text}")
        print(f"  Tools called: {capture.tool_names()}")
        print(f"  Scheduled: {[(r['due_local'], r['texte']) for r in scheduled]}")

        assert scheduled == [], (
            f"{case.text!r} put something on the clock: "
            f"{[(r['due_local'], r['texte']) for r in scheduled]!r}"
        )
