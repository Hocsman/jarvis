"""
Single-step plan direct execution evaluation (live small model).

Validates that when the planner emits a single-step tool plan (e.g. for
straightforward time or weather queries), the reply engine direct-executes
the planned tool without invoking the chat model for intermediate turns,
then synthesises the final grounded reply.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest

from conftest import requires_judge_llm
from helpers import (
    JUDGE_BASE_URL,
    JUDGE_MODEL,
    ToolCallCapture,
    assert_not_fallback_reply,
    create_mock_tool_run,
)


def _configure(mock_config):
    mock_config.ollama_base_url = JUDGE_BASE_URL
    mock_config.ollama_chat_model = JUDGE_MODEL
    mock_config.llm_chat_model = JUDGE_MODEL
    mock_config.planner_enabled = True
    mock_config.location_enabled = True
    return mock_config


@pytest.mark.eval
@requires_judge_llm
class TestSingleStepPlanDirectExecution:
    """Eval asserting single-step tool plans are direct-executed by the engine."""

    def test_single_step_tool_plan_is_executed_directly(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """When a single tool step is planned, the engine must execute it directly."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        time_payload = "Current time in Tokyo, Japan: 21:45 JST (UTC+9)."

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=create_mock_tool_run(capture, {"getTime": time_payload}),
        ), patch(
            "jarvis.reply.planner.plan_query",
            return_value=["getTime location='Tokyo'"],
        ):
            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text="What time is it in Tokyo?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Response: {response}")
        print(f"  Tools called: {capture.tool_names()}")

        assert_not_fallback_reply(response)
        assert capture.has_tool("getTime"), (
            f"getTime should be executed directly from the single-step plan; "
            f"called: {capture.tool_names()}"
        )
        args = capture.get_args("getTime") or {}
        assert args.get("location") == "Tokyo", f"getTime must be called with location='Tokyo'; got {args}"

    def test_live_planner_and_execution_for_single_step_query(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """End-to-end with live planner model: query requiring a single tool
        must plan and execute the tool call."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        weather_payload = "Current weather in Paris, France: 18C, sunny."

        with patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=create_mock_tool_run(capture, {"getWeather": weather_payload}),
        ):
            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text="What is the weather in Paris right now?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Live query response: {response}")
        print(f"  Tools called: {capture.tool_names()}")

        assert_not_fallback_reply(response)
        assert capture.has_tool("getWeather") or capture.has_tool("webSearch"), (
            f"Expected getWeather or webSearch to be called; got: {capture.tool_names()}"
        )
