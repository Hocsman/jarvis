"""Engine + planner integration tests.

Covers the direct-exec path end-to-end: when the planner emits a
multi-step plan and the model is SMALL (text_tools), the engine must
resolve each planned step to a concrete tool call without invoking the
chat model for intermediate turns, then call the chat model once for the
final synthesis.

Unlike `tests/test_planner.py`, these tests exercise the engine wiring:
system-message composition, direct-exec tool dispatch, progress-nudge
injection into the tool-result messages.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_tool_name_msg(name: str) -> dict:
    """Return a message dict that looks like a tool-result message from a prior query."""
    return {"role": "user", "content": f"[Tool result: {name}] some result", "tool_name": name}


def _assistant_content(text: str):
    return {"message": {"role": "assistant", "content": text}}


def test_plan_injects_action_plan_block_into_system_message(
    mock_config, db, dialogue_memory
):
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"  # LARGE → native tools, no direct-exec

    mock_config.llm_chat_model = "gpt-oss:20b"  # LARGE → native tools, no direct-exec

    captured_system_messages: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        for m in msgs:
            if m.get("role") == "system":
                captured_system_messages.append(m.get("content", ""))
                break
        return _assistant_content("All done.")

    def fake_tool_runner(*args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    plan = [
        "webSearch query='director of Possessor 2020'",
        "webSearch query='films by <director name from step 1>'",
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what films did the director of Possessor make?",
            dialogue_memory=dialogue_memory,
        )

    assert captured_system_messages, "chat model should have been called at least once"
    assert "ACTION PLAN" in captured_system_messages[0], (
        "Planner output must be visible to the chat model in the initial system message"
    )
    for step in plan:
        assert step in captured_system_messages[0], (
            f"Plan step not found in system message: {step!r}"
        )


def test_small_model_direct_execs_planned_tools_without_chat_llm(
    mock_config, db, dialogue_memory
):
    """SMALL model + multi-step plan → engine runs each tool via the
    plan step-resolver, skipping chat_with_messages until the final
    synthesis turn."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    mock_config.llm_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    chat_call_count = [0]

    def fake_chat(*args, **kwargs):
        chat_call_count[0] += 1
        return _assistant_content("Paul Hardiman directed Possessor and later made X and Y.")

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        if len(invoked_tools) == 1:
            return ToolExecutionResult(
                success=True, reply_text="Possessor (2020) directed by Brandon Cronenberg.",
                error_message=None,
            )
        return ToolExecutionResult(
            success=True,
            reply_text="Films by Brandon Cronenberg: Antiviral (2012), Possessor (2020), Infinity Pool (2023).",
            error_message=None,
        )

    plan = [
        "webSearch query='Possessor 2020 director'",
        "webSearch query='films directed by <director name from step 1>'",
        "Reply to the user with the combined findings.",
    ]

    # Step resolver returns concrete tool calls for each planned step,
    # then `null` for the synthesis step (handled by engine as no-op).
    resolved_calls = iter([
        ("webSearch", {"query": "Possessor 2020 director"}),
        ("webSearch", {"query": "films directed by Brandon Cronenberg"}),
    ])

    def fake_resolve(*args, **kwargs):
        try:
            return next(resolved_calls)
        except StopIteration:
            return None

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what films did the director of Possessor make?",
            dialogue_memory=dialogue_memory,
        )

    tool_names = [n for n, _ in invoked_tools]
    assert tool_names == ["webSearch", "webSearch"], (
        f"Both plan tool steps should be direct-executed in order; got {tool_names}"
    )
    assert invoked_tools[1][1]["query"] == "films directed by Brandon Cronenberg", (
        "Second direct-exec must substitute the placeholder with a concrete entity"
    )
    # The chat model runs only for the final synthesis turn, not for
    # intermediate steps that were already direct-executed.
    assert chat_call_count[0] == 1, (
        f"Chat model should only fire for the final synthesis turn; "
        f"called {chat_call_count[0]}×"
    )


def test_small_model_direct_execs_single_step_tool_plan(
    mock_config, db, dialogue_memory
):
    """SMALL model + 1-step tool plan → engine direct-executes the tool call
    on turn 1, then invokes the chat model once for synthesis."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.llm_chat_model = "gemma4:e2b"

    chat_call_count = [0]

    def fake_chat(*args, **kwargs):
        chat_call_count[0] += 1
        return _assistant_content("It is 15 degrees and sunny in Paris.")

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        return ToolExecutionResult(
            success=True,
            reply_text="Weather in Paris: 15C, sunny.",
            error_message=None,
        )

    plan = ["getWeather location='Paris'"]

    def fake_resolve(*args, **kwargs):
        return ("getWeather", {"location": "Paris"})

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="what's the weather in Paris?",
            dialogue_memory=dialogue_memory,
        )

    tool_names = [n for n, _ in invoked_tools]
    assert tool_names == ["getWeather"], (
        f"The single plan tool step should be direct-executed; got {tool_names}"
    )
    assert invoked_tools[0][1] == {"location": "Paris"}
    assert chat_call_count[0] == 1, (
        f"Chat model should only be called once for synthesis; called {chat_call_count[0]}×"
    )


def test_small_model_skips_direct_exec_for_single_step_reply_plan(
    mock_config, db, dialogue_memory
):
    """SMALL model + 1-step reply plan → direct-exec skipped, chat model called on turn 1."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.llm_chat_model = "gemma4:e2b"

    chat_call_count = [0]

    def fake_chat(*args, **kwargs):
        chat_call_count[0] += 1
        return _assistant_content("Hello! How can I help?")

    invoked_tools: list[tuple[str, dict]] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append((tool_name, dict(tool_args or {})))
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    plan = ["Reply to the user."]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="hello there",
            dialogue_memory=dialogue_memory,
        )

    assert invoked_tools == [], "No tools should be direct-executed for a pure-reply plan"
    assert chat_call_count[0] == 1, "Chat model should reply directly on turn 1"


def test_empty_plan_falls_through_to_existing_behaviour(
    mock_config, db, dialogue_memory
):
    """Planner returning [] must not change engine behaviour."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"

    mock_config.llm_chat_model = "gemma4:e2b"

    captured_system_messages: list[str] = []

    def fake_chat(*args, **kwargs):
        msgs = kwargs.get("messages") or (args[2] if len(args) > 2 else [])
        for m in msgs:
            if m.get("role") == "system":
                captured_system_messages.append(m.get("content", ""))
                break
        return _assistant_content("Hi!")

    with patch.object(
        engine_mod,
        "run_tool_with_retries",
        return_value=ToolExecutionResult(success=True, reply_text="ok", error_message=None),
    ), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=[]):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="hello",
            dialogue_memory=dialogue_memory,
        )

    assert captured_system_messages
    assert "ACTION PLAN" not in captured_system_messages[0], (
        "Empty plan must NOT inject an ACTION PLAN block"
    )


def test_resolver_failure_on_tool_step_falls_back_to_chat(
    mock_config, db, dialogue_memory
):
    """When resolve_next_tool_call returns None for a tool step (not synthesis),
    the engine must fall through to the normal chat-model turn for that step."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    mock_config.llm_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    chat_call_count = [0]

    def fake_chat(*args, **kwargs):
        chat_call_count[0] += 1
        # First fallback turn: model emits a tool call itself
        if chat_call_count[0] == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "tool_calls: [{\"id\": \"c1\", \"type\": \"function\", "
                               "\"function\": {\"name\": \"webSearch\", "
                               "\"arguments\": \"{\\\"search_query\\\": \\\"foo\\\"}\"}}]",
                }
            }
        return _assistant_content("Here is what I found.")

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        return ToolExecutionResult(success=True, reply_text="Result", error_message=None)

    plan = [
        "webSearch query='foo'",
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "_resolve_plan_step", return_value=None):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="search for foo and summarise",
            dialogue_memory=dialogue_memory,
        )

    assert chat_call_count[0] >= 1, (
        "Engine must call the chat model when the step resolver returns None"
    )
    assert "webSearch" in invoked_tools, (
        "Chat model's own tool call should still be dispatched after resolver failure"
    )


def test_paraphrased_plan_falls_back_to_tool_router(
    mock_config, db, dialogue_memory
):
    """Small models sometimes emit prose steps like "get the weather"
    instead of naming the tool. The plan is non-empty but references
    no known tool — the engine must fall back to `select_tools` so the
    chat model isn't left with only stop + toolSearchTool (and then
    hallucinate a tool name from priors)."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"  # LARGE → native tools

    mock_config.llm_chat_model = "gpt-oss:20b"  # LARGE → native tools

    select_tools_called = [0]

    def fake_select_tools(*args, **kwargs):
        select_tools_called[0] += 1
        return ["getWeather", "stop"]

    def fake_chat(*args, **kwargs):
        return _assistant_content("Sunny.")

    def fake_tool_runner(*args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    plan = [
        "get the weather",  # paraphrased — no tool name
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", side_effect=fake_select_tools), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="how's the weather today?",
            dialogue_memory=dialogue_memory,
        )

    assert select_tools_called[0] == 1, (
        "Paraphrased plan with unresolved tool steps must fall back to select_tools"
    )


def test_paraphrased_plan_skips_direct_exec_for_small_models(
    mock_config, db, dialogue_memory
):
    """Under-specified plans (prose steps, no tool names) would otherwise
    force the step resolver LLM to guess arguments from vague step text
    (e.g. emitting location='Nowhere' for a plain "get the weather"
    step). Skip direct-exec entirely in that case — let the chat model
    handle the turn with the router-selected allow-list."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → direct-exec path

    mock_config.llm_chat_model = "gemma4:e2b"  # SMALL → direct-exec path

    resolver_calls = [0]

    def fake_resolver(*args, **kwargs):
        resolver_calls[0] += 1
        return ("getWeather", {"location": "Nowhere"})

    def fake_chat(*args, **kwargs):
        return _assistant_content("Sunny.")

    def fake_tool_runner(*args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    plan = [
        "get the weather",  # paraphrased — no tool name
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolver):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="how's the weather today?",
            dialogue_memory=dialogue_memory,
        )

    assert resolver_calls[0] == 0, (
        "Direct-exec resolver must not run when the plan is under-specified"
    )


def test_router_always_runs_and_plan_tools_are_unioned(
    mock_config, db, dialogue_memory
):
    """select_tools is the authoritative picker. When the planner picks
    tools, the names are unioned into the router's allow-list, not used
    to replace it. Small models often pick the most universal tool
    (webSearch) instead of a dedicated one (getWeather); the router is
    tuned for that classification and must remain authoritative."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gpt-oss:20b"

    mock_config.llm_chat_model = "gpt-oss:20b"

    router_calls = [0]
    captured_allow_lists: list[list[str]] = []

    def fake_select_tools(*args, **kwargs):
        router_calls[0] += 1
        # Router picks getWeather — the dedicated tool for this question.
        return ["getWeather", "stop"]

    def fake_chat(*args, **kwargs):
        # Grab the schema from kwargs/args to inspect the allow-list.
        schema = kwargs.get("tools") or []
        names = [s.get("function", {}).get("name") for s in schema if isinstance(s, dict)]
        captured_allow_lists.append([n for n in names if n])
        return _assistant_content("Sunny.")

    def fake_tool_runner(*args, **kwargs):
        return ToolExecutionResult(success=True, reply_text="ok", error_message=None)

    # Planner picks webSearch (the weaker, more universal choice).
    plan = [
        "webSearch query='weather in Hackney'",
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", side_effect=fake_select_tools), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="how's the weather here?",
            dialogue_memory=dialogue_memory,
        )

    assert router_calls[0] == 1, (
        "select_tools must always run, even when the planner picks tools"
    )
    assert captured_allow_lists, "chat model must have been called"
    exposed = captured_allow_lists[0]
    # Router's pick (authoritative, specific) is present …
    assert "getWeather" in exposed, (
        "Router's dedicated pick must be preserved in the allow-list"
    )
    # … and the planner's pick is unioned in, not dropped.
    assert "webSearch" in exposed, (
        "Planner's tool picks must be unioned into the allow-list"
    )


def test_direct_exec_fires_despite_prior_query_tool_carryover(
    mock_config, db, dialogue_memory
):
    """Tool results carried over from a PREVIOUS query must NOT be counted
    as 'already-executed steps of the current plan'.

    Regression: _tool_results_so_far counted all tool_name messages in the
    message list — including those from dialogue carryover — so a plan with
    one tool step appeared 'already done' whenever the prior turn used any
    tool, and direct-exec silently skipped the current query's tool call.
    The LLM then produced an empty reply → 'Sorry, I had trouble processing
    that'. This test verifies direct-exec fires correctly when carryover is
    present.
    """
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    mock_config.llm_chat_model = "gemma4:e2b"  # SMALL → use_text_tools

    # Simulate a prior query that used a tool — this is what happens after the
    # "scientists similar to Einstein" query that ran webSearch successfully.
    # We need both a text message (so has_recent_messages() returns True) AND
    # a tool turn (the actual carryover messages that appear in messages list).
    dialogue_memory.add_message("user", "what scientists are similar to Einstein?")
    dialogue_memory.add_message("assistant", "Niels Bohr and Richard Feynman.")
    dialogue_memory.record_tool_turn([
        _make_tool_name_msg("webSearch"),
    ])

    invoked_tools: list[str] = []

    def fake_tool_runner(db, cfg, tool_name, tool_args, **kwargs):
        invoked_tools.append(tool_name)
        return ToolExecutionResult(
            success=True, reply_text="London: 17°C, overcast", error_message=None
        )

    def fake_chat(*args, **kwargs):
        return _assistant_content("Tomorrow in London will be overcast, 17°C.")

    def fake_resolve(*args, **kwargs):
        return ("getWeather", {"location": "London"})

    plan = [
        "getWeather location='London'",
        "Reply to the user with the combined findings.",
    ]

    with patch.object(engine_mod, "run_tool_with_retries", side_effect=fake_tool_runner), \
         patch.object(engine_mod, "chat_with_messages", side_effect=fake_chat), \
         patch.object(engine_mod, "select_tools", return_value=["getWeather", "stop"]), \
         patch.object(
             engine_mod,
             "extract_search_params_for_memory",
             return_value={"keywords": []},
         ), \
         patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="tell me about the weather tomorrow",
            dialogue_memory=dialogue_memory,
        )

    assert "getWeather" in invoked_tools, (
        "direct-exec must fire for the current plan's getWeather step even when "
        "prior-query tool results are present in dialogue carryover"
    )


def test_direct_exec_resolver_receives_the_memory_digest(
    mock_config, db, dialogue_memory
):
    """When enrichment produced a digest, the plan step resolver must see
    it: a recalled fact (the user's city) has to be able to ground the
    composed tool call, not only the final chat reply."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"  # SMALL → use_text_tools
    mock_config.llm_chat_model = "gemma4:e2b"

    plan = [
        "searchMemory topic='user home city and dining preferences'",
        "webSearch query='good restaurants tonight'",
        "Reply to the user with the combined findings.",
    ]

    captured: list[dict] = []
    resolved_calls = iter([
        ("webSearch", {"query": "good restaurants tonight"}),
    ])

    def fake_resolve(*args, **kwargs):
        captured.append(kwargs)
        try:
            return next(resolved_calls)
        except StopIteration:
            return None

    fake_graph_store = MagicMock()
    fake_graph_store.search_nodes.return_value = []
    fake_graph_store.get_ancestors.return_value = []

    with patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(engine_mod, "extract_search_params_for_memory",
                      return_value={"keywords": ["restaurants"]}), \
         patch("jarvis.memory.conversation.search_conversation_memory_by_keywords",
               return_value=["The user lives in Manchester."]), \
         patch("jarvis.memory.graph.GraphMemoryStore",
               return_value=fake_graph_store), \
         patch.object(engine_mod, "digest_memory_for_query",
                      return_value="The user lives in Manchester."), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve), \
         patch.object(engine_mod, "run_tool_with_retries",
                      return_value=ToolExecutionResult(success=True, reply_text="OK")), \
         patch.object(engine_mod, "chat_with_messages",
                      return_value=_assistant_content("Bundobust in Manchester.")):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="any good restaurants for me tonight?",
            dialogue_memory=dialogue_memory,
        )

    assert captured, "the webSearch plan step should have been direct-executed"
    assert "Manchester" in captured[0].get("memory_context", "")


def test_direct_exec_resolver_gets_empty_memory_when_nothing_recalled(
    mock_config, db, dialogue_memory
):
    """No recalled memory → the resolver receives an empty memory_context,
    never a stale or fabricated one."""
    from jarvis.reply import engine as engine_mod
    from jarvis.tools.types import ToolExecutionResult

    mock_config.ollama_chat_model = "gemma4:e2b"
    mock_config.llm_chat_model = "gemma4:e2b"

    plan = [
        "webSearch query='Possessor 2020 director'",
        "Reply to the user with the findings.",
    ]

    captured: list[dict] = []
    resolved_calls = iter([("webSearch", {"query": "Possessor 2020 director"})])

    def fake_resolve(*args, **kwargs):
        captured.append(kwargs)
        try:
            return next(resolved_calls)
        except StopIteration:
            return None

    fake_graph_store = MagicMock()
    fake_graph_store.search_nodes.return_value = []
    fake_graph_store.get_ancestors.return_value = []

    with patch.object(engine_mod, "plan_query", return_value=plan), \
         patch.object(engine_mod, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(engine_mod, "extract_search_params_for_memory",
                      return_value={}), \
         patch("jarvis.memory.graph.GraphMemoryStore",
               return_value=fake_graph_store), \
         patch.object(engine_mod, "_resolve_plan_step", side_effect=fake_resolve), \
         patch.object(engine_mod, "run_tool_with_retries",
                      return_value=ToolExecutionResult(success=True, reply_text="OK")), \
         patch.object(engine_mod, "chat_with_messages",
                      return_value=_assistant_content("Brandon Cronenberg.")):
        engine_mod.run_reply_engine(
            db=db,
            cfg=mock_config,
            tts=None,
            text="who directed Possessor?",
            dialogue_memory=dialogue_memory,
        )

    assert captured, "the webSearch plan step should have been direct-executed"
    assert captured[0].get("memory_context", "") == ""
