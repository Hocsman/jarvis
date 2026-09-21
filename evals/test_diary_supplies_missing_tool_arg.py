"""
End-to-end eval: single-turn flow where the user's home city lives only
in the diary from a past conversation. The diary must surface
"Manchester", and the composed ``webSearch`` query must carry it.

The behaviour under test: when a tool argument is missing from the user's
utterance, the diary supplies it. The query is a personalised
recommendation ("for me"), which the planner's contract routes through
``searchMemory`` (planner.py rule 2), and the tool whose argument has to
carry the recalled city is ``webSearch``. The weather tool cannot play this
role — its contract (src/jarvis/tools/builtin/weather.py) instructs the
model to call it with empty args and let the tool auto-derive the
location, so a diary-supplied city would never appear in its arguments.

This stresses the diary-recall path. It complements the carry-over
guard's hot-window path (covered by
``evals/test_followup_supplies_missing_tool_arg.py``) by exercising the
slower long-term-memory path: the user said "I live in Manchester" days
ago, the conversation has lapsed, and now the user asks for a restaurant
with no live geoip and nothing in the hot window.

The planner is pinned to the plan its contract gives this query when
``webSearch`` is routed: the ``searchMemory`` directive first (rule 2), a
``webSearch`` step with a concrete argument (rule 4), then the synthesis
step (rule 8). The pin makes the diary pass deterministic; whether the
planner emits the directive for a personalised query is guarded on its
own by ``evals/test_planner_personalisation.py``. The step carries no
city, because the planner runs before memory and never sees it. It uses
the ``query=`` key the planner's own examples teach rather than the
tool's ``search_query``, so the step resolver turns it into a call with a
model round-trip, as in production, not through its deterministic fast
path.

Once the engine strips the directive, the plan still holds a tool step,
which keeps two branches live. The ACTION PLAN block reaches the system
prompt on every tier, and on small models plan-driven direct-exec
resolves the ``webSearch`` call from the step text, the prior tool
results and the tool schema before the chat model runs. The memory digest
is not among those inputs, so on that path the diary city has no way into
the argument: that is the gap this eval exists to catch. Routing runs
before the planner and stays live (the pinned step also puts
``webSearch`` in the allow-list), as does everything after the planner:
keyword extraction, diary search, digest, the step resolver and the chat
loop.

Memory-recall reliability on small models is itself an open failure
mode separate from the tool carry-over guard. If gemma4:e2b consistently
deflects rather than grounding the search, this eval is best read as an
upper-bound regression guard: a green run on a reliable judge model
proves the wiring works, while a red run on a small model is expected
until follow-up memory work lands.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh diary_supplies_missing_tool_arg
"""

from unittest.mock import patch

import pytest

from conftest import requires_judge_llm
from helpers import (
    ToolCallCapture,
    assert_not_fallback_reply,
    seed_diary_summaries,
    JUDGE_MODEL,
)


_DIARY_MANCHESTER = [
    (
        "2026-04-26",
        "The user mentioned they live in Manchester and have been trying "
        "new vegetarian restaurants around the Northern Quarter.",
    ),
]


_MANCHESTER_RESTAURANTS = (
    "Top result: Bundobust, Manchester — vegetarian Indian street food in "
    "the city centre, highly rated. Also listed: Dishoom Manchester and "
    "The Allotment Vegan Eatery, both well reviewed."
)


def _make_runner(capture: ToolCallCapture):
    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        capture.record(tool_name, tool_args or {})
        if tool_name == "webSearch":
            return ToolExecutionResult(
                success=True,
                reply_text=_MANCHESTER_RESTAURANTS,
            )
        return ToolExecutionResult(success=True, reply_text="OK")

    return _runner


@pytest.mark.eval
@requires_judge_llm
class TestDiarySuppliesMissingToolArg:
    """Diary-recall path: the user's home city, surfaced from a prior
    conversation, grounds the composed webSearch query without the hot
    window or an explicit re-statement."""

    def test_diary_location_grounds_restaurant_search(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Geoip disabled — the only way the model gets a location is from
        # diary recall.
        mock_config.location_enabled = False
        mock_config.memory_enrichment_source = "diary"

        seed_diary_summaries(eval_db, _DIARY_MANCHESTER)

        capture = ToolCallCapture()

        # Pin the plan the planner's contract gives this query: memory
        # first, then a webSearch step written without memory (the planner
        # never sees it), then the reply. The tool step keeps the ACTION
        # PLAN block and, on small models, plan-driven direct-exec live,
        # so the webSearch argument is composed on the path production
        # takes. That composition is the behaviour under test.
        forced_plan = [
            "searchMemory topic='user home city and dining preferences'",
            "webSearch query='good restaurants tonight'",
            "Reply to the user with the combined findings.",
        ]

        with patch(
            "jarvis.reply.engine.plan_query",
            return_value=forced_plan,
        ), patch(
            "jarvis.reply.engine.run_tool_with_retries",
            side_effect=_make_runner(capture),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="know any good restaurants for me tonight?",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Diary Supplies Missing Tool Arg ({JUDGE_MODEL}):")
        print(f"  Tools called: {capture.tool_names()}")
        for c in capture.calls:
            print(f"    - {c['name']}({c['args']})")
        print(f"  Response: {(response or '')[:300]}")

        assert_not_fallback_reply(response, context="diary-recall")

        # The reply must actually use the recalled location, both at the
        # tool call layer and in the user-facing reply.
        search_calls = [c for c in capture.calls if c["name"] == "webSearch"]
        manchester_calls = [
            c for c in search_calls
            if "manchester" in (c["args"].get("search_query") or "").lower()
        ]
        assert manchester_calls, (
            "webSearch was not composed with Manchester even though the "
            "diary contains the user's stated home city. The memory "
            "enrichment → tool argument grounding path is broken. "
            f"All webSearch calls: {[c['args'] for c in search_calls]}. "
            f"Tools observed: {capture.tool_names()}. "
            f"Response: {(response or '')[:400]}"
        )

        response_lower = (response or "").lower()
        assert "manchester" in response_lower, (
            "Reply does not mention Manchester despite the diary stating "
            f"the user lives there. Response: {(response or '')[:400]}"
        )

        # Guard against a hardcoded-default leak: any reply that mentions
        # Hackney here is wrong (Hackney is the test fixture's geoip
        # default, but geoip is disabled in this test).
        assert "hackney" not in response_lower, (
            "Reply mentions Hackney — the diary clearly states Manchester, "
            "and geoip is disabled in this test. The model leaked a "
            f"hardcoded default. Response: {(response or '')[:400]}"
        )
