"""
Helper functions and data classes for eval tests.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, Tuple
import os


# LLM-as-judge / model-under-test configuration.
#
# This single knob does double duty: it's both the model the eval uses as
# the chat LLM being tested AND the judge used to assess open-ended
# responses. Field failures on the production default surface here first,
# so the default MUST match what users actually run — which is the smallest
# supported model in the README ("gemma4:e2b"), not the largest we
# internally test against. Opt into larger models with EVAL_JUDGE_MODEL=…
# when you want a sanity check of the upper tier.
#
# Historical note: the default was gpt-oss:20b until 2026-04-20, at which
# point two field regressions on gemma4:e2b (tool selected but not invoked;
# native "tool_code" fallback syntax) slipped past CI because the evals
# were only testing the 20B tier. Defaulting to the small tier is the
# cheapest way to stop that happening again.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gemma4:e2b")

# The small chain, separately from the chat model.
#
# They are one tier in the suite's defaults and two on a real machine:
# the chat model answers, and a smaller one routes tools, judges intent,
# extracts times and reads approvals. Pinning them together measures a
# configuration nobody runs, so this exists to measure the one they do.
SMALL_MODEL = os.environ.get("EVAL_SMALL_MODEL", "") or JUDGE_MODEL
JUDGE_BASE_URL = os.environ.get("EVAL_JUDGE_BASE_URL", "http://localhost:11434")

# Which backend the suite runs against. Ollama by default, because the
# default judge model is local and free. Point these at an
# OpenAI-compatible endpoint to measure the tier a user actually runs:
#
#   EVAL_LLM_PROVIDER=openai_compatible \
#   EVAL_LLM_BASE_URL=https://openrouter.ai/api/v1 \
#   EVAL_LLM_API_KEY_ENV=OPENROUTER_API_KEY \
#   EVAL_JUDGE_MODEL=openai/gpt-oss-120b ./scripts/run_evals.sh <suite>
#
# This is not a detail: the reply engine forces text-based tool calling
# for models it classifies as SMALL, so a local small-model run never
# exercises the native tool-call path at all. A suite about whether a
# tool gets invoked measures a different mechanism on each tier.
EVAL_LLM_PROVIDER = os.environ.get("EVAL_LLM_PROVIDER", "ollama")
EVAL_LLM_BASE_URL = os.environ.get("EVAL_LLM_BASE_URL", "")
EVAL_LLM_API_KEY_ENV = os.environ.get("EVAL_LLM_API_KEY_ENV", "")


# =============================================================================
# Tool Call Capture
# =============================================================================

# =============================================================================
# Fallback-reply detection
# =============================================================================
#
# When the malformed-output guard fires in the reply engine (engine.py), the
# user gets one of these canned strings. From the user's perspective that is
# a FAILURE — they asked a question and got a shrug — but historically several
# evals treated it as neutral because "no malformed text reached the user" is
# technically true. Treating these strings as test failures turns a silent
# shield into a loud alarm: if gemma keeps tripping the guard under a given
# context shape (warm memory, large digest, odd phrasing), the evals will
# finally flag it.
#
# The helper asserts at the call site of an eval rather than globally,
# because a handful of evals (e.g. `TestMalformedResponseAfterTools` itself)
# are specifically asserting the fallback fires and must NOT use this helper.

FALLBACK_REPLY_PHRASES = (
    "i had trouble understanding that request",
    "i had trouble processing that",
    "sorry, i had trouble",
)


def is_fallback_reply(response: Optional[str]) -> bool:
    """Return True when ``response`` is the engine's canned malformed-guard
    fallback reply — i.e. the user got a shrug instead of an answer."""
    if not response:
        return False
    lowered = response.lower()
    return any(phrase in lowered for phrase in FALLBACK_REPLY_PHRASES)


def assert_not_fallback_reply(response: Optional[str], context: str = "") -> None:
    """Fail the test when the response is the engine's canned fallback.

    A fallback reply means the malformed-output guard fired — which is a
    safety net masking an underlying model failure. In most evals, seeing
    this string means the test SHOULD fail even if the rest of the
    assertions happen to pass, because the user experience is "the
    assistant gave up".
    """
    import pytest

    if is_fallback_reply(response):
        prefix = f"[{context}] " if context else ""
        pytest.fail(
            f"{prefix}Response is the engine's canned malformed-guard "
            f"fallback reply — the model produced garbled output and the "
            f"guard shielded the user. From the user's perspective the "
            f"assistant gave up. Treat this as a real failure. "
            f"Response: {(response or '')[:400]}"
        )


# =============================================================================
# Max-turns digest caveat detection
# =============================================================================
#
# When the agentic loop exhausts ``agentic_max_turns`` without the evaluator
# ever firing terminal, ``digest_loop_for_max_turns`` in ``enrichment.py``
# produces a reply whose first sentence is a caveat noting the request was
# not fully finished (e.g. "I could not fully finish your request…").
#
# From the user's perspective that caveat is a FAILURE for simple,
# single-tool queries — the tool ran, the answer was in hand, and yet the
# evaluator kept saying "continue" until the turn cap fired the digest
# summariser. The answer that follows the caveat is typically correct, so
# naive grounding assertions pass and the regression hides. Treating the
# caveat as a failure turns that silent shield into a loud alarm for the
# evaluator's terminal-detection quality.
#
# The digest prompt (``_LOOP_DIGEST_SYSTEM_PROMPT`` in
# ``src/jarvis/reply/enrichment.py``) instructs the LLM to open with a
# caveat about not finishing. The phrases below are the canonical English
# shapes that prompt produces; a drift pin test keeps them aligned with
# the source prompt.

MAX_TURNS_DIGEST_PHRASES = (
    "could not fully finish",
    "couldn't fully finish",
    "was unable to fully finish",
    "wasn't able to fully finish",
)


def is_max_turns_digest(response: Optional[str]) -> bool:
    """Return True when ``response`` looks like the max-turns digest
    caveat — i.e. the agentic loop ran out of turns without the evaluator
    ever firing terminal."""
    if not response:
        return False
    lowered = response.lower()
    return any(phrase in lowered for phrase in MAX_TURNS_DIGEST_PHRASES)


def assert_not_max_turns_digest(response: Optional[str], context: str = "") -> None:
    """Fail the test when the response opens with the max-turns digest
    caveat. For simple single-tool queries, hitting the digest path means
    the evaluator failed to recognise a grounded, terminal reply — even if
    the content that follows the caveat happens to be correct."""
    import pytest

    if is_max_turns_digest(response):
        prefix = f"[{context}] " if context else ""
        pytest.fail(
            f"{prefix}Response begins with the max-turns digest caveat — "
            f"the agentic loop exhausted ``agentic_max_turns`` without the "
            f"evaluator returning terminal on a grounded reply. For simple "
            f"queries this is an evaluator quality failure, not a success. "
            f"Response: {(response or '')[:400]}"
        )


# =============================================================================
# Warm-memory seeding
# =============================================================================
#
# The default eval fixtures (`eval_db`, `eval_dialogue_memory`) start empty,
# which does NOT reproduce the real-world state where the user's memory
# already carries weeks of diary summaries. Field failures consistently
# correlate with loaded context: gemma produces clean tool calls on empty
# memory and slides into scaffolding leaks when a multi-hundred-char memory
# digest is prepended to the system message.
#
# This helper seeds the diary table with dated summaries on a given topic
# so the memory-search path hits real entries and produces a digest that
# matches the production shape.

def seed_diary_summaries(
    db,
    topic_summaries: List[Tuple[str, str]],
) -> None:
    """Seed ``conversation_summaries`` with the given (date_utc, summary) pairs.

    ``date_utc`` must be ``YYYY-MM-DD``. The helper is a thin wrapper around
    ``db.upsert_conversation_summary`` intended for evals that need a warm
    memory state — e.g. "user has asked about the weather ten times in the
    last fortnight" — to reproduce the loaded-context failure mode that the
    reply engine hits in production.
    """
    for date_utc, summary in topic_summaries:
        db.upsert_conversation_summary(
            date_utc=date_utc,
            summary=summary,
            topics=None,
            source_app="jarvis",
        )


@dataclass
class ToolCallCapture:
    """Captures tool calls during evaluation."""

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: Dict[str, Any]):
        self.calls.append({"name": name, "args": args})

    def has_tool(self, name: str) -> bool:
        return any(c["name"] == name for c in self.calls)

    def has_any_tool(self) -> bool:
        return len(self.calls) > 0

    def get_args(self, name: str) -> Optional[Dict[str, Any]]:
        for c in self.calls:
            if c["name"] == name:
                return c["args"]
        return None

    def tool_names(self) -> List[str]:
        return [c["name"] for c in self.calls]

    # Alias for backward compatibility
    tool_sequence = tool_names

    def clear(self):
        self.calls = []


# =============================================================================
# Mock Tool Run Factory
# =============================================================================

def create_mock_tool_run(
    capture: ToolCallCapture,
    responses: Optional[Dict[str, str]] = None,
):
    """Create a mock tool runner that captures calls and returns canned responses.

    Args:
        capture: ToolCallCapture instance to record calls
        responses: Dict mapping tool name → response text. Unmatched tools return "OK".

    Returns:
        A function suitable for patching ``run_tool_with_retries``.
    """
    responses = responses or {}

    def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
        from jarvis.tools.types import ToolExecutionResult
        capture.record(tool_name, tool_args or {})
        reply = responses.get(tool_name, "OK")
        return ToolExecutionResult(success=True, reply_text=reply)

    return mock_tool_run


_REAL_DEFAULTS = None


def _real_default_settings():
    """A real ``Settings`` built from defaults only, cached.

    Loaded against an empty config file so the developer's own
    ``config.json`` (cloud provider, API keys, home city…) cannot leak into
    eval runs and make them non-reproducible.
    """
    global _REAL_DEFAULTS
    if _REAL_DEFAULTS is None:
        import json
        import tempfile
        from jarvis.config import load_settings

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as fh:
                json.dump({}, fh)
            previous = os.environ.get("JARVIS_CONFIG_PATH")
            os.environ["JARVIS_CONFIG_PATH"] = path
            try:
                _REAL_DEFAULTS = load_settings()
            finally:
                if previous is None:
                    os.environ.pop("JARVIS_CONFIG_PATH", None)
                else:
                    os.environ["JARVIS_CONFIG_PATH"] = previous
    return _REAL_DEFAULTS


@dataclass
class MockConfig:
    """Minimal config object for eval tests.

    Only fields the evals actually pin are declared here; anything else
    falls through to the real ``Settings`` defaults (see ``__getattr__``).
    That indirection exists because this mock had drifted 72 fields behind
    ``Settings`` — every eval died on ``AttributeError`` before asserting
    anything, so the whole suite was reporting red for a reason unrelated to
    agent behaviour. Falling back means a newly added setting can no longer
    silently break the suite.
    """
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    # Backend selection. Declared rather than left to fall through so a
    # suite can be pointed at the tier the user actually runs; the
    # factory dispatches on ``llm_provider``.
    llm_provider: str = EVAL_LLM_PROVIDER
    llm_base_url: str = EVAL_LLM_BASE_URL
    llm_api_key_env: str = EVAL_LLM_API_KEY_ENV
    # Resolved here rather than left to ``Settings``, which reads the env
    # var named by ``llm_api_key_env`` only while loading a real config
    # file. The key is never written to disk or printed by the suite.
    llm_api_key: str = os.environ.get(EVAL_LLM_API_KEY_ENV, "") if EVAL_LLM_API_KEY_ENV else ""
    llm_chat_model: str = JUDGE_MODEL
    # The small chain follows the suite's model, rather than falling
    # through to `Settings`' Ollama defaults.
    #
    # It did fall through, and the cost was invisible: `intent_judge_model`
    # defaulted to `gemma4:e2b`, `resolve_tool_router_model` resolved to
    # it, and OpenRouter answered "gemma4:e2b is not a valid model ID" on
    # every routing call. `select_tools` fails open, so every eval run
    # against a remote provider was silently exercising the fall-open
    # path — "20 selected" is the whole catalogue, not a routing
    # decision — while appearing to pass. A suite that measures routing
    # has to actually route.
    intent_judge_model: str = SMALL_MODEL
    tool_router_model: str = SMALL_MODEL
    reminder_model: str = SMALL_MODEL
    confirmation_model: str = SMALL_MODEL
    ollama_embed_model: str = "nomic-embed-text"
    db_path: str = ":memory:"
    sqlite_vss_path: Optional[str] = None
    voice_debug: bool = True
    tts_enabled: bool = False
    tts_engine: str = "piper"  # "piper" (default) or "chatterbox"
    tts_voice: Optional[str] = None
    tts_rate: int = 200
    # Piper TTS settings
    tts_piper_model_path: Optional[str] = None
    tts_piper_speaker: Optional[int] = None
    tts_piper_length_scale: float = 1.0
    tts_piper_noise_scale: float = 0.667
    tts_piper_noise_w: float = 0.8
    tts_piper_sentence_silence: float = 0.2
    # Chatterbox TTS settings
    tts_chatterbox_device: str = "cpu"
    tts_chatterbox_audio_prompt: Optional[str] = None
    tts_chatterbox_exaggeration: float = 0.5
    tts_chatterbox_cfg_weight: float = 0.5
    web_search_enabled: bool = True
    brave_search_api_key: str = ""
    wikipedia_fallback_enabled: bool = True
    llm_profile_select_timeout_sec: float = 10.0
    llm_tools_timeout_sec: float = 8.0
    llm_embedding_timeout_sec: float = 10.0
    llm_chat_timeout_sec: float = 120.0
    agentic_max_turns: int = 8
    memory_enrichment_max_results: int = 5
    active_profiles: List[str] = field(default_factory=lambda: ["developer", "business", "life"])
    location_enabled: bool = True
    location_ip_address: Optional[str] = None
    location_auto_detect: bool = False
    location_cgnat_resolve_public_ip: bool = False
    dialogue_memory_timeout: int = 300
    mcps: Dict[str, Any] = field(default_factory=dict)
    use_stdin: bool = True

    def __getattr__(self, name: str):
        """Fall back to the real ``Settings`` default for undeclared fields.

        Only called for attributes the dataclass doesn't define, so declared
        fields keep their eval-specific values (local Ollama URL, test model,
        in-memory DB). Private/dunder names are refused so this never
        interferes with copy, pickle, or pytest introspection.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        defaults = _real_default_settings()
        try:
            return getattr(defaults, name)
        except AttributeError:
            raise AttributeError(
                f"{name!r} is neither a MockConfig field nor a Settings default"
            ) from None


@dataclass
class EvalResult:
    """Result of a single eval test case."""
    query: str
    response: Optional[str]
    is_passed: bool
    failure_reason: Optional[str] = None
    tool_calls_made: List[str] = field(default_factory=list)
    turn_count: int = 0

    def __str__(self) -> str:
        status = "✅ PASS" if self.is_passed else "❌ FAIL"
        lines = [
            f"{status}: {self.query[:50]}...",
            f"  Response: {(self.response or '')[:100]}...",
            f"  Tools used: {', '.join(self.tool_calls_made) or 'none'}",
            f"  Turns: {self.turn_count}",
        ]
        if self.failure_reason:
            lines.append(f"  Reason: {self.failure_reason}")
        return "\n".join(lines)


@dataclass
class EvalCase:
    """A single eval test case definition."""
    name: str
    query: str
    expected_tool_calls: List[str] = field(default_factory=list)
    response_should_contain: List[str] = field(default_factory=list)
    response_should_not_contain: List[str] = field(default_factory=list)
    custom_validator: Optional[Callable[[str], bool]] = None
    profile_hint: Optional[str] = None


def assert_response_quality(result: EvalResult, case: EvalCase) -> None:
    """Assert that the response meets quality criteria."""
    response = result.response or ""
    response_lower = response.lower()

    # Check expected content
    for expected in case.response_should_contain:
        assert expected.lower() in response_lower, (
            f"Response should contain '{expected}' but got: {response[:200]}..."
        )

    # Check excluded content
    for excluded in case.response_should_not_contain:
        assert excluded.lower() not in response_lower, (
            f"Response should NOT contain '{excluded}' but got: {response[:200]}..."
        )

    # Check custom validator
    if case.custom_validator:
        assert case.custom_validator(response), (
            f"Custom validation failed for response: {response[:200]}..."
        )


def is_generic_greeting(response: str) -> bool:
    """Check if response is a generic greeting that ignores the query."""
    generic_patterns = [
        "how can i help you",
        "what can i do for you",
        "what would you like",
        "how may i assist",
        "is there something",
        "let me know what",
        "feel free to ask",
    ]
    response_lower = response.lower()
    return any(pattern in response_lower for pattern in generic_patterns)


def response_addresses_topic(response: str, topic_keywords: List[str]) -> bool:
    """Check if response addresses the topic by mentioning relevant keywords."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in topic_keywords)


def create_mock_llm_response(content: str, tool_calls: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Create a mock LLM response in Ollama format."""
    message = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message}


def create_tool_call(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Create a tool call in OpenAI format."""
    return {
        "id": f"call_{name}_001",
        "function": {
            "name": name,
            "arguments": args
        }
    }


# =============================================================================
# LLM-as-Judge Evaluation
# =============================================================================

@dataclass
class JudgeVerdict:
    """Result from LLM judge evaluation."""
    is_passed: bool
    score: float  # 0.0 to 1.0
    reasoning: str
    criteria_scores: Dict[str, float] = field(default_factory=dict)


def is_judge_llm_available() -> bool:
    """Check whether the configured judge model can be reached.

    Provider-aware: an OpenAI-compatible endpoint is probed with a free
    model listing rather than Ollama's ``/api/tags``, which would report
    every remote model as missing and silently skip the whole suite.
    """
    import requests

    if EVAL_LLM_PROVIDER != "ollama":
        if not EVAL_LLM_BASE_URL:
            return False
        headers = {}
        key = os.environ.get(EVAL_LLM_API_KEY_ENV, "") if EVAL_LLM_API_KEY_ENV else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            resp = requests.get(
                f"{EVAL_LLM_BASE_URL.rstrip('/')}/models", headers=headers, timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    try:
        # First check if Ollama is running
        resp = requests.get(f"{JUDGE_BASE_URL.rstrip('/')}/api/tags", timeout=2)
        if resp.status_code != 200:
            return False

        # Check if the judge model is available
        data = resp.json()
        models = data.get("models", [])
        model_names = [m.get("name", "").split(":")[0] for m in models]

        # Check if our judge model (or a variant) is available
        judge_base = JUDGE_MODEL.split(":")[0]
        return any(judge_base in name for name in model_names)
    except Exception:
        return False


def call_judge_llm(system_prompt: str, user_prompt: str, timeout_sec: float = 120.0) -> Optional[str]:
    """Call the judge LLM with a prompt.

    Routes through the configured backend when the suite is pointed at a
    non-Ollama provider, so judge-scored cases measure the same tier as
    the behaviour under test.
    """
    import requests

    if EVAL_LLM_PROVIDER != "ollama":
        from jarvis.llm import get_llm_backend
        try:
            return get_llm_backend(MockConfig()).direct(
                JUDGE_MODEL, system_prompt, user_prompt, timeout_sec=timeout_sec,
            )
        except Exception as e:
            print(f"⚠️ Judge LLM call failed: {e}")
            return None

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {"num_ctx": 4096},
    }

    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout_sec
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "message" in data:
            return data["message"].get("content", "")
    except Exception as e:
        print(f"⚠️ Judge LLM call failed: {e}")
        return None
    return None


def judge_response_answers_query(query: str, response: str, context: Optional[str] = None) -> JudgeVerdict:
    """
    Use LLM to judge if the response actually answers the user's query.

    Args:
        query: The user's original question
        response: The assistant's response
        context: Optional context about what data was available (e.g., tool results)

    Returns:
        JudgeVerdict with pass/fail, score, and reasoning
    """
    system_prompt = """You are an evaluation judge for a voice assistant. Your job is to determine if the assistant's response actually answers the user's question with real information.

Score the response on these criteria (0-10 each):
1. RELEVANCE: Does the response address the specific question asked? Score 0 if it doesn't mention the topic at all.
2. COMPLETENESS: Does it provide the information the user was seeking? Score 0 for empty acknowledgments like "Sure!", "OK!", "Got it!" that provide no actual information.
3. ACCURACY: Is the information factually plausible (based on any context provided)? Score 0 if no factual information is provided.
4. NO_DEFLECTION: Does it avoid generic greetings, deflections like "How can I help you?", or empty acknowledgments? Score 0 for responses under 20 characters that don't answer the question.

IMPORTANT: A response that just acknowledges without providing any actual information (e.g., "Sure thing!", "OK!", "Got it!") should score 0 on COMPLETENESS and fail overall.

Output your evaluation in this EXACT format:
RELEVANCE: [0-10]
COMPLETENESS: [0-10]
ACCURACY: [0-10]
NO_DEFLECTION: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    user_prompt = f"""User Query: {query}

Assistant Response: {response}"""

    if context:
        user_prompt += f"\n\nContext (data available to assistant):\n{context[:2000]}"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Fallback to heuristic evaluation if judge fails
        return JudgeVerdict(
            is_passed=not is_generic_greeting(response) and len(response) > 50,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    # Parse the judge response
    return _parse_judge_response(judge_response)


def judge_search_query_quality(
    user_query: str,
    search_query: str,
    location: Optional[str] = None,
    time_context: Optional[str] = None
) -> JudgeVerdict:
    """
    Use LLM to judge if the search query is well-formed for the user's intent.

    Args:
        user_query: What the user asked
        search_query: The search query the assistant generated
        location: User's known location (should be included if relevant)
        time_context: Time-related context (e.g., "this week", "tomorrow")

    Returns:
        JudgeVerdict evaluating search query quality
    """
    system_prompt = """You are evaluating search queries generated by a voice assistant.

Score the search query on these criteria (0-10 each):
1. INTENT_MATCH: Does the search query capture the user's actual intent?
2. LOCATION_AWARENESS: If location is known and relevant, is it included appropriately?
3. TIME_AWARENESS: If the query has time context, is it reflected in the search?
4. SPECIFICITY: Is the query specific enough to get useful results?

Output your evaluation in this EXACT format:
INTENT_MATCH: [0-10]
LOCATION_AWARENESS: [0-10]
TIME_AWARENESS: [0-10]
SPECIFICITY: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    user_prompt = f"""User Query: "{user_query}"
Generated Search Query: "{search_query}"
"""
    if location:
        user_prompt += f"User's Known Location: {location}\n"
    if time_context:
        user_prompt += f"Time Context: {time_context}\n"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Heuristic fallback
        has_location = location and any(
            loc_part.lower() in search_query.lower()
            for loc_part in location.split(",")[0].split()
        )
        return JudgeVerdict(
            is_passed=has_location if location else True,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    return _parse_judge_response(judge_response)


def _parse_judge_response(response: str) -> JudgeVerdict:
    """Parse the structured judge response into a JudgeVerdict."""
    lines = response.strip().split("\n")
    criteria_scores = {}
    is_passed = False
    reasoning = ""

    for line in lines:
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()

            if key == "OVERALL":
                is_passed = "PASS" in value.upper()
            elif key == "REASONING":
                reasoning = value
            else:
                # Try to parse as score
                try:
                    score = float(value.split()[0])
                    criteria_scores[key.lower()] = score / 10.0  # Normalize to 0-1
                except (ValueError, IndexError):
                    pass

    # Calculate average score
    avg_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 0.5

    return JudgeVerdict(
        is_passed=is_passed,
        score=avg_score,
        reasoning=reasoning,
        criteria_scores=criteria_scores
    )


def judge_tool_usage_appropriateness(
    query: str,
    tools_called: List[str],
    tool_args: List[Dict[str, Any]],
    expected_tools: Optional[List[str]] = None
) -> JudgeVerdict:
    """
    Judge whether the tools used were appropriate for the query.

    Args:
        query: User's question
        tools_called: List of tool names that were called
        tool_args: List of arguments passed to each tool
        expected_tools: Optional list of tools that should have been called

    Returns:
        JudgeVerdict on tool usage
    """
    system_prompt = """You are evaluating tool usage by a voice assistant.

Score on these criteria (0-10 each):
1. TOOL_SELECTION: Were the right tools chosen for the task?
2. ARG_QUALITY: Were the tool arguments well-formed and appropriate?
3. EFFICIENCY: Was there unnecessary tool calling or missing necessary calls?

Output your evaluation in this EXACT format:
TOOL_SELECTION: [0-10]
ARG_QUALITY: [0-10]
EFFICIENCY: [0-10]
OVERALL: [PASS/FAIL]
REASONING: [One paragraph explaining your verdict]"""

    tool_info = "\n".join([
        f"- {name}: {args}" for name, args in zip(tools_called, tool_args)
    ]) if tools_called else "No tools called"

    user_prompt = f"""User Query: "{query}"

Tools Called:
{tool_info}
"""
    if expected_tools:
        user_prompt += f"\nExpected Tools: {', '.join(expected_tools)}"

    judge_response = call_judge_llm(system_prompt, user_prompt)

    if not judge_response:
        # Heuristic fallback
        has_expected = not expected_tools or all(t in tools_called for t in expected_tools)
        return JudgeVerdict(
            is_passed=has_expected,
            score=0.5,
            reasoning="Judge LLM unavailable, using heuristic fallback"
        )

    return _parse_judge_response(judge_response)

