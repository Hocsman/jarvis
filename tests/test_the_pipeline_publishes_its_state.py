"""The voice pipeline tells whoever is watching what it is doing.

The orb, the dashboard and the face read one shared state (``jarvis.state``);
this is the other side of that contract. Each stage of the pipeline publishes
the state it enters, with no desktop app in the process: the state a viewer
would read is observed here through the same reader a viewer uses.

Who watches is none of the pipeline's business, which is why none of these
tests import a window.
"""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from jarvis.state import JarvisState, get_jarvis_state


def _published() -> JarvisState:
    """What a viewer reading the shared state sees right now."""
    return get_jarvis_state().state


def _wait_for_published(expected: JarvisState, timeout: float = 2.0) -> bool:
    """Wait for a state published from a timer thread to become visible."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _published() == expected:
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Speech engines
# ---------------------------------------------------------------------------


def _piper():
    from jarvis.output.tts import PiperTTS

    return PiperTTS(enabled=False)


def _kokoro():
    from jarvis.output.tts import KokoroTTS

    return KokoroTTS.__new__(KokoroTTS)


def _chatterbox():
    from jarvis.output.tts import ChatterboxTTS

    return ChatterboxTTS.__new__(ChatterboxTTS)


@pytest.mark.unit
@pytest.mark.parametrize("make_engine", [_piper, _kokoro, _chatterbox], ids=["piper", "kokoro", "chatterbox"])
class TestSpeechEnginesPublishSpeaking:
    def test_the_state_becomes_speaking_when_speech_starts(self, make_engine):
        get_jarvis_state().set_state(JarvisState.THINKING)

        make_engine()._notify_speaking_state(True)

        assert _published() == JarvisState.SPEAKING

    def test_the_state_is_left_alone_when_speech_ends(self, make_engine):
        """What follows speech (a hot window, idle) is the daemon's call, so an
        engine that stops must not guess."""
        get_jarvis_state().set_state(JarvisState.LISTENING)

        make_engine()._notify_speaking_state(False)

        assert _published() == JarvisState.LISTENING


# ---------------------------------------------------------------------------
# Listening states
# ---------------------------------------------------------------------------


@pytest.fixture
def state_manager():
    from jarvis.listening.state_manager import StateManager

    manager = StateManager(hot_window_seconds=0.2, echo_tolerance=0.01)
    yield manager
    manager.stop()


@pytest.mark.unit
class TestListeningStatesArePublished:
    @patch("builtins.print")
    def test_collecting_a_query_is_listening(self, _print, state_manager):
        get_jarvis_state().set_state(JarvisState.IDLE)

        state_manager.start_collection("what time is it")

        assert _published() == JarvisState.LISTENING

    @patch("builtins.print")
    def test_a_hot_window_is_listening_and_its_expiry_is_idle(self, _print, state_manager):
        get_jarvis_state().set_state(JarvisState.IDLE)

        state_manager.schedule_hot_window_activation()

        assert _wait_for_published(JarvisState.LISTENING), "an open hot window shows as listening"
        assert _wait_for_published(JarvisState.IDLE), "a window that ran out shows as idle again"

    @patch("builtins.print")
    def test_closing_the_hot_window_by_hand_is_idle(self, _print, state_manager):
        state_manager.schedule_hot_window_activation()
        assert _wait_for_published(JarvisState.LISTENING)

        state_manager.expire_hot_window()

        assert _published() == JarvisState.IDLE


# ---------------------------------------------------------------------------
# Reply engine
# ---------------------------------------------------------------------------


def _engine_cfg():
    cfg = Mock()
    cfg.ollama_base_url = "http://localhost:11434"
    cfg.ollama_chat_model = "test-large"
    cfg.llm_chat_model = "test-large"
    cfg.voice_debug = False
    cfg.llm_tools_timeout_sec = 8.0
    cfg.tool_router_timeout_sec = 8.0
    cfg.llm_embedding_timeout_sec = 10.0
    cfg.llm_chat_timeout_sec = 45.0
    cfg.llm_digest_timeout_sec = 8.0
    cfg.memory_enrichment_max_results = 5
    cfg.memory_enrichment_source = "diary"
    cfg.memory_digest_enabled = False
    cfg.tool_result_digest_enabled = False
    cfg.location_ip_address = None
    cfg.location_auto_detect = False
    cfg.location_enabled = False
    cfg.agentic_max_turns = 8
    cfg.tool_search_max_calls = 3
    cfg.tool_selection_strategy = "all"
    cfg.tool_carryover_max_turns = 2
    cfg.tool_carryover_per_entry_chars = 1200
    cfg.mcps = {}
    cfg.llm_thinking_enabled = False
    cfg.tts_engine = "none"
    cfg.ollama_embed_model = "test-embed"
    return cfg


@pytest.mark.unit
@patch("src.jarvis.reply.engine.plan_query", return_value=[])
@patch("src.jarvis.reply.engine.extract_search_params_for_memory", return_value={})
@patch("src.jarvis.reply.engine.run_tool_with_retries")
@patch("src.jarvis.reply.engine.extract_text_from_response", return_value="")
@patch("src.jarvis.reply.engine.chat_with_messages")
def test_a_dismissal_sends_the_assistant_back_to_idle(
    mock_chat, _extract, mock_tool, _params, _plan
):
    """The stop tool ends the conversation without a reply; nothing will be
    spoken to move the state on, so the engine itself settles it."""
    from src.jarvis.memory.conversation import DialogueMemory
    from src.jarvis.reply.engine import run_reply_engine
    from src.jarvis.tools.builtin.stop import STOP_SIGNAL
    from src.jarvis.tools.types import ToolExecutionResult

    mock_tool.return_value = ToolExecutionResult(success=True, reply_text=STOP_SIGNAL, error_message=None)
    mock_chat.return_value = {"message": {"content": "", "tool_calls": [{
        "id": "c1", "type": "function", "function": {"name": "stop", "arguments": {}},
    }]}}
    get_jarvis_state().set_state(JarvisState.THINKING)

    with patch("builtins.print"):
        reply = run_reply_engine(db=Mock(), cfg=_engine_cfg(), tts=None, text="stop",
                                 dialogue_memory=DialogueMemory())

    assert reply is None
    assert _published() == JarvisState.IDLE
