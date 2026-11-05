"""
Tests for user input processing around the hot window.

These tests verify observable behaviour: given a sequence of events (TTS finishes,
user speaks, time passes), does the system accept or reject the input, and does
the accepted query contain the right text?

Tests exercise VoiceListener._process_transcript with mocked TTS and intent judge
but use real StateManager and EchoDetector instances to avoid coupling to internals.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from rapidfuzz import fuzz

from conftest import _created_listeners
from jarvis.listening.echo_detection import EchoDetector
from jarvis.listening.state_manager import StateManager, ListeningState
from jarvis.listening.intent_judge import IntentJudge, IntentJudgment
from jarvis.listening.listener import _Turn
from jarvis.listening.wake_detection import is_wake_word_detected
from jarvis.config import get_default_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_listener(**kwargs):
    """Create a VoiceListener with mocked heavy subsystems.

    Returns (listener, mock_tts) so tests can control TTS state.
    Uses real StateManager and EchoDetector — only Whisper, audio, and
    the intent judge are mocked.
    """
    mock_cfg = MagicMock()
    mock_cfg.whisper_model = "small"
    mock_cfg.whisper_device = "auto"
    mock_cfg.whisper_compute_type = "int8"
    mock_cfg.whisper_backend = "faster-whisper"
    mock_cfg.sample_rate = 16000
    mock_cfg.vad_enabled = False
    mock_cfg.vad_aggressiveness = 2
    mock_cfg.echo_tolerance = kwargs.get("echo_tolerance", 0.3)
    mock_cfg.echo_energy_threshold = 2.0
    mock_cfg.hot_window_seconds = kwargs.get("hot_window_seconds", 3.0)
    mock_cfg.hot_window_enabled = True
    mock_cfg.voice_collect_seconds = 2.0
    mock_cfg.voice_max_collect_seconds = 60.0
    mock_cfg.voice_device = None
    mock_cfg.voice_debug = False
    mock_cfg.voice_min_energy = 0.0045
    mock_cfg.tune_enabled = False
    mock_cfg.wake_word = "jarvis"
    mock_cfg.wake_aliases = []
    mock_cfg.wake_fuzzy_ratio = 0.78
    mock_cfg.stop_commands = ["stop", "quiet"]
    mock_cfg.tts_rate = 200
    mock_cfg.transcript_buffer_duration_sec = 120.0
    mock_cfg.intent_judge_model = "gemma4:e2b"
    mock_cfg.ollama_base_url = "http://127.0.0.1:11434"
    mock_cfg.intent_judge_timeout_sec = 3.0
    mock_cfg.whisper_language = None
    mock_db = MagicMock()
    mock_tts = MagicMock()
    mock_tts.enabled = True
    mock_tts.is_speaking.return_value = kwargs.get("tts_speaking", False)
    mock_dialogue_memory = MagicMock()

    with patch("jarvis.listening.listener.webrtcvad", None), \
         patch("jarvis.listening.listener.sd", None), \
         patch("jarvis.listening.listener.np", None), \
         patch("jarvis.listening.listener.create_intent_judge", return_value=None):
        from jarvis.listening.listener import VoiceListener
        listener = VoiceListener(mock_db, mock_cfg, mock_tts, mock_dialogue_memory)

    _created_listeners.append(listener)
    return listener, mock_tts


def _make_judgment(directed=True, query="", stop=False, confidence="high", reasoning="test",
                   noise=False):
    """Build an IntentJudgment."""
    return IntentJudgment(
        directed=directed, query=query, stop=stop,
        confidence=confidence, reasoning=reasoning, noise=noise,
    )


def _install_intent_judge(listener, judgment):
    """Replace the listener's intent judge with a mock returning *judgment*."""
    mock_judge = MagicMock()
    mock_judge.available = True
    mock_judge.judge.return_value = judgment
    listener._intent_judge = mock_judge
    return mock_judge


def _prompt_shown_to(judge, call=-1) -> str:
    """The prompt the mocked *judge* was given on that call, built the way
    the real judge builds it."""
    kwargs = judge.judge.call_args_list[call].kwargs
    return IntentJudge()._build_user_prompt(**{
        k: v for k, v in kwargs.items()
        if k in ("segments", "wake_timestamp", "last_tts_text",
                 "last_tts_finish_time", "in_hot_window", "current_text")})


def _simulate_tts_finish(listener):
    """Simulate TTS finishing: track finish time and schedule hot window activation."""
    listener.echo_detector.track_tts_finish()
    listener.state_manager.schedule_hot_window_activation()


def _wait_for_hot_window_active(listener, timeout=0.5):
    """Wait until hot window is formally active (past echo_tolerance delay)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if listener.state_manager.is_hot_window_active():
            return True
        time.sleep(0.01)
    return False


def _wait_for_hot_window_expiry(listener, timeout=2.0):
    """Wait until the expiry timer has closed the window.

    The window's clock runs on a timer thread, and a busy machine can wake
    that thread tens of milliseconds late; a fixed sleep sized to the window
    then finds it still open. Waiting on the span's end observes the timer
    itself, so the test measures the window and not the scheduler. A window
    that never opened never closes either, and shows up as a timeout.

    Before returning, step past the clock's granularity: the span's end is
    stamped when the timer observes the expiry, and the acceptance check
    includes that boundary (start <= span_end). A caller stamping an
    "after expiry" time the instant this returns can land on the same
    millisecond tick as the boundary and read as inside the window, which
    is the flake this margin removes.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if (listener.state_manager._hot_window_span_end > 0
                and not listener.state_manager.is_hot_window_active()):
            time.sleep(0.025)
            return True
        time.sleep(0.01)
    return False


def _accepted_query(listener) -> str:
    """Return the accepted query text, or empty string if input was rejected."""
    if listener.state_manager.get_pending_query():
        return listener.state_manager.get_pending_query()
    return ""


# ---------------------------------------------------------------------------
# Tests: User speaks during active hot window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserSpeaksDuringHotWindow:
    """TTS finishes, hot window activates, user speaks within the window."""

    @patch("builtins.print")
    def test_directed_follow_up_is_accepted(self, _print):
        """User's follow-up question during hot window is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The weather is sunny today.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_undirected_background_speech_is_accepted_in_hot_window(self, _print):
        """Non-echo speech during hot window is accepted even if judge says not directed.

        The 3s hot window is short enough that false positives (accepting
        background speech) are preferable to false negatives (ignoring genuine
        follow-ups like 'don't you already know that?'). Small LLMs sometimes
        reject valid follow-ups, so we override in hot window mode.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Here is your answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="background conversation"))

        listener._process_transcript("did you see the game last night", utterance_energy=0.01)

        # In hot window, non-echo speech is always accepted
        assert _accepted_query(listener) == "did you see the game last night"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_query_is_used_in_hot_window(self, _print):
        """In hot window, the intent judge's extracted query is authoritative.

        The judge is the canonical echo-stripper and noise-pruner; its output
        always wins over the raw transcript. This prevents partial-salvage
        leakage where echo fragments ride through on the raw text. If the
        judge returns an empty query, the listener falls back to raw text.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Do you want to know more?")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what is the weather tomorrow"))

        listener._process_transcript(
            "uh okay what is the weather tomorrow", utterance_energy=0.01)

        assert _accepted_query(listener) == "what is the weather tomorrow"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_empty_judge_query_falls_back_to_raw_text(self, _print):
        """If the judge is directed but returns no query, fall back to raw text."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Do you want to know more?")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query=""))

        listener._process_transcript("tell me a joke please", utterance_energy=0.01)

        assert _accepted_query(listener) == "tell me a joke please"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: User starts speaking during hot window, transcript arrives after expiry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTranscriptArrivesAfterHotWindowExpiry:
    """User speaks during hot window but Whisper is slow — transcript arrives after expiry.

    Uses timestamp-based detection: utterance_start_time is compared against the
    hot window's time span, so it doesn't matter when Whisper finishes."""

    @patch("builtins.print")
    def test_speech_started_during_window_accepted_after_expiry(self, _print):
        """Speech that STARTED during the hot window is accepted even after expiry.

        This is the core scenario: user starts speaking at 2.5s into a 3s window,
        Whisper takes 2s to transcribe, so transcript arrives at 4.5s — after
        "Returning to wake word mode". The timestamp check still detects the
        speech started during the window.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.08)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Speech starts during active window
        speech_start = time.time()

        # The transcript arrives after expiry (Whisper was slow)
        assert _wait_for_hot_window_expiry(listener)

        # Transcript arrives after expiry — but speech_start was during window
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_started_after_expiry_rejected(self, _print):
        """Speech starting AFTER window expired is rejected (requires wake word)."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Wait for hot window to expire
        assert _wait_for_hot_window_expiry(listener)

        # Speech starts AFTER expiry
        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_during_active_window_accepted_before_expiry(self, _print):
        """Voice processed while hot window is still active succeeds."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript(
            "tell me more", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_voice_during_pending_activation_accepted(self, _print):
        """Voice start during echo_tolerance delay (pending activation) still counts."""
        listener, _ = _create_listener(echo_tolerance=0.5, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Answer text.")
        _simulate_tts_finish(listener)

        # Hot window not yet active (still in echo_tolerance delay)
        assert not listener.state_manager.is_hot_window_active()

        # Speech starts now during pending period
        speech_start = time.time()

        _install_intent_judge(listener, _make_judgment(directed=True, query="yes please"))
        listener._process_transcript(
            "yes please", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=time.time())

        assert _accepted_query(listener) == "yes please"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_minutes_after_window_not_treated_as_hot(self, _print):
        """Speech a minute after hot window expired is NOT treated as hot window.

        Regression test: a stale boolean flag previously caused speech long
        after the window to be treated as hot window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Quick answer.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Wait for window to expire
        assert _wait_for_hot_window_expiry(listener)

        # Simulate speech "a minute later" (use a start time well after expiry)
        speech_start = time.time() + 0.5  # even 500ms later should be rejected

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="something funny"))
        listener._process_transcript(
            "something funny", utterance_energy=0.01,
            utterance_start_time=speech_start, utterance_end_time=speech_start + 1.0)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo and user speech in the same Whisper chunk
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoAndUserSpeechInSameChunk:
    """Whisper merges echo + user speech into one transcript chunk."""

    @patch("builtins.print")
    def test_mixed_echo_and_speech_after_tts_accepted_in_hot_window(self, _print):
        """When echo + user speech arrive as one chunk in hot window, input is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "here is the answer"
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        now = time.time()
        # Intent judge sees the mixed text and marks it directed
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="thanks can you also check email"))

        # Mixed chunk: echo + user speech
        listener._process_transcript(
            "here is the answer thanks can you also check email",
            utterance_energy=0.01,
            utterance_start_time=now - 3.0,
            utterance_end_time=now - 0.5,
        )

        # Hot window uses raw text (intent judge handles echo stripping)
        query = _accepted_query(listener)
        assert query != ""
        assert "thanks" in query or "check email" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_plus_speech_from_during_tts_accepted_after_expiry(self, _print):
        """Mixed echo+speech chunk where VAD triggered during TTS is accepted
        even after the hot window expires.

        Real scenario: TTS plays, mic picks up echo (VAD triggers during TTS),
        user speaks during hot window, Whisper takes >3s to transcribe the long
        combined audio, hot window expires, transcript arrives.

        The utterance started BEFORE the hot window span (during TTS) but
        ended DURING the span (user spoke during window). The system should
        recognise this overlap and treat it as hot window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "Got it. I will keep my responses short and to the point from now on."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Manually expire hot window (simulates Whisper taking >3s)
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Intent judge correctly extracts user speech from mixed transcript
        _install_intent_judge(listener, _make_judgment(
            directed=True,
            query="tell me something random"))

        # Mixed chunk: full TTS echo + user speech appended
        # utterance_start_time is BEFORE span_start (VAD triggered during TTS)
        # utterance_end_time is AFTER span_start (user spoke during window)
        mixed_text = (
            "Got it. I will keep my responses short and to the point from now on. "
            "Yeah, I guess that's fine, but tell me something random."
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech where utterance overlaps hot window should be "
            "accepted, not dropped because utterance_start_time < span_start"
        )
        assert "random" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_mixed_echo_speech_unsalvaged_uses_judge_extraction(self, _print):
        """When salvage fails to strip echo, the post-judge echo check should
        use the intent judge's extraction instead of rejecting everything.

        If the heard text is much longer than TTS (mixed content), the echo
        check should recognise it's not pure echo and fall through to use the
        judge's extracted query.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "The current temperature is around nine degrees celsius."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Intent judge correctly extracts user speech
        _install_intent_judge(listener, _make_judgment(
            directed=True,
            query="what will it be tomorrow"))

        # Mixed text where salvage won't work (Whisper transcribed echo differently
        # from TTS text, so exact word matching fails). User speech is substantially
        # longer than TTS echo so word count guard lets it through.
        mixed_text = (
            "the temperature is about 9 degrees. "
            "yeah I figured as much but what will it be like tomorrow afternoon"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech should not be rejected when text is longer than TTS"
        )
        assert "tomorrow" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_echo_reasoning_overridden_for_mixed_content_in_hot_window(self, _print):
        """When the intent judge says 'not directed' with echo reasoning but the
        utterance overlaps the hot window and text is longer than TTS (mixed
        echo+speech), the rejection should be overridden.

        Real scenario: TTS plays, mic picks up echo + user speaks during hot window,
        hot window expires, Whisper delivers mixed transcript. Intent judge sees TTS
        text in transcript and says 'echo, not directed'. But the word-count guard
        shows it's mixed content and could_be_hot_window is True, so the override
        should kick in.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "You are currently in Tbilisi, Georgia."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Hot window expires (Whisper is slow)
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Intent judge incorrectly classifies as echo (sees TTS text in transcript)
        _install_intent_judge(listener, _make_judgment(
            directed=False,
            query="",
            confidence="high",
            reasoning="echo of TTS output"))

        mixed_text = (
            "you are currently in T-Ballista Georgia and what do you think "
            "about Joseph Stalin and communism in general?"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Mixed echo+speech should be accepted in hot window even when "
            "intent judge says 'echo, not directed' — word count shows mixed content"
        )
        assert "stalin" in query.lower() or "communism" in query.lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_returns_none_hot_window_speech_still_accepted(self, _print):
        """When the intent judge times out or errors (returns None), hot window
        speech carrying words she did not say, in a voice, is still accepted.

        Real scenario: user speaks during hot window, Whisper delivers mixed
        echo+speech, intent judge times out on the long transcript. The beep
        started (early check passed) but the query is silently dropped because
        the judge-None path falls through to wake word detection.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "You are currently in Tbilisi, Georgia."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        span_start = listener.state_manager._hot_window_span_start

        # Hot window expires (Whisper is slow)
        listener.state_manager.expire_hot_window()

        # Intent judge returns None (timeout)
        _install_intent_judge(listener, None)

        mixed_text = (
            "you are currently in T-Ballista Georgia and what do you think "
            "about Joseph Stalin and communism in general?"
        )
        listener._process_transcript(
            mixed_text,
            utterance_energy=0.01,
            utterance_start_time=span_start - 2.0,
            utterance_end_time=span_start + 0.05,
        )

        query = _accepted_query(listener)
        assert query != "", (
            "Hot window speech should be accepted even when intent judge "
            "times out — it carries words she did not say, in a voice"
        )
        assert "stalin" in query.lower() or "communism" in query.lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_utterance_starting_during_tts_ending_after_treated_as_hot_window(self, _print):
        """Utterance that starts before TTS finishes is still treated as hot window context."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("some response text")
        tts_finish = time.time()
        listener.echo_detector.track_tts_finish()
        listener.state_manager.schedule_hot_window_activation()
        _wait_for_hot_window_active(listener)

        # Utterance started 0.5s BEFORE TTS finished, ended 1s after
        utterance_start = tts_finish - 0.5
        utterance_end = tts_finish + 1.0

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))

        listener._process_transcript(
            "tell me more",
            utterance_energy=0.01,
            utterance_start_time=utterance_start,
            utterance_end_time=utterance_end,
        )

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_early_echo_check_salvages_trailing_user_speech(self, _print):
        """Early echo check must salvage user speech appended after an echo prefix.

        Whisper often merges the tail of TTS echo with the user's follow-up into
        one transcript. The early fuzzy echo check used to reject the whole chunk,
        so the user's real speech was dropped before the intent judge could see it.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = (
            "I do have a tool to check the weather, but I need to use it with a "
            "location. I can check the forecast for London for you right now."
        )
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="yeah go ahead and do that"))

        # Mixed chunk: exact tail of TTS echo + user's follow-up
        listener._process_transcript(
            "I can check the forecast for London for you right now. "
            "Yeah, go ahead and do that.",
            utterance_energy=0.01,
        )

        query = _accepted_query(listener)
        assert query != "", "Trailing user speech should be salvaged, not rejected as echo"
        assert "go ahead" in query.lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_early_echo_salvage_accepts_at_minimum_word_count(self, _print):
        """Salvaged remainder at exactly min_salvage_words should be accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        min_words = listener.echo_detector.min_salvage_words

        tts_text = "The weather is going to be sunny today in London."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        follow_up_words = ["thanks", "tell", "me", "more", "please"][:min_words]
        follow_up = " ".join(follow_up_words)
        _install_intent_judge(listener, _make_judgment(directed=True, query=follow_up))

        listener._process_transcript(
            f"{tts_text} {follow_up}",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) != "", (
            f"Remainder of exactly {min_words} words should be salvaged"
        )
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_early_echo_salvage_rejects_below_minimum_word_count(self, _print):
        """Salvaged remainder below min_salvage_words should be rejected as echo."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        min_words = listener.echo_detector.min_salvage_words

        tts_text = "The weather is going to be sunny today in London."
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        short_tail = " ".join(["really", "nice"][: max(min_words - 1, 1)])
        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query=short_tail, reasoning="should not be consulted"))

        listener._process_transcript(
            f"{tts_text} {short_tail}",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == "", (
            f"Remainder below {min_words} words should be rejected"
        )
        judge.judge.assert_not_called()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_early_echo_salvage_rejects_when_no_prefix_match(self, _print):
        """If cleanup_leading_echo can't strip any prefix, fall back to rejection."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        tts_text = "alpha beta gamma delta epsilon zeta"
        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=False, query="", reasoning="should not be consulted"))

        # Shares enough words with TTS to clear partial_ratio >= 70 (marks it
        # echo) but the tokens are in a different order so cleanup_leading_echo
        # cannot find a matching prefix — nothing to salvage.
        listener._process_transcript(
            "beta alpha delta gamma zeta epsilon",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == "", (
            "Chunk with no strippable prefix should be rejected as pure echo"
        )
        judge.judge.assert_not_called()
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Grace period boundaries
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHotWindowOnlyFromStateManager:
    """Hot window status comes exclusively from the state manager's formal
    activation/expiry — not from time-based grace periods. This prevents
    false hot window claims after the user has seen 'Returning to wake word mode'."""

    @patch("builtins.print")
    def test_recent_tts_without_hot_window_activation_not_treated_as_hot(self, _print):
        """TTS finishing without hot window activation does not create a hot window."""
        listener, _ = _create_listener(
            hot_window_seconds=3.0,
            echo_tolerance=0.3,
        )

        # Track TTS finish but do NOT schedule hot window activation
        listener.echo_detector.track_tts_start("answer text")
        listener.echo_detector.track_tts_finish()

        # Judge says directed, but no wake word and no hot window
        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript("thanks", utterance_energy=0.01)

        # Should NOT be accepted — no hot window active, no wake word
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_formal_hot_window_activation_required(self, _print):
        """Only formally activated hot window allows wake-word-free input."""
        listener, _ = _create_listener(
            hot_window_seconds=3.0,
            echo_tolerance=0.02,
        )

        listener.echo_detector.track_tts_start("old answer")
        listener.echo_detector.track_tts_finish()
        tts_finish = listener.echo_detector._last_tts_finish_time

        # Judge says directed, but no wake word in text — should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="hello there"))

        listener._process_transcript(
            "hello there",
            utterance_energy=0.01,
            utterance_start_time=tts_finish + 0.5,
            utterance_end_time=tts_finish + 1.0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_with_active_hot_window_accepted(self, _print):
        """When Whisper provides no timestamps but hot window is active, accepted."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.02)

        listener.echo_detector.track_tts_start("recent response")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="and also"))

        listener._process_transcript(
            "and also",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted_query(listener) == "and also"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_timestamps_without_hot_window_rejected(self, _print):
        """When Whisper provides no timestamps and no hot window, requires wake word."""
        listener, _ = _create_listener(hot_window_seconds=3.0, echo_tolerance=0.3)

        listener.echo_detector.track_tts_start("stale response")
        # TTS finished but no hot window scheduled
        listener.echo_detector.track_tts_finish()

        _install_intent_judge(listener, _make_judgment(directed=True, query="random remark"))

        listener._process_transcript(
            "random remark",
            utterance_energy=0.01,
            utterance_start_time=0,
            utterance_end_time=0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo rejection does NOT extend the hot window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoRejectionDoesNotExtendFollowUpWindow:
    """Echo is caught early (instant fuzzy check), so it doesn't block the
    audio loop or extend the hot window. The original window duration applies."""

    @patch("builtins.print")
    def test_echo_does_not_reset_window_timer(self, _print):
        """Echo rejection leaves the original window timer untouched."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The answer is 42.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        original_start = listener.state_manager._hot_window_start_time

        # Feed echo — caught early
        listener._process_transcript("The answer is 42", utterance_energy=0.01)

        # Window timer should not have been reset
        assert listener.state_manager._hot_window_start_time == original_start
        # Window still active (within original 3s)
        assert listener.state_manager.is_hot_window_active()

        # User speaks within the original window
        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))
        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_after_window_expiry_does_not_reactivate(self, _print):
        """Late echo arrival after window expired does NOT reactivate the window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short reply.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Let hot window expire
        assert _wait_for_hot_window_expiry(listener)

        # Late echo arrives — window should stay expired
        listener._process_transcript("Short reply", utterance_energy=0.01)
        assert not listener.state_manager.is_hot_window_active()

        # Speech without wake word should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="one more thing"))
        listener._process_transcript("one more thing", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


@pytest.mark.unit
class TestLongTtsTailEcho:
    """Echoes of the TAIL of a long TTS response must still be rejected. The
    fuzzy echo check previously truncated TTS to 300 chars, so tail echoes from
    longer responses slipped through and were accepted as user speech."""

    @patch("builtins.print")
    def test_tail_echo_from_long_tts_rejected(self, _print):
        """Echo of the final clause of a ~370-char TTS is caught, not accepted."""
        long_tts = (
            "You asked for something interesting, so I found that there are "
            "over 1800 creative writing prompts available across various genres, "
            "including themes like a character losing the ability to create or "
            "an intangible concept becoming a real object. I also found that "
            "evolving marketing tactics rely on using data, leveraging "
            "analytics, and being agile to understand user behavior."
        )
        assert len(long_tts) > 300  # Guard: the bug only manifests past old cap

        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.echo_detector.track_tts_start(long_tts)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Mic picks up the tail of the TTS response — this is pure echo.
        tail_echo = "leveraging analytics and being agile to understand user behavior."
        _install_intent_judge(
            listener,
            _make_judgment(directed=False, reasoning="Segment is an echo"),
        )
        listener._process_transcript(tail_echo, utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Early beep and published state feedback
# ---------------------------------------------------------------------------

def _is_beeping(listener) -> bool:
    """Check if the thinking tune is currently active."""
    return listener._tune_player is not None


@pytest.mark.unit
class TestEarlyBeepFeedback:
    """Beep should start immediately after Whisper transcription, before the
    intent judge runs. This gives instant auditory feedback to the user."""

    @patch("builtins.print")
    def test_beep_starts_on_wake_word_before_intent_judge(self, _print):
        """Beep starts right after 'Heard' when wake word is present."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # No intent judge installed — beep should still start from the
        # early detection path, then fallback wake word check processes query.
        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert _accepted_query(listener) != ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_beep_starts_in_hot_window_before_intent_judge(self, _print):
        """Beep starts right after 'Heard' when in hot window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        listener.echo_detector.track_tts_start("Here is the answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == "tell me more"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_without_wake_word_or_hot_window(self, _print):
        """No beep when there's no wake word and not in hot window."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # Random speech, no wake word, no hot window
        listener._process_transcript("the weather is nice today", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        # Beep should not have been started (and if it was, it was stopped)
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_beep_stops_when_intent_judge_rejects(self, _print):
        """Early beep is stopped if intent judge rejects the input."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        # Install judge that rejects — speech has wake word so early beep fires,
        # but judge says not directed so beep should be stopped.
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="narrative mention"))

        listener._process_transcript("jarvis is a cool name", utterance_energy=0.01)

        # Query should NOT be accepted (judge rejected + fallback wake word
        # check won't find a query after "jarvis")
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_during_tts_playback(self, _print):
        """Beep does not start while TTS is actively speaking."""
        listener, mock_tts = _create_listener(
            echo_tolerance=0.02, hot_window_seconds=3.0, tts_speaking=True)
        listener.cfg.tune_enabled = True

        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        # Should not beep during TTS (stop command path handles TTS interrupts)
        assert not _is_beeping(listener)
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo caught early in hot window (no intent judge, no window reset)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoRejectionInHotWindow:
    """Echo in the hot window is caught by the early fuzzy check before
    the intent judge runs. The hot window timer is NOT reset."""

    @patch("builtins.print")
    def test_confirmed_echo_rejected_without_intent_judge(self, _print):
        """Echo matching TTS is caught early — intent judge never runs."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "The weather will be sunny tomorrow."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="echo of assistant speech"))

        listener._process_transcript(
            "the weather will be sunny tomorrow",
            utterance_energy=0.01)

        # Echo caught early — no query accepted, no intent judge called
        assert _accepted_query(listener) == ""
        judge.judge.assert_not_called()
        # Hot window still active (within original 3s, NOT reset)
        assert listener.state_manager.is_hot_window_active()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_rejected_before_intent_judge_can_accept(self, _print):
        """Echo is caught early even when intent judge would say directed.

        The mic picks up Jarvis's TTS output and Whisper transcribes it.
        The early fuzzy check catches it before the intent judge runs.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "Georgian cuisine is incredibly rich and you should try Khachapuri and Georgian bread."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="and kg chai like georgian bread",
            confidence="high", reasoning="user follow-up"))

        listener._process_transcript(
            "and kg chai like georgian bread",
            utterance_energy=0.01)

        # Echo caught early — no query accepted
        assert _accepted_query(listener) == ""
        # Intent judge never called
        judge.judge.assert_not_called()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_non_echo_speech_accepted_via_override(self, _print):
        """Non-echo speech in hot window is accepted even if judge rejects.

        In hot window, non-echo speech is always accepted (override), since
        small LLMs sometimes reject valid follow-ups.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "The weather will be sunny tomorrow."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Judge rejects unrelated speech
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="background conversation"))

        listener._process_transcript(
            "did you see the game last night",
            utterance_energy=0.01)

        # Non-echo speech in hot window is accepted via override
        assert _accepted_query(listener) == "did you see the game last night"
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: His answer, begun the moment her reply ends
# ---------------------------------------------------------------------------

# The microphone's level when she started speaking, and how loud each source
# reaches it, as multiples of the bar the echo detector sets over that floor.
_FLOOR = 0.005
_HIS_VOICE = 5.0
_HER_VOICE = 3.0     # her reply through the speakers clears the bar too
_ROOM_NOISE = 0.5

_HER_REPLY = "Bien reçu, Monsieur. Je reste à votre écoute : que souhaitez-vous savoir ce soir ?"
_HIS_QUESTION = "quel temps fera-t-il demain matin à lyon ?"
# Her reply coming back decimated by Whisper, words dropped and others bent,
# nearly all of them still hers: the contiguous echo check misses it.
_HER_WORDS_DECIMATED = "reçu monsieur reste écoute souhaitez vous soir"
# Her reply when it names her, and when it says a word the wake-word check
# takes for her name, each coming back decimated as she stops.
_HER_REPLY_NAMING_HER = ("Bien reçu, Monsieur, ici Jarvis. Je reste à votre écoute : "
                         "que souhaitez-vous savoir ce soir ?")
_HER_NAME_DECIMATED = "reçu monsieur jarvis reste écoute souhaitez vous soir"
_HER_REPLY_WITH_A_WORD_LIKE_HER_NAME = (
    "Bien reçu, Monsieur. À mon avis, il vaut mieux partir tôt demain matin pour "
    "éviter les bouchons. Qu'en pensez-vous ?")
_A_WORD_LIKE_HER_NAME_DECIMATED = "reçu monsieur mon avis mieux partir matin éviter bouchons pensez"
# Her reply holds no word the wake-word check takes for her name, and
# coming back Whisper bends one of hers into one: "servi" heard "servis",
# close to the alias "jervis".
_HER_REPLY_WITH_A_WORD_WHISPER_BENDS = (
    "Le dîner est servi, Monsieur. Voulez-vous que je baisse un peu la lumière du salon ?")
_A_WORD_BENT_INTO_HER_NAME = "dîner servis monsieur voulez baisse peu lumière salon"


def _with_the_default_aliases(listener):
    """The wake word as configured by default, with its aliases: the
    spellings Whisper gives her name."""
    listener.cfg.wake_aliases = list(get_default_config()["wake_aliases"])
    return listener


def _the_judge_gave_no_verdict(mock_print) -> bool:
    """Whether the console says the judge gave no verdict, as the listener
    says it before deciding without one."""
    return any("Intent judge: unavailable" in line for line in _console_lines(mock_print))


def _level(listener, multiple: float) -> float:
    detector = listener.echo_detector
    return _FLOOR * detector.energy_spike_threshold * multiple


def _just_after(listener, last_word: float) -> float:
    """When his answer starts if he answers at once: inside the echo
    tolerance after her last word, the voice detector's pre-roll counted."""
    return last_word + listener.echo_detector.echo_tolerance / 3


def _she_says(listener, reply: str) -> float:
    """She says *reply* to its last word: the end-of-reply callback stamps
    it and schedules the hot window. Returns when her voice stopped."""
    listener.echo_detector.track_tts_start(reply, baseline_energy=_FLOOR)
    last_word = time.time()
    _simulate_tts_finish(listener)
    return last_word


def _whisper_hears(listener, text: str, *, started: float, level: float) -> None:
    """One utterance through the audio loop's own path, from the voice
    detector closing it to the decision: it started at *started* on the
    capture clock (pre-roll included), closes now, its audio sits at
    *level*, and Whisper (faked) makes *text* of it."""
    cfg = listener.cfg
    cfg.whisper_min_audio_duration, cfg.whisper_min_confidence = 0.15, 0.3
    cfg.whisper_no_speech_threshold = 0.5
    listener._whisper_backend, listener._whisper_device = "faster-whisper", "cuda"
    listener.model = MagicMock()
    listener.model.transcribe.return_value = (
        [SimpleNamespace(text=f" {text}", avg_logprob=-0.1, no_speech_prob=0.01)],
        SimpleNamespace(language="fr"))
    listener.echo_detector.track_utterance_timing(started, 0.0)
    listener._frame_captured_at = time.time()
    listener._utterance_frames = [np.full(320, level, dtype=np.float32) for _ in range(50)]
    listener.is_speech_active = True
    listener._finalize_utterance()


def _current_line(prompt: str) -> str:
    """The line of a judge prompt that holds the utterance being judged."""
    return next((line for line in prompt.splitlines()
                 if "CURRENT - JUDGE THIS" in line), "")


def _judge_that_reads_the_marker(listener, on_hers=None):
    """A judge that reads the prompt the real judge builds, so it sees what
    the model sees: a current segment shown "(during TTS)" gets *on_hers*,
    anything else in the hot window is his. By default *on_hers* is what
    the prompt asks for, her echo, never extracted. Field trace: the echo
    verdict came at a confidence below "high"."""
    if on_hers is None:
        on_hers = _make_judgment(
            directed=False, confidence="medium",
            reasoning="the current segment is marked (during TTS), an echo "
                      "of the assistant's own speech, never extracted")

    def judge(**kwargs):
        current = _current_line(IntentJudge()._build_user_prompt(**kwargs))
        assert current, kwargs
        if "during TTS" in current:
            return on_hers
        return _make_judgment(directed=True, query=kwargs["current_text"],
                              reasoning="his follow-up in the hot window")

    mock_judge = MagicMock()
    mock_judge.available = True
    mock_judge.last_failure_reason = ""
    mock_judge.judge.side_effect = judge
    listener._intent_judge = mock_judge
    return mock_judge


def _current_line_shown_to(judge) -> str:
    """The line of the last prompt that holds the utterance being judged."""
    return _current_line(_prompt_shown_to(judge))


def _console_lines(mock_print) -> list:
    return [" ".join(str(arg) for arg in call.args) for call in mock_print.call_args_list]


def _console_says_what_became_of(mock_print, heard: str) -> bool:
    """Whether a console line, other than the transcription's own "📝 Heard",
    quotes *heard*, as far as the console quotes it: what she did with it,
    not only that it was heard."""
    quoted = heard.lower()[:50]
    return any(quoted in line.lower() and "📝" not in line
               for line in _console_lines(mock_print))


def _console_says_heard_as_her_echo(mock_print, heard: str) -> bool:
    """Whether the console says *heard* was set aside as her own echo."""
    quoted = heard.lower()[:50]
    return any("🔇 Heard (echo)" in line and quoted in line.lower()
               for line in _console_lines(mock_print))


@pytest.mark.unit
class TestHisAnswerTheMomentHerReplyEnds:
    """She ends her reply on a question and he answers at once, without the
    wake word, as people do.

    Field trace (2026-11-05): his question began within the echo tolerance
    of her last word, counting the voice detector's pre-roll, so it was shown
    to the judge "(during TTS)". The judge's prompt says such a segment is
    her echo and never extracted; it said so, he was not answered, and
    nothing on the console said why. The early text check had cleared it:
    hardly a word of it was hers.
    """

    def _her_reply_ends(self, reply=_HER_REPLY):
        # The window outlasts the echo tolerance, as every configuration's
        # does, kept short so that waiting for it to close stays quick.
        listener, _ = _create_listener(echo_tolerance=0.1, hot_window_seconds=0.2)
        last_word = _she_says(listener, reply)
        assert _wait_for_hot_window_active(listener)
        return listener, last_word

    @patch("builtins.print")
    def test_his_answer_is_answered(self, _print):
        listener, last_word = self._her_reply_ends()
        _judge_that_reads_the_marker(listener)
        # As in the field, his words reach the judge after the window closed.
        assert _wait_for_hot_window_expiry(listener)

        _whisper_hears(listener, _HIS_QUESTION, started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        assert "lyon" in _accepted_query(listener).lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_own_words_coming_back_then_are_not(self, _print):
        """Her echo, decimated by Whisper so that the text check misses it.
        It began while she spoke and closed as she stopped."""
        listener, last_word = self._her_reply_ends()
        _judge_that_reads_the_marker(listener)
        assert (fuzz.partial_ratio(_HER_WORDS_DECIMATED, _HER_REPLY.lower())
                < EchoDetector.PURE_ECHO_THRESHOLD)

        _whisper_hears(listener, _HER_WORDS_DECIMATED, started=last_word - 1.5,
                       level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_phrase_made_of_the_quiet_then_is_not_answered(self, mock_print):
        """Room noise just after her reply, which Whisper turns into a short
        thank-you: words she did not say, and nobody's voice. It is not
        answered, and the console says so."""
        listener, last_word = self._her_reply_ends()
        _judge_that_reads_the_marker(listener)

        _whisper_hears(listener, "Merci.", started=_just_after(listener, last_word),
                       level=_level(listener, _ROOM_NOISE))

        assert _accepted_query(listener) == ""
        assert _console_says_what_became_of(mock_print, "merci.")
        listener.state_manager.stop()

    @pytest.mark.parametrize("heard, starts, level", [
        # Her own words coming back as she stops, decimated by Whisper.
        (_HER_WORDS_DECIMATED, "while she spoke", _HER_VOICE),
        # What Whisper makes of the quiet right after her.
        ("Merci.", "just after", _ROOM_NOISE),
    ], ids=["her-words", "made-of-the-quiet"])
    @pytest.mark.parametrize("verdict", [
        _make_judgment(directed=False, confidence=confidence,
                       reasoning="the assistant's own speech, marked during TTS")
        for confidence in ("high", "medium", "low")
    ] + [
        _make_judgment(directed=True, query="", confidence=confidence,
                       reasoning="hot window, nothing to extract")
        for confidence in ("high", "medium", "low")
    ], ids=[f"{kind}-{confidence}" for kind in ("rejected", "directed-without-query")
            for confidence in ("high", "medium", "low")])
    @patch("builtins.print")
    def test_what_could_be_hers_is_not_answered_whatever_the_judge_says(
            self, mock_print, verdict, heard, starts, level):
        """The judge is shown these words "(during TTS)" and leaves them out
        in its own words: a rejection that never says "echo", or directed
        with nothing to extract. The listener knows they could be hers, and
        no override takes them."""
        listener, last_word = self._her_reply_ends()
        _judge_that_reads_the_marker(listener, on_hers=verdict)
        started = last_word - 1.5 if starts == "while she spoke" else _just_after(listener, last_word)

        _whisper_hears(listener, heard, started=started, level=_level(listener, level))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("reply, heard", [
        (_HER_REPLY_NAMING_HER, _HER_NAME_DECIMATED),
        (_HER_REPLY_WITH_A_WORD_LIKE_HER_NAME, _A_WORD_LIKE_HER_NAME_DECIMATED),
    ], ids=["her-name", "a-word-like-her-name"])
    @pytest.mark.parametrize("verdict", [
        _make_judgment(directed=False, confidence=confidence, reasoning=reasoning)
        for reasoning in ("an echo of the assistant's own speech",
                          "the assistant's own speech, marked during TTS")
        for confidence in ("high", "low")
    ] + [
        _make_judgment(directed=True, query="", confidence=confidence,
                       reasoning="hot window, nothing to extract")
        for confidence in ("high", "low")
    ], ids=[f"{kind}-{confidence}" for kind in ("echo", "rejected", "directed-without-query")
            for confidence in ("high", "low")])
    @patch("builtins.print")
    def test_her_own_words_carrying_the_wake_word_are_not_answered(
            self, mock_print, verdict, reply, heard):
        """Her reply named her, or said a word the wake-word check takes for
        her name, and comes back decimated as she stops. The wake word in it
        is hers, so it proves nothing: the words are set aside as hers,
        whatever the judge's words for leaving them out."""
        listener, last_word = self._her_reply_ends(reply)
        _judge_that_reads_the_marker(listener, on_hers=verdict)
        cfg = listener.cfg
        assert is_wake_word_detected(heard, cfg.wake_word, cfg.wake_aliases, cfg.wake_fuzzy_ratio)
        assert fuzz.partial_ratio(heard, reply.lower()) < EchoDetector.PURE_ECHO_THRESHOLD

        _whisper_hears(listener, heard, started=last_word - 1.5, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("verdict", [
        None,   # the judge's own reading of the mark: her echo
        _make_judgment(directed=False, confidence="medium",
                       reasoning="the assistant's own speech, marked during TTS"),
        _make_judgment(directed=True, query="", confidence="low",
                       reasoning="hot window, nothing to extract"),
    ], ids=["echo", "rejected", "directed-without-query"])
    @patch("builtins.print")
    def test_his_quiet_answer_with_the_wake_word_is_answered(self, _print, verdict):
        """He answers at once but quietly, below the voice bar, so the words
        could be hers and the judge is shown them as hers. He said the wake
        word, though, a word she did not say, and the wake word decides, as
        outside the window: what he said is answered, the wake word taken
        out."""
        listener, last_word = self._her_reply_ends()
        _judge_that_reads_the_marker(listener, on_hers=verdict)

        _whisper_hears(listener, f"Jarvis, {_HIS_QUESTION}",
                       started=_just_after(listener, last_word), level=_level(listener, 0.8))

        answered = _accepted_query(listener)
        assert "lyon" in answered and "jarvis" not in answered
        listener.state_manager.stop()

    @pytest.mark.parametrize("reply, heard, starts, level", [
        (_HER_REPLY, _HER_WORDS_DECIMATED, "while she spoke", _HER_VOICE),
        (_HER_REPLY, "Merci.", "just after", _ROOM_NOISE),
        (_HER_REPLY_NAMING_HER, _HER_NAME_DECIMATED, "while she spoke", _HER_VOICE),
    ], ids=["her-words", "made-of-the-quiet", "her-name"])
    @patch("builtins.print")
    def test_with_no_verdict_what_could_be_hers_is_not_answered(
            self, mock_print, reply, heard, starts, level):
        """The judge gives no verdict: a timeout, or its back-off after a
        connection error. The listener keeps his words without it, not words
        that could be hers, her name among them, and the console says so."""
        listener, last_word = self._her_reply_ends(reply)
        _install_intent_judge(listener, None).last_failure_reason = "timeout"
        started = last_word - 1.5 if starts == "while she spoke" else _just_after(listener, last_word)

        _whisper_hears(listener, heard, started=started, level=_level(listener, level))

        assert _the_judge_gave_no_verdict(mock_print)
        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_her_words_are_read_against_the_reply_they_came_from(
            self, mock_print):
        """A reply waiting its turn starts while the judge is thinking, and
        the judge then gives no verdict. Her decimated words, her name among
        them, are still read against the reply they came from."""
        listener, last_word = self._her_reply_ends(_HER_REPLY_NAMING_HER)
        mock_judge = _install_intent_judge(listener, None)
        mock_judge.last_failure_reason = "timeout"

        def judge(**_kwargs):
            listener.echo_detector.track_tts_start(
                "La réunion de demain est déplacée à neuf heures.", baseline_energy=_FLOOR)
            return None

        mock_judge.judge.side_effect = judge

        _whisper_hears(listener, _HER_NAME_DECIMATED, started=last_word - 1.5,
                       level=_level(listener, _HER_VOICE))

        assert _the_judge_gave_no_verdict(mock_print)
        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, _HER_NAME_DECIMATED)
        listener.state_manager.stop()

    @pytest.mark.parametrize("verdict", [
        _make_judgment(directed=False, confidence="medium",
                       reasoning="an echo of the assistant's own speech"),
        _make_judgment(directed=False, confidence="medium",
                       reasoning="the assistant's own speech, marked during TTS"),
        _make_judgment(directed=True, query="", confidence="low",
                       reasoning="hot window, nothing to extract"),
    ], ids=["echo", "rejected", "directed-without-query"])
    @patch("builtins.print")
    def test_her_name_is_read_against_the_reply_the_judge_was_shown(self, mock_print, verdict):
        """A reply waiting its turn starts while the judge is thinking, and
        replaces the reply her echo would be compared with. Her decimated
        words, her name among them, are read against the reply the judge was
        shown, which named her: the wake word in them is hers, and they are
        set aside as hers."""
        listener, last_word = self._her_reply_ends(_HER_REPLY_NAMING_HER)
        mock_judge = _install_intent_judge(listener, verdict)

        def judge(**_kwargs):
            listener.echo_detector.track_tts_start(
                "La réunion de demain est déplacée à neuf heures.", baseline_energy=_FLOOR)
            return verdict

        mock_judge.judge.side_effect = judge

        _whisper_hears(listener, _HER_NAME_DECIMATED, started=last_word - 1.5,
                       level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, _HER_NAME_DECIMATED)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_query_extracted_from_words_shown_as_hers_is_answered(self, _print):
        """His quiet answer just after her last word could be hers, and the
        judge is shown it "(during TTS)". A judge that extracts a query from
        it anyway is answered, as any directed verdict with a query is: the
        mark informs the judge, it does not overrule it."""
        listener, last_word = self._her_reply_ends()
        extracted = "quel temps fera-t-il demain matin"
        _judge_that_reads_the_marker(listener, on_hers=_make_judgment(
            directed=True, query=extracted, confidence="medium",
            reasoning="his follow-up, whatever the mark says"))

        _whisper_hears(listener, _HIS_QUESTION, started=_just_after(listener, last_word),
                       level=_level(listener, 0.8))

        assert _accepted_query(listener) == extracted
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_his_answer_is_answered(self, _print):
        listener, last_word = self._her_reply_ends()
        _install_intent_judge(listener, None).last_failure_reason = "timeout"

        _whisper_hears(listener, _HIS_QUESTION, started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        assert "lyon" in _accepted_query(listener).lower()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_his_quiet_answer_naming_her_is_answered(self, _print):
        """With no verdict, his quiet answer could be hers, but the wake word
        in it can only be his: it hands his words to the wake-word check, as
        it does behind a verdict, which answers them with the wake word
        taken out."""
        listener, last_word = self._her_reply_ends()
        _install_intent_judge(listener, None).last_failure_reason = "timeout"

        _whisper_hears(listener, f"Jarvis, {_HIS_QUESTION}",
                       started=_just_after(listener, last_word), level=_level(listener, 0.8))

        answered = _accepted_query(listener)
        assert "lyon" in answered and "jarvis" not in answered
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_his_words_holding_an_alias_are_answered_as_said(self, _print):
        """With no verdict, his words heard in a voice are answered as he
        said them, a word of his holding one of the wake word's aliases
        ("charismatique" holds "charis") included."""
        listener, last_word = self._her_reply_ends()
        _with_the_default_aliases(listener)
        _install_intent_judge(listener, None).last_failure_reason = "timeout"
        heard = "Est-elle charismatique ?"
        cfg = listener.cfg
        assert is_wake_word_detected(heard.lower(), cfg.wake_word, cfg.wake_aliases,
                                     cfg.wake_fuzzy_ratio)

        _whisper_hears(listener, heard, started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        assert _accepted_query(listener) == heard.lower()
        listener.state_manager.stop()

    @pytest.mark.parametrize("verdict", [
        _make_judgment(directed=False, confidence="medium", reasoning="background conversation"),
        _make_judgment(directed=False, confidence="high",
                       reasoning="an echo of the assistant's own speech"),
        _make_judgment(directed=True, query="", confidence="low",
                       reasoning="hot window, nothing to extract"),
        None,
    ], ids=["rejected", "echo", "directed-without-query", "no-verdict"])
    @patch("builtins.print")
    def test_his_answer_holding_a_word_like_her_name_is_answered(self, mock_print, verdict):
        """"Changé d'avis" holds "avis", a word the wake-word check takes for
        her name on its own, not inside "d'avis" as it reads the utterance.
        His answer is answered, and the console says so."""
        listener, last_word = self._her_reply_ends()
        _install_intent_judge(listener, verdict).last_failure_reason = "timeout"
        heard = "Non, j'ai changé d'avis."

        _whisper_hears(listener, heard, started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        assert "changé" in _accepted_query(listener)
        assert _console_says_what_became_of(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("spelling", ["chavis", "joris"])
    @pytest.mark.parametrize("judged", [True, False], ids=["echo-verdict", "no-verdict"])
    @patch("builtins.print")
    def test_her_name_coming_back_in_another_spelling_is_not_answered(
            self, mock_print, judged, spelling):
        """Her reply named her, and her name comes back in one of the
        spellings Whisper gives it, the ones the wake word's aliases list.
        After a reply holding the wake word, no wake word heard can be told
        from hers: the words are set aside as hers."""
        listener, last_word = self._her_reply_ends(_HER_REPLY_NAMING_HER)
        _with_the_default_aliases(listener)
        if judged:
            _judge_that_reads_the_marker(listener)
        else:
            _install_intent_judge(listener, None).last_failure_reason = "timeout"
        heard = _HER_NAME_DECIMATED.replace("jarvis", spelling)
        cfg = listener.cfg
        assert is_wake_word_detected(heard, cfg.wake_word, cfg.wake_aliases, cfg.wake_fuzzy_ratio)
        assert (fuzz.partial_ratio(heard, _HER_REPLY_NAMING_HER.lower())
                < EchoDetector.PURE_ECHO_THRESHOLD)

        _whisper_hears(listener, heard, started=last_word - 1.5, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("judged", [True, False], ids=["echo-verdict", "no-verdict"])
    @patch("builtins.print")
    def test_a_word_of_hers_bent_into_her_name_is_not_answered(self, mock_print, judged):
        """Her reply holds no word the wake-word check takes for her name,
        but coming back Whisper bends one of hers into one. It is still her
        word: the words are set aside as hers."""
        reply, heard = _HER_REPLY_WITH_A_WORD_WHISPER_BENDS, _A_WORD_BENT_INTO_HER_NAME
        listener, last_word = self._her_reply_ends(reply)
        _with_the_default_aliases(listener)
        if judged:
            _judge_that_reads_the_marker(listener)
        else:
            _install_intent_judge(listener, None).last_failure_reason = "timeout"
        cfg = listener.cfg
        assert is_wake_word_detected(heard, cfg.wake_word, cfg.wake_aliases, cfg.wake_fuzzy_ratio)
        assert not is_wake_word_detected(reply.lower(), cfg.wake_word, cfg.wake_aliases,
                                         cfg.wake_fuzzy_ratio)
        assert fuzz.partial_ratio(heard, reply.lower()) < EchoDetector.PURE_ECHO_THRESHOLD

        _whisper_hears(listener, heard, started=last_word - 1.5, level=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("spelling", ["Jarvis", "Jervis"])
    @pytest.mark.parametrize("judged", [True, False], ids=["echo-verdict", "no-verdict"])
    @patch("builtins.print")
    def test_after_a_reply_holding_a_word_like_her_name_his_quiet_wake_word_is_weighed_as_hers(
            self, mock_print, judged, spelling):
        """Her reply said "avis", which the wake-word check takes for her
        name. Coming back, that word could take any spelling Whisper gives
        her name, so after such a reply no wake word heard can be told from
        hers, in whichever spelling it is heard. His quiet answer naming her
        is weighed as her words: set aside, and the console says so."""
        listener, last_word = self._her_reply_ends(_HER_REPLY_WITH_A_WORD_LIKE_HER_NAME)
        if judged:
            _judge_that_reads_the_marker(listener)
        else:
            _install_intent_judge(listener, None).last_failure_reason = "timeout"
        heard = f"{spelling}, {_HIS_QUESTION}"
        cfg = listener.cfg
        assert is_wake_word_detected(heard.lower(), cfg.wake_word, cfg.wake_aliases,
                                     cfg.wake_fuzzy_ratio)

        _whisper_hears(listener, heard, started=_just_after(listener, last_word),
                       level=_level(listener, 0.8))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("reply", [
        _HER_REPLY_NAMING_HER, _HER_REPLY_WITH_A_WORD_LIKE_HER_NAME,
    ], ids=["her-name", "a-word-like-her-name"])
    @pytest.mark.parametrize("verdict", [
        _make_judgment(directed=False, confidence="medium", reasoning="background conversation"),
        _make_judgment(directed=False, confidence="high",
                       reasoning="an echo of the assistant's own speech"),
        _make_judgment(directed=True, query="", confidence="low",
                       reasoning="hot window, nothing to extract"),
        None,
    ], ids=["rejected", "echo", "directed-without-query", "no-verdict"])
    @patch("builtins.print")
    def test_after_a_reply_holding_her_name_his_answer_in_his_voice_is_answered(
            self, _print, verdict, reply):
        """After a reply holding her name, or a word the wake-word check
        takes for it, the wake word he says decides nothing. His answer
        naming her, heard in his voice, is still his: it is answered as any
        answer of his in the hot window is."""
        listener, last_word = self._her_reply_ends(reply)
        _install_intent_judge(listener, verdict).last_failure_reason = "timeout"

        _whisper_hears(listener, f"Jarvis, {_HIS_QUESTION}",
                       started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        assert "lyon" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_after_her_time_his_quiet_words_are_his(self, _print):
        """Once the echo tolerance has passed, her voice is no longer coming
        back: quiet words heard then are his, and a rejection in other words
        than echo does not set them aside."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        last_word = _she_says(listener, _HER_REPLY)
        assert _wait_for_hot_window_active(listener)
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", reasoning="background conversation"))
        started = last_word + 10 * listener.echo_detector.echo_tolerance
        while time.time() < started + 0.05:
            time.sleep(0.01)

        _whisper_hears(listener, "Oui, vas-y.", started=started, level=_level(listener, 0.8))

        assert "vas" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_his_words_after_her_tail_are_read_without_it(self, _print):
        """Whisper joins the tail of her reply, heard as she stops, to his
        question. Once the tail is stripped, what is left is read on its
        own: his words, which a rejection does not set aside."""
        listener, last_word = self._her_reply_ends()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", reasoning="background conversation"))
        heard = "je reste à votre écoute que souhaitez-vous savoir ce soir ? la météo à lyon"

        _whisper_hears(listener, heard, started=last_word - 1.0, level=_level(listener, _HIS_VOICE))

        answered = _accepted_query(listener)
        assert "météo" in answered and "écoute" not in answered
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_judge_is_shown_his_answer_as_his(self, _print):
        listener, last_word = self._her_reply_ends()
        judge = _judge_that_reads_the_marker(listener)

        _whisper_hears(listener, _HIS_QUESTION, started=_just_after(listener, last_word),
                       level=_level(listener, _HIS_VOICE))

        current = _current_line_shown_to(judge)
        assert "lyon" in current.lower() and "during TTS" not in current
        listener.state_manager.stop()

    @pytest.mark.parametrize("heard, starts, level", [
        (_HER_WORDS_DECIMATED, "while she spoke", _HER_VOICE),
        ("Merci.", "just after", _ROOM_NOISE),
    ], ids=["her-words", "made-of-the-quiet"])
    @patch("builtins.print")
    def test_the_judge_is_still_shown_what_could_be_hers_as_hers(
            self, _print, heard, starts, level):
        listener, last_word = self._her_reply_ends()
        judge = _judge_that_reads_the_marker(listener)
        started = last_word - 1.5 if starts == "while she spoke" else _just_after(listener, last_word)

        _whisper_hears(listener, heard, started=started, level=_level(listener, level))

        assert "during TTS" in _current_line_shown_to(judge)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_his_words_over_hers_are_shown_as_his(self, _print):
        """He talks while she is still speaking: her voice clears the bar
        too, but the words are not hers."""
        listener, _ = _create_listener(tts_speaking=True)
        listener.echo_detector.track_tts_start(_HER_REPLY, baseline_energy=_FLOOR)
        judge = _install_intent_judge(listener, _make_judgment(
            directed=False, reasoning="no wake word"))

        _whisper_hears(listener, "Non, je parlais de Lyon, pas de Paris.",
                       started=time.time() - 1.0, level=_level(listener, _HIS_VOICE))

        current = _current_line_shown_to(judge)
        assert "lyon" in current.lower() and "during TTS" not in current
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_own_words_heard_while_she_still_speaks_stay_hers(self, _print):
        """Her voice clears the bar while she speaks, so only her words keep
        the mark there. Asked about later, he is shown them as hers."""
        listener, tts = _create_listener(
            tts_speaking=True, echo_tolerance=0.1, hot_window_seconds=0.2)
        listener.echo_detector.track_tts_start(_HER_REPLY, baseline_energy=_FLOOR)
        judge = _judge_that_reads_the_marker(listener)
        _whisper_hears(listener, _HER_WORDS_DECIMATED, started=time.time() - 1.0,
                       level=_level(listener, _HER_VOICE))

        tts.is_speaking.return_value = False
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)
        _whisper_hears(listener, _HIS_QUESTION, started=time.time(),
                       level=_level(listener, _HIS_VOICE))

        her_line = next(line for line in _prompt_shown_to(judge).splitlines()
                        if "souhaitez vous soir" in line)
        assert "during TTS" in her_line
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: A hesitant verdict in the hot window
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAHesitantVerdictInTheHotWindow:
    """In the hot window a rejection the judge is unsure of is no stronger
    than one it is sure of: it goes through the same overrides, so his words
    are kept, hers are set aside, and either way the console says which.
    The judge's confidence is "low" whenever its answer leaves it out."""

    def _in_the_hot_window(self):
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        _she_says(listener, _HER_REPLY)
        assert _wait_for_hot_window_active(listener)
        return listener

    @pytest.mark.parametrize("confidence", ["medium", "low"])
    @patch("builtins.print")
    def test_a_hesitant_rejection_of_his_words_is_overridden(self, _print, confidence):
        listener = self._in_the_hot_window()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence=confidence, reasoning="background conversation"))

        listener._process_transcript(_HIS_QUESTION,
                                     utterance_energy=_level(listener, _HIS_VOICE))

        assert _accepted_query(listener) == _HIS_QUESTION
        listener.state_manager.stop()

    @pytest.mark.parametrize("confidence", ["medium", "low"])
    @patch("builtins.print")
    def test_a_hesitant_echo_verdict_on_his_words_is_overridden(self, _print, confidence):
        listener = self._in_the_hot_window()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence=confidence, reasoning="echo of the assistant"))

        listener._process_transcript(_HIS_QUESTION,
                                     utterance_energy=_level(listener, _HIS_VOICE))

        assert _accepted_query(listener) == _HIS_QUESTION
        listener.state_manager.stop()

    @pytest.mark.parametrize("confidence", ["medium", "low"])
    @patch("builtins.print")
    def test_a_hesitant_echo_verdict_on_her_words_is_said(self, mock_print, confidence):
        listener = self._in_the_hot_window()
        assert (fuzz.partial_ratio(_HER_WORDS_DECIMATED, _HER_REPLY.lower())
                < EchoDetector.PURE_ECHO_THRESHOLD)
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence=confidence, reasoning="echo of the assistant"))

        listener._process_transcript(_HER_WORDS_DECIMATED,
                                     utterance_energy=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_what_became_of(mock_print, _HER_WORDS_DECIMATED)
        listener.state_manager.stop()

    @pytest.mark.parametrize("confidence", ["medium", "low"])
    @patch("builtins.print")
    def test_a_hesitant_directed_verdict_without_a_query_takes_his_words(
            self, _print, confidence):
        listener = self._in_the_hot_window()
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="", confidence=confidence, reasoning="follow-up"))

        listener._process_transcript(_HIS_QUESTION,
                                     utterance_energy=_level(listener, _HIS_VOICE))

        assert _accepted_query(listener) == _HIS_QUESTION
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_her_own_words_the_early_check_did_not_see_are_said(self, mock_print):
        """She is thinking about his next question while the window is
        still open, so the early echo check stands aside, and the judge
        rejects her own words without calling them echo."""
        listener = self._in_the_hot_window()
        listener._tune_player = MagicMock()
        listener._tune_player.is_playing.return_value = True
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", reasoning="background conversation"))
        heard = "bien reçu, monsieur. je reste à votre écoute"

        listener._process_transcript(heard, utterance_energy=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_what_became_of(mock_print, heard)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_her_reply_coming_back_whole_is_said(self, mock_print):
        """She is thinking about his next question while the window is still
        open, so the early echo check stands aside, and the judge gives no
        verdict. Her reply, which named her, comes back whole: it is set
        aside as hers, and the console says so."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        _she_says(listener, _HER_REPLY_NAMING_HER)
        assert _wait_for_hot_window_active(listener)
        listener._tune_player = MagicMock()
        listener._tune_player.is_playing.return_value = True
        _install_intent_judge(listener, None).last_failure_reason = "timeout"
        heard = "bien reçu, monsieur, ici jarvis. je reste à votre écoute"

        listener._process_transcript(heard, utterance_energy=_level(listener, _HER_VOICE))

        assert _the_judge_gave_no_verdict(mock_print)
        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, heard)
        listener.state_manager.stop()

    @pytest.mark.parametrize("directed", [False, True], ids=["rejected", "directed-without-query"])
    @pytest.mark.parametrize("confidence", ["high", "medium", "low"])
    @patch("builtins.print")
    def test_a_wake_word_with_nothing_extracted_is_left_to_the_wake_word(
            self, _print, confidence, directed):
        """He says the wake word in the hot window and the judge extracts
        nothing: the wake-word check decides, as outside the window, and
        answers what he said, the wake word taken out."""
        listener = self._in_the_hot_window()
        _install_intent_judge(listener, _make_judgment(
            directed=directed, query="", confidence=confidence, reasoning="unclear"))

        listener._process_transcript(f"jarvis {_HIS_QUESTION}",
                                     utterance_energy=_level(listener, _HIS_VOICE))

        answered = _accepted_query(listener)
        assert "lyon" in answered and "jarvis" not in answered
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_an_echo_verdict_is_read_against_the_reply_the_judge_was_shown(self, mock_print):
        """A reply waiting its turn starts while the judge is thinking, and
        replaces the reply her echo would be compared with. The verdict is
        about the reply the judge was shown, so her decimated words are
        still read against that one."""
        listener = self._in_the_hot_window()
        verdict = _make_judgment(directed=False, confidence="high",
                                 reasoning="echo of the assistant")

        def judge(**_kwargs):
            listener.echo_detector.track_tts_start(
                "La réunion de demain est déplacée à neuf heures.", baseline_energy=_FLOOR)
            return verdict

        mock_judge = _install_intent_judge(listener, verdict)
        mock_judge.judge.side_effect = judge

        listener._process_transcript(_HER_WORDS_DECIMATED,
                                     utterance_energy=_level(listener, _HER_VOICE))

        assert _accepted_query(listener) == ""
        assert _console_says_heard_as_her_echo(mock_print, _HER_WORDS_DECIMATED)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_outside_it_a_hesitant_rejection_still_leaves_it_to_the_wake_word(self, _print):
        """Outside the hot window nothing changes: an unsure rejection falls
        to the wake-word check, which a wake word passes."""
        listener, _ = _create_listener()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", reasoning="unclear"))

        listener._process_transcript("jarvis quelle heure est-il à lyon",
                                     utterance_energy=_level(listener, _HIS_VOICE))

        assert "heure" in _accepted_query(listener)
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: A sentence heard while his question is collected completes it
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestASecondSentenceCompletesTheQuestion:
    """A sentence that reaches her while his question is still being
    collected completes it, and she answers both.

    Field trace (2026-10-30): "Salut Yuba, comment tu vas ?" then "Tu
    m'entends Yuba ?" inside the collection window. The second replaced the
    first, and only "tu m'entends ?" was answered.
    """

    @patch("builtins.print")
    def test_a_second_wake_word_sentence_adds_to_the_first(self, _print):
        listener, _ = _create_listener()

        listener._process_transcript("jarvis comment tu vas", utterance_energy=0.01)
        listener._process_transcript("jarvis tu m'entends", utterance_energy=0.01)

        query = _accepted_query(listener)
        assert "comment tu vas" in query and "tu m'entends" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_so_does_one_the_judge_accepts(self, _print):
        listener, _ = _create_listener()
        _install_intent_judge(listener, _make_judgment(directed=True, query="comment tu vas"))
        listener._process_transcript("jarvis comment tu vas", utterance_energy=0.01)

        _install_intent_judge(listener, _make_judgment(directed=True, query="tu m'entends"))
        listener._process_transcript("jarvis tu m'entends", utterance_energy=0.01)

        query = _accepted_query(listener)
        assert "comment tu vas" in query and "tu m'entends" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_stop_cancels_the_question_instead_of_joining_it(self, _print):
        listener, _ = _create_listener()
        listener._process_transcript("jarvis comment tu vas", utterance_energy=0.01)

        listener._process_transcript("jarvis stop", utterance_energy=0.01)

        assert not listener.state_manager.is_collecting()
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_what_joins_the_question_is_no_context_for_the_judge(self, _print):
        """Left visible, the judge could restate it in a later query and it
        would be collected twice."""
        listener, _ = _create_listener()

        def hear(text):
            now = time.time()
            listener._transcript_buffer.add(text, now - 1.0, now, 0.01)
            listener._process_transcript(text, utterance_energy=0.01)

        hear("jarvis")
        hear("quel temps fait-il demain")

        shown = IntentJudge()._build_user_prompt(
            listener._transcript_buffer.get_last_seconds(120.0), wake_timestamp=None,
            last_tts_text="", last_tts_finish_time=0.0, in_hot_window=False,
            current_text="jarvis réponds")
        assert "quel temps" not in shown
        assert "quel temps fait-il demain" in _accepted_query(listener)
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: What Whisper makes of silence is nobody's speech
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWhatWhisperMakesOfSilence:
    """Whisper turns silence and room noise into the closing lines of videos:
    "Thank you.", "ご視聴ありがとうございました", "Продолжение следует...".
    The judge says so with ``noise``, and no hot-window override may answer it.

    Field trace (2026-10-29): the judge called each of these a hallucination,
    the overrides answered them anyway, and she replied in English, then
    about Russian and Japanese, to a man speaking French.
    """

    def _in_the_window(self, reply):
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.echo_detector.track_tts_start(reply)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)
        return listener

    @patch("builtins.print")
    def test_a_phrase_nobody_said_is_not_answered(self, _print):
        listener = self._in_the_window("Pour demain, le ciel sera dégagé, Monsieur.")
        listener.cfg.tune_enabled = True
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle line produced from silence"))

        listener._process_transcript("продолжение следует...", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        assert not _is_beeping(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_nor_when_the_judge_also_speaks_of_her_echo(self, _print):
        """The echo override: "thank you." carries words she did not say, and
        its energy clears the bar over the ambient floor recorded when she
        started speaking, so only the noise verdict stands between it and a
        reply."""
        listener = self._in_the_window("Je reste à votre disposition, Monsieur.")
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="hallucination on silence; the other segment is echo"))

        listener._process_transcript("thank you.", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_same_words_said_by_him_are_still_answered(self, _print):
        """The same utterance and the same echo reasoning, judged speech: the
        barge-in override still takes it."""
        listener = self._in_the_window("Je reste à votre disposition, Monsieur.")
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=False,
            reasoning="his words; the other segment is echo"))

        listener._process_transcript("thank you.", utterance_energy=0.01)

        assert _accepted_query(listener) == "thank you."
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_judge_that_also_says_directed_is_believed(self, _print):
        """A verdict that contradicts itself falls to his side."""
        listener = self._in_the_window("Pour demain, le ciel sera dégagé, Monsieur.")
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="merci", confidence="high", noise=True))

        listener._process_transcript("merci", utterance_energy=0.01)

        assert _accepted_query(listener) == "merci"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_phantom_does_not_cost_him_his_follow_up(self, _print):
        listener = self._in_the_window("Pour demain, le ciel sera dégagé, Monsieur.")
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle line produced from silence"))
        listener._process_transcript("ご視聴ありがとうございました", utterance_energy=0.01)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="et après-demain ?", confidence="high"))
        listener._process_transcript("et après-demain ?", utterance_energy=0.01)

        assert _accepted_query(listener) == "et après-demain ?"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_phantom_is_no_context_for_what_he_says_next(self, _print):
        listener = self._in_the_window("Pour demain, le ciel sera dégagé, Monsieur.")

        def hear(text):
            # What the audio loop does before handing the text over.
            now = time.time()
            listener._transcript_buffer.add(text, now - 1.0, now, 0.01)
            listener._process_transcript(text, utterance_energy=0.01)

        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle line produced from silence"))
        hear("ご視聴ありがとうございました")

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="et après-demain ?", confidence="high"))
        hear("et après-demain ?")

        shown = _prompt_shown_to(judge)
        assert "ご視聴" not in shown
        assert "et après-demain" in shown
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_wake_word_outside_the_window_is_still_his(self, _print):
        """Outside the hot window the judge only runs on a wake word or while
        she speaks. A doubtful verdict there still falls through to the wake
        word, whatever it says about noise."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="medium", noise=True,
            reasoning="unsure"))

        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert "what time is it" in _accepted_query(listener)
        listener.state_manager.stop()


@pytest.mark.unit
class TestWhatWhisperMakesOfSilenceWhileHeAsks:
    """While his question is still being collected, what he adds needs no
    wake word, so Whisper's inventions reach the question too. The judge
    says which are not speech, and those are left out.

    Field trace (2026-11-02): "Hey Yuba, comment tu vas ? Ça va aujourd'hui ?"
    then "Sous-titrage ST' 501" made out of silence inside the collection
    window. It joined the question, and she asked him what "sous-titrage
    st' 501" was.
    """

    def _asking(self, **kwargs):
        listener, _ = _create_listener(**kwargs)
        listener.cfg.tune_enabled = True
        listener._process_transcript("jarvis comment tu vas", utterance_energy=0.01)
        assert listener.state_manager.is_collecting()
        return listener

    def _hear(self, listener, text, ended=None):
        # What the audio loop does before handing the text over.
        ended = time.time() if ended is None else ended
        listener._transcript_buffer.add(text, ended - 1.0, ended, 0.01)
        listener._process_transcript(text, 0.01, ended - 1.0, ended)

    def _handed_over(self, listener):
        """The turn his question became once its silence ran out."""
        listener.state_manager.voice_collect_seconds = 0.0
        listener._check_query_timeout()
        return listener._turns.get_nowait()

    @patch("builtins.print")
    def test_a_phrase_nobody_said_does_not_join_his_question(self, _print):
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle credit produced from silence"))

        self._hear(listener, "sous-titrage st' 501")

        assert listener.state_manager.is_collecting()
        assert "sous-titrage" not in _accepted_query(listener)
        assert "comment tu vas" in _accepted_query(listener)
        assert _is_beeping(listener), "the tune of his question stopped"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_his_second_sentence_still_joins_it(self, _print):
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="ça va aujourd'hui ?", confidence="high"))

        self._hear(listener, "ça va aujourd'hui ?")

        query = _accepted_query(listener)
        assert "comment tu vas" in query and "ça va aujourd'hui" in query
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_so_does_one_the_judge_calls_undirected_speech(self, _print):
        """Only the noise verdict counts here: he needs no wake word to
        finish his own question."""
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=False, reasoning="no wake word"))

        self._hear(listener, "et demain aussi")

        assert "et demain aussi" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_so_does_one_the_judge_also_calls_directed(self, _print):
        """His side wins a contradiction."""
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="et demain aussi", confidence="high", noise=True))

        self._hear(listener, "et demain aussi")

        assert "et demain aussi" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_with_no_verdict_it_joins(self, _print):
        listener = self._asking()
        _install_intent_judge(listener, None)

        self._hear(listener, "et demain aussi")

        assert "et demain aussi" in _accepted_query(listener)
        assert any("unavailable" in str(c.args[0]) for c in _print.call_args_list if c.args), \
            "the judge's silence was not shown"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_stop_the_judge_reads_cancels_his_question(self, _print):
        """A stop the judge reads out of a sentence counts as a stop word
        does: the question being collected is dropped, not completed."""
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=False, stop=True, confidence="high",
            reasoning="he asks her to drop it"))

        self._hear(listener, "non laisse tomber c'est pas grave")

        assert not listener.state_manager.is_collecting()
        assert listener._turns.empty()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_his_words_not_the_phantom_travel_with_his_question(self, _print):
        """His exact words and when he stopped speaking go with the query: a
        yes answering her spoken question is read from them."""
        listener = self._asking()
        said_at = time.time() - 2.0
        self._hear(listener, "oui envoie-le", ended=said_at)
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle credit produced from silence"))

        self._hear(listener, "sous-titrage st' 501")

        turn = self._handed_over(listener)
        assert turn.heard == "oui envoie-le"
        assert turn.heard_at == said_at
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judging_what_he_adds_does_not_hold_his_question_back(self, _print):
        """The collection's silence is counted from the end of his words:
        the judge's time is part of it, not added to it."""
        listener = self._asking()
        listener.state_manager.voice_collect_seconds = 0.3
        judge = _install_intent_judge(listener, None)

        def slow(**_kwargs):
            time.sleep(0.35)
            return _make_judgment(directed=True, query="et demain", confidence="high")
        judge.judge.side_effect = slow

        self._hear(listener, "et demain")

        assert "et demain" in _accepted_query(listener)
        assert listener.state_manager.check_collection_timeout()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_verdict_given_without_his_question_is_not_the_last_word(self, _print):
        """While she speaks, the judge reads a longer utterance as it reads
        any, without his question before it. What he adds is judged again,
        as a follow-up, before a noise verdict can leave it out."""
        listener = self._asking(tts_speaking=True)
        judge = _install_intent_judge(listener, None)
        judge.judge.side_effect = [
            _make_judgment(directed=False, confidence="medium", noise=True,
                           reasoning="reads like a fragment from nowhere"),
            _make_judgment(directed=True, query="et pour après-demain aussi",
                           confidence="high"),
        ]

        self._hear(listener, "et pour à près de main aussi")

        assert "et pour à près de main aussi" in _accepted_query(listener)
        assert "comment tu vas" in _prompt_shown_to(judge)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_phantom_does_not_hold_his_question_back(self, _print):
        """His question goes once he has been silent for the collection's
        silence, whatever Whisper made of that silence."""
        listener = self._asking()
        listener.state_manager.voice_collect_seconds = 0.3
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle credit produced from silence"))

        time.sleep(0.2)
        self._hear(listener, "sous-titrage st' 501")
        time.sleep(0.15)

        assert listener.state_manager.check_collection_timeout()
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_judge_reads_it_after_his_question(self, _print):
        """His question is processed, so the transcript no longer shows it,
        and a misheard addition read alone looks like noise."""
        listener = self._asking()
        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="et pour après-demain aussi", confidence="high"))

        self._hear(listener, "et pour à près de main aussi")

        shown = _prompt_shown_to(judge)
        assert shown.index("comment tu vas") < shown.index("et pour à près de main aussi")
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_the_phantom_is_no_context_for_what_he_says_next(self, _print):
        listener = self._asking()
        _install_intent_judge(listener, _make_judgment(
            directed=False, confidence="high", noise=True,
            reasoning="subtitle credit produced from silence"))
        self._hear(listener, "sous-titrage st' 501")

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="ça va aujourd'hui ?", confidence="high"))
        self._hear(listener, "ça va aujourd'hui ?")

        shown = _prompt_shown_to(judge)
        assert "sous-titrage" not in shown
        assert "ça va aujourd'hui" in shown
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Hot window boundary enforcement
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHotWindowBoundary:
    """The hot window has a strict time boundary. Speech arriving after
    the window expires should require wake word detection."""

    @patch("builtins.print")
    def test_speech_within_window_accepted(self, _print):
        """Speech processed while hot window is active is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(directed=True, query="thanks"))
        listener._process_transcript("thanks", utterance_energy=0.01)

        assert _accepted_query(listener) == "thanks"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_after_window_requires_wake_word(self, _print):
        """Speech arriving after hot window expired requires wake word."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Let hot window expire
        assert _wait_for_hot_window_expiry(listener)

        # Speech without wake word — should be rejected
        _install_intent_judge(listener, _make_judgment(directed=True, query="tell me more"))
        listener._process_transcript("tell me more", utterance_energy=0.01)

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_after_window_with_wake_word_accepted(self, _print):
        """Speech after hot window expired but containing wake word is accepted."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)

        listener.echo_detector.track_tts_start("Short answer.")
        _simulate_tts_finish(listener)
        assert _wait_for_hot_window_active(listener)

        # Let hot window expire
        assert _wait_for_hot_window_expiry(listener)

        # Speech with wake word — accepted via wake word detection fallback
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what time is it"))
        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert _accepted_query(listener) != ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Echo is caught early (before beep and intent judge)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEchoCaughtBeforeBeepAndIntentJudge:
    """Echo in the hot window must be caught BEFORE the thinking beep starts
    and before the intent judge is called. This prevents:
    1. False beep on echo (user hears beep then nothing happens)
    2. Intent judge blocking the audio loop for seconds on echo
    3. Hot window extending indefinitely from repeated echo resets
    """

    @patch("builtins.print")
    def test_echo_in_hot_window_does_not_trigger_beep(self, _print):
        """Echo matching TTS output should not start the thinking beep."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True
        tts_text = "Tbilisi is a must-see especially the colourful old town."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Install intent judge that should NOT be called for echo
        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="tbilisi is a must-see"))

        listener._process_transcript(
            "Tbilisi is a must-see especially the colourful old town",
            utterance_energy=0.01)

        # No beep should have started
        assert not _is_beeping(listener)
        # Echo should be rejected — no query accepted
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_in_hot_window_skips_intent_judge(self, _print):
        """Echo caught early should not invoke the intent judge at all."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        tts_text = "For breathtaking scenery you should explore the mountainous regions."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="explore the mountainous regions"))

        listener._process_transcript(
            "For breathtaking scenery you should explore the mountainous regions like Steneti",
            utterance_energy=0.01)

        # Intent judge should not have been called
        judge.judge.assert_not_called()
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_echo_does_not_extend_hot_window(self, _print):
        """Echo rejection must NOT reset/extend the hot window timer.

        The window measures one fixed span from the TTS finish. If each
        echo chunk restarted that span, a run of chunks would keep the
        window open for the sum of their arrivals instead of its own
        duration, and speech long after TTS would be treated as hot
        window input.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.10)
        tts_text = "The answer is sunny and warm."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Record when hot window started
        original_start = listener.state_manager._hot_window_start_time

        # Process echo — should be caught early
        listener._process_transcript(
            "the answer is sunny and warm",
            utterance_energy=0.01)

        # Hot window start time should NOT have been reset
        assert listener.state_manager._hot_window_start_time == original_start

        # Wait for original window to expire
        assert _wait_for_hot_window_expiry(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_non_echo_in_hot_window_still_triggers_beep(self, _print):
        """Non-echo speech in hot window should still get the early beep."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True
        tts_text = "The weather is sunny today."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what about tomorrow"))

        listener._process_transcript("what about tomorrow", utterance_energy=0.01)

        assert _accepted_query(listener) == "what about tomorrow"
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_multiple_echo_chunks_do_not_stack_window_extensions(self, _print):
        """Multiple echo chunks should not extend the hot window repeatedly.

        Real scenario: TTS response is split into 2+ Whisper chunks. Each
        previously reset the timer, creating a window of N*hot_window_seconds.
        Now echo is caught early without any timer reset.
        """
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.10)
        tts_text = "Tbilisi is a must-see. For breathtaking scenery explore Svaneti."

        listener.echo_detector.track_tts_start(tts_text)
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # First echo chunk
        listener._process_transcript(
            "Tbilisi is a must-see especially the colourful old town",
            utterance_energy=0.01)

        # Second echo chunk
        listener._process_transcript(
            "For breathtaking scenery you should explore Steneti",
            utterance_energy=0.01)

        # Both should be rejected
        assert _accepted_query(listener) == ""

        # Window should still expire on original schedule
        assert _wait_for_hot_window_expiry(listener)

        # Speech after expiry requires wake word
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what the hell"))
        listener._process_transcript("what the hell", utterance_energy=0.01)
        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Speech without wake word outside hot window is ignored
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpeechIgnoredOutsideHotWindow:
    """When no hot window is active and no wake word is present, all speech
    should be completely ignored — no beep, no intent judge query, no action.
    This is the default idle state."""

    @patch("builtins.print")
    def test_complete_sentence_without_wake_word_ignored(self, _print):
        """A full sentence without wake word and no hot window is ignored."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        # Judge would accept if asked — but it shouldn't matter
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="what is the meaning of life"))

        listener._process_transcript(
            "what is the meaning of life",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_no_beep_no_intent_for_background_chatter(self, _print):
        """Background conversation without wake word triggers no beep and
        no intent judge invocation."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)
        listener.cfg.tune_enabled = True

        judge = _install_intent_judge(listener, _make_judgment(
            directed=True, query="pass the salt"))

        listener._process_transcript(
            "hey can you pass the salt please",
            utterance_energy=0.01,
        )

        assert _accepted_query(listener) == ""
        # Intent judge should still be called (it's the decision-maker),
        # but since it returns directed without wake word, it's rejected
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_multiple_utterances_after_hot_window_all_ignored(self, _print):
        """Multiple consecutive utterances after hot window expires are all
        ignored if they lack a wake word. The system stays in wake word mode."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("The answer is 42.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        # Expire hot window
        listener.state_manager.expire_hot_window()
        assert not listener.state_manager.is_hot_window_active()

        # Install judge that would accept everything
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="first remark"))

        # First utterance — no wake word, no hot window
        listener._process_transcript("I think it might rain later", utterance_energy=0.01)
        assert _accepted_query(listener) == ""

        # Second utterance — still no wake word, still no hot window
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="second remark"))
        listener._process_transcript("yeah the forecast said so", utterance_energy=0.01)
        assert _accepted_query(listener) == ""

        # Third utterance with wake word — THIS should work
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="will it rain"))
        listener._process_transcript("jarvis will it rain today", utterance_energy=0.01)
        assert "rain" in _accepted_query(listener)
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_speech_long_after_any_tts_ignored(self, _print):
        """Speech arriving long after any TTS activity is ignored without
        wake word, even if the intent judge says directed."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        # TTS happened ages ago, hot window long expired
        listener.echo_detector.track_tts_start("Old response.")
        listener.echo_detector.track_tts_finish()
        # No hot window scheduled — simulates a stale session

        _install_intent_judge(listener, _make_judgment(
            directed=True, query="hey what time is it"))

        # Speech with timestamps well after any TTS
        now = time.time()
        listener._process_transcript(
            "hey what time is it",
            utterance_energy=0.01,
            utterance_start_time=now,
            utterance_end_time=now + 1.0,
        )

        assert _accepted_query(listener) == ""
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: Stale wake timestamp must not leak across utterances
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStaleWakeTimestampAcrossUtterances:
    """After the intent judge rejects a wake-worded utterance, the next
    utterance without a wake word must not be accepted just because the
    previous utterance had one.

    Real-world bug: user said "Jarvis, remember..." (rejected by judge),
    then said "Hey Google, TV off." The judge saw the previous "Jarvis"
    in its buffer and returned directed=true with query="tv off". The
    verification guard `_wake_timestamp is not None` short-circuited true
    because it was never cleared, so the unrelated "Hey Google" command
    was accepted.
    """

    @patch("builtins.print")
    def test_rejected_wake_utterance_does_not_vouch_for_next_utterance(self, _print):
        """A prior rejected wake-worded utterance must not authorise a later
        utterance that lacks a wake word."""
        listener, _ = _create_listener(echo_tolerance=0.3, hot_window_seconds=3.0)

        # First utterance: has "jarvis", judge rejects as not directed
        _install_intent_judge(listener, _make_judgment(
            directed=False, query="", confidence="high",
            reasoning="statement to self, not directed"))

        now = time.time()
        listener._process_transcript(
            "jarvis i want you to remember that my other office days are thursdays",
            utterance_energy=0.01,
            utterance_start_time=now,
            utterance_end_time=now + 2.0,
        )
        assert _accepted_query(listener) == ""

        # Second utterance: no wake word, judge hallucinates directed=true
        # (e.g. because the earlier "jarvis" is still in its context buffer)
        _install_intent_judge(listener, _make_judgment(
            directed=True, query="tv off", confidence="high",
            reasoning="synthesised from buffer"))

        listener._process_transcript(
            "hey google, tv off.",
            utterance_energy=0.01,
            utterance_start_time=now + 5.0,
            utterance_end_time=now + 6.0,
        )

        # Must be rejected — no wake word in this utterance, no hot window
        assert _accepted_query(listener) == "", (
            "Second utterance without wake word must not be accepted just "
            "because a prior utterance set _wake_timestamp")
        listener.state_manager.stop()


@pytest.mark.unit
class TestIntentJudgeGating:
    """The intent judge must not be called on pure ambient speech.

    Calling it on every utterance blocks the audio loop for up to
    `intent_judge_timeout_sec` on each background chatter, which can
    cascade into UI freezes when many utterances queue up during a slow
    or loaded Ollama. The judge adds value only when there's an
    engagement signal: wake word, hot window, or active TTS.
    """

    @patch("builtins.print")
    def test_judge_not_called_for_ambient_speech(self, _print):
        """Ambient speech with no wake word / hot window / TTS must not hit the judge."""
        listener, _ = _create_listener()

        mock_judge = _install_intent_judge(
            listener, _make_judgment(directed=False, query=""))

        # No hot window, no TTS, no wake word in the text
        listener._process_transcript(
            "random background chatter about the weather",
            utterance_energy=0.01,
        )

        assert mock_judge.judge.call_count == 0, (
            "Intent judge must be gated on an engagement signal; ambient "
            "speech should skip the judge to avoid blocking the audio loop")
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_called_when_wake_word_detected(self, _print):
        """Utterances containing the wake word do reach the judge."""
        listener, _ = _create_listener()

        mock_judge = _install_intent_judge(
            listener, _make_judgment(
                directed=True, query="what time is it"))

        listener._process_transcript(
            "jarvis what time is it", utterance_energy=0.01,
        )

        assert mock_judge.judge.call_count == 1
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_called_in_hot_window(self, _print):
        """Utterances during the hot window do reach the judge."""
        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=3.0)

        listener.echo_detector.track_tts_start("Here you go.")
        _simulate_tts_finish(listener)
        _wait_for_hot_window_active(listener)

        mock_judge = _install_intent_judge(
            listener, _make_judgment(directed=True, query="thanks"))

        listener._process_transcript("thanks", utterance_energy=0.01)

        assert mock_judge.judge.call_count == 1
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_skipped_for_short_utterance_during_tts(self, _print):
        """Short utterances (<=3 words) during active TTS bypass the judge.

        The fast text-based stop-command check already handles short
        interruptions like "stop" / "shut up" while TTS is speaking. Sending
        these to the judge would block the audio loop for the judge's
        timeout on every short echo chunk during playback.
        """
        listener, mock_tts = _create_listener(tts_speaking=True)

        mock_judge = _install_intent_judge(
            listener, _make_judgment(directed=False, query=""))

        listener._process_transcript(
            "uh huh yeah", utterance_energy=0.01,
        )

        assert mock_judge.judge.call_count == 0, (
            "Short utterances during TTS must be handled by the stop-command "
            "path, not the judge, to avoid blocking the audio loop")
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_judge_called_for_longer_utterance_during_tts(self, _print):
        """Longer utterances (>3 words) during TTS still reach the judge.

        Active TTS is itself an engagement signal — the user may be
        interrupting with a real follow-up or correction, and the judge
        needs to see it to catch intents the fast text-based stop-command
        check misses.
        """
        listener, mock_tts = _create_listener(tts_speaking=True)

        mock_judge = _install_intent_judge(
            listener, _make_judgment(
                directed=True, query="what about tomorrow's weather"))

        # >3 words, no stop-command keywords, not echo
        listener._process_transcript(
            "actually what about tomorrow's weather",
            utterance_energy=0.01,
        )

        assert mock_judge.judge.call_count == 1
        listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: The window's length does not depend on what activation announces
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHotWindowLengthIsMeasuredFromActivation:
    """``hot_window_seconds`` runs from the moment the window opens.

    Activation also publishes the assistant's state and tells the console,
    and either can be slow: a state file on a busy disk, a console that
    blocks. The clock is armed before that work, so the window is the same
    length whatever the announcement costs."""

    @patch("builtins.print")
    def test_a_slow_state_update_does_not_lengthen_the_window(self, _print):
        from unittest.mock import MagicMock

        listener, _ = _create_listener(echo_tolerance=0.02, hot_window_seconds=0.05)
        announcement_over = threading.Event()

        announcing = []

        def slow_publish():
            announcing.append(time.time())
            # The announcement's cost. The test ends it as soon as it has its
            # answer, so the timer threads still announcing do not outlive it.
            announcement_over.wait(0.6)
            return MagicMock()

        try:
            with patch("jarvis.listening.state_manager.get_jarvis_state", side_effect=slow_publish):
                listener.echo_detector.track_tts_start("Short answer.")
                _simulate_tts_finish(listener)
                assert _wait_for_hot_window_active(listener)
                opened_at = time.time()

                # Six times the window, a tenth of the announcement.
                assert _wait_for_hot_window_expiry(listener, timeout=0.3), (
                    f"still open {time.time() - opened_at:.2f}s after a 0.05s window"
                )
                assert announcing, "the activation never reached the slow state update"
        finally:
            announcement_over.set()
            listener.state_manager.stop()


# ---------------------------------------------------------------------------
# Tests: The listener publishes the assistant's state as a query moves through
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTheListenerPublishesItsState:
    """Whatever shows the assistant to the user (orb, dashboard) reads the
    shared state the listener publishes; these follow one query through it.
    The state is read back through the same reader a viewer uses."""

    @patch("builtins.print")
    def test_hearing_the_wake_word_is_listening(self, _print):
        from jarvis.state import JarvisState, get_jarvis_state

        listener, _ = _create_listener()
        get_jarvis_state().set_state(JarvisState.IDLE)

        listener._process_transcript("jarvis what time is it", utterance_energy=0.01)

        assert get_jarvis_state().state == JarvisState.LISTENING
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_query_is_thinking_while_the_engine_works_on_it(self, _print):
        from jarvis.state import JarvisState, get_jarvis_state

        listener, _ = _create_listener()
        get_jarvis_state().set_state(JarvisState.LISTENING)
        seen_by_the_engine = []

        def engine(*_args, **_kwargs):
            seen_by_the_engine.append(get_jarvis_state().state)
            return None

        with patch("jarvis.reply.engine.run_reply_engine", side_effect=engine):
            listener._dispatch_query(_Turn("what time is it"))

        assert seen_by_the_engine == [JarvisState.THINKING]
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_stop_word_leaves_the_assistant_idle(self, _print):
        """An interrupted reply is unheard, so the completion callback that
        would end the speech never fires: whoever stops her publishes IDLE."""
        from jarvis.state import JarvisState, get_jarvis_state

        listener, tts = _create_listener(tts_speaking=True)
        get_jarvis_state().set_state(JarvisState.SPEAKING)

        listener._process_transcript("stop", utterance_energy=0.01)

        tts.interrupt.assert_called_once()
        assert get_jarvis_state().state == JarvisState.IDLE
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_stop_heard_by_the_intent_judge_leaves_the_assistant_idle(self, _print):
        from jarvis.state import JarvisState, get_jarvis_state

        listener, tts = _create_listener(tts_speaking=True)
        _install_intent_judge(listener, _make_judgment(directed=False, stop=True))
        get_jarvis_state().set_state(JarvisState.SPEAKING)

        listener._process_transcript("that is enough talking from you now", utterance_energy=0.01)

        tts.interrupt.assert_called_once()
        assert get_jarvis_state().state == JarvisState.IDLE
        listener.state_manager.stop()

    @patch("builtins.print")
    def test_a_query_with_no_reply_leaves_the_assistant_idle(self, _print):
        """Nothing will be spoken to move the state on, so the tune that was
        covering the wait stops and the assistant goes back to idle."""
        from jarvis.state import JarvisState, get_jarvis_state

        listener, _ = _create_listener()
        listener._tune_player = MagicMock()
        get_jarvis_state().set_state(JarvisState.THINKING)

        listener._speak_reply(None)

        assert get_jarvis_state().state == JarvisState.IDLE
        listener.state_manager.stop()
