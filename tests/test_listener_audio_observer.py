"""Tests for the audio observer hook on the voice listener.

The orb UI registers an observer that receives every mic audio chunk
via ``_on_audio``. The contract this test verifies is the one that
matters most: a buggy observer must never break STT. If an observer
raises, the listener must still push the chunk onto its STT queue.

We bypass the heavy VoiceListener construction (which would load
Whisper, open the mic, etc.) and exercise the callback directly via
a minimal stub that carries the same attributes ``_on_audio`` reads.
This is a unit test of the *callback contract*, not an integration
test of the full listener.
"""

from __future__ import annotations

import queue
import threading

import numpy as np
import pytest

from jarvis.listening import listener as _listener
from jarvis.listening.listener import (
    VoiceListener,
    register_audio_observer,
    unregister_audio_observer,
)


class _AudioCallbackStub:
    """Tiny stand-in for the parts of VoiceListener that ``_on_audio``
    reads. Lets us exercise the callback without booting Whisper or
    opening an InputStream."""

    def __init__(self) -> None:
        self._should_stop = False
        self._dictation_active = False
        self._callback_count = 0
        self._audio_q: queue.Queue = queue.Queue(maxsize=64)

    # Bind the unbound method from the real listener so we exercise
    # the exact same code path the production callback runs.
    _on_audio = VoiceListener._on_audio


@pytest.fixture(autouse=True)
def _clean_registry():
    """Each test starts with an empty observer registry and ends with
    the same, so test ordering can't leak observers between cases."""
    snapshot = list(_listener._audio_observers)
    _listener._audio_observers.clear()
    yield
    _listener._audio_observers.clear()
    _listener._audio_observers.extend(snapshot)


class TestObserverResilience:
    """The privacy-of-STT contract: a faulty observer must never block
    or break the listener."""

    @pytest.mark.unit
    def test_throwing_observer_does_not_break_stt(self):
        """Register an observer that always raises, then invoke the
        callback. The chunk must still land on the STT queue.

        This is the test that proves the spec rule 'STT continue à
        fonctionner' under the (1a) decision (listener modify
        allowed only for observer fan-out)."""
        calls = []

        def angry_observer(chunk):
            calls.append(chunk)
            raise RuntimeError("observer is on fire")

        register_audio_observer(angry_observer)

        stub = _AudioCallbackStub()
        chunk = np.zeros(480, dtype=np.float32)
        stub._on_audio(chunk, frames=480, time_info=None, status=None)

        # The observer ran...
        assert len(calls) == 1
        # ...and STT got the chunk anyway.
        assert stub._audio_q.qsize() == 1

    @pytest.mark.unit
    def test_multiple_observers_isolated(self):
        """One observer raising must not prevent the next observer
        from running. Order matters here: we exercise the registry's
        per-observer try/except, not a global try/except."""
        log: list[str] = []

        def good_a(chunk): log.append("a")
        def bad(chunk):
            log.append("b")
            raise ValueError("boom")
        def good_c(chunk): log.append("c")

        register_audio_observer(good_a)
        register_audio_observer(bad)
        register_audio_observer(good_c)

        stub = _AudioCallbackStub()
        stub._on_audio(np.zeros(480, dtype=np.float32), 480, None, None)

        # All three were attempted.
        assert log == ["a", "b", "c"]
        # And STT still got the chunk.
        assert stub._audio_q.qsize() == 1


class TestObserverRegistryAPI:
    """Surface checks on register/unregister."""

    @pytest.mark.unit
    def test_double_register_is_idempotent(self):
        """Registering the same callable twice must not result in it
        being called twice per chunk. Common foot-gun if a widget
        re-registers on reload."""
        calls = []
        def obs(chunk): calls.append(1)
        register_audio_observer(obs)
        register_audio_observer(obs)

        stub = _AudioCallbackStub()
        stub._on_audio(np.zeros(480, dtype=np.float32), 480, None, None)
        assert len(calls) == 1

    @pytest.mark.unit
    def test_unregister_removes_observer(self):
        calls = []
        def obs(chunk): calls.append(1)
        register_audio_observer(obs)
        unregister_audio_observer(obs)

        stub = _AudioCallbackStub()
        stub._on_audio(np.zeros(480, dtype=np.float32), 480, None, None)
        assert calls == []
        assert stub._audio_q.qsize() == 1

    @pytest.mark.unit
    def test_unregister_unknown_is_silent(self):
        def obs(chunk): pass
        # Was never registered; must not raise.
        unregister_audio_observer(obs)

    @pytest.mark.unit
    def test_dictation_pause_skips_observers(self):
        """When dictation is active the listener returns early. Make
        sure observers see no audio during that window so the orb
        doesn't react to dictation recordings the rest of the pipeline
        is ignoring."""
        calls = []
        register_audio_observer(lambda chunk: calls.append(1))
        stub = _AudioCallbackStub()
        stub._dictation_active = True
        stub._on_audio(np.zeros(480, dtype=np.float32), 480, None, None)
        assert calls == []
        # And STT also skipped (existing behaviour, just verifying).
        assert stub._audio_q.qsize() == 0
