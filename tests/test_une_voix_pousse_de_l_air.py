"""Overriding an echo verdict takes a voice, and a voice moves air.

Field trace, 2026-08-17. Whisper turned background noise into "Thank
you." while she was speaking French. The judge called it her echo. The
guard added the day before asked "are there words here she did not say?"
— and there were, because "thank you" resembles nothing in a French
sentence. So the override fired and she answered a hallucination:
"De rien, Hocine."

The guard worked exactly as designed, and that is the fault. Its premise
was that words she did not say are the user's words. A hallucination
produces words *nobody* said, so the premise fails on it — the same
absence-read-as-presence this codebase keeps producing.

Text cannot separate the two: "thank you" and a real English interjection
are the same string. What separates them is that one of them moved air.
A person talking over her puts their voice on top of the echo and the
energy rises; noise Whisper dreamt into words does not.

The detector already tracks the ambient level at the moment she started
speaking, and already uses `energy_spike_threshold` for exactly this
comparison elsewhere. This applies it to the override.

Fails open: with no baseline recorded there is nothing to compare
against, and the utterance is treated as the user's.
"""

from __future__ import annotations

import pytest

from src.jarvis.listening.echo_detection import EchoDetector


def _detecteur(baseline: float):
    d = EchoDetector()
    d.track_tts_start("Salut Hocine, je vais bien merci.", baseline_energy=baseline)
    return d


def test_a_hallucination_over_silence_does_not_override():
    """The trace: energy at the ambient floor, so nothing was said."""
    d = _detecteur(0.0045)

    assert d.utterance_carries_a_voice(0.0045) is False


def test_a_voice_over_her_does_override():
    """The control, and the reason the override exists: he talks over her
    and his words must survive."""
    d = _detecteur(0.0045)

    assert d.utterance_carries_a_voice(0.02) is True


def test_the_bar_is_the_existing_spike_threshold():
    """Reuses the multiplier the detector already applies to this same
    question after TTS, rather than inventing a second number."""
    d = _detecteur(0.01)

    assert d.utterance_carries_a_voice(0.01 * d.energy_spike_threshold * 1.1) is True
    assert d.utterance_carries_a_voice(0.01 * d.energy_spike_threshold * 0.9) is False


def test_without_a_baseline_it_stays_out_of_the_way():
    """No recorded ambient level means no grounds to call anything noise.
    Failing shut here would eat real speech."""
    d = EchoDetector()

    assert d.utterance_carries_a_voice(0.0) is True
    assert d.utterance_carries_a_voice(0.5) is True


def test_an_unmeasured_utterance_is_not_judged():
    """Energy of zero is the 'not measured' value the listener passes when
    it has no frames, not a claim of silence."""
    d = _detecteur(0.0045)

    assert d.utterance_carries_a_voice(0.0) is True
