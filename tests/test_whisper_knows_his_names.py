"""The names he uses, given to Whisper before it hears him.

Whisper pulls a proper noun it has rarely seen towards a word it knows.
"Bagneux" came back "Bagnus", the weather request went to "Bagnus", and
the answer was the weather of a village in Switzerland. Measured on
synthetic speech with the same model and filters as the listener, the
name came out right 271 times in 672 without a hint and 502 with one.

The hint has a price, and the tests below hold the listener to paying
none of it: over noise Whisper repeats the hint back. Background noise
that comes out "Thank you." without the hint comes out "Bagneux" with
it, which clears both confidence filters and, in a hot window, would be
taken as a question.
So the hint may change how a word is spelled, never whether anything was
said.

The model here is a stand-in that behaves the way the real one was
measured to: given the hint it writes the name properly, and over noise
it repeats the hint.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.listening.listener import VoiceListener
from jarvis.listening.vocabulary import whisper_hint


SPOKEN = "Quel temps fait-il à {} ?"
MISHEARD = SPOKEN.format("Bagnus")


def _cfg(**overrides):
    values = dict(
        weather_city="Bagneux",
        whisper_vocabulary=[],
        wake_word="yuba",
        wake_aliases=["youba", "juba"],
        whisper_min_confidence=0.3,
        whisper_no_speech_threshold=0.5,
        voice_debug=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Whisper:
    """faster-whisper as measured.

    ``speech`` is a sentence with one proper noun in braces: without a
    hint the noun comes out as ``misheard``; with a hint naming it, as
    the hint spells it. ``speech=None`` is noise: nothing without a
    hint, the hint itself with one.
    """

    def __init__(self, speech=None, misheard="Bagnus", over_noise=None):
        self.speech = speech
        self.misheard = misheard
        self.over_noise = over_noise

    def heard(self, hint):
        if self.speech is None:
            if hint:
                return self.over_noise(hint) if self.over_noise else hint
            return ""
        names = [n.strip() for n in (hint or "").split(",")]
        for name in names:
            if name and name.casefold().startswith(self.misheard[:3].casefold()):
                return self.speech.format(name)
        return self.speech.format(self.misheard)

    def transcribe(self, audio, **kwargs):
        hint = kwargs.get("hotwords")
        text = self.heard(hint)
        segments = [SimpleNamespace(text=text, avg_logprob=-0.2, no_speech_prob=0.05)] if text else []
        return iter(segments), SimpleNamespace(language="fr")


class _MlxWhisper(_Whisper):
    """mlx_whisper takes the hint as ``initial_prompt`` and answers a dict."""

    def transcribe(self, audio, path_or_hf_repo=None, language=None, **kwargs):
        hint = kwargs.get("initial_prompt")
        text = self.heard(hint)
        segments = [{"text": text, "avg_logprob": -0.2, "no_speech_prob": 0.05}] if text else []
        return {"text": text, "segments": segments, "language": "fr"}


def _listener(cfg, model):
    listener = VoiceListener.__new__(VoiceListener)
    listener.cfg = cfg
    listener.model = model
    listener._whisper_backend = "faster-whisper"
    listener._whisper_device = "cuda"
    listener._mlx_model_repo = "mlx-community/whisper-large-v3-turbo"
    listener.transcribe_lock = threading.Lock()
    listener._last_detected_language = None
    return listener


AUDIO = np.zeros(16000, dtype=np.float32)


# ── The hint ──────────────────────────────────────────────────────────


def test_the_city_he_lives_in_is_in_it():
    assert "Bagneux" in whisper_hint(_cfg())


def test_the_names_he_listed_are_in_it_too():
    hint = whisper_hint(_cfg(whisper_vocabulary=["Axendly", "Mon Rempart"]))

    for name in ("Bagneux", "Axendly", "Mon Rempart"):
        assert name in hint


def test_no_names_means_no_hint():
    """Nothing to steer, so Whisper is called exactly as before."""
    assert whisper_hint(_cfg(weather_city="", whisper_vocabulary=[])) is None


def test_a_name_listed_twice_is_given_once():
    hint = whisper_hint(_cfg(whisper_vocabulary=["bagneux", "Axendly"]))

    assert hint.casefold().count("bagneux") == 1


def test_her_name_is_never_in_it():
    """The hint makes Whisper write its words more readily, and writing
    hers more readily is a false wake."""
    cfg = _cfg(whisper_vocabulary=["Yuba", "Axendly"])

    assert "yuba" not in whisper_hint(cfg).casefold()


def test_nor_is_a_word_she_answers_to():
    """Juba is a city, and also how Whisper spells her name."""
    cfg = _cfg(weather_city="Juba")

    assert whisper_hint(cfg) is None


def test_a_name_close_enough_to_wake_her_is_left_out():
    cfg = _cfg(whisper_vocabulary=["Youbah", "Axendly"])

    assert whisper_hint(cfg) == "Bagneux, Axendly"


def test_settings_that_are_not_names_are_ignored():
    cfg = _cfg(weather_city=None, whisper_vocabulary=[3, "", "  ", "Axendly"])

    assert whisper_hint(cfg) == "Axendly"


# ── What he said ──────────────────────────────────────────────────────


def test_the_name_comes_out_the_way_he_spells_it():
    listener = _listener(_cfg(), _Whisper(SPOKEN))

    assert listener._transcribe(AUDIO) == SPOKEN.format("Bagneux")


def test_without_names_nothing_changes():
    listener = _listener(_cfg(weather_city=""), _Whisper(SPOKEN))

    assert listener._transcribe(AUDIO) == MISHEARD


def test_on_a_mac_too():
    listener = _listener(_cfg(), None)
    listener._whisper_backend = "mlx"
    model = _MlxWhisper(SPOKEN)

    with patch("jarvis.listening.listener.mlx_whisper", model, create=True):
        assert listener._transcribe(AUDIO) == SPOKEN.format("Bagneux")


# ── What nobody said ──────────────────────────────────────────────────


def test_noise_does_not_become_his_city():
    listener = _listener(_cfg(), _Whisper(speech=None))

    assert listener._transcribe(AUDIO) == ""


def test_noise_reads_as_it_did_before_the_hint():
    """The hint adds nothing to noise and takes nothing away: whatever
    Whisper made of it unhinted is what the listener gets."""
    model = _Whisper(speech=None)
    model.heard = lambda hint: hint if hint else "Thank you."
    listener = _listener(_cfg(), model)

    assert listener._transcribe(AUDIO) == "Thank you."


def test_a_list_repeated_back_with_a_stray_word_is_still_the_list():
    """Measured with fourteen names: the list came back over noise with a
    word of its own tacked on ("…, Vercel, T.J."). Nearly every word is
    still the hint's."""
    names = ["Axendly", "Mon Rempart", "Playwright", "Vercel", "Supabase"]
    model = _Whisper(speech=None, over_noise=lambda hint: hint + ", T.J.")
    listener = _listener(_cfg(whisper_vocabulary=names), model)

    assert listener._transcribe(AUDIO) == ""


def test_on_a_mac_noise_stays_noise():
    listener = _listener(_cfg(), None)
    listener._whisper_backend = "mlx"

    with patch("jarvis.listening.listener.mlx_whisper", _MlxWhisper(speech=None), create=True):
        assert listener._transcribe(AUDIO) == ""


def test_a_sentence_that_merely_names_the_city_is_his():
    """One name among other words is the hint doing its job."""
    listener = _listener(_cfg(), _Whisper("Météo {}"))

    assert listener._transcribe(AUDIO) == "Météo Bagneux"


def test_his_name_alone_reads_as_it_did_before_the_hint():
    """The price of the guard: an answer that is nothing but one of his
    names cannot be told from the hint repeated back, so it is read
    without the hint."""
    listener = _listener(_cfg(), _Whisper("{}"))

    assert listener._transcribe(AUDIO) == "Bagnus"


# ── All the way to the transcript ─────────────────────────────────────


def _wired_listener(cfg, model):
    with patch("jarvis.listening.listener.FASTER_WHISPER_AVAILABLE", True), \
            patch("jarvis.listening.listener.MLX_WHISPER_AVAILABLE", False), \
            patch("jarvis.listening.listener.WhisperModel"):
        listener = VoiceListener(MagicMock(), cfg, MagicMock(), MagicMock())
    listener.model = model
    listener._whisper_backend = "faster-whisper"
    listener._whisper_device = "cuda"
    listener._samplerate = 16000
    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
    listener.echo_detector._utterance_start_time = time.time() - 1.0
    listener.is_speech_active = True
    listener._process_transcript = MagicMock()
    return listener


def _mock_cfg(**overrides):
    cfg = MagicMock()
    for key, value in dict(
        sample_rate=16000, vad_enabled=False, echo_tolerance=0.3,
        echo_energy_threshold=2.0, hot_window_seconds=3.0,
        voice_collect_seconds=2.0, voice_max_collect_seconds=60.0,
        tune_enabled=False, voice_debug=False, whisper_min_confidence=0.3,
        whisper_no_speech_threshold=0.5, whisper_min_audio_duration=0.15,
        wake_word="yuba", wake_aliases=["youba"], weather_city="Bagneux",
        whisper_vocabulary=[],
    ).items():
        setattr(cfg, key, value)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_the_corrected_sentence_is_what_reaches_the_transcript():
    listener = _wired_listener(_mock_cfg(), _Whisper(SPOKEN))

    listener._finalize_utterance()

    heard = [s.text for s in listener._transcript_buffer.get_since(0)]
    assert heard == [SPOKEN.format("Bagneux")]


def test_noise_in_a_hot_window_never_reaches_the_transcript_as_his_city():
    listener = _wired_listener(_mock_cfg(), _Whisper(speech=None))

    listener._finalize_utterance()

    assert listener._transcript_buffer.get_since(0) == []
    listener._process_transcript.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_hint_reaches_whisper_on_either_device(device):
    """On CPU the listener stops carrying text from one window to the
    next; the hint is not carried text and survives that."""
    model = _Whisper(SPOKEN)
    listener = _listener(_cfg(), model)
    listener._whisper_device = device

    assert listener._transcribe(AUDIO) == SPOKEN.format("Bagneux")


# ── From his settings ─────────────────────────────────────────────────


def _settings_from(tmp_path, monkeypatch, values: dict):
    import json

    from jarvis.config import load_settings

    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


def test_no_names_are_listed_by_default(tmp_path, monkeypatch):
    assert _settings_from(tmp_path, monkeypatch, {}).whisper_vocabulary == []


def test_the_names_he_lists_in_his_settings_reach_the_hint(tmp_path, monkeypatch):
    cfg = _settings_from(tmp_path, monkeypatch, {
        "weather_city": "Bagneux",
        "whisper_vocabulary": ["Axendly", " Mon Rempart "],
    })

    assert whisper_hint(cfg) == "Bagneux, Axendly, Mon Rempart"


def test_a_vocabulary_that_is_not_a_list_is_read_as_none(tmp_path, monkeypatch):
    cfg = _settings_from(tmp_path, monkeypatch, {"whisper_vocabulary": {"a": 1}})

    assert cfg.whisper_vocabulary == []
