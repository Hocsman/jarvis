"""The names he uses, given to Whisper before it hears him.

Whisper pulls a proper noun it has rarely seen towards a word it knows.
A town's name came back a letter short, the weather request went to that
spelling, and the answer was the weather of a village in another country.
Measured on synthetic speech with the same model and filters as the
listener, the name came out right 271 times in 672 without a hint and
502 with one.

The hint has a price, and the tests below hold the listener to paying
none of it: over noise Whisper repeats the hint back. Background noise
that comes out "Thank you." without the hint comes out as the town with
it, which clears both confidence filters and, in a hot window, would be
taken as a question.
So the hint may change how a word is spelled, never whether anything was
said.

The model here is a stand-in that behaves the way the real one was
measured to: given the hint it writes the name properly, and over noise
it repeats the hint. Every name is invented.
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


CITY = "Varnelle"
CITY_MISHEARD = "Varnel"
SPOKEN = "Quel temps fait-il à {} ?"
MISHEARD = SPOKEN.format(CITY_MISHEARD)

DETECTOR_MISSING = "[ONNXRuntimeError] : 3 : NO_SUCHFILE : silero_encoder_v5.onnx failed"


def _cfg(**overrides):
    values = dict(
        weather_city=CITY,
        whisper_vocabulary=[],
        wake_word="yuba",
        wake_aliases=["youba", "juba"],
        whisper_min_confidence=0.3,
        whisper_no_speech_threshold=0.5,
        whisper_language=None,
        whisper_vad=True,
        voice_debug=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Whisper:
    """faster-whisper as measured.

    ``speech`` is a sentence with one proper noun in braces: without a
    hint the noun comes out as ``misheard``; with a hint naming it, as
    the hint spells it. ``speech=None`` is noise its voice detector let
    through: nothing without a hint, the hint itself with one. With
    ``detector_fails``, asking for the voice detector raises what a build
    without its model files raises.
    """

    def __init__(self, speech=None, misheard=CITY_MISHEARD, over_noise=None,
                 detector_fails=False):
        self.speech = speech
        self.misheard = misheard
        self.over_noise = over_noise
        self.detector_fails = detector_fails
        self.calls = []

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
        self.calls.append(kwargs)
        if self.detector_fails and kwargs.get("vad_filter"):
            raise RuntimeError(DETECTOR_MISSING)
        hint = kwargs.get("hotwords")
        text = self.heard(hint)
        segments = [SimpleNamespace(text=text, avg_logprob=-0.2, no_speech_prob=0.05)] if text else []
        return iter(segments), SimpleNamespace(language="fr")


class _MlxWhisper(_Whisper):
    """mlx_whisper takes the hint as ``initial_prompt`` and answers a dict."""

    def transcribe(self, audio, path_or_hf_repo=None, language=None, **kwargs):
        self.calls.append(kwargs)
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


def _text(listener) -> str:
    """What the listener makes of one utterance."""
    text, _language, _pieces = listener._transcribe(AUDIO)
    return text


# ── The hint ──────────────────────────────────────────────────────────


def test_the_city_he_lives_in_is_in_it():
    assert CITY in whisper_hint(_cfg())


def test_the_names_he_listed_are_in_it_too():
    hint = whisper_hint(_cfg(whisper_vocabulary=["Ostrevane", "Pont Sarrel"]))

    for name in (CITY, "Ostrevane", "Pont Sarrel"):
        assert name in hint


def test_no_names_means_no_hint():
    """Nothing to steer, so Whisper is called exactly as without one."""
    assert whisper_hint(_cfg(weather_city="", whisper_vocabulary=[])) is None


def test_a_name_listed_twice_is_given_once():
    hint = whisper_hint(_cfg(whisper_vocabulary=[CITY.lower(), "Ostrevane"]))

    assert hint.casefold().count(CITY.casefold()) == 1


def test_her_name_is_never_in_it():
    """The hint makes Whisper write its words more readily, and writing
    hers more readily is a false wake."""
    cfg = _cfg(whisper_vocabulary=["Yuba", "Ostrevane"])

    assert "yuba" not in whisper_hint(cfg).casefold()


def test_nor_is_a_word_she_answers_to():
    """Juba is a city, and also how Whisper spells her name."""
    cfg = _cfg(weather_city="Juba")

    assert whisper_hint(cfg) is None


def test_a_name_close_enough_to_wake_her_is_left_out():
    cfg = _cfg(whisper_vocabulary=["Youbah", "Ostrevane"])

    assert whisper_hint(cfg) == f"{CITY}, Ostrevane"


def test_settings_that_are_not_names_are_ignored():
    cfg = _cfg(weather_city=None, whisper_vocabulary=[3, "", "  ", "Ostrevane"])

    assert whisper_hint(cfg) == "Ostrevane"


# ── What he said ──────────────────────────────────────────────────────


def test_the_name_comes_out_the_way_he_spells_it():
    listener = _listener(_cfg(), _Whisper(SPOKEN))

    assert _text(listener) == SPOKEN.format(CITY)


def test_without_names_nothing_changes():
    listener = _listener(_cfg(weather_city=""), _Whisper(SPOKEN))

    assert _text(listener) == MISHEARD


def test_without_names_whisper_is_given_no_argument_for_one():
    model = _Whisper(SPOKEN)
    listener = _listener(_cfg(weather_city=""), model)

    _text(listener)

    assert model.calls and all("hotwords" not in call for call in model.calls)


def test_on_a_mac_too():
    listener = _listener(_cfg(), None)
    listener._whisper_backend = "mlx"
    model = _MlxWhisper(SPOKEN)

    with patch("jarvis.listening.listener.mlx_whisper", model, create=True):
        assert _text(listener) == SPOKEN.format(CITY)


def test_on_a_mac_without_names_whisper_is_given_no_argument_for_one():
    listener = _listener(_cfg(weather_city=""), None)
    listener._whisper_backend = "mlx"
    model = _MlxWhisper(SPOKEN)

    with patch("jarvis.listening.listener.mlx_whisper", model, create=True):
        _text(listener)

    assert model.calls and all("initial_prompt" not in call for call in model.calls)


@pytest.mark.parametrize("language", [None, "fr"])
def test_the_name_comes_out_right_whether_or_not_his_language_is_named(language):
    listener = _listener(_cfg(whisper_language=language), _Whisper(SPOKEN))

    assert _text(listener) == SPOKEN.format(CITY)


def test_a_voice_detector_that_fails_does_not_take_the_hint_with_it():
    """Whisper's voice detector failing makes the listener hear the
    utterance again without it (`whisper_vad.py`). That second call still
    carries his names."""
    listener = _listener(_cfg(whisper_vad=True), _Whisper(SPOKEN, detector_fails=True))

    assert _text(listener) == SPOKEN.format(CITY)


def test_a_whisper_that_takes_only_the_language_still_hears_him():
    """A model that takes none of the listener's arguments is called with
    the language alone: no hint reaches it, and his sentence does."""
    model = MagicMock()
    model.transcribe.side_effect = lambda audio, language=None: (
        [SimpleNamespace(text=MISHEARD, avg_logprob=-0.2, no_speech_prob=0.05)],
        SimpleNamespace(language="fr"))
    listener = _listener(_cfg(), model)

    assert _text(listener) == MISHEARD


# ── What nobody said ──────────────────────────────────────────────────


def test_noise_does_not_become_his_city():
    listener = _listener(_cfg(), _Whisper(speech=None))

    assert _text(listener) == ""


def test_noise_reads_as_it_did_before_the_hint():
    """The hint adds nothing to noise and takes nothing away: whatever
    Whisper made of it unhinted is what the listener gets."""
    model = _Whisper(speech=None)
    model.heard = lambda hint: hint if hint else "Thank you."
    listener = _listener(_cfg(), model)

    assert _text(listener) == "Thank you."


def test_what_is_read_again_is_used_whole():
    """The language and the segments the listener goes on with are those
    of the reading it kept, like the text."""
    model = MagicMock()
    model.transcribe.side_effect = lambda audio, **kwargs: (
        ([SimpleNamespace(text=CITY, avg_logprob=-0.2, no_speech_prob=0.05)],
         SimpleNamespace(language="fr"))
        if kwargs.get("hotwords") else
        ([SimpleNamespace(text="Thank you.", avg_logprob=-0.2, no_speech_prob=0.05)],
         SimpleNamespace(language="en")))
    listener = _listener(_cfg(), model)

    assert listener._transcribe(AUDIO) == ("Thank you.", "en", ["Thank you."])


def test_a_list_repeated_back_with_a_stray_word_is_still_the_list():
    """Measured with fourteen names: the list came back over noise with a
    word of its own tacked on at the end. Nearly every word is still the
    hint's."""
    names = ["Ostrevane", "Pont Sarrel", "Tisseraud", "Brumaille", "Quillomb"]
    model = _Whisper(speech=None, over_noise=lambda hint: hint + ", T.J.")
    listener = _listener(_cfg(whisper_vocabulary=names), model)

    assert _text(listener) == ""


def test_on_a_mac_noise_stays_noise():
    listener = _listener(_cfg(), None)
    listener._whisper_backend = "mlx"

    with patch("jarvis.listening.listener.mlx_whisper", _MlxWhisper(speech=None), create=True):
        assert _text(listener) == ""


def test_a_sentence_that_merely_names_the_city_is_his():
    """One name among other words is the hint doing its job."""
    listener = _listener(_cfg(), _Whisper("Météo {}"))

    assert _text(listener) == f"Météo {CITY}"


def test_his_name_alone_reads_as_it_did_before_the_hint():
    """The price of the guard: an answer that is nothing but one of his
    names cannot be told from the hint repeated back, so it is read
    without the hint."""
    listener = _listener(_cfg(), _Whisper("{}"))

    assert _text(listener) == CITY_MISHEARD


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
        whisper_language=None, whisper_vad=True,
        wake_word="yuba", wake_aliases=["youba"], weather_city=CITY,
        whisper_vocabulary=[],
    ).items():
        setattr(cfg, key, value)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_the_corrected_sentence_is_what_reaches_the_transcript():
    listener = _wired_listener(_mock_cfg(), _Whisper(SPOKEN))

    listener._finalize_utterance()
    listener.state_manager.stop()

    heard = [s.text for s in listener._transcript_buffer.get_since(0)]
    assert heard == [SPOKEN.format(CITY)]


def test_the_language_he_spoke_in_is_still_the_one_last_heard():
    listener = _wired_listener(_mock_cfg(), _Whisper(SPOKEN))

    listener._finalize_utterance()
    listener.state_manager.stop()

    assert listener._last_detected_language == "fr"


def test_noise_in_a_hot_window_never_reaches_the_transcript_as_his_city():
    listener = _wired_listener(_mock_cfg(), _Whisper(speech=None))

    listener._finalize_utterance()
    listener.state_manager.stop()

    assert listener._transcript_buffer.get_since(0) == []
    listener._process_transcript.assert_not_called()


def test_noise_read_again_as_nothing_leaves_the_language_last_heard():
    """Whisper names a language for the hint it wrote back over noise.
    Nothing was said, so the language last heard is still his."""
    listener = _wired_listener(_mock_cfg(), _Whisper(speech=None))
    listener._last_detected_language = "de"

    listener._finalize_utterance()
    listener.state_manager.stop()

    assert listener._last_detected_language == "de"


@patch("builtins.print")
def test_her_sentences_at_the_edges_are_still_left_out_of_what_he_said(_print):
    """Whisper's segments reach the edge check with his names configured
    as they do without: her sentence coming back before his question is
    taken out of it."""
    from test_her_sentences_at_the_edges import (
        _HER_MIDDLE_SENTENCE, _HIS_QUESTION, _her_words_in, _while_she_speaks,
    )
    from test_hot_window_input import _HIS_VOICE, _accepted_query, _level, _whisper_hears

    listener = _while_she_speaks()
    listener.cfg.weather_city = CITY
    listener.cfg.whisper_vocabulary = ["Ostrevane"]

    _whisper_hears(listener, [_HER_MIDDLE_SENTENCE, _HIS_QUESTION],
                   started=time.time() - 3.0, level=_level(listener, _HIS_VOICE))

    query = _accepted_query(listener)
    listener.state_manager.stop()
    assert "hotwords" in listener.model.transcribe.call_args.kwargs
    assert "nantes" in query.lower()
    assert not _her_words_in(query)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_hint_reaches_whisper_on_either_device(device):
    """On CPU the listener stops carrying text from one window to the
    next; the hint is not carried text and survives that."""
    model = _Whisper(SPOKEN)
    listener = _listener(_cfg(), model)
    listener._whisper_device = device

    assert _text(listener) == SPOKEN.format(CITY)


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
        "weather_city": CITY,
        "whisper_vocabulary": ["Ostrevane", " Pont Sarrel "],
    })

    assert whisper_hint(cfg) == f"{CITY}, Ostrevane, Pont Sarrel"


def test_a_vocabulary_that_is_not_a_list_is_read_as_none(tmp_path, monkeypatch):
    cfg = _settings_from(tmp_path, monkeypatch, {"whisper_vocabulary": {"a": 1}})

    assert cfg.whisper_vocabulary == []
