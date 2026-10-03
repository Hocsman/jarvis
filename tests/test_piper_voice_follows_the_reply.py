"""The voice that speaks a reply follows the language of that reply.

`response_language` fixes one voice for the whole session, which serves a
user who has chosen a language and fails one who has not: with it empty
the assistant answers in whatever language she was spoken to, and an
English voice then reads French words with English phonetics.

So the voice is chosen per item, at synthesis time, from a map of
language to voice (`tts_piper_voices`) and the language of the reply. The
pinned `tts_piper_model_path` is the voice for every language the map does
not name. When the two languages in play disagree, the configured one
wins, because it is what the persona prompt tells the model to write in.

The tests run the real engine over a fake `piper` and a fake
`sounddevice`, so what they observe is which voice file was asked to
synthesise, and at what sample rate it was played.
"""

from __future__ import annotations

import os
import sys
import threading
import types

import numpy as np
import pytest

from jarvis.output import tts as tts_module
from jarvis.output.tts import PiperTTS, select_tts_voice


FRENCH = "fr_FR-siwis-medium"
GERMAN = "de_DE-thorsten-medium"
ENGLISH = "en_GB-alan-medium"

RATES = {FRENCH: 22050, GERMAN: 16000, ENGLISH: 24000}


class _World:
    """What the fake audio stack saw."""

    def __init__(self, models):
        self.models = models
        self.synthesised: list[tuple[str, str]] = []  # (voice, text)
        self.stream_rates: list[int] = []
        self.loaded: list[str] = []

    def voice_path(self, name: str) -> str:
        return str(self.models / f"{name}.onnx")

    @property
    def last_voice(self) -> str:
        return self.synthesised[-1][0]


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Three voices on disk, a fake piper that records which one speaks, and
    a fake output stream that records the rate it was opened at. Nothing is
    downloaded: a request for a missing voice fails loudly."""
    models = tmp_path / "piper"
    models.mkdir()
    for nom in (FRENCH, GERMAN, ENGLISH):
        (models / f"{nom}.onnx").write_bytes(b"stub")
        (models / f"{nom}.onnx.json").write_bytes(b"{}")
    seen = _World(models)

    class _Voice:
        def __init__(self, model_path):
            self.name = os.path.basename(model_path).replace(".onnx", "")
            self.config = types.SimpleNamespace(sample_rate=RATES[self.name])

        @staticmethod
        def load(model_path, config_path):
            voice = _Voice(model_path)
            seen.loaded.append(voice.name)
            return voice

        def synthesize(self, text, syn_config):
            seen.synthesised.append((self.name, text))
            return [types.SimpleNamespace(
                audio_int16_array=np.zeros(RATES[self.name] // 10, dtype=np.int16))]

    class _Stream:
        active = False  # playback is over by the time anyone looks

        def __init__(self, samplerate, **kwargs):
            seen.stream_rates.append(samplerate)

        def start(self):
            pass

        def abort(self):
            pass

        def close(self):
            pass

    class _Abort(Exception):
        pass

    class _Stop(Exception):
        pass

    monkeypatch.setitem(sys.modules, "piper", types.ModuleType("piper"))
    monkeypatch.setitem(sys.modules, "piper.voice", types.SimpleNamespace(PiperVoice=_Voice))
    monkeypatch.setitem(sys.modules, "piper.config", types.SimpleNamespace(
        SynthesisConfig=lambda **kwargs: kwargs))
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        OutputStream=_Stream, CallbackAbort=_Abort, CallbackStop=_Stop))

    monkeypatch.setattr(tts_module, "_get_piper_models_dir", lambda: models)

    def _no_network(voice_name, progress_callback=None):
        raise AssertionError(f"tried to download {voice_name}")

    monkeypatch.setattr(tts_module, "_download_piper_voice", _no_network)
    monkeypatch.setattr(PiperTTS, "_notify_speaking_state", lambda self, speaking: None)
    return seen


def _engine(world, voices=None, pinned=ENGLISH, response_language=None):
    return PiperTTS(
        enabled=True,
        model_path=world.voice_path(pinned) if pinned else None,
        voices={code: world.voice_path(nom) for code, nom in (voices or {}).items()},
        response_language=response_language,
    )


# ── The language of the reply picks the voice ──────────────────────────


@pytest.mark.unit
def test_a_french_reply_is_spoken_with_the_french_voice(world):
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN})

    moteur._speak_once("Il fait beau à Bagneux.", language="fr")

    assert world.last_voice == FRENCH


@pytest.mark.unit
def test_a_language_the_map_does_not_name_gets_the_fallback_voice(world):
    """The pinned model is the voice for everything the map leaves out."""
    moteur = _engine(world, voices={"fr": FRENCH}, pinned=ENGLISH)

    moteur._speak_once("Es ist schön heute.", language="de")

    assert world.last_voice == ENGLISH


@pytest.mark.unit
def test_with_no_language_known_the_fallback_speaks(world):
    moteur = _engine(world, voices={"fr": FRENCH}, pinned=ENGLISH)

    moteur._speak_once("Hello.", language=None)

    assert world.last_voice == ENGLISH


@pytest.mark.unit
def test_a_map_entry_beats_the_pinned_model(world):
    """Pinning a voice names the fallback, not a veto: a user who also lists
    a French voice wants French replies in it."""
    moteur = _engine(world, voices={"fr": FRENCH}, pinned=GERMAN)

    moteur._speak_once("Bonjour.", language="fr")
    assert world.last_voice == FRENCH

    moteur._speak_once("Guten Tag.", language="en")
    assert world.last_voice == GERMAN


@pytest.mark.unit
def test_every_voice_is_played_at_its_own_sample_rate(world):
    """Two voices rarely share a rate. Playing a 16 kHz voice at 22 kHz is
    chipmunks, and the exact duration handed to the echo detector would be
    wrong with it."""
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN})
    durees = []
    moteur._duration_callback = durees.append

    moteur._speak_once("Bonjour.", language="fr")
    moteur._speak_once("Guten Tag.", language="de")

    assert world.stream_rates == [RATES[FRENCH], RATES[GERMAN]]
    # Each fake voice makes a tenth of a second of audio at its own rate.
    assert durees == [pytest.approx(0.1), pytest.approx(0.1)]


# ── Which language wins ────────────────────────────────────────────────


@pytest.mark.unit
def test_auto_follows_the_detected_language(world):
    """`response_language` empty is auto: she is expected to answer in the
    language she was spoken to, so that is the language the voice follows."""
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN}, response_language="")

    moteur._speak_once("Bonjour.", language="fr")
    assert world.last_voice == FRENCH

    moteur._speak_once("Guten Tag.", language="de")
    assert world.last_voice == GERMAN


@pytest.mark.unit
def test_the_configured_language_outranks_what_was_detected(world):
    """The persona prompt makes the model write in `response_language`
    whatever language it is spoken to, so detection describes the question
    and not the reply."""
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN}, response_language="deutsch")

    moteur._speak_once("Guten Tag.", language="fr")

    assert world.last_voice == GERMAN


@pytest.mark.unit
def test_the_configured_language_speaks_when_nothing_was_detected(world):
    """A line spoken with no utterance behind it still has a language."""
    moteur = _engine(world, voices={"de": GERMAN}, response_language="Deutsch")

    moteur._speak_once("Guten Tag.", language=None)

    assert world.last_voice == GERMAN


@pytest.mark.unit
@pytest.mark.parametrize("cle", ["fr", "FR", "français", "Francais", "french", "fr-FR", "fr_CA"])
def test_a_map_key_is_a_language_in_any_spelling(world, cle):
    """Keys are written by a person and the language arrives as a Whisper
    code, so both are reduced to one language before they are compared."""
    moteur = PiperTTS(
        enabled=True,
        model_path=world.voice_path(ENGLISH),
        voices={cle: world.voice_path(FRENCH)},
    )

    moteur._speak_once("Bonjour.", language="fr")

    assert world.last_voice == FRENCH


# ── A voice that is not there ──────────────────────────────────────────


@pytest.mark.unit
def test_a_mapped_voice_that_fails_to_download_falls_back_to_the_default(world, monkeypatch):
    monkeypatch.setattr(tts_module, "_download_piper_voice", lambda *a, **k: None)
    moteur = PiperTTS(
        enabled=True,
        model_path=world.voice_path(ENGLISH),
        voices={"fr": str(world.models / "fr_FR-absente-medium.onnx")},
    )

    moteur._speak_once("Bonjour.", language="fr")

    assert world.last_voice == ENGLISH


@pytest.mark.unit
def test_a_failed_voice_is_not_fetched_again_on_every_sentence(world, monkeypatch):
    """A fetch that fails retries a rate-limited server for half a minute.
    Doing that per sentence would hold the reply hostage."""
    tentatives = []
    monkeypatch.setattr(
        tts_module, "_download_piper_voice",
        lambda nom, progress_callback=None: tentatives.append(nom) or None)
    moteur = PiperTTS(
        enabled=True,
        model_path=world.voice_path(ENGLISH),
        voices={"fr": str(world.models / "fr_FR-absente-medium.onnx")},
    )

    for _ in range(3):
        moteur._speak_once("Bonjour.", language="fr")

    assert len(tentatives) == 1


# ── Choosing, without an engine ────────────────────────────────────────


@pytest.mark.unit
def test_select_tts_voice_names_the_voice_for_the_reply_language(tmp_path):
    voices = {"fr": "/v/fr.onnx", "de": "/v/de.onnx"}

    assert select_tts_voice(voices, "/v/en.onnx", detected_language="de") == "/v/de.onnx"
    assert select_tts_voice(voices, "/v/en.onnx", detected_language="it") == "/v/en.onnx"
    assert select_tts_voice({}, "/v/en.onnx", detected_language="fr") == "/v/en.onnx"


@pytest.mark.unit
def test_select_tts_voice_prefers_the_configured_language_to_the_detected_one():
    voices = {"fr": "/v/fr.onnx", "de": "/v/de.onnx"}

    chosen = select_tts_voice(
        voices, "/v/en.onnx", response_language="français", detected_language="de")

    assert chosen == "/v/fr.onnx"


@pytest.mark.unit
@pytest.mark.parametrize("vide", [None, "", "   "])
def test_an_empty_configured_language_is_auto(vide):
    voices = {"de": "/v/de.onnx"}

    chosen = select_tts_voice(
        voices, "/v/en.onnx", response_language=vide, detected_language="de")

    assert chosen == "/v/de.onnx"


@pytest.mark.unit
def test_a_language_outside_the_table_is_keyed_by_its_iso_code():
    """Whisper reports `ja`, not "Japanese". A language the built-in table
    does not list has no names to recognise, so only its ISO 639-1 code (or
    a regional variant of it) meets what the recogniser reports."""
    fallback = "/v/en.onnx"

    by_name = select_tts_voice({"Japanese": "/v/ja.onnx"}, fallback, detected_language="ja")
    by_code = select_tts_voice({"ja": "/v/ja.onnx"}, fallback, detected_language="ja")
    by_variant = select_tts_voice({"ja-JP": "/v/ja.onnx"}, fallback, detected_language="ja")

    assert by_name == fallback
    assert by_code == "/v/ja.onnx"
    assert by_variant == "/v/ja.onnx"


@pytest.mark.unit
def test_a_bare_voice_name_is_a_file_in_the_models_directory(world):
    """A path in a config file is a chore to type and to move between
    machines. A bare name is what Piper calls the voice, and it keeps
    working through the auto-download: the engine fetches by name."""
    chosen = select_tts_voice({"fr": FRENCH}, "/v/en.onnx", detected_language="fr")

    assert chosen == str(world.models / f"{FRENCH}.onnx")


def _voice_lines(lines, *noms):
    """The lines that name a voice as the one chosen, not the ones that
    report a model file being loaded."""
    return [
        line for line in lines
        if any(nom in line for nom in noms) and "loading" not in line.lower()
    ]


@pytest.mark.unit
def test_the_choice_is_named_in_the_debug_log(world, monkeypatch):
    """The model file being loaded is logged for its own reasons and names the
    voice too, so only the lines that are not about loading count here."""
    lines = []
    monkeypatch.setattr(tts_module, "debug_log", lambda msg, *a, **k: lines.append(msg))
    moteur = _engine(world, voices={"fr": FRENCH})

    moteur._speak_once("Bonjour.", language="fr")

    assert _voice_lines(lines, FRENCH), lines


@pytest.mark.unit
def test_a_reply_of_many_sentences_names_its_voice_once(world, monkeypatch):
    """A streamed reply reaches the engine a sentence at a time. One line per
    sentence would bury the log; the voice is news only when it changes."""
    lines = []
    monkeypatch.setattr(tts_module, "debug_log", lambda msg, *a, **k: lines.append(msg))
    moteur = _engine(world, voices={"fr": FRENCH})

    for phrase in ("Premiere phrase.", "Deuxieme phrase.", "Troisieme phrase."):
        moteur._speak_once(phrase, language="fr")

    assert len(_voice_lines(lines, FRENCH)) == 1, lines


@pytest.mark.unit
def test_a_change_of_voice_is_named_again(world, monkeypatch):
    lines = []
    monkeypatch.setattr(tts_module, "debug_log", lambda msg, *a, **k: lines.append(msg))
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN})

    moteur._speak_once("Bonjour.", language="fr")
    moteur._speak_once("Guten Tag.", language="de")
    moteur._speak_once("Encore un mot.", language="fr")

    assert len(_voice_lines(lines, FRENCH)) == 2, lines
    assert len(_voice_lines(lines, GERMAN)) == 1, lines


# ── From speak() to the worker ─────────────────────────────────────────


@pytest.mark.unit
def test_speak_carries_the_language_to_the_voice_that_says_it(world):
    """The language is a property of the item, not of the engine: sentences
    of two replies in two languages can sit in the queue together."""
    moteur = _engine(world, voices={"fr": FRENCH, "de": GERMAN})
    fini = threading.Event()
    try:
        moteur.start()
        moteur.speak("Bonjour tout le monde.", language="fr")
        moteur.speak("Guten Tag zusammen.", completion_callback=fini.set, language="de")
        assert fini.wait(10), "the queue never reached its end"
    finally:
        moteur.stop()

    assert [voix for voix, _ in world.synthesised] == [FRENCH, GERMAN]


# ── From the config file to the engine ─────────────────────────────────
#
# Every test above builds the engine by hand. What the user touches is the
# config file, and a map that is read, shown in the settings window and
# then dropped on the way to the engine would leave all of them green.


def _settings_from(config: dict):
    from unittest.mock import patch

    from jarvis.config import load_settings

    with patch("jarvis.config._load_json", return_value={"_config_version": 1, **config}), \
         patch("jarvis.config._save_json", return_value=True):
        return load_settings()


@pytest.mark.unit
def test_the_factory_hands_the_map_to_the_piper_engine(world):
    moteur = tts_module.create_tts_engine(
        engine="piper",
        piper_model_path=world.voice_path(ENGLISH),
        piper_voices={"fr": world.voice_path(FRENCH)},
    )

    moteur._speak_once("Bonjour.", language="fr")
    assert world.last_voice == FRENCH

    moteur._speak_once("Hello.", language="en")
    assert world.last_voice == ENGLISH


@pytest.mark.unit
def test_a_voice_listed_in_the_config_file_is_the_one_the_daemon_speaks_with(world):
    from jarvis import daemon

    reglages = _settings_from({
        "tts_engine": "piper",
        "tts_enabled": True,
        "tts_piper_model_path": world.voice_path(ENGLISH),
        "tts_piper_voices": {"fr": world.voice_path(FRENCH), "de": world.voice_path(GERMAN)},
    })

    moteur = daemon.build_tts_engine(reglages)

    moteur._speak_once("Bonjour.", language="fr")
    assert world.last_voice == FRENCH
    moteur._speak_once("Guten Tag.", language="de")
    assert world.last_voice == GERMAN
    moteur._speak_once("Hello.", language="en")
    assert world.last_voice == ENGLISH


@pytest.mark.unit
def test_the_configured_response_language_reaches_the_engine_from_the_config_file(world):
    """The language that wins over detection is read from the same file, so
    it has to travel the same road."""
    from jarvis import daemon

    reglages = _settings_from({
        "tts_engine": "piper",
        "tts_enabled": True,
        "response_language": "german",
        "tts_piper_model_path": world.voice_path(ENGLISH),
        "tts_piper_voices": {"fr": world.voice_path(FRENCH), "de": world.voice_path(GERMAN)},
    })

    moteur = daemon.build_tts_engine(reglages)

    moteur._speak_once("Guten Tag.", language="fr")
    assert world.last_voice == GERMAN
