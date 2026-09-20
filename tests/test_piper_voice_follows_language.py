"""The spoken voice follows the configured language.

Piper voices are trained per language. Reading French with an English voice
does not produce accented French, it produces English phonetics applied to
French words, which is what a user hears as "that is not my language".

The mapping has to be stated rather than derived: Piper names its voices by
locale and speaker (``fr_FR-siwis-medium``), and nothing in the string
"français" yields that. What must not be stated is a preference for one
language over the others, which is why an unlisted language falls back
instead of failing, and why the listed spellings are each language's own
name, its English name and its ISO code, compared without case or accents.
"""

from __future__ import annotations

import pytest

from jarvis.output.tts import (
    PIPER_FALLBACK_VOICE,
    _piper_voice_for_language,
    _get_default_piper_model_path,
)


@pytest.mark.unit
@pytest.mark.parametrize("langue", ["français", "Français", "francais", "FRANCAIS", "fr", "French"])
def test_french_in_any_spelling_gets_the_french_voice(langue):
    """Case and accents are noise; the language underneath is what counts."""
    assert _piper_voice_for_language(langue).startswith("fr_")


@pytest.mark.unit
@pytest.mark.parametrize(
    "langue,prefixe",
    [("español", "es_"), ("deutsch", "de_"), ("italiano", "it_"),
     ("türkçe", "tr_"), ("polski", "pl_"), ("english", "en_")],
)
def test_each_listed_language_gets_its_own_voice(langue, prefixe):
    assert _piper_voice_for_language(langue).startswith(prefixe)


@pytest.mark.unit
@pytest.mark.parametrize("langue", [None, "", "   ", "klingon", "espéranto"])
def test_an_unlisted_language_falls_back_rather_than_failing(langue):
    """Silence would be worse than an accent: an unknown language still speaks."""
    assert _piper_voice_for_language(langue) == PIPER_FALLBACK_VOICE


@pytest.mark.unit
def test_the_default_model_path_carries_the_language_s_voice():
    """The path is what the engine downloads, so the language has to reach it."""
    chemin = _get_default_piper_model_path("français")
    assert chemin.endswith(".onnx")
    assert "fr_FR" in chemin


@pytest.mark.unit
def test_the_engine_resolves_its_voice_from_the_configured_language():
    """End to end: a French install downloads a French voice, not an English one."""
    from jarvis.output.tts import PiperTTS

    moteur = PiperTTS(response_language="français")
    assert "fr_FR" in moteur._resolved_model_path()

    anglophone = PiperTTS(response_language="english")
    assert "en_" in anglophone._resolved_model_path()


@pytest.mark.unit
def test_an_explicit_model_path_still_wins_over_the_language():
    """A user who names a voice gets that voice, whatever the language says."""
    from jarvis.output.tts import PiperTTS

    moteur = PiperTTS(response_language="français", model_path="/tmp/ma_voix.onnx")
    assert moteur._resolved_model_path() == "/tmp/ma_voix.onnx"
