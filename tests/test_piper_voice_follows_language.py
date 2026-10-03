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
    select_tts_voice,
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
@pytest.mark.parametrize(
    "variante,base",
    [("fr-FR", "fr"), ("fr_FR", "fr"), ("pt_BR", "pt"), ("FR", "fr"),
     ("de-AT", "de"), ("zh-TW", "zh"), ("Français-CA", "fr")],
)
def test_a_regional_variant_speaks_with_the_voice_of_its_base_language(variante, base):
    """`fr-FR` is French. The map already reads it that way, so the default
    voice has to as well, or one setting would mean two languages."""
    assert _piper_voice_for_language(base) != PIPER_FALLBACK_VOICE
    assert _piper_voice_for_language(variante) == _piper_voice_for_language(base)


@pytest.mark.unit
@pytest.mark.parametrize("variante", ["ja-JP", "kl_KL", "xx-YY"])
def test_a_variant_of_an_unlisted_language_still_falls_back(variante):
    assert _piper_voice_for_language(variante) == PIPER_FALLBACK_VOICE


@pytest.mark.unit
@pytest.mark.parametrize(
    "langue,base",
    [("fr-FR", "fr"), ("fr_FR", "fr"), ("pt_BR", "pt"), ("de-AT", "de"), ("FR", "fr")],
)
def test_the_map_and_the_default_voice_read_a_variant_as_the_same_language(langue, base):
    """One value, one language: a map keyed on the base language catches the
    variant exactly when the default voice treats it as that language."""
    mapped = select_tts_voice(
        {base: "/v/mapped.onnx"}, "/v/fallback.onnx", response_language=langue)

    assert mapped == "/v/mapped.onnx"
    assert _piper_voice_for_language(langue) != PIPER_FALLBACK_VOICE


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

    regionale = PiperTTS(response_language="fr-FR")
    assert "fr_FR" in regionale._resolved_model_path()

    anglophone = PiperTTS(response_language="english")
    assert "en_" in anglophone._resolved_model_path()


@pytest.mark.unit
def test_an_explicit_model_path_still_wins_over_the_language():
    """A user who names a voice gets that voice, whatever the language says."""
    from jarvis.output.tts import PiperTTS

    moteur = PiperTTS(response_language="français", model_path="/tmp/ma_voix.onnx")
    assert moteur._resolved_model_path() == "/tmp/ma_voix.onnx"
