"""A voice that cannot be fetched must not cost the user her speech.

The default voice follows `response_language`, so an install upgrading into
that behaviour asks for a file it has never held, while the voice it spoke
with yesterday still sits beside it. Going mute rather than reaching for
that one turns a wrong accent into silence, and `_initialized` never resets,
so the silence would last the whole session.
"""

from __future__ import annotations

import sys
import types

import pytest

from jarvis.output.tts import PiperTTS, PIPER_FALLBACK_VOICE


@pytest.fixture
def piper_charge(monkeypatch):
    """Record what `PiperVoice.load` is handed, without a real model."""
    charges = []

    class _Voice:
        config = types.SimpleNamespace(sample_rate=22050)

        @staticmethod
        def load(model_path, config_path):
            charges.append(model_path)
            return _Voice()

    monkeypatch.setitem(
        sys.modules, "piper.voice", types.SimpleNamespace(PiperVoice=_Voice)
    )
    return charges


@pytest.fixture
def cache_avec_repli(tmp_path, monkeypatch):
    """A models directory holding the fallback voice and nothing else."""
    models = tmp_path / "piper"
    models.mkdir()
    for suffixe in (".onnx", ".onnx.json"):
        (models / f"{PIPER_FALLBACK_VOICE}{suffixe}").write_bytes(b"stub")
    monkeypatch.setattr("jarvis.output.tts._get_piper_models_dir", lambda: models)
    return models


@pytest.mark.unit
def test_a_failed_download_falls_back_to_a_voice_already_on_disk(
    cache_avec_repli, piper_charge, monkeypatch
):
    """Offline, she speaks with what she has rather than not at all."""
    tentatives = []

    def echec(voice_name, progress_callback=None):
        tentatives.append(voice_name)
        return None

    monkeypatch.setattr("jarvis.output.tts._download_piper_voice", echec)

    moteur = PiperTTS(enabled=True, response_language="français")
    ok = moteur._ensure_initialized()

    assert tentatives and tentatives[0].startswith("fr_"), "the language voice was tried first"
    assert ok, "the engine stayed usable"
    assert piper_charge and PIPER_FALLBACK_VOICE in piper_charge[0]


@pytest.mark.unit
def test_no_fallback_is_invented_when_nothing_is_cached(tmp_path, piper_charge, monkeypatch):
    """With an empty cache there is nothing to fall back to, and it says so."""
    models = tmp_path / "piper"
    models.mkdir()
    monkeypatch.setattr("jarvis.output.tts._get_piper_models_dir", lambda: models)
    monkeypatch.setattr("jarvis.output.tts._download_piper_voice", lambda *a, **k: None)

    moteur = PiperTTS(enabled=True, response_language="français")

    assert moteur._ensure_initialized() is False
    assert "fr_" in (moteur._init_error or "")


@pytest.mark.unit
def test_a_pinned_voice_is_never_swapped_for_the_fallback(
    cache_avec_repli, piper_charge, monkeypatch
):
    """Naming a voice is a choice; failing to fetch it is no licence to substitute."""
    monkeypatch.setattr("jarvis.output.tts._download_piper_voice", lambda *a, **k: None)

    moteur = PiperTTS(
        enabled=True,
        response_language="français",
        model_path=str(cache_avec_repli / "ma_voix.onnx"),
    )

    assert moteur._ensure_initialized() is False
    assert not piper_charge, "nothing was loaded in place of the pinned voice"
