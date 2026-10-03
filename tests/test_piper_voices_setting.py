"""`tts_piper_voices` maps a language to the Piper voice that speaks it.

It is read once, at start-up, from a file the user edits by hand or
through the settings window, so a malformed value has to degrade to "no
map" rather than take the daemon down or put a non-path in front of the
engine.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jarvis.config import get_default_config, load_settings


def _settings_from(config: dict):
    with patch("jarvis.config._load_json", return_value={"_config_version": 1, **config}), \
         patch("jarvis.config._save_json", return_value=True):
        return load_settings()


@pytest.mark.unit
def test_no_map_is_the_default():
    assert get_default_config()["tts_piper_voices"] == {}
    assert _settings_from({}).tts_piper_voices == {}


@pytest.mark.unit
def test_the_map_is_read_as_written():
    settings = _settings_from({"tts_piper_voices": {
        "fr": "~/voices/fr_FR-siwis-medium.onnx",
        "de": "de_DE-thorsten-medium",
    }})

    assert settings.tts_piper_voices == {
        "fr": "~/voices/fr_FR-siwis-medium.onnx",
        "de": "de_DE-thorsten-medium",
    }


@pytest.mark.unit
def test_whitespace_around_keys_and_values_is_not_part_of_them():
    settings = _settings_from({"tts_piper_voices": {" fr ": "  fr_FR-siwis-medium "}})

    assert settings.tts_piper_voices == {"fr": "fr_FR-siwis-medium"}


@pytest.mark.unit
@pytest.mark.parametrize("junk", [None, "fr_FR-siwis-medium", ["fr", "de"], 7, True])
def test_a_map_that_is_not_a_map_is_no_map(junk):
    assert _settings_from({"tts_piper_voices": junk}).tts_piper_voices == {}


@pytest.mark.unit
def test_an_entry_that_names_no_language_or_no_voice_is_dropped():
    settings = _settings_from({"tts_piper_voices": {
        "fr": "fr_FR-siwis-medium",
        "": "de_DE-thorsten-medium",
        "es": "",
        "it": None,
        "nl": 3,
    }})

    assert settings.tts_piper_voices == {"fr": "fr_FR-siwis-medium"}
