"""The settings window can edit a mapping, and uses it for the Piper voices.

Every other field in the metadata is a scalar or a list of strings, so a
language-to-voice map had nothing to be edited with. The `map` field type
edits a dict as one `key -> value` row per entry, the same shape the
custom dictionary already uses, and stores it in `config.json` as the
object the engine reads.

Nothing here opens a window: the conversions are plain functions, and value
extraction is exercised through stand-ins for the Qt list widget.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from desktop_app.settings_window import (
    CATEGORIES,
    FIELD_METADATA,
    SettingsWindow,
    _is_default_value,
    _map_to_rows,
    _rows_to_map,
)
from jarvis.config import get_default_config


def _field(key):
    return next(fm for fm in FIELD_METADATA if fm.key == key)


@pytest.mark.unit
def test_the_voice_map_is_offered_with_the_piper_settings():
    fm = _field("tts_piper_voices")

    assert fm.field_type == "map"
    assert fm.category == "piper"
    assert fm.category in {key for key, _ in CATEGORIES}


@pytest.mark.unit
def test_the_voice_map_default_is_an_empty_map():
    assert get_default_config()["tts_piper_voices"] == {}


@pytest.mark.unit
def test_an_emptied_voice_map_is_not_written_to_the_config_file():
    """The minimal-config rule: a value equal to the default is omitted."""
    assert _is_default_value({}, get_default_config()["tts_piper_voices"])


@pytest.mark.unit
def test_a_map_is_shown_as_one_row_per_entry():
    rows = _map_to_rows({"fr": "fr_FR-siwis-medium", "de": "de_DE-thorsten-medium"})

    assert rows == ["fr -> fr_FR-siwis-medium", "de -> de_DE-thorsten-medium"]


@pytest.mark.unit
@pytest.mark.parametrize("junk", [None, "", [], 3, ["fr"]])
def test_a_value_that_is_not_a_map_shows_no_rows(junk):
    assert _map_to_rows(junk) == []


@pytest.mark.unit
def test_rows_become_the_map_they_were_made_from():
    original = {"fr": "fr_FR-siwis-medium", "de": r"C:\voices\de.onnx"}

    assert _rows_to_map(_map_to_rows(original)) == original


@pytest.mark.unit
def test_rows_tolerate_spacing_around_the_arrow():
    assert _rows_to_map(["fr->a", "  de  ->  b  "]) == {"fr": "a", "de": "b"}


@pytest.mark.unit
def test_a_row_without_a_language_and_a_voice_is_dropped():
    rows = ["fr -> a", "no arrow here", " -> b", "es ->", "-> ", ""]

    assert _rows_to_map(rows) == {"fr": "a"}


@pytest.mark.unit
def test_a_value_may_itself_contain_an_arrow_character_run():
    """Only the first arrow separates: paths and names can contain `>`."""
    assert _rows_to_map(["fr -> a->b"]) == {"fr": "a->b"}


@pytest.mark.unit
def test_a_language_listed_twice_keeps_its_last_voice():
    assert _rows_to_map(["fr -> old", "fr -> new"]) == {"fr": "new"}


def _stand_in_window(fm, rows):
    """What `_get_value` and `_set_widget_value` touch, without Qt."""
    texts = list(rows)
    list_widget = MagicMock()
    list_widget.count.side_effect = lambda: len(texts)
    list_widget.item.side_effect = lambda i: SimpleNamespace(text=lambda: texts[i])
    list_widget.clear.side_effect = texts.clear
    list_widget.addItem.side_effect = texts.append
    container = SimpleNamespace(_list_widget=list_widget)
    return SimpleNamespace(_widgets={fm.key: container}), texts


@pytest.mark.unit
def test_the_window_reads_the_rows_back_as_a_map():
    fm = _field("tts_piper_voices")
    window, _ = _stand_in_window(fm, ["fr -> fr_FR-siwis-medium", "de -> de_DE-thorsten-medium"])

    value = SettingsWindow._get_value(window, fm)

    assert value == {"fr": "fr_FR-siwis-medium", "de": "de_DE-thorsten-medium"}


@pytest.mark.unit
def test_reset_puts_the_default_map_back_in_the_widget():
    fm = _field("tts_piper_voices")
    window, texts = _stand_in_window(fm, ["fr -> fr_FR-siwis-medium"])

    SettingsWindow._set_widget_value(window, fm, {"de": "de_DE-thorsten-medium"})
    assert texts == ["de -> de_DE-thorsten-medium"]

    SettingsWindow._set_widget_value(window, fm, {})
    assert texts == []
