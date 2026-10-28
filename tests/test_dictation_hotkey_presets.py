"""The dictation hotkey dropdowns: one list of presets, offered by two windows.

The setup wizard and the settings window both let the user pick the dictation
hotkey. They read the same preset list, so a preset added there reaches both,
and the platform default is marked the same way in each.
"""

import sys

import pytest
from PyQt6.QtWidgets import QComboBox

import jarvis.config
from jarvis.config import DICTATION_HOTKEY_PRESETS, _default_dictation_hotkey


def _settings_window_choices():
    from desktop_app.settings_window import _build_field_metadata

    (field,) = [m for m in _build_field_metadata() if m.key == "dictation_hotkey"]
    return list(field.choices)


def _wizard_choices():
    from desktop_app.setup_wizard import DictationPage

    page = DictationPage()
    combo = page.findChild(QComboBox)
    return [(combo.itemData(i), combo.itemText(i)) for i in range(combo.count())]


def test_both_windows_offer_the_same_presets(qapp):
    settings = _settings_window_choices()

    assert settings == _wizard_choices()
    assert [value for value, _ in settings] == list(DICTATION_HOTKEY_PRESETS)


def test_a_preset_added_to_the_list_reaches_both_windows(qapp, monkeypatch):
    added = "ctrl+alt+space"
    monkeypatch.setattr(jarvis.config, "DICTATION_HOTKEY_PRESETS", (*DICTATION_HOTKEY_PRESETS, added))

    assert added in [value for value, _ in _settings_window_choices()]
    assert added in [value for value, _ in _wizard_choices()]


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_the_platform_default_is_the_one_preset_marked_default(platform, monkeypatch):
    from jarvis.dictation.hotkeys import dictation_hotkey_choices

    monkeypatch.setattr(sys, "platform", platform)
    default = _default_dictation_hotkey()
    marked = [value for value, label in dictation_hotkey_choices() if label.endswith("(default)")]

    assert default in DICTATION_HOTKEY_PRESETS
    assert marked == [default]
