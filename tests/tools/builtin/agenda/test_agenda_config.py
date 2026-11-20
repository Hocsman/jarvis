"""The agenda is off until the user names where it is read from.

See src/jarvis/tools/builtin/agenda/read_agenda.spec.md, *Where the
agenda is read from*.
"""

from __future__ import annotations

import json

from jarvis.config import get_default_config, load_settings
from jarvis.tools.builtin.agenda import ReadAgendaTool


def _load(tmp_path, monkeypatch, **values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


def test_off_by_default():
    assert get_default_config()["agenda_source"] == ""


def test_with_nothing_in_the_file_the_tool_is_not_offered(tmp_path, monkeypatch):
    settings = _load(tmp_path, monkeypatch)

    assert settings.agenda_source == ""
    assert ReadAgendaTool().offered(settings) is False


def test_a_source_in_the_file_is_carried_trimmed_and_the_tool_is_offered(tmp_path, monkeypatch):
    address = "https://agenda.invented.example/ical/jeton-invente/basic.ics"
    settings = _load(tmp_path, monkeypatch, agenda_source=f"  {address}  ")

    assert settings.agenda_source == address
    assert ReadAgendaTool().offered(settings) is True


def test_a_null_in_the_file_reads_as_no_source(tmp_path, monkeypatch):
    """The settings window writes null for a field emptied by hand."""
    settings = _load(tmp_path, monkeypatch, agenda_source=None)

    assert settings.agenda_source == ""


def test_the_settings_window_has_a_field_for_it():
    from desktop_app.settings_window import FIELD_METADATA

    fields = {field.key: field for field in FIELD_METADATA}

    assert "agenda_source" in fields
