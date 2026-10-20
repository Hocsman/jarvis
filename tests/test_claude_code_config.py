"""Claude Code delegation is off until the user names a folder, and its
deadline is bounded whatever the file says.

See src/jarvis/projets/projets.spec.md, *Settings*.
"""

from __future__ import annotations

import json

import pytest

from jarvis.config import get_default_config, load_settings


def _load(tmp_path, monkeypatch, **values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    return load_settings()


def test_off_by_default():
    """The feature sends the user's code to a cloud: nothing reaches it
    until a folder is named."""
    assert get_default_config()["claude_code_projects_dir"] == ""


def test_the_settings_carry_what_the_file_says(tmp_path, monkeypatch):
    settings = _load(tmp_path, monkeypatch,
                     claude_code_projects_dir=str(tmp_path / "Projets"),
                     claude_code_command="C:/outils/claude.exe",
                     claude_code_timeout_sec=900)

    assert settings.claude_code_projects_dir == str(tmp_path / "Projets")
    assert settings.claude_code_command == "C:/outils/claude.exe"
    assert settings.claude_code_timeout_sec == 900.0


@pytest.mark.parametrize("given, kept", [(1, 60.0), (10**9, 7200.0)])
def test_the_deadline_is_bounded(tmp_path, monkeypatch, given, kept):
    """Zero would kill every job at birth; unbounded would let a stuck
    agent hold the only slot for good."""
    settings = _load(tmp_path, monkeypatch, claude_code_timeout_sec=given)

    assert settings.claude_code_timeout_sec == kept


def test_an_empty_command_means_claude_and_a_padded_folder_is_trimmed(tmp_path, monkeypatch):
    settings = _load(tmp_path, monkeypatch,
                     claude_code_command="  ",
                     claude_code_projects_dir=f"  {tmp_path / 'Projets'}  ")

    assert settings.claude_code_command == "claude"
    assert settings.claude_code_projects_dir == str(tmp_path / "Projets")
