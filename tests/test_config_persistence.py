"""Tests for config atomic write, tilde path expansion, and dependencies."""

from pathlib import Path
import json
import os
import pytest
from unittest.mock import patch


class TestConfigAtomicSave:
    def test_save_json_atomic_write_replaces_file(self, tmp_path):
        """_save_json should write to a temp file and replace target atomically."""
        from jarvis.config import _save_json, _load_json

        target = tmp_path / "config.json"
        data = {"tts_engine": "piper", "sample_rate": 16000}

        # First save
        success = _save_json(target, data)
        assert success is True
        assert target.exists()
        loaded = _load_json(target)
        assert loaded == data

        # Overwrite atomically
        new_data = {"tts_engine": "kokoro", "sample_rate": 24000}
        success = _save_json(target, new_data)
        assert success is True
        assert _load_json(target) == new_data

        # Ensure no temp files (.tmp.) left behind in the directory
        leftovers = list(tmp_path.glob("*.tmp*"))
        assert leftovers == []

    def test_save_json_cleans_up_temp_on_write_failure(self, tmp_path, monkeypatch):
        """If writing or serializing fails, temporary file is cleaned up and False returned."""
        from jarvis.config import _save_json

        target = tmp_path / "config.json"

        # Non-serializable object
        bad_data = {"unserializable": object()}
        success = _save_json(target, bad_data)
        assert success is False
        assert not target.exists()

        leftovers = list(tmp_path.glob("*.tmp*"))
        assert leftovers == []


class TestConfigTildePathExpansion:
    def test_load_settings_expands_tilde_in_paths(self, tmp_path, monkeypatch):
        """load_settings should expand tildes in path fields."""
        from jarvis.config import load_settings

        config_file = tmp_path / "config.json"
        config_data = {
            "db_path": "~/custom/jarvis.db",
            "sqlite_vss_path": "~/custom/vss.so",
            "tts_piper_model_path": "~/voices/model.onnx",
            "tts_chatterbox_audio_prompt": "~/prompts/voice.wav",
            "whisper_model": "~/models/whisper-large",
        }
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(config_file))
        settings = load_settings()

        home = str(Path.home())
        assert settings.db_path.startswith(home)
        assert not settings.db_path.startswith("~")
        assert settings.sqlite_vss_path.startswith(home)
        assert not settings.sqlite_vss_path.startswith("~")
        assert settings.tts_piper_model_path.startswith(home)
        assert not settings.tts_piper_model_path.startswith("~")
        assert settings.tts_chatterbox_audio_prompt.startswith(home)
        assert not settings.tts_chatterbox_audio_prompt.startswith("~")
        assert settings.whisper_model.startswith(home)
        assert not settings.whisper_model.startswith("~")

    def test_database_init_expands_tilde(self, tmp_path, monkeypatch):
        """Database constructor should expand tilde in db_path and not create a literal '~' directory."""
        from jarvis.memory.db import Database

        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("USERPROFILE", str(fake_home))
        monkeypatch.setenv("HOME", str(fake_home))

        with patch("pathlib.Path.home", return_value=fake_home):
            db = Database(db_path="~/test_sub/test.db")
            try:
                assert str(fake_home) in db.db_path
                assert not db.db_path.startswith("~")
                assert not Path("~").exists(), "Must not create a literal '~' directory on disk"
            finally:
                db.close()


class TestDependencies:
    def test_requests_version_bumped(self):
        """requirements.txt must specify requests==2.32.4 for the .netrc CVE fix."""
        req_file = Path(__file__).parent.parent / "requirements.txt"
        content = req_file.read_text(encoding="utf-8")
        assert "requests==2.32.4" in content, "requests must be pinned to 2.32.4"
