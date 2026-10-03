"""Config keys that have no setting, sitting in a user's config.json.

A config file is written by one release and read by the next, so a file can
carry keys the running release has no setting for. The loader has to take
such a file without complaint, ignore the keys, and never be the one that
writes them. The settings window preserves keys it does not manage, so a
retired key the user typed stays exactly as he typed it until he deletes it
himself.
"""

from __future__ import annotations

import dataclasses
import json
import os
from unittest.mock import patch

import pytest

from jarvis.config import Settings, get_default_config, load_settings

# Keys of a per-turn reply evaluator. The engine has no such evaluator, so these
# keys have no setting and the loader ignores them.
RETIRED_KEYS = {
    "evaluator_model": "pinned-judge:1b",
    "evaluator_enabled": True,
    "evaluator_nudge_max": 5,
}


def _config_file(tmp_path, monkeypatch, values: dict):
    path = tmp_path / "jarvis" / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return path


class TestTheLoaderIgnoresRetiredKeys:
    def test_no_default_names_a_retired_key(self):
        assert not set(RETIRED_KEYS) & set(get_default_config())

    def test_settings_has_no_field_for_a_retired_key(self):
        fields = {f.name for f in dataclasses.fields(Settings)}
        assert not set(RETIRED_KEYS) & fields

    def test_a_file_carrying_them_loads_and_is_left_untouched(
        self, tmp_path, monkeypatch
    ):
        path = _config_file(tmp_path, monkeypatch, {
            "_config_version": 2,
            "ollama_chat_model": "chat:8b",
            "intent_judge_model": "judge:1b",
            **RETIRED_KEYS,
        })
        before = path.read_bytes()

        settings = load_settings()

        assert settings.llm_chat_model == "chat:8b"
        assert settings.intent_judge_model == "judge:1b"
        assert path.read_bytes() == before
        assert not any(hasattr(settings, key) for key in RETIRED_KEYS)

    def test_a_migration_rewrite_never_adds_a_retired_key(
        self, tmp_path, monkeypatch
    ):
        path = _config_file(tmp_path, monkeypatch, {
            "ollama_chat_model": "chat:8b",
        })

        load_settings()

        rewritten = json.loads(path.read_text(encoding="utf-8"))
        assert rewritten["_config_version"] >= 2
        assert not set(RETIRED_KEYS) & set(rewritten)

    def test_a_migration_rewrite_keeps_the_retired_keys_the_user_wrote(
        self, tmp_path, monkeypatch
    ):
        path = _config_file(tmp_path, monkeypatch, {
            "ollama_chat_model": "chat:8b",
            **RETIRED_KEYS,
        })

        load_settings()

        rewritten = json.loads(path.read_text(encoding="utf-8"))
        assert {k: rewritten[k] for k in RETIRED_KEYS} == RETIRED_KEYS

    def test_a_retired_model_pin_does_not_steer_the_max_turn_digest(
        self, tmp_path, monkeypatch
    ):
        """The key has no setting, so it cannot name the model of the
        max-turn digest: the digest follows the intent judge like any other
        small classification-shaped pass."""
        from jarvis.reply.enrichment import digest_loop_for_max_turns

        _config_file(tmp_path, monkeypatch, {
            "_config_version": 2,
            "ollama_chat_model": "chat:8b",
            "intent_judge_model": "judge:1b",
            "evaluator_model": "pinned-judge:1b",
        })
        settings = load_settings()
        loop = [{"role": "assistant", "content": "Let me check that."}]

        with patch(
            "jarvis.reply.enrichment.call_llm_direct", return_value="Done."
        ) as call:
            digest_loop_for_max_turns("what is the weather", loop, settings)

        assert call.call_args.kwargs["chat_model"] == "judge:1b"


class TestTheSettingsWindowLeavesRetiredKeysAlone:
    @pytest.fixture
    def save_from_window(self, qapp, tmp_path, monkeypatch):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        def _save(values: dict) -> dict:
            path = _config_file(tmp_path, monkeypatch, values)
            from desktop_app import settings_window as sw

            window = sw.SettingsWindow()
            with patch.object(sw, "QMessageBox"):
                window._on_save()
            return json.loads(path.read_text(encoding="utf-8"))

        return _save

    def test_a_retired_key_survives_a_save_untouched(self, save_from_window):
        saved = save_from_window({"tts_enabled": False, **RETIRED_KEYS})

        assert {k: saved[k] for k in RETIRED_KEYS} == RETIRED_KEYS
        assert saved["tts_enabled"] is False

    def test_a_save_never_writes_a_retired_key_of_its_own(self, save_from_window):
        saved = save_from_window({"tts_enabled": False})

        assert not set(RETIRED_KEYS) & set(saved)
