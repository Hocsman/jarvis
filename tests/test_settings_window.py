"""
Tests for settings window metadata and config I/O logic.

Tests verify the metadata registry, value extraction, and save/load behaviour.
Most run without the GUI, using mock Qt objects where needed; the save round
trips open a real window on the offscreen Qt platform.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from desktop_app.settings_window import (
    FIELD_METADATA,
    CATEGORIES,
    FieldMeta,
    get_input_devices,
    _build_field_metadata,
    _is_default_value,
    _MCPCatalogueDialog,
    _MCPEditDialog,
)
from desktop_app.mcp_catalogue import CATALOGUE_BY_NAME
from jarvis.config import get_default_config


class TestFieldMetadata:
    """Tests for the config field metadata registry."""

    def test_all_fields_reference_valid_categories(self):
        """Every field's category must appear in CATEGORIES."""
        valid_cats = {key for key, _ in CATEGORIES}
        for fm in FIELD_METADATA:
            assert fm.category in valid_cats, (
                f"Field '{fm.key}' references unknown category '{fm.category}'"
            )

    def test_all_fields_reference_existing_config_keys(self):
        """Every field key must exist in get_default_config()."""
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            assert fm.key in defaults, (
                f"Field '{fm.key}' not found in default config"
            )

    def test_no_duplicate_keys(self):
        """Each config key should appear at most once in the metadata."""
        keys = [fm.key for fm in FIELD_METADATA]
        assert len(keys) == len(set(keys)), (
            f"Duplicate keys: {[k for k in keys if keys.count(k) > 1]}"
        )

    def test_field_types_are_valid(self):
        """All field_type values must be from the allowed set."""
        valid_types = {"bool", "int", "float", "str", "choice", "device", "list", "map", "password"}
        for fm in FIELD_METADATA:
            assert fm.field_type in valid_types, (
                f"Field '{fm.key}' has invalid type '{fm.field_type}'"
            )

    def test_choice_fields_have_choices(self):
        """Fields with type 'choice' must have a non-empty choices list."""
        for fm in FIELD_METADATA:
            if fm.field_type == "choice":
                assert fm.choices and len(fm.choices) > 0, (
                    f"Choice field '{fm.key}' has no choices defined"
                )

    def test_numeric_fields_have_bounds(self):
        """Numeric fields (int/float) should have min and max defined."""
        for fm in FIELD_METADATA:
            if fm.field_type in ("int", "float") and not fm.nullable:
                assert fm.min_val is not None, (
                    f"Numeric field '{fm.key}' missing min_val"
                )
                assert fm.max_val is not None, (
                    f"Numeric field '{fm.key}' missing max_val"
                )

    def test_labels_are_nonempty(self):
        """Every field must have a non-empty label."""
        for fm in FIELD_METADATA:
            assert fm.label.strip(), f"Field '{fm.key}' has empty label"

    def test_descriptions_are_nonempty(self):
        """Every field must have a non-empty description."""
        for fm in FIELD_METADATA:
            assert fm.description.strip(), f"Field '{fm.key}' has empty description"

    def test_build_returns_consistent_results(self):
        """_build_field_metadata() should return the same structure on repeated calls."""
        a = _build_field_metadata()
        b = _build_field_metadata()
        assert len(a) == len(b)
        for fa, fb in zip(a, b):
            assert fa.key == fb.key
            assert fa.category == fb.category

    def test_low_power_mode_is_exposed_as_feature_toggle(self):
        """Low-power mode should be available without hand-editing config.json."""
        field = next((fm for fm in FIELD_METADATA if fm.key == "low_power_mode"), None)
        assert field is not None
        assert field.category == "features"
        assert field.field_type == "bool"


class TestLLMProviderFields:
    """The settings UI must expose the provider-aware LLM config so a user
    can select an OpenAI-compatible backend without editing config.json by
    hand."""

    def _field(self, key):
        for fm in FIELD_METADATA:
            if fm.key == key:
                return fm
        return None

    def test_provider_category_present(self):
        """A dedicated 'LLM Provider' category must exist in the sidebar."""
        cat_keys = [k for k, _ in CATEGORIES]
        assert "llm_provider" in cat_keys

    def test_provider_fields_present(self):
        """All eight provider-aware config keys are surfaced."""
        expected = {
            "llm_provider", "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_provider", "embedding_base_url", "embedding_api_key",
            "embedding_model",
        }
        present = {fm.key for fm in FIELD_METADATA}
        missing = expected - present
        assert not missing, f"Provider fields missing from settings UI: {missing}"

    def test_provider_fields_live_in_provider_category(self):
        """The provider connection/credential fields group under the
        'LLM Provider' category, not scattered across 'llm'."""
        for key in (
            "llm_provider", "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_provider", "embedding_base_url", "embedding_api_key",
            "embedding_model",
        ):
            fm = self._field(key)
            assert fm is not None and fm.category == "llm_provider", (
                f"'{key}' should be in the 'llm_provider' category"
            )

    def test_llm_provider_choices_match_config(self):
        """The provider dropdown offers exactly the values the config loader
        accepts."""
        from jarvis.config import LLM_PROVIDERS

        fm = self._field("llm_provider")
        assert fm is not None and fm.field_type == "choice"
        values = {v for v, _ in (fm.choices or [])}
        assert values == set(LLM_PROVIDERS)

    def test_the_key_can_live_in_an_environment_variable(self):
        """Naming the variable keeps the secret out of config.json; the
        field holds a name, not a secret, so it is plain text."""
        fm = self._field("llm_api_key_env")
        assert fm is not None
        assert fm.category == "llm_provider"
        assert fm.field_type == "str"

    def test_embedding_provider_offers_inherit_option(self):
        """embedding_provider includes the empty 'same as chat provider'
        option plus the two concrete providers."""
        fm = self._field("embedding_provider")
        assert fm is not None and fm.field_type == "choice"
        values = {v for v, _ in (fm.choices or [])}
        assert "" in values, "must offer an inherit-from-chat-provider option"
        assert {"ollama", "openai_compatible"} <= values

    def test_api_key_fields_are_password_type(self):
        """API keys must use the password field type so they render masked."""
        for key in ("llm_api_key", "embedding_api_key"):
            fm = self._field(key)
            assert fm is not None and fm.field_type == "password", (
                f"'{key}' should be a password field"
            )

    def test_model_fields_are_freetext(self):
        """The provider model fields are free text — an OpenAI-compatible
        server's model name is not in the Ollama SUPPORTED_CHAT_MODELS
        catalogue, so a choice dropdown would be wrong."""
        for key in ("llm_chat_model", "embedding_model"):
            fm = self._field(key)
            assert fm is not None and fm.field_type == "str", (
                f"'{key}' should be a free-text str field"
            )

    def test_connection_fields_are_nullable(self):
        """Connection/credential/model fields are nullable so leaving them
        empty falls back to the Ollama settings and keeps config.json minimal."""
        for key in (
            "llm_base_url", "llm_api_key", "llm_chat_model",
            "embedding_base_url", "embedding_api_key", "embedding_model",
        ):
            fm = self._field(key)
            assert fm is not None and fm.nullable, f"'{key}' should be nullable"


class TestKokoroFields:
    """Kokoro's voice and language code are a pair: the pipeline is built
    with ``lang_code`` and a voice whose prefix belongs to another language
    cannot synthesise, so a GUI-only user must be able to set both."""

    def _field(self, key):
        for fm in FIELD_METADATA:
            if fm.key == key:
                return fm
        return None

    def test_kokoro_category_exposes_voice_language_and_speed(self):
        for key in ("tts_kokoro_voice", "tts_kokoro_lang_code", "tts_kokoro_speed"):
            fm = self._field(key)
            assert fm is not None and fm.category == "kokoro", (
                f"'{key}' should be exposed in the 'kokoro' category"
            )

    def test_kokoro_lang_code_default_matches_config(self):
        fm = self._field("tts_kokoro_lang_code")
        assert fm is not None and fm.field_type == "str"
        assert get_default_config()["tts_kokoro_lang_code"] == "f"


class TestMinimalConfigInvariant:
    """``_is_default_value`` decides whether a field is omitted from
    config.json. An emptied nullable provider field (reads back as None)
    whose default is an empty string must be omitted, not persisted as null."""

    def test_value_equal_to_default_is_omitted(self):
        assert _is_default_value("ollama", "ollama") is True

    def test_changed_value_is_kept(self):
        assert _is_default_value("openai_compatible", "ollama") is False

    def test_emptied_field_with_empty_string_default_is_omitted(self):
        # llm_base_url etc.: default "", user clears it -> _get_value returns None
        assert _is_default_value(None, "") is True

    def test_emptied_field_with_none_default_is_omitted(self):
        assert _is_default_value(None, None) is True

    def test_set_value_over_empty_default_is_kept(self):
        assert _is_default_value("http://localhost:1234/v1", "") is False

    def test_none_over_nonempty_default_is_kept(self):
        # A nullable field whose default is a real value, cleared by the user,
        # is a genuine change and must be written.
        assert _is_default_value(None, "gemma4:e2b") is False


class TestCategories:
    """Tests for category definitions."""

    def test_no_duplicate_category_keys(self):
        """Category keys should be unique."""
        keys = [k for k, _ in CATEGORIES]
        assert len(keys) == len(set(keys))

    def test_every_category_has_fields(self):
        """Every defined category should have at least one field.

        The 'mcps' category uses a custom page, not FIELD_METADATA, so it's excluded.
        """
        cats_with_fields = {fm.category for fm in FIELD_METADATA}
        custom_page_categories = {"mcps"}
        for key, label in CATEGORIES:
            if key in custom_page_categories:
                continue
            assert key in cats_with_fields, (
                f"Category '{key}' ({label}) has no fields"
            )

    def test_mcps_category_exists(self):
        """The MCP Servers category must be present in the sidebar."""
        cat_keys = [k for k, _ in CATEGORIES]
        assert "mcps" in cat_keys


class TestInputDevices:
    """Tests for audio device enumeration."""

    def test_always_includes_system_default(self):
        """get_input_devices() always returns at least the system default."""
        # Even if sounddevice fails, we should get the default option
        with patch.dict("sys.modules", {"sounddevice": None}):
            devices = get_input_devices()
        assert len(devices) >= 1
        assert devices[0][0] == ""  # empty string = system default

    def test_with_mock_sounddevice(self):
        """With mock devices, returns them plus system default."""
        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = [
            {"name": "Built-in Mic", "max_input_channels": 2, "default_samplerate": 44100},
            {"name": "USB Speaker", "max_input_channels": 0, "default_samplerate": 48000},
            {"name": "External Mic", "max_input_channels": 1, "default_samplerate": 16000},
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Need to reimport to pick up the mock
            import importlib
            import desktop_app.settings_window as sw
            importlib.reload(sw)
            devices = sw.get_input_devices()

        # System default + 2 input devices (USB Speaker has 0 input channels)
        assert len(devices) == 3
        assert devices[0][0] == ""
        assert "Built-in Mic" in devices[1][1]
        assert "External Mic" in devices[2][1]

    def test_handles_sounddevice_import_error(self):
        """Gracefully handles missing sounddevice."""
        devices = get_input_devices()
        # Should always at least have the default
        assert len(devices) >= 1


class TestConfigSaveLogic:
    """Tests for save/load round-trip behaviour."""

    def test_only_non_defaults_are_saved(self):
        """Saving default values should produce an empty config file."""
        defaults = get_default_config()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            f.write('{}')
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            # Simulate: all values match defaults, so nothing should be written
            config = {}
            for fm in FIELD_METADATA:
                val = defaults.get(fm.key)
                default_val = defaults.get(fm.key)
                if val != default_val:
                    config[fm.key] = val

            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert saved == {}
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_changed_values_are_preserved(self):
        """Non-default values should survive a save/load round-trip."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            f.write('{}')
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            config = {
                "ollama_chat_model": "gemma4:e4b",
                "tts_enabled": False,
                "hot_window_seconds": 5.0,
            }
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert saved["ollama_chat_model"] == "gemma4:e4b"
            assert saved["tts_enabled"] is False
            assert saved["hot_window_seconds"] == 5.0
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_unknown_keys_preserved_on_save(self):
        """Keys not in FIELD_METADATA (e.g. mcps) should survive save."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            json.dump({"mcps": {"test": {"url": "http://example.com"}},
                        "_config_version": 1}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            existing = _load_json(cfg_path)
            # Simulate settings save: add a changed value, keep existing keys
            existing["tts_enabled"] = False
            _save_json(cfg_path, existing)

            saved = _load_json(cfg_path)
            assert "mcps" in saved
            assert saved["mcps"]["test"]["url"] == "http://example.com"
            assert saved["_config_version"] == 1
            assert saved["tts_enabled"] is False
        finally:
            cfg_path.unlink(missing_ok=True)


REMOTE_PINS = ["google/gemini-3.1-flash-lite", "claude-haiku-5-5"]

NAMED_MIC = "MSI GV60"
AUDIO_DEVICES = [
    {"name": "Speakers (USB2.0)", "max_input_channels": 0},
    {"name": f"Microphone ({NAMED_MIC})", "max_input_channels": 1},
    {"name": "Microphone Array (Realtek)", "max_input_channels": 2},
]


def _a_value_not_in(fm: FieldMeta):
    """A value the choice field does not list, of the type its default has."""
    if isinstance(get_default_config()[fm.key], int):
        return max(int(value) for value, _ in fm.choices) + 1
    return "pinned-by-hand"


class TestASaveKeepsWhatTheWindowDoesNotList:
    """A value the window has no option for, such as a model served by a
    remote provider, is written back exactly as it was read when any setting
    is saved. Falling back to the first option would move a pin the user
    never touched, without a word."""

    @pytest.fixture
    def settings_file(self, qapp, tmp_path, monkeypatch):
        from desktop_app import settings_window as sw

        path = tmp_path / "jarvis" / "config.json"
        path.parent.mkdir(parents=True)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # The window resolves the path through its own import of
        # default_config_path, which TestInputDevices' module reload can
        # leave bound to another test's sandbox, so patch it there too.
        monkeypatch.setattr(sw, "default_config_path", lambda: path)
        return path

    @staticmethod
    def _open(path: Path, values: dict):
        from desktop_app import settings_window as sw

        path.write_text(json.dumps(values), encoding="utf-8")
        return sw.SettingsWindow()

    @staticmethod
    def _save(window, path: Path) -> dict:
        from desktop_app import settings_window as sw

        with patch.object(sw, "QMessageBox"):
            window._on_save()
        return json.loads(path.read_text(encoding="utf-8"))

    @pytest.mark.parametrize("pin", REMOTE_PINS)
    def test_a_remote_judge_pin_survives_saving_another_setting(self, settings_file, pin):
        window = self._open(settings_file, {"intent_judge_model": pin})
        window._widgets["low_power_mode"].setChecked(True)

        saved = self._save(window, settings_file)

        assert saved["intent_judge_model"] == pin
        assert saved["low_power_mode"] is True

    @pytest.mark.parametrize("pin", REMOTE_PINS)
    def test_the_window_shows_the_pin_it_keeps(self, settings_file, pin):
        window = self._open(settings_file, {"intent_judge_model": pin})

        assert window._widgets["intent_judge_model"].currentText() == pin

    def test_every_model_key_keeps_a_remote_pin(self, settings_file):
        pins = {
            key: REMOTE_PINS[0]
            for key in get_default_config()
            if key.endswith("_model")
        }

        saved = self._save(self._open(settings_file, pins), settings_file)

        assert {key: saved.get(key) for key in pins} == pins

    def test_every_choice_field_keeps_a_value_it_does_not_list(self, settings_file):
        unlisted = {
            fm.key: _a_value_not_in(fm)
            for fm in FIELD_METADATA
            if fm.field_type == "choice"
        }

        saved = self._save(self._open(settings_file, unlisted), settings_file)

        assert {key: saved.get(key) for key in unlisted} == unlisted

    def test_an_unlisted_value_of_another_type_keeps_its_value(self, settings_file):
        """A number in a text field stays that number, and a sample rate
        written as a float is saved as the rate it names."""
        odd = {}
        for fm in FIELD_METADATA:
            if fm.field_type != "choice":
                continue
            unlisted = _a_value_not_in(fm)
            odd[fm.key] = float(unlisted) if isinstance(unlisted, int) else 3

        saved = self._save(self._open(settings_file, odd), settings_file)

        assert {key: saved.get(key) for key in odd} == odd

    @pytest.mark.parametrize("empty", ["", None])
    def test_an_empty_pin_is_never_kept_as_a_blank_option(self, settings_file, empty):
        fm = next(fm for fm in FIELD_METADATA if fm.key == "intent_judge_model")
        listed = [value for value, _ in fm.choices]
        default = get_default_config()["intent_judge_model"]
        window = self._open(settings_file, {"intent_judge_model": empty})
        combo = window._widgets["intent_judge_model"]

        saved = self._save(window, settings_file)

        assert [combo.itemData(i) for i in range(combo.count())] == listed
        assert saved.get("intent_judge_model", default) in listed

    def test_a_listed_model_still_replaces_the_pin(self, settings_file):
        from jarvis.config import SUPPORTED_CHAT_MODELS

        default = get_default_config()["intent_judge_model"]
        listed = next(model for model in SUPPORTED_CHAT_MODELS if model != default)
        window = self._open(settings_file, {"intent_judge_model": REMOTE_PINS[0]})
        combo = window._widgets["intent_judge_model"]
        combo.setCurrentIndex(combo.findData(listed))

        saved = self._save(window, settings_file)

        assert saved["intent_judge_model"] == listed

    def test_choosing_the_default_drops_the_pin(self, settings_file):
        default = get_default_config()["intent_judge_model"]
        window = self._open(settings_file, {"intent_judge_model": REMOTE_PINS[0]})
        combo = window._widgets["intent_judge_model"]
        combo.setCurrentIndex(combo.findData(default))

        saved = self._save(window, settings_file)

        assert "intent_judge_model" not in saved

    def test_reset_drops_the_pin(self, settings_file):
        from desktop_app import settings_window as sw

        window = self._open(settings_file, {"intent_judge_model": REMOTE_PINS[0]})
        with patch.object(
            sw.QMessageBox, "question",
            return_value=sw.QMessageBox.StandardButton.Yes,
        ):
            window._on_reset()

        saved = self._save(window, settings_file)

        assert "intent_judge_model" not in saved

    @pytest.fixture
    def microphones(self, monkeypatch):
        sd = MagicMock()
        sd.query_devices.return_value = AUDIO_DEVICES
        monkeypatch.setitem(sys.modules, "sounddevice", sd)
        return sd

    @pytest.mark.parametrize("plugged_in", [True, False])
    def test_a_microphone_set_by_name_survives_saving_another_setting(
        self, settings_file, microphones, plugged_in,
    ):
        """The listener finds a microphone by name, and a name stays true
        across a replug, so the window keeps one whether or not it is
        plugged in when the window opens."""
        if not plugged_in:
            microphones.query_devices.return_value = [
                dev for dev in AUDIO_DEVICES if NAMED_MIC not in dev["name"]
            ]
        window = self._open(settings_file, {"voice_device": NAMED_MIC})
        window._widgets["low_power_mode"].setChecked(True)

        saved = self._save(window, settings_file)

        assert saved["voice_device"] == NAMED_MIC
        assert saved["low_power_mode"] is True

    def test_the_window_shows_the_name_apart_from_the_devices_it_lists(
        self, settings_file, microphones,
    ):
        listed = [display for _, display in get_input_devices()]
        window = self._open(settings_file, {"voice_device": NAMED_MIC})
        combo = window._widgets["voice_device"]

        assert NAMED_MIC in combo.currentText()
        assert combo.currentText() not in listed

    @pytest.mark.parametrize("index", ["7", 7, "0"])
    def test_an_index_the_list_lacks_falls_back_to_system_default(
        self, settings_file, microphones, index,
    ):
        """An index can name another device after a replug, so the window
        does not keep one it cannot show. Device 0 is output-only."""
        window = self._open(settings_file, {"voice_device": index})
        combo = window._widgets["voice_device"]

        assert combo.currentData() == ""
        assert "voice_device" not in self._save(window, settings_file)

    @pytest.mark.parametrize("index", ["1", 1])
    def test_a_listed_index_is_selected_and_kept(self, settings_file, microphones, index):
        shown = dict(get_input_devices())[str(index)]
        window = self._open(settings_file, {"voice_device": index})
        combo = window._widgets["voice_device"]

        assert combo.currentText() == shown
        assert int(self._save(window, settings_file)["voice_device"]) == int(index)

    @pytest.mark.parametrize("default", ["default", "System", "  "])
    def test_a_word_for_the_system_default_shows_as_system_default(
        self, settings_file, microphones, default,
    ):
        """The config reads these as the system default, so the window shows
        System Default rather than a microphone of that name."""
        window = self._open(settings_file, {"voice_device": default})
        combo = window._widgets["voice_device"]

        assert combo.currentData() == ""
        assert combo.count() == len(get_input_devices())
        assert "voice_device" not in self._save(window, settings_file)

    @pytest.mark.parametrize("choice", ["2", ""])
    def test_choosing_a_listed_option_replaces_the_name(
        self, settings_file, microphones, choice,
    ):
        window = self._open(settings_file, {"voice_device": NAMED_MIC})
        combo = window._widgets["voice_device"]
        combo.setCurrentIndex(combo.findData(choice))

        saved = self._save(window, settings_file)

        assert saved.get("voice_device") == (choice or None)

    def test_reset_drops_the_name(self, settings_file, microphones):
        from desktop_app import settings_window as sw

        window = self._open(settings_file, {"voice_device": NAMED_MIC})
        with patch.object(
            sw.QMessageBox, "question",
            return_value=sw.QMessageBox.StandardButton.Yes,
        ):
            window._on_reset()

        assert "voice_device" not in self._save(window, settings_file)


class TestDefaultValueTypes:
    """Verify that default values match the declared field types."""

    def test_bool_defaults_are_bool(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "bool":
                val = defaults.get(fm.key)
                assert isinstance(val, bool), (
                    f"Field '{fm.key}' default {val!r} is not bool"
                )

    def test_int_defaults_are_numeric(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "int" and not fm.nullable:
                val = defaults.get(fm.key)
                assert isinstance(val, (int, float)), (
                    f"Field '{fm.key}' default {val!r} is not numeric"
                )

    def test_float_defaults_are_numeric(self):
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "float":
                val = defaults.get(fm.key)
                assert isinstance(val, (int, float)), (
                    f"Field '{fm.key}' default {val!r} is not numeric"
                )

    def test_choice_defaults_are_in_choices(self):
        """Default values for choice fields must be one of the valid choices."""
        defaults = get_default_config()
        for fm in FIELD_METADATA:
            if fm.field_type == "choice" and fm.choices:
                val = str(defaults.get(fm.key))
                valid_values = [c[0] for c in fm.choices]
                assert val in valid_values, (
                    f"Field '{fm.key}' default '{val}' not in choices {valid_values}"
                )


class TestMCPEditDialogLogic:
    """Tests for the MCP edit dialog's get_result() logic (no GUI)."""

    def test_get_result_basic(self):
        """get_result parses name, command, args, and env correctly."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "test-server"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "npx"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = "-y @test/server ~"
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = "API_KEY=abc123"

        name, cfg = dlg.get_result()
        assert name == "test-server"
        assert cfg["transport"] == "stdio"
        assert cfg["command"] == "npx"
        assert cfg["args"] == ["-y", "@test/server", "~"]
        assert cfg["env"] == {"API_KEY": "abc123"}

    def test_get_result_empty_env(self):
        """When env is empty, env key should not be in config."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "test"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "node"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = ""
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = ""

        name, cfg = dlg.get_result()
        assert name == "test"
        assert cfg["command"] == "node"
        assert cfg["args"] == []
        assert "env" not in cfg

    def test_get_result_multiple_env_vars(self):
        """Multiple KEY=VALUE pairs are parsed correctly."""
        dlg = _MCPEditDialog.__new__(_MCPEditDialog)
        dlg._name_edit = MagicMock()
        dlg._name_edit.text.return_value = "srv"
        dlg._command_edit = MagicMock()
        dlg._command_edit.text.return_value = "cmd"
        dlg._args_edit = MagicMock()
        dlg._args_edit.text.return_value = ""
        dlg._env_edit = MagicMock()
        dlg._env_edit.text.return_value = "A=1 B=two C=three=four"

        _, cfg = dlg.get_result()
        assert cfg["env"] == {"A": "1", "B": "two", "C": "three=four"}


class TestMCPCatalogueDialogLogic:
    """Tests for the MCP catalogue dialog's Node.js detection (no GUI)."""

    def test_is_node_available_returns_true_when_found(self):
        """_is_node_available returns True when _resolve_command succeeds."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", return_value="/usr/bin/npx"):
            assert _MCPCatalogueDialog._is_node_available() is True

    def test_is_node_available_returns_false_when_missing(self):
        """_is_node_available returns False when _resolve_command raises."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", side_effect=FileNotFoundError("not found")):
            assert _MCPCatalogueDialog._is_node_available() is False


class TestMCPConfigSaveLogic:
    """Tests for MCP config preservation during save."""

    def test_mcps_saved_when_present(self):
        """MCP configs should be written to the config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            config = {
                "mcps": {
                    "filesystem": {
                        "transport": "stdio",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "~"],
                    }
                }
            }
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert "mcps" in saved
            assert "filesystem" in saved["mcps"]
            assert saved["mcps"]["filesystem"]["command"] == "npx"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_empty_mcps_not_saved(self):
        """When mcps is empty, it should not be written to config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            from jarvis.config import _save_json, _load_json

            # Simulate: mcps is empty so should not be written
            config = {"tts_enabled": False}
            _save_json(cfg_path, config)
            saved = _load_json(cfg_path)
            assert "mcps" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)
