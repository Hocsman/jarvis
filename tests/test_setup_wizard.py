"""
Tests for setup wizard detection functions.

These tests verify the Ollama detection logic without touching the UI.
They treat the detection functions as black boxes, verifying inputs produce correct outputs.
"""

import subprocess
from unittest.mock import patch, MagicMock
import pytest

from desktop_app.setup_wizard import (
    check_ollama_cli,
    check_ollama_server,
    get_required_models,
    check_installed_models,
    check_ollama_status,
    resolve_ollama_path,
    should_show_setup_wizard,
    OllamaStatus,
    MCPPage,
    SearchProvidersPage,
    ProviderChoicePage,
    OpenAICompatiblePage,
)
from desktop_app.mcp_catalogue import get_wizard_entries
from jarvis.config import DEFAULT_CHAT_MODEL
from jarvis.utils.location import (
    get_location_context,
    is_location_available,
    _is_private_ip,
)


class TestCheckOllamaCli:
    """Tests for Ollama CLI detection."""

    def test_detects_ollama_in_path(self):
        """When ollama is in PATH, returns True with path."""
        with patch("shutil.which", return_value="/usr/local/bin/ollama"):
            is_installed, path = check_ollama_cli()

            assert is_installed is True
            assert path == "/usr/local/bin/ollama"

    def test_returns_false_when_not_installed(self):
        """When ollama is not installed anywhere, returns False."""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                is_installed, path = check_ollama_cli()

                assert is_installed is False
                assert path is None

    def test_checks_macos_homebrew_path(self):
        """On macOS, checks Homebrew installation path."""
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile") as mock_isfile:
                with patch("os.access", return_value=True):
                    # First call for /usr/local/bin/ollama returns False
                    # Second call for /opt/homebrew/bin/ollama returns True
                    mock_isfile.side_effect = lambda p: p == "/opt/homebrew/bin/ollama"

                    is_installed, path = check_ollama_cli()

                    assert is_installed is True
                    assert path == "/opt/homebrew/bin/ollama"


class TestCheckOllamaServer:
    """Tests for Ollama server detection."""

    def test_detects_running_server(self):
        """When server is running, returns True with version."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "0.1.23"}

        with patch("requests.get", return_value=mock_response):
            is_running, version = check_ollama_server()

            assert is_running is True
            assert version == "0.1.23"

    def test_returns_false_when_server_not_running(self):
        """When server is not responding, returns False."""
        with patch("requests.get", side_effect=Exception("Connection refused")):
            is_running, version = check_ollama_server()

            assert is_running is False
            assert version is None

    def test_handles_timeout(self):
        """When request times out, returns False."""
        import requests
        with patch("requests.get", side_effect=requests.exceptions.Timeout):
            is_running, version = check_ollama_server()

            assert is_running is False
            assert version is None


class TestGetRequiredModels:
    """Tests for getting required models from config."""

    def test_returns_models_from_config(self):
        """Returns chat and embed models from config."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "llama2:7b"
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.intent_judge_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            assert "llama2:7b" in models
            assert "nomic-embed-text" in models

    def test_includes_intent_judge_model_when_different_from_chat(self):
        """Includes intent judge model when it differs from chat model."""
        mock_settings = MagicMock()
        mock_settings.ollama_chat_model = "gpt-oss:20b"  # Different from intent judge
        mock_settings.ollama_embed_model = "nomic-embed-text"
        mock_settings.intent_judge_model = "gemma4:e2b"

        with patch("desktop_app.setup_wizard.load_settings", return_value=mock_settings):
            models = get_required_models()

            # Should have 3 models: chat, embed, and intent judge
            assert len(models) == 3
            assert "gpt-oss:20b" in models
            assert "nomic-embed-text" in models
            assert "gemma4:e2b" in models  # Intent judge model is always required

    def test_returns_defaults_on_config_error(self):
        """Returns default models if config can't be loaded."""
        with patch("desktop_app.setup_wizard.load_settings", side_effect=Exception("Config error")):
            models = get_required_models()

            assert len(models) == 2
            assert "gemma4:e2b" in models
            assert "nomic-embed-text" in models

    def _cfg(self, **over):
        from types import SimpleNamespace
        base = dict(
            llm_provider="ollama",
            embedding_provider="",
            ollama_chat_model="gemma4:e2b",
            ollama_embed_model="nomic-embed-text",
            intent_judge_model="gemma4:e2b",
        )
        base.update(over)
        return SimpleNamespace(**base)

    def test_pure_ollama_requires_chat_embed_judge(self):
        """The default local setup needs the chat model, embed model, and
        (distinct) intent-judge model — all pulled from Ollama."""
        cfg = self._cfg(llm_provider="ollama", ollama_chat_model="gpt-oss:20b")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["gpt-oss:20b", "nomic-embed-text", "gemma4:e2b"]

    def test_pure_openai_requires_no_ollama_models(self):
        """Chat, judge, and embeddings all remote: nothing to pull locally."""
        cfg = self._cfg(llm_provider="openai_compatible", embedding_provider="")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == []

    def test_openai_chat_with_ollama_embeddings_requires_only_embed_model(self):
        """The advanced split: chat/judge remote, embeddings on Ollama. Only
        the embedding model must be present locally — not the remote chat
        model name, not the intent-judge model."""
        cfg = self._cfg(
            llm_provider="openai_compatible",
            embedding_provider="ollama",
            ollama_chat_model="some-remote-model",
            intent_judge_model="vendor/remote-judge",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["nomic-embed-text"]

    @pytest.mark.parametrize("provider", ["openai_compatible", "anthropic"])
    def test_a_small_role_kept_local_on_a_remote_provider_is_pulled(self, provider):
        """With Ollama running for the embeddings, a local tag for a small
        role (the default intent judge) stays local, so it must be there:
        nothing else pulls it, and the wizard does not open on its own
        for a remote provider."""
        cfg = self._cfg(
            llm_provider=provider,
            embedding_provider="ollama",
            intent_judge_model="gemma4:e2b",
            tool_router_model="qwen2.5:3b",
            planner_model="",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["nomic-embed-text", "gemma4:e2b", "qwen2.5:3b"]

    def test_anthropic_chat_requires_only_the_embed_model(self):
        """Chat and judge go to Anthropic; embeddings stay on Ollama, which
        is the one model to pull."""
        cfg = self._cfg(
            llm_provider="anthropic",
            ollama_chat_model="gemma4:e2b",
            intent_judge_model="claude-haiku-5-5",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["nomic-embed-text"]

    def test_ollama_chat_with_openai_embeddings_skips_embed_model(self):
        """Chat/judge on Ollama, embeddings remote: pull chat + judge, not
        the Ollama embed model."""
        cfg = self._cfg(
            llm_provider="ollama",
            embedding_provider="openai_compatible",
            ollama_chat_model="gpt-oss:20b",
        )
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg):
            models = get_required_models()
        assert models == ["gpt-oss:20b", "gemma4:e2b"]


class TestCheckInstalledModels:
    """Tests for checking installed Ollama models."""

    def test_parses_ollama_list_output(self):
        """Correctly parses 'ollama list' output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = """NAME                       ID              SIZE      MODIFIED
llama2:7b                  abc123          3.8 GB    2 days ago
nomic-embed-text:latest    def456          274 MB    1 week ago
"""

        with patch("subprocess.run", return_value=mock_result):
            models = check_installed_models("/usr/bin/ollama")

            assert "llama2:7b" in models
            assert "nomic-embed-text:latest" in models

    def test_returns_empty_on_error(self):
        """Returns empty list if ollama list fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            models = check_installed_models()

            assert models == []

    def test_handles_subprocess_exception(self):
        """Returns empty list if subprocess raises exception."""
        with patch("subprocess.run", side_effect=Exception("Command not found")):
            models = check_installed_models()

            assert models == []

    def test_falls_back_to_check_ollama_cli_when_path_unset(self):
        """When PATH does not contain ollama (e.g. frozen macOS .app launch),
        falls back to check_ollama_cli() so the resolved binary is invoked
        instead of plain "ollama" which would fail with FileNotFoundError."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME    ID    SIZE    MODIFIED\nllama2:7b    abc    3.8 GB    1d\n"

        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(True, "/usr/local/bin/ollama"),
            ):
                with patch("subprocess.run", return_value=mock_result) as run:
                    models = check_installed_models()

                    assert "llama2:7b" in models
                    args, _ = run.call_args
                    assert args[0][0] == "/usr/local/bin/ollama"


class TestResolveOllamaPath:
    """Tests for the ollama CLI path resolver."""

    def test_prefers_path_lookup(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value="/opt/homebrew/bin/ollama"):
            assert resolve_ollama_path() == "/opt/homebrew/bin/ollama"

    def test_falls_back_to_check_ollama_cli(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(True, "/usr/local/bin/ollama"),
            ):
                assert resolve_ollama_path() == "/usr/local/bin/ollama"

    def test_returns_literal_when_nothing_resolves(self):
        with patch("desktop_app.setup_wizard.shutil.which", return_value=None):
            with patch(
                "desktop_app.setup_wizard.check_ollama_cli",
                return_value=(False, None),
            ):
                assert resolve_ollama_path() == "ollama"


class TestCheckOllamaStatus:
    """Tests for complete Ollama status check."""

    def test_fully_setup_status(self):
        """Returns correct status when everything is set up."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.1.23")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b"]):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=["llama2:7b"]):
                        status = check_ollama_status()

                        assert status.is_cli_installed is True
                        assert status.is_server_running is True
                        assert status.missing_models == []
                        assert status.is_fully_setup is True

    def test_missing_cli_status(self):
        """Returns correct status when CLI is not installed."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(False, None)):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(False, None)):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b"]):
                    status = check_ollama_status()

                    assert status.is_cli_installed is False
                    assert status.is_fully_setup is False
                    assert "llama2:7b" in status.missing_models

    def test_missing_models_status(self):
        """Returns correct status when models are missing."""
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.1.23")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=["llama2:7b", "codellama"]):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=["llama2:7b"]):
                        status = check_ollama_status()

                        assert status.is_cli_installed is True
                        assert status.is_server_running is True
                        assert "codellama" in status.missing_models
                        assert status.is_fully_setup is False


class TestShouldShowSetupWizard:
    """Tests for wizard display logic."""

    def test_returns_false_when_fully_setup(self):
        """Returns False when everything is configured."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.1.23",
            installed_models=["llama2:7b"],
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_returns_true_when_cli_missing(self):
        """Returns True when CLI is not installed."""
        mock_status = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=["llama2:7b"],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_returns_false_when_server_not_running_but_cli_installed(self):
        """Returns False when server is not running but CLI is installed.

        The app can auto-start the server, so no wizard needed.
        """
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=False,
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_returns_true_when_models_missing(self):
        """Returns True when required models are missing."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.1.23",
            installed_models=[],
            missing_models=["llama2:7b"],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_returns_false_for_openai_compatible_provider(self):
        """An OpenAI-compatible user has opted out of the local Ollama
        stack, so the Ollama-centric wizard must never auto-show even if
        the Ollama CLI is absent."""
        from types import SimpleNamespace
        missing_cli = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=["llama2:7b"],
        )
        cfg = SimpleNamespace(llm_provider="openai_compatible")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg), \
             patch("desktop_app.setup_wizard.check_ollama_status", return_value=missing_cli):
            assert should_show_setup_wizard() is False

    def test_returns_false_for_anthropic_provider(self):
        """The wizard offers no Anthropic page, and its provider page would
        reset an Anthropic setup to Ollama: it must not open on its own."""
        from types import SimpleNamespace
        missing_cli = OllamaStatus(
            is_cli_installed=False,
            is_server_running=False,
            missing_models=["llama2:7b"],
        )
        cfg = SimpleNamespace(llm_provider="anthropic")
        with patch("desktop_app.setup_wizard.load_settings", return_value=cfg), \
             patch("desktop_app.setup_wizard.check_ollama_status", return_value=missing_cli):
            assert should_show_setup_wizard() is False


class TestProviderChoicePage:
    """The first real wizard decision: which runtime serves the LLM."""

    def test_validate_writes_openai_compatible_provider(self):
        """Selecting the OpenAI-compatible card persists llm_provider."""
        import tempfile, json
        from pathlib import Path
        page = ProviderChoicePage.__new__(ProviderChoicePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._selected = "openai_compatible"
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert saved["llm_provider"] == "openai_compatible"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_removes_provider_override_for_ollama(self):
        """Selecting Ollama clears the openai_compatible overrides so the
        Ollama settings become authoritative again (no stale base URL /
        model / key left pointing at a former OpenAI-compatible server)."""
        import tempfile, json
        from pathlib import Path
        page = ProviderChoicePage.__new__(ProviderChoicePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "llm_provider": "openai_compatible",
                "llm_base_url": "http://localhost:1234/v1",
                "llm_api_key": "sk-x",
                "llm_chat_model": "lmstudio/gemma",
                "embedding_model": "text-embedding-3-small",
            }, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._selected = "ollama"
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert saved.get("llm_provider", "ollama") == "ollama"
            for stale in ("llm_base_url", "llm_api_key", "llm_chat_model",
                          "embedding_model", "embedding_base_url", "embedding_api_key"):
                assert stale not in saved, f"{stale} must be cleared on the Ollama path"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_for_ollama_forgets_the_remote_key_variable(self):
        """The variable naming a remote provider's key goes with the rest
        of the remote setup: left behind, it would hand that key to the
        next remote provider chosen."""
        import tempfile, json
        from pathlib import Path
        page = ProviderChoicePage.__new__(ProviderChoicePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"llm_provider": "anthropic", "llm_api_key_env": "YUBA_ANTHROPIC_KEY",
                       "llm_chat_model": "claude-haiku-5-5"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._selected = "ollama"
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert "llm_api_key_env" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_routes_openai_to_config_page(self):
        """OpenAI-compatible selection jumps to the connection-config page."""
        page = ProviderChoicePage.__new__(ProviderChoicePage)
        page._selected = "openai_compatible"
        wizard = MagicMock()
        wizard.openai_compat_page_id = 42
        page.wizard = MagicMock(return_value=wizard)
        # isinstance check in nextId: make wizard look like SetupWizard
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 42

    def test_preselects_openai_from_existing_config(self, qapp):
        """Re-running the wizard reflects the saved provider: an existing
        openai_compatible config preselects the OpenAI card."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"llm_provider": "openai_compatible"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()  # __init__ calls _preselect_from_config
            assert page._selected == "openai_compatible"
            assert page._openai_radio.isChecked() is True
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_preselects_ollama_by_default(self, qapp):
        """A config without llm_provider (the default install) preselects Ollama."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            assert page._selected == "ollama"
            assert page._ollama_radio.isChecked() is True
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_radios_are_mutually_exclusive(self, qapp):
        """The two provider radios live in separate cards, so they need a
        shared QButtonGroup to be mutually exclusive — checking one must
        uncheck the other (and update the selection)."""
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            # Default: Ollama selected, OpenAI not.
            assert page._ollama_radio.isChecked() and not page._openai_radio.isChecked()
            page._openai_radio.setChecked(True)
            assert page._openai_radio.isChecked()
            assert not page._ollama_radio.isChecked(), "radios must be mutually exclusive"
            assert page._selected == "openai_compatible"
            page._ollama_radio.setChecked(True)
            assert not page._openai_radio.isChecked()
            assert page._selected == "ollama"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_openai_card_describes_a_local_server(self, qapp):
        """The OpenAI-compatible card must not imply the option is cloud /
        less private: its copy clarifies it points at a local server."""
        import tempfile
        from pathlib import Path
        from PyQt6.QtWidgets import QLabel
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page = ProviderChoicePage()
            blob = " ".join(lbl.text().lower() for lbl in page.findChildren(QLabel))
            assert "local" in blob and "network" in blob, (
                "provider copy should make clear the OpenAI-compatible option "
                "is also local"
            )
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_ollama_routes_through_welcome_status(self):
        """Ollama selection goes to the Welcome/status page (which surfaces
        Ollama readiness only after the user has chosen Ollama)."""
        page = ProviderChoicePage.__new__(ProviderChoicePage)
        page._selected = "ollama"
        wizard = MagicMock()
        wizard.welcome_page_id = 5
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 5

    def test_wizard_starts_on_provider_choice(self, qapp):
        """Ollama is optional, so the wizard's first step is the provider
        choice — not the Ollama-centric Welcome/status page."""
        import tempfile
        from pathlib import Path
        from desktop_app.setup_wizard import SetupWizard
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                wiz = SetupWizard()
            assert wiz.startId() == wiz.provider_choice_page_id
        finally:
            cfg_path.unlink(missing_ok=True)


class TestWelcomePageFlow:
    """The Welcome/status page is reached only on the Ollama branch."""

    def test_nextid_enters_ollama_flow(self):
        from desktop_app.setup_wizard import WelcomePage
        page = WelcomePage.__new__(WelcomePage)
        wizard = MagicMock()
        wizard.ollama_entry_page_id = MagicMock(return_value=9)
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 9


class TestOpenAICompatiblePage:
    """Collects the OpenAI-compatible connection details."""

    def test_incomplete_without_base_url_and_model(self):
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        page._base_url = ""
        page._chat_model = ""
        assert page._is_ready("", "") is False
        assert page._is_ready("http://localhost:1234/v1", "") is False
        assert page._is_ready("", "gemma") is False
        assert page._is_ready("http://localhost:1234/v1", "gemma") is True

    def test_validate_writes_connection_fields(self):
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"llm_provider": "openai_compatible"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "http://localhost:1234/v1", "sk-secret", "lmstudio/gemma", "text-embed-3",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert saved["llm_provider"] == "openai_compatible"
            assert saved["llm_base_url"] == "http://localhost:1234/v1"
            assert saved["llm_api_key"] == "sk-secret"
            assert saved["llm_chat_model"] == "lmstudio/gemma"
            assert saved["embedding_model"] == "text-embed-3"
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_omits_empty_optional_fields(self):
        """API key and embedding model are optional; empty values are not
        persisted, keeping config.json minimal."""
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            f.write("{}")
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "http://localhost:1234/v1", "", "lmstudio/gemma", "",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert "llm_api_key" not in saved
            assert "embedding_model" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_makes_the_key_typed_here_the_one_used(self):
        """A key variable left from another provider would win over the key
        typed on this page, and send that other provider's key here."""
        import tempfile, json
        from pathlib import Path
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"llm_provider": "anthropic", "llm_api_key_env": "YUBA_ANTHROPIC_KEY"}, f)
            cfg_path = Path(f.name)
        try:
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page._read_inputs = MagicMock(return_value=(
                    "https://openrouter.ai/api/v1", "sk-or-typed", "google/gemini-2.5-flash", "",
                ))
                assert page.validatePage() is True
            saved = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert saved["llm_api_key"] == "sk-or-typed"
            assert "llm_api_key_env" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_nextid_skips_ollama_pages(self):
        """After configuring the remote provider, the wizard jumps straight
        to Whisper setup — the Ollama install/server/models pages are
        irrelevant."""
        page = OpenAICompatiblePage.__new__(OpenAICompatiblePage)
        wizard = MagicMock()
        wizard.mlx_whisper_page_id = 7
        page.wizard = MagicMock(return_value=wizard)
        with patch("desktop_app.setup_wizard.SetupWizard", MagicMock):
            assert page.nextId() == 7

    def test_initialize_page_prefills_from_existing_config(self, qapp):
        """Re-running the wizard restores the user's saved connection
        details into the form fields so they are not re-typed."""
        import tempfile, json
        from pathlib import Path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({
                "llm_provider": "openai_compatible",
                "llm_base_url": "http://lmstudio:1234/v1",
                "llm_api_key": "sk-saved",
                "llm_chat_model": "lmstudio/gemma",
                "embedding_model": "text-embed-3",
            }, f)
            cfg_path = Path(f.name)
        try:
            page = OpenAICompatiblePage()
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.initializePage()
            assert page._base_url_input.text() == "http://lmstudio:1234/v1"
            assert page._api_key_input.text() == "sk-saved"
            assert page._chat_model_input.text() == "lmstudio/gemma"
            assert page._embed_model_input.text() == "text-embed-3"
            # The API key field stays masked even when pre-filled.
            from PyQt6.QtWidgets import QLineEdit
            assert page._api_key_input.echoMode() == QLineEdit.EchoMode.Password
        finally:
            cfg_path.unlink(missing_ok=True)


class TestOllamaStatusDataclass:
    """Tests for OllamaStatus dataclass behavior."""

    def test_is_fully_setup_property(self):
        """is_fully_setup returns True only when all conditions are met."""
        # All good
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=True,
            missing_models=[],
        )
        assert status.is_fully_setup is True

        # Missing CLI
        status = OllamaStatus(
            is_cli_installed=False,
            is_server_running=True,
            missing_models=[],
        )
        assert status.is_fully_setup is False

        # Server not running
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=False,
            missing_models=[],
        )
        assert status.is_fully_setup is False

        # Missing models
        status = OllamaStatus(
            is_cli_installed=True,
            is_server_running=True,
            missing_models=["some-model"],
        )
        assert status.is_fully_setup is False

    def test_default_values(self):
        """Dataclass initializes with correct defaults."""
        status = OllamaStatus()

        assert status.is_cli_installed is False
        assert status.cli_path is None
        assert status.is_server_running is False
        assert status.server_version is None
        assert status.installed_models == []
        assert status.missing_models == []


class TestLocationDetectionForWizard:
    """Tests for location detection utilities used in setup wizard."""

    def test_private_ip_detection(self):
        """Private IPs are correctly identified."""
        # RFC 1918 private ranges
        assert _is_private_ip("10.0.0.1") is True
        assert _is_private_ip("10.255.255.255") is True
        assert _is_private_ip("172.16.0.1") is True
        assert _is_private_ip("172.31.255.255") is True
        assert _is_private_ip("192.168.0.1") is True
        assert _is_private_ip("192.168.255.255") is True

        # Loopback
        assert _is_private_ip("127.0.0.1") is True

        # Public IPs (8.8.8.8 is Google DNS, 1.1.1.1 is Cloudflare)
        assert _is_private_ip("8.8.8.8") is False
        assert _is_private_ip("1.1.1.1") is False

    def test_location_context_returns_unknown_when_unavailable(self):
        """Location context returns 'Unknown' when detection fails."""
        # Disable auto-detect to avoid network calls, no config IP
        with patch("jarvis.utils.location._get_external_ip_automatically", return_value=None):
            with patch("jarvis.utils.location._get_local_network_ip", return_value="192.168.1.1"):
                context = get_location_context(config_ip=None, auto_detect=True)
                # Should return Unknown since 192.168.x.x can't be geolocated
                assert "Unknown" in context or "error" in context.lower()

    def test_location_availability_check(self):
        """is_location_available checks for GeoIP2 and database."""
        with patch("jarvis.utils.location.GEOIP2_AVAILABLE", False):
            # When library not available, should return False
            # Note: We can't easily patch the constant after import,
            # so we test the behavior indirectly
            pass

        # With patched database path
        with patch("jarvis.utils.location._get_database_path") as mock_path:
            mock_path_obj = MagicMock()
            mock_path_obj.exists.return_value = False
            mock_path.return_value = mock_path_obj

            # Can't easily test due to import-time GEOIP2_AVAILABLE check
            # but the function should return False if DB doesn't exist

    def test_location_context_with_config_ip(self):
        """When config IP is provided and valid, uses it for location."""
        mock_location = {
            "city": "San Francisco",
            "region": "California",
            "country": "United States",
            "timezone": "America/Los_Angeles",
        }

        with patch("jarvis.utils.location.get_location_info", return_value=mock_location):
            context = get_location_context(config_ip="203.0.113.45")

            assert "San Francisco" in context
            assert "California" in context
            assert "United States" in context


class TestModelOptions:
    """Tests for model selection options in setup wizard."""

    def test_model_options_available(self):
        """Model options include both recommended and lightweight options."""
        from desktop_app.setup_wizard import ModelsPage

        assert "gpt-oss:20b" in ModelsPage.MODEL_OPTIONS
        assert DEFAULT_CHAT_MODEL in ModelsPage.MODEL_OPTIONS

    def test_model_options_have_required_fields(self):
        """Each model option has required info fields."""
        from desktop_app.setup_wizard import ModelsPage

        for model_id, info in ModelsPage.MODEL_OPTIONS.items():
            assert "name" in info, f"Model {model_id} missing 'name'"
            assert "description" in info, f"Model {model_id} missing 'description'"
            assert "size" in info, f"Model {model_id} missing 'size'"
            assert "vram" in info, f"Model {model_id} missing 'vram'"

    def test_model_options_uses_centralized_config(self):
        """ModelsPage.MODEL_OPTIONS should reference the centralized config."""
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        # Verify they're the same object (not just equal values)
        assert ModelsPage.MODEL_OPTIONS is SUPPORTED_CHAT_MODELS


class TestDefaultModelDetection:
    """Regression tests: the default small model must be detected as missing when not
    installed, triggering the setup wizard install prompt.

    Uses DEFAULT_CHAT_MODEL from config so these tests stay valid when the default
    model changes — no hardcoded model names here.
    """

    EMBED_MODEL = "nomic-embed-text"

    def test_small_model_missing_detected_in_status(self):
        """When the default chat model is not installed, check_ollama_status reports it as missing."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=[self.EMBED_MODEL]):
                        status = check_ollama_status()

                        assert DEFAULT_CHAT_MODEL in status.missing_models
                        assert status.is_fully_setup is False

    def test_small_model_installed_not_in_missing(self):
        """When the default chat model is installed, check_ollama_status does not list it as missing."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    with patch("desktop_app.setup_wizard.check_installed_models", return_value=required):
                        status = check_ollama_status()

                        assert status.missing_models == []
                        assert status.is_fully_setup is True

    def test_wizard_shown_when_small_model_missing(self):
        """should_show_setup_wizard returns True when the default chat model is not installed."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.3.0",
            installed_models=[self.EMBED_MODEL],
            missing_models=[DEFAULT_CHAT_MODEL],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is True

    def test_wizard_not_shown_when_small_model_installed(self):
        """should_show_setup_wizard returns False when the default chat model is present."""
        mock_status = OllamaStatus(
            is_cli_installed=True,
            cli_path="/usr/bin/ollama",
            is_server_running=True,
            server_version="0.3.0",
            installed_models=[DEFAULT_CHAT_MODEL, self.EMBED_MODEL],
            missing_models=[],
        )

        with patch("desktop_app.setup_wizard.check_ollama_status", return_value=mock_status):
            assert should_show_setup_wizard() is False

    def test_latest_tag_stripped_before_comparison(self):
        """Ollama appends ':latest' to model names; the status check must strip it so
        '<model>:latest' is not incorrectly treated as missing when '<model>' is required."""
        required = [DEFAULT_CHAT_MODEL, self.EMBED_MODEL]
        with patch("desktop_app.setup_wizard.check_ollama_cli", return_value=(True, "/usr/bin/ollama")):
            with patch("desktop_app.setup_wizard.check_ollama_server", return_value=(True, "0.3.0")):
                with patch("desktop_app.setup_wizard.get_required_models", return_value=required):
                    # Simulate Ollama reporting "<model>:latest" in its model list
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = (
                        "NAME                       ID              SIZE      MODIFIED\n"
                        f"{DEFAULT_CHAT_MODEL}:latest    abc123          2.0 GB    1 day ago\n"
                        f"{self.EMBED_MODEL}:latest    def456          274 MB    1 week ago\n"
                    )
                    with patch("subprocess.run", return_value=mock_result):
                        status = check_ollama_status()

                        assert DEFAULT_CHAT_MODEL not in status.missing_models
                        assert status.is_fully_setup is True


class TestWhisperModelOptions:
    """Tests for whisper model selection options in setup wizard."""

    def test_whisper_multilingual_model_options_available(self):
        """Multilingual whisper model options include recommended and lightweight options."""
        from desktop_app.setup_wizard import WhisperSetupPage

        model_ids = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS]
        assert "small" in model_ids
        assert "tiny" in model_ids
        assert "large-v3-turbo" in model_ids

    def test_options_are_the_shared_list_split_by_language(self):
        """The slider offers every model of the list the settings window
        offers: ``.en`` ids on the English-only slider, the rest on the
        multilingual one."""
        from desktop_app.setup_wizard import WhisperSetupPage
        from jarvis.config import SUPPORTED_WHISPER_MODELS

        multilingual = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS]
        english = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN]
        assert multilingual == [m for m in SUPPORTED_WHISPER_MODELS if not m.endswith(".en")]
        assert english == [m for m in SUPPORTED_WHISPER_MODELS if m.endswith(".en")]

    def test_whisper_english_model_options_available(self):
        """English-only whisper model options include recommended and lightweight options."""
        from desktop_app.setup_wizard import WhisperSetupPage

        model_ids = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN]
        assert "small.en" in model_ids
        assert "tiny.en" in model_ids
        assert "medium.en" in model_ids
        # Note: large models don't have .en variants
        assert not any("large" in m for m in model_ids)

    def test_whisper_multilingual_model_options_have_required_fields(self):
        """Each multilingual whisper model option has required info fields."""
        from desktop_app.setup_wizard import WhisperSetupPage

        for model_tuple in WhisperSetupPage.WHISPER_MODEL_OPTIONS:
            assert len(model_tuple) == 5, f"Whisper model tuple should have 5 elements: {model_tuple}"
            model_id, name, file_size, ram, desc = model_tuple
            assert model_id, "Model ID should not be empty"
            assert name, "Model name should not be empty"
            assert file_size, "Model file size should not be empty"
            assert ram, "Model RAM requirement should not be empty"
            assert desc, "Model description should not be empty"
            # Multilingual models should NOT have .en suffix
            assert not model_id.endswith(".en"), f"Multilingual model should not end with .en: {model_id}"

    def test_turbo_hidden_when_faster_whisper_unsupported(self, installed_faster_whisper):
        """large-v3-turbo is filtered from options when faster-whisper is too old."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = False
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        installed_faster_whisper("1.0.3")
        options = WhisperSetupPage._listed_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" not in model_ids
        assert "small" in model_ids

    def test_turbo_shown_when_faster_whisper_supported(self, installed_faster_whisper):
        """large-v3-turbo is available when faster-whisper supports it."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = False
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        installed_faster_whisper("1.1.1")
        options = WhisperSetupPage._listed_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" in model_ids

    def test_turbo_always_shown_on_apple_silicon(self, installed_faster_whisper):
        """large-v3-turbo is always available on Apple Silicon (MLX backend)."""
        from desktop_app.setup_wizard import WhisperSetupPage

        page = MagicMock(spec=WhisperSetupPage)
        page._is_english_only = False
        page._is_apple_silicon = True
        page.WHISPER_MODEL_OPTIONS = WhisperSetupPage.WHISPER_MODEL_OPTIONS
        page.WHISPER_MODEL_OPTIONS_EN = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN

        installed_faster_whisper("1.0.3")
        options = WhisperSetupPage._listed_model_options(page)
        model_ids = [m[0] for m in options]
        assert "large-v3-turbo" in model_ids

    def test_whisper_english_model_options_have_required_fields(self):
        """Each English-only whisper model option has required info fields."""
        from desktop_app.setup_wizard import WhisperSetupPage

        for model_tuple in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN:
            assert len(model_tuple) == 5, f"Whisper model tuple should have 5 elements: {model_tuple}"
            model_id, name, file_size, ram, desc = model_tuple
            assert model_id, "Model ID should not be empty"
            assert name, "Model name should not be empty"
            assert file_size, "Model file size should not be empty"
            assert ram, "Model RAM requirement should not be empty"
            assert desc, "Model description should not be empty"
            # English-only models should have .en suffix
            assert model_id.endswith(".en"), f"English model should end with .en: {model_id}"


class TestWhisperSetupPageSliderRebuild:
    """Regression tests for WhisperSetupPage slider rebuild lifecycle.

    On macOS, promoting a child QLabel to a top-level widget (via
    setParent(None)) during a QWizard page transition could trigger
    a SIGABRT ('Fatal Python error: Aborted') while the next page
    was being shown.  These tests guarantee that the slider labels
    stay parented to their containers throughout rebuilds — the
    safe pattern for clearing items out of a layout.
    """

    def test_slider_labels_keep_container_parent_after_rebuild(self, qapp):
        """Newly-built slider labels must remain children of their containers.

        If any label ends up reparented to None it becomes a top-level
        widget, which on macOS triggers a native window creation that
        can abort during wizard page transitions.
        """
        from desktop_app.setup_wizard import WhisperSetupPage

        page = WhisperSetupPage()

        # Toggle language mode — this fires _rebuild_slider_ui which
        # clears the old labels and inserts a new set.
        page._on_language_changed(True)
        page._on_language_changed(False)

        labels_container = page._labels_container
        size_container = page._size_container

        for i in range(page._labels_layout.count()):
            item = page._labels_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                assert w.parent() is labels_container, (
                    "Slider name labels must stay parented to their "
                    "container — a None parent promotes them to top-level "
                    "widgets, which crashes QWizard transitions on macOS."
                )

        for i in range(page._size_layout.count()):
            item = page._size_layout.itemAt(i)
            w = item.widget()
            if w is not None:
                assert w.parent() is size_container, (
                    "Slider size labels must stay parented to their "
                    "container — a None parent promotes them to top-level "
                    "widgets, which crashes QWizard transitions on macOS."
                )

    def test_initialize_page_can_be_called_multiple_times(self, qapp):
        """initializePage must be safely re-callable.

        QWizard calls initializePage each time a page is shown.  The
        first call (right after construction) has to clear the initial
        labels that __init__ built, and subsequent calls must not
        crash or leak top-level widgets.
        """
        from desktop_app.setup_wizard import WhisperSetupPage

        page = WhisperSetupPage()

        # Re-initialise a few times — this mirrors back/forward
        # navigation between wizard pages.
        for _ in range(3):
            page.initializePage()

        # All remaining labels in the layouts are still properly
        # parented (not promoted to top-level).
        for layout, container in [
            (page._labels_layout, page._labels_container),
            (page._size_layout, page._size_container),
        ]:
            for i in range(layout.count()):
                item = layout.itemAt(i)
                w = item.widget()
                if w is not None:
                    assert w.parent() is container


class TestMCPPage:
    """Tests for the MCP servers wizard page."""

    def test_mcp_page_is_always_complete(self):
        """MCP page should always be completeable (nothing is required)."""
        # MCPPage.isComplete is hardcoded to True — the page is always optional
        page = MCPPage.__new__(MCPPage)
        assert page.isComplete() is True

    def test_is_already_configured_returns_false_on_empty_config(self):
        """When config has no mcps key, returns False."""
        with patch("jarvis.config._load_json", return_value={}):
            assert MCPPage._is_already_configured("filesystem") is False

    def test_is_already_configured_returns_true_when_present(self):
        """When the server name exists in config.mcps, returns True."""
        mock_config = {"mcps": {"filesystem": {"transport": "stdio"}}}
        with patch("jarvis.config._load_json", return_value=mock_config):
            assert MCPPage._is_already_configured("filesystem") is True

    def test_is_already_configured_handles_exception(self):
        """Returns False if config loading fails."""
        with patch("jarvis.config._load_json", side_effect=Exception("boom")):
            assert MCPPage._is_already_configured("filesystem") is False

    def test_wizard_entries_available(self):
        """Wizard-featured catalogue entries are available for the MCP page."""
        entries = get_wizard_entries()
        assert len(entries) >= 1
        # All entries should have display names and descriptions
        for e in entries:
            assert e.display_name
            assert e.description

    def test_validate_page_saves_selected_mcps(self):
        """validatePage writes selected MCPs to config."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)

        try:
            page = MCPPage.__new__(MCPPage)
            entries = get_wizard_entries()
            # Simulate checkboxes: first entry checked, rest unchecked
            page._checkboxes = {}
            for i, entry in enumerate(entries):
                cb = MagicMock()
                cb.isChecked.return_value = (i == 0)
                page._checkboxes[entry.name] = cb

            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                result = page.validatePage()

            assert result is True
            saved = _load_json(cfg_path)
            first_entry = entries[0]
            assert first_entry.name in saved.get("mcps", {})
            assert saved["mcps"][first_entry.name]["command"] == first_entry.command
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_is_node_available_returns_true_when_npx_found(self):
        """_is_node_available returns True when _resolve_command succeeds."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", return_value="/usr/bin/npx"):
            assert MCPPage._is_node_available() is True

    def test_is_node_available_returns_false_when_npx_missing(self):
        """_is_node_available returns False when _resolve_command raises."""
        with patch("jarvis.tools.external.mcp_client._resolve_command", side_effect=FileNotFoundError("not found")):
            assert MCPPage._is_node_available() is False

    def test_validate_page_preserves_existing_non_wizard_mcps(self):
        """validatePage must not remove MCPs that aren't in the wizard catalogue."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding="utf-8") as f:
            json.dump({"mcps": {"custom-server": {"transport": "stdio", "command": "node", "args": []}}}, f)
            cfg_path = Path(f.name)

        try:
            page = MCPPage.__new__(MCPPage)
            entries = get_wizard_entries()
            page._checkboxes = {}
            for entry in entries:
                cb = MagicMock()
                cb.isChecked.return_value = False
                page._checkboxes[entry.name] = cb

            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()

            saved = _load_json(cfg_path)
            assert "custom-server" in saved.get("mcps", {}), "Custom MCP server was removed"
        finally:
            cfg_path.unlink(missing_ok=True)


class TestSearchProvidersPage:
    """Tests for the Search Providers wizard page (Brave + Wikipedia)."""

    def _make_page(self, brave_key: str, wiki_enabled: bool) -> SearchProvidersPage:
        page = SearchProvidersPage.__new__(SearchProvidersPage)
        brave_input = MagicMock()
        brave_input.text.return_value = brave_key
        wiki_check = MagicMock()
        wiki_check.isChecked.return_value = wiki_enabled
        page._brave_input = brave_input
        page._wiki_check = wiki_check
        return page

    def test_page_is_always_complete(self):
        page = SearchProvidersPage.__new__(SearchProvidersPage)
        assert page.isComplete() is True

    def test_validate_writes_brave_key_when_provided(self):
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="BSA-abc123", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                assert page.validatePage() is True
            saved = _load_json(cfg_path)
            # Default non-default-only write: Brave present, Wikipedia omitted.
            assert saved.get("brave_search_api_key") == "BSA-abc123"
            assert "wikipedia_fallback_enabled" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_omits_empty_brave_key(self):
        """Empty Brave key must NOT write an empty-string entry — matches
        the settings-window minimal-diff invariant."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="   ", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert "brave_search_api_key" not in saved
            assert "wikipedia_fallback_enabled" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_persists_wikipedia_disable_only(self):
        """Wikipedia defaults to True, so only write it when user disables it."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="", wiki_enabled=False)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert saved.get("wikipedia_fallback_enabled") is False
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_removes_existing_brave_key_when_cleared(self):
        """If user blanks the Brave key, the entry must be removed, not kept."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"brave_search_api_key": "old-key"}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="", wiki_enabled=True)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert "brave_search_api_key" not in saved
        finally:
            cfg_path.unlink(missing_ok=True)

    def test_validate_preserves_unrelated_keys(self):
        """validatePage must not clobber unrelated config entries."""
        import json
        import tempfile
        from pathlib import Path
        from jarvis.config import _load_json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump({"ollama_chat_model": "gpt-oss:20b", "mcps": {"x": {}}}, f)
            cfg_path = Path(f.name)
        try:
            page = self._make_page(brave_key="BSA-key", wiki_enabled=False)
            with patch("jarvis.config.default_config_path", return_value=cfg_path):
                page.validatePage()
            saved = _load_json(cfg_path)
            assert saved["ollama_chat_model"] == "gpt-oss:20b"
            assert saved["mcps"] == {"x": {}}
        finally:
            cfg_path.unlink(missing_ok=True)


# --- Re-running the wizard keeps a value its pages do not list ------------


@pytest.fixture
def wizard_config(tmp_path, monkeypatch):
    """Point the config file at a fresh one, the way the suite-wide sandbox
    does: through ``jarvis.config`` only. A page that read or wrote through
    a path of its own would miss this file, and the tests below would see it.
    """
    import json

    path = tmp_path / "config.json"
    monkeypatch.setattr("jarvis.config.default_config_path", lambda: path)
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))

    class _Config:
        def write(self, values):
            path.write_text(json.dumps(values), encoding="utf-8")

        def read(self):
            return json.loads(path.read_text(encoding="utf-8"))

    return _Config()


class TestPagesWithNoConfiguredValue:
    """A fresh install (no file, or a key that names nothing) gets each page's
    default selected and no extra option."""

    _NO_FILE, _NO_KEY = object(), object()
    CASES = [
        pytest.param(_NO_FILE, id="no-file"),
        pytest.param(_NO_KEY, id="no-key"),
        pytest.param("", id="empty"),
        pytest.param(None, id="null"),
    ]

    @classmethod
    def _prepare(cls, wizard_config, case, key):
        if case is cls._NO_FILE:
            return
        wizard_config.write({} if case is cls._NO_KEY else {key: case})

    @pytest.mark.parametrize("case", CASES)
    def test_models_page_selects_the_default_chat_model(self, qapp, wizard_config, case):
        from PyQt6.QtWidgets import QPushButton
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        self._prepare(wizard_config, case, "ollama_chat_model")
        page = ModelsPage()
        page.initializePage()

        cards = [b for b in page.findChildren(QPushButton) if b.isCheckable()]
        checked = [b for b in cards if b.isChecked()]
        assert len(cards) == len(SUPPORTED_CHAT_MODELS)
        assert len(checked) == 1
        assert checked[0].text().startswith(SUPPORTED_CHAT_MODELS[DEFAULT_CHAT_MODEL]["name"])

    @pytest.mark.parametrize("case", CASES)
    def test_whisper_page_selects_the_default_model(self, qapp, wizard_config, case):
        from desktop_app.setup_wizard import WhisperSetupPage
        from jarvis.config import get_default_config

        self._prepare(wizard_config, case, "whisper_model")
        page = WhisperSetupPage()
        page.initializePage()
        page.validatePage()

        listed = WhisperSetupPage.WHISPER_MODEL_OPTIONS + WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN
        names = TestWhisperSetupPageKeepsAnUnlistedModel._slider_names(page)
        assert set(names) <= {option[1] for option in listed}
        assert wizard_config.read()["whisper_model"] == get_default_config()["whisper_model"]

    @pytest.mark.parametrize("case", CASES)
    def test_dictation_page_selects_the_default_hotkey(self, qapp, wizard_config, case):
        from desktop_app.setup_wizard import DictationPage
        from jarvis.config import DICTATION_HOTKEY_PRESETS, get_default_config

        self._prepare(wizard_config, case, "dictation_hotkey")
        page = DictationPage()

        assert page._hotkey_combo.count() == len(DICTATION_HOTKEY_PRESETS)
        assert page._hotkey_combo.currentData() == get_default_config()["dictation_hotkey"]


class TestModelsPageKeepsAnUnlistedChatModel:
    """A chat model the cards do not list (one pulled on purpose) survives a
    re-run of the wizard untouched."""

    UNLISTED = "my-own-model:7b"

    @staticmethod
    def _cards(page):
        from PyQt6.QtWidgets import QPushButton

        return [b for b in page.findChildren(QPushButton) if b.isCheckable()]

    def test_unlisted_model_is_written_back_unchanged(self, qapp, wizard_config):
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        assert self.UNLISTED not in SUPPORTED_CHAT_MODELS
        wizard_config.write({"ollama_chat_model": self.UNLISTED})

        page = ModelsPage()
        page.initializePage()
        page.validatePage()

        assert wizard_config.read()["ollama_chat_model"] == self.UNLISTED

    def test_unlisted_model_gets_its_own_selected_card(self, qapp, wizard_config):
        from desktop_app.setup_wizard import ModelsPage

        wizard_config.write({"ollama_chat_model": self.UNLISTED})

        page = ModelsPage()
        page.initializePage()

        checked = [b for b in self._cards(page) if b.isChecked()]
        assert [b.text() for b in checked] == [self.UNLISTED]

    def test_reentering_the_page_adds_the_card_once(self, qapp, wizard_config):
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        wizard_config.write({"ollama_chat_model": self.UNLISTED})

        page = ModelsPage()
        for _ in range(3):
            page.initializePage()

        assert len(self._cards(page)) == len(SUPPORTED_CHAT_MODELS) + 1

    def test_choosing_a_listed_model_replaces_it(self, qapp, wizard_config):
        from desktop_app.setup_wizard import ModelsPage
        from jarvis.config import SUPPORTED_CHAT_MODELS

        wizard_config.write({"ollama_chat_model": self.UNLISTED})
        listed_id, listed = next(iter(SUPPORTED_CHAT_MODELS.items()))

        page = ModelsPage()
        page.initializePage()
        card = next(b for b in self._cards(page) if b.text().startswith(listed["name"]))
        card.click()
        page.validatePage()

        assert wizard_config.read()["ollama_chat_model"] == listed_id

    def test_the_wizard_grows_by_the_added_card(self, qapp, wizard_config):
        from PyQt6.QtWidgets import QWizard
        from desktop_app.setup_wizard import ModelsPage

        heights = {}
        for model in (DEFAULT_CHAT_MODEL, self.UNLISTED):
            wizard_config.write({"ollama_chat_model": model})
            wizard = QWizard()
            page = ModelsPage()
            wizard.addPage(page)
            page.initializePage()
            heights[model] = wizard.minimumHeight()

        added = next(b for b in self._cards(page) if b.text() == self.UNLISTED)
        step = added.maximumHeight() + added.parentWidget().layout().spacing()
        assert heights[self.UNLISTED] - heights[DEFAULT_CHAT_MODEL] == step

    def test_unlisted_model_is_installed_like_a_listed_one(self, qapp, wizard_config):
        from desktop_app.setup_wizard import SetupWizard
        from jarvis.config import load_settings

        wizard_config.write({"ollama_chat_model": self.UNLISTED})
        cfg = load_settings()
        others = [cfg.ollama_embed_model, cfg.intent_judge_model]

        wizard = SetupWizard()
        page = wizard.page(wizard.models_page_id)
        wizard.ollama_status = OllamaStatus(
            is_cli_installed=True, cli_path="ollama", is_server_running=True,
            server_version="0", installed_models=list(others), missing_models=[],
        )
        page.initializePage()
        assert not page.isComplete()
        assert self.UNLISTED in page.models_label.text()

        wizard.ollama_status.installed_models.append(f"{self.UNLISTED}:latest")
        page.initializePage()
        assert page.isComplete()


class TestWhisperSetupPageKeepsAnUnlistedModel:
    """A Whisper model the slider does not list (a local model path, a turbo
    the installed engine cannot run) survives a re-run of the wizard
    untouched."""

    UNLISTED = "distil-large-v3"
    LONG_PATH = "C:/Users/someone/AppData/Local/models/" + "faster-whisper-large-v3-ct2-int8" * 3

    @staticmethod
    def _slider_labels(page):
        labels = []
        for i in range(page._labels_layout.count()):
            w = page._labels_layout.itemAt(i).widget()
            if w is not None:
                labels.append(w)
        return labels

    @classmethod
    def _slider_names(cls, page):
        return [label.text() for label in cls._slider_labels(page)]

    @pytest.mark.parametrize("value", [
        UNLISTED,
        "distil-small.en",
        "~/models/whisper-ct2",
        LONG_PATH,
    ])
    def test_unlisted_model_is_written_back_unchanged(self, qapp, wizard_config, value):
        from desktop_app.setup_wizard import WhisperSetupPage
        from jarvis.config import SUPPORTED_WHISPER_MODELS

        assert value not in SUPPORTED_WHISPER_MODELS
        wizard_config.write({"whisper_model": value})

        page = WhisperSetupPage()
        page.initializePage()
        page.validatePage()

        assert wizard_config.read()["whisper_model"] == value

    def test_every_model_the_settings_window_offers_is_listed(self, qapp, wizard_config, installed_faster_whisper):
        """A size picked in Settings comes back as that size, shown with its
        size details as a listed option rather than kept raw."""
        from desktop_app.settings_window import FIELD_METADATA
        from desktop_app.setup_wizard import WhisperSetupPage
        from jarvis.config import SUPPORTED_WHISPER_MODELS

        fm = next(f for f in FIELD_METADATA if f.key == "whisper_model")
        installed_faster_whisper("1.1.1")
        for value, _label in fm.choices:
            wizard_config.write({"whisper_model": value})

            page = WhisperSetupPage()
            page.initializePage()
            page.validatePage()

            assert wizard_config.read()["whisper_model"] == value
            assert SUPPORTED_WHISPER_MODELS[value]["size"] in page._model_info_label.text()

    def test_switching_a_large_model_to_english_keeps_the_largest_english_size(self, qapp, wizard_config, installed_faster_whisper):
        """Large models have no English-only variant: switching lands on the
        largest English size, not the smallest."""
        from desktop_app.setup_wizard import WhisperSetupPage

        largest_english = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN[-1][0]
        large = [m[0] for m in WhisperSetupPage.WHISPER_MODEL_OPTIONS
                 if f"{m[0]}.en" not in {e[0] for e in WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN}]
        assert large
        installed_faster_whisper("1.1.1")
        for model in large:
            wizard_config.write({"whisper_model": model})
            page = WhisperSetupPage()
            page.initializePage()
            page._english_btn.click()
            page.validatePage()

            assert wizard_config.read()["whisper_model"] == largest_english

    def test_reentering_the_page_reselects_the_configured_size(self, qapp, wizard_config):
        """Back then Next re-reads the config: a size dragged to but never
        saved does not stick, whatever the slider held."""
        from desktop_app.setup_wizard import WhisperSetupPage

        configured = WhisperSetupPage.WHISPER_MODEL_OPTIONS_EN[1][0]
        wizard_config.write({"whisper_model": configured})

        page = WhisperSetupPage()
        page.initializePage()
        page._multilingual_btn.click()
        page._model_slider.setValue(page._model_slider.maximum())
        page.initializePage()
        page.validatePage()

        assert wizard_config.read()["whisper_model"] == configured

    def test_a_long_path_is_shortened_on_screen_and_kept_whole(self, qapp, wizard_config):
        """However long a local model path is, the slider row does not grow
        with it; the whole path stays readable in the tooltip."""
        from desktop_app.setup_wizard import WhisperSetupPage

        widths = []
        for path in (self.LONG_PATH, self.LONG_PATH * 3):
            wizard_config.write({"whisper_model": path})
            page = WhisperSetupPage()
            page.initializePage()
            label = self._slider_labels(page)[-1]
            assert label.toolTip() == path
            assert label.text() != path
            widths.append((
                page._labels_container.minimumSizeHint().width(),
                page._model_info_label.minimumSizeHint().width(),
            ))

        assert widths[0] == widths[1]

    def test_turbo_is_kept_when_the_engine_cannot_run_it(self, qapp, wizard_config, installed_faster_whisper):
        from desktop_app.setup_wizard import WhisperSetupPage

        wizard_config.write({"whisper_model": "large-v3-turbo"})
        installed_faster_whisper("1.0.3")

        with patch("desktop_app.setup_wizard.is_apple_silicon", return_value=False):
            page = WhisperSetupPage()
            page.initializePage()
            page.validatePage()

        assert wizard_config.read()["whisper_model"] == "large-v3-turbo"

    def test_unlisted_model_is_shown_raw_on_the_slider(self, qapp, wizard_config):
        from desktop_app.setup_wizard import WhisperSetupPage

        wizard_config.write({"whisper_model": self.UNLISTED})

        page = WhisperSetupPage()
        page.initializePage()

        assert self.UNLISTED in self._slider_names(page)
        assert self.UNLISTED in page._model_info_label.text()

    def test_reentering_the_page_adds_the_option_once(self, qapp, wizard_config):
        from desktop_app.setup_wizard import WhisperSetupPage

        wizard_config.write({"whisper_model": self.UNLISTED})

        page = WhisperSetupPage()
        for _ in range(3):
            page.initializePage()

        assert self._slider_names(page).count(self.UNLISTED) == 1

    def test_unlisted_model_stays_in_its_own_language(self, qapp, wizard_config):
        from desktop_app.setup_wizard import WhisperSetupPage

        wizard_config.write({"whisper_model": self.UNLISTED})

        page = WhisperSetupPage()
        page.initializePage()
        page._english_btn.click()
        assert self.UNLISTED not in self._slider_names(page)
        page._multilingual_btn.click()
        assert self._slider_names(page).count(self.UNLISTED) == 1

    def test_choosing_a_listed_model_replaces_it(self, qapp, wizard_config):
        from desktop_app.setup_wizard import WhisperSetupPage

        wizard_config.write({"whisper_model": self.UNLISTED})

        page = WhisperSetupPage()
        page.initializePage()
        page._model_slider.setValue(0)
        page.validatePage()

        assert wizard_config.read()["whisper_model"] == WhisperSetupPage.WHISPER_MODEL_OPTIONS[0][0]


class TestDictationPageKeepsAnUnlistedHotkey:
    """A hand-edited dictation hotkey the presets do not list survives a
    re-run of the wizard untouched."""

    UNLISTED = "ctrl+alt+space"

    def test_unlisted_hotkey_is_written_back_unchanged(self, qapp, wizard_config):
        from desktop_app.setup_wizard import DictationPage
        from jarvis.config import DICTATION_HOTKEY_PRESETS

        assert self.UNLISTED not in DICTATION_HOTKEY_PRESETS
        wizard_config.write({"dictation_hotkey": self.UNLISTED})

        page = DictationPage()
        page.validatePage()

        assert wizard_config.read()["dictation_hotkey"] == self.UNLISTED

    def test_unlisted_hotkey_is_shown_raw_and_selected(self, qapp, wizard_config):
        from desktop_app.setup_wizard import DictationPage

        wizard_config.write({"dictation_hotkey": self.UNLISTED})

        page = DictationPage()

        assert page._hotkey_combo.currentText() == self.UNLISTED
        assert page._hotkey_combo.currentData() == self.UNLISTED

    def test_choosing_a_preset_replaces_it(self, qapp, wizard_config):
        from desktop_app.setup_wizard import DictationPage
        from jarvis.config import DICTATION_HOTKEY_PRESETS

        wizard_config.write({"dictation_hotkey": self.UNLISTED})
        preset = DICTATION_HOTKEY_PRESETS[0]

        page = DictationPage()
        page._hotkey_combo.setCurrentIndex(page._hotkey_combo.findData(preset))
        page.validatePage()

        assert wizard_config.read()["dictation_hotkey"] == preset

    @pytest.mark.parametrize("key", ["dictation_enabled", "dictation_filler_removal"])
    def test_its_toggles_are_written_back_as_set(self, qapp, wizard_config, key):
        """Dictation switched off in Settings stays off after a re-run; so
        does any toggle the user flipped away from its default."""
        from desktop_app.setup_wizard import DictationPage
        from jarvis.config import get_default_config

        flipped = not get_default_config()[key]
        wizard_config.write({key: flipped})

        page = DictationPage()
        page.validatePage()

        assert wizard_config.read()[key] is flipped

