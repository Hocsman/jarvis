"""Phase 2E LaunchAgent tests.

Three families:

1. ``TestPlistTemplateXml`` — substitute the four placeholders with
   dummy values, run the result through ``plistlib.loads()``, and
   assert it parses as a well-formed plist with the expected top-
   level keys + ProgramArguments shape. Catches any future syntax
   error in the template before it reaches a live ``launchctl
   load``.

2. ``TestInstallScriptDryRun`` — invoke ``bash
   scripts/install_launch_agent.sh --dry-run`` in a subprocess.
   Verify it exits 0, prints the planned ``security`` + ``launchctl
   load`` actions, and DOES NOT create the plist destination file
   or touch the keychain. The dry-run is the safety net we tell the
   user to lean on; if it silently mutates anything we've failed.

3. ``TestKeychainHelper`` — unit-level: mock subprocess.run and
   verify ``set / get / delete`` issue the right ``security``
   invocations and parse stdout/exit codes correctly. Bonus
   integration-marked roundtrip against the real keychain for the
   macOS CI path.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLIST_TEMPLATE = REPO_ROOT / "scripts" / "launchagent" / "com.jarvis.daemon.plist.template"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_launch_agent.sh"
UNINSTALL_SCRIPT = REPO_ROOT / "scripts" / "uninstall_launch_agent.sh"


# ── Family 1: plist template parses as valid XML plist ──────────────────


def _render_template_with_dummies() -> bytes:
    """Substitute the four placeholders with safe dummy values."""
    text = PLIST_TEMPLATE.read_text(encoding="utf-8")
    return (
        text
        .replace("__WRAPPER_PATH__", "/tmp/dummy_wrapper.sh")
        .replace("__JARVIS_REPO__", "/tmp/dummy_repo")
        .replace("__LOG_DIR__", "/tmp/dummy_logs")
        .replace("__USER__", "dummyuser")
    ).encode("utf-8")


class TestPlistTemplateXml:
    """If the template ever picks up bad XML or a missing closing
    tag, this catches it without anyone needing to load it via
    launchctl first."""

    @pytest.mark.unit
    def test_template_renders_to_valid_plist(self) -> None:
        rendered = _render_template_with_dummies()
        parsed = plistlib.loads(rendered)
        assert isinstance(parsed, dict), (
            f"Plist must parse to a dict at the top level, got {type(parsed).__name__}"
        )

    @pytest.mark.unit
    def test_label_is_com_jarvis_daemon(self) -> None:
        parsed = plistlib.loads(_render_template_with_dummies())
        assert parsed.get("Label") == "com.jarvis.daemon", (
            f"Plist Label is {parsed.get('Label')!r}, expected "
            f"'com.jarvis.daemon'. install / uninstall both grep on "
            f"this exact label — a rename would silently break them."
        )

    @pytest.mark.unit
    def test_program_arguments_points_at_wrapper(self) -> None:
        parsed = plistlib.loads(_render_template_with_dummies())
        args = parsed.get("ProgramArguments")
        assert isinstance(args, list) and args, (
            "ProgramArguments must be a non-empty list"
        )
        assert args[0] == "/tmp/dummy_wrapper.sh", (
            f"ProgramArguments[0] should be the substituted wrapper path; "
            f"got {args[0]!r}"
        )

    @pytest.mark.unit
    def test_environment_does_not_contain_api_key(self) -> None:
        """The whole point of routing the secret through the keychain
        is that the plist never carries it. A regression here is the
        worst kind of leak."""
        parsed = plistlib.loads(_render_template_with_dummies())
        env = parsed.get("EnvironmentVariables", {})
        assert "ANTHROPIC_API_KEY" not in env, (
            f"EnvironmentVariables contains ANTHROPIC_API_KEY — "
            f"that would surface the secret in ``launchctl print`` "
            f"and defeat the whole keychain design."
        )

    @pytest.mark.unit
    def test_run_at_load_is_true(self) -> None:
        parsed = plistlib.loads(_render_template_with_dummies())
        assert parsed.get("RunAtLoad") is True

    @pytest.mark.unit
    def test_keep_alive_is_false(self) -> None:
        """KeepAlive=True would respawn the daemon immediately after
        ``launchctl unload``, defeating the uninstall script."""
        parsed = plistlib.loads(_render_template_with_dummies())
        assert parsed.get("KeepAlive") is False

    @pytest.mark.unit
    def test_log_paths_under_log_dir(self) -> None:
        parsed = plistlib.loads(_render_template_with_dummies())
        assert parsed.get("StandardOutPath") == "/tmp/dummy_logs/stdout.log"
        assert parsed.get("StandardErrorPath") == "/tmp/dummy_logs/stderr.log"


# ── Family 2: install script --dry-run is side-effect-free ──────────────


class TestInstallScriptDryRun:
    """The dry-run is the safety net. It MUST not write the plist,
    MUST not call ``security`` for real, and MUST exit 0 with output
    that names the planned actions."""

    @pytest.mark.unit
    def test_dry_run_exits_zero(self, tmp_path, monkeypatch) -> None:
        """Run the script with HOME pointed at a tmp_path so even if
        a regression bypasses the dry-run guard, no user files are
        touched. Exit 0 + planned actions in stdout."""
        env = {**os.environ, "HOME": str(tmp_path)}
        # Make sure ANTHROPIC_API_KEY is unset so the test path is
        # deterministic (we test the with-key branch elsewhere).
        env.pop("ANTHROPIC_API_KEY", None)

        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--dry-run"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0, (
            f"--dry-run should exit 0; got {proc.returncode}. "
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )
        assert "Dry-run mode" in proc.stdout
        assert "launchctl load" in proc.stdout
        # plist path is mentioned in the planned-actions list.
        assert "com.jarvis.daemon.plist" in proc.stdout

    @pytest.mark.unit
    def test_dry_run_does_not_create_plist(self, tmp_path) -> None:
        """The actual side-effect we care about: no plist file under
        the simulated HOME's LaunchAgents directory."""
        env = {**os.environ, "HOME": str(tmp_path)}
        env.pop("ANTHROPIC_API_KEY", None)

        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--dry-run"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0
        plist = tmp_path / "Library" / "LaunchAgents" / "com.jarvis.daemon.plist"
        assert not plist.exists(), (
            f"--dry-run created the plist at {plist}; should be a no-op"
        )

    @pytest.mark.unit
    def test_dry_run_with_key_mentions_keychain(self, tmp_path) -> None:
        """When ANTHROPIC_API_KEY is exported, the dry-run output
        names the keychain action. The key itself must NOT appear in
        the output (we mask it as '***')."""
        env = {
            **os.environ,
            "HOME": str(tmp_path),
            "ANTHROPIC_API_KEY": "sk-ant-test-very-secret-value-DO-NOT-LEAK",
        }

        proc = subprocess.run(
            ["bash", str(INSTALL_SCRIPT), "--dry-run"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0
        assert "security add-generic-password" in proc.stdout, (
            "Expected planned-actions to mention the keychain command"
        )
        assert "sk-ant-test-very-secret-value-DO-NOT-LEAK" not in proc.stdout, (
            "Dry-run leaked the real API key in stdout — that defeats "
            "the keychain-only design"
        )

    @pytest.mark.unit
    def test_uninstall_dry_run_exits_zero(self, tmp_path) -> None:
        """Symmetric check for uninstall."""
        env = {**os.environ, "HOME": str(tmp_path)}
        proc = subprocess.run(
            ["bash", str(UNINSTALL_SCRIPT), "--dry-run"],
            env=env, capture_output=True, text=True, timeout=20,
        )
        assert proc.returncode == 0
        assert "Dry-run" in proc.stdout
        assert "launchctl unload" in proc.stdout


# ── Family 3: keychain helper unit + integration ────────────────────────


class TestKeychainHelperUnit:
    """Mock subprocess.run; verify the helper issues the right
    arguments and interprets exit codes."""

    @pytest.mark.unit
    def test_set_calls_security_with_update_flag(self) -> None:
        """``security add-generic-password -U`` is what lets re-runs
        of the install script replace an existing key. The ``-U``
        flag is non-obvious; pin it."""
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stderr = ""
            keychain.set_anthropic_key("sk-test-1234", account="alice")

            args, _kwargs = mock_run.call_args
            cmd = args[0]
            assert cmd[:2] == ["/usr/bin/security", "add-generic-password"]
            assert "-U" in cmd, "Missing -U flag (update if exists)"
            assert cmd[cmd.index("-s") + 1] == "jarvis-anthropic"
            assert cmd[cmd.index("-a") + 1] == "alice"
            assert cmd[cmd.index("-w") + 1] == "sk-test-1234"

    @pytest.mark.unit
    def test_set_raises_on_security_failure(self) -> None:
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stderr = "simulated denial"
            with pytest.raises(RuntimeError, match="security add-generic-password failed"):
                keychain.set_anthropic_key("sk-test", account="alice")

    @pytest.mark.unit
    def test_get_returns_key_on_success(self) -> None:
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "sk-stored-key\n"
            value = keychain.get_anthropic_key(account="alice")
            assert value == "sk-stored-key"

    @pytest.mark.unit
    def test_get_returns_none_when_missing(self) -> None:
        """``security find-generic-password`` exits 44 with "The
        specified item could not be found" when there's no entry.
        We map any non-zero to None — caller decides what to do."""
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 44
            mock_run.return_value.stdout = ""
            mock_run.return_value.stderr = "item not found"
            assert keychain.get_anthropic_key(account="alice") is None

    @pytest.mark.unit
    def test_delete_returns_true_on_success(self) -> None:
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert keychain.delete_anthropic_key(account="alice") is True

    @pytest.mark.unit
    def test_delete_returns_false_when_absent(self) -> None:
        """Idempotent uninstall: deleting a non-existent entry
        returns False, not raises."""
        from jarvis.utils import keychain

        with patch.object(keychain, "is_macos", return_value=True), \
             patch("jarvis.utils.keychain.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 44
            assert keychain.delete_anthropic_key(account="alice") is False

    @pytest.mark.unit
    def test_non_macos_set_raises(self) -> None:
        from jarvis.utils import keychain
        with patch.object(keychain, "is_macos", return_value=False):
            with pytest.raises(RuntimeError, match="requires macOS"):
                keychain.set_anthropic_key("sk-test")

    @pytest.mark.unit
    def test_non_macos_get_returns_none(self) -> None:
        from jarvis.utils import keychain
        with patch.object(keychain, "is_macos", return_value=False):
            assert keychain.get_anthropic_key() is None

    @pytest.mark.unit
    def test_non_macos_delete_returns_false(self) -> None:
        from jarvis.utils import keychain
        with patch.object(keychain, "is_macos", return_value=False):
            assert keychain.delete_anthropic_key() is False


class TestKeychainHelperRoundtripReal:
    """Integration roundtrip against the real macOS keychain. Uses
    a randomised service name to avoid stomping the real
    ``jarvis-anthropic`` entry; the helper's service is hardcoded so
    we monkey-patch ``KEYCHAIN_SERVICE`` for this test only."""

    @pytest.mark.integration
    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only test")
    def test_set_get_delete_real_keychain(self, monkeypatch) -> None:
        from jarvis.utils import keychain

        unique_service = f"jarvis-test-{uuid.uuid4().hex[:12]}"
        monkeypatch.setattr(keychain, "KEYCHAIN_SERVICE", unique_service)

        secret = f"sk-test-{uuid.uuid4().hex}"

        # Start: get returns None.
        assert keychain.get_anthropic_key() is None

        try:
            keychain.set_anthropic_key(secret)
            roundtrip = keychain.get_anthropic_key()
            assert roundtrip == secret, (
                f"keychain roundtrip mismatch: stored {secret!r}, "
                f"got {roundtrip!r}"
            )
        finally:
            # Cleanup. delete returns True if it removed something.
            deleted = keychain.delete_anthropic_key()
            assert deleted is True
            assert keychain.get_anthropic_key() is None
