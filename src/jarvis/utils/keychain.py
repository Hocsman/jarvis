"""Thin wrapper around macOS ``security`` for the Anthropic API key.

Phase 2E uses the macOS keychain instead of a ``.env`` file or a
hardcoded plist value because the keychain is the only place that
keeps the secret out of every visible surface:

- Not on disk in cleartext (.env files leak via Time Machine, sync
  services, screen-sharing screenshots).
- Not in ``launchctl print`` output (the plist's
  ``EnvironmentVariables`` block would surface it; we deliberately
  don't put it there).
- Not in ``ps auxe`` output (we don't pass it as a process argument
  or environment that's visible to other processes' /proc readers).

The runtime path is in scripts/launchagent/jarvis_daemon_wrapper.sh:
it shells out to ``security find-generic-password -w -s
jarvis-anthropic -a "$USER"`` and exports the result. This module is
the Python side used at install / uninstall time + tests.

All operations are explicit no-ops on non-macOS so test suites run
cross-platform without skip noise.
"""

from __future__ import annotations

import platform
import subprocess
from typing import Optional


# Constants used by the install / uninstall scripts AND this module.
# Centralised so a future rename only touches one place.
KEYCHAIN_SERVICE: str = "jarvis-anthropic"


def is_macos() -> bool:
    """``True`` when running on macOS. Every public function below
    is a no-op (or raises ``RuntimeError``) on other platforms."""
    return platform.system() == "Darwin"


def _current_account() -> str:
    """The keychain ``-a`` field. We use the macOS short username,
    which matches what the wrapper bash sees at boot via ``$USER``."""
    import getpass
    return getpass.getuser()


def set_anthropic_key(value: str, *, account: Optional[str] = None) -> None:
    """Store the Anthropic API key in the user's login keychain.

    ``-U`` updates the entry if it already exists rather than
    erroring; this is the right behaviour when re-running the
    install script with a new key.

    ``-T ""`` would normally restrict the entry to specific apps;
    we leave it open so the wrapper bash (run by launchd, not
    by the install script's process) can still read it.

    Raises ``RuntimeError`` on non-macOS hosts so the caller can
    fail loudly rather than silently dropping the key.
    """
    if not is_macos():
        raise RuntimeError("set_anthropic_key requires macOS")
    if not value:
        raise ValueError("Refusing to store empty Anthropic key")
    acct = account or _current_account()
    cmd = [
        "/usr/bin/security", "add-generic-password",
        "-U",  # update if exists
        "-s", KEYCHAIN_SERVICE,
        "-a", acct,
        "-w", value,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"security add-generic-password failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )


def get_anthropic_key(*, account: Optional[str] = None) -> Optional[str]:
    """Return the key from the keychain, or ``None`` if it isn't set.

    ``security find-generic-password -w`` prints just the password
    when found, or exits non-zero with ``"The specified item could
    not be found in the keychain."`` on stderr otherwise — we map
    that to ``None``.
    """
    if not is_macos():
        return None
    acct = account or _current_account()
    cmd = [
        "/usr/bin/security", "find-generic-password", "-w",
        "-s", KEYCHAIN_SERVICE,
        "-a", acct,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return proc.stdout.rstrip("\n")
    return None


def delete_anthropic_key(*, account: Optional[str] = None) -> bool:
    """Remove the entry. Returns ``True`` if a deletion happened,
    ``False`` if there was nothing to delete (idempotent uninstall)."""
    if not is_macos():
        return False
    acct = account or _current_account()
    cmd = [
        "/usr/bin/security", "delete-generic-password",
        "-s", KEYCHAIN_SERVICE,
        "-a", acct,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0
