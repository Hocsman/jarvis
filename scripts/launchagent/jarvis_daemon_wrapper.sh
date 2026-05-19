#!/usr/bin/env bash
# Jarvis daemon LaunchAgent wrapper (Phase 2E).
#
# This script is what launchd execs at user login. Its only job is
# to pull the Anthropic API key out of the macOS keychain, export
# it, then hand off to ``python -m jarvis.main``. The key never
# touches disk in cleartext; ``launchctl print`` never sees it.
#
# Substitutions performed by install_launch_agent.sh:
#   __PYTHON_BIN__  -> absolute path to the mamba_env Python binary
#   __JARVIS_REPO__ -> absolute path to the repo (for PYTHONPATH=src)
#   __USER__        -> $USER captured at install time (keychain account)
#
# Fail-soft contract:
# - If the keychain entry is missing, log a clear breadcrumb and
#   start the daemon WITHOUT the key. The router falls back to
#   local_only as a result (cf. llm_router_telemetry rows showing
#   local_fallback). That's the right behaviour: a missing key
#   should not block the daemon's mic / TTS / reply loop.
# - If the python binary is missing, we exit with status 1 so
#   launchd surfaces the failure in ``launchctl print``.

set -u  # -e omitted: we want explicit error handling on the security call.

# ── Resolved paths (substituted at install time) ────────────────
PYTHON_BIN="__PYTHON_BIN__"
JARVIS_REPO="__JARVIS_REPO__"
KEYCHAIN_USER="__USER__"

# ── Pre-flight ───────────────────────────────────────────────────
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "❌ jarvis-wrapper: python binary not found at ${PYTHON_BIN}" >&2
    echo "   Reinstall the LaunchAgent via scripts/install_launch_agent.sh" >&2
    exit 1
fi

# ── Keychain lookup ──────────────────────────────────────────────
# ``security find-generic-password -w`` prints just the password.
# We swallow stderr so a missing entry doesn't pollute the log.
ANTHROPIC_KEY=""
if KEY_RAW=$(/usr/bin/security find-generic-password -w -s jarvis-anthropic -a "${KEYCHAIN_USER}" 2>/dev/null); then
    ANTHROPIC_KEY="${KEY_RAW}"
fi

if [[ -n "${ANTHROPIC_KEY}" ]]; then
    export ANTHROPIC_API_KEY="${ANTHROPIC_KEY}"
    echo "🔑 jarvis-wrapper: Anthropic key loaded from keychain (jarvis-anthropic)"
else
    echo "🔇 jarvis-wrapper: no Anthropic key in keychain — daemon will run in local-only mode"
    echo "   To enable hybrid mode, run: scripts/install_launch_agent.sh (with ANTHROPIC_API_KEY exported)"
fi

# ── Hand off to the daemon ───────────────────────────────────────
cd "${JARVIS_REPO}"
export PYTHONPATH="${JARVIS_REPO}/src:${PYTHONPATH:-}"
exec "${PYTHON_BIN}" -m jarvis.main
