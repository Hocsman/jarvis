#!/usr/bin/env bash
# Launch the desktop app with the reactive orb window enabled.
#
# Usage:
#   bash scripts/run_orb.sh
#
# This is a thin wrapper over the standard desktop_app entry point
# with the --with-orb flag. Use it when you want to demo or develop
# the orb without remembering the flag.
#
# Requirements:
#   - .mamba_env/ already bootstrapped (see scripts/run_macos.sh).
#   - Ollama running locally with the configured chat model pulled.
#
# Environment overrides:
#   JARVIS_ORB_FORCE_DEV=1     # force the DEV badge on
#   JARVIS_ORB_FORCE_PROD=1    # force the DEV badge off
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

if [ ! -d .mamba_env ]; then
  echo "❌ .mamba_env/ not found. Bootstrap it first via scripts/run_macos.sh." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT/src"
exec .mamba_env/bin/python -m desktop_app --with-orb "$@"
