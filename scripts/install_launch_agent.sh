#!/usr/bin/env bash
# Install the Jarvis macOS LaunchAgent so the daemon starts on login.
#
# Usage:
#   bash scripts/install_launch_agent.sh           # install + load
#   bash scripts/install_launch_agent.sh --dry-run # show planned actions, no changes
#
# What this does:
#   1. Verify we're on macOS and the mamba env is present.
#   2. If ANTHROPIC_API_KEY is exported in the current shell, store
#      it in the macOS login keychain under service ``jarvis-anthropic``.
#      The key is fetched at boot by the wrapper bash; it never ends
#      up in the plist or in process environment listings.
#   3. Render the plist template into ~/Library/LaunchAgents/
#      com.jarvis.daemon.plist with absolute paths substituted.
#   4. launchctl unload (silent if not currently loaded) + load -w.
#   5. Verify the agent is listed by launchctl.
#
# Re-run the script any time the repo path / mamba env / API key
# changes — ``security add-generic-password -U`` and ``launchctl load
# -w`` both update in place rather than erroring.

set -uo pipefail

# ── Parse flags ──────────────────────────────────────────────────
DRY_RUN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help)
            sed -n '2,22p' "$0"
            exit 0
            ;;
        *)
            echo "❌ unknown argument: ${arg}" >&2
            echo "   use --dry-run or --help" >&2
            exit 2
            ;;
    esac
done

# ── Resolved paths ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${REPO_ROOT}/.mamba_env/bin/python"
WRAPPER_SRC="${REPO_ROOT}/scripts/launchagent/jarvis_daemon_wrapper.sh"
PLIST_TEMPLATE="${REPO_ROOT}/scripts/launchagent/com.jarvis.daemon.plist.template"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
PLIST_DEST="${LAUNCH_AGENTS_DIR}/com.jarvis.daemon.plist"
LOG_DIR="${HOME}/Library/Logs/Jarvis"

# Used by both the wrapper substitution AND the keychain entry.
KEYCHAIN_USER="${USER:-$(/usr/bin/id -un)}"

# ── Pre-flight checks ────────────────────────────────────────────
echo "🔍 Pre-flight checks"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "   ❌ This script is macOS-only (you're on $(uname -s))" >&2
    echo "      The LaunchAgent design uses macOS-specific tooling" >&2
    echo "      (launchctl, security). Linux users can wire a systemd" >&2
    echo "      user unit pointing at scripts/run_linux.sh." >&2
    exit 1
fi
echo "   ✅ macOS detected"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "   ❌ Mamba env missing at ${PYTHON_BIN}" >&2
    echo "      Run the project setup first." >&2
    exit 1
fi
echo "   ✅ python found: ${PYTHON_BIN}"

if [[ ! -f "${WRAPPER_SRC}" ]]; then
    echo "   ❌ Wrapper missing: ${WRAPPER_SRC}" >&2
    exit 1
fi
echo "   ✅ wrapper template found"

if [[ ! -f "${PLIST_TEMPLATE}" ]]; then
    echo "   ❌ Plist template missing: ${PLIST_TEMPLATE}" >&2
    exit 1
fi
echo "   ✅ plist template found"
echo ""

# ── Plan summary ─────────────────────────────────────────────────
echo "📋 Plan"
echo "   Repo root:    ${REPO_ROOT}"
echo "   Python:       ${PYTHON_BIN}"
echo "   Plist dest:   ${PLIST_DEST}"
echo "   Log dir:      ${LOG_DIR}"
echo "   Keychain user: ${KEYCHAIN_USER}"
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "   API key:      detected in env → will store in keychain"
else
    echo "   API key:      not set in env → daemon will run local-only"
fi
echo ""

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "🧪 Dry-run mode — listing actions only, no changes:"
    echo "   1. mkdir -p ${LAUNCH_AGENTS_DIR}"
    echo "   2. mkdir -p ${LOG_DIR}"
    if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        echo "   3. security add-generic-password -U -s jarvis-anthropic -a ${KEYCHAIN_USER} -w '***'"
    else
        echo "   3. (skip keychain — no key in env)"
    fi
    echo "   4. Render ${PLIST_TEMPLATE} → ${PLIST_DEST}"
    echo "   5. launchctl unload ${PLIST_DEST} (silent on first install)"
    echo "   6. launchctl load -w ${PLIST_DEST}"
    echo "   7. launchctl list | grep com.jarvis.daemon"
    echo ""
    echo "✅ Dry-run complete. Re-run without --dry-run to apply."
    exit 0
fi

# ── Wet run ──────────────────────────────────────────────────────
echo "🚀 Installing"

# 1. Ensure target directories exist.
mkdir -p "${LAUNCH_AGENTS_DIR}"
mkdir -p "${LOG_DIR}"
echo "   ✅ ${LAUNCH_AGENTS_DIR} ready"
echo "   ✅ ${LOG_DIR} ready"

# 2. Store the API key in keychain if available.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
    if /usr/bin/security add-generic-password \
        -U \
        -s jarvis-anthropic \
        -a "${KEYCHAIN_USER}" \
        -w "${ANTHROPIC_API_KEY}" >/dev/null 2>&1; then
        echo "   ✅ Anthropic key stored in keychain (service: jarvis-anthropic)"
    else
        echo "   ⚠️  Failed to update keychain entry; daemon will run local-only" >&2
    fi
else
    echo "   ⚠️  ANTHROPIC_API_KEY not exported — daemon will run in local-only mode"
    echo "        To enable hybrid mode later: export ANTHROPIC_API_KEY=sk-... && re-run this script"
fi

# 3. Render the plist template. Use sed because envsubst isn't
# always available on macOS without homebrew, and our substitutions
# are simple literal replacements.
sed_safe() {
    # sed -i requires '' on macOS BSD sed; use a temp file pattern
    # so the script works the same on both BSD and GNU sed.
    local pattern="$1"
    local replacement="$2"
    local src="$3"
    local dst="$4"
    # Use a sentinel delimiter unlikely to appear in paths.
    /usr/bin/sed "s|${pattern}|${replacement}|g" "${src}" > "${dst}.tmp"
    mv "${dst}.tmp" "${dst}"
}

# Render wrapper script too (the placeholders in there need
# substituting just like the plist).
WRAPPER_RENDERED="${REPO_ROOT}/scripts/launchagent/.jarvis_daemon_wrapper.rendered.sh"
/usr/bin/sed \
    -e "s|__PYTHON_BIN__|${PYTHON_BIN}|g" \
    -e "s|__JARVIS_REPO__|${REPO_ROOT}|g" \
    -e "s|__USER__|${KEYCHAIN_USER}|g" \
    "${WRAPPER_SRC}" > "${WRAPPER_RENDERED}"
chmod +x "${WRAPPER_RENDERED}"
echo "   ✅ wrapper rendered → ${WRAPPER_RENDERED}"

# Render the plist itself.
/usr/bin/sed \
    -e "s|__WRAPPER_PATH__|${WRAPPER_RENDERED}|g" \
    -e "s|__JARVIS_REPO__|${REPO_ROOT}|g" \
    -e "s|__LOG_DIR__|${LOG_DIR}|g" \
    -e "s|__USER__|${KEYCHAIN_USER}|g" \
    "${PLIST_TEMPLATE}" > "${PLIST_DEST}"
chmod 600 "${PLIST_DEST}"
echo "   ✅ plist rendered → ${PLIST_DEST}"

# 4. Reload via launchctl. ``unload`` is best-effort: it silently
# fails if the agent wasn't loaded, which is the common case on a
# first install.
/bin/launchctl unload "${PLIST_DEST}" 2>/dev/null || true
if /bin/launchctl load -w "${PLIST_DEST}"; then
    echo "   ✅ launchctl load OK"
else
    echo "   ❌ launchctl load failed; check ${LOG_DIR}/stderr.log" >&2
    exit 1
fi

# 5. Verify.
echo ""
echo "🔎 Status"
if /bin/launchctl list | grep -q com.jarvis.daemon; then
    /bin/launchctl list | grep com.jarvis.daemon | head -1 | awk '{print "   PID="$1, "exit="$2, "label="$3}'
    echo "   ✅ Jarvis daemon is loaded; it will auto-start at next login"
    echo "      Logs: ${LOG_DIR}/{stdout,stderr}.log"
else
    echo "   ⚠️  launchctl list does not show com.jarvis.daemon — check ${LOG_DIR}/stderr.log" >&2
fi
