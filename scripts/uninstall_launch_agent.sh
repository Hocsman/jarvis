#!/usr/bin/env bash
# Uninstall the Jarvis macOS LaunchAgent.
#
# Usage:
#   bash scripts/uninstall_launch_agent.sh                # remove plist + unload
#   bash scripts/uninstall_launch_agent.sh --purge-keychain  # also delete the Anthropic key
#   bash scripts/uninstall_launch_agent.sh --dry-run      # show planned actions
#
# Symmetric with install_launch_agent.sh. Safe to run when nothing
# is installed — every step is best-effort and reports what it
# actually did vs. skipped.

set -uo pipefail

# ── Parse flags ──────────────────────────────────────────────────
DRY_RUN=0
PURGE_KEYCHAIN=0
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --purge-keychain) PURGE_KEYCHAIN=1 ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "❌ unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

# ── Resolved paths ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PLIST_DEST="${HOME}/Library/LaunchAgents/com.jarvis.daemon.plist"
WRAPPER_RENDERED="${REPO_ROOT}/scripts/launchagent/.jarvis_daemon_wrapper.rendered.sh"
KEYCHAIN_USER="${USER:-$(/usr/bin/id -un)}"

# ── Plan summary ─────────────────────────────────────────────────
echo "📋 Uninstall plan"
echo "   Plist:    ${PLIST_DEST}"
echo "   Wrapper:  ${WRAPPER_RENDERED}"
echo "   Keychain: $([[ ${PURGE_KEYCHAIN} -eq 1 ]] && echo 'WILL be purged' || echo 'preserved (use --purge-keychain to remove)')"
echo ""

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "🧪 Dry-run mode — actions:"
    echo "   1. launchctl unload ${PLIST_DEST}"
    echo "   2. rm ${PLIST_DEST}"
    echo "   3. rm ${WRAPPER_RENDERED}"
    if [[ ${PURGE_KEYCHAIN} -eq 1 ]]; then
        echo "   4. security delete-generic-password -s jarvis-anthropic -a ${KEYCHAIN_USER}"
    fi
    echo ""
    echo "✅ Dry-run complete. Re-run without --dry-run to apply."
    exit 0
fi

echo "🧹 Uninstalling"

# 1. Unload the LaunchAgent. Best-effort: silent if not currently
# loaded (which is the common case if the user already moved on).
if /bin/launchctl list | grep -q com.jarvis.daemon; then
    /bin/launchctl unload "${PLIST_DEST}" 2>/dev/null || true
    echo "   ✅ launchctl unload OK"
else
    echo "   ⏭️  agent not currently loaded — skipping unload"
fi

# 2. Remove the rendered plist.
if [[ -f "${PLIST_DEST}" ]]; then
    rm -f "${PLIST_DEST}"
    echo "   ✅ removed ${PLIST_DEST}"
else
    echo "   ⏭️  plist not present — nothing to remove"
fi

# 3. Remove the rendered wrapper (the .rendered.sh in scripts/launchagent/).
if [[ -f "${WRAPPER_RENDERED}" ]]; then
    rm -f "${WRAPPER_RENDERED}"
    echo "   ✅ removed ${WRAPPER_RENDERED}"
else
    echo "   ⏭️  rendered wrapper not present — nothing to remove"
fi

# 4. Optionally purge the keychain entry.
if [[ ${PURGE_KEYCHAIN} -eq 1 ]]; then
    if /usr/bin/security delete-generic-password \
        -s jarvis-anthropic \
        -a "${KEYCHAIN_USER}" >/dev/null 2>&1; then
        echo "   ✅ deleted keychain entry jarvis-anthropic"
    else
        echo "   ⏭️  no keychain entry to delete (already absent)"
    fi
else
    echo "   ℹ️  keychain entry preserved (use --purge-keychain to remove)"
fi

echo ""
echo "✅ Uninstall complete."
