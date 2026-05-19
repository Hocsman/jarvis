#!/usr/bin/env bash
# Tiny append-only logger for Jarvis dogfooding friction notes.
#
# Usage:
#     bash scripts/dogfooding/jarvis_friction_log.sh "voulais convertir PDF md / a halluciné section / 2"
#
# Each call appends one line to ~/.jarvis_dogfooding_notes.md in the
# format:
#     - 2026-05-19T14:30 · <description> · friction X
#
# The trailing "friction X" suffix is optional — if your note ends with
# a number, the script treats it as the friction score (1 = mild, 5 =
# severe). Otherwise it's left as-is.
#
# First run prints the suggested zsh alias so you can call it as just
# ``jflog "..."`` in your shell history.

set -uo pipefail

NOTES="${HOME}/.jarvis_dogfooding_notes.md"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '2,15p' "$0"
    exit 0
fi

# Concatenate all positional args so the user can quote loosely.
NOTE="$*"

# Optional friction score (trailing integer 1-5).
if [[ "${NOTE}" =~ ([0-9])[[:space:]]*$ ]]; then
    SCORE="${BASH_REMATCH[1]}"
    BODY="${NOTE% ${SCORE}}"
    LINE="- $(date +%Y-%m-%dT%H:%M) · ${BODY} · friction ${SCORE}"
else
    LINE="- $(date +%Y-%m-%dT%H:%M) · ${NOTE}"
fi

# Create the file with a header on first use.
if [[ ! -f "${NOTES}" ]]; then
    cat > "${NOTES}" <<EOF
# Jarvis dogfooding notes

Append-only friction log. One line per moment.

Format:
\`- <iso-timestamp> · <description> · friction <1-5>\`

Friction scale: 1 = barely noticed, 3 = noticeable annoyance,
5 = stopped using Jarvis for that task.

---

EOF
    FIRST_RUN=1
fi

echo "${LINE}" >> "${NOTES}"
echo "📝 logged: ${LINE}"
echo "   file:   ${NOTES}"

if [[ "${FIRST_RUN:-0}" -eq 1 ]]; then
    echo ""
    echo "💡 First run. Suggested zsh alias for daily use:"
    echo "     alias jflog='bash ${SCRIPT_DIR}/jarvis_friction_log.sh'"
    echo "   Add it to ~/.zshrc and reload (source ~/.zshrc) so you can"
    echo "   call \`jflog \"...\"\` directly without the long path."
fi
