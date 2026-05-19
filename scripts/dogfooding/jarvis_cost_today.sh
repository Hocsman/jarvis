#!/usr/bin/env bash
# Daily Anthropic spend overview from the local llm_router_stats table.
#
# Reads ~/.local/share/jarvis/jarvis.db directly via sqlite3 — no Python,
# no daemon dependency. Safe to run while the daemon is live (sqlite WAL
# mode means concurrent readers don't block the writer).
#
# Usage:
#     bash scripts/dogfooding/jarvis_cost_today.sh
#     bash scripts/dogfooding/jarvis_cost_today.sh --all-time   # show ever-since totals too
#
# Output shape (today's section):
#
#   📊 Jarvis usage today (2026-05-19)
#   provider          calls   tokens_in   tokens_out   cost_usd     avg_latency_ms
#   local              42       0           0          $0.0000       240
#   cloud               8     12340         847        $0.0497      3120
#   local_fallback      1       0           0          $0.0000      1820
#   ─────────────────────────────────────────────────────────────────────────
#   Total cloud cost today:       $0.0497
#   Total cloud cost this month:  $0.31

set -uo pipefail

DB="${HOME}/.local/share/jarvis/jarvis.db"

if [[ ! -f "${DB}" ]]; then
    echo "❌ jarvis.db not found at ${DB}"
    echo "   Run Jarvis at least once so the database is created."
    exit 1
fi

# Honour an optional --all-time flag.
SHOW_ALL_TIME=0
for arg in "$@"; do
    case "${arg}" in
        --all-time) SHOW_ALL_TIME=1 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *)
            echo "❌ unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

TODAY="$(date +%Y-%m-%d)"
MONTH_START="$(date +%Y-%m-01)"

echo "📊 Jarvis usage today (${TODAY})"

# Per-provider breakdown for today (UTC vs local: ts_utc is stored as UTC,
# but we filter by the local date prefix on the localtime conversion).
printf '   %-16s %5s %11s %12s %11s %15s\n' \
    "provider" "calls" "tokens_in" "tokens_out" "cost_usd" "avg_latency_ms"

sqlite3 -separator '|' "${DB}" "
SELECT
    provider,
    COUNT(*),
    COALESCE(SUM(tokens_in), 0),
    COALESCE(SUM(tokens_out), 0),
    PRINTF('\$%.4f', COALESCE(SUM(cost_estimate_usd), 0)),
    PRINTF('%d', COALESCE(CAST(AVG(latency_ms) AS INTEGER), 0))
FROM llm_router_stats
WHERE DATE(ts_utc, 'localtime') = '${TODAY}'
GROUP BY provider
ORDER BY provider;
" 2>/dev/null | awk -F'|' '{printf "   %-16s %5s %11s %12s %11s %15s\n", $1, $2, $3, $4, $5, $6}'

echo "   ─────────────────────────────────────────────────────────────────────"

# Cloud totals (we surface these prominently — they are the ones that
# cost money).
TODAY_COST="$(sqlite3 "${DB}" "
SELECT PRINTF('\$%.4f', COALESCE(SUM(cost_estimate_usd), 0))
FROM llm_router_stats
WHERE provider='cloud' AND DATE(ts_utc, 'localtime') = '${TODAY}';
")"
MONTH_COST="$(sqlite3 "${DB}" "
SELECT PRINTF('\$%.4f', COALESCE(SUM(cost_estimate_usd), 0))
FROM llm_router_stats
WHERE provider='cloud' AND DATE(ts_utc, 'localtime') >= '${MONTH_START}';
")"

echo "   Total cloud cost today:       ${TODAY_COST}"
echo "   Total cloud cost this month:  ${MONTH_COST}"

if [[ ${SHOW_ALL_TIME} -eq 1 ]]; then
    echo ""
    echo "📈 All-time"

    printf '   %-16s %5s %11s %12s %11s\n' \
        "provider" "calls" "tokens_in" "tokens_out" "cost_usd"

    sqlite3 -separator '|' "${DB}" "
    SELECT
        provider,
        COUNT(*),
        COALESCE(SUM(tokens_in), 0),
        COALESCE(SUM(tokens_out), 0),
        PRINTF('\$%.4f', COALESCE(SUM(cost_estimate_usd), 0))
    FROM llm_router_stats
    GROUP BY provider
    ORDER BY provider;
    " 2>/dev/null | awk -F'|' '{printf "   %-16s %5s %11s %12s %11s\n", $1, $2, $3, $4, $5}'

    echo "   ──────────────────────────────────────────────────────"
    ALL_TIME="$(sqlite3 "${DB}" "
    SELECT PRINTF('\$%.4f', COALESCE(SUM(cost_estimate_usd), 0))
    FROM llm_router_stats WHERE provider='cloud';
    ")"
    echo "   All-time cloud cost:          ${ALL_TIME}"
fi
