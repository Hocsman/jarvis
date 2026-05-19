#!/usr/bin/env bash
# Interactive router validation walkthrough — run mid-week to confirm
# the hybrid routing is doing what you think it's doing.
#
# Walks through four checks:
#   1. Trivial query  -> expect local_only (gemma4:e2b)
#   2. Complex query  -> expect cloud (claude-sonnet-4-6)
#   3. Wifi-off       -> expect local_fallback (cloud unreachable)
#   4. Telemetry recap -> show last 10 rows + cost tally
#
# For each check the script:
#   - Prints the question to ask Jarvis aloud
#   - Pauses with a press-enter prompt
#   - Queries llm_router_stats to inspect the latest row(s)
#   - Surfaces the verdict (expected vs observed) with ✅/❌ visual.
#
# Usage:
#   bash scripts/dogfooding/jarvis_router_test.sh
#
# Prerequisites:
#   - Jarvis daemon must be running (otherwise no rows get written).
#   - ANTHROPIC_API_KEY must be exported in the daemon's environment
#     for checks 2 and 3 to be meaningful.

set -uo pipefail

DB="${HOME}/.local/share/jarvis/jarvis.db"

if [[ ! -f "${DB}" ]]; then
    echo "❌ jarvis.db not found at ${DB}"
    exit 1
fi

# Snapshot the row count at start so each check inspects only rows
# created after that moment. Avoids confusion from historical rows.
SESSION_BASE_ID="$(sqlite3 "${DB}" 'SELECT COALESCE(MAX(id), 0) FROM llm_router_stats;')"

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════"
}

pause_for_user() {
    echo ""
    read -r -p "  ⏸️  $1 (press Enter when done) "
}

# Latest row added by the daemon since check_start_id.
last_row_since() {
    local start_id="$1"
    sqlite3 -separator '|' "${DB}" "
    SELECT id, datetime(ts_utc, 'localtime'), provider, model, intent,
           tokens_in, tokens_out, latency_ms,
           PRINTF('\$%.4f', cost_estimate_usd)
    FROM llm_router_stats
    WHERE id > ${start_id}
    ORDER BY id DESC LIMIT 1;
    "
}

# All rows added since check_start_id (for tool-use turns that produce
# multiple llm calls).
rows_since() {
    local start_id="$1"
    sqlite3 -separator '|' "${DB}" "
    SELECT id, datetime(ts_utc, 'localtime'), provider, model, intent,
           tokens_in, tokens_out, latency_ms,
           PRINTF('\$%.4f', cost_estimate_usd)
    FROM llm_router_stats
    WHERE id > ${start_id}
    ORDER BY id ASC;
    "
}

format_rows() {
    awk -F'|' 'BEGIN {
        printf "   %-4s %-19s %-15s %-22s %-22s %5s %5s %6s %s\n",
            "id", "ts", "provider", "model", "intent", "in", "out", "ms", "cost"
    }
    {
        printf "   %-4s %-19s %-15s %-22s %-22s %5s %5s %6s %s\n",
            $1, $2, $3, $4, $5, $6, $7, $8, $9
    }'
}

# ── Check 1 — trivial query stays local ─────────────────────────────────
print_header "Check 1/4 — trivial query (expect: local)"
CHECK1_START="$(sqlite3 "${DB}" 'SELECT COALESCE(MAX(id), 0) FROM llm_router_stats;')"
echo ""
echo "  🎤 Ask Jarvis:"
echo '       "Jarvis, salut, comment vas-tu ?"'
echo ""
echo "  Expected: gemma4:e2b answers locally (intent = casual_chat),"
echo "            no cloud call, no API spend."
pause_for_user "Spoken the question + got the reply?"

rows_since "${CHECK1_START}" | format_rows
LAST="$(last_row_since "${CHECK1_START}")"
PROVIDER="$(echo "${LAST}" | cut -d'|' -f3)"
if [[ "${PROVIDER}" == "local" ]]; then
    echo "   ✅ provider = local — routing correct"
elif [[ -z "${PROVIDER}" ]]; then
    echo "   ❌ no rows recorded since check start — is the daemon running?"
else
    echo "   ❌ provider = '${PROVIDER}' — expected 'local'"
fi

# ── Check 2 — complex query goes cloud ─────────────────────────────────
print_header "Check 2/4 — complex query (expect: cloud + Sonnet 4.6)"
CHECK2_START="$(sqlite3 "${DB}" 'SELECT COALESCE(MAX(id), 0) FROM llm_router_stats;')"
echo ""
echo "  🎤 Ask Jarvis:"
echo '       "Jarvis, écris-moi une fonction Python qui détecte les'
echo '        nombres premiers, avec deux variantes — une naïve et une'
echo '        optimisée — et explique-moi la complexité de chaque."'
echo ""
echo "  Expected: code_complex intent → cloud → claude-sonnet-4-6,"
echo "            non-zero tokens, response in French."
pause_for_user "Got Jarvis's reply?"

rows_since "${CHECK2_START}" | format_rows
LAST="$(last_row_since "${CHECK2_START}")"
PROVIDER="$(echo "${LAST}" | cut -d'|' -f3)"
MODEL="$(echo "${LAST}" | cut -d'|' -f4)"
TOKENS_IN="$(echo "${LAST}" | cut -d'|' -f6)"
if [[ "${PROVIDER}" == "cloud" && "${MODEL}" == "claude-sonnet-4-6" ]]; then
    if [[ "${TOKENS_IN}" -gt 0 ]]; then
        echo "   ✅ cloud + sonnet-4-6 + ${TOKENS_IN} tokens in — routing + telemetry correct"
    else
        echo "   ⚠️  cloud + sonnet-4-6 but tokens_in=0 — telemetry capture path may be off"
    fi
elif [[ "${PROVIDER}" == "local_fallback" ]]; then
    echo "   ❌ provider = local_fallback — cloud call failed (key invalid? no network?)"
elif [[ -z "${PROVIDER}" ]]; then
    echo "   ❌ no rows recorded — is the daemon running?"
else
    echo "   ❌ provider='${PROVIDER}' model='${MODEL}' — expected cloud + claude-sonnet-4-6"
fi

# ── Check 3 — wifi off triggers local_fallback ─────────────────────────
print_header "Check 3/4 — wifi-off (expect: local_fallback)"
echo ""
echo "  This check verifies the router's graceful degradation when"
echo "  the cloud is unreachable. We'll turn wifi OFF, ask a query"
echo "  that would normally route to cloud, then turn wifi back ON."
echo ""
echo "  🔌 To turn wifi OFF:"
echo "       networksetup -setairportpower en0 off"
echo ""
read -r -p "  ⏸️  Wifi turned OFF? (press Enter to continue) "

CHECK3_START="$(sqlite3 "${DB}" 'SELECT COALESCE(MAX(id), 0) FROM llm_router_stats;')"
echo ""
echo "  🎤 Ask Jarvis:"
echo '       "Jarvis, raconte-moi comment fonctionne la blockchain en'
echo '        détail, avec ses limites et alternatives modernes."'
echo ""
echo "  Expected: classifier → multi_step_reasoning → tries cloud →"
echo "            cloud unreachable → local_fallback → gemma4:e2b answers."
pause_for_user "Got Jarvis's reply?"

rows_since "${CHECK3_START}" | format_rows
LAST="$(last_row_since "${CHECK3_START}")"
PROVIDER="$(echo "${LAST}" | cut -d'|' -f3)"
if [[ "${PROVIDER}" == "local_fallback" ]]; then
    echo "   ✅ provider = local_fallback — graceful degradation correct"
elif [[ "${PROVIDER}" == "local" ]]; then
    echo "   ⚠️  provider = local — classifier sent it to local directly (not a fallback)"
    echo "       Possible cause: classifier itself needs internet for cloud classification,"
    echo "       and without wifi it returned an intent that's not in cloud_intents."
elif [[ -z "${PROVIDER}" ]]; then
    echo "   ❌ no rows recorded since check start"
else
    echo "   ❌ provider = '${PROVIDER}' — expected 'local_fallback'"
fi

echo ""
echo "  🔌 Now turn wifi back ON:"
echo "       networksetup -setairportpower en0 on"
read -r -p "  ⏸️  Wifi turned ON? (press Enter to continue) "

# ── Check 4 — telemetry recap ──────────────────────────────────────────
print_header "Check 4/4 — telemetry recap (last 10 rows of this session)"

sqlite3 -separator '|' "${DB}" "
SELECT id, datetime(ts_utc, 'localtime'), provider, model, intent,
       tokens_in, tokens_out, latency_ms,
       PRINTF('\$%.4f', cost_estimate_usd)
FROM llm_router_stats
WHERE id > ${SESSION_BASE_ID}
ORDER BY id DESC LIMIT 10;
" | format_rows

echo ""
echo "  Session cost so far:"
sqlite3 "${DB}" "
SELECT '   provider=' || provider || ': ' || COUNT(*) || ' calls, '
       || PRINTF('\$%.4f', SUM(cost_estimate_usd))
FROM llm_router_stats
WHERE id > ${SESSION_BASE_ID}
GROUP BY provider
ORDER BY provider;"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Router test walkthrough complete. Notes ?"
echo "       bash scripts/dogfooding/jarvis_friction_log.sh '<note>'"
echo "════════════════════════════════════════════════════════════════"
