#!/bin/bash
# Run Jarvis evaluation suite
#
# Usage:
#   ./scripts/run_evals.sh              # Run all evals with both models (live + judge enabled)
#   ./scripts/run_evals.sh weather      # Run only weather-related evals
#   ./scripts/run_evals.sh -v           # Verbose output
#   ./scripts/run_evals.sh --no-live    # Exclude live LLM tests
#   ./scripts/run_evals.sh --no-judge   # Exclude LLM-as-judge tests
#   ./scripts/run_evals.sh --no-report  # Skip EVALS.md generation
#   ./scripts/run_evals.sh --single     # Run with single model only (EVAL_JUDGE_MODEL)
#
# Environment variables:
#   EVAL_JUDGE_MODEL    - Model to use for LLM-as-judge (default: gemma4:e2b,
#                         matching evals/helpers.py — the smallest supported
#                         model is the canary, see MODEL_SMALL below)
#   EVAL_JUDGE_BASE_URL - Ollama base URL (default: http://localhost:11434)
#   EVAL_REPEAT_COUNT   - Number of times to run each test (default: 1; use 3 when tuning prompts to surface flakiness)
#   PYTHON              - Interpreter to run pytest with (default: the first
#                         of python3 / python found on PATH)

set -e

# Resolve the interpreter rather than assuming a bare `python` exists. On a
# machine with only `python3`, or inside an environment that exposes its
# interpreter by full path, the bare name is not found: pytest never starts,
# every suite reports nothing, and the run looks like a completed one that
# happened to fail. Measured here on 2026-08-16 — 338 eval cases, zero of
# them executed.
if [ -z "${PYTHON:-}" ]; then
    if command -v python3 > /dev/null 2>&1; then
        PYTHON="python3"
    elif command -v python > /dev/null 2>&1; then
        PYTHON="python"
    else
        echo "  ❌ No Python interpreter found. Set PYTHON=/path/to/python." >&2
        exit 127
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Officially supported models (from config.py)
MODEL_SMALL="gemma4:e2b"
MODEL_LARGE="gpt-oss:20b"

echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│                  🧪 Jarvis Evaluation Suite                │"
echo "└────────────────────────────────────────────────────────────┘"
echo ""

# Check if Ollama is available
OLLAMA_AVAILABLE=false
OLLAMA_URL="${EVAL_JUDGE_BASE_URL:-http://localhost:11434}"
if curl -s "${OLLAMA_URL}/api/tags" > /dev/null 2>&1; then
    OLLAMA_AVAILABLE=true
    echo "  ✅ Ollama detected at ${OLLAMA_URL}"
else
    echo "  ⚠️  Ollama not detected at ${OLLAMA_URL}"
    echo "     LLM-as-judge tests will be skipped"
fi
echo ""

# Parse arguments (defaults: live=true, judge=true, report=true, multi_model=true)
PYTEST_ARGS="-v"
FILTER=""
INCLUDE_LIVE=true
INCLUDE_JUDGE=true
GENERATE_REPORT=true
MULTI_MODEL=true

for arg in "$@"; do
    case $arg in
        --no-live)
            INCLUDE_LIVE=false
            ;;
        --no-judge)
            INCLUDE_JUDGE=false
            ;;
        --no-report)
            GENERATE_REPORT=false
            ;;
        --single)
            MULTI_MODEL=false
            ;;
        --live)
            INCLUDE_LIVE=true
            ;;
        --judge)
            INCLUDE_JUDGE=true
            ;;
        -v|--verbose)
            PYTEST_ARGS="$PYTEST_ARGS -v"
            ;;
        -vv)
            PYTEST_ARGS="$PYTEST_ARGS -vv"
            ;;
        --*)
            PYTEST_ARGS="$PYTEST_ARGS $arg"
            ;;
        *)
            FILTER="$arg"
            ;;
    esac
done

# Build exclusion filter
EXCLUDE_PATTERNS=""
if [ "$INCLUDE_LIVE" = false ]; then
    EXCLUDE_PATTERNS="Live"
    echo "  ⏭️  Skipping live LLM tests (remove --no-live to include)"
fi

# Function to run evals for a specific model
run_evals_for_model() {
    local model="$1"
    local report_suffix="$2"

    export EVAL_JUDGE_MODEL="$model"

    echo ""
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "  🤖 Running evals with model: $model"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    # Build the pytest command (--tb=short for cleaner tracebacks, -s to capture stdout for judge notes)
    # Each test runs REPEAT_COUNT times for pass rate calculation
    local REPEAT_COUNT="${EVAL_REPEAT_COUNT:-1}"
    local CMD="$PYTHON -m pytest evals/ $PYTEST_ARGS --tb=short --count=$REPEAT_COUNT"

    if [ -n "$FILTER" ]; then
        if [ -n "$EXCLUDE_PATTERNS" ]; then
            CMD="$CMD -k '$FILTER and not $EXCLUDE_PATTERNS'"
        else
            CMD="$CMD -k '$FILTER'"
        fi
    elif [ -n "$EXCLUDE_PATTERNS" ]; then
        CMD="$CMD -k 'not $EXCLUDE_PATTERNS'"
    fi

    echo "  🚀 Command: $CMD"
    echo ""

    # Run with report generation if enabled
    if [ "$GENERATE_REPORT" = true ]; then
        export EVAL_GENERATE_REPORT=1
        export EVAL_REPORT_SUFFIX="$report_suffix"
    fi

    # Run and capture exit code (don't exit on failure)
    set +e
    eval $CMD
    local exit_code=$?
    set -e

    return $exit_code
}

# Run evals
if [ "$GENERATE_REPORT" = true ]; then
    echo "  📄 Report will be saved to EVALS.md"
fi

FINAL_EXIT_CODE=0

if [ "$MULTI_MODEL" = true ] && [ "$OLLAMA_AVAILABLE" = true ]; then
    echo "  🔄 Running evals with both supported models for comparison"

    # Create temp files for individual model reports
    TEMP_DIR=$(mktemp -d)

    # Run with small model
    export EVAL_REPORT_PATH="${TEMP_DIR}/evals_small.md"
    run_evals_for_model "$MODEL_SMALL" "_small" || FINAL_EXIT_CODE=$?

    # Unload all models to avoid VRAM corruption when switching
    echo "  🔄 Unloading models before switching..."
    curl -s "${OLLAMA_URL}/api/generate" -d "{\"model\":\"$MODEL_SMALL\",\"keep_alive\":0}" > /dev/null 2>&1
    sleep 2

    # Run with large model
    export EVAL_REPORT_PATH="${TEMP_DIR}/evals_large.md"
    run_evals_for_model "$MODEL_LARGE" "_large" || FINAL_EXIT_CODE=$?

    # Merge reports into final EVALS.md
    if [ "$GENERATE_REPORT" = true ]; then
        "$PYTHON" "${SCRIPT_DIR}/merge_eval_reports.py" \
            "${TEMP_DIR}/evals_small.md" "$MODEL_SMALL" \
            "${TEMP_DIR}/evals_large.md" "$MODEL_LARGE" \
            > "${PROJECT_ROOT}/EVALS.md"
        echo ""
        echo "  📄 Combined report saved to EVALS.md"
    fi

    # Cleanup temp directory
    rm -rf "$TEMP_DIR"
else
    # Single model mode. Defaults to the small tier, matching
    # evals/helpers.py: two field regressions once slipped past CI because
    # the evals only ran against the larger model, so the canary is the
    # default and the upper tier is the opt-in.
    export EVAL_JUDGE_MODEL="${EVAL_JUDGE_MODEL:-$MODEL_SMALL}"
    export EVAL_REPORT_PATH="${PROJECT_ROOT}/EVALS.md"
    run_evals_for_model "$EVAL_JUDGE_MODEL" "" || FINAL_EXIT_CODE=$?
fi

echo ""
echo "────────────────────────────────────────────────────────────────"
if [ $FINAL_EXIT_CODE -eq 0 ]; then
    echo "  ✅ All evaluations passed!"
elif [ $FINAL_EXIT_CODE -eq 1 ]; then
    echo "  ⚠️  Some evaluations failed (exit code: 1)"
else
    # Anything else is pytest failing to run rather than tests failing:
    # 5 is "no tests collected", 4 a usage error, 126/127 a shell-level
    # failure to start the interpreter at all. Reporting those as failed
    # evaluations is the shape of the bug this suite exists to catch —
    # a run that measured nothing, wearing the face of one that did.
    echo "  ❌ The suite did not run (exit code: $FINAL_EXIT_CODE)."
    case $FINAL_EXIT_CODE in
        5)   echo "     pytest collected no tests. Check the filter argument." ;;
        4)   echo "     pytest usage error. Check the arguments passed through." ;;
        126|127) echo "     '$PYTHON' could not be executed. Set PYTHON=/path/to/python." ;;
        *)   echo "     pytest exited $FINAL_EXIT_CODE before reporting results." ;;
    esac
    echo "     Nothing was measured — this is not a set of failing evals."
fi
echo ""
echo "  📖 Legend:"
echo "     PASSED  → Test passed"
echo "     FAILED  → Test failed"
echo "     SKIPPED → Test skipped (missing dependencies)"
echo "     XFAIL   → Expected failure (documents known limitation)"
echo "     XPASS   → Bug fixed! (expected failure now passes)"
echo ""
if [ "$GENERATE_REPORT" = true ]; then
    echo "  📄 Full report: EVALS.md"
    echo ""
fi
echo "────────────────────────────────────────────────────────────────"

exit $FINAL_EXIT_CODE
