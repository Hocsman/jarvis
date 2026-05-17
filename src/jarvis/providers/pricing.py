"""Cloud LLM pricing snapshot.

Standalone module so the pricing table is reviewed and updated in one
place. The telemetry layer (``llm_router_telemetry``) reads from here;
no other module hardcodes per-token rates.

# Pricing snapshot vérifié le 2026-05-17, source :
# https://docs.claude.com/en/docs/about-claude/pricing
"""

from __future__ import annotations

from typing import Optional


# Rates are USD per million tokens. Keys are model IDs (or prefixes).
# Lookup matches the longest prefix that fits, so dated suffix variants
# fall back to the base alias automatically (e.g.
# ``claude-opus-4-7-20251201`` resolves to ``claude-opus-4-7``).
_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "claude-opus-4-7":   {"input": 5.00, "output": 25.00},
    "claude-opus-4-6":   {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input": 1.00, "output":  5.00},
}


# Anthropic cache pricing multipliers applied on the input side.
# Cache writes pay 1.25x the base input rate, reads pay 0.10x.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


def get_pricing(model: str) -> Optional[dict[str, float]]:
    """Return ``{"input": $/1M, "output": $/1M}`` for ``model``.

    Returns ``None`` when the model is not in the table — callers must
    treat unknown models as zero-cost (best-effort, never crash). This
    keeps a future provider rollout from accidentally over-billing the
    telemetry record because someone added a model ID we haven't priced.
    """
    if not model:
        return None
    if model in _PRICING_PER_1M:
        return dict(_PRICING_PER_1M[model])
    # Longest-prefix match for dated variants (e.g. "-20251201").
    candidates = sorted(
        (k for k in _PRICING_PER_1M if model.startswith(k)),
        key=len,
        reverse=True,
    )
    if candidates:
        return dict(_PRICING_PER_1M[candidates[0]])
    return None
