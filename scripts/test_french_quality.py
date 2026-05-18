"""French-quality A/B harness.

Runs ten French questions through both the local Ollama chat model
(``cfg.ollama_chat_model``, default ``gemma4:e2b``) and the cloud
Anthropic model (``cfg.llm_router.anthropic_model``, default
``claude-sonnet-4-6``) side by side. The output is a Markdown table
that captures each response, its latency, and its token cost so the
user can pick the right routing strategy for their daily workflow.

The script bypasses the router intentionally: we want a clean A/B,
not the router's intent-based decision. Each question hits both
providers regardless of intent classification.

Usage:
    ANTHROPIC_API_KEY=sk-... PYTHONPATH=src \\
      .mamba_env/bin/python scripts/test_french_quality.py

Output:
    /tmp/jarvis_french_quality_<unix_ts>.md
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Ten questions spread across four intent buckets so the result
# reflects realistic daily use. Two each of casual_chat / simple_query
# / multi_step_reasoning / code_complex, plus two tool_use_chain
# (here we just send the question raw — no tool catalogue — and see
# how each model handles a tool-shaped request).
QUESTIONS: List[Tuple[str, str]] = [
    ("casual_chat", "Salut, tu vas bien ?"),
    ("casual_chat", "Tu peux me raconter une blague courte ?"),
    ("simple_query", "Quelle est la capitale de l'Australie ?"),
    ("simple_query", "Combien de continents sur Terre ?"),
    ("multi_step_reasoning",
     "Compare brièvement le télétravail et le présentiel pour un développeur senior — donne deux avantages et un inconvénient pour chaque."),
    ("multi_step_reasoning",
     "Si je pars à 8h, que je conduis 1h30 puis je m'arrête 20 minutes, et que je reprends pour 45 minutes, à quelle heure j'arrive ?"),
    ("code_complex",
     "Écris une fonction Python qui prend une liste d'entiers et renvoie la somme des nombres premiers qu'elle contient. Commente le code."),
    ("code_complex",
     "Explique en deux phrases pourquoi un async/await mal utilisé peut bloquer une boucle d'événement."),
    ("tool_use_chain",
     "Cherche dans Wikipedia ce qu'est l'effet Doppler puis résume-le en trois phrases simples."),
    ("tool_use_chain",
     "Vérifie la météo à Paris pour demain et propose une activité adaptée."),
]


# A short system prompt that mirrors the production persona without
# the full memory / tool context. We want to measure raw FR quality,
# not the whole agentic loop.
SYSTEM_PROMPT_FR = (
    "Tu es un assistant vocal bref et utile. Réponds toujours en français, "
    "même si la question contient des mots en anglais. Sois concis : "
    "1 à 4 phrases sauf pour le code, où tu donnes le bloc minimum demandé."
)


def _run_local(cfg: Any, system_prompt: str, user_content: str, timeout_sec: float) -> Dict[str, Any]:
    """Call the local Ollama model directly via the private helper so
    we bypass the hybrid router."""
    from jarvis.llm import _call_llm_direct_local

    started = time.monotonic()
    text = _call_llm_direct_local(
        cfg.ollama_base_url,
        cfg.ollama_chat_model,
        system_prompt,
        user_content,
        timeout_sec=timeout_sec,
    )
    elapsed = time.monotonic() - started
    return {
        "provider": "local",
        "model": cfg.ollama_chat_model,
        "text": (text or "").strip() or "(empty)",
        "latency_s": elapsed,
        "tokens_in": None,
        "tokens_out": None,
        "cost_usd": 0.0,
    }


def _run_cloud(cfg: Any, system_prompt: str, user_content: str, timeout_sec: float) -> Dict[str, Any]:
    """Call Anthropic directly via the provider module so the router
    classifier doesn't get involved."""
    from jarvis.providers.anthropic_provider import call_direct, last_usage
    from jarvis.llm_router_telemetry import estimate_cost_usd

    started = time.monotonic()
    text = call_direct(
        base_url=cfg.ollama_base_url,  # accepted for signature compat, unused
        chat_model=cfg.llm_router.anthropic_model,
        system_prompt=system_prompt,
        user_content=user_content,
        timeout_sec=timeout_sec,
        cfg=cfg,
    )
    elapsed = time.monotonic() - started
    usage = last_usage() or {}
    tokens_in = int(usage.get("input_tokens") or 0)
    tokens_out = int(usage.get("output_tokens") or 0)
    cost = estimate_cost_usd(
        model=cfg.llm_router.anthropic_model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
    )
    return {
        "provider": "cloud",
        "model": cfg.llm_router.anthropic_model,
        "text": (text or "").strip() or "(empty)",
        "latency_s": elapsed,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": cost,
    }


def _format_cell(text: str) -> str:
    """Markdown-table-safe cell: replace newlines with <br>, escape pipes."""
    return text.replace("|", "\\|").replace("\n", " <br> ").strip()


def _emoji_for_latency(seconds: float) -> str:
    if seconds < 2.0:
        return "🟢"
    if seconds < 5.0:
        return "🟡"
    return "🔴"


def main() -> int:
    # The script must be invoked from the repo root with PYTHONPATH=src
    # (per the project convention). Bail out clearly if not.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY is not set. Export it and retry.", file=sys.stderr)
        return 2

    try:
        from jarvis.config import load_settings
    except Exception as exc:
        print(f"❌ Cannot import jarvis.config: {exc}. Did you set PYTHONPATH=src?", file=sys.stderr)
        return 2

    cfg = load_settings()
    if (cfg.llm_router.mode or "").lower() == "local_only":
        print("⚠️  cfg.llm_router.mode is 'local_only' — temporarily forcing hybrid for the A/B run.", flush=True)
        cfg.llm_router.mode = "hybrid"

    timestamp = int(time.time())
    output_path = Path(f"/tmp/jarvis_french_quality_{timestamp}.md")

    print(f"🇫🇷 French quality A/B — {len(QUESTIONS)} questions", flush=True)
    print(f"     local model: {cfg.ollama_chat_model}", flush=True)
    print(f"     cloud model: {cfg.llm_router.anthropic_model}", flush=True)
    print(f"     output:      {output_path}", flush=True)
    print(flush=True)

    rows: List[Dict[str, Any]] = []
    total_cost = 0.0
    for i, (intent, question) in enumerate(QUESTIONS, start=1):
        print(f"  [{i:>2}/{len(QUESTIONS)}] [{intent}] {question[:60]}{'…' if len(question) > 60 else ''}", flush=True)
        local = _run_local(cfg, SYSTEM_PROMPT_FR, question, timeout_sec=60.0)
        cloud = _run_cloud(cfg, SYSTEM_PROMPT_FR, question, timeout_sec=60.0)
        total_cost += cloud["cost_usd"]
        rows.append({
            "intent": intent,
            "question": question,
            "local": local,
            "cloud": cloud,
        })
        print(
            f"          local {_emoji_for_latency(local['latency_s'])} {local['latency_s']:>5.2f}s | "
            f"cloud {_emoji_for_latency(cloud['latency_s'])} {cloud['latency_s']:>5.2f}s "
            f"({cloud['tokens_in']}→{cloud['tokens_out']}, ${cloud['cost_usd']:.4f})",
            flush=True,
        )

    # ── Markdown report ─────────────────────────────────────────────
    lines: List[str] = []
    lines.append(f"# French Quality A/B — {timestamp}")
    lines.append("")
    lines.append(f"- Local model: `{cfg.ollama_chat_model}`")
    lines.append(f"- Cloud model: `{cfg.llm_router.anthropic_model}`")
    lines.append(f"- Questions: {len(QUESTIONS)}")
    lines.append(f"- Total cloud cost: **${total_cost:.4f}**")
    lines.append("")
    lines.append("## Side-by-side")
    lines.append("")
    lines.append("| # | Intent | Question | Local reply | Cloud reply | Local lat. | Cloud lat. | Cloud tokens | Cloud $ |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {row['intent']} | {_format_cell(row['question'])} "
            f"| {_format_cell(row['local']['text'])} "
            f"| {_format_cell(row['cloud']['text'])} "
            f"| {row['local']['latency_s']:.2f}s "
            f"| {row['cloud']['latency_s']:.2f}s "
            f"| {row['cloud']['tokens_in']}→{row['cloud']['tokens_out']} "
            f"| ${row['cloud']['cost_usd']:.4f} |"
        )
    lines.append("")

    # Aggregate summary so the user gets a TL;DR before diving into
    # the per-question rows.
    total_local_lat = sum(r["local"]["latency_s"] for r in rows)
    total_cloud_lat = sum(r["cloud"]["latency_s"] for r in rows)
    total_cloud_in = sum(r["cloud"]["tokens_in"] or 0 for r in rows)
    total_cloud_out = sum(r["cloud"]["tokens_out"] or 0 for r in rows)
    lines.append("## Aggregates")
    lines.append("")
    lines.append(f"- **Local total latency**: {total_local_lat:.2f}s ({total_local_lat/len(rows):.2f}s/q)")
    lines.append(f"- **Cloud total latency**: {total_cloud_lat:.2f}s ({total_cloud_lat/len(rows):.2f}s/q)")
    lines.append(f"- **Cloud total tokens**: {total_cloud_in} in / {total_cloud_out} out")
    lines.append(f"- **Cloud total cost**: ${total_cost:.4f}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(flush=True)
    print(f"✅ A/B done — total cloud cost: ${total_cost:.4f}", flush=True)
    print(f"   report: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
