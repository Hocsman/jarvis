# Phase 2B Quarantine Note

**Date imposée**: 2026-05-19
**Branche concernée**: `feature/orb-as-primary` (this branch)
**Statut**: en attente de stabilisation upstream

## Raison de la quarantaine

Phase 2B (FR quality + hybrid LLM router + cloud telemetry capture) entre en collision structurelle avec trois PRs upstream livrées par `isair/jarvis` le 2026-05-08 :

- **#388**: refactor du monolithe `src/jarvis/llm.py` (238 LOC) en un package `src/jarvis/llm/` avec une ABC `LLMBackend` à 5 méthodes (`direct`, `streaming`, `chat`, `embed`, `list_models`).
- **#389**: ajout d'`OpenAICompatibleBackend` + factory dispatch sur `cfg.llm_provider ∈ {"ollama", "openai_compatible"}`.
- **#390**: rename `llm_embed_timeout_sec` → `llm_embedding_timeout_sec` (mécanique, sans impact direct).

Découvertes clés de l'investigation (rapport complet 2026-05-19) :

- L'ABC `LLMBackend` a **exactement le même trou token-usage** que celui que Phase 2B comblait via `_cloud_usage_payload` : `direct()` et `streaming()` renvoient des `str`, donc aucune surface pour les tokens du retour. Adopter l'ABC en l'état ne résout pas le problème, il faudrait l'étendre upstream.
- La factory upstream est **session-statique** (une seule backend par session, choisie par config). Le routing per-call par intent classifié (le cœur de Phase 2B) n'a pas de slot équivalent dans la nouvelle architecture.
- `AnthropicBackend` est mentionné comme provider futur dans le docstring de `factory.py` mais **n'est pas encore implémenté**. Coder le nôtre maintenant risque de dupliquer la PR upstream à venir.

## Conditions de sortie de quarantaine

L'une ou l'autre des conditions suivantes :

1. **`isair:develop` livre un `AnthropicBackend`** — on adapte notre Phase 2B au shape officiel. Notre Anthropic provider devient un concrete `LLMBackend` ; le router hybride devient soit une décoration au-dessus de `get_llm_backend()` soit migre dans le factory si upstream ajoute un hook per-call.
2. **`isair:develop` ajoute un per-call routing hook** dans l'ABC ou dans une couche au-dessus — on contribue notre intent classifier upstream.
3. **4 semaines sans signal upstream sur le sujet** (deadline interne : 2026-06-16). On évalue R1 (rebase + AnthropicBackend custom maintenu localement) comme stratégie finale, assumant la divergence.

## Surveillance recommandée

- Watch `isair/jarvis` releases + PRs taggés `llm` / `backend` / `anthropic` / `router`.
- Cron hebdomadaire : `git fetch origin && git log origin/develop --grep -E "llm|backend|anthropic|router" --since "1 week ago"`.
- Vérifier `factory.py` à chaque release upstream : le retour d'`anthropic_compatible` dans les providers valides est le signal le plus probable.

## Continuer à utiliser Phase 2B en local

La branche `feature/orb-as-primary` reste fonctionnelle pour usage personnel quotidien. Mise en route :

```bash
git checkout feature/orb-as-primary
source .mamba_env/bin/activate

# Configurer le router hybride dans ~/.config/jarvis/config.json
# (cf. docs/HYBRID_LLM.md pour le snippet complet)
export ANTHROPIC_API_KEY=sk-ant-...

bash scripts/run_macos.sh
```

Le router hybride, la capture des tokens cloud, le pipeline FR (`response_language` + `tts_voices` + `select_tts_voice`), et les scripts A/B `scripts/test_french_quality.py` sont tous testés et fonctionnels sur cette branche.

## État des PRs associées

- **`Hocsman/jarvis#1`** (cette branche, full Phase 2) : draft, commenté avec lien vers cette note + PR #2.
- **`Hocsman/jarvis#2`** (`feature/orb-default-and-macos`) : ready for review, base `develop`. Carry Phase 2A + 2C + 2D + 2E sans dépendance LLM. Aucun conflit avec #388/#389/#390 attendu.

## Liens

- Rapport d'investigation : conversation Claude Code datée 2026-05-19, voir transcript local
- PRs upstream : isair/jarvis#388, isair/jarvis#389, isair/jarvis#390 (toutes mergées dans `develop`, pas encore dans `main`)
