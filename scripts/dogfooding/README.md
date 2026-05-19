# Dogfooding helpers

Three small scripts to support a week of personal Jarvis usage with
the Phase 2 stack (orb default, hybrid router, FR pipeline, SHM
audio, optional macOS LaunchAgent). All read-only against the
`llm_router_stats` table and the daemon's stdout log — never write
to anything but `~/.jarvis_dogfooding_notes.md`.

## What's in here

- `jarvis_cost_today.sh` — daily Anthropic spend overview from
  `~/.local/share/jarvis/jarvis.db`. Pass `--all-time` for the
  ever-since totals.
- `jarvis_router_test.sh` — interactive walkthrough that validates
  the four routing flavours (local, cloud, local_fallback, telemetry
  recap). Run mid-week with the daemon already running.
- `jarvis_friction_log.sh` — one-liner append to
  `~/.jarvis_dogfooding_notes.md` for "this annoyed me" moments.
  Optional trailing integer becomes the friction score (1-5).

## Suggested zsh aliases

Add to `~/.zshrc` then `source ~/.zshrc`:

```sh
alias jcost='bash /Users/hocine/jarvis/.claude/worktrees/nice-khayyam-d5b235/scripts/dogfooding/jarvis_cost_today.sh'
alias jtest='bash /Users/hocine/jarvis/.claude/worktrees/nice-khayyam-d5b235/scripts/dogfooding/jarvis_router_test.sh'
alias jflog='bash /Users/hocine/jarvis/.claude/worktrees/nice-khayyam-d5b235/scripts/dogfooding/jarvis_friction_log.sh'
```

Then:

```sh
jcost              # cost today
jcost --all-time   # plus all-time totals
jtest              # router validation walkthrough
jflog "voulais convertir PDF md / hallucination / 4"
```

## Notes

- All three scripts are bash + sqlite3, no Python deps. Fast (<100ms each).
- `jarvis_router_test.sh` snapshots `MAX(id)` on `llm_router_stats` at
  start so each check inspects only rows produced in that session,
  filtering out historical noise.
- `jarvis_friction_log.sh` writes to `~/.jarvis_dogfooding_notes.md`
  with sane formatting on first run; subsequent runs just append.
