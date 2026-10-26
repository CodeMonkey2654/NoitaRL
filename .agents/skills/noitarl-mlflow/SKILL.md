---
name: noitarl-mlflow
description: Configures NoitaRL MLflow runs via Tracker—params, tags, settings.json, metrics, checkpoints, traces, chart artifacts. Use when logging training, evaluation, or documenting metrics for fine-tuning.
disable-model-invocation: true
---

# NoitaRL MLflow

Read `.cursor/rules/architecture.mdc` and `documentation/mlflow.md` after changing behavior.

## Run record

`Tracker(settings)` owns the active run. On `start`:

- Set tracking URI and experiment from settings.
- Log flattened settings as MLflow params (truncate long values; full copy in `settings.json` artifact).
- Set tags: `environment.kind`, `agent.kind`, run name, and stage when present.

On `finish`, end the run with FINISHED, FAILED, or KILLED.

Only `tracker.py` imports `mlflow` for lifecycle. Other code sends metrics through `Tracker` methods or episode hooks the worker owns.

## Metrics

Prefer names a human can tune without reading code:

| Key | Meaning |
|-----|---------|
| `return` | Episode total reward (step = episode index) |
| `eval/<suite>/return_mean` | Mean return over a fixed eval seed list (future) |
| `env/desync_per_1k` | Noita lockstep health; must stay 0 in production |

Add new keys to `documentation/design.md` in the same change.

## Artifacts

- `settings.json` — full resolved settings for reproduction.
- `checkpoints/` — policy zip at intervals.
- `charts/return.png` — matplotlib return curve after train, evaluate, or record.
- `eval/eval_summary.json` — seeds, returns, mean return after `evaluate`.
- Optional traces/spans for play or debug runs; keep training traces bounded.

## Fine-tuning story

Params come from env vars on `Settings`. Document which env var moves which metric. Avoid duplicate logging paths (wrapper plus tracker logging the same scalar twice).
