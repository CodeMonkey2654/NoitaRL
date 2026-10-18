# MLflow

Store: SQLite at `mlruns/mlflow.db` (override with `TRACKING__ROOT`).

Experiment name: `TRACKING__EXPERIMENT` (default `noitarl`).

## Metrics

| Key | Step | Meaning | Tune via |
|-----|------|---------|----------|
| `return` | Episode index | Total reward for the episode | Reward comes from the env; agent explores via `AGENT__ENT_COEF` |
| `score` | Environment step | Crafter geometric-mean achievement score (percent) | `AGENT__DISTILL__ENABLED`, encoder and PPO settings |
| `achievement/<name>` | Environment step | Success rate for one achievement (percent) | Same as `score` |

## Params

Flattened settings at run start (`agent.learning_rate`, `environment.episode_length`, …). Reproduce by reading `config/settings.json` on the run and exporting the same env vars.

## Artifacts

| Path | Contents |
|------|----------|
| `config/settings.json` | Resolved settings for the run |
| `episodes.json` | Table of per-episode return, steps, outcome, action counts |
| `charts/return.png` | Return curve over episodes |
| `checkpoints/` | `policy.pt` (Impala PPO state dict) |
| `eval/eval_summary.json` | Seeds, returns, mean return after `evaluate` |
| `films/` | Episode GIFs when `record` uses `RUN__FILM_DIR` |

## Traces

`record` and rollout play log spans for choose action and step the world. Training logs a single train span with rollout summary.

## UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns/mlflow.db
```

Open the run, compare params, plot `return`, download checkpoints and charts.
