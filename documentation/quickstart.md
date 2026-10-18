# Quickstart

## Install

```bash
uv sync --group dev
```

## Check

```bash
uv run ruff check src tests
uv run ty check src
uv run pytest
```

## Train

```bash
RUN__NAME=baseline AGENT__TOTAL_STEPS=4096 uv run noitarl train
```

Paper-scale Crafter run (1M environment steps, 8 workers, achievement distillation on, MPS when available):

```bash
RUN__NAME=crafter-1m AGENT__TOTAL_STEPS=1024000 uv run noitarl train
```

Policy weights: `mlruns/policy.pt`. Metrics include `score` and `return`; `charts/return.png` and `config/settings.json` are on the run in MLflow.

## Evaluate

After training:

```bash
AGENT__CHECKPOINT=mlruns/policy.pt uv run noitarl evaluate
```

Uses `EVALUATION__SEEDS` and `EVALUATION__EPISODES` from settings. Writes `eval/eval_summary.json`.

## Record

GIFs and traces:

```bash
AGENT__CHECKPOINT=mlruns/policy.pt RUN__FILM_DIR=runs/watch uv run noitarl record
```

## MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns/mlflow.db
```
