# NoitaRL

Reinforcement learning for Noita with MLflow-backed training on **Crafter** today and **Noita** lockstep on Windows next.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --group dev
uv run pytest
RUN__NAME=baseline uv run noitarl train
uv run mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns/mlflow.db
```

See [documentation/quickstart.md](documentation/quickstart.md) and [documentation/design.md](documentation/design.md).
