# Design

NoitaRL trains PPO agents on real simulators and logs every run to MLflow. Crafter is the default environment on Mac and Linux. Noita lockstep is the Windows target (`ENVIRONMENT__KIND=noita`, added in a later milestone).

## Entry

`Worker(Settings()).run("train" | "evaluate" | "record")`. Settings come from code defaults, optional `.env`, and env vars with `__` nesting (`AGENT__TOTAL_STEPS=8192`).

## Environment kinds

| Kind | Role |
|------|------|
| `crafter` | Survival benchmark; rgb observations; 17 discrete actions |
| `noita` | Real game via mod + wire (Windows, `@pytest.mark.game`) |

Register a new kind on `Worker.ENVIRONMENTS` and extend `EnvironmentSettings.kind`.

## MLflow contract

One tracked run per Worker invocation. See [mlflow.md](mlflow.md) for metrics and artifacts. Tune behavior by changing settings params logged at run start; full copy lives in `config/settings.json`.

## Tests

- Unit: `tests/unit/` mirrors `src/noitarl/`
- Integration: `tests/integration/` for Worker smoke tests
- Default: `uv run pytest` (excludes `game`)

## Modules

`settings`, `tracker`, `worker`, `agent/ppo` (Impala PPO plus optional achievement distillation), `agent/encoder`, `agent/policy`, `environment/crafter`, `environment/crafter_score`, wrappers `track` and `film`. Only `tracker` owns MLflow run lifecycle. Training logs `score` from the agent loop; evaluate/record use `track` for episode tables.
