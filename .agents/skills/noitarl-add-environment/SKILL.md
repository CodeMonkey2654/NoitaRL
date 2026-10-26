---
name: noitarl-add-environment
description: Registers a new environment.kind in NoitaRL—adapter module, Settings fields, Worker registry line, unit TestClass, integration smoke. Use when adding Crafter-like adapters or wiring Noita lockstep.
disable-model-invocation: true
---

# Add an environment kind

Read `.cursor/rules/architecture.mdc` and `.cursor/rules/environment.mdc` (Noita only).

## Checklist

1. **Module** — `src/noitarl/environment/<name>.py` defines a gymnasium env (or adapter) with `reset`, `step`, `close`, spaces, and project `info` fields agreed in design.md.
2. **Settings** — extend `EnvironmentSettings` with a `Literal` kind and any nested config (paths, ports, seeds).
3. **Registry** — one line on `Worker.ENVIRONMENTS` mapping kind string to class.
4. **Unit tests** — `tests/unit/noitarl/environment/test_<name>.py` with `Test<EnvClass>`: obs in space, step five-tuple, seed determinism where promised, documented errors.
5. **Integration** — if the kind is default for CI, add or extend `tests/integration/test_worker_train.py` with a short train smoke.

## Noita

Kind `noita` uses lockstep wire, frame skip K from settings, and `@pytest.mark.game` integration tests. Do not fake actuation or rewards in Python.

## Non-Noita (e.g. Crafter)

Wrap the upstream env; translate obs/action to what the agent expects. Keep Mac-default pytest on this kind until Noita game tests pass on Windows.

## Do not

- Add a second registry or factory function beside `Worker.ENVIRONMENTS`.
- Import MLflow inside the environment module.
- Leave the old kind registered after removal.
