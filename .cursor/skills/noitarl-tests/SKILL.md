---
name: noitarl-tests
description: Adds or rewrites NoitaRL pytest under tests/unit and tests/integration with TestClass per production type. Use when writing tests, fixing test layout, or when the user mentions unit vs integration or class-based pytest.
disable-model-invocation: true
---

# NoitaRL tests

Read `.cursor/rules/testing.mdc` and `.cursor/rules/modules.mdc` first.

## Unit file template

Path mirrors source: `tests/unit/noitarl/test_<module>.py` for `src/noitarl/<module>.py`.

```python
from __future__ import annotations

import pytest

from noitarl.settings import Settings
from noitarl.worker import Worker


class TestWorker:
    def test_run_unknown_command_exits(self, settings: Settings) -> None:
        worker = Worker(settings)
        with pytest.raises(SystemExit):
            worker.run("not-a-command")
```

One `Test<ClassName>` per class defined in the source module. Name methods after behavior: `test_reset_observation_in_space`, not `test_reset_1`.

## Fixtures

Use shared `settings` and MLflow isolation from `tests/conftest.py`. Never point tests at the developer's `mlruns/` or `.env` unless the test explicitly builds overrides.

## Integration

Place in `tests/integration/test_<invariant>.py`. Build `Worker(settings)` with tmp tracking URI and small step or episode budgets. Assert run finished, artifact paths, and metric keys—not loss curves.

Mark Noita-only files with `@pytest.mark.game`.

## Do not

- Add flat `tests/test_foo.py` files.
- Run long training as the only assertion.
- Split one class under test across multiple `Test*` classes without a boundary reason.
