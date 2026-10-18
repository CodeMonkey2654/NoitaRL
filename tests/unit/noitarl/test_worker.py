from __future__ import annotations

import pytest

from noitarl.settings import Settings
from noitarl.worker import Worker


class TestWorker:
    def test_run_unknown_command_exits(self, settings: Settings) -> None:
        with pytest.raises(SystemExit) as exit_info:
            Worker(settings).run("not-a-command")
        assert exit_info.value.code == 2

    def test_train_creates_mlflow_store(self, settings: Settings) -> None:
        Worker(settings).run("train")
        assert (settings.tracking.root / "mlflow.db").exists()
