from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from noitarl.settings import Settings


class Tracker:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._run_id: str | None = None

    @property
    def uri(self) -> str:
        root = self._settings.tracking.root.resolve()
        return f"sqlite:///{root / 'mlflow.db'}"

    @property
    def root(self) -> Path:
        return self._settings.tracking.root

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def __enter__(self) -> Tracker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if exc_type is None:
            self.finish("FINISHED")
        elif exc_type is KeyboardInterrupt:
            self.finish("KILLED")
        else:
            self.finish("FAILED")

    def start(self) -> None:
        root = self._settings.tracking.root
        root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self._settings.tracking.experiment)
        run = mlflow.start_run(run_name=self._settings.run.name)
        self._run_id = run.info.run_id
        if self._settings.run.notes:
            mlflow.set_tag("mlflow.note.content", self._settings.run.notes)
        mlflow.log_params(self._settings.flat())
        self._log_settings_artifact()
        mlflow.set_tag("environment.kind", self._settings.environment.kind)
        mlflow.set_tag("agent.kind", self._settings.agent.kind)

    def finish(self, status: Literal["FINISHED", "FAILED", "KILLED"] = "FINISHED") -> None:
        mlflow.flush_trace_async_logging()
        mlflow.end_run(status=status)
        self._run_id = None

    def log_metric(self, key: str, value: float, step: int) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_table(self, columns: dict[str, list[Any]], name: str) -> None:
        mlflow.log_table(columns, name)

    def log_artifact(self, path: Path, artifact_path: str) -> None:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_image(self, array: np.ndarray, key: str) -> None:
        mlflow.log_image(array, key=key, synchronous=True)

    def log_policy(self, path: Path) -> None:
        mlflow.log_artifact(str(path), artifact_path="checkpoints")

    def log_return_chart(self, returns: list[float]) -> None:
        if not returns:
            return
        figure, axis = plt.subplots(figsize=(6, 3))
        axis.plot(range(1, len(returns) + 1), returns)
        axis.set_xlabel("episode")
        axis.set_ylabel("return")
        path = self._settings.tracking.root / "return.png"
        figure.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(figure)
        mlflow.log_artifact(str(path), artifact_path="charts")
        path.unlink(missing_ok=True)

    def log_eval_summary(self, payload: dict[str, Any]) -> None:
        path = self._settings.tracking.root / "eval_summary.json"
        path.write_text(json.dumps(payload, indent=2))
        mlflow.log_artifact(str(path), artifact_path="eval")
        path.unlink(missing_ok=True)

    def _log_settings_artifact(self) -> None:
        path = self._settings.tracking.root / "settings.json"
        path.write_text(self._settings.json_text())
        mlflow.log_artifact(str(path), artifact_path="config")
        path.unlink(missing_ok=True)
