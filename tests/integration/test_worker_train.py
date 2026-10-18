from __future__ import annotations

import mlflow

from noitarl.settings import Settings
from noitarl.tracker import Tracker
from noitarl.worker import Worker


def _client(settings: Settings) -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient(tracking_uri=Tracker(settings).uri)


class TestWorkerTrain:
    def test_train_writes_policy(self, settings: Settings) -> None:
        Worker(settings).run("train")
        assert (settings.tracking.root / "policy.pt").exists()

    def test_train_logs_config_and_chart(self, settings: Settings) -> None:
        Worker(settings).run("train")
        client = _client(settings)
        experiment = client.get_experiment_by_name(settings.tracking.experiment)
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id], max_results=1)
        run_id = runs[0].info.run_id
        config = client.list_artifacts(run_id, "config")
        charts = client.list_artifacts(run_id, "charts")
        assert any(path.path.endswith("settings.json") for path in config)
        assert any(path.path.endswith("return.png") for path in charts)

    def test_train_logs_score(self, settings: Settings) -> None:
        Worker(settings).run("train")
        client = _client(settings)
        experiment = client.get_experiment_by_name(settings.tracking.experiment)
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id], max_results=1)
        metrics = client.get_metric_history(runs[0].info.run_id, "score")
        assert metrics

    def test_train_then_evaluate(self, settings: Settings) -> None:
        Worker(settings).run("train")
        checkpoint = settings.tracking.root / "policy.pt"
        evaluate = Settings(
            tracking=settings.tracking,
            run=settings.run,
            environment=settings.environment,
            agent=settings.agent.model_copy(update={"checkpoint": checkpoint}),
            evaluation=settings.evaluation,
        )
        Worker(evaluate).run("evaluate")
        client = _client(settings)
        experiment = client.get_experiment_by_name(settings.tracking.experiment)
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        eval_artifacts = client.list_artifacts(runs[0].info.run_id, "eval")
        assert any(path.path.endswith("eval_summary.json") for path in eval_artifacts)
