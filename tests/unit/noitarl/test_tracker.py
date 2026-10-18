from __future__ import annotations

import mlflow

from noitarl.settings import Settings
from noitarl.tracker import Tracker


class TestTracker:
    def test_uri_points_at_sqlite_under_tracking_root(self, settings: Settings) -> None:
        tracker = Tracker(settings)
        expected = settings.tracking.root.resolve() / "mlflow.db"
        assert tracker.uri == f"sqlite:///{expected}"

    def test_start_logs_settings_artifact(self, settings: Settings) -> None:
        tracker = Tracker(settings)
        tracker.start()
        try:
            run = mlflow.active_run()
            assert run is not None
            client = mlflow.tracking.MlflowClient(tracking_uri=tracker.uri)
            artifacts = client.list_artifacts(run.info.run_id, "config")
            assert any(item.path.endswith("settings.json") for item in artifacts)
        finally:
            tracker.finish()
