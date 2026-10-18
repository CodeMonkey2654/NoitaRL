from __future__ import annotations

import mlflow
import pytest

from noitarl.environment.crafter import CrafterEnvironment
from noitarl.environment.wrappers.track import Track
from noitarl.settings import EnvironmentSettings
from noitarl.tracker import Tracker


class TestTrack:
    @pytest.fixture
    def tracked(self, tracker: Tracker) -> Track:
        env = CrafterEnvironment(EnvironmentSettings(episode_length=24))
        return Track(env, tracker)

    def test_summary_empty_before_episode_ends(self, tracked: Track) -> None:
        tracked.reset(seed=0)
        tracked.step(0)
        assert tracked.summary["episodes"] == 0

    def test_logs_return_when_episode_ends(self, tracked: Track, tracker: Tracker) -> None:
        tracked.reset(seed=0)
        for _ in range(30):
            _observation, _reward, terminated, truncated, _info = tracked.step(0)
            if terminated or truncated:
                break
        assert tracked.summary["episodes"] == 1
        run = mlflow.active_run()
        assert run is not None
        client = mlflow.tracking.MlflowClient(tracking_uri=tracker.uri)
        history = client.get_metric_history(run.info.run_id, "return")
        assert len(history) == 1
