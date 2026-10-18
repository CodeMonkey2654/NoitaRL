from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from noitarl.settings import (
    AgentSettings,
    DistillSettings,
    EncoderSettings,
    EnvironmentSettings,
    RunSettings,
    Settings,
    TrackingSettings,
)
from noitarl.tracker import Tracker


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        tracking=TrackingSettings(root=tmp_path / "mlruns"),
        run=RunSettings(name="test"),
        environment=EnvironmentSettings(episode_length=64),
        agent=AgentSettings(
            total_steps=256,
            play_episodes=1,
            n_workers=1,
            n_steps=64,
            distill=DistillSettings(enabled=False),
            encoder=EncoderSettings(channels=[16, 32, 32], outsize=64, hidden=128, nblock=1),
        ),
    )


@pytest.fixture
def tracker(settings: Settings) -> Tracker:
    session = Tracker(settings)
    session.start()
    yield session
    session.finish()


@pytest.fixture(autouse=True)
def _end_mlflow_run() -> None:
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
