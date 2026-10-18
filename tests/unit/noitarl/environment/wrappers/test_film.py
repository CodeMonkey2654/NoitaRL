from __future__ import annotations

from pathlib import Path

from noitarl.environment.crafter import CrafterEnvironment
from noitarl.environment.wrappers.film import Film
from noitarl.settings import EnvironmentSettings
from noitarl.tracker import Tracker


class TestFilm:
    def test_writes_gif_when_episode_ends(self, tmp_path: Path, tracker: Tracker) -> None:
        base = CrafterEnvironment(EnvironmentSettings(episode_length=24))
        env = Film(base, tmp_path / "films", tracker)
        env.reset(seed=0)
        for _ in range(30):
            _observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
        gifs = list((tmp_path / "films").glob("*.gif"))
        assert len(gifs) == 1
