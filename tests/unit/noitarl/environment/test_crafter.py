from __future__ import annotations

import numpy as np
import pytest

from noitarl.environment.crafter import CrafterEnvironment
from noitarl.settings import EnvironmentSettings


class TestCrafterEnvironment:
    @pytest.fixture
    def env(self) -> CrafterEnvironment:
        return CrafterEnvironment(EnvironmentSettings(episode_length=32))

    def test_reset_observation_in_space(self, env: CrafterEnvironment) -> None:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        assert info["seed"] == 0

    def test_same_seed_same_initial_observation(self, env: CrafterEnvironment) -> None:
        first, _ = env.reset(seed=3)
        other = CrafterEnvironment(EnvironmentSettings(episode_length=32))
        second, _ = other.reset(seed=3)
        np.testing.assert_array_equal(first, second)

    def test_step_returns_five_tuple(self, env: CrafterEnvironment) -> None:
        env.reset(seed=0)
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "outcome" in info

    def test_episode_length_truncates(self, env: CrafterEnvironment) -> None:
        env.reset(seed=1)
        truncated = False
        for _ in range(40):
            _observation, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                break
        assert truncated
