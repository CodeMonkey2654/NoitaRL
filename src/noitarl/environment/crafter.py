from __future__ import annotations

from typing import Any

import crafter
import gymnasium
import numpy as np
from gymnasium import spaces

from noitarl.settings import EnvironmentSettings


class CrafterEnvironment(gymnasium.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, settings: EnvironmentSettings) -> None:
        super().__init__()
        self._settings = settings
        self._seed = 0
        self._core = crafter.Env(length=settings.episode_length, seed=self._seed)
        self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.action_space = spaces.Discrete(len(self._core.action_names))
        self.render_mode = "rgb_array"

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(self._core.action_names)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self._core = crafter.Env(
                length=self._settings.episode_length,
                seed=seed,
            )
            self.action_space = spaces.Discrete(len(self._core.action_names))
        observation = self._core.reset()
        return observation, {"seed": self._seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, done, crafter_info = self._core.step(int(action))
        dead = crafter_info["discount"] == 0.0
        terminated = bool(done and dead)
        truncated = bool(done and not dead)
        outcome = ""
        if terminated:
            outcome = "death"
        elif truncated:
            outcome = "timeout"
        info: dict[str, Any] = {
            "outcome": outcome,
            "achievements": crafter_info["achievements"],
        }
        return observation, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._core.render()
