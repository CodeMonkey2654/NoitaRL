from __future__ import annotations

from collections import Counter
from typing import Any

import gymnasium
import numpy as np

from noitarl.tracker import Tracker


class Track(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, tracker: Tracker) -> None:
        super().__init__(env)
        self._tracker = tracker
        self._return = 0.0
        self._length = 0
        self._episode = 0
        self._rows: list[dict[str, str | int | float]] = []
        self._actions: Counter[int] = Counter()
        self._action_names = tuple(getattr(env.unwrapped, "action_names", ()))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._return = 0.0
        self._length = 0
        self._actions = Counter()
        return self.env.reset(seed=seed, options=options)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        index = int(np.asarray(action).reshape(-1)[0])
        observation, reward, terminated, truncated, info = self.env.step(index)
        self._return += float(reward)
        self._length += 1
        self._actions[index] += 1
        if terminated or truncated:
            self._log_episode(str(info.get("outcome", "timeout")))
        return observation, float(reward), terminated, truncated, info

    def _log_episode(self, outcome: str) -> None:
        number = self._episode + 1
        self._tracker.log_metric("return", self._return, step=number)
        row: dict[str, str | int | float] = {
            "episode": number,
            "return": self._return,
            "steps": self._length,
            "outcome": outcome,
        }
        for index, count in self._actions.items():
            label = self._action_names[index] if index < len(self._action_names) else str(index)
            row[f"action {label}"] = count
        self._rows.append(row)
        self._episode += 1

    def publish(self) -> None:
        if not self._rows:
            return
        keys: list[str] = []
        for row in self._rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        columns = {key: [row.get(key, 0) for row in self._rows] for key in keys}
        self._tracker.log_table(columns, "episodes.json")

    @property
    def returns(self) -> list[float]:
        return [float(row["return"]) for row in self._rows]

    @property
    def summary(self) -> dict[str, float | int]:
        if not self._rows:
            return {"episodes": 0, "mean return": 0.0, "last return": 0.0}
        returns = self.returns
        return {
            "episodes": len(returns),
            "mean return": sum(returns) / len(returns),
            "last return": returns[-1],
        }
