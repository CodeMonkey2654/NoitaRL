from __future__ import annotations

import math

import numpy as np

ACHIEVEMENTS: tuple[str, ...] = (
    "collect_coal",
    "collect_diamond",
    "collect_drink",
    "collect_iron",
    "collect_sapling",
    "collect_stone",
    "collect_wood",
    "defeat_skeleton",
    "defeat_zombie",
    "eat_cow",
    "eat_plant",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_stone_pickaxe",
    "make_stone_sword",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_furnace",
    "place_plant",
    "place_stone",
    "place_table",
    "wake_up",
)


def crafter_score(success_rates_percent: np.ndarray) -> float:
    rates = np.asarray(success_rates_percent, dtype=np.float64)
    return float(math.exp(np.mean(np.log1p(rates))) - 1.0)


class Scoreboard:
    def __init__(self) -> None:
        self._successes = np.zeros(len(ACHIEVEMENTS), dtype=np.int64)
        self._episodes = 0

    def record_episode(self, achievements: dict[str, int]) -> None:
        self._episodes += 1
        for index, name in enumerate(ACHIEVEMENTS):
            if achievements.get(name, 0) > 0:
                self._successes[index] += 1

    def record_success_vector(self, success: np.ndarray) -> None:
        self._episodes += 1
        self._successes += np.asarray(success, dtype=np.int64).reshape(-1)

    @property
    def episodes(self) -> int:
        return self._episodes

    @property
    def success_rates_percent(self) -> np.ndarray:
        if self._episodes == 0:
            return np.zeros(len(ACHIEVEMENTS), dtype=np.float64)
        return 100.0 * self._successes / self._episodes

    @property
    def score(self) -> float:
        return crafter_score(self.success_rates_percent)

    def achievement_metrics(self) -> dict[str, float]:
        rates = self.success_rates_percent
        return {
            f"achievement/{name}": float(rates[index]) for index, name in enumerate(ACHIEVEMENTS)
        }
