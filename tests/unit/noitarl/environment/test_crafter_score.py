from __future__ import annotations

import numpy as np
import pytest

from noitarl.environment.crafter_score import Scoreboard, crafter_score


class TestCrafterScore:
    def test_geometric_mean_formula(self) -> None:
        rates = np.array([100.0, 0.0, 0.0])
        score = crafter_score(rates)
        expected = float(np.exp(np.mean(np.log1p(rates))) - 1.0)
        assert score == pytest.approx(expected)

    def test_scoreboard_tracks_episodes(self) -> None:
        board = Scoreboard()
        board.record_episode({"collect_wood": 1, "wake_up": 1})
        board.record_episode({"wake_up": 1})
        assert board.episodes == 2
        assert board.score >= 0.0
