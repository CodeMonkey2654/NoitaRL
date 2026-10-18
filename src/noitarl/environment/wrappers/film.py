from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from PIL import Image

from noitarl.tracker import Tracker


class Film(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, path: Path, tracker: Tracker) -> None:
        super().__init__(env)
        self._path = path
        self._tracker = tracker
        self._frames: list[Image.Image] = []
        self._episode = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._frames = [Image.fromarray(np.asarray(self.env.render()))]
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(Image.fromarray(np.asarray(self.env.render())))
        if terminated or truncated:
            self._keep()
        return observation, float(reward), bool(terminated), bool(truncated), info

    def _keep(self) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        gif = self._path / f"episode_{self._episode}.gif"
        self._frames[0].save(
            gif,
            save_all=True,
            append_images=self._frames[1:],
            duration=100,
            loop=0,
        )
        self._tracker.log_artifact(gif, "films")
        kept = self._frames[:: max(1, len(self._frames) // 12)]
        thumb_width = 180
        thumbs = [
            frame.resize((thumb_width, int(frame.height * thumb_width / frame.width)))
            for frame in kept
        ]
        sheet = Image.new("RGB", (thumb_width * len(thumbs), thumbs[0].height))
        for index, thumb in enumerate(thumbs):
            sheet.paste(thumb, (index * thumb_width, 0))
        self._tracker.log_image(np.asarray(sheet), f"episode_{self._episode}")
        self._episode += 1
        self._frames = []
