from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import gymnasium
import mlflow
import numpy as np
from mlflow.entities import SpanType

from noitarl.agent.ppo import Agent
from noitarl.environment.crafter import CrafterEnvironment
from noitarl.environment.crafter_score import Scoreboard
from noitarl.environment.wrappers.film import Film
from noitarl.environment.wrappers.track import Track
from noitarl.settings import EnvironmentSettings, Settings
from noitarl.tracker import Tracker


class Worker:
    ENVIRONMENTS: dict[str, Callable[[EnvironmentSettings], gymnasium.Env]] = {
        "crafter": CrafterEnvironment,
    }
    AGENTS: dict[str, type[Agent]] = {
        "ppo": Agent,
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracker = Tracker(settings)

    def run(self, command: str) -> None:
        match command:
            case "train":
                self.train()
            case "evaluate":
                self.evaluate()
            case "record":
                self.record()
            case _:
                print("usage: noitarl train | evaluate | record", file=sys.stderr)
                raise SystemExit(2)

    def train(self) -> None:
        with self._tracker:
            agent = self._build_agent(None)
            mlflow.set_tag(
                "mlflow.note.content",
                "Score and return are logged during training. Policy is policy.pt.",
            )
            with mlflow.start_span(name="train", span_type=SpanType.CHAIN) as span:
                span.set_inputs({"training steps": self._settings.agent.total_steps})
                agent.learn(self._settings.agent.total_steps, self._tracker)
                span.set_outputs(
                    {
                        "episodes": len(agent.returns),
                        "mean return": (
                            sum(agent.returns) / len(agent.returns) if agent.returns else 0.0
                        ),
                        "score": agent.scoreboard.score,
                    }
                )
            self._tracker.log_return_chart(agent.returns)
            self._publish_policy(agent)

    def evaluate(self) -> None:
        with self._tracker:
            tracked, agent = self._build_tracked_agent()
            if self._settings.agent.checkpoint is not None:
                agent.load(self._settings.agent.checkpoint)
            agent.reset_playback()
            seeds = self._settings.evaluation.seeds[: self._settings.evaluation.episodes]
            for index, seed in enumerate(seeds):
                observation, _info = tracked.reset(seed=seed)
                self._play_episode_number(tracked, agent, index + 1, observation)
            self._finalize(tracked, eval_seeds=list(seeds), scoreboard=agent.scoreboard)
            self._publish_policy(agent)

    def record(self) -> None:
        with self._tracker:
            base = self._build_base_env()
            film_dir = self._settings.run.film_dir or Path("runs/watch")
            env: gymnasium.Env = Film(base, film_dir, self._tracker)
            tracked = Track(env, self._tracker)
            agent = self._build_agent(tracked)
            if self._settings.agent.checkpoint is not None:
                agent.load(self._settings.agent.checkpoint)
            for index in range(self._settings.agent.play_episodes):
                observation, _info = tracked.reset()
                self._play_episode_number(tracked, agent, index + 1, observation)
            self._finalize(tracked)
            self._publish_policy(agent)

    def _finalize(
        self,
        tracked: Track,
        eval_seeds: list[int] | None = None,
        scoreboard: object | None = None,
    ) -> None:
        tracked.publish()
        self._tracker.log_return_chart(tracked.returns)
        if eval_seeds is not None:
            summary: dict[str, object] = {
                "seeds": eval_seeds,
                "returns": tracked.returns,
                "mean return": tracked.summary["mean return"],
            }
            if isinstance(scoreboard, Scoreboard):
                summary["score"] = scoreboard.score
                summary["achievement rates"] = dict(scoreboard.achievement_metrics())
            self._tracker.log_eval_summary(summary)

    def _build_tracked_agent(self) -> tuple[Track, Agent]:
        tracked = Track(self._build_base_env(), self._tracker)
        return tracked, self._build_agent(tracked)

    def _build_base_env(self) -> gymnasium.Env:
        factory = self.ENVIRONMENTS[self._settings.environment.kind]
        return factory(self._settings.environment)

    def _build_agent(self, env: gymnasium.Env | None) -> Agent:
        agent_cls = self.AGENTS[self._settings.agent.kind]
        if env is None:
            env = self._build_base_env()
        agent = agent_cls(self._settings.agent, env, self._settings.environment)
        if self._settings.agent.checkpoint is not None:
            agent.load(self._settings.agent.checkpoint)
        return agent

    def _play_episode_number(
        self,
        env: Track,
        agent: Agent,
        number: int,
        observation: np.ndarray,
    ) -> None:
        name = f"episode {number}" if number else "episode"
        with mlflow.start_span(name=name, span_type=SpanType.AGENT) as episode:
            episode.set_inputs({"episode": number, **_obs_summary(observation)})
            total = 0.0
            steps = 0
            outcome = "timed out"
            terminated = False
            truncated = False
            while not terminated and not truncated:
                with mlflow.start_span(name="choose action", span_type=SpanType.AGENT) as choice:
                    action = agent.predict(observation)
                    choice.set_inputs(_obs_summary(observation))
                    choice.set_outputs({"action": action})
                with mlflow.start_span(name="step the world", span_type=SpanType.TOOL) as world:
                    observation, reward, terminated, truncated, info = env.step(action)
                    agent.observe_step(observation, info)
                    total += float(reward)
                    steps += 1
                    payload = {**_obs_summary(observation), "reward": float(reward)}
                    if terminated or truncated:
                        outcome = str(info.get("outcome", outcome))
                        payload["outcome"] = outcome
                    world.set_outputs(payload)
                    world.set_inputs({"action": action})
            achievements = info.get("achievements")
            if isinstance(achievements, dict):
                agent.scoreboard.record_episode(achievements)
            episode.set_outputs({"return": total, "steps": steps, "outcome": outcome})

    def _publish_policy(self, agent: Agent) -> None:
        path = self._tracker.root / "policy.pt"
        agent.save(path)
        self._tracker.log_policy(path)


def _obs_summary(observation: np.ndarray) -> dict[str, float]:
    return {"obs mean": float(observation.mean())}
