from __future__ import annotations

import random
from functools import partial
from pathlib import Path

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import SyncVectorEnv

from noitarl.agent.policy import (
    CrafterPolicy,
    DistillBuffer,
    RolloutStorage,
    run_distill_update,
    run_ppo_update,
)
from noitarl.environment.crafter import CrafterEnvironment
from noitarl.environment.crafter_score import ACHIEVEMENTS, Scoreboard
from noitarl.settings import AgentSettings, EnvironmentSettings
from noitarl.tracker import Tracker


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(name)


class Agent:
    def __init__(
        self,
        settings: AgentSettings,
        env: gymnasium.Env,
        environment: EnvironmentSettings | None = None,
    ) -> None:
        self._settings = settings
        self._environment = environment or EnvironmentSettings()
        self._device = resolve_device(settings.device)
        num_actions = int(env.action_space.n)
        self._policy = CrafterPolicy(num_actions, settings).to(self._device)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=settings.learning_rate)
        self._match_optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=settings.learning_rate
        )
        self._pred_optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=settings.learning_rate
        )
        self._memory = torch.zeros(settings.encoder.hidden, device=self._device)
        self._last_achievements: dict[str, int] | None = None
        self._returns: list[float] = []
        self._scoreboard = Scoreboard()
        random.seed(settings.seed)
        np.random.seed(settings.seed)
        torch.manual_seed(settings.seed)

    def predict(self, observation: np.ndarray) -> int:
        obs = self._obs_tensor(observation)
        state = self._memory.unsqueeze(0)
        with torch.no_grad():
            outputs = self._policy.act(obs, state)
        return int(outputs["actions"].reshape(-1)[0].item())

    def observe_step(self, observation: np.ndarray, info: dict[str, object]) -> None:
        achievements = info.get("achievements")
        if not isinstance(achievements, dict):
            return
        if self._last_achievements is None:
            self._last_achievements = dict(achievements)
            return
        unlocked = any(
            achievements.get(name, 0) > self._last_achievements.get(name, 0)
            for name in ACHIEVEMENTS
        )
        self._last_achievements = dict(achievements)
        if not unlocked:
            return
        obs = self._obs_tensor(observation)
        with torch.no_grad():
            latents = self._policy.encode(obs)
            state = F.normalize(latents, dim=-1).squeeze(0)
        self._memory = state

    def learn(self, steps: int, tracker: Tracker | None = None) -> None:
        seeds = np.random.randint(0, 2**31 - 1, size=self._settings.n_workers)
        env_settings = self._environment
        factories = [
            partial(CrafterEnvironment, env_settings)
            for _ in range(self._settings.n_workers)
        ]
        venv = SyncVectorEnv(factories)
        obs_hwc, _info = venv.reset(seed=seeds.tolist())
        obs = _to_model_obs(obs_hwc, self._device)
        storage = RolloutStorage(
            self._settings.n_steps,
            self._settings.n_workers,
            obs.shape[1:],
            self._settings.encoder.hidden,
            self._device,
        )
        storage.obs[0].copy_(obs)
        buffer = DistillBuffer(maxlen=self._settings.distill.buffer_rollouts)
        rollout_size = self._settings.n_steps * self._settings.n_workers
        updates = max(1, steps // rollout_size)
        env_steps = 0
        ppo_count = 0
        episode_index = 0
        for _ in range(updates):
            rollout_stats = self._sample_rollout(venv, storage)
            storage.compute_returns(self._settings.gamma, self._settings.gae_lambda)
            run_ppo_update(self._policy, storage, self._optimizer, self._settings)
            ppo_count += 1
            keys = ["obs", "actions", "states", "returns", "masks", "rewards", "successes"]
            buffer.insert({key: getattr(storage, key).detach().cpu() for key in keys})
            if (
                self._settings.distill.enabled
                and ppo_count % self._settings.distill.buffer_rollouts == 0
            ):
                run_distill_update(
                    self._policy,
                    buffer,
                    (self._match_optimizer, self._pred_optimizer),
                    self._settings,
                    self._device,
                )
            storage.reset()
            env_steps += rollout_size
            for episode_return, success in zip(
                rollout_stats["returns"], rollout_stats["successes"], strict=False
            ):
                self._scoreboard.record_success_vector(success)
                episode_index += 1
                self._returns.append(float(episode_return))
                if tracker is not None:
                    tracker.log_metric("return", float(episode_return), step=episode_index)
            if tracker is not None:
                tracker.log_metric("score", self._scoreboard.score, step=env_steps)
                for key, value in self._scoreboard.achievement_metrics().items():
                    tracker.log_metric(key, value, step=env_steps)
        venv.close()

    def save(self, path: Path) -> None:
        torch.save(
            {
                "policy": self._policy.state_dict(),
                "agent": self._settings.model_dump(mode="json"),
            },
            path,
        )

    def load(self, path: Path) -> None:
        payload = torch.load(path, map_location=self._device, weights_only=False)
        if isinstance(payload, dict) and "policy" in payload:
            state = payload["policy"]
            if "agent" in payload:
                self._settings = AgentSettings.model_validate(payload["agent"])
                num_actions = int(state["pi_head.linear.weight"].shape[0])
                self._policy = CrafterPolicy(num_actions, self._settings).to(self._device)
        else:
            state = payload
        self._policy.load_state_dict(state)

    def reset_playback(self) -> None:
        self._scoreboard = Scoreboard()
        self._memory.zero_()
        self._last_achievements = None

    @property
    def returns(self) -> list[float]:
        return list(self._returns)

    @property
    def scoreboard(self) -> Scoreboard:
        return self._scoreboard

    @property
    def parameters(self) -> dict[str, str | float | int]:
        return {
            "algorithm": "ppo",
            "learning rate": self._settings.learning_rate,
            "discount": self._settings.gamma,
            "rollout steps": self._settings.n_steps,
            "workers": self._settings.n_workers,
            "entropy bonus": self._settings.ent_coef,
            "device": str(self._device),
        }

    def _obs_tensor(self, observation: np.ndarray) -> torch.Tensor:
        array = np.asarray(observation, dtype=np.float32)
        if array.ndim == 3 and array.shape[-1] == 3:
            array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array / 255.0).unsqueeze(0).to(self._device)
        return tensor

    def _sample_rollout(
        self,
        venv: SyncVectorEnv,
        storage: RolloutStorage,
    ) -> dict[str, np.ndarray]:
        self._policy.eval()
        episode_returns: list[float] = []
        episode_successes: list[np.ndarray] = []
        pending_returns = np.zeros(self._settings.n_workers, dtype=np.float32)
        for step in range(storage.nstep):
            inputs = storage.get_inputs(step)
            outputs = self._policy.act(**inputs)
            actions = outputs["actions"].reshape(-1).cpu().numpy()
            obs_hwc, rewards, terminated, truncated, infos = venv.step(actions)
            dones = np.logical_or(terminated, truncated)
            obs = _to_model_obs(obs_hwc, self._device)
            rewards_t = torch.from_numpy(rewards.astype(np.float32)).to(self._device)
            rewards_t = rewards_t.view(-1, 1)
            masks = torch.from_numpy(1.0 - dones.astype(np.float32)).to(self._device)
            masks = masks.view(-1, 1)
            successes = _success_tensor(infos, self._settings.n_workers, self._device)
            storage.insert(
                self._policy,
                obs,
                outputs["latents"],
                outputs["actions"],
                rewards_t,
                masks,
                outputs["vpreds"],
                outputs["log_probs"],
                successes,
            )
            pending_returns += rewards.astype(np.float32)
            for index, done in enumerate(dones):
                if not done:
                    continue
                episode_returns.append(float(pending_returns[index]))
                pending_returns[index] = 0.0
                achievements = _step_achievements(infos, index)
                episode_successes.append(
                    np.array([1 if achievements.get(name, 0) > 0 else 0 for name in ACHIEVEMENTS])
                )
        bootstrap = storage.get_inputs(-1)
        last_values = self._policy.act(**bootstrap)["vpreds"]
        storage.vpreds[-1].copy_(last_values)
        return {
            "returns": np.array(episode_returns, dtype=np.float32),
            "successes": episode_successes,
        }


def _to_model_obs(obs_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    obs = np.transpose(obs_hwc, (0, 3, 1, 2)).astype(np.float32) / 255.0
    return torch.from_numpy(obs).to(device)


def _step_achievements(infos: dict[str, object], index: int) -> dict[str, int]:
    achievements = infos.get("achievements")
    if isinstance(achievements, dict):
        row: dict[str, int] = {}
        for name in ACHIEVEMENTS:
            if name not in achievements:
                continue
            value = achievements[name]
            if isinstance(value, np.ndarray):
                row[name] = int(value[index])
            else:
                row[name] = int(value)
        if row:
            return row
    if isinstance(achievements, np.ndarray):
        row = achievements[index]
        if isinstance(row, dict):
            return row
    final = infos.get("final_info")
    if isinstance(final, np.ndarray) and final[index] is not None:
        entry = final[index]
        if isinstance(entry, dict):
            return _step_achievements(entry, 0)
    row = {}
    for name in ACHIEVEMENTS:
        if name not in infos:
            continue
        value = infos[name]
        if isinstance(value, np.ndarray):
            row[name] = int(value[index])
    return row


def _success_tensor(
    infos: dict[str, object], count: int, device: torch.device
) -> torch.Tensor:
    rows = [
        [1 if _step_achievements(infos, index).get(name, 0) > 0 else 0 for name in ACHIEVEMENTS]
        for index in range(count)
    ]
    return torch.tensor(rows, device=device, dtype=torch.long)
