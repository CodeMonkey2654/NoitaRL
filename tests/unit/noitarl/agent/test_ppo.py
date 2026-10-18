from __future__ import annotations

import pytest

from noitarl.agent.ppo import Agent
from noitarl.environment.crafter import CrafterEnvironment
from noitarl.settings import AgentSettings, DistillSettings, EncoderSettings, EnvironmentSettings


class TestAgent:
    @pytest.fixture
    def env(self) -> CrafterEnvironment:
        return CrafterEnvironment(EnvironmentSettings(episode_length=64))

    @pytest.fixture
    def agent(self, env: CrafterEnvironment) -> Agent:
        settings = AgentSettings(
            n_workers=1,
            n_steps=64,
            total_steps=128,
            distill=DistillSettings(enabled=False),
            encoder=EncoderSettings(channels=[16, 32, 32], outsize=64, hidden=128, nblock=1),
        )
        return Agent(settings, env)

    def test_predict_returns_action_in_space(self, env: CrafterEnvironment, agent: Agent) -> None:
        observation, _info = env.reset(seed=0)
        action = agent.predict(observation)
        assert env.action_space.contains(action)

    def test_learn_then_predict_still_in_space(self, env: CrafterEnvironment, agent: Agent) -> None:
        observation, _info = env.reset(seed=0)
        agent.learn(128)
        action = agent.predict(observation)
        assert env.action_space.contains(action)

    def test_parameters_lists_hyperparameters(self, agent: Agent) -> None:
        params = agent.parameters
        assert params["algorithm"] == "ppo"
        assert "learning rate" in params
