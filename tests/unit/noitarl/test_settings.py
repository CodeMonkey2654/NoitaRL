from __future__ import annotations

import pytest

from noitarl.settings import Settings


class TestSettings:
    def test_defaults_load_without_env(self) -> None:
        settings = Settings()
        assert settings.environment.kind == "crafter"
        assert settings.agent.kind == "ppo"

    def test_agent_total_steps_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT__TOTAL_STEPS", "1024")
        settings = Settings()
        assert settings.agent.total_steps == 1024

    def test_flat_keys_use_dotted_paths(self) -> None:
        flat = Settings().flat()
        assert flat["agent.total_steps"] == "4096"
        assert flat["environment.kind"] == "crafter"

    def test_hpc_environment_sentinel_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "BATCH")
        settings = Settings()
        assert settings.environment.kind == "crafter"

    def test_environment_nested_override_with_hpc_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "BATCH")
        monkeypatch.setenv("ENVIRONMENT__EPISODE_LENGTH", "128")
        settings = Settings()
        assert settings.environment.episode_length == 128

    def test_environment_json_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", '{"kind": "crafter", "episode_length": 500}')
        settings = Settings()
        assert settings.environment.episode_length == 500
