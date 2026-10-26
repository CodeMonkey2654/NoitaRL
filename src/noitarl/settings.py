from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource, PydanticBaseSettingsSource


class RunSettings(BaseModel):
    name: str = "unnamed"
    notes: str = ""
    film_dir: Path | None = None


class EnvironmentSettings(BaseModel):
    kind: Literal["crafter"] = "crafter"
    episode_length: int = 1000


class EncoderSettings(BaseModel):
    channels: list[int] = Field(default_factory=lambda: [64, 128, 128])
    outsize: int = 256
    nblock: int = 2
    hidden: int = 1024


class DistillSettings(BaseModel):
    enabled: bool = True
    buffer_rollouts: int = 8
    aux_epochs: int = 6
    transport_reg: float = 0.05
    match_threshold: float = 0.5
    pi_coef: float = 1.0
    vf_coef: float = 1.0
    temperature: float = 0.1


class AgentSettings(BaseModel):
    kind: Literal["ppo"] = "ppo"
    total_steps: int = 4096
    play_episodes: int = 3
    learning_rate: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.65
    n_workers: int = 8
    n_steps: int = 512
    ppo_epochs: int = 3
    n_minibatches: int = 8
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    value_ewma: float = 0.99
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 0
    checkpoint: Path | None = None
    encoder: EncoderSettings = Field(default_factory=EncoderSettings)
    distill: DistillSettings = Field(default_factory=DistillSettings)


class EvaluationSettings(BaseModel):
    episodes: int = 5
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class TrackingSettings(BaseModel):
    root: Path = Path("mlruns")
    experiment: str = "noitarl"


class _HpcSafeEnvSettingsSource(EnvSettingsSource):
    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        env_val, field_key, value_is_complex = super().get_field_value(field, field_name)
        if (
            field_name == "environment"
            and isinstance(env_val, str)
            and not env_val.lstrip().startswith(("{", "["))
        ):
            return None, field_key, value_is_complex
        return env_val, field_key, value_is_complex


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    run: RunSettings = RunSettings()
    environment: EnvironmentSettings = EnvironmentSettings()
    agent: AgentSettings = AgentSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    tracking: TrackingSettings = TrackingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        hpc_env = _HpcSafeEnvSettingsSource(settings_cls, _init_state=env_settings._init_state)
        return init_settings, hpc_env, dotenv_settings, file_secret_settings

    def flat(self) -> dict[str, str]:
        return _flatten("", self.model_dump(mode="json"))

    def json_text(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2)


def _flatten(prefix: str, value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(path, item))
        return out
    return {prefix: str(value)}
