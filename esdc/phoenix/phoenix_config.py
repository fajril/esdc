from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class PhoenixConfig:
    enabled: bool
    collector_endpoint: str
    project_name: str

    @classmethod
    def from_config(cls) -> PhoenixConfig:
        from esdc.configs import Config

        cfg = Config.get_phoenix_config()
        return cls(
            enabled=cfg["enabled"],
            collector_endpoint=cfg["collector_endpoint"],
            project_name=cfg["project_name"],
        )

    @classmethod
    def from_env(cls) -> PhoenixConfig:
        enabled = os.environ.get("PHOENIX_ENABLED", "").lower() in ("true", "1", "yes")
        collector_endpoint = os.environ.get(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "http://localhost:4317",
        )
        project_name = os.environ.get("PHOENIX_PROJECT_NAME", "iris")
        return cls(
            enabled=enabled,
            collector_endpoint=collector_endpoint,
            project_name=project_name,
        )
