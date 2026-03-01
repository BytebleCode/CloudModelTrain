"""
Configuration loader.

Merges: base.yaml  <-  agents/<agent>.yaml  <-  gpu/<profile>.yaml  <-  CLI overrides
Produces a single flat-ish dict consumed by every other module.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT_DIR / "configs"
REGISTRY_PATH = ROOT_DIR / "registry" / "agents.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_base_config() -> dict:
    return load_yaml(CONFIGS_DIR / "base.yaml")


def load_agent_config(agent_name: str) -> dict:
    path = CONFIGS_DIR / "agents" / f"{agent_name}.yaml"
    if path.exists():
        return load_yaml(path)
    return {}


def load_gpu_profile(profile_name: str, agent_name: str | None = None) -> dict:
    """Load a GPU profile and extract agent-specific overrides if present."""
    path = CONFIGS_DIR / "gpu" / f"{profile_name}.yaml"
    if not path.exists():
        available = [p.stem for p in (CONFIGS_DIR / "gpu").glob("*.yaml")]
        raise ValueError(
            f"Unknown GPU profile '{profile_name}'. Available: {', '.join(available)}"
        )
    data = load_yaml(path)

    # Extract agent-specific overrides from the gpu profile
    agent_overrides = data.pop("agent_overrides", {})
    if agent_name and agent_name in agent_overrides:
        _deep_merge(data, agent_overrides[agent_name])

    return data


def load_registry() -> dict:
    data = load_yaml(REGISTRY_PATH)
    return data.get("agents", {})


def resolve_config(
    agent_name: str,
    run_name: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    gpu_profile: str | None = None,
) -> dict:
    """Build the final merged config for a training run."""

    registry = load_registry()
    if agent_name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(
            f"Unknown agent '{agent_name}'. Available: {available}"
        )

    # --- merge configs: base <- agent <- gpu <- CLI ---
    # GPU profile goes AFTER agent config so hardware-specific batch sizes
    # override the agent's defaults (e.g. H200 batch_size=4 wins over
    # code_writer.yaml's batch_size=2).
    cfg = load_base_config()

    agent_cfg = load_agent_config(agent_name)
    _deep_merge(cfg, agent_cfg)

    if gpu_profile:
        gpu_cfg = load_gpu_profile(gpu_profile, agent_name)
        _deep_merge(cfg, gpu_cfg)
        cfg["gpu_profile"] = gpu_profile

    if cli_overrides:
        _deep_merge(cfg, cli_overrides)

    # --- attach registry info ---
    agent_reg = registry[agent_name]
    cfg["agent_name"] = agent_name
    cfg["agent_registry"] = agent_reg

    # --- resolve output dir ---
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg["run_name"] = run_name
    cfg["output_dir"] = str(
        Path(cfg["output"]["base_dir"]) / agent_name / run_name
    )

    # --- resolve data cache dir ---
    cfg["data_cache_dir"] = str(Path("data_cache") / agent_name)

    return cfg


def save_config_snapshot(cfg: dict, output_dir: str) -> Path:
    """Persist the resolved config for reproducibility."""
    out = Path(output_dir) / "config_snapshot.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out
