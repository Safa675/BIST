from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Models.common.enums import RegimeLabel

logger = logging.getLogger(__name__)
REGIME_ALLOCATIONS: dict[RegimeLabel, float] = {
    RegimeLabel.BULL: 1.0,
    RegimeLabel.RECOVERY: 1.0,
    RegimeLabel.STRESS: 0.0,
    RegimeLabel.BEAR: 0.0,
}


DEFAULT_PORTFOLIO_OPTIONS = {
    "use_regime_filter": True,
    "use_vol_targeting": True,
    "target_downside_vol": 0.20,
    "vol_lookback": 63,
    "vol_floor": 0.10,
    "vol_cap": 1.0,
    "use_inverse_vol_sizing": True,
    "inverse_vol_lookback": 60,
    "max_position_weight": 0.25,
    "use_stop_loss": True,
    "stop_loss_threshold": 0.15,
    "use_liquidity_filter": True,
    "liquidity_quantile": 0.25,
    "use_slippage": True,
    "slippage_bps": 5.0,
    "use_mcap_slippage": True,
    "small_cap_slippage_bps": 20.0,
    "mid_cap_slippage_bps": 10.0,
    "top_n": 20,
    "signal_lag_days": 1,
}


TOP_N = 20
LIQUIDITY_QUANTILE = 0.25
POSITION_STOP_LOSS = 0.15
SLIPPAGE_BPS = 5.0
TARGET_DOWNSIDE_VOL = 0.20
VOL_LOOKBACK = 63
VOL_FLOOR = 0.10
VOL_CAP = 1.0
INVERSE_VOL_LOOKBACK = 60
MAX_POSITION_WEIGHT = 0.25

ConfigDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised when strategy configuration is invalid."""


@dataclass(frozen=True)
class ConfigManager:
    project_root: Path
    models_dir: Path

    @classmethod
    def from_default_paths(cls) -> ConfigManager:
        models_dir = Path(__file__).resolve().parents[1]
        project_root = models_dir.parent
        return cls(project_root=project_root, models_dir=models_dir)

    @staticmethod
    def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager.deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _validate_strategy_config(name: str, config: ConfigDict) -> ConfigDict:
        if not isinstance(config, dict):
            raise ConfigError(f"Strategy '{name}' config must be a dict")

        validated = dict(config)
        validated["name"] = name

        description = validated.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ConfigError(f"Strategy '{name}' missing required non-empty 'description'")

        timeline = validated.get("timeline")
        if timeline is not None and not isinstance(timeline, dict):
            raise ConfigError(f"Strategy '{name}' field 'timeline' must be a dict when present")

        parameters = validated.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            raise ConfigError(f"Strategy '{name}' field 'parameters' must be a dict when present")

        signal_params = validated.get("signal_params")
        if signal_params is not None and not isinstance(signal_params, dict):
            raise ConfigError(f"Strategy '{name}' field 'signal_params' must be a dict when present")

        portfolio_options = validated.get("portfolio_options")
        if portfolio_options is not None and not isinstance(portfolio_options, dict):
            raise ConfigError(
                f"Strategy '{name}' field 'portfolio_options' must be a dict when present"
            )

        return validated

    def _yaml_path_candidates(self) -> list[Path]:
        return [
            self.project_root / "configs" / "strategies.yaml",
            self.models_dir / "configs" / "strategies.yaml",
        ]

    def load_yaml_configs(self) -> dict[str, ConfigDict]:
        try:
            import yaml
        except ImportError:
            logger.warning("⚠️  PyYAML not installed. Run: pip install pyyaml")
            return {}

        yaml_path = next((path for path in self._yaml_path_candidates() if path.exists()), None)
        if yaml_path is None:
            return {}

        with open(yaml_path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        if not isinstance(raw, dict):
            raise ConfigError(f"Invalid {yaml_path}: expected top-level mapping")

        defaults = raw.get("defaults", {})
        strategies = raw.get("strategies", {})

        if not isinstance(defaults, dict):
            raise ConfigError(f"Invalid {yaml_path}: 'defaults' must be a mapping")
        if not isinstance(strategies, dict):
            raise ConfigError(f"Invalid {yaml_path}: 'strategies' must be a mapping")

        configs: dict[str, ConfigDict] = {}
        for name, strategy_override in strategies.items():
            if strategy_override is None:
                strategy_override = {}
            if not isinstance(strategy_override, dict):
                raise ConfigError(f"Strategy '{name}' override must be a mapping")

            merged = self.deep_merge(defaults, strategy_override)
            configs[name] = self._validate_strategy_config(name, merged)

        return configs

    def load_legacy_py_configs(self) -> dict[str, ConfigDict]:
        configs: dict[str, ConfigDict] = {}
        config_dir = self.models_dir / "configs"
        if not config_dir.exists():
            return configs

        for config_file in config_dir.glob("*.py"):
            if config_file.name == "__init__.py":
                continue

            try:
                module_name = config_file.stem
                spec = importlib.util.spec_from_file_location(module_name, config_file)
                if spec is None or spec.loader is None:
                    raise ConfigError(f"Cannot load module spec for '{config_file.name}'")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                raw_config = getattr(module, "SIGNAL_CONFIG", None)
                if raw_config is None:
                    continue
                if not isinstance(raw_config, dict):
                    raise ConfigError(
                        f"Legacy config '{config_file.name}' SIGNAL_CONFIG must be a dict"
                    )

                name = raw_config.get("name") or config_file.stem
                configs[name] = self._validate_strategy_config(name, raw_config)
            except Exception as exc:
                logger.warning(f"⚠️  Failed to load config {config_file.name}: {exc}")

        return configs

    def load_signal_configs(self, prefer_yaml: bool = True) -> dict[str, ConfigDict]:
        if prefer_yaml:
            try:
                yaml_configs = self.load_yaml_configs()
                if yaml_configs:
                    return yaml_configs
            except Exception as exc:
                logger.warning(f"⚠️  Failed to load YAML configs: {exc}")
                logger.info("    Falling back to legacy .py configs...")

        configs = self.load_legacy_py_configs()
        if not configs:
            logger.warning("⚠️  No configs loaded from YAML or .py files")
        return configs


def load_signal_configs(prefer_yaml: bool = True) -> dict[str, ConfigDict]:
    manager = ConfigManager.from_default_paths()
    return manager.load_signal_configs(prefer_yaml=prefer_yaml)
