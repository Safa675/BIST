from __future__ import annotations

from pathlib import Path

import pytest

from Models.common.config_manager import ConfigError, ConfigManager


def test_yaml_config_loading_and_deep_merge(tmp_path: Path) -> None:
    project_root = tmp_path
    models_dir = tmp_path / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = project_root / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    (cfg_dir / "strategies.yaml").write_text(
        """
defaults:
  description: Base strategy
  parameters:
    lookback: 20
    threshold: 0.5
  portfolio_options:
    use_regime_filter: true
strategies:
  demo:
    description: Demo strategy
    parameters:
      threshold: 0.8
""".strip(),
        encoding="utf-8",
    )

    manager = ConfigManager(project_root=project_root, models_dir=models_dir)
    configs = manager.load_signal_configs(prefer_yaml=True)

    assert "demo" in configs
    assert configs["demo"]["parameters"]["lookback"] == 20
    assert configs["demo"]["parameters"]["threshold"] == 0.8
    assert configs["demo"]["portfolio_options"]["use_regime_filter"] is True


def test_validate_strategy_config_rejects_missing_description() -> None:
    with pytest.raises(ConfigError, match="missing required non-empty 'description'"):
        ConfigManager._validate_strategy_config("bad", {"parameters": {"x": 1}})
