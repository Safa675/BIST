from typing import Any

import pandas as pd

from Models.signals.composite import BUILDERS as COMPOSITE_BUILDERS
from Models.signals.momentum import BUILDERS as MOMENTUM_BUILDERS
from Models.signals.protocol import SignalBuilder
from Models.signals.quality import BUILDERS as QUALITY_BUILDERS
from Models.signals.technical import BUILDERS as TECHNICAL_BUILDERS
from Models.signals.value import BUILDERS as VALUE_BUILDERS

ConfigDict = dict[str, Any]

BUILDERS: dict[str, SignalBuilder] = {
    **MOMENTUM_BUILDERS,
    **VALUE_BUILDERS,
    **QUALITY_BUILDERS,
    **TECHNICAL_BUILDERS,
    **COMPOSITE_BUILDERS,
}


def get_available_signals() -> list[str]:
    return sorted(BUILDERS.keys())


def _resolve_signal_params(name: str, config: ConfigDict) -> ConfigDict:
    signal_params = config.get("signal_params", {})
    if signal_params is None:
        signal_params = {}
    if not isinstance(signal_params, dict):
        raise TypeError(
            f"Signal '{name}' expects config['signal_params'] to be dict, got {type(signal_params).__name__}"
        )

    legacy_params = config.get("parameters", {})
    if legacy_params is None:
        legacy_params = {}
    if not isinstance(legacy_params, dict):
        raise TypeError(
            f"Signal '{name}' expects config['parameters'] to be dict, got {type(legacy_params).__name__}"
        )

    merged_params = dict(legacy_params)
    merged_params.update(signal_params)
    return merged_params


def build_signal(
    name: str,
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
) -> pd.DataFrame:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Signal name must be a non-empty string")
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"'dates' must be pd.DatetimeIndex, got {type(dates).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"'config' must be dict, got {type(config).__name__}")

    builder = BUILDERS.get(name)
    if builder is None:
        available = ", ".join(get_available_signals())
        raise ValueError(f"Unknown signal: {name}. Available signals: {available}")

    signal_params = _resolve_signal_params(name, config)
    result = builder(dates=dates, loader=loader, config=config, signal_params=signal_params)

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Signal builder '{name}' must return pd.DataFrame, got {type(result).__name__}"
        )

    return result
