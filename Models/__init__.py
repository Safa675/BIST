"""Public package API for the BIST quant library."""

from importlib import import_module
from typing import Any

__all__ = [
    "PortfolioEngine",
    "build_signal",
    "get_available_signals",
    "load_signal_configs",
]


def __getattr__(name: str) -> Any:
    if name == "PortfolioEngine":
        return getattr(import_module("Models.portfolio_engine"), name)
    if name in {"build_signal", "get_available_signals"}:
        return getattr(import_module("Models.signals.factory"), name)
    if name == "load_signal_configs":
        return getattr(import_module("Models.common.config_manager"), name)
    raise AttributeError(f"module 'Models' has no attribute {name!r}")
