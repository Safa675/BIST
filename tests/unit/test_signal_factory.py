from __future__ import annotations

import pandas as pd
import pytest

from Models.signals import factory


def test_get_available_signals_is_non_empty() -> None:
    names = factory.get_available_signals()
    assert isinstance(names, list)
    assert names


def test_resolve_signal_params_merges_legacy_and_new() -> None:
    config = {
        "parameters": {"lookback": 20, "alpha": 1},
        "signal_params": {"alpha": 2, "beta": 3},
    }
    merged = factory._resolve_signal_params("demo", config)
    assert merged == {"lookback": 20, "alpha": 2, "beta": 3}


def test_build_signal_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown signal"):
        factory.build_signal(
            name="__does_not_exist__",
            dates=pd.bdate_range("2025-01-02", periods=5),
            loader=None,
            config={},
        )


def test_build_signal_validates_builder_return_type(monkeypatch: pytest.MonkeyPatch) -> None:
    def _bad_builder(*_args, **_kwargs):
        return "not-a-dataframe"

    monkeypatch.setattr(factory, "BUILDERS", {"demo": _bad_builder})

    with pytest.raises(TypeError, match="must return pd.DataFrame"):
        factory.build_signal(
            name="demo",
            dates=pd.bdate_range("2025-01-02", periods=4),
            loader=None,
            config={},
        )


def test_build_signal_passes_merged_params_to_builder(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, dict] = {}

    def _builder(*, dates, loader, config, signal_params):
        calls["signal_params"] = dict(signal_params)
        return pd.DataFrame(1.0, index=dates, columns=["AAA"])

    monkeypatch.setattr(factory, "BUILDERS", {"demo": _builder})

    out = factory.build_signal(
        name="demo",
        dates=pd.bdate_range("2025-01-02", periods=4),
        loader=object(),
        config={"parameters": {"x": 1}, "signal_params": {"x": 2, "y": 3}},
    )

    assert isinstance(out, pd.DataFrame)
    assert calls["signal_params"] == {"x": 2, "y": 3}
