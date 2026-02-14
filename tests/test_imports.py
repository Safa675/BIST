from __future__ import annotations


def test_core_package_imports() -> None:
    import Models
    import Models.common
    import Models.signals

    assert Models is not None
    assert Models.common is not None
    assert Models.signals is not None


def test_signal_config_loads() -> None:
    from Models.common.config_manager import load_signal_configs

    configs = load_signal_configs()
    assert isinstance(configs, dict)
    assert len(configs) > 0
