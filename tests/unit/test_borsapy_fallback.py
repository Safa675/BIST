from __future__ import annotations

import importlib.util
import ssl
import types
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest


def _load_borsapy_client_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "Fetcher-Scrapper"
        / "borsapy_client.py"
    )
    spec = importlib.util.spec_from_file_location("test_borsapy_client_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_client(module, tmp_path: Path):
    config_path = tmp_path / "missing_config.yaml"
    return module.BorsapyClient(
        cache_dir=tmp_path / "cache",
        use_mcp_fallback=True,
        config_path=config_path,
    )


def test_get_financial_statements_ssl_error_fallback(tmp_path: Path):
    module = _load_borsapy_client_module()
    module.BORSAPY_AVAILABLE = True
    module.bp = types.SimpleNamespace(Ticker=Mock(side_effect=ssl.SSLError("ssl failed")))

    client = _build_client(module, tmp_path)
    expected = {
        "balance_sheet": pd.DataFrame({"col1": [1, 2]}),
        "income_stmt": pd.DataFrame({"col2": [3, 4]}),
        "cash_flow": pd.DataFrame({"col3": [5, 6]}),
    }
    client._get_financials_via_mcp = Mock(return_value=expected)

    result = client.get_financial_statements("TEST")

    pd.testing.assert_frame_equal(result["balance_sheet"], expected["balance_sheet"])
    pd.testing.assert_frame_equal(result["income_stmt"], expected["income_stmt"])
    pd.testing.assert_frame_equal(result["cash_flow"], expected["cash_flow"])
    client._get_financials_via_mcp.assert_called_once_with("TEST")
    client.close()


def test_screen_stocks_ssl_error_fallback(tmp_path: Path):
    module = _load_borsapy_client_module()
    module.BORSAPY_AVAILABLE = True
    module.bp = types.SimpleNamespace(
        screen_stocks=Mock(side_effect=ssl.SSLError("ssl failed"))
    )

    client = _build_client(module, tmp_path)
    expected = pd.DataFrame({"symbol": ["TEST"], "price": [100.0]})
    client._screen_via_mcp = Mock(return_value=expected)

    result = client.screen_stocks(template="high_dividend")

    pd.testing.assert_frame_equal(result, expected)
    client._screen_via_mcp.assert_called_once_with(template="high_dividend", filters={})
    client.close()


def test_get_financial_ratios_empty_fallback(tmp_path: Path):
    module = _load_borsapy_client_module()
    module.BORSAPY_AVAILABLE = True
    module.bp = types.SimpleNamespace(
        Ticker=Mock(return_value=types.SimpleNamespace(financial_ratios=pd.DataFrame()))
    )

    client = _build_client(module, tmp_path)
    expected = pd.DataFrame({"pe": [9.8], "pb": [1.2]})
    client._get_ratios_via_mcp = Mock(return_value=expected)

    result = client.get_financial_ratios("TEST")

    pd.testing.assert_frame_equal(result, expected)
    client._get_ratios_via_mcp.assert_called_once_with("TEST")
    client.close()


def test_circuit_breaker_functionality(tmp_path: Path):
    module = _load_borsapy_client_module()
    module.BORSAPY_AVAILABLE = True
    module.bp = types.SimpleNamespace(Ticker=Mock())

    client = _build_client(module, tmp_path)
    breaker = client._circuit_breaker

    for _ in range(breaker.failure_threshold):
        breaker.on_failure()

    assert breaker.state == module.CircuitState.OPEN
    with pytest.raises(module.CircuitBreakerError, match="OPEN"):
        breaker.call(lambda: "ok")
    client.close()


def test_retry_with_backoff_retries_until_success(monkeypatch: pytest.MonkeyPatch):
    module = _load_borsapy_client_module()
    attempts = {"count": 0}
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    @module.retry_with_backoff(max_retries=2, base_delay=0, max_delay=0, jitter=False)
    def flaky():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    assert flaky() == "ok"
    assert attempts["count"] == 3
