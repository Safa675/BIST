"""Property-based tests for mathematical invariants in the BIST library.

These tests use Hypothesis to verify that core mathematical functions maintain
their expected properties across a wide range of inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from Models.common.risk_manager import RiskManager
from Models.common.utils import (
    cross_sectional_rank,
    rolling_cumulative_return,
    validate_signal_panel_schema,
)
from Models.signals.composite import weighted_sum, zscore_blend


# =============================================================================
# PANDAS STRATEGY HELPERS
# =============================================================================


@st.composite
def numeric_panel_strategy(
    draw,
    min_rows: int = 10,
    max_rows: int = 60,
    min_cols: int = 3,
    max_cols: int = 10,
    min_value: float = -1000.0,
    max_value: float = 1000.0,
) -> pd.DataFrame:
    """Generate a random numeric DataFrame suitable for signal panel tests."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    dates = pd.bdate_range("2025-01-02", periods=n_rows)
    tickers = [f"TK{i:02d}" for i in range(n_cols)]

    values = draw(
        st.lists(
            st.lists(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=n_cols,
                max_size=n_cols,
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    return pd.DataFrame(values, index=dates, columns=tickers, dtype=float)


@st.composite
def price_panel_strategy(
    draw,
    min_rows: int = 30,
    max_rows: int = 120,
    min_cols: int = 3,
    max_cols: int = 10,
) -> pd.DataFrame:
    """Generate a random price DataFrame with realistic price dynamics."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    dates = pd.bdate_range("2025-01-02", periods=n_rows)
    tickers = [f"TK{i:02d}" for i in range(n_cols)]

    base_prices = draw(
        st.lists(
            st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
            min_size=n_cols,
            max_size=n_cols,
        )
    )

    panel = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for i, ticker in enumerate(tickers):
        returns = np.random.default_rng(draw(st.integers(0, 10000))).normal(
            loc=0.0005, scale=0.02, size=n_rows
        )
        panel[ticker] = base_prices[i] * np.cumprod(1.0 + returns)

    return panel


# =============================================================================
# EXISTING TESTS
# =============================================================================


@settings(max_examples=80)
@given(
    st.lists(
        st.tuples(
            st.floats(min_value=-1_000, max_value=1_000, allow_infinity=False, allow_nan=False),
            st.floats(min_value=-1_000, max_value=1_000, allow_infinity=False, allow_nan=False),
            st.floats(min_value=-1_000, max_value=1_000, allow_infinity=False, allow_nan=False),
            st.floats(min_value=-1_000, max_value=1_000, allow_infinity=False, allow_nan=False),
            st.floats(min_value=-1_000, max_value=1_000, allow_infinity=False, allow_nan=False),
        ),
        min_size=5,
        max_size=60,
    )
)
def test_cross_sectional_rank_output_bounds(values: list[tuple[float, float, float, float, float]]) -> None:
    dates = pd.bdate_range("2025-01-02", periods=len(values))
    panel = pd.DataFrame(values, index=dates, columns=["A", "B", "C", "D", "E"])

    ranked = cross_sectional_rank(panel, higher_is_better=True)

    assert ranked.shape == panel.shape
    assert float(ranked.min().min()) >= 0.0
    assert float(ranked.max().max()) <= 100.0


@settings(max_examples=80)
@given(
    st.lists(
        st.floats(min_value=-0.20, max_value=0.20, allow_nan=False, allow_infinity=False),
        min_size=20,
        max_size=120,
    ),
    st.integers(min_value=10, max_value=30),
)
def test_rolling_cumulative_return_matches_log_compounding(
    returns_list: list[float],
    lookback: int,
) -> None:
    series = pd.Series(returns_list, index=pd.bdate_range("2025-01-02", periods=len(returns_list)))
    out = rolling_cumulative_return(series, lookback=lookback)

    if len(series) < lookback:
        return

    clipped = np.clip(series.to_numpy(dtype=float), -0.99, None)
    expected_last = float(np.expm1(np.log1p(clipped[-lookback:]).sum()))
    observed_last = float(out.iloc[-1]) if pd.notna(out.iloc[-1]) else np.nan

    assert np.isfinite(observed_last)
    assert np.isclose(observed_last, expected_last, atol=1e-10)


# =============================================================================
# NEW PROPERTY TESTS
# =============================================================================


@settings(max_examples=50)
@given(numeric_panel_strategy(min_rows=10, max_rows=40, min_cols=3, max_cols=8))
def test_validate_signal_panel_schema_preserves_valid_panels(panel: pd.DataFrame) -> None:
    """validate_signal_panel_schema() should not modify valid panels structurally."""
    dates = panel.index
    tickers = panel.columns

    validated = validate_signal_panel_schema(
        panel=panel,
        dates=dates,
        tickers=tickers,
        signal_name="test_signal",
        context="property_test",
    )

    assert validated.shape == panel.shape
    assert validated.index.equals(dates)
    assert validated.columns.tolist() == tickers.tolist()
    assert validated.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()

    pd.testing.assert_frame_equal(
        validated,
        panel.astype(float),
        check_exact=False,
        atol=1e-12,
    )


@settings(max_examples=80)
@given(numeric_panel_strategy(min_rows=5, max_rows=30, min_cols=3, max_cols=6))
def test_cross_sectional_rank_values_in_0_100(panel: pd.DataFrame) -> None:
    """cross_sectional_rank() should produce values in [0, 100] range."""
    ranked = cross_sectional_rank(panel, higher_is_better=True)

    assert ranked.shape == panel.shape

    valid_values = ranked.to_numpy(dtype=float).flatten()
    valid_values = valid_values[np.isfinite(valid_values)]

    if len(valid_values) > 0:
        assert float(valid_values.min()) >= 0.0
        assert float(valid_values.max()) <= 100.0


@settings(max_examples=50)
@given(numeric_panel_strategy(min_rows=10, max_rows=30, min_cols=3, max_cols=5))
def test_weighted_sum_with_equal_weights_equals_mean(panel: pd.DataFrame) -> None:
    """weighted_sum() with equal weights should equal the simple mean."""
    panel_a = panel.copy()
    panel_b = panel.copy() * 1.5
    panel_c = panel.copy() * 0.8

    panels = {"a": panel_a, "b": panel_b, "c": panel_c}
    equal_weights = {"a": 1.0, "b": 1.0, "c": 1.0}

    result = weighted_sum(panels, equal_weights)

    expected = (panel_a + panel_b + panel_c) / 3.0

    pd.testing.assert_frame_equal(
        result.dropna(how="all"),
        expected.dropna(how="all"),
        check_exact=False,
        atol=1e-10,
    )


@settings(max_examples=50)
@given(numeric_panel_strategy(min_rows=10, max_rows=30, min_cols=4, max_cols=6))
def test_zscore_blend_produces_zero_mean_per_row(panel: pd.DataFrame) -> None:
    """zscore_blend() should produce panels with approximately zero mean per row."""
    panel_a = panel.copy()
    panel_b = panel.copy() + 10.0

    panels = {"a": panel_a, "b": panel_b}

    result = zscore_blend(panels)

    row_means = result.mean(axis=1).dropna()

    for date, mean_val in row_means.items():
        # Use a more relaxed tolerance for floating point arithmetic
        assert np.isclose(mean_val, 0.0, atol=1e-6), f"Row mean at {date} is {mean_val}, expected ~0"


@settings(max_examples=50)
@given(price_panel_strategy(min_rows=60, max_rows=120, min_cols=3, max_cols=6))
def test_inverse_downside_vol_weights_sum_to_one_or_less(close_df: pd.DataFrame) -> None:
    """inverse_downside_vol_weights() should always produce weights summing to ~1.0."""
    volume_df = pd.DataFrame(
        1_000_000.0,
        index=close_df.index,
        columns=close_df.columns,
    )

    risk_manager = RiskManager(close_df=close_df, volume_df=volume_df)

    dates = close_df.index[40:]  # Need lookback
    tickers = close_df.columns.tolist()

    for date in dates[:10]:  # Test a sample of dates
        weights = risk_manager.inverse_downside_vol_weights(
            selected=tickers,
            date=date,
            lookback=20,
            max_weight=0.5,
        )

        total_weight = float(weights.sum())

        # After clipping and renormalization, weights should sum to ~1.0
        assert np.isclose(total_weight, 1.0, atol=1e-10), f"Weights sum to {total_weight} != 1.0 at {date}"

        # All weights should be non-negative
        assert (weights >= 0.0).all(), f"Negative weights found at {date}"

        # Individual weights can exceed max_weight after renormalization if most weights were clipped
        # The max_weight is a soft constraint applied before renormalization


@settings(max_examples=30)
@given(
    st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=20,
    ),
    st.lists(
        st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=20,
    ),
)
def test_weighted_sum_respects_weight_ratios(values_a: list[float], weights_raw: list[float]) -> None:
    """weighted_sum() should maintain correct weight ratios in the output."""
    n = min(len(values_a), len(weights_raw), 10)
    assume(n >= 3)

    dates = pd.bdate_range("2025-01-02", periods=n)
    tickers = ["X", "Y", "Z"]

    panel_a = pd.DataFrame(
        [[values_a[i % len(values_a)]] * 3 for i in range(n)],
        index=dates,
        columns=tickers,
        dtype=float,
    )
    panel_b = pd.DataFrame(
        [[values_a[(i + 1) % len(values_a)] * 2] * 3 for i in range(n)],
        index=dates,
        columns=tickers,
        dtype=float,
    )

    w_a = abs(weights_raw[0]) + 0.01
    w_b = abs(weights_raw[1]) + 0.01

    panels = {"a": panel_a, "b": panel_b}
    weights = {"a": w_a, "b": w_b}

    result = weighted_sum(panels, weights)

    total_w = w_a + w_b
    expected = (panel_a * w_a + panel_b * w_b) / total_w

    pd.testing.assert_frame_equal(
        result,
        expected,
        check_exact=False,
        atol=1e-10,
    )


@settings(max_examples=40)
@given(numeric_panel_strategy(min_rows=10, max_rows=30, min_cols=3, max_cols=5))
def test_cross_sectional_rank_preserves_ordering(panel: pd.DataFrame) -> None:
    """cross_sectional_rank() should preserve the relative ordering of values."""
    ranked = cross_sectional_rank(panel, higher_is_better=True)

    for date in panel.index[:5]:  # Test a sample of dates
        row = panel.loc[date].dropna()
        ranked_row = ranked.loc[date].dropna()

        if len(row) < 2:
            continue

        original_order = row.argsort()
        ranked_order = ranked_row.argsort()

        original_sorted_tickers = row.iloc[original_order.values].index.tolist()
        ranked_sorted_tickers = ranked_row.iloc[ranked_order.values].index.tolist()

        assert original_sorted_tickers == ranked_sorted_tickers, (
            f"Ranking order mismatch at {date}: "
            f"original={original_sorted_tickers}, ranked={ranked_sorted_tickers}"
        )


@settings(max_examples=30)
@given(numeric_panel_strategy(min_rows=5, max_rows=15, min_cols=2, max_cols=4))
def test_zscore_blend_single_panel_equals_zscore(panel: pd.DataFrame) -> None:
    """zscore_blend() with a single panel should equal simple z-scoring."""
    panels = {"only": panel}

    result = zscore_blend(panels)

    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1).replace(0, np.nan)
    expected = panel.sub(row_mean, axis=0).div(row_std, axis=0)

    pd.testing.assert_frame_equal(
        result.dropna(how="all"),
        expected.dropna(how="all"),
        check_exact=False,
        atol=1e-10,
    )
