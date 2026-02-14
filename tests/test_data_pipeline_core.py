from __future__ import annotations

import pandas as pd

from Models.data_pipeline.freshness import (
    compute_staleness_report,
    evaluate_freshness,
    summarize_quality_metrics,
)
from Models.data_pipeline.merge import merge_consolidated_panels
from Models.data_pipeline.types import FreshnessThresholds


def _make_panel(
    index_rows: list[tuple[str, str, str]],
    values: dict[str, list[float | None]],
) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(index_rows, names=["ticker", "sheet_name", "row_name"])
    frame = pd.DataFrame(values, index=index)
    return frame


def test_vectorized_merge_aligns_row_name_and_fills_missing_cells() -> None:
    existing = _make_panel(
        [("AAA", "Gelir Tablosu (Çeyreklik)", "Net Satışlar")],
        {"2025/9": [100.0], "2025/12": [None]},
    )
    new_data = _make_panel(
        [("AAA", "Gelir Tablosu (Çeyreklik)", "  Net Satışlar  ")],
        {"2025/12": [120.0]},
    )

    merged, stats = merge_consolidated_panels(
        existing=existing,
        new_data=new_data,
        prefer_existing_values=True,
    )

    assert ("AAA", "Gelir Tablosu (Çeyreklik)", "Net Satışlar") in merged.index
    assert merged.loc[("AAA", "Gelir Tablosu (Çeyreklik)", "Net Satışlar"), "2025/12"] == 120.0
    assert stats["cells_filled_from_new"] >= 1


def test_freshness_evaluation_flags_stale_dataset() -> None:
    panel = _make_panel(
        [("AAA", "Bilanço", "Toplam Varlıklar"), ("BBB", "Bilanço", "Toplam Varlıklar")],
        {"2024/9": [100.0, 200.0], "2025/12": [None, None]},
    )
    staleness = compute_staleness_report(panel, reference_date=pd.Timestamp("2026-02-14"))
    quality = summarize_quality_metrics(staleness)

    violations = evaluate_freshness(
        quality_metrics=quality,
        thresholds=FreshnessThresholds(
            max_median_staleness_days=120,
            max_pct_over_120_days=0.50,
            min_q4_coverage_pct=0.10,
            max_max_staleness_days=300,
            grace_days=0,
        ),
    )
    assert violations
