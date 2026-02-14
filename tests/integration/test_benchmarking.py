from __future__ import annotations

from pathlib import Path

from Models.common.benchmarking import BenchmarkConfig, compare_with_baseline, run_benchmark_suite


def test_benchmark_suite_generates_expected_contract(
    tmp_path: Path,
    raw_fundamentals_payload: dict,
) -> None:
    report = run_benchmark_suite(
        raw_payload=raw_fundamentals_payload,
        config=BenchmarkConfig(repeats=1, warmup=0, days=60, tickers=8, top_n=3),
        tmp_root=tmp_path / "bench_tmp",
    )

    assert "benchmarks" in report
    assert {"backtester_run", "fundamentals_pipeline"}.issubset(report["benchmarks"].keys())
    for name in ("backtester_run", "fundamentals_pipeline"):
        target = report["benchmarks"][name]
        assert target["runs"] == 1
        assert target["median_elapsed_seconds"] >= 0.0
        assert target["median_peak_memory_mb"] >= 0.0


def test_compare_with_baseline_detects_regressions() -> None:
    baseline = {
        "benchmarks": {
            "backtester_run": {
                "median_elapsed_seconds": 1.0,
                "median_peak_memory_mb": 100.0,
            },
            "fundamentals_pipeline": {
                "median_elapsed_seconds": 0.5,
                "median_peak_memory_mb": 50.0,
            },
        }
    }
    current = {
        "benchmarks": {
            "backtester_run": {
                "median_elapsed_seconds": 1.5,
                "median_peak_memory_mb": 130.0,
            },
            "fundamentals_pipeline": {
                "median_elapsed_seconds": 0.6,
                "median_peak_memory_mb": 52.0,
            },
        }
    }

    issues = compare_with_baseline(
        current=current,
        baseline=baseline,
        max_slowdown_pct=20.0,
        max_memory_regression_pct=20.0,
    )
    assert issues
    assert any("backtester_run" in issue for issue in issues)

