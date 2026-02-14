from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from Models.data_pipeline.fetcher import FundamentalsFetcher
from Models.data_pipeline.pipeline import (
    FundamentalsPipeline,
    build_default_config,
    build_default_paths,
)
from Models.data_pipeline.types import RawDataBundle


def _make_pipeline(tmp_path: Path) -> FundamentalsPipeline:
    paths = build_default_paths(base_dir=tmp_path)
    config = build_default_config(enforce_freshness_gate=False, allow_stale_override=True)
    config = replace(config, request_delay_seconds=0.0, max_retries=1)
    return FundamentalsPipeline(paths=paths, config=config)


def test_pipeline_process_raw_bundle_writes_outputs(
    tmp_path: Path,
    raw_fundamentals_payload: dict,
) -> None:
    pipeline = _make_pipeline(tmp_path)
    raw_bundle = RawDataBundle(
        raw_by_ticker=raw_fundamentals_payload,
        errors=[],
        source_name="fixture_payload",
        fetched_at=datetime.now(timezone.utc),
    )

    result = pipeline.process_raw_bundle(raw_bundle=raw_bundle)

    assert result.outputs
    assert pipeline.paths.normalized_parquet.exists()
    assert pipeline.paths.normalized_csv.exists()
    assert pipeline.paths.consolidated_parquet.exists()
    assert pipeline.paths.freshness_report_csv.exists()
    assert pipeline.paths.quality_metrics_json.exists()
    assert pipeline.paths.cache_state_json.exists()

    quality = json.loads(pipeline.paths.quality_metrics_json.read_text(encoding="utf-8"))
    assert quality["ticker_count"] >= 2


def test_pipeline_cache_hit_on_repeated_payload(
    tmp_path: Path,
    raw_fundamentals_payload: dict,
) -> None:
    pipeline = _make_pipeline(tmp_path)
    raw_bundle = RawDataBundle(
        raw_by_ticker=raw_fundamentals_payload,
        errors=[],
        source_name="fixture_payload",
        fetched_at=datetime.now(timezone.utc),
    )

    first = pipeline.process_raw_bundle(raw_bundle=raw_bundle)
    second = pipeline.process_raw_bundle(raw_bundle=raw_bundle)

    assert first.merged_bundle is not None
    assert second.merged_bundle is not None
    assert second.merged_bundle.merge_stats.get("cache_hit") is True


def test_fetcher_uses_mocked_http_and_writes_raw_cache(
    tmp_path: Path,
    monkeypatch,
    api_value_response: dict,
) -> None:
    import Models.data_pipeline.fetcher as fetcher_module

    paths = build_default_paths(base_dir=tmp_path)
    config = build_default_config(enforce_freshness_gate=False, allow_stale_override=True)
    config = replace(config, request_delay_seconds=0.0, max_retries=1)
    pipeline = FundamentalsPipeline(paths=paths, config=config)
    fetcher = FundamentalsFetcher(config=config, paths=paths, logger=pipeline.logger)

    class _FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.status_code = 200
            self.url = "https://mocked.local/Data.aspx/MaliTablo"
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return dict(self._payload)

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, *args, **kwargs):
            return _FakeResponse(api_value_response)

    monkeypatch.setattr(fetcher_module.httpx, "Client", lambda *args, **kwargs: _FakeClient())

    bundle = fetcher.fetch_tickers(tickers=["AAA"], force=True)

    assert "AAA" in bundle.raw_by_ticker
    assert bundle.errors == []
    assert (paths.raw_dir / "AAA.json").exists()
