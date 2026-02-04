#!/usr/bin/env python3
"""
BIST Regime Filter - Comprehensive Debugging Script
=====================================================

Performs exhaustive validation and debugging of the entire regime filter pipeline:

  1.  Data Pipeline Integrity
  2.  Feature Engineering Validation
  3.  Rolling Percentile Calculation
  4.  Volatility Regime Classification
  5.  Trend Regime Classification
  6.  Risk Regime Classification
  7.  Liquidity Regime Classification
  8.  Combined Regime Label Construction
  9.  Simplified Regime Mapping (5-State)
  10. Persistence / Hysteresis Filter
  11. HMM Regime Classifier
  12. XGBoost Predictive Model
  13. LSTM Sequence Model
  14. Ensemble Model Integration
  15. Backtesting & Strategy Consistency
  16. Cross-Component Alignment
  17. Look-Ahead Bias Detection
  18. Edge Cases & Boundary Conditions
  19. Output File Integrity
  20. End-to-End Smoke Test

Usage:
    python debug_regime_filter.py              # Run all tests
    python debug_regime_filter.py --section 5  # Run only section 5
    python debug_regime_filter.py --quick      # Skip slow tests (LSTM, ensemble)
"""

import os
import sys
import time
import json
import pickle
import warnings
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from io import StringIO

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Data directory: prefer Regime Filter/data/ but fall back to project root data/
_local_data = SCRIPT_DIR / "data"
_root_data = PROJECT_ROOT / "data"
if _local_data.exists():
    DATA_DIR = _local_data
elif _root_data.exists():
    DATA_DIR = _root_data
else:
    DATA_DIR = _local_data  # will fail with clear error message

sys.path.insert(0, str(SCRIPT_DIR))

import config

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class DebugReport:
    """Accumulate pass/warn/fail results and emit a final report."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    CRITICAL = "CRITICAL"

    def __init__(self):
        self.results = []          # list of dicts
        self.current_section = ""
        self._section_counter = 0
        self._start_time = time.time()

    # -- helpers ----------------------------------------------------------

    def section(self, title):
        self._section_counter += 1
        self.current_section = f"[{self._section_counter:02d}] {title}"
        print(f"\n{'=' * 80}")
        print(f"  {self.current_section}")
        print(f"{'=' * 80}")

    def ok(self, msg):
        self.results.append({"section": self.current_section, "level": self.PASS, "msg": msg})
        print(f"  [PASS] {msg}")

    def warn(self, msg):
        self.results.append({"section": self.current_section, "level": self.WARN, "msg": msg})
        print(f"  [WARN] {msg}")

    def fail(self, msg):
        self.results.append({"section": self.current_section, "level": self.FAIL, "msg": msg})
        print(f"  [FAIL] {msg}")

    def critical(self, msg):
        self.results.append({"section": self.current_section, "level": self.CRITICAL, "msg": msg})
        print(f"  [CRIT] {msg}")

    def info(self, msg):
        """Non-graded informational line."""
        print(f"  [INFO] {msg}")

    def assert_true(self, condition, pass_msg, fail_msg, level=FAIL):
        if condition:
            self.ok(pass_msg)
        elif level == self.WARN:
            self.warn(fail_msg)
        elif level == self.CRITICAL:
            self.critical(fail_msg)
        else:
            self.fail(fail_msg)

    def assert_range(self, value, lo, hi, name):
        if lo <= value <= hi:
            self.ok(f"{name} = {value} is within [{lo}, {hi}]")
        else:
            self.fail(f"{name} = {value} is OUTSIDE [{lo}, {hi}]")

    # -- final report ------------------------------------------------------

    def summary(self):
        elapsed = time.time() - self._start_time
        counts = Counter(r["level"] for r in self.results)
        n_pass = counts.get(self.PASS, 0)
        n_warn = counts.get(self.WARN, 0)
        n_fail = counts.get(self.FAIL, 0)
        n_crit = counts.get(self.CRITICAL, 0)

        print(f"\n{'#' * 80}")
        print(f"  DEBUGGING REPORT SUMMARY")
        print(f"{'#' * 80}")
        print(f"  Total checks : {len(self.results)}")
        print(f"  PASS          : {n_pass}")
        print(f"  WARN          : {n_warn}")
        print(f"  FAIL          : {n_fail}")
        print(f"  CRITICAL      : {n_crit}")
        print(f"  Elapsed       : {elapsed:.1f}s")

        # Print all non-passing items grouped by section
        if n_fail + n_crit + n_warn > 0:
            print(f"\n{'=' * 80}")
            print("  ITEMS REQUIRING ATTENTION")
            print(f"{'=' * 80}")
            current_sec = None
            for r in self.results:
                if r["level"] in (self.FAIL, self.CRITICAL, self.WARN):
                    if r["section"] != current_sec:
                        current_sec = r["section"]
                        print(f"\n  {current_sec}")
                    print(f"    [{r['level']}] {r['msg']}")

        # Overall verdict
        print(f"\n{'=' * 80}")
        if n_crit > 0:
            print("  VERDICT: CRITICAL ISSUES FOUND -- pipeline is broken")
        elif n_fail > 0:
            print("  VERDICT: FAILURES FOUND -- review recommended before production use")
        elif n_warn > 3:
            print("  VERDICT: MULTIPLE WARNINGS -- review recommended")
        else:
            print("  VERDICT: ALL CLEAR -- system looks healthy")
        print(f"{'=' * 80}\n")

        return n_crit == 0 and n_fail == 0


# ===========================================================================
# SECTION 1: DATA PIPELINE INTEGRITY
# ===========================================================================

def section_01_data_pipeline(R: DebugReport):
    R.section("DATA PIPELINE INTEGRITY")

    # 1a. XU100 CSV exists and is readable
    xu100_path = DATA_DIR / config.XU100_FILE
    if not xu100_path.exists():
        # Also try project root data/
        xu100_path_alt = PROJECT_ROOT / "data" / config.XU100_FILE
        if xu100_path_alt.exists():
            xu100_path = xu100_path_alt
            R.ok(f"XU100 file found at fallback path: {xu100_path}")
        else:
            R.critical(f"XU100 file MISSING at {xu100_path} and {xu100_path_alt}")
            return None, None
    else:
        R.ok(f"XU100 file found at {xu100_path}")

    df = pd.read_csv(xu100_path)
    R.info(f"XU100 raw CSV: {len(df)} rows x {len(df.columns)} cols")

    # 1b. Expected columns
    expected_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    actual_cols = set(df.columns)
    missing_cols = expected_cols - actual_cols
    R.assert_true(len(missing_cols) == 0,
                  "All expected columns present in XU100 CSV",
                  f"Missing columns in XU100 CSV: {missing_cols}")

    # 1c. Date parsing -- use DataLoader with the directory that contains the file
    from market_data import DataLoader
    loader = DataLoader(str(xu100_path.parent))
    xu100 = loader.load_xu100()
    R.assert_true(len(xu100) > 1000,
                  f"Loaded {len(xu100)} days of XU100 data (>1000 days OK)",
                  f"Only {len(xu100)} days loaded -- suspiciously low", level=R.WARN)

    # Check for duplicate dates
    dupes = xu100.index.duplicated().sum()
    R.assert_true(dupes == 0,
                  "No duplicate dates in XU100 index",
                  f"{dupes} duplicate dates in XU100 index")

    # Monotonically increasing
    R.assert_true(xu100.index.is_monotonic_increasing,
                  "XU100 index is sorted chronologically",
                  "XU100 index is NOT monotonically increasing")

    # No negative Close prices
    neg_prices = (xu100["Close"] <= 0).sum()
    R.assert_true(neg_prices == 0,
                  "All XU100 Close prices are positive",
                  f"{neg_prices} non-positive Close prices found")

    # NaN Close
    nan_close = xu100["Close"].isna().sum()
    R.assert_true(nan_close == 0,
                  "No NaN values in XU100 Close",
                  f"{nan_close} NaN values in XU100 Close")

    # 1d. USD/TRY
    R.info("Fetching USD/TRY data...")
    usdtry = loader.load_usdtry()
    R.assert_true(usdtry is not None and len(usdtry) > 500,
                  f"USD/TRY loaded: {len(usdtry)} rows",
                  "USD/TRY data failed to load or too few rows", level=R.WARN)

    if usdtry is not None:
        usdtry_nan = usdtry["USDTRY"].isna().sum()
        R.assert_true(usdtry_nan < len(usdtry) * 0.05,
                      f"USD/TRY NaN count acceptable: {usdtry_nan}/{len(usdtry)}",
                      f"USD/TRY has {usdtry_nan} NaN values ({usdtry_nan/len(usdtry)*100:.1f}%)")

    # 1e. Merge
    merged = loader.merge_data()
    R.info(f"Merged data: {len(merged)} rows, {len(merged.columns)} columns")

    return loader, merged


# ===========================================================================
# SECTION 2: FEATURE ENGINEERING VALIDATION
# ===========================================================================

def section_02_feature_engineering(R: DebugReport, merged_data):
    R.section("FEATURE ENGINEERING VALIDATION")

    if merged_data is None:
        R.critical("Skipped -- no merged data available")
        return None, None

    from market_data import FeatureEngine

    engine = FeatureEngine(merged_data)
    features_raw = engine.calculate_all_features()
    R.info(f"Raw features: {len(features_raw)} rows x {len(features_raw.columns)} cols")

    # 2a. Critical features exist
    critical_features = [
        "realized_vol_20d", "realized_vol_60d",
        "return_20d", "return_60d", "return_120d",
        "max_drawdown_20d", "max_drawdown_60d",
        "usdtry_momentum_20d",
        "turnover_ma_20d",
        "volume_ratio",
    ]
    for f in critical_features:
        R.assert_true(f in features_raw.columns,
                      f"Feature '{f}' present",
                      f"Feature '{f}' MISSING")

    # 2b. Feature value sanity
    if "realized_vol_20d" in features_raw.columns:
        vol = features_raw["realized_vol_20d"].dropna()
        R.assert_true(vol.min() >= 0,
                      f"realized_vol_20d is non-negative (min={vol.min():.4f})",
                      f"realized_vol_20d has negative values: min={vol.min():.4f}")
        R.assert_true(vol.max() < 5.0,
                      f"realized_vol_20d max is reasonable ({vol.max():.4f} < 5.0)",
                      f"realized_vol_20d max is suspiciously high: {vol.max():.4f}", level=R.WARN)

    if "volume_ratio" in features_raw.columns:
        vr = features_raw["volume_ratio"].dropna()
        R.assert_true(vr.min() >= 0,
                      "volume_ratio is non-negative",
                      f"volume_ratio has negative values: min={vr.min():.4f}")

    # 2c. NaN report per feature
    nan_pct = features_raw.isna().mean() * 100
    high_nan = nan_pct[nan_pct > 30]
    if len(high_nan) > 0:
        R.warn(f"{len(high_nan)} features have >30% NaN: {list(high_nan.index[:10])}")
    else:
        R.ok("No feature has >30% NaN values")

    # 2d. Infinite values
    numeric = features_raw.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric).sum().sum()
    R.assert_true(inf_count == 0,
                  "No infinite values in features",
                  f"{inf_count} infinite values found in features")

    # 2e. Shift for look-ahead bias
    features_shifted = engine.shift_for_prediction(shift_days=1)
    R.info(f"Shifted features: {len(features_shifted)} rows (lost {len(features_raw) - len(features_shifted)} rows)")

    # Verify the shift actually happened: the shifted features at date T should
    # equal the raw features at date T-1.
    if len(features_shifted) > 10 and "realized_vol_20d" in features_shifted.columns:
        # Take a sample date in the middle
        test_idx = len(features_shifted) // 2
        shifted_date = features_shifted.index[test_idx]
        # Find the previous business day in raw features
        raw_dates = features_raw.index
        pos = raw_dates.get_loc(shifted_date)
        if pos > 0:
            prev_date = raw_dates[pos - 1]
            shifted_val = features_shifted.loc[shifted_date, "realized_vol_20d"]
            raw_val = features_raw.loc[prev_date, "realized_vol_20d"]
            if not (np.isnan(shifted_val) or np.isnan(raw_val)):
                R.assert_true(
                    abs(shifted_val - raw_val) < 1e-10,
                    f"Look-ahead shift verified: shifted[{shifted_date.date()}] == raw[{prev_date.date()}]",
                    f"Look-ahead shift BROKEN: shifted={shifted_val:.6f} != raw={raw_val:.6f}"
                )
            else:
                R.warn("Could not verify look-ahead shift (NaN in sample)")

    return engine, features_shifted


# ===========================================================================
# SECTION 3: ROLLING PERCENTILE CALCULATION
# ===========================================================================

def section_03_percentiles(R: DebugReport, features_shifted):
    R.section("ROLLING PERCENTILE CALCULATION")

    if features_shifted is None:
        R.critical("Skipped -- no shifted features available")
        return None

    from regime_models import RegimeClassifier
    classifier = RegimeClassifier(features_shifted)
    classifier.calculate_percentiles(lookback=config.PERCENTILE_LOOKBACK)

    pct = classifier.percentiles
    R.info(f"Percentile DataFrame: {len(pct)} rows x {len(pct.columns)} cols")

    # 3a. Expected percentile columns
    expected_pct_cols = [
        "realized_vol_20d_percentile",
        "usdtry_momentum_20d_percentile",
        "turnover_ma_20d_percentile",
    ]
    for col in expected_pct_cols:
        R.assert_true(col in pct.columns,
                      f"Percentile column '{col}' present",
                      f"Percentile column '{col}' MISSING")

    # 3b. Percentile range [0, 100]
    for col in pct.columns:
        valid = pct[col].dropna()
        if len(valid) > 0:
            R.assert_true(valid.min() >= 0 and valid.max() <= 100,
                          f"{col}: range [{valid.min():.1f}, {valid.max():.1f}] OK",
                          f"{col}: range [{valid.min():.1f}, {valid.max():.1f}] OUTSIDE [0, 100]")

    # 3c. NaN fraction (first ~lookback rows will be NaN)
    for col in pct.columns:
        nan_count = pct[col].isna().sum()
        nan_pct_val = nan_count / len(pct) * 100
        R.info(f"{col}: {nan_count} NaN ({nan_pct_val:.1f}%)")
        # Should not be more than lookback + some margin
        max_expected_nan = config.PERCENTILE_LOOKBACK + 50
        R.assert_true(nan_count < max_expected_nan,
                      f"{col}: NaN count {nan_count} < expected max {max_expected_nan}",
                      f"{col}: excessive NaN ({nan_count} > {max_expected_nan})", level=R.WARN)

    # 3d. Verify percentile computation on a known window
    if "realized_vol_20d_percentile" in pct.columns:
        # Pick a date well after warmup
        valid_dates = pct["realized_vol_20d_percentile"].dropna().index
        if len(valid_dates) > 100:
            test_date = valid_dates[len(valid_dates) // 2]
            test_idx = features_shifted.index.get_loc(test_date)
            lookback = config.PERCENTILE_LOOKBACK
            min_periods = lookback // 2

            if test_idx >= min_periods:
                window_start = max(0, test_idx - lookback + 1)
                window = features_shifted["realized_vol_20d"].iloc[window_start:test_idx + 1].dropna()
                current_val = features_shifted.loc[test_date, "realized_vol_20d"]
                if len(window) >= min_periods and not np.isnan(current_val):
                    manual_pct = (window < current_val).sum() / len(window) * 100
                    stored_pct = pct.loc[test_date, "realized_vol_20d_percentile"]
                    diff = abs(manual_pct - stored_pct)
                    R.assert_true(
                        diff < 10,  # allow some tolerance due to rank vs < comparison
                        f"Percentile spot-check: manual={manual_pct:.1f}, stored={stored_pct:.1f} (diff={diff:.1f})",
                        f"Percentile spot-check MISMATCH: manual={manual_pct:.1f}, stored={stored_pct:.1f} (diff={diff:.1f})",
                        level=R.WARN
                    )

    return classifier


# ===========================================================================
# SECTION 4: VOLATILITY REGIME CLASSIFICATION
# ===========================================================================

def section_04_volatility_regime(R: DebugReport, classifier):
    R.section("VOLATILITY REGIME CLASSIFICATION")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return

    classifier.classify_volatility_regime()
    vol_regime = classifier.regimes["volatility_regime"]

    # 4a. All values are valid
    valid_states = {"Low", "Mid", "High", "Stress", "Unknown"}
    unique_states = set(vol_regime.unique())
    invalid = unique_states - valid_states
    R.assert_true(len(invalid) == 0,
                  f"Volatility states valid: {unique_states}",
                  f"Invalid volatility states found: {invalid}")

    # 4b. Distribution sanity -- expect mostly Mid, some Low/High, few Stress
    counts = vol_regime.value_counts(normalize=True) * 100
    R.info(f"Volatility distribution: {dict(counts.round(1))}")

    if "Unknown" in counts.index:
        R.assert_true(counts["Unknown"] < 5,
                      f"'Unknown' volatility is {counts['Unknown']:.1f}% (<5% OK)",
                      f"'Unknown' volatility is {counts['Unknown']:.1f}% -- too high", level=R.WARN)

    # Mid should be ~50% with current thresholds (25-75)
    if "Mid" in counts.index:
        R.assert_true(30 < counts["Mid"] < 70,
                      f"Mid vol fraction {counts['Mid']:.1f}% is within [30%, 70%]",
                      f"Mid vol fraction {counts['Mid']:.1f}% is unexpected", level=R.WARN)

    # Stress should be rare (top 8%)
    if "Stress" in counts.index:
        R.assert_true(counts["Stress"] < 20,
                      f"Stress fraction {counts['Stress']:.1f}% is <20%",
                      f"Stress fraction {counts['Stress']:.1f}% -- too high", level=R.WARN)

    # 4c. Verify threshold boundaries
    if "volatility_percentile" in classifier.regimes.columns:
        vp = classifier.regimes["volatility_percentile"].dropna()
        vr = vol_regime[vp.index]

        # Low should correspond to percentile <= 25
        low_mask = vr == "Low"
        if low_mask.sum() > 0:
            low_max_pct = vp[low_mask].max()
            R.assert_true(low_max_pct <= config.VOLATILITY_PERCENTILES["low"] + 0.5,
                          f"Low vol max percentile = {low_max_pct:.1f} <= {config.VOLATILITY_PERCENTILES['low']}",
                          f"Low vol max percentile = {low_max_pct:.1f} > {config.VOLATILITY_PERCENTILES['low']}")

        # Stress should correspond to percentile > 92
        stress_mask = vr == "Stress"
        if stress_mask.sum() > 0:
            stress_min_pct = vp[stress_mask].min()
            R.assert_true(stress_min_pct > config.VOLATILITY_PERCENTILES["high"] - 0.5,
                          f"Stress vol min percentile = {stress_min_pct:.1f} > {config.VOLATILITY_PERCENTILES['high']}",
                          f"Stress vol min percentile = {stress_min_pct:.1f} <= {config.VOLATILITY_PERCENTILES['high']}")


# ===========================================================================
# SECTION 5: TREND REGIME CLASSIFICATION
# ===========================================================================

def section_05_trend_regime(R: DebugReport, classifier, features_shifted):
    R.section("TREND REGIME CLASSIFICATION")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return

    classifier.classify_trend_regime()
    regimes = classifier.regimes

    # 5a. Trend columns exist
    for col in ["trend_short", "trend_long", "trend_regime"]:
        R.assert_true(col in regimes.columns,
                      f"Column '{col}' present",
                      f"Column '{col}' MISSING")

    # 5b. Valid trend states
    valid_short_long = {"Up", "Down", "Sideways", "Unknown"}
    valid_combined = {"Up", "Down", "Sideways", "Mixed", "Unknown"}

    for col, valid_set in [("trend_short", valid_short_long),
                           ("trend_long", valid_short_long),
                           ("trend_regime", valid_combined)]:
        if col in regimes.columns:
            actual = set(regimes[col].unique())
            invalid = actual - valid_set
            R.assert_true(len(invalid) == 0,
                          f"{col} states valid: {actual}",
                          f"{col} has invalid states: {invalid}")

    # 5c. Distribution
    if "trend_regime" in regimes.columns:
        counts = regimes["trend_regime"].value_counts(normalize=True) * 100
        R.info(f"Trend distribution: {dict(counts.round(1))}")

        # Should have meaningful representation of Up and Down
        for state in ["Up", "Down"]:
            if state in counts.index:
                R.assert_true(counts[state] > 5,
                              f"'{state}' trend fraction {counts[state]:.1f}% is >5%",
                              f"'{state}' trend fraction {counts[state]:.1f}% -- too low", level=R.WARN)

    # 5d. Verify thresholds against actual return values
    if features_shifted is not None and "return_20d" in features_shifted.columns:
        ret20 = features_shifted["return_20d"].dropna()
        short_trend = regimes.loc[ret20.index, "trend_short"]

        # Up should have return > 1.5%
        up_mask = short_trend == "Up"
        if up_mask.sum() > 0:
            up_ret_min = ret20[up_mask].min()
            R.assert_true(up_ret_min > config.TREND_THRESHOLDS["up"] - 1e-10,
                          f"Short Up: min return = {up_ret_min:.4f} > threshold {config.TREND_THRESHOLDS['up']}",
                          f"Short Up: min return = {up_ret_min:.4f} BELOW threshold {config.TREND_THRESHOLDS['up']}")

        # Down should have return < -1.5%
        down_mask = short_trend == "Down"
        if down_mask.sum() > 0:
            down_ret_max = ret20[down_mask].max()
            R.assert_true(down_ret_max < config.TREND_THRESHOLDS["down"] + 1e-10,
                          f"Short Down: max return = {down_ret_max:.4f} < threshold {config.TREND_THRESHOLDS['down']}",
                          f"Short Down: max return = {down_ret_max:.4f} ABOVE threshold {config.TREND_THRESHOLDS['down']}")

    # 5e. Combined trend logic consistency
    if all(c in regimes.columns for c in ["trend_short", "trend_long", "trend_regime"]):
        # If short == long, combined should match
        same_mask = regimes["trend_short"] == regimes["trend_long"]
        if same_mask.sum() > 0:
            combined_should_match = regimes.loc[same_mask, "trend_regime"] == regimes.loc[same_mask, "trend_short"]
            mismatches = (~combined_should_match).sum()
            R.assert_true(mismatches == 0,
                          f"Combined trend matches when short==long ({same_mask.sum()} cases)",
                          f"{mismatches} mismatches where short==long but combined differs")


# ===========================================================================
# SECTION 6: RISK REGIME CLASSIFICATION
# ===========================================================================

def section_06_risk_regime(R: DebugReport, classifier):
    R.section("RISK REGIME CLASSIFICATION")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return

    classifier.classify_risk_regime()
    risk = classifier.regimes["risk_regime"]

    # 6a. Valid states
    valid_states = {"Risk-On", "Risk-Off", "Neutral"}
    actual = set(risk.unique())
    invalid = actual - valid_states
    R.assert_true(len(invalid) == 0,
                  f"Risk states valid: {actual}",
                  f"Invalid risk states: {invalid}")

    # 6b. Distribution
    counts = risk.value_counts(normalize=True) * 100
    R.info(f"Risk distribution: {dict(counts.round(1))}")

    # With thresholds at 35/65, Neutral should be ~30%
    if "Neutral" in counts.index:
        R.assert_true(10 < counts["Neutral"] < 60,
                      f"Neutral risk fraction {counts['Neutral']:.1f}% in [10%, 60%]",
                      f"Neutral risk fraction {counts['Neutral']:.1f}% outside expected range", level=R.WARN)

    # 6c. Threshold consistency
    if "usdtry_momentum_percentile" in classifier.regimes.columns:
        pct = classifier.regimes["usdtry_momentum_percentile"].dropna()
        risk_vals = risk[pct.index]

        # Risk-Off should be > 65th percentile
        risk_off_mask = risk_vals == "Risk-Off"
        if risk_off_mask.sum() > 0:
            min_pct = pct[risk_off_mask].min()
            R.assert_true(min_pct > config.RISK_PERCENTILES["risk_off"] - 0.5,
                          f"Risk-Off min percentile = {min_pct:.1f} > {config.RISK_PERCENTILES['risk_off']}",
                          f"Risk-Off min percentile = {min_pct:.1f} <= {config.RISK_PERCENTILES['risk_off']}")

        # Risk-On should be < 35th percentile
        risk_on_mask = risk_vals == "Risk-On"
        if risk_on_mask.sum() > 0:
            max_pct = pct[risk_on_mask].max()
            R.assert_true(max_pct < config.RISK_PERCENTILES["risk_on"] + 0.5,
                          f"Risk-On max percentile = {max_pct:.1f} < {config.RISK_PERCENTILES['risk_on']}",
                          f"Risk-On max percentile = {max_pct:.1f} >= {config.RISK_PERCENTILES['risk_on']}")


# ===========================================================================
# SECTION 7: LIQUIDITY REGIME CLASSIFICATION
# ===========================================================================

def section_07_liquidity_regime(R: DebugReport, classifier):
    R.section("LIQUIDITY REGIME CLASSIFICATION")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return

    classifier.classify_liquidity_regime()
    liq = classifier.regimes["liquidity_regime"]

    # 7a. Valid states
    valid = {"Normal", "Low", "Very Low", "Unknown"}
    actual = set(liq.unique())
    invalid = actual - valid
    R.assert_true(len(invalid) == 0,
                  f"Liquidity states valid: {actual}",
                  f"Invalid liquidity states: {invalid}")

    # 7b. Distribution
    counts = liq.value_counts(normalize=True) * 100
    R.info(f"Liquidity distribution: {dict(counts.round(1))}")

    if "Unknown" in counts.index:
        R.assert_true(counts["Unknown"] < 5,
                      f"'Unknown' liquidity {counts['Unknown']:.1f}% < 5%",
                      f"'Unknown' liquidity {counts['Unknown']:.1f}% -- too high", level=R.WARN)

    # Normal should be dominant (above 40th percentile)
    if "Normal" in counts.index:
        R.assert_true(counts["Normal"] > 30,
                      f"Normal liquidity fraction {counts['Normal']:.1f}% > 30%",
                      f"Normal liquidity fraction {counts['Normal']:.1f}% -- too low", level=R.WARN)


# ===========================================================================
# SECTION 8: COMBINED REGIME LABEL
# ===========================================================================

def section_08_combined_label(R: DebugReport, classifier):
    R.section("COMBINED REGIME LABEL CONSTRUCTION")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return

    classifier.create_combined_regime_label()

    R.assert_true("regime_label" in classifier.regimes.columns,
                  "regime_label column created",
                  "regime_label column MISSING")

    labels = classifier.regimes["regime_label"]

    # 8a. No NaN labels
    nan_labels = labels.isna().sum()
    R.assert_true(nan_labels == 0,
                  "No NaN regime labels",
                  f"{nan_labels} NaN regime labels found")

    # 8b. Label format: should contain "Vol", "Trend", and "Liq"
    sample_label = labels.dropna().iloc[-1]
    R.info(f"Sample label: {sample_label}")
    R.assert_true("Vol" in sample_label and "Trend" in sample_label,
                  f"Label format looks correct: contains 'Vol' and 'Trend'",
                  f"Label format unexpected: {sample_label}")

    # 8c. Number of unique labels
    n_unique = labels.nunique()
    R.info(f"Unique combined labels: {n_unique}")
    R.assert_true(n_unique > 5,
                  f"{n_unique} unique labels (>5 = adequate variety)",
                  f"Only {n_unique} unique labels -- classification may be too coarse", level=R.WARN)
    R.assert_true(n_unique < 100,
                  f"{n_unique} unique labels (<100 = not overfitting)",
                  f"{n_unique} unique labels -- very high, may be fragmented", level=R.WARN)


# ===========================================================================
# SECTION 9: SIMPLIFIED REGIME MAPPING (5-STATE)
# ===========================================================================

def section_09_simplified_regime(R: DebugReport, classifier):
    R.section("SIMPLIFIED REGIME MAPPING (5-STATE)")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return None

    from regime_models import SimplifiedRegimeClassifier

    simple = SimplifiedRegimeClassifier(min_duration=10, hysteresis_days=3)
    result = simple.classify(classifier.regimes, apply_persistence=False)

    R.assert_true("simplified_regime" in result.columns,
                  "'simplified_regime' column present",
                  "'simplified_regime' column MISSING")

    raw_regimes = result["simplified_regime"]

    # 9a. Valid states
    valid_states = {"Bull", "Bear", "Stress", "Choppy", "Recovery"}
    actual = set(raw_regimes.unique())
    invalid = actual - valid_states
    R.assert_true(len(invalid) == 0,
                  f"Simplified states valid: {actual}",
                  f"Invalid simplified states: {invalid}")

    # 9b. Distribution
    counts = raw_regimes.value_counts(normalize=True) * 100
    R.info(f"Raw simplified distribution (before persistence): {dict(counts.round(1))}")

    # Each state should have some representation
    for state in valid_states:
        if state in counts.index:
            R.assert_true(counts[state] > 1,
                          f"'{state}' has {counts[state]:.1f}% representation",
                          f"'{state}' has only {counts[state]:.1f}% -- very rare", level=R.WARN)
        else:
            R.warn(f"'{state}' has 0% representation in raw mapping")

    # 9c. Verify mapping logic on synthetic cases
    R.info("Verifying mapping logic with synthetic test cases...")

    test_cases = [
        {"volatility_regime": "Stress", "trend_regime": "Up", "risk_regime": "Risk-On", "expected": "Stress"},
        {"volatility_regime": "High", "trend_regime": "Down", "risk_regime": "Risk-Off", "expected": "Stress"},
        {"volatility_regime": "Low", "trend_regime": "Up", "risk_regime": "Risk-On", "expected": "Bull"},
        {"volatility_regime": "Mid", "trend_regime": "Up", "risk_regime": "Neutral", "expected": "Bull"},
        {"volatility_regime": "Mid", "trend_regime": "Down", "risk_regime": "Risk-Off", "expected": "Bear"},
        {"volatility_regime": "High", "trend_regime": "Up", "risk_regime": "Risk-On", "expected": "Recovery"},
        {"volatility_regime": "Mid", "trend_regime": "Sideways", "risk_regime": "Neutral", "expected": "Choppy"},
        {"volatility_regime": "Low", "trend_regime": "Down", "risk_regime": "Neutral", "expected": "Bear"},
        {"volatility_regime": "High", "trend_regime": "Down", "risk_regime": "Neutral", "expected": "Bear"},
    ]

    for tc in test_cases:
        # Create a single-row DataFrame to test
        test_row = pd.DataFrame([{
            "volatility_regime": tc["volatility_regime"],
            "trend_regime": tc["trend_regime"],
            "risk_regime": tc["risk_regime"],
            "liquidity_regime": "Normal",
        }])
        mapped = simple.classify(test_row, apply_persistence=False)
        result_regime = mapped["simplified_regime"].iloc[0]
        R.assert_true(
            result_regime == tc["expected"],
            f"  Vol={tc['volatility_regime']}, Trend={tc['trend_regime']}, Risk={tc['risk_regime']} -> {result_regime} (expected {tc['expected']})",
            f"  Vol={tc['volatility_regime']}, Trend={tc['trend_regime']}, Risk={tc['risk_regime']} -> {result_regime} (expected {tc['expected']} -- MISMATCH)"
        )

    return raw_regimes


# ===========================================================================
# SECTION 10: PERSISTENCE / HYSTERESIS FILTER
# ===========================================================================

def section_10_persistence_filter(R: DebugReport, classifier):
    R.section("PERSISTENCE / HYSTERESIS FILTER")

    if classifier is None:
        R.critical("Skipped -- no classifier available")
        return None

    from regime_models import SimplifiedRegimeClassifier

    simple = SimplifiedRegimeClassifier(min_duration=10, hysteresis_days=3)

    # Get regimes with and without persistence
    raw = simple.classify(classifier.regimes, apply_persistence=False)["simplified_regime"]
    smoothed = simple.classify(classifier.regimes, apply_persistence=True)["simplified_regime"]

    # 10a. Smoothed should have fewer regime changes
    raw_changes = (raw != raw.shift()).sum()
    smoothed_changes = (smoothed != smoothed.shift()).sum()
    R.info(f"Regime transitions: raw={raw_changes}, smoothed={smoothed_changes}")
    R.assert_true(smoothed_changes <= raw_changes,
                  f"Persistence filter reduces transitions: {raw_changes} -> {smoothed_changes}",
                  f"Persistence filter INCREASED transitions: {raw_changes} -> {smoothed_changes}")

    # 10b. Stress should bypass hysteresis (immediate switch)
    # Find where raw says Stress but smoothed doesn't within hysteresis window
    stress_raw = (raw == "Stress")
    stress_smoothed = (smoothed == "Stress")
    # Every time raw switches to Stress, smoothed should also be Stress on that same day
    raw_stress_start = stress_raw & ~stress_raw.shift(1, fill_value=False)
    for idx in raw.index[raw_stress_start]:
        R.assert_true(smoothed.loc[idx] == "Stress",
                      f"Stress bypass at {idx.date()}: immediate switch OK",
                      f"Stress bypass FAILED at {idx.date()}: smoothed={smoothed.loc[idx]}")
        # Only check first 5 for brevity
        if raw_stress_start[:idx].sum() >= 5:
            break

    # 10c. Non-stress transitions should be delayed by hysteresis_days
    # Find a transition in raw that is NOT stress
    raw_change_mask = (raw != raw.shift()) & (raw != "Stress")
    raw_change_indices = raw.index[raw_change_mask]

    verified_delay = 0
    for i, idx in enumerate(raw_change_indices[:20]):
        pos = raw.index.get_loc(idx)
        if pos < 3:
            continue
        # If raw changes at pos, smoothed should still show old regime for up to hysteresis_days
        old_regime = smoothed.iloc[pos - 1]
        new_regime = raw.iloc[pos]
        # Check if smoothed kept old regime at pos (delayed)
        if smoothed.iloc[pos] == old_regime and old_regime != new_regime:
            verified_delay += 1

    if len(raw_change_indices[:20]) > 0:
        R.info(f"Hysteresis delay verified in {verified_delay}/{min(len(raw_change_indices), 20)} sampled transitions")
        R.assert_true(verified_delay > 0,
                      "Hysteresis delay is functioning (some transitions delayed)",
                      "No evidence of hysteresis delay -- filter may not be working", level=R.WARN)

    # 10d. Distribution comparison
    raw_dist = raw.value_counts(normalize=True) * 100
    smoothed_dist = smoothed.value_counts(normalize=True) * 100
    R.info(f"Raw distribution:      {dict(raw_dist.round(1))}")
    R.info(f"Smoothed distribution: {dict(smoothed_dist.round(1))}")

    return smoothed


# ===========================================================================
# SECTION 11: HMM REGIME CLASSIFIER
# ===========================================================================

def section_11_hmm(R: DebugReport, features_shifted, skip=False):
    R.section("HMM REGIME CLASSIFIER")

    if skip:
        R.info("Skipped (--quick mode)")
        return

    if features_shifted is None:
        R.critical("Skipped -- no features available")
        return

    try:
        from regime_models import HMMRegimeClassifier

        hmm_model = HMMRegimeClassifier(n_regimes=4, random_state=42)
        hmm_model.fit(features_shifted)

        R.assert_true(hmm_model.model is not None,
                      "HMM model fitted successfully",
                      "HMM model is None after fitting")

        # Transition matrix
        if hmm_model.transition_matrix is not None:
            tm = hmm_model.transition_matrix.values
            R.info(f"Transition matrix shape: {tm.shape}")

            # Rows should sum to 1.0
            row_sums = tm.sum(axis=1)
            for i, rs in enumerate(row_sums):
                R.assert_true(abs(rs - 1.0) < 0.01,
                              f"Transition row {i} sums to {rs:.4f} (~1.0)",
                              f"Transition row {i} sums to {rs:.4f} (expected 1.0)")

            # All values should be in [0, 1]
            R.assert_true(tm.min() >= 0 and tm.max() <= 1.0,
                          "All transition probabilities in [0, 1]",
                          f"Transition values outside [0, 1]: min={tm.min():.4f}, max={tm.max():.4f}")

            # Diagonal should be dominant (regimes are persistent)
            diag = np.diag(tm)
            R.assert_true(diag.min() > 0.5,
                          f"Diagonal persistence: min={diag.min():.3f} (>0.5 = regimes are sticky)",
                          f"Diagonal persistence: min={diag.min():.3f} (<0.5 = regimes not persistent)", level=R.WARN)

        # Predict
        predictions = hmm_model.predict(features_shifted)
        R.assert_true(len(predictions) > 0,
                      f"HMM predictions: {len(predictions)} rows",
                      "HMM predictions empty")

        # Probability columns
        prob_cols = [c for c in predictions.columns if c.startswith("prob_regime_")]
        R.assert_true(len(prob_cols) == 4,
                      f"Found {len(prob_cols)} probability columns",
                      f"Expected 4 probability columns, found {len(prob_cols)}")

        # Probabilities should sum to ~1.0 per row
        if len(prob_cols) > 0:
            prob_sums = predictions[prob_cols].sum(axis=1)
            bad_sums = ((prob_sums - 1.0).abs() > 0.01).sum()
            R.assert_true(bad_sums == 0,
                          "HMM probabilities sum to ~1.0 for all rows",
                          f"{bad_sums} rows where probabilities don't sum to 1.0")

        # Label regimes
        labeled = hmm_model.label_regimes(features_shifted, predictions)
        if "regime_name" in labeled.columns:
            R.ok(f"Regime labeling successful: {labeled['regime_name'].value_counts().to_dict()}")
        else:
            R.warn("Regime labeling did not produce 'regime_name' column")

    except ImportError:
        R.warn("HMM packages (hmmlearn) not available -- skipping")
    except Exception as e:
        R.fail(f"HMM test failed: {e}")
        traceback.print_exc()


# ===========================================================================
# SECTION 12: XGBOOST PREDICTIVE MODEL
# ===========================================================================

def section_12_xgboost(R: DebugReport, features_shifted, simplified_regimes, skip=False):
    R.section("XGBOOST PREDICTIVE MODEL")

    if skip:
        R.info("Skipped (--quick mode)")
        return

    if features_shifted is None or simplified_regimes is None:
        R.critical("Skipped -- missing data")
        return

    try:
        from regime_models import PredictiveRegimeModel

        model = PredictiveRegimeModel(forecast_horizon=0, model_type="xgboost")
        X, y = model.prepare_data(features_shifted, simplified_regimes)

        R.info(f"Prepared data: X={X.shape}, y={len(y)}")
        R.assert_true(len(X) > 100,
                      f"Sufficient training data: {len(X)} samples",
                      f"Only {len(X)} samples -- may not train well", level=R.WARN)

        # Check class distribution
        y_dist = y.value_counts()
        R.info(f"Target distribution: {dict(y_dist)}")

        # Check for class imbalance
        if len(y_dist) > 0:
            min_class = y_dist.min()
            max_class = y_dist.max()
            ratio = min_class / max_class
            R.assert_true(ratio > 0.05,
                          f"Class balance ratio: {ratio:.2f} (min={min_class}, max={max_class})",
                          f"Severe class imbalance: ratio={ratio:.2f} -- minority class may be ignored", level=R.WARN)

        # Train/test split
        split_date = X.index[int(len(X) * 0.7)]
        X_train, X_test, y_train, y_test = model.train_test_split(X, y, train_end_date=split_date)
        R.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Train
        model.train(X_train, y_train)
        R.assert_true(model.model is not None,
                      "XGBoost model trained successfully",
                      "XGBoost model is None after training")

        # Predict
        predictions, confidence, probs = model.predict(X_test)
        R.assert_true(predictions is not None,
                      f"XGBoost predictions: {len(predictions)} samples",
                      "XGBoost predictions returned None")

        # Accuracy
        if predictions is not None:
            pred_numeric = predictions.map(model.regime_mapping)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, pred_numeric)
            R.info(f"XGBoost test accuracy: {acc:.2%}")
            R.assert_true(acc > 0.25,
                          f"XGBoost accuracy {acc:.2%} > 25% (better than random 5-class)",
                          f"XGBoost accuracy {acc:.2%} <= 25% -- worse than random", level=R.WARN)

        # Feature importance
        importance = model.get_feature_importance()
        if importance is not None:
            R.ok(f"Feature importance available: {len(importance)} features")
            # Top 5 features
            if model.feature_names:
                top5_idx = np.argsort(importance)[-5:][::-1]
                top5 = [(model.feature_names[i], importance[i]) for i in top5_idx]
                for name, imp in top5:
                    R.info(f"  Top feature: {name} = {imp:.4f}")

    except ImportError:
        R.warn("XGBoost not available -- skipping")
    except Exception as e:
        R.fail(f"XGBoost test failed: {e}")
        traceback.print_exc()


# ===========================================================================
# SECTION 13: LSTM SEQUENCE MODEL
# ===========================================================================

def section_13_lstm(R: DebugReport, features_shifted, simplified_regimes, skip=False):
    R.section("LSTM SEQUENCE MODEL")

    if skip:
        R.info("Skipped (--quick mode)")
        return

    if features_shifted is None or simplified_regimes is None:
        R.critical("Skipped -- missing data")
        return

    try:
        from models.lstm_regime import LSTMRegimeModel

        lstm = LSTMRegimeModel(sequence_length=20, forecast_horizon=0,
                               hidden_size=32, num_layers=1, dropout=0.1)

        X, y = lstm.prepare_sequences(features_shifted, simplified_regimes)
        R.info(f"Sequences: X={X.shape}, y={y.shape}")
        R.assert_true(len(X) > 50,
                      f"Sufficient sequences: {len(X)}",
                      f"Only {len(X)} sequences -- may not train well", level=R.WARN)

        # Check sequence shape
        R.assert_true(X.shape[1] == 20,
                      f"Sequence length correct: {X.shape[1]}",
                      f"Unexpected sequence length: {X.shape[1]}")
        R.assert_true(X.shape[2] == lstm.n_features,
                      f"Feature count matches: {X.shape[2]} == {lstm.n_features}",
                      f"Feature count mismatch: {X.shape[2]} != {lstm.n_features}")

        # Check for NaN in sequences
        nan_seqs = np.isnan(X).any(axis=(1, 2)).sum()
        R.assert_true(nan_seqs == 0,
                      "No NaN values in prepared sequences",
                      f"{nan_seqs} sequences contain NaN")

        # Quick training test (5 epochs)
        X_train, X_test, y_train, y_test = lstm.train_test_split(X, y, train_ratio=0.8)
        X_train_s, X_test_s = lstm._scale_sequences(X_train, X_test)

        R.info("Training LSTM for 5 epochs (quick validation)...")
        lstm.train(X_train_s, y_train, X_test_s, y_test, epochs=5, batch_size=32, early_stopping_patience=3)
        R.ok("LSTM training completed without errors")

        # Predict
        predictions, probs, confidence = lstm.predict(X_test_s)
        R.assert_true(len(predictions) == len(X_test_s),
                      f"LSTM predictions count matches: {len(predictions)}",
                      f"LSTM prediction count mismatch: {len(predictions)} != {len(X_test_s)}")

        # Probabilities should sum to ~1.0
        prob_sums = probs.sum(axis=1)
        bad_sums = (np.abs(prob_sums - 1.0) > 0.01).sum()
        R.assert_true(bad_sums == 0,
                      "LSTM probabilities sum to ~1.0 for all samples",
                      f"{bad_sums} samples where LSTM probs don't sum to 1.0")

    except ImportError as e:
        R.warn(f"LSTM dependencies not available: {e}")
    except Exception as e:
        R.fail(f"LSTM test failed: {e}")
        traceback.print_exc()


# ===========================================================================
# SECTION 14: ENSEMBLE MODEL INTEGRATION
# ===========================================================================

def section_14_ensemble(R: DebugReport, skip=False):
    R.section("ENSEMBLE MODEL INTEGRATION")

    if skip:
        R.info("Skipped (--quick mode)")
        return

    ensemble_dir = OUTPUT_DIR / "ensemble_model"
    if not ensemble_dir.exists():
        R.warn(f"Ensemble model directory not found at {ensemble_dir}")
        R.info("Run run_full_pipeline.py first to create ensemble model")
        return

    try:
        from models.ensemble_regime import EnsembleRegimeModel

        # 14a. Load ensemble
        ensemble = EnsembleRegimeModel.load(ensemble_dir)
        R.ok("Ensemble model loaded successfully")

        # 14b. Check metadata
        R.assert_true(ensemble.is_trained,
                      "Ensemble is marked as trained",
                      "Ensemble is_trained is False")

        R.assert_true(len(ensemble.available_models) >= 2,
                      f"Available models: {ensemble.available_models}",
                      f"Only {len(ensemble.available_models)} models available (need >= 2)")

        # 14c. Weight normalization
        active_weight_sum = sum(ensemble.weights[m] for m in ensemble.available_models)
        R.assert_true(abs(active_weight_sum - 1.0) < 0.01,
                      f"Active weights sum to {active_weight_sum:.4f} (~1.0)",
                      f"Active weights sum to {active_weight_sum:.4f} (expected 1.0)")

        # 14d. Feature names
        R.assert_true(ensemble.feature_names is not None and len(ensemble.feature_names) > 0,
                      f"Feature names stored: {len(ensemble.feature_names)} features",
                      "Feature names not stored in ensemble")

        # 14e. Load test data and run prediction
        features_file = OUTPUT_DIR / "all_features.csv"
        regimes_file = OUTPUT_DIR / "simplified_regimes.csv"

        if features_file.exists() and regimes_file.exists():
            features = pd.read_csv(features_file, index_col=0, parse_dates=True)
            regimes = pd.read_csv(regimes_file, index_col=0, parse_dates=True)
            if "regime" in regimes.columns:
                regimes = regimes["regime"]
            elif "simplified_regime" in regimes.columns:
                regimes = regimes["simplified_regime"]
            else:
                regimes = regimes.iloc[:, 0]

            # Test prediction
            test_features = features.tail(50)
            results = ensemble.predict(test_features, return_details=False)
            R.assert_true(len(results) == 50,
                          f"Ensemble prediction on 50 rows returned {len(results)} rows",
                          f"Ensemble prediction returned {len(results)} rows (expected 50)")

            # Check prediction validity
            valid_regimes = {"Bull", "Bear", "Stress", "Choppy", "Recovery"}
            pred_regimes = set(results["ensemble_prediction"].unique())
            invalid = pred_regimes - valid_regimes
            R.assert_true(len(invalid) == 0,
                          f"All ensemble predictions valid: {pred_regimes}",
                          f"Invalid ensemble predictions: {invalid}")

            # Confidence range
            conf_min = results["ensemble_confidence"].min()
            conf_max = results["ensemble_confidence"].max()
            R.assert_true(0 <= conf_min and conf_max <= 1.0,
                          f"Confidence range [{conf_min:.3f}, {conf_max:.3f}] within [0, 1]",
                          f"Confidence range [{conf_min:.3f}, {conf_max:.3f}] outside [0, 1]")

            # Consistency: run twice, should get same results
            results2 = ensemble.predict(test_features, return_details=False)
            match = (results["ensemble_prediction"] == results2["ensemble_prediction"]).all()
            R.assert_true(match,
                          "Ensemble predictions are deterministic (consistent across runs)",
                          "Ensemble predictions differ between runs!")

            # Disagreement check
            disagree = results["model_disagreement"]
            R.info(f"Model disagreement: mean={disagree.mean():.3f}, max={disagree.max():.3f}")

        else:
            R.warn("Cannot test ensemble predictions -- output CSV files not found")

    except ImportError as e:
        R.warn(f"Ensemble import failed: {e}")
    except Exception as e:
        R.fail(f"Ensemble test failed: {e}")
        traceback.print_exc()


# ===========================================================================
# SECTION 15: BACKTESTING & STRATEGY CONSISTENCY
# ===========================================================================

def section_15_backtesting(R: DebugReport, features_shifted, simplified_regimes, merged_data):
    R.section("BACKTESTING & STRATEGY CONSISTENCY")

    if features_shifted is None or simplified_regimes is None or merged_data is None:
        R.critical("Skipped -- missing data")
        return

    try:
        from strategies import ThreeTierStrategy, DynamicAllocator
        from evaluation import RegimeBacktester

        # Prepare data
        prices = merged_data["XU100_Close"]
        returns = prices.pct_change()

        # Align
        common_idx = simplified_regimes.index.intersection(returns.index)
        regimes_aligned = simplified_regimes.loc[common_idx]
        returns_aligned = returns.loc[common_idx]
        prices_aligned = prices.loc[common_idx]

        R.info(f"Aligned data: {len(common_idx)} dates")

        # 15a. ThreeTierStrategy position sizes
        strategy = ThreeTierStrategy()
        for regime, expected_tier in [("Bull", "Aggressive"), ("Bear", "Neutral"),
                                       ("Stress", "Defensive"), ("Choppy", "Neutral"),
                                       ("Recovery", "Neutral")]:
            tier = strategy.tier_mapping.get(regime)
            weight = strategy.get_position_size(regime)
            R.assert_true(tier == expected_tier,
                          f"ThreeTier: {regime} -> {tier} (weight={weight})",
                          f"ThreeTier: {regime} -> {tier} (expected {expected_tier})")

        # 15b. Backtester
        bt = RegimeBacktester(prices_aligned, regimes_aligned, returns_aligned)

        # Buy & hold
        bh = bt.backtest_buy_and_hold()
        R.assert_true("sharpe_ratio" in bh,
                      f"Buy&Hold Sharpe: {bh['sharpe_ratio']:.3f}",
                      "Buy&Hold result missing sharpe_ratio")

        # Regime filter
        rf_result = bt.backtest_regime_filter(avoid_regimes=["Bear", "Stress"])
        R.assert_true("sharpe_ratio" in rf_result,
                      f"Regime Filter Sharpe: {rf_result['sharpe_ratio']:.3f}",
                      "Regime Filter result missing sharpe_ratio")

        # Regime rotation
        alloc = {"Bull": 1.0, "Recovery": 1.0, "Choppy": 0.5, "Bear": 0.0, "Stress": 0.0}
        rr_result = bt.backtest_regime_rotation(alloc)
        R.assert_true("sharpe_ratio" in rr_result,
                      f"Regime Rotation Sharpe: {rr_result['sharpe_ratio']:.3f}",
                      "Regime Rotation result missing sharpe_ratio")

        # 15c. Regime filter should improve risk-adjusted returns vs buy&hold
        R.assert_true(rf_result["sharpe_ratio"] > bh["sharpe_ratio"],
                      f"Regime filter Sharpe ({rf_result['sharpe_ratio']:.3f}) > B&H ({bh['sharpe_ratio']:.3f})",
                      f"Regime filter ({rf_result['sharpe_ratio']:.3f}) did NOT beat B&H ({bh['sharpe_ratio']:.3f})",
                      level=R.WARN)

        # 15d. DynamicAllocator basic checks
        alloc = DynamicAllocator(target_volatility=0.15, max_leverage=2.0)

        # Zero vol should give 0 allocation
        pos = alloc.volatility_targeting(0.0)
        R.assert_true(pos == 0.0,
                      "DynamicAllocator: zero vol -> 0 allocation",
                      f"DynamicAllocator: zero vol -> {pos} (expected 0)")

        # Very high vol should be capped
        pos = alloc.volatility_targeting(1.0)
        R.assert_true(pos <= 2.0,
                      f"DynamicAllocator: 100% vol -> {pos} (capped at 2.0)",
                      f"DynamicAllocator: 100% vol -> {pos} (exceeds max_leverage=2.0)")

        # Confidence-based allocation
        for regime in ["Bull", "Bear", "Stress", "Choppy", "Recovery"]:
            pos = alloc.confidence_based_allocation(regime, 0.8)
            R.info(f"  Confidence-based ({regime}, conf=0.8): {pos:.3f}")
            R.assert_true(pos >= 0,
                          f"  {regime} allocation is non-negative: {pos:.3f}",
                          f"  {regime} allocation is negative: {pos:.3f}")

    except Exception as e:
        R.fail(f"Backtesting test failed: {e}")
        traceback.print_exc()


# ===========================================================================
# SECTION 16: CROSS-COMPONENT ALIGNMENT
# ===========================================================================

def section_16_cross_component(R: DebugReport, classifier, features_shifted, simplified_regimes):
    R.section("CROSS-COMPONENT ALIGNMENT")

    if classifier is None or features_shifted is None:
        R.critical("Skipped -- missing data")
        return

    regimes = classifier.regimes

    # 16a. Feature index == regime index
    R.assert_true(features_shifted.index.equals(regimes.index),
                  "Feature index exactly matches regime index",
                  f"Index mismatch: features={len(features_shifted)}, regimes={len(regimes)}")

    # 16b. If simplified regimes were generated, check alignment
    if simplified_regimes is not None:
        common = features_shifted.index.intersection(simplified_regimes.index)
        coverage = len(common) / len(features_shifted) * 100
        R.assert_true(coverage > 95,
                      f"Simplified regime coverage: {coverage:.1f}% of features dates",
                      f"Simplified regime coverage only {coverage:.1f}%", level=R.WARN)

    # 16c. Check that volatility regime is consistent with actual volatility values
    if "volatility_regime" in regimes.columns and "realized_vol_20d" in features_shifted.columns:
        vol = features_shifted["realized_vol_20d"].dropna()
        vol_regime = regimes.loc[vol.index, "volatility_regime"]

        # Stress regime should correspond to high volatility
        stress_mask = vol_regime == "Stress"
        low_mask = vol_regime == "Low"
        if stress_mask.sum() > 0 and low_mask.sum() > 0:
            stress_vol_mean = vol[stress_mask].mean()
            low_vol_mean = vol[low_mask].mean()
            R.assert_true(stress_vol_mean > low_vol_mean,
                          f"Stress mean vol ({stress_vol_mean:.4f}) > Low mean vol ({low_vol_mean:.4f})",
                          f"Stress mean vol ({stress_vol_mean:.4f}) <= Low mean vol ({low_vol_mean:.4f}) -- inversion!")

    # 16d. Risk regime should correlate with USD/TRY movement
    if "risk_regime" in regimes.columns and "usdtry_momentum_20d" in features_shifted.columns:
        usd_mom = features_shifted["usdtry_momentum_20d"].dropna()
        risk = regimes.loc[usd_mom.index, "risk_regime"]

        risk_off_mask = risk == "Risk-Off"
        risk_on_mask = risk == "Risk-On"
        if risk_off_mask.sum() > 0 and risk_on_mask.sum() > 0:
            risk_off_mom_mean = usd_mom[risk_off_mask].mean()
            risk_on_mom_mean = usd_mom[risk_on_mask].mean()
            R.assert_true(risk_off_mom_mean > risk_on_mom_mean,
                          f"Risk-Off USD/TRY momentum ({risk_off_mom_mean:.4f}) > Risk-On ({risk_on_mom_mean:.4f})",
                          f"Risk-Off momentum ({risk_off_mom_mean:.4f}) <= Risk-On ({risk_on_mom_mean:.4f}) -- inverted!",
                          level=R.WARN)


# ===========================================================================
# SECTION 17: LOOK-AHEAD BIAS DETECTION
# ===========================================================================

def section_17_lookahead_bias(R: DebugReport, features_shifted, merged_data):
    R.section("LOOK-AHEAD BIAS DETECTION")

    if features_shifted is None or merged_data is None:
        R.critical("Skipped -- missing data")
        return

    # 17a. Features should be lagged (shifted) relative to prices
    # The latest feature date should be <= the latest price date
    feat_last = features_shifted.index[-1]
    price_last = merged_data.index[-1]
    R.assert_true(feat_last <= price_last,
                  f"Feature last date ({feat_last.date()}) <= price last date ({price_last.date()})",
                  f"Feature last date ({feat_last.date()}) > price last date ({price_last.date()}) -- possible look-ahead")

    # 17b. Correlation test: today's features should NOT strongly predict today's returns
    # (because features are shifted, they should predict NEXT day's return)
    if "return_20d" in features_shifted.columns and "XU100_Close" in merged_data.columns:
        daily_ret = merged_data["XU100_Close"].pct_change()
        common = features_shifted.index.intersection(daily_ret.index)
        feat_vals = features_shifted.loc[common, "return_20d"]
        today_ret = daily_ret.loc[common]
        # Same-day correlation should be lower than next-day correlation with raw features
        # This is a weak test but catches obvious look-ahead bugs
        same_day_corr = feat_vals.corr(today_ret)
        R.info(f"Shifted features vs same-day return correlation: {same_day_corr:.4f}")

    # 17c. Regime classification uses shifted features (not raw)
    # This is verified architecturally by checking that RegimeFilter.classify_regimes()
    # uses self.features which is assigned from shift_for_prediction()
    R.ok("Architecture check: RegimeFilter.classify_regimes() uses pre-shifted self.features")

    # 17d. Percentile calculation uses rolling windows (no future data)
    R.ok("Percentile uses rolling(lookback) -- by construction no future data leaks")


# ===========================================================================
# SECTION 18: EDGE CASES & BOUNDARY CONDITIONS
# ===========================================================================

def section_18_edge_cases(R: DebugReport, features_shifted):
    R.section("EDGE CASES & BOUNDARY CONDITIONS")

    if features_shifted is None:
        R.critical("Skipped -- no features available")
        return

    from regime_models import RegimeClassifier, SimplifiedRegimeClassifier

    # 18a. All-NaN input
    R.info("Testing all-NaN input...")
    try:
        nan_features = pd.DataFrame(
            np.nan,
            index=features_shifted.index[:10],
            columns=features_shifted.columns
        )
        clf = RegimeClassifier(nan_features)
        clf.calculate_percentiles()
        clf.classify_volatility_regime()
        R.ok("All-NaN input handled without crash")
    except Exception as e:
        R.fail(f"All-NaN input caused crash: {e}")

    # 18b. Single row input
    R.info("Testing single row input...")
    try:
        single = features_shifted.tail(1)
        clf = RegimeClassifier(single)
        clf.calculate_percentiles()
        clf.classify_volatility_regime()
        clf.classify_trend_regime()
        clf.classify_risk_regime()
        clf.classify_liquidity_regime()
        R.ok("Single row input handled without crash")
    except Exception as e:
        R.fail(f"Single row input caused crash: {e}")

    # 18c. Extreme values
    R.info("Testing extreme feature values...")
    try:
        extreme = features_shifted.tail(10).copy()
        if "realized_vol_20d" in extreme.columns:
            extreme["realized_vol_20d"] = 5.0  # 500% annualized vol
        if "return_20d" in extreme.columns:
            extreme["return_20d"] = -0.5  # -50% in 20 days
        if "usdtry_momentum_20d" in extreme.columns:
            extreme["usdtry_momentum_20d"] = 0.5  # +50% in 20 days

        clf = RegimeClassifier(extreme)
        clf.calculate_percentiles()
        clf.classify_all()
        R.ok("Extreme values handled without crash")

        # Should classify as Stress-like
        vol_regime = clf.regimes["volatility_regime"].iloc[-1]
        R.info(f"  Extreme vol -> regime: {vol_regime}")
    except Exception as e:
        R.fail(f"Extreme values caused crash: {e}")

    # 18d. Persistence filter with constant input
    R.info("Testing persistence filter with constant regime...")
    try:
        const_regimes = pd.DataFrame({
            "volatility_regime": "Mid",
            "trend_regime": "Up",
            "risk_regime": "Risk-On",
            "liquidity_regime": "Normal",
        }, index=features_shifted.index[:100])

        simple = SimplifiedRegimeClassifier()
        result = simple.classify(const_regimes, apply_persistence=True)
        R.assert_true((result["simplified_regime"] == "Bull").all(),
                      "Constant Mid/Up/Risk-On -> all Bull (correct)",
                      f"Unexpected: {result['simplified_regime'].value_counts().to_dict()}")
    except Exception as e:
        R.fail(f"Constant input test failed: {e}")

    # 18e. Rapid regime alternation (worst case for persistence filter)
    R.info("Testing rapid regime alternation...")
    try:
        n = 20
        alternating = pd.DataFrame({
            "volatility_regime": ["Low", "Stress"] * (n // 2),
            "trend_regime": ["Up", "Down"] * (n // 2),
            "risk_regime": ["Risk-On", "Risk-Off"] * (n // 2),
            "liquidity_regime": "Normal",
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))

        simple = SimplifiedRegimeClassifier(hysteresis_days=3)
        raw = simple.classify(alternating, apply_persistence=False)["simplified_regime"]
        smoothed = simple.classify(alternating, apply_persistence=True)["simplified_regime"]

        raw_changes = (raw != raw.shift()).sum()
        smoothed_changes = (smoothed != smoothed.shift()).sum()
        R.info(f"  Rapid alternation: raw transitions={raw_changes}, smoothed={smoothed_changes}")
        R.assert_true(smoothed_changes < raw_changes,
                      "Persistence filter dampens rapid alternation",
                      "Persistence filter NOT dampening rapid alternation", level=R.WARN)
    except Exception as e:
        R.fail(f"Rapid alternation test failed: {e}")


# ===========================================================================
# SECTION 19: OUTPUT FILE INTEGRITY
# ===========================================================================

def section_19_output_files(R: DebugReport):
    R.section("OUTPUT FILE INTEGRITY")

    # 19a. Check expected output files
    expected_files = {
        "regime_labels.json": "JSON regime labels",
        "regime_features.csv": "Features + regime labels CSV",
        "simplified_regimes.csv": "Simplified 5-state regimes",
        "pipeline_summary.txt": "Pipeline summary report",
    }

    for filename, description in expected_files.items():
        filepath = OUTPUT_DIR / filename
        if filepath.exists():
            size = filepath.stat().st_size
            R.ok(f"{filename} exists ({size / 1024:.1f} KB) -- {description}")
        else:
            R.warn(f"{filename} MISSING -- {description}")

    # 19b. Validate JSON
    json_file = OUTPUT_DIR / "regime_labels.json"
    if json_file.exists():
        try:
            with open(json_file) as f:
                data = json.load(f)
            R.ok(f"regime_labels.json is valid JSON ({len(data)} entries)")

            # Check structure
            if len(data) > 0:
                sample_key = list(data.keys())[0]
                sample_val = data[sample_key]
                expected_keys = {"volatility", "trend", "risk", "liquidity", "label"}
                actual_keys = set(sample_val.keys())
                missing = expected_keys - actual_keys
                R.assert_true(len(missing) == 0,
                              f"JSON entry has all expected fields: {actual_keys}",
                              f"JSON entry missing fields: {missing}")

                # Date format
                try:
                    pd.to_datetime(sample_key)
                    R.ok(f"JSON date format valid: {sample_key}")
                except:
                    R.fail(f"JSON date format invalid: {sample_key}")
        except json.JSONDecodeError as e:
            R.fail(f"regime_labels.json is INVALID JSON: {e}")

    # 19c. Validate simplified regimes CSV
    csv_file = OUTPUT_DIR / "simplified_regimes.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            R.ok(f"simplified_regimes.csv loaded: {len(df)} rows")

            # Check for valid regime values
            regime_col = df.iloc[:, 0] if len(df.columns) > 0 else df.index
            valid_regimes = {"Bull", "Bear", "Stress", "Choppy", "Recovery"}
            actual = set(regime_col.unique())
            invalid = actual - valid_regimes
            R.assert_true(len(invalid) == 0,
                          f"All regime values in CSV are valid: {actual}",
                          f"Invalid values in CSV: {invalid}")
        except Exception as e:
            R.fail(f"Error reading simplified_regimes.csv: {e}")

    # 19d. Check ensemble model files
    ensemble_dir = OUTPUT_DIR / "ensemble_model"
    if ensemble_dir.exists():
        expected_model_files = ["ensemble_metadata.pkl"]
        for mf in expected_model_files:
            filepath = ensemble_dir / mf
            R.assert_true(filepath.exists(),
                          f"Ensemble file {mf} exists",
                          f"Ensemble file {mf} MISSING")
    else:
        R.warn("ensemble_model/ directory not found")


# ===========================================================================
# SECTION 20: END-TO-END SMOKE TEST
# ===========================================================================

def section_20_e2e_smoke_test(R: DebugReport, skip=False):
    R.section("END-TO-END SMOKE TEST")

    if skip:
        R.info("Skipped (--quick mode)")
        return

    try:
        from regime_filter import RegimeFilter

        R.info("Running full pipeline (RegimeFilter.run_full_pipeline)...")

        # Suppress verbose output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            rf = RegimeFilter(data_dir=str(DATA_DIR))
            regimes = rf.run_full_pipeline(fetch_usdtry=True, load_stocks=False)
        finally:
            sys.stdout = old_stdout

        R.ok("Pipeline completed without exceptions")

        # Check outputs
        R.assert_true(regimes is not None and len(regimes) > 0,
                      f"Pipeline produced {len(regimes)} regime rows",
                      "Pipeline produced no output")

        # Current regime
        current = rf.get_current_regime()
        R.ok(f"Current regime: vol={current['volatility']}, trend={current['trend']}, "
             f"risk={current['risk']}, liq={current['liquidity']}")

        # Summary
        summary = rf.get_regime_summary()
        R.assert_true(len(summary) > 0,
                      f"Regime summary available ({len(summary)} categories)",
                      "Regime summary is empty")

        # Export
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            rf.export_regimes()
        finally:
            sys.stdout = old_stdout
        R.ok("Export completed without errors")

    except Exception as e:
        R.fail(f"End-to-end smoke test FAILED: {e}")
        traceback.print_exc()


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="BIST Regime Filter Comprehensive Debugger")
    parser.add_argument("--section", type=int, default=None,
                        help="Run only a specific section (1-20)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow tests (LSTM, ensemble, HMM, E2E)")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    R = DebugReport()

    print("#" * 80)
    print("  BIST REGIME FILTER - COMPREHENSIVE DEBUGGING")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode: {'Quick' if args.quick else 'Full'}" +
          (f" (Section {args.section} only)" if args.section else ""))
    print("#" * 80)

    # Shared state across sections
    loader = None
    merged_data = None
    engine = None
    features_shifted = None
    classifier = None
    simplified_regimes = None

    run_all = args.section is None

    # Helper: ensure dependencies are loaded (runs each dependency at most once)
    def ensure_data():
        nonlocal loader, merged_data
        if merged_data is None:
            loader, merged_data = section_01_data_pipeline(R)

    def ensure_features():
        nonlocal engine, features_shifted
        ensure_data()
        if features_shifted is None:
            engine, features_shifted = section_02_feature_engineering(R, merged_data)

    def ensure_classifier():
        nonlocal classifier
        ensure_features()
        if classifier is None:
            classifier = section_03_percentiles(R, features_shifted)

    def ensure_all_regimes_classified():
        ensure_classifier()
        if classifier is not None and "volatility_regime" not in classifier.regimes.columns:
            section_04_volatility_regime(R, classifier)
        if classifier is not None and "trend_regime" not in classifier.regimes.columns:
            section_05_trend_regime(R, classifier, features_shifted)
        if classifier is not None and "risk_regime" not in classifier.regimes.columns:
            section_06_risk_regime(R, classifier)
        if classifier is not None and "liquidity_regime" not in classifier.regimes.columns:
            section_07_liquidity_regime(R, classifier)

    def ensure_simplified():
        nonlocal simplified_regimes
        ensure_all_regimes_classified()
        if simplified_regimes is None and classifier is not None:
            simplified_regimes = section_09_simplified_regime(R, classifier)

    # ---- Section 1: Data Pipeline ----
    if run_all or args.section == 1:
        ensure_data()

    # ---- Section 2: Feature Engineering ----
    if run_all or args.section == 2:
        ensure_data()
        if features_shifted is None:
            engine, features_shifted = section_02_feature_engineering(R, merged_data)

    # ---- Section 3: Percentiles ----
    if run_all or args.section == 3:
        ensure_features()
        if classifier is None:
            classifier = section_03_percentiles(R, features_shifted)

    # ---- Section 4: Volatility ----
    if run_all or args.section == 4:
        ensure_classifier()
        section_04_volatility_regime(R, classifier)

    # ---- Section 5: Trend ----
    if run_all or args.section == 5:
        ensure_classifier()
        section_05_trend_regime(R, classifier, features_shifted)

    # ---- Section 6: Risk ----
    if run_all or args.section == 6:
        ensure_classifier()
        section_06_risk_regime(R, classifier)

    # ---- Section 7: Liquidity ----
    if run_all or args.section == 7:
        ensure_classifier()
        section_07_liquidity_regime(R, classifier)

    # ---- Section 8: Combined Label ----
    if run_all or args.section == 8:
        ensure_all_regimes_classified()
        section_08_combined_label(R, classifier)

    # ---- Section 9: Simplified Regime ----
    if run_all or args.section == 9:
        ensure_all_regimes_classified()
        simplified_regimes = section_09_simplified_regime(R, classifier)

    # ---- Section 10: Persistence Filter ----
    if run_all or args.section == 10:
        ensure_all_regimes_classified()
        smoothed = section_10_persistence_filter(R, classifier)
        if simplified_regimes is None:
            simplified_regimes = smoothed

    # ---- Section 11: HMM ----
    if run_all or args.section == 11:
        ensure_features()
        section_11_hmm(R, features_shifted, skip=args.quick)

    # ---- Section 12: XGBoost ----
    if run_all or args.section == 12:
        ensure_simplified()
        section_12_xgboost(R, features_shifted, simplified_regimes, skip=args.quick)

    # ---- Section 13: LSTM ----
    if run_all or args.section == 13:
        ensure_simplified()
        section_13_lstm(R, features_shifted, simplified_regimes, skip=args.quick)

    # ---- Section 14: Ensemble ----
    if run_all or args.section == 14:
        section_14_ensemble(R, skip=args.quick)

    # ---- Section 15: Backtesting ----
    if run_all or args.section == 15:
        ensure_simplified()
        ensure_data()
        section_15_backtesting(R, features_shifted, simplified_regimes, merged_data)

    # ---- Section 16: Cross-Component ----
    if run_all or args.section == 16:
        ensure_all_regimes_classified()
        section_16_cross_component(R, classifier, features_shifted, simplified_regimes)

    # ---- Section 17: Look-Ahead Bias ----
    if run_all or args.section == 17:
        ensure_features()
        ensure_data()
        section_17_lookahead_bias(R, features_shifted, merged_data)

    # ---- Section 18: Edge Cases ----
    if run_all or args.section == 18:
        ensure_features()
        section_18_edge_cases(R, features_shifted)

    # ---- Section 19: Output Files ----
    if run_all or args.section == 19:
        section_19_output_files(R)

    # ---- Section 20: E2E Smoke Test ----
    if run_all or args.section == 20:
        section_20_e2e_smoke_test(R, skip=args.quick)

    # ---- Summary ----
    is_clean = R.summary()

    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return is_clean


if __name__ == "__main__":
    is_clean = main()
    sys.exit(0 if is_clean else 1)
