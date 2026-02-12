"""
TCMB EVDS Data Fetcher for Leading Indicators
Fetches Turkish-specific market data: VIOP30 (implied vol), CDS spreads, yield curve, policy rates

EVDS Series Codes Reference:
- Yield curve: TP.DK.TRE.YSKG (government bond yields by maturity)
- Policy rate: TP.PH.S01 (1-week repo rate)
- Inflation: TP.FG.J0 (CPI annual)
- Inflation expectations: TP.BEK.S01.A (12-month ahead)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import os

class TCMBDataFetcher:
    """
    Fetch Turkey-specific leading indicators from TCMB EVDS

    Data Sources:
    1. VIOP30 - Turkish VIX (computed from options or proxy)
    2. Turkey CDS 5Y - Credit risk indicator
    3. Yield Curve - 2Y vs 10Y spread
    4. Policy Rate - TCMB 1-week repo
    5. Inflation & Expectations
    """

    # EVDS Series Codes
    SERIES_CODES = {
        # Government Bond Yields (various maturities)
        'yield_2y': 'TP.DK.TRE.YSKG.02',      # 2-year bond yield
        'yield_5y': 'TP.DK.TRE.YSKG.05',      # 5-year bond yield
        'yield_10y': 'TP.DK.TRE.YSKG.10',     # 10-year bond yield

        # Policy Rate
        'policy_rate': 'TP.PH.S01',            # 1-week repo rate

        # Inflation
        'cpi_annual': 'TP.FG.J0',              # CPI year-over-year
        'inflation_exp': 'TP.BEK.S01.A',       # 12-month inflation expectations

        # Deposit Rates
        'deposit_rate': 'TP.FG2.A01',          # Weighted avg deposit rate

        # Interbank rates
        'overnight_rate': 'TP.PH.S03',         # Overnight lending rate
    }

    def __init__(self, api_key=None, cache_dir=None):
        """
        Args:
            api_key: EVDS API key (get from evds2.tcmb.gov.tr)
                     Can also be set via TCMB_EVDS_API_KEY environment variable
            cache_dir: Directory to cache downloaded data
        """
        self.api_key = api_key or os.environ.get('TCMB_EVDS_API_KEY')

        if cache_dir is None:
            current_file = Path(__file__).resolve()
            regime_filter_dir = current_file.parent
            bist_dir = regime_filter_dir.parent
            self.cache_dir = bist_dir / "data" / "tcmb_cache"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.data = {}
        self._evds_client = None

    @staticmethod
    def _required_cache_end(end_ts: pd.Timestamp, max_stale_days: int = 7) -> pd.Timestamp:
        """Required minimum cache end date for requested window."""
        now = pd.Timestamp.now().normalize()
        cutoff = now - pd.Timedelta(days=max_stale_days)
        return end_ts if end_ts < cutoff else end_ts - pd.Timedelta(days=max_stale_days)

    @classmethod
    def _cache_covers_window(
        cls,
        cached_index: pd.DatetimeIndex,
        start_date: str,
        end_date: str,
        max_stale_days: int = 7,
    ) -> bool:
        if len(cached_index) == 0:
            return False
        idx = pd.DatetimeIndex(cached_index).dropna().sort_values()
        if len(idx) == 0:
            return False

        start_ts = pd.to_datetime(start_date).normalize()
        end_ts = pd.to_datetime(end_date).normalize()
        if idx.min() > start_ts:
            return False

        required_end = cls._required_cache_end(end_ts, max_stale_days=max_stale_days)
        return idx.max() >= required_end

    @staticmethod
    def _slice_series_to_window(series: pd.Series | None, start_date: str, end_date: str) -> pd.Series | None:
        if series is None:
            return None
        start_ts = pd.to_datetime(start_date).normalize()
        end_ts = pd.to_datetime(end_date).normalize()
        out = series.sort_index()
        out = out[(out.index >= start_ts) & (out.index <= end_ts)]
        return out if len(out) > 0 else None

    def _get_evds_client(self):
        """Lazy initialization of EVDS client"""
        if self._evds_client is None:
            try:
                import evds
                if not self.api_key:
                    raise ValueError(
                        "EVDS API key required. Get one from: https://evds2.tcmb.gov.tr/\n"
                        "Set via: TCMBDataFetcher(api_key='your_key') or "
                        "TCMB_EVDS_API_KEY environment variable"
                    )
                self._evds_client = evds.evdsAPI(self.api_key)
            except ImportError:
                raise ImportError(
                    "evds package not installed. Install with: pip install evds"
                )
        return self._evds_client

    def fetch_all(self, start_date='2013-01-01', end_date=None, use_cache=True):
        """
        Fetch all Turkish leading indicators

        Returns:
            DataFrame with all indicators aligned by date
        """
        print("="*70)
        print("FETCHING TCMB LEADING INDICATORS")
        print("="*70)

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Try to fetch from EVDS
        try:
            self.fetch_yield_curve(start_date, end_date, use_cache)
            self.fetch_policy_rates(start_date, end_date, use_cache)
            self.fetch_inflation_data(start_date, end_date, use_cache)
            self.fetch_cds_proxy(start_date, end_date)  # Computed proxy
            self.fetch_viop30_proxy(start_date, end_date)  # Computed proxy
        except Exception as e:
            print(f"\nWarning: EVDS fetch failed: {e}")
            print("Falling back to proxy calculations...")
            self._calculate_all_proxies(start_date, end_date)

        # Combine all data
        combined = self._combine_data()

        # Calculate derived indicators
        combined = self._calculate_derived_indicators(combined)

        print(f"\n{'='*70}")
        print(f"Fetched {len(combined.columns)} TCMB indicators")
        if len(combined) > 0:
            print(f"Date range: {combined.index[0]} to {combined.index[-1]}")
            print(f"Observations: {len(combined)}")
        print("="*70)

        return combined

    def fetch_yield_curve(self, start_date, end_date, use_cache=True):
        """Fetch government bond yields for yield curve"""
        print("\n[1/5] Fetching Yield Curve Data...")

        cache_file = self.cache_dir / "yield_curve.csv"

        if use_cache and cache_file.exists():
            print("  Loading from cache...")
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if self._cache_covers_window(cached.index, start_date, end_date):
                y2 = self._slice_series_to_window(cached.get('yield_2y'), start_date, end_date)
                y5 = self._slice_series_to_window(cached.get('yield_5y'), start_date, end_date)
                y10 = self._slice_series_to_window(cached.get('yield_10y'), start_date, end_date)
                if y2 is not None:
                    self.data['yield_2y'] = y2
                if y5 is not None:
                    self.data['yield_5y'] = y5
                if y10 is not None:
                    self.data['yield_10y'] = y10
                if any(x is not None for x in (y2, y5, y10)):
                    print(f"  Loaded {len(cached)} observations from cache")
                    return

        try:
            client = self._get_evds_client()

            # Format dates for EVDS (DD-MM-YYYY)
            start_fmt = pd.to_datetime(start_date).strftime('%d-%m-%Y')
            end_fmt = pd.to_datetime(end_date).strftime('%d-%m-%Y')

            # Fetch yields
            for name, code in [('yield_2y', self.SERIES_CODES['yield_2y']),
                              ('yield_5y', self.SERIES_CODES['yield_5y']),
                              ('yield_10y', self.SERIES_CODES['yield_10y'])]:
                try:
                    print(f"  - Fetching {name} ({code})...")
                    data = client.get_data([code], startdate=start_fmt, enddate=end_fmt)
                    if data is not None and len(data) > 0:
                        data.columns = ['Date', name]
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        data[name] = pd.to_numeric(data[name], errors='coerce') / 100  # Convert to decimal
                        self.data[name] = data[name]
                        print(f"    Fetched {len(data)} observations")
                except Exception as e:
                    print(f"    Warning: Could not fetch {name}: {e}")

            # Cache the data
            if self.data:
                yield_df = pd.DataFrame(self.data)
                yield_df.to_csv(cache_file)
                print(f"  Cached to {cache_file}")

        except Exception as e:
            print(f"  Error fetching yield curve: {e}")
            self._calculate_yield_proxy(start_date, end_date)

    def fetch_policy_rates(self, start_date, end_date, use_cache=True):
        """Fetch TCMB policy rates"""
        print("\n[2/5] Fetching Policy Rates...")

        cache_file = self.cache_dir / "policy_rates.csv"

        if use_cache and cache_file.exists():
            print("  Loading from cache...")
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if self._cache_covers_window(cached.index, start_date, end_date):
                policy = self._slice_series_to_window(cached.get('policy_rate'), start_date, end_date)
                overnight = self._slice_series_to_window(cached.get('overnight_rate'), start_date, end_date)
                if policy is not None:
                    self.data['policy_rate'] = policy
                if overnight is not None:
                    self.data['overnight_rate'] = overnight
                if policy is not None or overnight is not None:
                    print(f"  Loaded {len(cached)} observations from cache")
                    return

        try:
            client = self._get_evds_client()
            start_fmt = pd.to_datetime(start_date).strftime('%d-%m-%Y')
            end_fmt = pd.to_datetime(end_date).strftime('%d-%m-%Y')

            for name, code in [('policy_rate', self.SERIES_CODES['policy_rate']),
                              ('overnight_rate', self.SERIES_CODES['overnight_rate'])]:
                try:
                    print(f"  - Fetching {name}...")
                    data = client.get_data([code], startdate=start_fmt, enddate=end_fmt)
                    if data is not None and len(data) > 0:
                        data.columns = ['Date', name]
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        data[name] = pd.to_numeric(data[name], errors='coerce') / 100
                        self.data[name] = data[name]
                        print(f"    Fetched {len(data)} observations")
                except Exception as e:
                    print(f"    Warning: Could not fetch {name}: {e}")

            # Cache
            if 'policy_rate' in self.data:
                rates_df = pd.DataFrame({k: v for k, v in self.data.items()
                                        if k in ['policy_rate', 'overnight_rate']})
                rates_df.to_csv(cache_file)

        except Exception as e:
            print(f"  Error fetching policy rates: {e}")

    def fetch_inflation_data(self, start_date, end_date, use_cache=True):
        """Fetch inflation and inflation expectations"""
        print("\n[3/5] Fetching Inflation Data...")

        cache_file = self.cache_dir / "inflation.csv"

        if use_cache and cache_file.exists():
            print("  Loading from cache...")
            cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if self._cache_covers_window(cached.index, start_date, end_date):
                cpi = self._slice_series_to_window(cached.get('cpi_annual'), start_date, end_date)
                infl_exp = self._slice_series_to_window(cached.get('inflation_exp'), start_date, end_date)
                if cpi is not None:
                    self.data['cpi_annual'] = cpi
                if infl_exp is not None:
                    self.data['inflation_exp'] = infl_exp
                if cpi is not None or infl_exp is not None:
                    print(f"  Loaded {len(cached)} observations from cache")
                    return

        try:
            client = self._get_evds_client()
            start_fmt = pd.to_datetime(start_date).strftime('%d-%m-%Y')
            end_fmt = pd.to_datetime(end_date).strftime('%d-%m-%Y')

            for name, code in [('cpi_annual', self.SERIES_CODES['cpi_annual']),
                              ('inflation_exp', self.SERIES_CODES['inflation_exp'])]:
                try:
                    print(f"  - Fetching {name}...")
                    data = client.get_data([code], startdate=start_fmt, enddate=end_fmt)
                    if data is not None and len(data) > 0:
                        data.columns = ['Date', name]
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        data[name] = pd.to_numeric(data[name], errors='coerce') / 100
                        self.data[name] = data[name]
                        print(f"    Fetched {len(data)} observations")
                except Exception as e:
                    print(f"    Warning: Could not fetch {name}: {e}")

            # Cache
            if 'cpi_annual' in self.data:
                infl_df = pd.DataFrame({k: v for k, v in self.data.items()
                                       if k in ['cpi_annual', 'inflation_exp']})
                infl_df.to_csv(cache_file)

        except Exception as e:
            print(f"  Error fetching inflation data: {e}")

    def fetch_cds_proxy(self, start_date, end_date):
        """
        Calculate Turkey CDS proxy from bond spreads
        CDS data typically requires Bloomberg/Refinitiv, so we compute a proxy

        Proxy: Turkey 5Y yield - US 5Y yield (simplified credit spread)
        """
        print("\n[4/5] Calculating CDS Proxy...")

        try:
            import yfinance as yf

            # Fetch US 5Y yield
            print("  - Fetching US 5Y Treasury yield...")
            us5y = yf.download('^FVX', start=start_date, end=end_date, progress=False)

            if not us5y.empty:
                if isinstance(us5y['Close'], pd.DataFrame):
                    us5y_close = us5y['Close'].iloc[:, 0]
                else:
                    us5y_close = us5y['Close']

                self.data['us_5y_yield'] = us5y_close / 100  # Convert to decimal
                print(f"    Fetched {len(us5y)} observations")

                # If we have Turkey 5Y, calculate spread
                if 'yield_5y' in self.data and self.data['yield_5y'] is not None:
                    # Align indices
                    common_idx = self.data['yield_5y'].index.intersection(us5y_close.index)
                    tr_5y = self.data['yield_5y'].reindex(common_idx)
                    us_5y = (us5y_close / 100).reindex(common_idx)

                    self.data['cds_proxy'] = tr_5y - us_5y
                    print(f"  CDS proxy calculated: {len(common_idx)} observations")
                else:
                    # Use USD/TRY volatility as alternative proxy
                    print("  Turkey yield not available, using USD/TRY vol as CDS proxy")
                    self._calculate_cds_from_fx(start_date, end_date)
            else:
                print("  Warning: US yield data not available")

        except Exception as e:
            print(f"  Error calculating CDS proxy: {e}")
            self._calculate_cds_from_fx(start_date, end_date)

    def _calculate_cds_from_fx(self, start_date, end_date):
        """Alternative CDS proxy using USD/TRY volatility"""
        try:
            import yfinance as yf

            usdtry = yf.download('TRY=X', start=start_date, end=end_date, progress=False)
            if not usdtry.empty:
                if isinstance(usdtry['Close'], pd.DataFrame):
                    close = usdtry['Close'].iloc[:, 0]
                else:
                    close = usdtry['Close']

                # CDS proxy: 20-day realized vol of USD/TRY (higher vol = higher credit risk)
                usdtry_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
                self.data['cds_proxy'] = usdtry_vol * 100  # Scale to basis points style
                print(f"  CDS proxy (from FX vol): {len(usdtry_vol)} observations")
        except Exception as e:
            print(f"  Could not calculate CDS proxy: {e}")

    def fetch_viop30_proxy(self, start_date, end_date):
        """
        Calculate VIOP30 (Turkish VIX) proxy

        VIOP30 is not readily available via free APIs, so we compute:
        1. Implied vol proxy from price dynamics
        2. VIX-based adjustment for Turkish market
        """
        print("\n[5/5] Calculating VIOP30 Proxy...")

        try:
            import yfinance as yf

            # Method 1: Scale VIX by Turkey/US volatility ratio
            print("  - Fetching VIX and XU100 for proxy calculation...")

            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            xu100 = yf.download('XU100.IS', start=start_date, end=end_date, progress=False)
            spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

            if not vix.empty and not xu100.empty and not spx.empty:
                # Get close prices
                vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
                xu100_close = xu100['Close'].iloc[:, 0] if isinstance(xu100['Close'], pd.DataFrame) else xu100['Close']
                spx_close = spx['Close'].iloc[:, 0] if isinstance(spx['Close'], pd.DataFrame) else spx['Close']

                # Calculate realized vols
                xu100_rvol = xu100_close.pct_change().rolling(20).std() * np.sqrt(252) * 100
                spx_rvol = spx_close.pct_change().rolling(20).std() * np.sqrt(252) * 100

                # Volatility ratio (Turkey typically 1.5-2x US vol)
                vol_ratio = xu100_rvol / spx_rvol.reindex(xu100_rvol.index, method='ffill')
                vol_ratio = vol_ratio.clip(1.0, 3.0)  # Cap at reasonable range

                # VIOP30 proxy = VIX * vol_ratio
                vix_aligned = vix_close.reindex(xu100_rvol.index, method='ffill')
                viop30_proxy = vix_aligned * vol_ratio

                self.data['viop30_proxy'] = viop30_proxy
                self.data['vix'] = vix_aligned
                self.data['xu100_rvol'] = xu100_rvol

                print(f"  VIOP30 proxy calculated: {len(viop30_proxy)} observations")
                print(f"  Average VIOP30 proxy: {viop30_proxy.mean():.1f}")
                print(f"  Average vol ratio (TR/US): {vol_ratio.mean():.2f}")
            else:
                print("  Warning: Could not fetch required data for VIOP30 proxy")

        except Exception as e:
            print(f"  Error calculating VIOP30 proxy: {e}")

    def _calculate_yield_proxy(self, start_date, end_date):
        """Calculate yield curve proxy when EVDS unavailable"""
        print("  Calculating yield proxy from USD/TRY...")
        try:
            import yfinance as yf

            usdtry = yf.download('TRY=X', start=start_date, end=end_date, progress=False)
            if not usdtry.empty:
                close = usdtry['Close'].iloc[:, 0] if isinstance(usdtry['Close'], pd.DataFrame) else usdtry['Close']

                # Proxy: base yield + depreciation premium
                depreciation = close.pct_change(252).clip(lower=0)
                base_yield = 0.15  # 15% base

                self.data['yield_proxy'] = base_yield + depreciation * 0.5
                print(f"  Yield proxy calculated: {len(self.data['yield_proxy'])} observations")
        except Exception as e:
            print(f"  Could not calculate yield proxy: {e}")

    def _calculate_all_proxies(self, start_date, end_date):
        """Calculate all proxies when EVDS is unavailable"""
        print("\nCalculating all proxies (EVDS unavailable)...")
        self._calculate_yield_proxy(start_date, end_date)
        self._calculate_cds_from_fx(start_date, end_date)
        self.fetch_viop30_proxy(start_date, end_date)

    def _combine_data(self):
        """Combine all fetched data into single DataFrame"""
        if not self.data:
            return pd.DataFrame()

        # Filter out None values
        valid_data = {k: v for k, v in self.data.items() if v is not None}

        if not valid_data:
            return pd.DataFrame()

        combined = pd.concat(valid_data, axis=1)
        combined = combined.ffill()  # Forward fill for missing values

        return combined

    def _calculate_derived_indicators(self, df):
        """Calculate derived indicators from raw data"""
        if df.empty:
            return df

        # Yield Curve Slope (10Y - 2Y)
        if 'yield_10y' in df.columns and 'yield_2y' in df.columns:
            df['yield_curve_slope'] = df['yield_10y'] - df['yield_2y']
            print("  + Calculated yield_curve_slope")

        # Real Interest Rate (policy rate - inflation)
        if 'policy_rate' in df.columns and 'cpi_annual' in df.columns:
            df['real_rate'] = df['policy_rate'] - df['cpi_annual']
            print("  + Calculated real_rate")

        # Inflation Surprise (actual - expected)
        if 'cpi_annual' in df.columns and 'inflation_exp' in df.columns:
            df['inflation_surprise'] = df['cpi_annual'] - df['inflation_exp'].shift(12)  # 12-month lag
            print("  + Calculated inflation_surprise")

        # IV-RV Spread (implied - realized volatility)
        if 'viop30_proxy' in df.columns and 'xu100_rvol' in df.columns:
            df['iv_rv_spread'] = df['viop30_proxy'] - df['xu100_rvol']
            print("  + Calculated iv_rv_spread")

        # CDS momentum
        if 'cds_proxy' in df.columns:
            df['cds_change_20d'] = df['cds_proxy'].diff(20)
            df['cds_ma_ratio'] = df['cds_proxy'] / df['cds_proxy'].rolling(60).mean()
            print("  + Calculated CDS momentum indicators")

        # VIOP30 momentum
        if 'viop30_proxy' in df.columns:
            df['viop30_change_5d'] = df['viop30_proxy'].pct_change(5)
            df['viop30_change_20d'] = df['viop30_proxy'].pct_change(20)
            df['viop30_ma_ratio'] = df['viop30_proxy'] / df['viop30_proxy'].rolling(60).mean()
            print("  + Calculated VIOP30 momentum indicators")

        # Policy rate momentum
        if 'policy_rate' in df.columns:
            df['policy_rate_change'] = df['policy_rate'].diff(20)
            print("  + Calculated policy rate change")

        return df

    def get_data_summary(self):
        """Get summary of available data"""
        if not self.data:
            return "No data fetched yet. Call fetch_all() first."

        lines = []
        lines.append("\n" + "="*70)
        lines.append("TCMB DATA SUMMARY")
        lines.append("="*70)

        for name, series in self.data.items():
            if series is not None and len(series) > 0:
                non_null = series.notna().sum()
                coverage = (non_null / len(series)) * 100
                lines.append(f"{name:25s}: {len(series):5d} obs, {coverage:5.1f}% coverage")

        return "\n".join(lines)


if __name__ == "__main__":
    print("="*70)
    print("TCMB DATA FETCHER TEST")
    print("="*70)

    # Initialize fetcher (will use proxies if no API key)
    fetcher = TCMBDataFetcher()

    # Fetch all data
    data = fetcher.fetch_all(start_date='2020-01-01')

    # Print summary
    print(fetcher.get_data_summary())

    # Show sample data
    print("\n" + "="*70)
    print("SAMPLE DATA (Last 10 rows)")
    print("="*70)
    if not data.empty:
        print(data.tail(10))

    # Show key statistics
    print("\n" + "="*70)
    print("KEY STATISTICS")
    print("="*70)
    if not data.empty:
        key_cols = ['viop30_proxy', 'cds_proxy', 'yield_curve_slope', 'real_rate']
        available_cols = [c for c in key_cols if c in data.columns]
        if available_cols:
            print(data[available_cols].describe())

    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("To use actual TCMB data:")
    print("1. Get API key from: https://evds2.tcmb.gov.tr/")
    print("2. Set environment variable: export TCMB_EVDS_API_KEY='your_key'")
    print("3. Or pass directly: TCMBDataFetcher(api_key='your_key')")
    print("\nWithout API key, proxy calculations are used.")
    print("="*70)
