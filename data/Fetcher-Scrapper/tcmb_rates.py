"""
TCMB (Turkish Central Bank) Risk-Free Rate Fetcher
Fetches Turkish deposit rates from EVDS (Electronic Data Distribution System)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings

class TCMBRateFetcher:
    """
    Fetch Turkish deposit rates from TCMB EVDS
    Falls back to approximation if API unavailable
    """
    
    def __init__(self, api_key=None, cache_dir=None):
        """
        Args:
            api_key: EVDS API key (get from evds2.tcmb.gov.tr)
            cache_dir: Directory to cache downloaded rates
        """
        self.api_key = api_key
        
        if cache_dir is None:
            # Default to BIST/data directory
            current_file = Path(__file__).resolve()
            regime_filter_dir = current_file.parent
            bist_dir = regime_filter_dir.parent
            self.cache_dir = bist_dir / "data"
        else:
            self.cache_dir = Path(cache_dir)
            
        self.cache_file = self.cache_dir / "tcmb_deposit_rates.csv"
        self.rates_data = None
        
    def fetch_rates(self, start_date='2013-01-01', end_date=None, force_refresh=False):
        """
        Fetch Turkish deposit rates from TCMB EVDS
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            force_refresh: Force re-download even if cached
        
        Returns:
            DataFrame with Date index and deposit_rate column
        """
        # Check cache first
        if not force_refresh and self.cache_file.exists():
            print(f"Loading cached TCMB rates from {self.cache_file}")
            self.rates_data = pd.read_csv(self.cache_file, index_col=0, parse_dates=True)
            return self.rates_data
        
        # Try to fetch from EVDS
        try:
            self.rates_data = self._fetch_from_evds(start_date, end_date)
            
            # Cache the data
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            self.rates_data.to_csv(self.cache_file)
            print(f"TCMB rates cached to {self.cache_file}")
            
        except Exception as e:
            print(f"Warning: Could not fetch from EVDS: {e}")
            print("Falling back to approximation based on USD/TRY...")
            self.rates_data = self._approximate_rates(start_date, end_date)
        
        return self.rates_data
    
    def _fetch_from_evds(self, start_date, end_date):
        """Fetch actual rates from TCMB EVDS API"""
        try:
            import evds
        except ImportError:
            raise ImportError(
                "evds package not installed. Install with: pip install evds\n"
                "Or get API key from: https://evds2.tcmb.gov.tr/"
            )
        
        if not self.api_key:
            raise ValueError(
                "EVDS API key required. Get one from: https://evds2.tcmb.gov.tr/\n"
                "Then pass it to TCMBRateFetcher(api_key='your_key')"
            )
        
        # Initialize EVDS client
        evds_client = evds.evdsAPI(self.api_key)
        
        # TCMB deposit rate series codes
        # TP.DK.USD.A.YTL - USD Deposit Rate (most commonly used as risk-free proxy)
        # TP.FG2.A01 - Weighted Average Interest Rate on Deposits
        series_code = 'TP.FG2.A01'  # Weighted average deposit rate
        
        print(f"Fetching TCMB deposit rates (series: {series_code})...")
        
        # Fetch data
        if end_date is None:
            end_date = datetime.now().strftime('%d-%m-%Y')
        else:
            end_date = pd.to_datetime(end_date).strftime('%d-%m-%Y')
        
        start_date = pd.to_datetime(start_date).strftime('%d-%m-%Y')
        
        # Get data from EVDS
        data = evds_client.get_data([series_code], startdate=start_date, enddate=end_date)
        
        # Process data
        data.columns = ['Date', 'deposit_rate']
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        
        # Convert to decimal (EVDS returns percentages)
        data['deposit_rate'] = data['deposit_rate'] / 100.0
        
        # Forward fill missing values
        data = data.resample('D').ffill()
        
        print(f"Fetched {len(data)} days of TCMB deposit rates")
        print(f"Rate range: {data['deposit_rate'].min():.2%} to {data['deposit_rate'].max():.2%}")
        
        return data
    
    def _approximate_rates(self, start_date, end_date):
        """
        Approximate Turkish deposit rates based on USD/TRY dynamics
        Used as fallback when EVDS API unavailable
        """
        import yfinance as yf
        
        print("Approximating Turkish deposit rates from USD/TRY...")
        
        # Fetch USD/TRY
        usdtry = yf.download('TRY=X', start=start_date, end=end_date, progress=False)
        
        if usdtry.empty:
            raise ValueError("Could not fetch USD/TRY data for approximation")
        
        # Extract close prices (handle both Series and DataFrame)
        if isinstance(usdtry['Close'], pd.DataFrame):
            usdtry_close = usdtry['Close'].iloc[:, 0]  # Get first column if DataFrame
        else:
            usdtry_close = usdtry['Close']
        
        # Calculate 1-year USD/TRY change (proxy for TRY weakness)
        usdtry_change = usdtry_close.pct_change(252)
        
        # Approximate deposit rate formula:
        # Base rate + premium based on TRY weakness
        # Historical calibration: 
        # - Base rate: 8% (minimum during stable periods)
        # - Each 10% TRY depreciation → ~3% rate increase
        
        base_rate = 0.08
        sensitivity = 0.30  # 30% of depreciation translates to rate increase
        
        approx_rate = base_rate + (usdtry_change * sensitivity).clip(lower=0)
        
        # Cap at reasonable levels (TCMB rarely goes above 50%)
        approx_rate = approx_rate.clip(upper=0.50)
        
        # Fill initial NaN values with base rate
        approx_rate = approx_rate.fillna(base_rate)
        
        # Create DataFrame
        rates_df = pd.DataFrame(index=approx_rate.index)
        rates_df['deposit_rate'] = approx_rate.values
        
        print(f"Approximated {len(rates_df)} days of deposit rates")
        print(f"Rate range: {rates_df['deposit_rate'].min():.2%} to {rates_df['deposit_rate'].max():.2%}")
        print("Note: These are approximations. For accurate rates, use EVDS API with api_key.")
        
        return rates_df
    
    def get_rate_for_date(self, date):
        """Get deposit rate for a specific date"""
        if self.rates_data is None:
            raise ValueError("No rates data loaded. Call fetch_rates() first.")
        
        # Find nearest date
        if date not in self.rates_data.index:
            # Get closest previous date
            valid_dates = self.rates_data.index[self.rates_data.index <= date]
            if len(valid_dates) == 0:
                return self.rates_data['deposit_rate'].iloc[0]
            date = valid_dates[-1]
        
        return self.rates_data.loc[date, 'deposit_rate']
    
    def get_rates_series(self, dates):
        """Get deposit rates for a series of dates"""
        if self.rates_data is None:
            raise ValueError("No rates data loaded. Call fetch_rates() first.")
        
        # Reindex to match dates, forward fill
        aligned_rates = self.rates_data.reindex(dates, method='ffill')
        return aligned_rates['deposit_rate']


if __name__ == "__main__":
    # Test the fetcher
    print("="*70)
    print("TCMB DEPOSIT RATE FETCHER TEST")
    print("="*70)
    
    # Initialize fetcher (without API key, will use approximation)
    fetcher = TCMBRateFetcher()
    
    # Fetch rates
    rates = fetcher.fetch_rates(start_date='2013-01-01', end_date='2026-01-23')
    
    print("\n" + "="*70)
    print("DEPOSIT RATE STATISTICS")
    print("="*70)
    print(f"\nPeriod: {rates.index[0]} to {rates.index[-1]}")
    print(f"Observations: {len(rates)}")
    print(f"\nRate Statistics:")
    print(f"  Mean: {rates['deposit_rate'].mean():.2%}")
    print(f"  Median: {rates['deposit_rate'].median():.2%}")
    print(f"  Min: {rates['deposit_rate'].min():.2%}")
    print(f"  Max: {rates['deposit_rate'].max():.2%}")
    print(f"  Std: {rates['deposit_rate'].std():.2%}")
    
    print("\n" + "="*70)
    print("SAMPLE RATES (Recent)")
    print("="*70)
    print(rates.tail(10))
    
    print("\n" + "="*70)
    print("USAGE INSTRUCTIONS")
    print("="*70)
    print("\nTo use actual TCMB rates:")
    print("1. Get API key from: https://evds2.tcmb.gov.tr/")
    print("2. Install evds: pip install evds")
    print("3. Use: fetcher = TCMBRateFetcher(api_key='YOUR_KEY')")
    print("\nWithout API key, approximation based on USD/TRY is used.")
    print("="*70)
