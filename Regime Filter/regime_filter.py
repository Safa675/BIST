"""
Main Regime Filter orchestration class
Combines data loading, feature engineering, and regime classification
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from market_data import DataLoader, FeatureEngine
from regime_models import RegimeClassifier
import config

class RegimeFilter:
    """Main regime filter system for BIST market"""
    
    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir
        self.loader = DataLoader(data_dir)
        self.data = None
        self.features = None
        self.regimes = None
        self.classifier = None
        
    def load_data(self, fetch_usdtry=True, load_stocks=False):
        """Load all required data"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        self.data = self.loader.load_all(
            fetch_usdtry=fetch_usdtry,
            load_stocks=load_stocks
        )
        
        return self.data
    
    def calculate_features(self):
        """Calculate all features"""
        if self.data is None:
            raise ValueError("Load data first using load_data()")
        
        print("\n" + "="*60)
        print("CALCULATING FEATURES")
        print("="*60)
        
        engine = FeatureEngine(self.data)
        features_raw = engine.calculate_all_features()
        
        # CRITICAL: Shift features to eliminate look-ahead bias
        print("\nShifting features to eliminate look-ahead bias...")
        self.features = engine.shift_for_prediction(shift_days=1)
        print(f"  Features: {len(features_raw)} → {len(self.features)} rows")
        
        return self.features
    
    def classify_regimes(self):
        """Classify market regimes"""
        if self.features is None:
            raise ValueError("Calculate features first using calculate_features()")
        
        print("\n" + "="*60)
        print("CLASSIFYING REGIMES")
        print("="*60)
        
        # Features are already shifted in calculate_features()
        self.classifier = RegimeClassifier(self.features)
        self.regimes = self.classifier.classify_all()
        
        return self.regimes
    
    def run_full_pipeline(self, fetch_usdtry=True, load_stocks=False):
        """Run the complete regime filter pipeline"""
        print("\n" + "="*60)
        print("BIST REGIME FILTER SYSTEM")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load data
        self.load_data(fetch_usdtry=fetch_usdtry, load_stocks=load_stocks)
        
        # Calculate features
        self.calculate_features()
        
        # Classify regimes
        self.classify_regimes()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return self.regimes
    
    def get_current_regime(self):
        """Get current market regime"""
        if self.classifier is None:
            raise ValueError("Run classification first")
        
        return self.classifier.get_current_regime()
    
    def get_regime_summary(self):
        """Get summary of regime classifications"""
        if self.classifier is None:
            raise ValueError("Run classification first")
        
        return self.classifier.get_regime_summary()
    
    def get_regime_history(self, start_date=None, end_date=None):
        """Get regime history for a date range"""
        if self.regimes is None:
            raise ValueError("Run classification first")
        
        regimes = self.regimes.copy()
        
        if start_date:
            regimes = regimes[regimes.index >= start_date]
        if end_date:
            regimes = regimes[regimes.index <= end_date]
        
        return regimes
    
    def export_regimes(self, output_dir=None):
        """Export regime classifications to JSON and CSV"""
        if self.regimes is None:
            raise ValueError("Run classification first")
        
        if output_dir is None:
            output_dir = Path(config.OUTPUT_DIR)
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Export to JSON
        json_file = output_dir / config.REGIME_JSON
        regime_dict = {}
        
        for date, row in self.regimes.iterrows():
            regime_dict[date.strftime('%Y-%m-%d')] = {
                'volatility': row.get('volatility_regime', 'Unknown'),
                'trend': row.get('trend_regime', 'Unknown'),
                'trend_short': row.get('trend_short', 'Unknown'),
                'trend_long': row.get('trend_long', 'Unknown'),
                'risk': row.get('risk_regime', 'Unknown'),
                'liquidity': row.get('liquidity_regime', 'Unknown'),
                'label': row.get('regime_label', 'Unknown')
            }
        
        with open(json_file, 'w') as f:
            json.dump(regime_dict, f, indent=2)
        
        print(f"\nRegimes exported to: {json_file}")
        
        # Export features to CSV
        if self.features is not None:
            features_file = output_dir / config.FEATURES_CSV
            combined = pd.concat([self.features, self.regimes], axis=1)
            combined.to_csv(features_file)
            print(f"Features exported to: {features_file}")
        
        return json_file
    
    def print_summary(self):
        """Print a summary of the regime filter results"""
        if self.regimes is None:
            print("No regimes calculated yet. Run the pipeline first.")
            return
        
        print("\n" + "="*60)
        print("REGIME FILTER SUMMARY")
        print("="*60)
        
        # Date range
        print(f"\nDate range: {self.regimes.index[0].date()} to {self.regimes.index[-1].date()}")
        print(f"Total periods: {len(self.regimes)}")
        
        # Current regime
        current = self.get_current_regime()
        print(f"\n{'CURRENT REGIME':-^60}")
        print(f"Date: {current['date'].date()}")
        print(f"Volatility: {current['volatility']}")
        print(f"Trend (Short): {current['trend_short']}")
        print(f"Trend (Long): {current['trend_long']}")
        print(f"Trend (Combined): {current['trend']}")
        print(f"Risk: {current['risk']}")
        print(f"Liquidity: {current['liquidity']}")
        print(f"\nLabel: {current['label']}")
        
        # Regime distribution
        summary = self.get_regime_summary()
        
        print(f"\n{'REGIME DISTRIBUTION':-^60}")
        for regime_type, counts in summary.items():
            print(f"\n{regime_type.replace('_', ' ').title()}:")
            total = sum(counts.values())
            for state, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
                pct = (count / total) * 100
                print(f"  {state:15s}: {count:5d} ({pct:5.1f}%)")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    # Run the regime filter
    rf = RegimeFilter()
    
    # Run full pipeline
    regimes = rf.run_full_pipeline(fetch_usdtry=True, load_stocks=False)
    
    # Print summary
    rf.print_summary()
    
    # Export results
    rf.export_regimes()
    
    print("\nDone!")
