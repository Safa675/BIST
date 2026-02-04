#!/usr/bin/env python3
"""
Ensemble Model Debugging Script

Performs comprehensive checks on the ensemble model:
1. Data integrity checks
2. Model loading and state verification
3. Prediction consistency tests
4. Feature alignment checks
5. NaN/Inf handling
6. Edge case testing
7. Memory and performance profiling
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import traceback

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(SCRIPT_DIR))

from models.ensemble_regime import EnsembleRegimeModel


class EnsembleDebugger:
    """Comprehensive debugging for ensemble model"""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.passed = []
        
    def log_issue(self, test_name, message, severity='ERROR'):
        """Log an issue found during testing"""
        self.issues.append({
            'test': test_name,
            'message': message,
            'severity': severity
        })
        print(f"  ❌ {severity}: {message}")
        
    def log_warning(self, test_name, message):
        """Log a warning"""
        self.warnings.append({
            'test': test_name,
            'message': message
        })
        print(f"  ⚠️  WARNING: {message}")
        
    def log_pass(self, test_name, message):
        """Log a passed test"""
        self.passed.append({
            'test': test_name,
            'message': message
        })
        print(f"  ✓ {message}")
    
    def print_section(self, title):
        """Print section header"""
        print("\n" + "="*80)
        print(f" {title}")
        print("="*80)
    
    def test_data_loading(self):
        """Test 1: Data Loading and Integrity"""
        self.print_section("TEST 1: DATA LOADING AND INTEGRITY")
        
        try:
            # Load features
            features_file = OUTPUT_DIR / "all_features.csv"
            if not features_file.exists():
                self.log_issue("data_loading", "Features file not found", "CRITICAL")
                return None, None
            
            features = pd.read_csv(features_file, index_col=0, parse_dates=True)
            self.log_pass("data_loading", f"Features loaded: {len(features)} rows, {len(features.columns)} columns")
            
            # Check for duplicate indices
            if features.index.duplicated().any():
                n_dupes = features.index.duplicated().sum()
                self.log_issue("data_loading", f"Found {n_dupes} duplicate dates in features")
            else:
                self.log_pass("data_loading", "No duplicate dates in features")
            
            # Check for sorted index
            if not features.index.is_monotonic_increasing:
                self.log_warning("data_loading", "Features index is not sorted chronologically")
            else:
                self.log_pass("data_loading", "Features index is properly sorted")
            
            # Check for NaN columns
            nan_cols = features.columns[features.isna().all()].tolist()
            if nan_cols:
                self.log_warning("data_loading", f"Found {len(nan_cols)} completely empty columns: {nan_cols[:5]}")
            else:
                self.log_pass("data_loading", "No completely empty columns")
            
            # Check NaN percentage
            nan_pct = features.isna().sum().sum() / (len(features) * len(features.columns)) * 100
            if nan_pct > 10:
                self.log_warning("data_loading", f"High NaN percentage: {nan_pct:.1f}%")
            else:
                self.log_pass("data_loading", f"Acceptable NaN percentage: {nan_pct:.1f}%")
            
            # Check for infinite values
            inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
            if inf_count > 0:
                self.log_issue("data_loading", f"Found {inf_count} infinite values in features")
            else:
                self.log_pass("data_loading", "No infinite values in features")
            
            # Load regimes
            regimes_file = OUTPUT_DIR / "simplified_regimes.csv"
            if not regimes_file.exists():
                self.log_issue("data_loading", "Regimes file not found", "CRITICAL")
                return features, None
            
            regimes = pd.read_csv(regimes_file, index_col=0, parse_dates=True)['regime']
            self.log_pass("data_loading", f"Regimes loaded: {len(regimes)} rows")
            
            # Check regime values
            valid_regimes = {'Bull', 'Bear', 'Stress', 'Choppy', 'Recovery'}
            invalid_regimes = set(regimes.unique()) - valid_regimes
            if invalid_regimes:
                self.log_issue("data_loading", f"Invalid regime values found: {invalid_regimes}")
            else:
                self.log_pass("data_loading", "All regime values are valid")
            
            # Check alignment
            common_idx = features.index.intersection(regimes.index)
            if len(common_idx) < len(features) * 0.9:
                self.log_warning("data_loading", f"Poor alignment: only {len(common_idx)}/{len(features)} dates match")
            else:
                self.log_pass("data_loading", f"Good alignment: {len(common_idx)}/{len(features)} dates match")
            
            return features, regimes
            
        except Exception as e:
            self.log_issue("data_loading", f"Exception during data loading: {str(e)}", "CRITICAL")
            traceback.print_exc()
            return None, None
    
    def test_model_loading(self):
        """Test 2: Model Loading and State"""
        self.print_section("TEST 2: MODEL LOADING AND STATE")
        
        try:
            model_dir = OUTPUT_DIR / "ensemble_model"
            if not model_dir.exists():
                self.log_issue("model_loading", "Ensemble model directory not found", "CRITICAL")
                return None
            
            # Check required files
            required_files = ['ensemble_metadata.pkl', 'xgboost_model.pkl', 'lstm_model.pt', 'hmm_model.pkl']
            for file in required_files:
                if not (model_dir / file).exists():
                    self.log_warning("model_loading", f"Missing model file: {file}")
                else:
                    self.log_pass("model_loading", f"Found {file}")
            
            # Load ensemble
            ensemble = EnsembleRegimeModel.load(model_dir)
            self.log_pass("model_loading", "Ensemble model loaded successfully")
            
            # Check model state
            if not ensemble.is_trained:
                self.log_issue("model_loading", "Model is_trained flag is False", "CRITICAL")
            else:
                self.log_pass("model_loading", "Model is marked as trained")
            
            # Check available models
            if len(ensemble.available_models) == 0:
                self.log_issue("model_loading", "No models available in ensemble", "CRITICAL")
            else:
                self.log_pass("model_loading", f"Available models: {ensemble.available_models}")
            
            # Check weights
            weight_sum = sum(ensemble.weights[m] for m in ensemble.available_models)
            if abs(weight_sum - 1.0) > 0.01:
                self.log_warning("model_loading", f"Weights don't sum to 1.0: {weight_sum:.3f}")
            else:
                self.log_pass("model_loading", f"Weights properly normalized: {weight_sum:.3f}")
            
            # Check feature names
            if ensemble.feature_names is None:
                self.log_issue("model_loading", "Feature names not stored in ensemble")
            else:
                self.log_pass("model_loading", f"Feature names stored: {len(ensemble.feature_names)} features")
            
            # Check individual models
            for model_name in ensemble.available_models:
                model = getattr(ensemble, f"{model_name}_model", None)
                if model is None:
                    self.log_issue("model_loading", f"{model_name} model is None")
                else:
                    self.log_pass("model_loading", f"{model_name} model loaded")
                    
                    # Check XGBoost feature names
                    if model_name == 'xgboost' and hasattr(model, 'feature_names'):
                        if model.feature_names is None:
                            self.log_issue("model_loading", "XGBoost feature_names is None")
                        else:
                            self.log_pass("model_loading", f"XGBoost has {len(model.feature_names)} feature names")
                    
                    # Check LSTM feature names
                    if model_name == 'lstm' and hasattr(model, 'feature_names'):
                        if model.feature_names is None:
                            self.log_issue("model_loading", "LSTM feature_names is None")
                        else:
                            self.log_pass("model_loading", f"LSTM has {len(model.feature_names)} feature names")
            
            return ensemble
            
        except Exception as e:
            self.log_issue("model_loading", f"Exception during model loading: {str(e)}", "CRITICAL")
            traceback.print_exc()
            return None
    
    def test_prediction_consistency(self, ensemble, features, regimes):
        """Test 3: Prediction Consistency"""
        self.print_section("TEST 3: PREDICTION CONSISTENCY")
        
        if ensemble is None or features is None:
            self.log_issue("prediction_consistency", "Cannot test - model or data not loaded", "CRITICAL")
            return
        
        try:
            # Test on recent data
            recent_features = features.tail(100)
            recent_regimes = regimes.tail(100)
            
            # First prediction
            print("  Running first prediction...")
            results1 = ensemble.predict(recent_features, recent_regimes, return_details=False)
            self.log_pass("prediction_consistency", "First prediction completed")
            
            # Second prediction (should be identical)
            print("  Running second prediction...")
            results2 = ensemble.predict(recent_features, recent_regimes, return_details=False)
            self.log_pass("prediction_consistency", "Second prediction completed")
            
            # Check consistency
            pred_match = (results1['ensemble_prediction'] == results2['ensemble_prediction']).all()
            if not pred_match:
                n_diff = (~(results1['ensemble_prediction'] == results2['ensemble_prediction'])).sum()
                self.log_issue("prediction_consistency", f"Predictions differ between runs: {n_diff} differences")
            else:
                self.log_pass("prediction_consistency", "Predictions are consistent across runs")
            
            # Check confidence consistency
            conf_diff = (results1['ensemble_confidence'] - results2['ensemble_confidence']).abs().max()
            if conf_diff > 0.001:
                self.log_warning("prediction_consistency", f"Confidence scores differ: max diff = {conf_diff:.6f}")
            else:
                self.log_pass("prediction_consistency", "Confidence scores are consistent")
            
            # Check for NaN in predictions
            if results1['ensemble_prediction'].isna().any():
                n_nan = results1['ensemble_prediction'].isna().sum()
                self.log_issue("prediction_consistency", f"Found {n_nan} NaN predictions")
            else:
                self.log_pass("prediction_consistency", "No NaN predictions")
            
            # Check prediction distribution
            pred_dist = results1['ensemble_prediction'].value_counts()
            print(f"\n  Prediction distribution:")
            for regime, count in pred_dist.items():
                print(f"    {regime}: {count} ({count/len(results1)*100:.1f}%)")
            
            # Check if all predictions are same class (red flag)
            if len(pred_dist) == 1:
                self.log_warning("prediction_consistency", "All predictions are the same class!")
            else:
                self.log_pass("prediction_consistency", f"Predictions span {len(pred_dist)} different regimes")
            
            # Check confidence range
            conf_min = results1['ensemble_confidence'].min()
            conf_max = results1['ensemble_confidence'].max()
            conf_mean = results1['ensemble_confidence'].mean()
            print(f"\n  Confidence stats: min={conf_min:.2%}, max={conf_max:.2%}, mean={conf_mean:.2%}")
            
            if conf_min == conf_max:
                self.log_warning("prediction_consistency", "All confidence scores are identical")
            else:
                self.log_pass("prediction_consistency", f"Confidence varies: {conf_min:.2%} to {conf_max:.2%}")
            
            return results1
            
        except Exception as e:
            self.log_issue("prediction_consistency", f"Exception during prediction: {str(e)}", "CRITICAL")
            traceback.print_exc()
            return None
    
    def test_feature_alignment(self, ensemble, features):
        """Test 4: Feature Alignment"""
        self.print_section("TEST 4: FEATURE ALIGNMENT")
        
        if ensemble is None or features is None:
            self.log_issue("feature_alignment", "Cannot test - model or data not loaded", "CRITICAL")
            return
        
        try:
            # Check ensemble feature names
            if ensemble.feature_names is None:
                self.log_issue("feature_alignment", "Ensemble has no feature names stored")
                return
            
            ensemble_features = set(ensemble.feature_names)
            data_features = set(features.columns)
            
            # Check for missing features
            missing_in_data = ensemble_features - data_features
            if missing_in_data:
                self.log_warning("feature_alignment", f"Features in model but not in data: {missing_in_data}")
            else:
                self.log_pass("feature_alignment", "All model features present in data")
            
            # Check for extra features
            extra_in_data = data_features - ensemble_features
            if extra_in_data:
                self.log_pass("feature_alignment", f"Extra features in data (OK): {len(extra_in_data)} features")
            
            # Check XGBoost feature alignment
            if ensemble.xgboost_model and hasattr(ensemble.xgboost_model, 'feature_names'):
                if ensemble.xgboost_model.feature_names:
                    xgb_features = set(ensemble.xgboost_model.feature_names)
                    missing_xgb = xgb_features - data_features
                    if missing_xgb:
                        self.log_issue("feature_alignment", f"XGBoost missing features: {missing_xgb}")
                    else:
                        self.log_pass("feature_alignment", "XGBoost features aligned with data")
            
            # Check LSTM feature alignment
            if ensemble.lstm_model and hasattr(ensemble.lstm_model, 'feature_names'):
                if ensemble.lstm_model.feature_names:
                    lstm_features = set(ensemble.lstm_model.feature_names)
                    missing_lstm = lstm_features - data_features
                    if missing_lstm:
                        self.log_issue("feature_alignment", f"LSTM missing features: {missing_lstm}")
                    else:
                        self.log_pass("feature_alignment", "LSTM features aligned with data")
            
        except Exception as e:
            self.log_issue("feature_alignment", f"Exception during feature alignment check: {str(e)}")
            traceback.print_exc()
    
    def test_edge_cases(self, ensemble, features):
        """Test 5: Edge Cases"""
        self.print_section("TEST 5: EDGE CASE TESTING")
        
        if ensemble is None or features is None:
            self.log_issue("edge_cases", "Cannot test - model or data not loaded", "CRITICAL")
            return
        
        try:
            # Test 1: Single row prediction
            print("  Testing single row prediction...")
            try:
                single_row = features.tail(1)
                result = ensemble.predict(single_row, return_details=False)
                if len(result) == 1:
                    self.log_pass("edge_cases", "Single row prediction works")
                else:
                    self.log_warning("edge_cases", f"Single row returned {len(result)} results")
            except Exception as e:
                self.log_issue("edge_cases", f"Single row prediction failed: {str(e)}")
            
            # Test 2: Small batch
            print("  Testing small batch (5 rows)...")
            try:
                small_batch = features.tail(5)
                result = ensemble.predict(small_batch, return_details=False)
                if len(result) == 5:
                    self.log_pass("edge_cases", "Small batch prediction works")
                else:
                    self.log_warning("edge_cases", f"Small batch returned {len(result)} results")
            except Exception as e:
                self.log_issue("edge_cases", f"Small batch prediction failed: {str(e)}")
            
            # Test 3: Data with NaN
            print("  Testing data with NaN values...")
            try:
                nan_data = features.tail(10).copy()
                # Introduce some NaN
                nan_data.iloc[0, 0] = np.nan
                result = ensemble.predict(nan_data, return_details=False)
                self.log_pass("edge_cases", "Handles NaN in input data")
            except Exception as e:
                self.log_issue("edge_cases", f"Failed to handle NaN: {str(e)}")
            
            # Test 4: predict_current
            print("  Testing predict_current method...")
            try:
                current = ensemble.predict_current(features.tail(50))
                required_keys = ['prediction', 'confidence', 'disagreement', 'probabilities', 'model_agreement']
                missing_keys = [k for k in required_keys if k not in current]
                if missing_keys:
                    self.log_warning("edge_cases", f"predict_current missing keys: {missing_keys}")
                else:
                    self.log_pass("edge_cases", "predict_current returns all required keys")
                    print(f"    Current prediction: {current['prediction']} (confidence: {current['confidence']:.1%})")
            except Exception as e:
                self.log_issue("edge_cases", f"predict_current failed: {str(e)}")
            
        except Exception as e:
            self.log_issue("edge_cases", f"Exception during edge case testing: {str(e)}")
            traceback.print_exc()
    
    def test_memory_and_performance(self, ensemble, features):
        """Test 6: Memory and Performance"""
        self.print_section("TEST 6: MEMORY AND PERFORMANCE")
        
        if ensemble is None or features is None:
            self.log_issue("performance", "Cannot test - model or data not loaded", "CRITICAL")
            return
        
        try:
            import time
            
            # Test prediction speed
            test_data = features.tail(100)
            
            start = time.time()
            result = ensemble.predict(test_data, return_details=False)
            elapsed = time.time() - start
            
            per_sample = elapsed / len(test_data) * 1000  # ms per sample
            
            print(f"  Prediction time: {elapsed:.3f}s for {len(test_data)} samples")
            print(f"  Speed: {per_sample:.2f} ms/sample")
            
            if per_sample > 100:
                self.log_warning("performance", f"Slow prediction: {per_sample:.1f} ms/sample")
            else:
                self.log_pass("performance", f"Good prediction speed: {per_sample:.1f} ms/sample")
            
            # Test with larger batch
            if len(features) > 500:
                large_batch = features.tail(500)
                start = time.time()
                result = ensemble.predict(large_batch, return_details=False)
                elapsed = time.time() - start
                print(f"  Large batch (500): {elapsed:.3f}s ({elapsed/500*1000:.2f} ms/sample)")
            
        except Exception as e:
            self.log_issue("performance", f"Exception during performance testing: {str(e)}")
            traceback.print_exc()
    
    def generate_report(self):
        """Generate debugging report"""
        self.print_section("DEBUGGING SUMMARY")
        
        print(f"\n✓ PASSED: {len(self.passed)} tests")
        print(f"⚠️  WARNINGS: {len(self.warnings)} warnings")
        print(f"❌ ISSUES: {len(self.issues)} issues")
        
        if self.issues:
            print("\n" + "="*80)
            print("CRITICAL ISSUES FOUND:")
            print("="*80)
            for issue in self.issues:
                print(f"\n[{issue['severity']}] {issue['test']}")
                print(f"  {issue['message']}")
        
        if self.warnings:
            print("\n" + "="*80)
            print("WARNINGS:")
            print("="*80)
            for warning in self.warnings:
                print(f"\n[WARNING] {warning['test']}")
                print(f"  {warning['message']}")
        
        # Overall status
        print("\n" + "="*80)
        critical_issues = [i for i in self.issues if i['severity'] == 'CRITICAL']
        
        if critical_issues:
            print("STATUS: ❌ CRITICAL ISSUES FOUND - MODEL NOT READY")
            return False
        elif len(self.issues) > 0:
            print("STATUS: ⚠️  ISSUES FOUND - REVIEW RECOMMENDED")
            return True
        elif len(self.warnings) > 5:
            print("STATUS: ⚠️  MULTIPLE WARNINGS - REVIEW RECOMMENDED")
            return True
        else:
            print("STATUS: ✅ MODEL IS CLEAN AND READY TO USE")
            return True


def main():
    """Run all debugging tests"""
    print("="*80)
    print("ENSEMBLE MODEL DEBUGGING")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    debugger = EnsembleDebugger()
    
    # Run tests
    features, regimes = debugger.test_data_loading()
    ensemble = debugger.test_model_loading()
    results = debugger.test_prediction_consistency(ensemble, features, regimes)
    debugger.test_feature_alignment(ensemble, features)
    debugger.test_edge_cases(ensemble, features)
    debugger.test_memory_and_performance(ensemble, features)
    
    # Generate report
    is_clean = debugger.generate_report()
    
    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return is_clean


if __name__ == "__main__":
    is_clean = main()
    sys.exit(0 if is_clean else 1)
