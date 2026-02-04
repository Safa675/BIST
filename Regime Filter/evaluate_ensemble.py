#!/usr/bin/env python3
"""
Evaluate Ensemble Model Performance
Compares ensemble predictions with random baseline on out-of-sample data
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

sys.path.insert(0, str(SCRIPT_DIR))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models.ensemble_regime import EnsembleRegimeModel


def calculate_random_baseline(y_true, n_simulations=1000):
    """
    Calculate random baseline performance
    
    Args:
        y_true: True labels
        n_simulations: Number of random simulations to average
    
    Returns:
        Dictionary with random baseline metrics
    """
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    
    # Get class distribution
    class_counts = pd.Series(y_true).value_counts(normalize=True)
    
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    
    for _ in range(n_simulations):
        # Random predictions (uniform distribution)
        random_preds = np.random.choice(unique_classes, size=len(y_true))
        
        acc = accuracy_score(y_true, random_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, random_preds, average='weighted', zero_division=0
        )
        
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    
    # Also calculate stratified random (matching class distribution)
    stratified_accuracies = []
    for _ in range(n_simulations):
        stratified_preds = np.random.choice(
            unique_classes, 
            size=len(y_true),
            p=class_counts.values
        )
        stratified_accuracies.append(accuracy_score(y_true, stratified_preds))
    
    return {
        'uniform': {
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s)
        },
        'stratified': {
            'accuracy': np.mean(stratified_accuracies),
            'accuracy_std': np.std(stratified_accuracies)
        },
        'n_classes': n_classes,
        'class_distribution': class_counts.to_dict()
    }


def evaluate_ensemble():
    """Evaluate ensemble model on out-of-sample data"""
    
    print("="*80)
    print("ENSEMBLE MODEL EVALUATION - OUT OF SAMPLE")
    print("="*80)
    print(f"Evaluation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("Loading data...")
    features = pd.read_csv(OUTPUT_DIR / "all_features.csv", index_col=0, parse_dates=True)
    simplified_regimes = pd.read_csv(
        OUTPUT_DIR / "simplified_regimes.csv", 
        index_col=0, 
        parse_dates=True
    )['regime']
    
    print(f"  Features: {len(features)} rows, {len(features.columns)} columns")
    print(f"  Regimes: {len(simplified_regimes)} rows")
    print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}\n")
    
    # Load ensemble model
    print("Loading ensemble model...")
    ensemble = EnsembleRegimeModel.load(OUTPUT_DIR / "ensemble_model")
    print(f"  Models: {ensemble.available_models}")
    print(f"  Weights: {ensemble.weights}")
    print(f"  Forecast horizon: {ensemble.forecast_horizon} days\n")
    
    # Define train/test split (same as training)
    train_end_date = '2023-12-31'
    test_start_date = '2024-01-01'
    
    print(f"Train period: {features.index[0].date()} to {train_end_date}")
    print(f"Test period: {test_start_date} to {features.index[-1].date()}\n")
    
    # Get test data
    test_features = features[features.index >= test_start_date]
    test_regimes = simplified_regimes[simplified_regimes.index >= test_start_date]
    
    print(f"Test samples: {len(test_features)}\n")
    
    # Get ensemble predictions
    print("Generating ensemble predictions...")
    results, details = ensemble.predict(test_features, test_regimes, return_details=True)
    
    # Prepare true labels (shifted by forecast horizon)
    y_true = test_regimes.shift(-ensemble.forecast_horizon)
    
    # Align predictions and true labels
    common_idx = results.index.intersection(y_true.index)
    results = results.loc[common_idx]
    y_true = y_true.loc[common_idx]
    
    # Remove NaN
    valid_mask = ~y_true.isna()
    y_true = y_true[valid_mask]
    y_pred = results.loc[valid_mask, 'ensemble_prediction']
    confidence = results.loc[valid_mask, 'ensemble_confidence']
    disagreement = results.loc[valid_mask, 'model_disagreement']
    
    print(f"Valid predictions: {len(y_true)}\n")
    
    # Calculate ensemble metrics
    print("="*80)
    print("ENSEMBLE MODEL PERFORMANCE")
    print("="*80)
    
    ensemble_acc = accuracy_score(y_true, y_pred)
    ensemble_prec, ensemble_rec, ensemble_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    print(f"Accuracy:  {ensemble_acc:.2%}")
    print(f"Precision: {ensemble_prec:.2%}")
    print(f"Recall:    {ensemble_rec:.2%}")
    print(f"F1-Score:  {ensemble_f1:.2%}")
    print(f"Avg Confidence: {confidence.mean():.2%}")
    print(f"Avg Disagreement: {disagreement.mean():.2%}\n")
    
    # Per-regime metrics
    regime_metrics = {}
    for regime in y_true.unique():
        mask = y_true == regime
        regime_acc = accuracy_score(y_true[mask], y_pred[mask])
        regime_metrics[regime] = {
            'accuracy': regime_acc,
            'samples': mask.sum(),
            'avg_confidence': confidence[mask].mean()
        }
    
    print("Per-Regime Accuracy:")
    for regime, metrics in sorted(regime_metrics.items()):
        print(f"  {regime:10s}: {metrics['accuracy']:5.1%} (n={metrics['samples']:3d}, conf={metrics['avg_confidence']:.1%})")
    print()
    
    # Individual model performance
    print("="*80)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*80)
    
    individual_metrics = {}
    
    for model_name, preds in details['model_predictions'].items():
        # Handle NaN in predictions
        valid_preds_mask = ~np.isnan(preds[valid_mask.values])
        
        if valid_preds_mask.sum() > 0:
            model_y_true = y_true.values[valid_preds_mask]
            model_y_pred = preds[valid_mask.values][valid_preds_mask].astype(int)
            
            # Map numeric to regime names
            model_y_pred_labels = [ensemble.INVERSE_MAPPING[p] for p in model_y_pred]
            
            model_acc = accuracy_score(model_y_true, model_y_pred_labels)
            model_prec, model_rec, model_f1, _ = precision_recall_fscore_support(
                model_y_true, model_y_pred_labels, average='weighted', zero_division=0
            )
            
            individual_metrics[model_name] = {
                'accuracy': model_acc,
                'precision': model_prec,
                'recall': model_rec,
                'f1': model_f1,
                'samples': valid_preds_mask.sum()
            }
            
            print(f"{model_name.upper()}:")
            print(f"  Accuracy:  {model_acc:.2%}")
            print(f"  Precision: {model_prec:.2%}")
            print(f"  Recall:    {model_rec:.2%}")
            print(f"  F1-Score:  {model_f1:.2%}")
            print(f"  Samples:   {valid_preds_mask.sum()}\n")
    
    # Random baseline
    print("="*80)
    print("RANDOM BASELINE COMPARISON")
    print("="*80)
    
    random_metrics = calculate_random_baseline(y_true.values, n_simulations=1000)
    
    print(f"Number of classes: {random_metrics['n_classes']}")
    print(f"\nClass distribution:")
    for regime, pct in sorted(random_metrics['class_distribution'].items()):
        print(f"  {regime:10s}: {pct:.1%}")
    
    print(f"\nUniform Random Baseline:")
    print(f"  Accuracy:  {random_metrics['uniform']['accuracy']:.2%} ± {random_metrics['uniform']['accuracy_std']:.2%}")
    print(f"  Precision: {random_metrics['uniform']['precision']:.2%}")
    print(f"  Recall:    {random_metrics['uniform']['recall']:.2%}")
    print(f"  F1-Score:  {random_metrics['uniform']['f1']:.2%}")
    
    print(f"\nStratified Random Baseline (matching class distribution):")
    print(f"  Accuracy:  {random_metrics['stratified']['accuracy']:.2%} ± {random_metrics['stratified']['accuracy_std']:.2%}")
    
    # Improvement over random
    print(f"\n{'='*80}")
    print("IMPROVEMENT OVER RANDOM")
    print("="*80)
    
    uniform_improvement = (ensemble_acc - random_metrics['uniform']['accuracy']) / random_metrics['uniform']['accuracy'] * 100
    stratified_improvement = (ensemble_acc - random_metrics['stratified']['accuracy']) / random_metrics['stratified']['accuracy'] * 100
    
    print(f"Ensemble vs Uniform Random:    {uniform_improvement:+.1f}% improvement")
    print(f"Ensemble vs Stratified Random: {stratified_improvement:+.1f}% improvement")
    
    # Statistical significance (simple z-test)
    z_score = (ensemble_acc - random_metrics['uniform']['accuracy']) / random_metrics['uniform']['accuracy_std']
    print(f"\nZ-score vs uniform random: {z_score:.2f}")
    if z_score > 2.58:
        print("  ✓ Statistically significant at p < 0.01")
    elif z_score > 1.96:
        print("  ✓ Statistically significant at p < 0.05")
    else:
        print("  ✗ Not statistically significant")
    
    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX")
    print("="*80)
    
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    cm_df = pd.DataFrame(
        cm, 
        index=sorted(y_true.unique()), 
        columns=sorted(y_true.unique())
    )
    print(cm_df)
    
    # Return all metrics for report generation
    return {
        'ensemble': {
            'accuracy': ensemble_acc,
            'precision': ensemble_prec,
            'recall': ensemble_rec,
            'f1': ensemble_f1,
            'avg_confidence': confidence.mean(),
            'avg_disagreement': disagreement.mean()
        },
        'individual': individual_metrics,
        'random': random_metrics,
        'regime_metrics': regime_metrics,
        'improvements': {
            'uniform': uniform_improvement,
            'stratified': stratified_improvement,
            'z_score': z_score
        },
        'confusion_matrix': cm_df,
        'test_period': {
            'start': test_start_date,
            'end': str(features.index[-1].date()),
            'samples': len(y_true)
        },
        'train_period': {
            'start': str(features.index[0].date()),
            'end': train_end_date
        }
    }


def generate_report(metrics):
    """Generate markdown report"""
    
    report = []
    report.append("# Ensemble Model Evaluation Report")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    report.append(f"The ensemble regime prediction model was evaluated on out-of-sample data from **{metrics['test_period']['start']}** to **{metrics['test_period']['end']}** ({metrics['test_period']['samples']} trading days).")
    report.append("")
    report.append(f"**Key Findings:**")
    report.append(f"- Ensemble accuracy: **{metrics['ensemble']['accuracy']:.1%}**")
    report.append(f"- Improvement over uniform random: **{metrics['improvements']['uniform']:+.1f}%**")
    report.append(f"- Improvement over stratified random: **{metrics['improvements']['stratified']:+.1f}%**")
    report.append(f"- Statistical significance: **Z-score = {metrics['improvements']['z_score']:.2f}**")
    report.append("")
    
    # Data Overview
    report.append("---")
    report.append("")
    report.append("## Data Overview")
    report.append("")
    report.append("### Training Period")
    report.append(f"- **Start:** {metrics['train_period']['start']}")
    report.append(f"- **End:** {metrics['train_period']['end']}")
    report.append("")
    report.append("### Test Period (Out-of-Sample)")
    report.append(f"- **Start:** {metrics['test_period']['start']}")
    report.append(f"- **End:** {metrics['test_period']['end']}")
    report.append(f"- **Samples:** {metrics['test_period']['samples']} trading days")
    report.append("")
    report.append("### Regime Distribution in Test Set")
    report.append("")
    report.append("| Regime | Percentage | Count |")
    report.append("|--------|-----------|-------|")
    for regime, pct in sorted(metrics['random']['class_distribution'].items()):
        count = int(pct * metrics['test_period']['samples'])
        report.append(f"| {regime} | {pct:.1%} | {count} |")
    report.append("")
    
    # Ensemble Performance
    report.append("---")
    report.append("")
    report.append("## Ensemble Model Performance")
    report.append("")
    report.append("### Overall Metrics")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Accuracy** | {metrics['ensemble']['accuracy']:.2%} |")
    report.append(f"| **Precision** | {metrics['ensemble']['precision']:.2%} |")
    report.append(f"| **Recall** | {metrics['ensemble']['recall']:.2%} |")
    report.append(f"| **F1-Score** | {metrics['ensemble']['f1']:.2%} |")
    report.append(f"| **Avg Confidence** | {metrics['ensemble']['avg_confidence']:.2%} |")
    report.append(f"| **Avg Disagreement** | {metrics['ensemble']['avg_disagreement']:.2%} |")
    report.append("")
    
    # Per-regime performance
    report.append("### Per-Regime Accuracy")
    report.append("")
    report.append("| Regime | Accuracy | Samples | Avg Confidence |")
    report.append("|--------|----------|---------|----------------|")
    for regime, rm in sorted(metrics['regime_metrics'].items()):
        report.append(f"| {regime} | {rm['accuracy']:.1%} | {rm['samples']} | {rm['avg_confidence']:.1%} |")
    report.append("")
    
    # Individual models
    report.append("---")
    report.append("")
    report.append("## Individual Model Performance")
    report.append("")
    report.append("| Model | Accuracy | Precision | Recall | F1-Score | Samples |")
    report.append("|-------|----------|-----------|--------|----------|---------|")
    for model_name, im in sorted(metrics['individual'].items()):
        report.append(f"| {model_name.upper()} | {im['accuracy']:.2%} | {im['precision']:.2%} | {im['recall']:.2%} | {im['f1']:.2%} | {im['samples']} |")
    report.append("")
    
    # Random baseline
    report.append("---")
    report.append("")
    report.append("## Random Baseline Comparison")
    report.append("")
    report.append("### Uniform Random Baseline")
    report.append("")
    report.append("Random predictions with equal probability for each regime class.")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Accuracy** | {metrics['random']['uniform']['accuracy']:.2%} ± {metrics['random']['uniform']['accuracy_std']:.2%} |")
    report.append(f"| **Precision** | {metrics['random']['uniform']['precision']:.2%} |")
    report.append(f"| **Recall** | {metrics['random']['uniform']['recall']:.2%} |")
    report.append(f"| **F1-Score** | {metrics['random']['uniform']['f1']:.2%} |")
    report.append("")
    report.append("### Stratified Random Baseline")
    report.append("")
    report.append("Random predictions matching the actual class distribution in the test set.")
    report.append("")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| **Accuracy** | {metrics['random']['stratified']['accuracy']:.2%} ± {metrics['random']['stratified']['accuracy_std']:.2%} |")
    report.append("")
    
    # Improvement analysis
    report.append("---")
    report.append("")
    report.append("## Improvement Over Random")
    report.append("")
    report.append("| Comparison | Ensemble Accuracy | Random Accuracy | Improvement |")
    report.append("|------------|-------------------|-----------------|-------------|")
    report.append(f"| vs Uniform Random | {metrics['ensemble']['accuracy']:.2%} | {metrics['random']['uniform']['accuracy']:.2%} | **{metrics['improvements']['uniform']:+.1f}%** |")
    report.append(f"| vs Stratified Random | {metrics['ensemble']['accuracy']:.2%} | {metrics['random']['stratified']['accuracy']:.2%} | **{metrics['improvements']['stratified']:+.1f}%** |")
    report.append("")
    
    # Statistical significance
    report.append("### Statistical Significance")
    report.append("")
    report.append(f"**Z-score:** {metrics['improvements']['z_score']:.2f}")
    report.append("")
    if metrics['improvements']['z_score'] > 2.58:
        report.append("> [!NOTE]")
        report.append("> The ensemble model's performance is **statistically significant at p < 0.01** compared to uniform random baseline.")
    elif metrics['improvements']['z_score'] > 1.96:
        report.append("> [!NOTE]")
        report.append("> The ensemble model's performance is **statistically significant at p < 0.05** compared to uniform random baseline.")
    else:
        report.append("> [!WARNING]")
        report.append("> The ensemble model's performance is **not statistically significant** compared to uniform random baseline.")
    report.append("")
    
    # Confusion matrix
    report.append("---")
    report.append("")
    report.append("## Confusion Matrix")
    report.append("")
    report.append("Rows represent true regimes, columns represent predicted regimes.")
    report.append("")
    
    # Convert confusion matrix to markdown table
    cm = metrics['confusion_matrix']
    header = "| True \\ Predicted | " + " | ".join(cm.columns) + " |"
    separator = "|" + "|".join(["---"] * (len(cm.columns) + 1)) + "|"
    report.append(header)
    report.append(separator)
    for idx, row in cm.iterrows():
        row_str = f"| **{idx}** | " + " | ".join([str(int(v)) for v in row.values]) + " |"
        report.append(row_str)
    report.append("")
    
    # Interpretation
    report.append("---")
    report.append("")
    report.append("## Interpretation")
    report.append("")
    
    # Determine if model is better than random
    if metrics['improvements']['uniform'] > 50 and metrics['improvements']['z_score'] > 2.58:
        report.append("> [!IMPORTANT]")
        report.append("> The ensemble model demonstrates **strong predictive power**, significantly outperforming random baselines with high statistical confidence.")
    elif metrics['improvements']['uniform'] > 20 and metrics['improvements']['z_score'] > 1.96:
        report.append("> [!NOTE]")
        report.append("> The ensemble model shows **moderate predictive power**, performing better than random with statistical significance.")
    elif metrics['improvements']['uniform'] > 0:
        report.append("> [!CAUTION]")
        report.append("> The ensemble model shows **marginal improvement** over random baseline. Consider model refinement or feature engineering.")
    else:
        report.append("> [!WARNING]")
        report.append("> The ensemble model **does not outperform random baseline**. Significant model improvements are needed.")
    report.append("")
    
    # Key insights
    report.append("### Key Insights")
    report.append("")
    
    # Find best and worst performing regimes
    best_regime = max(metrics['regime_metrics'].items(), key=lambda x: x[1]['accuracy'])
    worst_regime = min(metrics['regime_metrics'].items(), key=lambda x: x[1]['accuracy'])
    
    report.append(f"1. **Best Predicted Regime:** {best_regime[0]} ({best_regime[1]['accuracy']:.1%} accuracy)")
    report.append(f"2. **Worst Predicted Regime:** {worst_regime[0]} ({worst_regime[1]['accuracy']:.1%} accuracy)")
    
    # Find best individual model
    best_model = max(metrics['individual'].items(), key=lambda x: x[1]['accuracy'])
    report.append(f"3. **Best Individual Model:** {best_model[0].upper()} ({best_model[1]['accuracy']:.1%} accuracy)")
    
    # Confidence analysis
    if metrics['ensemble']['avg_confidence'] > 0.7:
        report.append(f"4. **High Confidence:** Average confidence of {metrics['ensemble']['avg_confidence']:.1%} suggests the model is confident in its predictions")
    else:
        report.append(f"4. **Low Confidence:** Average confidence of {metrics['ensemble']['avg_confidence']:.1%} suggests uncertainty in predictions")
    
    # Disagreement analysis
    if metrics['ensemble']['avg_disagreement'] > 0.5:
        report.append(f"5. **High Model Disagreement:** {metrics['ensemble']['avg_disagreement']:.1%} disagreement indicates models often predict different regimes")
    else:
        report.append(f"5. **Model Consensus:** {metrics['ensemble']['avg_disagreement']:.1%} disagreement shows good agreement between models")
    
    report.append("")
    
    # Recommendations
    report.append("---")
    report.append("")
    report.append("## Recommendations")
    report.append("")
    
    if metrics['ensemble']['accuracy'] < 0.5:
        report.append("1. **Model Performance:** Consider retraining with different features or hyperparameters")
        report.append("2. **Feature Engineering:** Add more predictive features or transform existing ones")
        report.append("3. **Class Imbalance:** Address regime imbalance through resampling or class weights")
    elif metrics['ensemble']['accuracy'] < 0.7:
        report.append("1. **Feature Selection:** Analyze feature importance and remove noisy features")
        report.append("2. **Hyperparameter Tuning:** Optimize model parameters for better performance")
        report.append("3. **Ensemble Weights:** Consider dynamic weight adjustment based on recent performance")
    else:
        report.append("1. **Production Ready:** Model shows strong performance and can be used for regime prediction")
        report.append("2. **Monitoring:** Implement ongoing performance monitoring to detect degradation")
        report.append("3. **Refinement:** Continue to improve by analyzing misclassified cases")
    
    report.append("")
    report.append("---")
    report.append("")
    report.append("*Report generated by Ensemble Model Evaluation Script*")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate_ensemble()
    
    # Generate report
    print("\n" + "="*80)
    print("GENERATING MARKDOWN REPORT")
    print("="*80)
    
    report_content = generate_report(metrics)
    report_file = OUTPUT_DIR / "ensemble_evaluation_report.md"
    
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    print(f"\nReport saved to: {report_file}")
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
