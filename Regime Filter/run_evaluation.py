"""
Evaluate the Simple Regime Filter against baselines.
Runs automatically — no plots, just results.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from simple_regime import DataLoader, SimpleRegimeClassifier, SimpleBacktester, CONFIG


def evaluate():
    print("=" * 70)
    print("SIMPLE REGIME FILTER — FULL EVALUATION")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    loader = DataLoader()
    xu100 = loader.load_xu100()
    prices = xu100['Close']
    returns = prices.pct_change()

    # ===================================================================
    # 1. Simple Regime Filter (our new model)
    # ===================================================================
    print("\n" + "=" * 70)
    print("1. SIMPLE REGIME FILTER (200 MA + Vol)")
    print("=" * 70)

    classifier = SimpleRegimeClassifier()
    regimes_simple = classifier.classify(prices)

    dist = classifier.get_distribution()
    for regime, row in dist.iterrows():
        print(f"  {regime:10s}: {row['Count']:5.0f} ({row['Percent']:5.1f}%)")

    print(f"  Transitions: {classifier.get_transitions()}")
    avg_dur = len(regimes_simple) / max(classifier.get_transitions(), 1)
    print(f"  Avg duration: {avg_dur:.1f} days")

    backtester_simple = SimpleBacktester(prices, regimes_simple)
    results_simple = backtester_simple.run()
    quality_simple = backtester_simple.regime_quality()

    # ===================================================================
    # 2. Pure 200 MA (binary: above=100%, below=0%)
    # ===================================================================
    print("\n" + "=" * 70)
    print("2. SIMPLE 200 MA (Pure Trend)")
    print("=" * 70)

    ma_200 = prices.rolling(200).mean()
    regime_200ma = pd.Series('Bull', index=prices.index)
    regime_200ma[prices < ma_200] = 'Bear'
    regime_200ma = regime_200ma.loc[ma_200.dropna().index]

    transitions_200 = (regime_200ma != regime_200ma.shift(1)).sum()
    print(f"  Transitions: {transitions_200}")
    print(f"  Avg duration: {len(regime_200ma) / max(transitions_200, 1):.1f} days")

    alloc_200 = {'Bull': 1.0, 'Bear': 0.0}
    backtester_200 = SimpleBacktester(prices, regime_200ma, alloc_200)
    results_200 = backtester_200.run()

    # ===================================================================
    # 3. Simple 50 MA
    # ===================================================================
    print("\n" + "=" * 70)
    print("3. SIMPLE 50 MA")
    print("=" * 70)

    ma_50 = prices.rolling(50).mean()
    regime_50ma = pd.Series('Bull', index=prices.index)
    regime_50ma[prices < ma_50] = 'Bear'
    regime_50ma = regime_50ma.loc[ma_50.dropna().index]

    transitions_50 = (regime_50ma != regime_50ma.shift(1)).sum()
    print(f"  Transitions: {transitions_50}")
    print(f"  Avg duration: {len(regime_50ma) / max(transitions_50, 1):.1f} days")

    alloc_50 = {'Bull': 1.0, 'Bear': 0.0}
    backtester_50 = SimpleBacktester(prices, regime_50ma, alloc_50)
    results_50 = backtester_50.run()

    # ===================================================================
    # 4. Buy & Hold
    # ===================================================================
    results_bh = backtester_simple.run_buy_and_hold()

    # ===================================================================
    # 5. Load old sophisticated filter for comparison (if available)
    # ===================================================================
    results_old = None
    try:
        import sys
        sys.path.insert(0, str(loader.data_dir.parent / "Regime Filter"))
        from market_data import FeatureEngine
        from regime_models import RegimeClassifier, SimplifiedRegimeClassifier as OldSimplified

        data_merged = pd.DataFrame({
            'XU100_Close': xu100['Close'],
            'XU100_Open': xu100.get('Open', xu100['Close']),
            'XU100_High': xu100.get('High', xu100['Close']),
            'XU100_Low': xu100.get('Low', xu100['Close']),
            'XU100_Volume': xu100.get('Volume', pd.Series(0, index=xu100.index)),
        })

        # Try to add USDTRY
        try:
            import yfinance as yf
            usdtry = yf.download('TRY=X', start=xu100.index[0] - pd.Timedelta(days=365),
                                 end=xu100.index[-1], progress=False)
            if not usdtry.empty:
                usdtry_close = usdtry['Close'] if 'Close' in usdtry.columns else usdtry
                if isinstance(usdtry_close, pd.DataFrame):
                    usdtry_close = usdtry_close.iloc[:, 0]
                data_merged['USDTRY'] = usdtry_close.reindex(data_merged.index).ffill()
        except Exception:
            pass

        engine = FeatureEngine(data_merged)
        features_raw = engine.calculate_all_features()
        features = engine.shift_for_prediction(shift_days=1)

        old_classifier = RegimeClassifier(features)
        detailed = old_classifier.classify_all()
        old_simple = OldSimplified()
        old_regimes = old_simple.classify(detailed)['simplified_regime']

        alloc_old = {'Bull': 1.0, 'Bear': 0.2, 'Stress': 0.0, 'Choppy': 0.5, 'Recovery': 0.8}
        common = prices.index.intersection(old_regimes.index)
        backtester_old = SimpleBacktester(prices.loc[common], old_regimes.loc[common], alloc_old)
        results_old = backtester_old.run()
        quality_old = backtester_old.regime_quality()

        print("\n" + "=" * 70)
        print("5. OLD SOPHISTICATED FILTER (for comparison)")
        print("=" * 70)
        old_transitions = (old_regimes != old_regimes.shift(1)).sum()
        print(f"  Transitions: {old_transitions}")
        print(f"  Avg duration: {len(old_regimes) / max(old_transitions, 1):.1f} days")
    except Exception as e:
        print(f"\n(Could not load old filter for comparison: {e})")

    # ===================================================================
    # RESULTS COMPARISON
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    strategies = [
        ('Simple Regime (NEW)', results_simple),
        ('Pure 200 MA', results_200),
        ('Pure 50 MA', results_50),
        ('Buy & Hold', results_bh),
    ]
    if results_old is not None:
        strategies.append(('Old Sophisticated', results_old))

    # Sort by Sharpe
    strategies.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    print(f"\n{'Strategy':<25} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>10} {'Vol':>8}")
    print("-" * 63)
    for name, r in strategies:
        marker = " ←" if name == 'Simple Regime (NEW)' else ""
        print(f"{name:<25} {r['annual_return']:>9.1%} "
              f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>9.1%} "
              f"{r['annual_volatility']:>7.1%}{marker}")

    # ===================================================================
    # REGIME QUALITY ANALYSIS
    # ===================================================================
    print("\n\n" + "=" * 70)
    print("REGIME QUALITY — Does it separate bull from bear?")
    print("=" * 70)

    print(f"\nSimple Regime Filter (NEW):")
    bull_ret = quality_simple.get('Bull', {}).get('avg_annual_return', 0)
    bear_ret = quality_simple.get('Bear', {}).get('avg_annual_return', 0)
    recovery_ret = quality_simple.get('Recovery', {}).get('avg_annual_return', 0)
    stress_ret = quality_simple.get('Stress', {}).get('avg_annual_return', 0)
    separation = bull_ret - bear_ret

    print(f"  Bull     avg return: {bull_ret:>7.1%}  ({quality_simple.get('Bull', {}).get('pct', 0):>5.1f}% of time)")
    print(f"  Recovery avg return: {recovery_ret:>7.1%}  ({quality_simple.get('Recovery', {}).get('pct', 0):>5.1f}% of time)")
    print(f"  Bear     avg return: {bear_ret:>7.1%}  ({quality_simple.get('Bear', {}).get('pct', 0):>5.1f}% of time)")
    print(f"  Stress   avg return: {stress_ret:>7.1%}  ({quality_simple.get('Stress', {}).get('pct', 0):>5.1f}% of time)")
    print(f"\n  Bull-Bear separation: {separation:>7.1%}", end="")
    if separation > 0.50:
        print("  ✅ EXCELLENT")
    elif separation > 0.30:
        print("  ✅ GOOD")
    elif separation > 0.10:
        print("  ⚠️  WEAK")
    else:
        print("  ❌ BROKEN")

    if results_old is not None and quality_old:
        print(f"\nOld Sophisticated Filter (for reference):")
        old_bull = quality_old.get('Bull', {}).get('avg_annual_return', 0)
        old_bear = quality_old.get('Bear', {}).get('avg_annual_return', 0)
        old_sep = old_bull - old_bear
        print(f"  Bull avg return: {old_bull:>7.1%}")
        print(f"  Bear avg return: {old_bear:>7.1%}")
        print(f"  Separation:      {old_sep:>7.1%}", end="")
        if old_sep > 0.30:
            print("  ✅")
        elif old_sep > 0.10:
            print("  ⚠️")
        else:
            print("  ❌")

    # ===================================================================
    # HEAD-TO-HEAD: New vs Old
    # ===================================================================
    if results_old is not None:
        print("\n\n" + "=" * 70)
        print("HEAD-TO-HEAD: New Simple vs Old Sophisticated")
        print("=" * 70)

        metrics = [
            ('Annual Return', 'annual_return', True),
            ('Sharpe Ratio', 'sharpe_ratio', True),
            ('Max Drawdown', 'max_drawdown', False),
            ('Volatility', 'annual_volatility', False),
        ]

        for label, key, higher_better in metrics:
            new_val = results_simple[key]
            old_val = results_old[key]

            if key in ('annual_return', 'max_drawdown', 'annual_volatility'):
                new_str = f"{new_val:>8.1%}"
                old_str = f"{old_val:>8.1%}"
            else:
                new_str = f"{new_val:>8.2f}"
                old_str = f"{old_val:>8.2f}"

            if higher_better:
                winner = "NEW ✅" if new_val > old_val else "OLD"
            else:
                winner = "NEW ✅" if new_val > old_val else "OLD"
                # For drawdown, less negative = better
                if key == 'max_drawdown':
                    winner = "NEW ✅" if new_val > old_val else "OLD"
                # For vol, lower = better
                if key == 'annual_volatility':
                    winner = "NEW ✅" if new_val < old_val else "OLD"

            print(f"  {label:<18} New: {new_str}  Old: {old_str}  → {winner}")

        # Improvement summary
        ret_improve = results_simple['annual_return'] - results_old['annual_return']
        sharpe_improve = results_simple['sharpe_ratio'] - results_old['sharpe_ratio']
        print(f"\n  Return improvement:  {ret_improve:+.1%}")
        print(f"  Sharpe improvement:  {sharpe_improve:+.2f}")

    # ===================================================================
    # CURRENT REGIME
    # ===================================================================
    current = classifier.get_current_regime()
    print("\n\n" + "=" * 70)
    print(f"CURRENT REGIME ({current['date'].date()})")
    print("=" * 70)
    print(f"  Regime:      {current['regime']}")
    print(f"  Allocation:  {current['allocation']:.0%} stocks, {1-current['allocation']:.0%} gold")
    print(f"  Above 200MA: {'Yes' if current['above_ma'] else 'No'}")
    print(f"  Vol %ile:    {current['vol_percentile']:.0f}th percentile")
    print(f"  Vol (20d):   {current['realized_vol']:.1%} annualized")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return {
        'simple': results_simple,
        'ma_200': results_200,
        'ma_50': results_50,
        'buy_hold': results_bh,
        'old': results_old,
        'quality': quality_simple,
    }


if __name__ == "__main__":
    evaluate()
