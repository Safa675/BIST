"""
Strategies Module
Consolidates trading strategies and allocation logic.

Includes:
1. DynamicAllocator (Adaptive position sizing)
2. ThreeTierStrategy (Simplified 3-tier sizing)
"""

import pandas as pd
import numpy as np

class DynamicAllocator:
    """
    Calculate optimal position sizes dynamically
    
    Methods:
    1. Volatility Targeting: Scale to maintain constant risk
    2. Kelly Criterion: Optimal bet sizing based on edge
    3. Confidence-Based: Scale by prediction confidence
    """
    
    def __init__(self, target_volatility=0.15, max_leverage=2.0):
        self.target_volatility = target_volatility
        self.max_leverage = max_leverage
    
    def volatility_targeting(self, current_volatility):
        if current_volatility == 0 or np.isnan(current_volatility):
            return 0.0
        position_size = self.target_volatility / current_volatility
        return np.clip(position_size, 0, self.max_leverage)
    
    def kelly_criterion(self, win_rate, avg_win, avg_loss):
        if avg_loss == 0 or np.isnan(avg_loss): return 0.0
        edge = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        if edge <= 0: return 0.0
        kelly = edge / avg_win
        return np.clip(kelly * 0.5, 0, 1.0) # Half-Kelly
    
    def confidence_based_allocation(self, predicted_regime, confidence, base_allocations=None):
        if base_allocations is None:
            base_allocations = {'Bull': 1.5, 'Recovery': 1.0, 'Choppy': 0.5, 'Bear': 0.2, 'Stress': 0.0}
        base = base_allocations.get(predicted_regime, 0.5)
        confidence_multiplier = 0.5 + (confidence * 0.5)
        return base * confidence_multiplier
    
    def combined_allocation(self, predicted_regime, confidence, current_volatility, regime_stats=None):
        vol_allocation = self.volatility_targeting(current_volatility)
        conf_allocation = self.confidence_based_allocation(predicted_regime, confidence)
        
        if regime_stats and predicted_regime in regime_stats:
            stats = regime_stats[predicted_regime]
            kelly_allocation = self.kelly_criterion(stats['win_rate'], stats['avg_win'], stats['avg_loss'])
        else:
            kelly_allocation = 0.5

        final_allocation = min(vol_allocation, conf_allocation, kelly_allocation * 2)
        final_allocation = np.clip(final_allocation, 0, self.max_leverage)
        
        return final_allocation, {
            'volatility_target': vol_allocation,
            'confidence_based': conf_allocation,
            'kelly_based': kelly_allocation,
            'final': final_allocation
        }


class ThreeTierStrategy:
    """
    Three-tier position sizing based on simplified regime classification
    Tiers: Aggressive (Bull), Neutral (Bear/Choppy/Recovery), Defensive (Stress)
    """
    
    def __init__(self, aggressive_weight=1.0, neutral_weight=0.5, defensive_weight=0.0):
        self.aggressive_weight = aggressive_weight
        self.neutral_weight = neutral_weight
        self.defensive_weight = defensive_weight
        self.tier_mapping = {
            'Bull': 'Aggressive',
            'Bear': 'Neutral', 'Choppy': 'Neutral', 'Recovery': 'Neutral',
            'Stress': 'Defensive'
        }
        self.weight_mapping = {
            'Aggressive': aggressive_weight,
            'Neutral': neutral_weight,
            'Defensive': defensive_weight
        }
    
    def get_position_size(self, regime):
        tier = self.tier_mapping.get(regime, 'Neutral')
        return self.weight_mapping[tier]
    
    def apply_strategy(self, regimes):
        regime_series = regimes['simplified_regime'] if isinstance(regimes, pd.DataFrame) else regimes
        return regime_series.map(lambda r: self.get_position_size(r))
    
    def backtest(self, regimes, returns):
        positions = self.apply_strategy(regimes)
        strategy_returns = positions.shift(1) * returns
        strategy_returns = strategy_returns.dropna()
        
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        cumulative = (1 + strategy_returns).cumprod()
        drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
        max_drawdown = drawdown.min()
        
        regime_series = regimes['simplified_regime'] if isinstance(regimes, pd.DataFrame) else regimes
        regime_exposure = {}
        for regime in ['Bull', 'Bear', 'Choppy', 'Recovery', 'Stress']:
            mask = regime_series == regime
            regime_exposure[regime] = mask.sum() / len(regime_series) if len(regime_series) > 0 else 0
            
        return {
            'total_return': total_return, 'annual_return': annual_return,
            'annual_volatility': annual_vol, 'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown, 'regime_exposure': regime_exposure,
            'avg_position': positions.mean()
        }


def create_dynamic_strategy(prices, predicted_regimes, confidence, volatility, allocator=None):
    """Create dynamic allocation strategy based on predictions"""
    if allocator is None: allocator = DynamicAllocator()
    
    allocations = pd.Series(index=prices.index, dtype=float)
    for date in prices.index:
        if date not in predicted_regimes.index or date not in volatility.index:
            allocations[date] = 0.0; continue
        
        regime = predicted_regimes[date]
        conf = confidence[date]
        vol = volatility[date]
        allocation, _ = allocator.combined_allocation(regime, conf, vol)
        allocations[date] = allocation
        
    market_returns = prices.pct_change()
    strategy_returns = market_returns * allocations.shift(1)
    return allocations, strategy_returns
