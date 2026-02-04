"""
Dashboard and visualization module for regime filter
Creates interactive Plotly visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import config

class RegimeDashboard:
    """Create interactive dashboard for regime analysis"""
    
    def __init__(self, data, features, regimes):
        """
        Args:
            data: Raw market data
            features: Calculated features
            regimes: Regime classifications
        """
        self.data = data
        self.features = features
        self.regimes = regimes
        
    def create_dashboard(self, output_file=None):
        """Create comprehensive dashboard"""
        print("Creating dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=5, cols=1,
            subplot_titles=(
                'XU100 Index Price & Regime Timeline',
                'Volatility Regime',
                'Trend & Momentum',
                'Risk-On/Off (USD/TRY)',
                'Liquidity Metrics'
            ),
            vertical_spacing=0.08,
            row_heights=[0.25, 0.2, 0.2, 0.2, 0.15]
        )
        
        # 1. Price chart with regime background
        self._add_price_chart(fig, row=1)
        
        # 2. Volatility chart
        self._add_volatility_chart(fig, row=2)
        
        # 3. Trend chart
        self._add_trend_chart(fig, row=3)
        
        # 4. Risk chart
        self._add_risk_chart(fig, row=4)
        
        # 5. Liquidity chart
        self._add_liquidity_chart(fig, row=5)
        
        # Update layout
        fig.update_layout(
            height=1800,
            title_text="BIST Regime Filter Dashboard",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Save to HTML
        if output_file is None:
            output_dir = Path(config.OUTPUT_DIR)
            output_dir.mkdir(exist_ok=True, parents=True)
            output_file = output_dir / config.DASHBOARD_FILE
        
        fig.write_html(output_file)
        print(f"Dashboard saved to: {output_file}")
        
        return fig
    
    def _add_price_chart(self, fig, row):
        """Add price chart with regime coloring"""
        # Price line
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['XU100_Close'],
                name='XU100',
                line=dict(color='black', width=1.5)
            ),
            row=row, col=1
        )
        
        # Add regime background colors
        self._add_regime_backgrounds(fig, row)
        
    def _add_regime_backgrounds(self, fig, row):
        """Add colored backgrounds for different regimes"""
        if 'volatility_regime' not in self.regimes.columns:
            return
        
        # Color map for volatility regimes
        color_map = {
            'Low': 'rgba(0, 255, 0, 0.1)',      # Green
            'Mid': 'rgba(255, 255, 0, 0.1)',    # Yellow
            'High': 'rgba(255, 165, 0, 0.1)',   # Orange
            'Stress': 'rgba(255, 0, 0, 0.2)'    # Red
        }
        
        # Find regime changes
        regime_changes = self.regimes['volatility_regime'].ne(self.regimes['volatility_regime'].shift())
        change_dates = self.regimes.index[regime_changes].tolist()
        
        if len(change_dates) == 0:
            return
        
        # Add shapes for each regime period
        for i in range(len(change_dates)):
            start_date = change_dates[i]
            end_date = change_dates[i+1] if i < len(change_dates)-1 else self.regimes.index[-1]
            regime = self.regimes.loc[start_date, 'volatility_regime']
            
            if regime in color_map:
                fig.add_vrect(
                    x0=start_date, x1=end_date,
                    fillcolor=color_map[regime],
                    layer="below",
                    line_width=0,
                    row=row, col=1
                )
    
    def _add_volatility_chart(self, fig, row):
        """Add volatility metrics"""
        # Realized volatility
        if 'realized_vol_20d' in self.features.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.features.index,
                    y=self.features['realized_vol_20d'] * 100,
                    name='Realized Vol (20d)',
                    line=dict(color='blue')
                ),
                row=row, col=1
            )
        
        if 'realized_vol_60d' in self.features.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.features.index,
                    y=self.features['realized_vol_60d'] * 100,
                    name='Realized Vol (60d)',
                    line=dict(color='lightblue', dash='dash')
                ),
                row=row, col=1
            )
        
        fig.update_yaxes(title_text="Volatility (%)", row=row, col=1)
    
    def _add_trend_chart(self, fig, row):
        """Add trend and momentum indicators"""
        # Price vs MA
        if 'price_to_ma_50' in self.features.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.features.index,
                    y=self.features['price_to_ma_50'] * 100,
                    name='Price vs MA50 (%)',
                    line=dict(color='green')
                ),
                row=row, col=1
            )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)
        
        fig.update_yaxes(title_text="Trend (%)", row=row, col=1)
    
    def _add_risk_chart(self, fig, row):
        """Add risk-on/off indicators (USD/TRY)"""
        if 'usdtry_momentum_20d' in self.features.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.features.index,
                    y=self.features['usdtry_momentum_20d'] * 100,
                    name='USD/TRY Momentum (20d)',
                    line=dict(color='red')
                ),
                row=row, col=1
            )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)
        
        fig.update_yaxes(title_text="USD/TRY Change (%)", row=row, col=1)
    
    def _add_liquidity_chart(self, fig, row):
        """Add liquidity metrics"""
        if 'volume_ratio' in self.features.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.features.index,
                    y=self.features['volume_ratio'],
                    name='Volume Ratio',
                    line=dict(color='purple')
                ),
                row=row, col=1
            )
        
        # Add reference line at 1.0
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=row, col=1)
        
        fig.update_yaxes(title_text="Volume Ratio", row=row, col=1)
    
    def create_regime_distribution_chart(self):
        """Create pie charts showing regime distributions"""
        # Create subplots for each regime type
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Volatility Regime', 'Trend Regime', 'Risk Regime', 'Liquidity Regime'),
            specs=[[{'type':'pie'}, {'type':'pie'}],
                   [{'type':'pie'}, {'type':'pie'}]]
        )
        
        regime_cols = ['volatility_regime', 'trend_regime', 'risk_regime', 'liquidity_regime']
        positions = [(1,1), (1,2), (2,1), (2,2)]
        
        for col, (row, col_pos) in zip(regime_cols, positions):
            if col in self.regimes.columns:
                counts = self.regimes[col].value_counts()
                
                fig.add_trace(
                    go.Pie(
                        labels=counts.index,
                        values=counts.values,
                        name=col.replace('_', ' ').title()
                    ),
                    row=row, col=col_pos
                )
        
        fig.update_layout(
            title_text="Regime Distribution",
            height=800
        )
        
        return fig


if __name__ == "__main__":
    # Test dashboard creation
    from market_data import DataLoader, FeatureEngine
    from regime_models import RegimeClassifier
    
    loader = DataLoader()
    data = loader.load_all(fetch_usdtry=True)
    
    engine = FeatureEngine(data)
    features = engine.calculate_all_features()
    
    classifier = RegimeClassifier(features)
    regimes = classifier.classify_all()
    
    dashboard = RegimeDashboard(data, features, regimes)
    dashboard.create_dashboard()
    
    print("Dashboard created successfully!")
