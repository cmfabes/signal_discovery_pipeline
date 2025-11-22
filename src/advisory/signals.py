"""
Core signal generation and processing logic.
Simplified from original analysis.py to focus only on trading signals.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..data.ingest import load_market_data
from ..data.process import compute_rolling_zscore

class SignalGenerator:
    def __init__(self,
                 min_correlation: float = 0.3,
                 min_confidence: float = 0.95,
                 lookback_days: int = 90):
        self.min_correlation = min_correlation
        self.min_confidence = min_confidence
        self.lookback_days = lookback_days
        
    def analyze_port_activity(self, port_data: pd.DataFrame, port: str) -> Dict:
        """Analyze single port's activity for significant changes"""
        port_metrics = port_data[port_data['port'] == port].copy()
        
        signals = []
        for metric in ['anchored_count', 'berth_count', 'queue_length']:
            if metric not in port_metrics.columns:
                continue
                
            zscore = compute_rolling_zscore(port_metrics[metric])
            latest_z = zscore.iloc[-1] if not zscore.empty else 0
            
            if abs(latest_z) >= 2.0:  # Significant deviation
                signals.append({
                    'metric': metric,
                    'zscore': latest_z,
                    'value': float(port_metrics[metric].iloc[-1]),
                    'direction': 'increase' if latest_z > 0 else 'decrease'
                })
                
        return {
            'port': port,
            'signals': signals,
            'timestamp': datetime.now().isoformat()
        }
    
    def find_market_correlations(self, 
                               port_data: pd.DataFrame,
                               market_data: pd.DataFrame) -> List[Dict]:
        """Find correlations between port activity and market movements"""
        correlations = []
        
        for port in port_data['port'].unique():
            port_metrics = port_data[port_data['port'] == port]
            
            for metric in ['anchored_count', 'berth_count', 'queue_length']:
                if metric not in port_metrics.columns:
                    continue
                    
                metric_zscore = compute_rolling_zscore(port_metrics[metric])
                
                for ticker in market_data.columns:
                    # Look for leading relationships (port activity leading market)
                    for lag in range(1, 11):  # 1 to 10 days forward
                        corr = self._compute_lagged_correlation(
                            metric_zscore,
                            market_data[ticker],
                            lag
                        )
                        
                        if abs(corr['coefficient']) >= self.min_correlation:
                            correlations.append({
                                'port': port,
                                'metric': metric,
                                'ticker': ticker,
                                'lag_days': lag,
                                'correlation': corr['coefficient'],
                                'confidence': corr['confidence'],
                                'latest_zscore': float(metric_zscore.iloc[-1]),
                                'latest_value': float(port_metrics[metric].iloc[-1])
                            })
        
        return sorted(correlations, 
                     key=lambda x: (abs(x['correlation']), x['confidence']),
                     reverse=True)
    
    def generate_trading_signals(self,
                               port_data: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict:
        """Generate actionable trading signals from port and market data"""
        # Get correlations
        correlations = self.find_market_correlations(port_data, market_data)
        
        # Filter for high-confidence signals
        strong_signals = [
            c for c in correlations 
            if c['confidence'] >= self.min_confidence
        ]
        
        # Generate trade recommendations
        recommendations = []
        for signal in strong_signals:
            if abs(signal['latest_zscore']) >= 1.5:  # Significant current activity
                recommendations.append({
                    'ticker': signal['ticker'],
                    'action': 'BUY' if signal['correlation'] * signal['latest_zscore'] > 0 else 'SELL',
                    'confidence': signal['confidence'],
                    'port': signal['port'],
                    'metric': signal['metric'],
                    'expected_days': signal['lag_days'],
                    'zscore': signal['latest_zscore'],
                    'correlation': signal['correlation']
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'signals': recommendations,
            'correlations': correlations
        }
    
    def _compute_lagged_correlation(self,
                                  series1: pd.Series,
                                  series2: pd.Series,
                                  lag_days: int) -> Dict:
        """Compute correlation with statistical confidence"""
        # Align and lag the series
        s1 = series1.copy()
        s2 = series2.shift(-lag_days)  # Shift for forward-looking correlation
        
        # Remove NaN values
        valid = ~(s1.isna() | s2.isna())
        s1 = s1[valid]
        s2 = s2[valid]
        
        if len(s1) < 30:  # Need enough data points
            return {'coefficient': 0, 'confidence': 0}
        
        # Compute correlation
        coef = s1.corr(s2)
        
        # Compute confidence using bootstrap
        n_bootstrap = 1000
        bootstrap_corrs = []
        for _ in range(n_bootstrap):
            idx = np.random.randint(0, len(s1), len(s1))
            boot_corr = s1.iloc[idx].corr(s2.iloc[idx])
            bootstrap_corrs.append(boot_corr)
        
        confidence = 1 - (sum(abs(c) >= abs(coef) for c in bootstrap_corrs) / n_bootstrap)
        
        return {
            'coefficient': coef,
            'confidence': confidence
        }