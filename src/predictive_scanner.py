"""
Comprehensive shipping-to-market predictive scanner.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from .data_ingest import load_market_data
from .duckdb_io import read_port_features
from .analysis import lagged_correlation, to_log_returns, newey_west_pvalue
from .transforms import rolling_zscore as compute_rolling_zscore, detect_momentum_shift

class SignalStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

class SignalDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class MarketTarget:
    """Market target with related industry/sector context"""
    ticker: str
    name: str
    type: str  # 'stock', 'etf', 'index'
    sector: Optional[str] = None
    industry: Optional[str] = None

# Key markets to monitor
MARKET_TARGETS = [
    # Broad Market
    MarketTarget("SPY", "S&P 500", "etf"),
    MarketTarget("QQQ", "Nasdaq 100", "etf"),
    MarketTarget("IWM", "Russell 2000", "etf"),
    
    # Sectors
    MarketTarget("XLI", "Industrial Select Sector", "etf", "Industrials"),
    MarketTarget("XLK", "Technology Select Sector", "etf", "Technology"),
    MarketTarget("XLY", "Consumer Discretionary Select Sector", "etf", "Consumer Discretionary"),
    MarketTarget("XLP", "Consumer Staples Select Sector", "etf", "Consumer Staples"),
    MarketTarget("XLE", "Energy Select Sector", "etf", "Energy"),
    
    # Industry-Specific
    MarketTarget("SOXL", "Semiconductor Bull 3X", "etf", "Technology", "Semiconductors"),
    MarketTarget("FDX", "FedEx Corporation", "stock", "Industrials", "Logistics"),
    MarketTarget("UPS", "United Parcel Service", "stock", "Industrials", "Logistics"),
    MarketTarget("ZIM", "ZIM Integrated Shipping", "stock", "Industrials", "Shipping"),
    MarketTarget("MAERSK-B.CO", "Maersk", "stock", "Industrials", "Shipping"),
]

# Major ports and regions to analyze
MONITORED_PORTS = [
    # US West Coast
    "Los Angeles / Long Beach",
    "Oakland",
    "Seattle/Tacoma",
    
    # US East Coast
    "New York / New Jersey",
    "Savannah",
    "Virginia",
    
    # Asia
    "Shanghai",
    "Ningbo-Zhoushan",
    "Shenzhen/Yantian",
    "Singapore",
    "Busan",
    
    # Europe
    "Rotterdam",
    "Hamburg",
    "Antwerp"
]

# Port metrics to analyze
PORT_METRICS = {
    "anchored_count": "Ships at anchor",
    "berth_count": "Ships at berth",
    "arrivals_count": "Daily arrivals",
    "departures_count": "Daily departures",
    "queue_length": "Queue length",
    "total_capacity": "Total ship capacity",
    "avg_wait_time": "Average wait time"
}

class PredictiveScanner:
    def __init__(self, 
                 lookback_days: int = 90,
                 min_correlation: float = 0.3,
                 min_confidence: float = 0.95,
                 required_lag_days: int = 2):
        """
        Initialize the predictive scanner with parameters
        
        Args:
            lookback_days: Historical period to analyze
            min_correlation: Minimum correlation coefficient to consider
            min_confidence: Minimum statistical confidence (1 - p_value)
            required_lag_days: Minimum number of days the port signal must lead the market
        """
        self.lookback_days = lookback_days
        self.min_correlation = min_correlation
        self.min_confidence = min_confidence
        self.required_lag_days = required_lag_days
        
        # Load and cache data
        self.port_data = self._load_port_data()
        self.market_data = self._load_market_data()
        
    def _load_port_data(self) -> pd.DataFrame:
        """Load and preprocess port activity data"""
        df = read_port_features("data/ports.duckdb", "port_features")
        return df
    
    def _load_market_data(self) -> Dict[str, pd.Series]:
        """Load market data for all tracked instruments"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        tickers = [t.ticker for t in MARKET_TARGETS]
        return load_market_data(tickers, 
                              start_date=start_date.strftime("%Y-%m-%d"),
                              end_date=end_date.strftime("%Y-%m-%d"))
    
    def detect_momentum_patterns(self, port: str, metric: str) -> Optional[Dict]:
        """Detect significant momentum shifts in port metrics"""
        if port not in self.port_data["port"].unique():
            return None
            
        port_data = self.port_data[self.port_data["port"] == port]
        if metric not in port_data.columns:
            return None
            
        # Get the metric series
        series = port_data[metric].dropna()
        if len(series) < 30:  # Need enough data
            return None
            
        # Detect momentum shifts
        shifts = detect_momentum_shift(series, 
                                     window=14, 
                                     threshold=2.0)
        
        if shifts.empty:
            return None
            
        # Get the most recent shift
        latest_shift = shifts.iloc[-1]
        
        return {
            "port": port,
            "metric": metric,
            "shift_date": latest_shift.name,
            "direction": "up" if latest_shift["direction"] > 0 else "down",
            "magnitude": float(latest_shift["magnitude"]),
            "zscore": float(latest_shift["zscore"])
        }
    
    def scan_for_signals(self) -> Dict:
        """
        Scan all ports and markets for predictive signals.
        Returns dictionary with various signal categories and metadata.
        """
        signals = []
        port_patterns = []
        
        # Scan each port-metric combination
        for port in MONITORED_PORTS:
            port_data = self.port_data[self.port_data["port"] == port].copy()
            if port_data.empty:
                continue
                
            # Check for momentum patterns
            for metric in PORT_METRICS:
                pattern = self.detect_momentum_patterns(port, metric)
                if pattern:
                    port_patterns.append(pattern)
            
            # Analyze correlations with markets
            for metric in PORT_METRICS:
                if metric not in port_data.columns:
                    continue
                    
                metric_zscore = compute_rolling_zscore(port_data[metric], window=30)
                
                # Test correlation with each market
                for target in MARKET_TARGETS:
                    if target.ticker not in self.market_data:
                        continue
                        
                    market_series = self.market_data[target.ticker]
                    
                    # Convert to returns
                    port_returns = to_log_returns(metric_zscore)
                    market_returns = to_log_returns(market_series)
                    
                    # Test forward-looking lags
                    for lag in range(self.required_lag_days, 21):
                        corr = lagged_correlation(port_returns, market_returns, lags=[lag])
                        if corr.empty:
                            continue
                            
                        coef = float(corr["coef"].iloc[0])
                        
                        if abs(coef) >= self.min_correlation:
                            # Verify statistical significance
                            _, p_val = newey_west_pvalue(port_returns, market_returns, 
                                                      lag=lag, use_returns=True)
                            
                            if (1 - p_val) >= self.min_confidence:
                                latest_val = float(port_data[metric].iloc[-1])
                                latest_z = float(metric_zscore.iloc[-1])
                                
                                # Determine signal strength
                                if abs(coef) >= 0.6:
                                    strength = SignalStrength.STRONG
                                elif abs(coef) >= 0.45:
                                    strength = SignalStrength.MODERATE
                                else:
                                    strength = SignalStrength.WEAK
                                
                                # Determine direction
                                if latest_z * coef > 1.0:
                                    direction = SignalDirection.BULLISH
                                elif latest_z * coef < -1.0:
                                    direction = SignalDirection.BEARISH
                                else:
                                    direction = SignalDirection.NEUTRAL
                                
                                signal = {
                                    "port": port,
                                    "metric": metric,
                                    "metric_name": PORT_METRICS[metric],
                                    "target": target.__dict__,
                                    "lag_days": lag,
                                    "correlation": coef,
                                    "confidence": 1 - p_val,
                                    "latest_value": latest_val,
                                    "latest_zscore": latest_z,
                                    "strength": strength.value,
                                    "direction": direction.value,
                                    "expected_impact_date": (
                                        datetime.now() + timedelta(days=lag)
                                    ).strftime("%Y-%m-%d"),
                                    "signal_date": datetime.now().strftime("%Y-%m-%d")
                                }
                                
                                signals.append(signal)
        
        # Sort signals by strength and correlation
        signals.sort(key=lambda x: (
            SignalStrength[x["strength"]].value,
            abs(x["correlation"])
        ), reverse=True)
        
        # Group signals by sector/industry
        sector_signals = {}
        industry_signals = {}
        
        for signal in signals:
            target = signal["target"]
            if target["sector"]:
                if target["sector"] not in sector_signals:
                    sector_signals[target["sector"]] = []
                sector_signals[target["sector"]].append(signal)
            
            if target["industry"]:
                if target["industry"] not in industry_signals:
                    industry_signals[target["industry"]] = []
                industry_signals[target["industry"]].append(signal)
        
        return {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "lookback_days": self.lookback_days,
            "signals": signals,
            "port_patterns": port_patterns,
            "sector_signals": sector_signals,
            "industry_signals": industry_signals,
            "strong_signals": [s for s in signals if s["strength"] == SignalStrength.STRONG.value],
            "bullish_signals": [s for s in signals if s["direction"] == SignalDirection.BULLISH.value],
            "bearish_signals": [s for s in signals if s["direction"] == SignalDirection.BEARISH.value]
        }