"""
Day trading signals and patterns derived from maritime/port activity.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .predictive_scanner import MarketTarget, SignalStrength, SignalDirection
from .data_ingest import load_market_data, load_intraday_data, _get_logger
from .duckdb_io import read_port_features as read_port_features_df
from .transforms import rolling_zscore as compute_rolling_zscore, detect_momentum_shift

# Additional day-trading focused markets
DAY_TRADING_TARGETS = [
    # High-Volume Tech
    MarketTarget("AAPL", "Apple Inc.", "stock", "Technology", "Consumer Electronics"),
    MarketTarget("NVDA", "NVIDIA", "stock", "Technology", "Semiconductors"),
    MarketTarget("AMD", "Advanced Micro Devices", "stock", "Technology", "Semiconductors"),
    
    # Shipping/Logistics
    MarketTarget("MATX", "Matson Inc.", "stock", "Industrials", "Marine Shipping"),
    MarketTarget("KEX", "Kirby Corporation", "stock", "Industrials", "Marine Shipping"),
    MarketTarget("GOGL", "Golden Ocean Group", "stock", "Industrials", "Marine Shipping"),
    
    # Container Leasing
    MarketTarget("TGH", "Textainer Group", "stock", "Industrials", "Container Leasing"),
    MarketTarget("CAI", "CAI International", "stock", "Industrials", "Container Leasing"),
    
    # Port Operations
    MarketTarget("CMRE", "Costamare Inc.", "stock", "Industrials", "Port Operations"),
    MarketTarget("SSW", "Seaspan Corporation", "stock", "Industrials", "Port Operations"),
    
    # Related Industries
    MarketTarget("AAWW", "Atlas Air Worldwide", "stock", "Industrials", "Air Freight"),
    MarketTarget("EXPD", "Expeditors International", "stock", "Industrials", "Freight Forwarding"),
    
    # Leveraged ETFs
    MarketTarget("SOXL", "Direxion Semiconductor Bull 3X", "etf", "Technology", "Leveraged Semiconductor"),
    MarketTarget("TQQQ", "ProShares UltraPro QQQ", "etf", "Technology", "Leveraged Nasdaq"),
    MarketTarget("SPXL", "Direxion Daily S&P 500 Bull 3X", "etf", "Broad Market", "Leveraged S&P 500")
]

class DayTradingScanner:
    def __init__(self):
        self.intraday_patterns = {}
        self.volatility_alerts = {}
        self.volume_spikes = {}
        self.correlated_moves = {}
        
    def scan_premarket_signals(self) -> Dict:
        """Analyze pre-market conditions and port activity for day trading setups"""
        signals = []
        
        # Get overnight port activity changes
        port_data = read_port_features_df("data/ports.duckdb", "port_features")
        latest_date = port_data.index.max()
        
        # Look for significant overnight changes
        for port in port_data["port"].unique():
            port_metrics = port_data[port_data["port"] == port].iloc[-1]
            prev_metrics = port_data[port_data["port"] == port].iloc[-2]
            
            for metric in ["anchored_count", "berth_count", "queue_length"]:
                if metric not in port_metrics:
                    continue
                    
                pct_change = (port_metrics[metric] - prev_metrics[metric]) / prev_metrics[metric]
                zscore = compute_rolling_zscore(port_data[port_data["port"] == port][metric]).iloc[-1]
                
                if abs(pct_change) >= 0.1 or abs(zscore) >= 2.0:  # Significant change
                    impact = self._analyze_historical_impact(
                        port, metric, pct_change, zscore
                    )
                    
                    if impact:
                        signals.append({
                            "port": port,
                            "metric": metric,
                            "change_pct": pct_change,
                            "zscore": zscore,
                            "historical_impact": impact
                        })
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signals": signals
        }
    
    def _analyze_historical_impact(self, 
                                 port: str, 
                                 metric: str, 
                                 change_pct: float, 
                                 zscore: float) -> Optional[Dict]:
        """Analyze how similar historical patterns impacted related stocks"""
        # TODO: Implement historical pattern matching and impact analysis
        pass
    
    def get_intraday_setups(self) -> Dict:
        """Identify potential intraday trading setups based on port activity"""
        setups = []
        
        # Get latest port data
        port_data = read_port_features_df("data/ports.duckdb", "port_features")
        
        # Get intraday market data for related stocks
        market_data = {}
        
        # Try to get intraday data for all targets at once
        all_tickers = [t.ticker for t in DAY_TRADING_TARGETS]
        combined_data = load_intraday_data(all_tickers)
        
        if not combined_data.empty:
            # Split data by ticker
            for target in DAY_TRADING_TARGETS:
                ticker_data = combined_data[combined_data['ticker'] == target.ticker]
                if not ticker_data.empty:
                    market_data[target.ticker] = ticker_data
        
        # If no intraday data available, return empty results
        if not market_data:
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "setups": [],
                "message": "No intraday data available - market may be closed"
            }
        
        # Look for correlations between port changes and stock moves
        for port in port_data["port"].unique():
            port_changes = self._get_port_changes(port_data, port)
            
            for target, data in market_data.items():
                try:
                    # Calculate various technical indicators
                    rsi = self._calculate_rsi(data["Close"])  # Note: Capital C in Close
                    volume_ma = data["Volume"].rolling(20).mean()
                    price_ma = data["Close"].rolling(20).mean()
                    
                    # Look for setup conditions (with additional error checking)
                    if (len(rsi) > 0 and len(volume_ma) > 0 and
                        (rsi.iloc[-1] < 30 or rsi.iloc[-1] > 70) and  # Oversold/Overbought
                        data["Volume"].iloc[-1] > volume_ma.iloc[-1] * 1.5 and  # Volume spike
                        any(abs(chg) > 0.02 for chg in port_changes.values())  # Significant port change
                    ):
                        setup = {
                            "ticker": target,
                            "type": "reversal" if rsi.iloc[-1] < 30 else "momentum",
                            "price": float(data["Close"].iloc[-1]),
                            "volume_ratio": float(data["Volume"].iloc[-1] / volume_ma.iloc[-1]),
                            "rsi": float(rsi.iloc[-1]),
                            "port_catalyst": port,
                            "port_changes": port_changes,
                            "support_levels": self._find_support_levels(data),
                            "resistance_levels": self._find_resistance_levels(data)
                        }
                        setups.append(setup)
                except Exception as e:
                    log = _get_logger()
                    log.error(f"Error processing setup for {target}: {str(e)}")
                    continue
        
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "setups": setups
        }
    
    def _get_port_changes(self, df: pd.DataFrame, port: str) -> Dict[str, float]:
        """Calculate recent changes in port metrics"""
        port_data = df[df["port"] == port].copy()
        changes = {}
        
        metrics = ["anchored_count", "berth_count", "queue_length", "total_capacity"]
        for metric in metrics:
            if metric in port_data.columns:
                try:
                    pct_change = port_data[metric].pct_change().iloc[-1]
                    changes[metric] = pct_change
                except:
                    continue
        
        return changes
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        delta = prices.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _find_support_levels(data: pd.DataFrame, lookback: int = 20) -> List[float]:
        """Find potential support levels using recent lows"""
        window = data.tail(lookback)
        levels = []
        
        # Find swing lows
        for i in range(1, len(window) - 1):
            if (window["Low"].iloc[i] < window["Low"].iloc[i-1] and 
                window["Low"].iloc[i] < window["Low"].iloc[i+1]):
                levels.append(window["Low"].iloc[i])
        
        # Cluster nearby levels
        if levels:
            levels = pd.Series(levels).rolling(2).mean().dropna().tolist()
        
        return sorted(levels)
    
    @staticmethod
    def _find_resistance_levels(data: pd.DataFrame, lookback: int = 20) -> List[float]:
        """Find potential resistance levels using recent highs"""
        window = data.tail(lookback)
        levels = []
        
        # Find swing highs
        for i in range(1, len(window) - 1):
            if (window["High"].iloc[i] > window["High"].iloc[i-1] and 
                window["High"].iloc[i] > window["High"].iloc[i+1]):
                levels.append(window["High"].iloc[i])
        
        # Cluster nearby levels
        if levels:
            levels = pd.Series(levels).rolling(2).mean().dropna().tolist()
        
        return sorted(levels)
    
    def get_real_time_alerts(self) -> Dict:
        """Generate real-time alerts for day traders"""
        now = datetime.now()
        market_open = datetime.combine(now.date(), time(9, 30))
        market_close = datetime.combine(now.date(), time(16, 0))
        
        if now < market_open or now > market_close:
            return {"message": "Market closed - no real-time alerts"}
            
        alerts = []
        
        # Get latest data
        port_data = read_port_features_df("data/ports.duckdb", "port_features")
        
        # Scan for alert conditions
        for target in DAY_TRADING_TARGETS:
            data = load_intraday_data(target.ticker)
            if data.empty:
                continue
                
            # Volume spike alert
            vol_ma = data["Volume"].rolling(20).mean()
            if data["Volume"].iloc[-1] > vol_ma.iloc[-1] * 2:
                alerts.append({
                    "type": "VOLUME_SPIKE",
                    "ticker": target.ticker,
                    "message": f"Volume spike: {target.ticker} trading at {data['Volume'].iloc[-1]/vol_ma.iloc[-1]:.1f}x average volume"
                })
            
            # Breakout alert
            if len(self._find_resistance_levels(data)) > 0:
                last_price = data["Close"].iloc[-1]
                nearest_resist = min([r for r in self._find_resistance_levels(data) if r > last_price], default=None)
                
                if nearest_resist and last_price > nearest_resist * 0.99:
                    alerts.append({
                        "type": "BREAKOUT",
                        "ticker": target.ticker,
                        "message": f"Potential breakout: {target.ticker} approaching resistance at {nearest_resist:.2f}"
                    })
            
            # Correlation alert
            port_changes = {}
            for port in port_data["port"].unique():
                changes = self._get_port_changes(port_data, port)
                if any(abs(chg) > 0.05 for chg in changes.values()):
                    port_changes[port] = changes
            
            if port_changes:
                alerts.append({
                    "type": "PORT_ACTIVITY",
                    "ticker": target.ticker,
                    "message": f"Significant port activity detected: {len(port_changes)} ports showing >5% changes"
                })
        
        return {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "alerts": alerts
        }