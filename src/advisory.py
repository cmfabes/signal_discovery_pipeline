"""
Advisory signal generation based on port activity correlation with market movements.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from .data_ingest import load_market_data
from .duckdb_io import read_port_features as read_port_features_df
from .analysis import lagged_correlation, to_log_returns, newey_west_pvalue
from .transforms import compute_rolling_zscore

DEFAULT_PORTS = [
    "Los Angeles / Long Beach",
    "New York / New Jersey",
    "Shanghai",
    "Shenzhen/Yantian",
    "Singapore"
]

DEFAULT_TICKERS = ["SPY", "XLI", "XLK"]  # S&P 500, Industrials, Technology
LOOKBACK_DAYS = 90
MIN_CORRELATION = 0.3
SIGNIFICANCE_THRESHOLD = 0.05

def get_trading_signals() -> Dict:
    """
    Generate trading signals and recommendations based on port activity correlation with markets.
    Returns a dictionary with signals, strengths, and specific recommendations.
    """
    # Read latest port data
    df = read_port_features_df("data/ports.duckdb", "port_features")
    if df.empty:
        return {"error": "No port data available"}
    
    end_date = df.index.max()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    signals = []
    
    # Get market data
    mkt_data = load_market_data(DEFAULT_TICKERS, 
                               start_date=start_date.strftime("%Y-%m-%d"),
                               end_date=end_date.strftime("%Y-%m-%d"))
    
    for port in DEFAULT_PORTS:
        port_data = df[df["port"] == port].copy()
        if port_data.empty:
            continue
            
        # Key metrics we care about
        metrics = ["anchored_count", "arrivals_count", "berth_count"]
        
        for metric in metrics:
            if metric not in port_data.columns:
                continue
                
            # Compute z-scores for the metric
            metric_zscore = compute_rolling_zscore(port_data[metric], window=30)
            
            for ticker in DEFAULT_TICKERS:
                if ticker not in mkt_data.columns:
                    continue
                    
                # Convert both series to returns
                port_returns = to_log_returns(metric_zscore)
                mkt_returns = to_log_returns(mkt_data[ticker])
                
                # Test different lags
                for lag in [-10, -5, -3, -1, 0, 1, 3, 5, 10]:
                    corr = lagged_correlation(port_returns, mkt_returns, lags=[lag])
                    if corr.empty:
                        continue
                        
                    coef = float(corr["coef"].iloc[0])
                    
                    # Only consider strong correlations
                    if abs(coef) >= MIN_CORRELATION:
                        # Test statistical significance
                        _, p_val = newey_west_pvalue(port_returns, mkt_returns, 
                                                    lag=lag, use_returns=True)
                        
                        if p_val <= SIGNIFICANCE_THRESHOLD:
                            # Get latest metric value and z-score
                            latest_val = float(port_data[metric].iloc[-1])
                            latest_z = float(metric_zscore.iloc[-1])
                            
                            signal = {
                                "port": port,
                                "metric": metric,
                                "ticker": ticker,
                                "lag_days": lag,
                                "correlation": coef,
                                "p_value": p_val,
                                "latest_value": latest_val,
                                "latest_zscore": latest_z,
                                "signal_date": end_date.strftime("%Y-%m-%d")
                            }
                            
                            # Add trading recommendation
                            if lag < 0:  # Market leads port
                                signal["interpretation"] = "Reactive"
                                signal["recommendation"] = "Monitor port activity"
                            else:  # Port leads market or concurrent
                                signal["interpretation"] = "Predictive"
                                if latest_z > 1.0 and coef > 0:
                                    signal["recommendation"] = f"Consider LONG {ticker}"
                                elif latest_z < -1.0 and coef > 0:
                                    signal["recommendation"] = f"Consider SHORT {ticker}"
                                elif latest_z > 1.0 and coef < 0:
                                    signal["recommendation"] = f"Consider SHORT {ticker}"
                                elif latest_z < -1.0 and coef < 0:
                                    signal["recommendation"] = f"Consider LONG {ticker}"
                                else:
                                    signal["recommendation"] = "No clear signal"
                            
                            signals.append(signal)
    
    # Sort by absolute correlation strength
    signals.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    
    # Prepare the advisory output
    advisory = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": f"Analyzed {len(DEFAULT_PORTS)} ports vs {len(DEFAULT_TICKERS)} markets",
        "signals": signals,
        "top_signals": [s for s in signals if s["interpretation"] == "Predictive"][:3]
    }
    
    return advisory