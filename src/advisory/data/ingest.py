"""
Data ingestion utilities focused on market and port activity data.
"""
import os
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def load_market_data(tickers: List[str],
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Load market data for specified tickers.
    Returns daily returns.
    """
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=180)  # 6 months history
        
    # Download data
    market_data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                # Calculate daily returns
                returns = data['Adj Close'].pct_change()
                market_data[ticker] = returns
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
            
    return market_data

def load_port_data(db_path: str) -> pd.DataFrame:
    """
    Load port activity data from DuckDB.
    Returns daily metrics per port.
    """
    import duckdb
    
    with duckdb.connect(db_path) as conn:
        df = conn.execute("""
            SELECT date, port_name as port,
                   anchored_count, berth_count, 
                   queue_length, total_tonnage
            FROM port_features 
            ORDER BY date, port_name
        """).fetchdf()
        
    return df

def load_recent_data(db_path: str,
                    tickers: List[str],
                    days: int = 180) -> Dict[str, pd.DataFrame]:
    """
    Load both port and market data for recent period.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    market_data = load_market_data(tickers, start_date, end_date)
    port_data = load_port_data(db_path)
    
    # Filter port data to match date range
    port_data = port_data[
        (port_data['date'] >= start_date) &
        (port_data['date'] <= end_date)
    ]
    
    return {
        'market': market_data,
        'ports': port_data
    }