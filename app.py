"""
Advanced maritime trading platform with real-time analysis and visualization
"""
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import os

from src.predictive_scanner import PredictiveScanner

# Load environment variables
load_dotenv()

def main():
    # Page config
    st.set_page_config(page_title="Signal Discovery", layout="wide")
    st.title("Signal Discovery Dashboard")
    
    try:
        # Initialize scanner
        scanner = PredictiveScanner()
        results = scanner.scan_for_signals()
        
        if not results:
            st.error("No signals available")
            return
        
        # Display metrics
        cols = st.columns(4)
        with cols[0]:
            st.metric("Signals", len(results.get('signals', [])))
        with cols[1]:
            st.metric("Bullish", len(results.get('bullish_signals', [])))
        with cols[2]:
            st.metric("Bearish", len(results.get('bearish_signals', [])))
        with cols[3]:
            st.metric("High Conf.", len([s for s in results.get('strong_signals', []) 
                                    if s.get('confidence', 0) >= 0.9]))
        
        # Display sector analysis
        st.subheader("Sector Analysis")
        sector_signals = results.get('sector_signals', {})
        
        if sector_signals:
            for sector, signals in sector_signals.items():
                with st.expander(f"{sector} - {len(signals)} signals"):
                    for signal in signals:
                        st.write(f"- {signal.get('target', {}).get('ticker')}: "
                               f"{signal.get('direction')} signal")
        else:
            st.info("No sector signals available")
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()