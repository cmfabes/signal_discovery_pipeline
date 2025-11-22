"""
Advanced market scanner with customizable filters and real-time alerts.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

@dataclass
class ScanCriteria:
    """Criteria for market scanning"""
    field: str
    operator: str  # '>', '<', '==', 'between'
    value: float
    value2: Optional[float] = None  # For 'between' operator
    
@dataclass
class ScanResult:
    """Result from market scan"""
    ticker: str
    metric_name: str
    current_value: float
    signal_type: str
    confidence: float
    port_catalyst: Optional[str] = None
    additional_data: Optional[Dict] = None

class MarketScanner:
    def __init__(self):
        self.available_metrics = [
            "rsi",
            "volume_ratio",
            "price_change",
            "correlation",
            "port_impact",
            "momentum_score"
        ]
        
        self.available_operators = [
            ">", "<", "==", "between",
            "crossed_above", "crossed_below"
        ]
        
    def create_scan(self, name: str, criteria: List[ScanCriteria]) -> Dict:
        """Create a new market scan configuration"""
        return {
            "name": name,
            "criteria": criteria,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
    def run_scan(self, 
                market_data: pd.DataFrame,
                port_data: pd.DataFrame,
                criteria: List[ScanCriteria]) -> List[ScanResult]:
        """Run market scan with given criteria"""
        results = []
        
        # Process each market
        for ticker in market_data.columns:
            matches_all = True
            additional_data = {}
            
            for criterion in criteria:
                value = self._get_metric_value(
                    criterion.field,
                    ticker,
                    market_data,
                    port_data
                )
                
                if value is None:
                    matches_all = False
                    break
                    
                # Store the value for reporting
                additional_data[criterion.field] = value
                
                # Check if meets criterion
                if not self._check_criterion(value, criterion):
                    matches_all = False
                    break
            
            if matches_all:
                # Get contextual data for the signal
                signal_type = self._determine_signal_type(additional_data)
                confidence = self._calculate_confidence(additional_data)
                port_catalyst = self._find_port_catalyst(
                    ticker, market_data, port_data
                )
                
                results.append(ScanResult(
                    ticker=ticker,
                    metric_name=criteria[0].field,  # Use first criterion as main metric
                    current_value=value,
                    signal_type=signal_type,
                    confidence=confidence,
                    port_catalyst=port_catalyst,
                    additional_data=additional_data
                ))
        
        return results
    
    def _get_metric_value(self,
                         metric: str,
                         ticker: str,
                         market_data: pd.DataFrame,
                         port_data: pd.DataFrame) -> Optional[float]:
        """Calculate metric value for a given market"""
        try:
            if metric == "rsi":
                return self._calculate_rsi(market_data[ticker])
            elif metric == "volume_ratio":
                return self._calculate_volume_ratio(market_data[ticker])
            elif metric == "price_change":
                return self._calculate_price_change(market_data[ticker])
            elif metric == "correlation":
                return self._calculate_port_correlation(
                    market_data[ticker], port_data
                )
            elif metric == "momentum_score":
                return self._calculate_momentum_score(market_data[ticker])
        except Exception:
            return None
            
        return None
    
    def _check_criterion(self,
                        value: float,
                        criterion: ScanCriteria) -> bool:
        """Check if value meets the criterion"""
        if criterion.operator == ">":
            return value > criterion.value
        elif criterion.operator == "<":
            return value < criterion.value
        elif criterion.operator == "==":
            return abs(value - criterion.value) < 0.0001
        elif criterion.operator == "between":
            return criterion.value <= value <= criterion.value2
        elif criterion.operator == "crossed_above":
            # Would need historical values to implement
            return False
        elif criterion.operator == "crossed_below":
            # Would need historical values to implement
            return False
        return False
    
    def _calculate_rsi(self, prices: pd.Series, periods: int = 14) -> float:
        """Calculate RSI for latest period"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_volume_ratio(self,
                              market_data: pd.Series,
                              lookback: int = 20) -> float:
        """Calculate volume ratio vs average"""
        if 'Volume' not in market_data:
            return 0.0
        avg_volume = market_data['Volume'].rolling(lookback).mean()
        return market_data['Volume'].iloc[-1] / avg_volume.iloc[-1]
    
    def _calculate_price_change(self,
                              market_data: pd.Series,
                              periods: int = 1) -> float:
        """Calculate price change over periods"""
        if isinstance(market_data, pd.DataFrame) and 'Close' in market_data:
            close_prices = market_data['Close']
        else:
            close_prices = market_data
            
        return (
            (close_prices.iloc[-1] - close_prices.iloc[-periods-1]) 
            / close_prices.iloc[-periods-1]
        ) * 100
    
    def _calculate_port_correlation(self,
                                  market_data: pd.Series,
                                  port_data: pd.DataFrame,
                                  lookback: int = 30) -> float:
        """Calculate correlation with port metrics"""
        # Implementation would depend on how port data is structured
        return 0.0
    
    def _calculate_momentum_score(self,
                                market_data: pd.Series,
                                timeframes: List[int] = [20, 50, 200]) -> float:
        """Calculate momentum score based on multiple timeframes"""
        score = 0.0
        weights = [0.5, 0.3, 0.2]  # More weight to shorter timeframes
        
        for timeframe, weight in zip(timeframes, weights):
            pct_change = self._calculate_price_change(market_data, timeframe)
            score += pct_change * weight
            
        return score
    
    def _determine_signal_type(self, data: Dict) -> str:
        """Determine the type of signal based on metrics"""
        if 'rsi' in data:
            if data['rsi'] < 30:
                return 'OVERSOLD'
            elif data['rsi'] > 70:
                return 'OVERBOUGHT'
                
        if 'volume_ratio' in data and data['volume_ratio'] > 2.0:
            return 'VOLUME_SPIKE'
            
        if 'momentum_score' in data:
            if data['momentum_score'] > 5.0:
                return 'STRONG_MOMENTUM'
            elif data['momentum_score'] < -5.0:
                return 'STRONG_REVERSAL'
                
        return 'NEUTRAL'
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence score for the signal"""
        confidence = 0.5  # Base confidence
        
        # Add confidence based on volume confirmation
        if 'volume_ratio' in data:
            confidence += min(0.2, (data['volume_ratio'] - 1.0) * 0.1)
            
        # Add confidence based on momentum
        if 'momentum_score' in data:
            confidence += min(0.2, abs(data['momentum_score']) * 0.02)
            
        # Add confidence based on RSI extremes
        if 'rsi' in data:
            if data['rsi'] < 20 or data['rsi'] > 80:
                confidence += 0.1
            elif data['rsi'] < 30 or data['rsi'] > 70:
                confidence += 0.05
                
        return min(1.0, confidence)
    
    def _find_port_catalyst(self,
                          ticker: str,
                          market_data: pd.DataFrame,
                          port_data: pd.DataFrame) -> Optional[str]:
        """Find port activity that might be catalyst"""
        # Implementation would depend on port data structure
        return None