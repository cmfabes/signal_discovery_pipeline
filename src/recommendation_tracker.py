"""
Track recommended trades and active signals.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

@dataclass
class TradeRecommendation:
    ticker: str
    action: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    reason: str
    confidence: float
    signal_date: datetime
    port_catalyst: str
    active: bool = True

class RecommendationTracker:
    def __init__(self):
        self.active_recommendations: List[TradeRecommendation] = []
        self.historical_recommendations: List[TradeRecommendation] = []
        
    def add_recommendation(self, signal: Dict) -> TradeRecommendation:
        """Create a new trade recommendation from a signal"""
        price = float(signal["current_price"])
        
        # Calculate suggested levels
        stop_loss = price * 0.95 if signal["direction"] == "BULLISH" else price * 1.05
        take_profit_1 = price * 1.05 if signal["direction"] == "BULLISH" else price * 0.95
        take_profit_2 = price * 1.10 if signal["direction"] == "BULLISH" else price * 0.90
        
        rec = TradeRecommendation(
            ticker=signal["target"]["ticker"],
            action="BUY" if signal["direction"] == "BULLISH" else "SELL",
            entry_price=price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            reason=f"Signal from {signal['port']} port activity",
            confidence=signal["confidence"],
            signal_date=datetime.now(),
            port_catalyst=signal["port"],
            active=True
        )
        
        self.active_recommendations.append(rec)
        return rec
    
    def deactivate_recommendation(self, ticker: str) -> None:
        """Move a recommendation from active to historical"""
        for rec in self.active_recommendations:
            if rec.ticker == ticker:
                rec.active = False
                self.historical_recommendations.append(rec)
                self.active_recommendations.remove(rec)
                break
    
    def get_active_recommendations(self) -> List[TradeRecommendation]:
        """Get all currently active recommendations"""
        return sorted(
            self.active_recommendations,
            key=lambda x: (x.confidence, x.signal_date),
            reverse=True
        )
    
    def get_historical_recommendations(self, 
                                     days: Optional[int] = None) -> List[TradeRecommendation]:
        """Get historical recommendations, optionally limited to recent days"""
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            return [r for r in self.historical_recommendations if r.signal_date >= cutoff]
        return self.historical_recommendations
    
    def get_recommendation_summary(self) -> Dict:
        """Get a summary of current recommendations"""
        active = self.get_active_recommendations()
        return {
            "total_active": len(active),
            "high_confidence": len([r for r in active if r.confidence >= 0.9]),
            "moderate_confidence": len([r for r in active if 0.7 <= r.confidence < 0.9]),
            "buy_signals": len([r for r in active if r.action == "BUY"]),
            "sell_signals": len([r for r in active if r.action == "SELL"]),
            "recent_signals": len(self.get_historical_recommendations(days=5))
        }