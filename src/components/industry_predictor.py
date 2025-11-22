"""
Industry impact prediction component.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

@dataclass
class IndustryImpact:
    industry: str
    confidence: float
    expected_date: datetime
    impact_type: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    magnitude: float  # Scale of 1-10
    key_tickers: List[str]
    catalyst: str
    supporting_data: Dict[str, float]

class IndustryPredictor:
    def __init__(self):
        # Industry categories and their related commodities/cargo types
        self.industry_mapping = {
            "Semiconductors": ["semiconductor equipment", "silicon wafers", "chip manufacturing equipment"],
            "Consumer Electronics": ["electronics", "computer parts", "mobile devices"],
            "Automotive": ["vehicles", "auto parts", "lithium", "batteries"],
            "Energy": ["oil", "gas", "solar panels", "wind turbines"],
            "Basic Materials": ["iron ore", "copper", "aluminum", "chemicals"],
            "Consumer Goods": ["clothing", "furniture", "appliances"],
            "Healthcare": ["medical equipment", "pharmaceuticals"],
            "Industrial Equipment": ["machinery", "manufacturing equipment"],
            "Agriculture": ["fertilizers", "agricultural equipment", "grain"]
        }
        
        # Port specializations
        self.port_specializations = {
            "Los Angeles / Long Beach": ["electronics", "consumer goods"],
            "Oakland": ["agriculture", "technology"],
            "Shanghai": ["electronics", "industrial equipment"],
            "Rotterdam": ["energy", "chemicals"],
            "Singapore": ["electronics", "oil"]
        }
    
    def analyze_port_data(self, 
                         port_data: pd.DataFrame,
                         cargo_data: pd.DataFrame) -> List[IndustryImpact]:
        """Analyze port and cargo data to predict industry impacts"""
        impacts = []
        
        for port in port_data['port'].unique():
            port_metrics = port_data[port_data['port'] == port].iloc[-1]
            
            # Get cargo composition for this port
            port_cargo = cargo_data[cargo_data['port'] == port].iloc[-1] if not cargo_data.empty else None
            
            # Analyze each industry that could be affected
            for industry, commodities in self.industry_mapping.items():
                if port in self.port_specializations:
                    relevance = any(comm in self.port_specializations[port] 
                                  for comm in commodities)
                    if not relevance:
                        continue
                
                # Calculate impact based on port metrics and cargo data
                impact = self._calculate_industry_impact(
                    industry, port_metrics, port_cargo
                )
                
                if impact:
                    impacts.append(impact)
        
        return impacts
    
    def _calculate_industry_impact(self,
                                 industry: str,
                                 port_metrics: pd.Series,
                                 cargo_data: Optional[pd.Series]) -> Optional[IndustryImpact]:
        """Calculate specific impact on an industry based on port and cargo data"""
        # Base impact calculation
        queue_change = port_metrics.get('queue_length_change', 0)
        capacity_change = port_metrics.get('total_capacity_change', 0)
        
        # Impact factors
        factors = {
            'queue_impact': queue_change * -0.5,  # Negative impact from longer queues
            'capacity_impact': capacity_change * 0.3,
            'efficiency_impact': port_metrics.get('efficiency_score', 0) * 0.2
        }
        
        # Calculate overall impact
        total_impact = sum(factors.values())
        
        # Determine impact type
        if abs(total_impact) < 0.5:
            return None  # No significant impact
            
        impact_type = "POSITIVE" if total_impact > 0 else "NEGATIVE"
        
        # Calculate confidence based on data quality and historical correlation
        confidence = self._calculate_confidence(factors, industry)
        
        # Estimate impact date based on queue length and processing time
        days_to_impact = max(1, int(port_metrics.get('queue_length', 0) / 
                                  port_metrics.get('daily_processing_rate', 5)))
        impact_date = datetime.now() + timedelta(days=days_to_impact)
        
        # Get relevant tickers for this industry
        key_tickers = self._get_industry_tickers(industry)
        
        return IndustryImpact(
            industry=industry,
            confidence=confidence,
            expected_date=impact_date,
            impact_type=impact_type,
            magnitude=abs(total_impact),
            key_tickers=key_tickers,
            catalyst=f"Port activity changes and cargo composition",
            supporting_data=factors
        )
    
    def _calculate_confidence(self, 
                            factors: Dict[str, float], 
                            industry: str) -> float:
        """Calculate confidence score for the prediction"""
        # Base confidence
        confidence = 0.5
        
        # Add confidence based on data quality
        if all(abs(v) > 0.1 for v in factors.values()):
            confidence += 0.2
        
        # Add confidence based on historical accuracy (would need historical data)
        confidence += 0.1
        
        # Cap at 0.95
        return min(0.95, confidence)
    
    def _get_industry_tickers(self, industry: str) -> List[str]:
        """Get key tickers for an industry"""
        # This would ideally come from a more complete mapping
        industry_tickers = {
            "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "ASML"],
            "Consumer Electronics": ["AAPL", "DELL", "HPQ", "SNE"],
            "Automotive": ["TSLA", "GM", "F", "TM"],
            "Energy": ["XOM", "CVX", "BP", "SHELL"],
            "Basic Materials": ["BHP", "RIO", "FCX", "DOW"],
            "Consumer Goods": ["PG", "KO", "WMT", "COST"],
            "Healthcare": ["JNJ", "PFE", "MRK", "ABT"],
            "Industrial Equipment": ["CAT", "DE", "MMM", "HON"],
            "Agriculture": ["NTR", "CF", "ADM", "BG"]
        }
        
        return industry_tickers.get(industry, [])
    
    def create_impact_timeline(self, impacts: List[IndustryImpact]) -> go.Figure:
        """Create a timeline visualization of expected industry impacts"""
        fig = go.Figure()
        
        # Sort impacts by date
        impacts.sort(key=lambda x: x.expected_date)
        
        for impact in impacts:
            color = ("#00CC96" if impact.impact_type == "POSITIVE" 
                    else "#EF553B" if impact.impact_type == "NEGATIVE" 
                    else "#FFA15A")
            
            size = impact.confidence * 20  # Size based on confidence
            
            # Create hover text
            hover_text = f"""
            Industry: {impact.industry}<br>
            Impact: {impact.impact_type}<br>
            Magnitude: {impact.magnitude:.1f}<br>
            Confidence: {impact.confidence:.1%}<br>
            Key Tickers: {', '.join(impact.key_tickers)}<br>
            Catalyst: {impact.catalyst}
            """
            
            fig.add_trace(go.Scatter(
                x=[impact.expected_date],
                y=[impact.industry],
                mode="markers",
                marker=dict(
                    size=size,
                    color=color,
                    symbol="diamond" if impact.confidence > 0.7 else "circle"
                ),
                name=f"{impact.industry} ({impact.impact_type})",
                text=hover_text,
                hoverinfo="text"
            ))
        
        fig.update_layout(
            title="Industry Impact Timeline",
            xaxis_title="Expected Impact Date",
            yaxis_title="Industry",
            height=400,
            showlegend=True,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    def create_confidence_heatmap(self, impacts: List[IndustryImpact]) -> go.Figure:
        """Create a heatmap of prediction confidence by industry"""
        industries = list(self.industry_mapping.keys())
        impact_scores = {ind: 0.0 for ind in industries}
        confidence_scores = {ind: 0.0 for ind in industries}
        
        # Aggregate impacts and confidence
        for impact in impacts:
            current_score = impact_scores[impact.industry]
            current_conf = confidence_scores[impact.industry]
            
            # Weight the impact by confidence
            weighted_impact = impact.magnitude * (1 if impact.impact_type == "POSITIVE" else -1)
            impact_scores[impact.industry] = (current_score + weighted_impact) / 2
            confidence_scores[impact.industry] = (current_conf + impact.confidence) / 2
        
        # Create heatmap data
        fig = go.Figure(data=go.Heatmap(
            z=[[score] for score in impact_scores.values()],
            y=list(impact_scores.keys()),
            x=['Impact Score'],
            colorscale=[
                [0, '#EF553B'],      # Red for negative
                [0.5, '#FFA15A'],    # Yellow for neutral
                [1, '#00CC96']       # Green for positive
            ],
            colorbar=dict(title='Impact Score')
        ))
        
        fig.update_layout(
            title='Industry Impact Heatmap',
            height=400,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white')
        )
        
        return fig