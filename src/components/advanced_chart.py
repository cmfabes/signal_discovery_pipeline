"""
Advanced charting component with candlesticks, indicators, and annotations.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class TechnicalIndicator:
    name: str
    values: pd.Series
    color: str
    overlay: bool = True  # If False, will be plotted in separate subplot
    
def calculate_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices: pd.Series, 
                  fast: int = 12, 
                  slow: int = 26, 
                  signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD and signal line"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return {"macd": macd, "signal": signal_line, "histogram": macd - signal_line}

class AdvancedChart:
    def __init__(self):
        self.indicators: List[TechnicalIndicator] = []
        self.annotations = []
        
    def create_candlestick_chart(self,
                                df: pd.DataFrame,
                                volume: bool = True,
                                show_ma: bool = True,
                                show_bbands: bool = True) -> go.Figure:
        """Create an advanced candlestick chart with volume and indicators"""
        # Determine how many rows needed for subplots
        n_rows = 1
        if volume:
            n_rows += 1
            
        # Count non-overlay indicators
        n_rows += len([i for i in self.indicators if not i.overlay])
        
        # Create figure with subplots
        fig = make_subplots(rows=n_rows, 
                           cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.6] + [0.2] * (n_rows-1))
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Add moving averages if requested
        if show_ma:
            for period in [20, 50, 200]:
                ma = df['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma,
                        name=f"MA{period}",
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands if requested
        if show_bbands:
            period = 20
            std_dev = 2
            ma = df['Close'].rolling(window=period).mean()
            std = df['Close'].rolling(window=period).std()
            upper = ma + (std * std_dev)
            lower = ma - (std * std_dev)
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=upper,
                    name='Upper BB',
                    line=dict(width=1, dash='dash'),
                    opacity=0.5
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=lower,
                    name='Lower BB',
                    line=dict(width=1, dash='dash'),
                    opacity=0.5,
                    fill='tonexty'  # Fill area between upper and lower bands
                ),
                row=1, col=1
            )
        
        # Add volume bar chart
        current_row = 2
        if volume:
            colors = np.where(df['Open'].values > df['Close'].values, 'red', 'green')
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.5
                ),
                row=current_row, col=1
            )
            current_row += 1
        
        # Add additional technical indicators
        for indicator in self.indicators:
            if indicator.overlay:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator.values,
                        name=indicator.name,
                        line=dict(color=indicator.color)
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicator.values,
                        name=indicator.name,
                        line=dict(color=indicator.color)
                    ),
                    row=current_row, col=1
                )
                current_row += 1
        
        # Add any annotations
        for annotation in self.annotations:
            fig.add_annotation(annotation)
        
        # Update layout
        fig.update_layout(
            title_text="Advanced Technical Analysis",
            xaxis_title="Date",
            yaxis_title="Price",
            height=800,  # Taller to accommodate all elements
            xaxis_rangeslider_visible=False,  # Disable rangeslider
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Make it look more like ThinkOrSwim
        fig.update_layout(
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
        
    def add_indicator(self, name: str, values: pd.Series, 
                     color: str, overlay: bool = True):
        """Add a technical indicator to the chart"""
        self.indicators.append(
            TechnicalIndicator(name=name, values=values, 
                             color=color, overlay=overlay)
        )
        
    def add_annotation(self, x, y, text, color="white"):
        """Add an annotation to the chart"""
        self.annotations.append(dict(
            x=x, y=y,
            text=text,
            showarrow=True,
            arrowhead=1,
            font=dict(color=color)
        ))
        
    def clear_indicators(self):
        """Clear all indicators"""
        self.indicators = []
        
    def clear_annotations(self):
        """Clear all annotations"""
        self.annotations = []