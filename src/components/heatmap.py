"""
Heatmap components for visualizing port activity and market correlations.
"""
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class ActivityHeatmap:
    def __init__(self):
        self.color_scale = [
            [0.0, 'rgb(49,54,149)'],
            [0.2, 'rgb(69,117,180)'],
            [0.4, 'rgb(116,173,209)'],
            [0.6, 'rgb(224,243,248)'],
            [0.8, 'rgb(254,224,144)'],
            [1.0, 'rgb(215,48,39)']
        ]
    
    def create_port_activity_heatmap(self, 
                                   port_data: pd.DataFrame,
                                   metric: str,
                                   normalize: bool = True) -> go.Figure:
        """Create a heatmap showing port activity levels"""
        # Pivot data to ports x dates format
        pivot_data = port_data.pivot(
            columns='port',
            values=metric
        )
        
        if normalize:
            # Normalize each port's data to z-scores
            pivot_data = (pivot_data - pivot_data.mean()) / pivot_data.std()
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=self.color_scale,
            colorbar=dict(
                title=dict(
                    text='Activity Level' if not normalize else 'Z-Score',
                    side='right'
                )
            )
        ))
        
        fig.update_layout(
            title=f'Port Activity Heatmap: {metric}',
            xaxis_title='Ports',
            yaxis_title='Date',
            height=600,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white')
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def create_correlation_heatmap(self,
                                 correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap between markets and port metrics"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale=self.color_scale,
            colorbar=dict(
                title=dict(
                    text='Correlation',
                    side='right'
                )
            ),
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title='Market-Port Correlation Matrix',
            xaxis_title='Markets',
            yaxis_title='Port Metrics',
            height=800,
            width=1200,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white')
        )
        
        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)
        
        return fig
        
    def create_lag_impact_heatmap(self,
                                 impact_data: pd.DataFrame) -> go.Figure:
        """Create a heatmap showing lagged impact of port activity on markets"""
        fig = go.Figure(data=go.Heatmap(
            z=impact_data.values,
            x=impact_data.columns,  # Lag days
            y=impact_data.index,    # Port-Market pairs
            colorscale=self.color_scale,
            colorbar=dict(
                title=dict(
                    text='Impact Score',
                    side='right'
                )
            )
        ))
        
        fig.update_layout(
            title='Lagged Impact Analysis',
            xaxis_title='Lag (Days)',
            yaxis_title='Port-Market Pairs',
            height=800,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white')
        )
        
        return fig

class MarketHeatmap:
    def __init__(self):
        self.color_scale = [
            [0.0, 'rgb(215,48,39)'],     # Deep red for negative
            [0.4, 'rgb(254,224,144)'],    # Light red
            [0.5, 'rgb(255,255,255)'],    # White for neutral
            [0.6, 'rgb(144,238,144)'],    # Light green
            [1.0, 'rgb(0,128,0)']         # Deep green for positive
        ]
    
    def create_market_heatmap(self,
                            returns_data: pd.DataFrame,
                            timeframes: List[str],
                            sectors: Optional[List[str]] = None) -> go.Figure:
        """Create a market returns heatmap grouped by sector"""
        if sectors is None:
            # Group by the index level containing sectors if available
            if isinstance(returns_data.index, pd.MultiIndex):
                sectors = returns_data.index.get_level_values(1).unique()
            else:
                # No sector grouping
                sectors = ['All Markets']
        
        fig = go.Figure()
        
        # Create a heatmap for each sector
        y_labels = []
        z_values = []
        
        for sector in sectors:
            if sector == 'All Markets':
                sector_data = returns_data
            else:
                sector_data = returns_data[returns_data.index.get_level_values(1) == sector]
            
            # Add sector name as a header
            y_labels.extend([sector, ''])
            z_values.extend([np.full(len(timeframes), np.nan), np.full(len(timeframes), np.nan)])
            
            # Add market data
            for idx in sector_data.index:
                market_name = idx[0] if isinstance(idx, tuple) else idx
                y_labels.append(market_name)
                z_values.append(sector_data.loc[idx].values)
                
            # Add spacing between sectors
            y_labels.append('')
            z_values.append(np.full(len(timeframes), np.nan))
        
        fig.add_trace(go.Heatmap(
            z=z_values,
            x=timeframes,
            y=y_labels,
            colorscale=self.color_scale,
            colorbar=dict(
                title='Returns %',
                titleside='right'
            ),
            zmin=-5,  # -5% returns
            zmax=5,   # +5% returns
        ))
        
        fig.update_layout(
            title='Market Returns by Timeframe',
            xaxis_title='Timeframe',
            yaxis_title='Markets',
            height=1000,
            paper_bgcolor='rgb(25,25,25)',
            plot_bgcolor='rgb(25,25,25)',
            font=dict(color='white')
        )
        
        return fig