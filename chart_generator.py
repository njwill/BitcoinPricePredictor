import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st

class ChartGenerator:
    """Generates interactive Bitcoin charts with technical indicators"""
    
    def __init__(self):
        self.colors = {
            'bullish': '#00ff88',
            'bearish': '#ff4444',
            'volume': '#9467bd',
            'ma': '#ff7f0e',
            'bb_upper': '#d62728',
            'bb_lower': '#d62728',
            'bb_middle': '#2ca02c',
            'rsi': '#17becf',
            'macd': '#e377c2',
            'signal': '#7f7f7f'
        }
    
    def create_comprehensive_chart(self, data, indicators, title="Bitcoin Analysis", 
                                 show_indicators=True, show_volume=True, theme="light", 
                                 display_from_index=None):
        """
        Create a comprehensive chart with price, volume, and technical indicators
        
        Args:
            data: pandas.DataFrame with OHLCV data
            indicators: dict with technical indicators
            title: Chart title
            show_indicators: Whether to show technical indicators
            show_volume: Whether to show volume subplot
            
        Returns:
            plotly.graph_objects.Figure
        """
        try:
            # Ensure data is a proper DataFrame
            if not isinstance(data, pd.DataFrame):
                st.error("Invalid data format for chart generation")
                return self._create_empty_chart(title)
            
            # Handle display range trimming if specified
            if display_from_index is not None and display_from_index > 0:
                data = data.iloc[display_from_index:].copy()
                # Trim indicators to match the display data
                for key, value in indicators.items():
                    if isinstance(value, (pd.Series, list)) and len(value) > display_from_index:
                        if isinstance(value, pd.Series):
                            indicators[key] = value.iloc[display_from_index:].copy()
                        else:
                            indicators[key] = value[display_from_index:]
            
            # Ensure indicators are properly formatted
            if indicators:
                for key, value in indicators.items():
                    if isinstance(value, str):
                        # Convert string representation back to array
                        try:
                            # This is a fallback - ideally shouldn't happen
                            indicators[key] = pd.Series(data['Close'].values * 0)  # Create empty series same length
                        except:
                            indicators[key] = []
                    elif isinstance(value, list):
                        indicators[key] = pd.Series(value, index=data.index)
                    elif not isinstance(value, pd.Series):
                        indicators[key] = pd.Series(value, index=data.index)
            # Calculate number of subplots
            subplot_count = 1  # Main price chart
            subplot_titles = [title]
            
            if show_volume:
                subplot_count += 1
                subplot_titles.append("Volume")
            
            if show_indicators and indicators:
                if 'RSI' in indicators:
                    subplot_count += 1
                    subplot_titles.append("RSI")
                if 'MACD' in indicators:
                    subplot_count += 1
                    subplot_titles.append("MACD")
            
            # Create subplots
            row_heights = [0.5]  # Main chart takes 50%
            if show_volume:
                row_heights.append(0.15)
            if show_indicators and indicators:
                remaining_height = 0.5 if not show_volume else 0.35
                indicator_count = sum([1 for key in ['RSI', 'MACD'] if key in indicators])
                if indicator_count > 0:
                    indicator_height = remaining_height / indicator_count
                    row_heights.extend([indicator_height] * indicator_count)
            
            fig = make_subplots(
                rows=subplot_count,
                cols=1,
                subplot_titles=subplot_titles,
                vertical_spacing=0.08,
                row_heights=row_heights,
                specs=[[{"secondary_y": False}]] * subplot_count
            )
            
            # Main candlestick chart
            candlestick = go.Candlestick(
                x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Bitcoin Price',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            )
            fig.add_trace(candlestick, row=1, col=1)
            
            # Define x_axis for all subsequent traces
            x_axis = data.index if 'Datetime' not in data.columns else data['Datetime']
            
            # Add technical indicators to main chart
            if show_indicators and indicators:
                
                # Bollinger Bands
                if all(key in indicators for key in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=indicators['BB_Upper'],
                        line=dict(color=self.colors['bb_upper'], width=1),
                        name='BB Upper',
                        showlegend=False
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=indicators['BB_Lower'],
                        line=dict(color=self.colors['bb_lower'], width=1),
                        fill='tonexty',
                        fillcolor='rgba(214, 39, 40, 0.1)',
                        name='BB Lower',
                        showlegend=False
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=indicators['BB_Middle'],
                        line=dict(color=self.colors['bb_middle'], width=1),
                        name='BB Middle'
                    ), row=1, col=1)
                
                # EMA lines
                for ema_period in [20, 50]:
                    ema_key = f'EMA_{ema_period}'
                    if ema_key in indicators:
                        fig.add_trace(go.Scatter(
                            x=x_axis,
                            y=indicators[ema_key],
                            line=dict(color=self.colors['ma'], width=2 if ema_period == 20 else 1),
                            name=f'EMA {ema_period}'
                        ), row=1, col=1)
            
            current_row = 2
            
            # Volume chart
            if show_volume:
                volume_colors = ['red' if close < open else 'green' 
                               for close, open in zip(data['Close'], data['Open'])]
                
                fig.add_trace(go.Bar(
                    x=data.index if 'Datetime' not in data.columns else data['Datetime'],
                    y=data['Volume'],
                    name='Volume',
                    marker_color=volume_colors,
                    opacity=0.7
                ), row=current_row, col=1)
                current_row += 1
            
            # RSI subplot
            if show_indicators and indicators and 'RSI' in indicators:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=indicators['RSI'],
                    line=dict(color=self.colors['rsi'], width=2),
                    name='RSI'
                ), row=current_row, col=1)
                
                # RSI overbought/oversold levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                             annotation_text="Overbought", row=str(current_row), col="1")
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                             annotation_text="Oversold", row=str(current_row), col="1")
                fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                             row=str(current_row), col="1")
                
                current_row += 1
            
            # MACD subplot
            if show_indicators and indicators and 'MACD' in indicators:
                fig.add_trace(go.Scatter(
                    x=x_axis,
                    y=indicators['MACD'],
                    line=dict(color=self.colors['macd'], width=2),
                    name='MACD'
                ), row=current_row, col=1)
                
                if 'MACD_Signal' in indicators:
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=indicators['MACD_Signal'],
                        line=dict(color=self.colors['signal'], width=1),
                        name='Signal'
                    ), row=current_row, col=1)
                
                if 'MACD_Histogram' in indicators:
                    colors = ['green' if val >= 0 else 'red' for val in indicators['MACD_Histogram']]
                    fig.add_trace(go.Bar(
                        x=x_axis,
                        y=indicators['MACD_Histogram'],
                        name='MACD Histogram',
                        marker_color=colors,
                        opacity=0.6
                    ), row=current_row, col=1)
                
                fig.add_hline(y=0, line_color="gray", row=str(current_row), col="1")
            
            # Light theme colors
            bg_color = '#FFFFFF'
            text_color = '#262730'
            grid_color = '#E0E0E0'
            
            # Update layout with theme colors
            fig.update_layout(
                title=title,
                xaxis_rangeslider_visible=False,
                showlegend=True,
                height=650 if subplot_count <= 2 else 850,
                margin=dict(l=0, r=0, t=50, b=20),
                font=dict(size=12, color=text_color),
                paper_bgcolor=bg_color,
                plot_bgcolor=bg_color
            )
            
            # Update x-axes with theme colors
            for i in range(1, subplot_count + 1):
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, 
                               tickcolor=text_color, color=text_color, row=i, col=1)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_color,
                               tickcolor=text_color, color=text_color, row=i, col=1)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return self._create_empty_chart(title)
    
    def _create_empty_chart(self, title):
        """Create an empty chart as fallback"""
        fig = go.Figure()
        fig.update_layout(
            title=f"{title} - Error",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            height=600,
            showlegend=False
        )
        fig.add_annotation(
            text="Chart data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        return fig
    
    def create_price_prediction_chart(self, historical_data, prediction_data, confidence_intervals=None):
        """
        Create a chart showing historical prices and future predictions
        
        Args:
            historical_data: pandas.DataFrame with historical OHLCV data
            prediction_data: dict with prediction information
            confidence_intervals: dict with upper and lower bounds
            
        Returns:
            plotly.graph_objects.Figure
        """
        try:
            fig = go.Figure()
            
            # Historical price line
            fig.add_trace(go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            ))
            
            # Prediction point
            if prediction_data:
                fig.add_trace(go.Scatter(
                    x=[prediction_data.get('target_date')],
                    y=[prediction_data.get('predicted_price')],
                    mode='markers',
                    name='Predicted Price',
                    marker=dict(color='red', size=10, symbol='star')
                ))
            
            # Confidence intervals
            if confidence_intervals:
                fig.add_trace(go.Scatter(
                    x=[prediction_data.get('target_date')],
                    y=[confidence_intervals.get('upper')],
                    mode='markers',
                    name='Upper Bound',
                    marker=dict(color='lightcoral', size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=[prediction_data.get('target_date')],
                    y=[confidence_intervals.get('lower')],
                    mode='markers',
                    name='Lower Bound',
                    marker=dict(color='lightcoral', size=8)
                ))
            
            fig.update_layout(
                title="Bitcoin Price Prediction",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                showlegend=True,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating prediction chart: {str(e)}")
            return go.Figure()
    
    def create_probability_visualization(self, probabilities):
        """
        Create visualization for probability assessments
        
        Args:
            probabilities: dict with probability data
            
        Returns:
            plotly.graph_objects.Figure
        """
        try:
            labels = ['Higher', 'Lower']
            values = [probabilities.get('higher', 0), probabilities.get('lower', 0)]
            colors = ['#00ff88', '#ff4444']
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=14
            )])
            
            fig.update_layout(
                title="Probability Assessment - Friday 4PM ET",
                showlegend=True,
                height=300,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating probability visualization: {str(e)}")
            return go.Figure()
