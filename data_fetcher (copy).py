import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import requests
import json

class BitcoinDataFetcher:
    """Handles fetching Bitcoin price data from various sources"""
    
    def __init__(self):
        self.btc_ticker = "BTC-USD"
        
    def get_bitcoin_data(self, period='3mo', interval='1d', extended_for_indicators=True):
        """
        Fetch Bitcoin price data using yfinance
        
        Args:
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            extended_for_indicators: If True, fetch extra data for technical indicator warm-up periods
        
        Returns:
            pandas.DataFrame: Bitcoin price data with OHLCV columns
        """
        try:
            # Create ticker object
            btc = yf.Ticker(self.btc_ticker)
            
            # Fetch historical data with extension for technical indicators
            if period == '1wk':
                # For 1 week data, extend to get more hourly data for indicators
                if extended_for_indicators:
                    # Fetch 14 days of hourly data, then trim to last 7 days for display
                    data = btc.history(period='14d', interval='1h')
                    if not data.empty and len(data) > 168:  # 7 days * 24 hours
                        # Keep the extended data for indicators, but mark the display range
                        data.attrs['display_from_index'] = len(data) - 168
                else:
                    data = btc.history(period='7d', interval='1h')
            elif period == '3mo':
                if extended_for_indicators:
                    # Fetch 6 months of daily data, then trim to last 3 months for display  
                    data = btc.history(period='6mo', interval='1d')
                    if not data.empty and len(data) > 90:  # Approximate 3 months
                        # Keep the extended data for indicators, but mark the display range
                        data.attrs['display_from_index'] = len(data) - 90
                else:
                    data = btc.history(period=period, interval=interval)
            else:
                data = btc.history(period=period, interval=interval)
            
            if data.empty:
                st.error("No data received from Yahoo Finance")
                return pd.DataFrame()
            
            # Clean and prepare data
            data = data.dropna()
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    st.error(f"Missing required column: {col}")
                    return pd.DataFrame()
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename datetime column if needed
            if 'Date' in data.columns:
                data = data.rename(columns={'Date': 'Datetime'})
            elif 'Datetime' not in data.columns and data.index.name in ['Date', 'Datetime']:
                data = data.reset_index()
                if len(data.columns) > 0:
                    data = data.rename(columns={data.columns[0]: 'Datetime'})
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching Bitcoin data: {str(e)}")
            return pd.DataFrame()
    
    def get_market_events(self):
        """
        Fetch upcoming market events that might impact Bitcoin
        This is a placeholder for integrating with financial news APIs
        """
        try:
            # This would integrate with financial news APIs like Alpha Vantage, NewsAPI, etc.
            # For now, return some common recurring events
            events = [
                {
                    'date': datetime.now() + timedelta(days=1),
                    'event': 'Federal Reserve Meeting',
                    'impact': 'High',
                    'description': 'Monetary policy decisions can significantly impact cryptocurrency markets'
                },
                {
                    'date': datetime.now() + timedelta(days=3),
                    'event': 'Bitcoin Futures Expiry',
                    'impact': 'Medium',
                    'description': 'Monthly futures expiry can cause increased volatility'
                },
                {
                    'date': datetime.now() + timedelta(days=5),
                    'event': 'Weekly Options Expiry',
                    'impact': 'Medium',
                    'description': 'Weekly options expiry typically occurs on Fridays'
                }
            ]
            
            return events
            
        except Exception as e:
            st.error(f"Error fetching market events: {str(e)}")
            return []
    
    def get_bitcoin_fundamentals(self):
        """
        Fetch Bitcoin fundamental data like market cap, active addresses, etc.
        """
        try:
            btc = yf.Ticker(self.btc_ticker)
            info = btc.info
            
            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'volume_24h': info.get('volume24Hr', 0),
                'circulating_supply': info.get('circulatingSupply', 0),
                'max_supply': info.get('maxSupply', 21000000),
                'price_change_24h': info.get('regularMarketChangePercent', 0)
            }
            
            return fundamentals
            
        except Exception as e:
            st.error(f"Error fetching Bitcoin fundamentals: {str(e)}")
            return {}
    
    def validate_data(self, data):
        """
        Validate the fetched data for completeness and accuracy
        
        Args:
            data: pandas.DataFrame with Bitcoin price data
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for reasonable price ranges (Bitcoin should be > $1000 and < $1000000)
        if data['Close'].min() < 1000 or data['Close'].max() > 1000000:
            st.warning("Bitcoin price data seems unusual - please verify")
        
        # Check for missing data
        missing_data = data[required_columns].isnull().sum().sum()
        if missing_data > 0:
            st.warning(f"Found {missing_data} missing data points")
        
        return True
    
    def get_bitcoin_fear_greed_index(self):
        """
        Fetch Bitcoin Fear & Greed Index from Alternative.me API
        """
        try:
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    latest = data['data'][0]
                    return {
                        'value': int(latest['value']),
                        'classification': latest['value_classification'],
                        'timestamp': latest['timestamp']
                    }
            
            return None
            
        except Exception as e:
            st.warning(f"Could not fetch Fear & Greed Index: {str(e)}")
            return None
