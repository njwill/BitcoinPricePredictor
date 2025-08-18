import pandas as pd
import numpy as np
import talib
import streamlit as st
from typing import Dict, Optional

class TechnicalAnalyzer:
    """Calculates technical indicators for Bitcoin price analysis"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict:
        """
        Calculate all technical indicators for the given price data
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Dictionary containing all calculated indicators
        """
        try:
            if data.empty or len(data) < 50:
                st.warning("Insufficient data for technical analysis")
                return {}
            
            indicators = {}
            
            # Convert to numpy arrays for talib with proper data types
            high = data['High'].values.astype(np.float64)
            low = data['Low'].values.astype(np.float64)
            close = data['Close'].values.astype(np.float64)
            volume = data['Volume'].values.astype(np.float64)
            
            # Moving Averages
            indicators.update(self._calculate_moving_averages(close))
            
            # Bollinger Bands
            indicators.update(self._calculate_bollinger_bands(close))
            
            # RSI
            indicators.update(self._calculate_rsi(close))
            
            # MACD
            indicators.update(self._calculate_macd(close))
            
            # Stochastic Oscillator
            indicators.update(self._calculate_stochastic(high, low, close))
            
            # Average True Range (ATR)
            indicators.update(self._calculate_atr(high, low, close))
            
            # Williams %R
            indicators.update(self._calculate_williams_r(high, low, close))
            
            # Commodity Channel Index (CCI)
            indicators.update(self._calculate_cci(high, low, close))
            
            # Volume indicators
            indicators.update(self._calculate_volume_indicators(high, low, close, volume))
            
            # Support and Resistance levels
            indicators.update(self._calculate_support_resistance(high, low, close))
            
            # Convert numpy arrays back to pandas Series with proper index
            for key, value in indicators.items():
                if isinstance(value, np.ndarray):
                    indicators[key] = pd.Series(value, index=data.index)
            
            return indicators
            
        except Exception as e:
            st.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def _calculate_moving_averages(self, close: np.ndarray) -> Dict:
        """Calculate various moving averages"""
        indicators = {}
        
        try:
            # Ensure proper data type
            close_float = close.astype(np.float64)
            
            # Simple Moving Averages
            for period in [10, 20, 50, 200]:
                if len(close_float) >= period:
                    indicators[f'SMA_{period}'] = talib.SMA(close_float, timeperiod=period)
            
            # Exponential Moving Averages
            for period in [12, 20, 26, 50]:
                if len(close_float) >= period:
                    indicators[f'EMA_{period}'] = talib.EMA(close_float, timeperiod=period)
            
            # Weighted Moving Average
            if len(close_float) >= 14:
                indicators['WMA_14'] = talib.WMA(close_float, timeperiod=14)
            
        except Exception as e:
            st.warning(f"Error calculating moving averages: {str(e)}")
        
        return indicators
    
    def _calculate_bollinger_bands(self, close: np.ndarray) -> Dict:
        """Calculate Bollinger Bands"""
        indicators = {}
        
        try:
            # Ensure proper data type
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close_float, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
                )
                indicators['BB_Upper'] = bb_upper
                indicators['BB_Middle'] = bb_middle
                indicators['BB_Lower'] = bb_lower
                
                # Bollinger Band Width
                indicators['BB_Width'] = (bb_upper - bb_lower) / bb_middle
                
                # Bollinger Band %B
                indicators['BB_PercentB'] = (close_float - bb_lower) / (bb_upper - bb_lower)
                
        except Exception as e:
            st.warning(f"Error calculating Bollinger Bands: {str(e)}")
        
        return indicators
    
    def _calculate_rsi(self, close: np.ndarray) -> Dict:
        """Calculate Relative Strength Index"""
        indicators = {}
        
        try:
            # Ensure proper data type
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 14:
                indicators['RSI'] = talib.RSI(close_float, timeperiod=14)
                
                # RSI with different periods
                if len(close_float) >= 21:
                    indicators['RSI_21'] = talib.RSI(close_float, timeperiod=21)
                
        except Exception as e:
            st.warning(f"Error calculating RSI: {str(e)}")
        
        return indicators
    
    def _calculate_macd(self, close: np.ndarray) -> Dict:
        """Calculate MACD indicators"""
        indicators = {}
        
        try:
            # Ensure proper data type
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 34:  # Need at least 34 periods for MACD
                macd, macd_signal, macd_histogram = talib.MACD(
                    close_float, fastperiod=12, slowperiod=26, signalperiod=9
                )
                indicators['MACD'] = macd
                indicators['MACD_Signal'] = macd_signal
                indicators['MACD_Histogram'] = macd_histogram
                
        except Exception as e:
            st.warning(f"Error calculating MACD: {str(e)}")
        
        return indicators
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Calculate Stochastic Oscillator"""
        indicators = {}
        
        try:
            # Ensure proper data types
            high_float = high.astype(np.float64)
            low_float = low.astype(np.float64)
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 14:
                slowk, slowd = talib.STOCH(
                    high_float, low_float, close_float,
                    fastk_period=14, slowk_period=3, slowk_matype=0,
                    slowd_period=3, slowd_matype=0
                )
                indicators['STOCH_K'] = slowk
                indicators['STOCH_D'] = slowd
                
        except Exception as e:
            st.warning(f"Error calculating Stochastic: {str(e)}")
        
        return indicators
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Calculate Average True Range"""
        indicators = {}
        
        try:
            # Ensure proper data types
            high_float = high.astype(np.float64)
            low_float = low.astype(np.float64)
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 14:
                indicators['ATR'] = talib.ATR(high_float, low_float, close_float, timeperiod=14)
                
        except Exception as e:
            st.warning(f"Error calculating ATR: {str(e)}")
        
        return indicators
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Calculate Williams %R"""
        indicators = {}
        
        try:
            # Ensure proper data types
            high_float = high.astype(np.float64)
            low_float = low.astype(np.float64)
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 14:
                indicators['WILLR'] = talib.WILLR(high_float, low_float, close_float, timeperiod=14)
                
        except Exception as e:
            st.warning(f"Error calculating Williams %R: {str(e)}")
        
        return indicators
    
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Calculate Commodity Channel Index"""
        indicators = {}
        
        try:
            # Ensure proper data types
            high_float = high.astype(np.float64)
            low_float = low.astype(np.float64)
            close_float = close.astype(np.float64)
            
            if len(close_float) >= 14:
                indicators['CCI'] = talib.CCI(high_float, low_float, close_float, timeperiod=14)
                
        except Exception as e:
            st.warning(f"Error calculating CCI: {str(e)}")
        
        return indicators
    
    def _calculate_volume_indicators(self, high: np.ndarray, low: np.ndarray, 
                                   close: np.ndarray, volume: np.ndarray) -> Dict:
        """Calculate volume-based indicators"""
        indicators = {}
        
        try:
            if len(close) >= 20:
                # Ensure data types are correct for TA-Lib
                close_float = close.astype(np.float64)
                volume_float = volume.astype(np.float64)
                
                # On Balance Volume
                indicators['OBV'] = talib.OBV(close_float, volume_float)
                
                # Volume SMA
                indicators['Volume_SMA'] = talib.SMA(volume_float, timeperiod=20)
                
                # Price Volume Trend
                if len(close) >= 1:
                    pct_change = np.zeros(len(close), dtype=np.float64)
                    pct_change[1:] = (close_float[1:] - close_float[:-1]) / close_float[:-1]
                    indicators['PVT'] = np.cumsum(volume_float * pct_change)
                
        except Exception as e:
            st.warning(f"Error calculating volume indicators: {str(e)}")
        
        return indicators
    
    def _calculate_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Dict:
        """Calculate dynamic support and resistance levels"""
        indicators = {}
        
        try:
            if len(close) >= 20:
                # Simple support/resistance based on recent highs and lows
                window = min(20, len(close))
                recent_high = np.max(high[-window:])
                recent_low = np.min(low[-window:])
                
                indicators['Resistance'] = np.full(len(close), recent_high)
                indicators['Support'] = np.full(len(close), recent_low)
                
                # Pivot points (simplified)
                if len(close) >= 3:
                    typical_price = (high + low + close) / 3
                    pivot = typical_price[-1]
                    resistance1 = 2 * pivot - low[-1]
                    support1 = 2 * pivot - high[-1]
                    
                    indicators['Pivot'] = np.full(len(close), pivot)
                    indicators['R1'] = np.full(len(close), resistance1)
                    indicators['S1'] = np.full(len(close), support1)
                
        except Exception as e:
            st.warning(f"Error calculating support/resistance: {str(e)}")
        
        return indicators
    
    def get_signal_summary(self, indicators: Dict, current_price: float) -> Dict:
        """
        Generate trading signal summary based on technical indicators
        
        Args:
            indicators: Dictionary of calculated indicators
            current_price: Current Bitcoin price
            
        Returns:
            Dictionary with signal analysis
        """
        signals = {
            'overall_signal': 'NEUTRAL',
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'strength': 0
        }
        
        try:
            bullish_count = 0
            bearish_count = 0
            total_signals = 0
            
            # RSI Analysis
            if 'RSI' in indicators:
                rsi_current = indicators['RSI'].iloc[-1]
                if not np.isnan(rsi_current):
                    total_signals += 1
                    if rsi_current < 30:
                        signals['bullish_signals'].append(f"RSI oversold ({rsi_current:.1f})")
                        bullish_count += 1
                    elif rsi_current > 70:
                        signals['bearish_signals'].append(f"RSI overbought ({rsi_current:.1f})")
                        bearish_count += 1
                    else:
                        signals['neutral_signals'].append(f"RSI neutral ({rsi_current:.1f})")
            
            # MACD Analysis
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                macd_current = indicators['MACD'].iloc[-1]
                signal_current = indicators['MACD_Signal'].iloc[-1]
                if not (np.isnan(macd_current) or np.isnan(signal_current)):
                    total_signals += 1
                    if macd_current > signal_current:
                        signals['bullish_signals'].append("MACD above signal line")
                        bullish_count += 1
                    else:
                        signals['bearish_signals'].append("MACD below signal line")
                        bearish_count += 1
            
            # Bollinger Bands Analysis
            if all(key in indicators for key in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                bb_upper = indicators['BB_Upper'].iloc[-1]
                bb_lower = indicators['BB_Lower'].iloc[-1]
                bb_middle = indicators['BB_Middle'].iloc[-1]
                
                if not any(np.isnan([bb_upper, bb_lower, bb_middle])):
                    total_signals += 1
                    if current_price < bb_lower:
                        signals['bullish_signals'].append("Price below Bollinger Band lower")
                        bullish_count += 1
                    elif current_price > bb_upper:
                        signals['bearish_signals'].append("Price above Bollinger Band upper")
                        bearish_count += 1
                    else:
                        signals['neutral_signals'].append("Price within Bollinger Bands")
            
            # EMA Analysis
            if 'EMA_20' in indicators:
                ema_20 = indicators['EMA_20'].iloc[-1]
                if not np.isnan(ema_20):
                    total_signals += 1
                    if current_price > ema_20:
                        signals['bullish_signals'].append("Price above EMA 20")
                        bullish_count += 1
                    else:
                        signals['bearish_signals'].append("Price below EMA 20")
                        bearish_count += 1
            
            # Overall signal determination
            if total_signals > 0:
                bullish_ratio = bullish_count / total_signals
                bearish_ratio = bearish_count / total_signals
                
                if bullish_ratio >= 0.6:
                    signals['overall_signal'] = 'BULLISH'
                    signals['strength'] = bullish_ratio
                elif bearish_ratio >= 0.6:
                    signals['overall_signal'] = 'BEARISH'
                    signals['strength'] = bearish_ratio
                else:
                    signals['overall_signal'] = 'NEUTRAL'
                    signals['strength'] = 0.5
            
        except Exception as e:
            st.warning(f"Error generating signal summary: {str(e)}")
        
        return signals
