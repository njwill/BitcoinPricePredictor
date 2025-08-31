# ai_analysis_enhanced.py
# Enhanced Bitcoin Technical Analysis with Advanced Trend Detection and Multi-Indicator Analysis

from __future__ import annotations

import os
import json
import re
import contextlib
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytz
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# --- Streamlit-safe import ---
try:
    import streamlit as st
except Exception:
    class _DummySt:
        def write(self, *a, **k):
            print(*a)
        def warning(self, *a, **k):
            print("[warning]", *a)
        def error(self, *a, **k):
            print("[error]", *a)
        def info(self, *a, **k):
            print("[info]", *a)
    st = _DummySt()

# --- OpenAI client wrapper ---
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class _OpenAIWrapper:
    def __init__(self, api_key: Optional[str], model: str, debug: bool = False):
        self.model = model
        self.debug = debug
        self.client = None

        if api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                st.warning(f"Could not initialize OpenAI client: {e}")
                self.client = None
        else:
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY.")
            if OpenAI is None:
                st.error("openai SDK not available in environment.")

    def _extract_text_from_responses(self, resp: Any) -> str:
        # Handle new Responses API
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        # Handle structured output
        try:
            output = getattr(resp, "output", None)
            if output:
                chunks: List[str] = []
                for node in output:
                    content = getattr(node, "content", None)
                    if content:
                        for c in content:
                            t = getattr(c, "text", None)
                            if isinstance(t, str):
                                chunks.append(t)
                if chunks:
                    return "\n".join(chunks).strip()
        except Exception:
            pass

        # Handle chat completions
        try:
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        return content
        except Exception:
            pass

        return ""

    def generate(self, system_msg: str, user_msg: str) -> str:
        if not self.client:
            return ""

        # Try Responses API first
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = self._extract_text_from_responses(resp)
            if text:
                return text
        except Exception as e:
            if self.debug:
                st.warning(f"Responses API failed: {e}")

        # Fallback to Chat Completions
        try:
            chat = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            return self._extract_text_from_responses(chat)
        except Exception as e:
            if self.debug:
                st.error(f"Chat Completions failed: {e}")
            return ""

class TechnicalIndicators:
    """Enhanced technical indicator calculations"""

    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        }

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    @staticmethod
    def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels"""
        diff = high - low
        return {
            'fib_0': high,
            'fib_236': high - diff * 0.236,
            'fib_382': high - diff * 0.382,
            'fib_500': high - diff * 0.500,
            'fib_618': high - diff * 0.618,
            'fib_786': high - diff * 0.786,
            'fib_1000': low
        }

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index for trend strength"""
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = TechnicalIndicators.calculate_atr(high, low, close, 1)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return {
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di
        }

class PatternRecognition:
    """Chart pattern detection algorithms"""

    @staticmethod
    def detect_double_top(prices: pd.Series, window: int = 20, threshold: float = 0.98) -> Optional[Dict[str, Any]]:
        """Detect double top pattern"""
        if len(prices) < window * 2:
            return None

        peaks, properties = find_peaks(prices.values, distance=window//2)
        if len(peaks) < 2:
            return None

        last_two_peaks = peaks[-2:]
        peak1_price = prices.iloc[last_two_peaks[0]]
        peak2_price = prices.iloc[last_two_peaks[1]]

        if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) < (1 - threshold):
            valley = prices.iloc[last_two_peaks[0]:last_two_peaks[1]].min()
            return {
                'pattern': 'double_top',
                'peak1': {'index': last_two_peaks[0], 'price': float(peak1_price)},
                'peak2': {'index': last_two_peaks[1], 'price': float(peak2_price)},
                'neckline': float(valley),
                'target': float(valley - (max(peak1_price, peak2_price) - valley))
            }
        return None

    @staticmethod
    def detect_double_bottom(prices: pd.Series, window: int = 20, threshold: float = 0.98) -> Optional[Dict[str, Any]]:
        """Detect double bottom pattern"""
        if len(prices) < window * 2:
            return None

        troughs, properties = find_peaks(-prices.values, distance=window//2)
        if len(troughs) < 2:
            return None

        last_two_troughs = troughs[-2:]
        trough1_price = prices.iloc[last_two_troughs[0]]
        trough2_price = prices.iloc[last_two_troughs[1]]

        if abs(trough1_price - trough2_price) / max(trough1_price, trough2_price) < (1 - threshold):
            peak = prices.iloc[last_two_troughs[0]:last_two_troughs[1]].max()
            return {
                'pattern': 'double_bottom',
                'trough1': {'index': last_two_troughs[0], 'price': float(trough1_price)},
                'trough2': {'index': last_two_troughs[1], 'price': float(trough2_price)},
                'neckline': float(peak),
                'target': float(peak + (peak - min(trough1_price, trough2_price)))
            }
        return None

    @staticmethod
    def detect_triangle_pattern(prices: pd.Series, min_touches: int = 3) -> Optional[Dict[str, Any]]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(prices) < 20:
            return None

        highs = prices.rolling(window=5).max()
        lows = prices.rolling(window=5).min()

        # Fit trend lines
        x = np.arange(len(prices))
        high_slope, high_intercept, _, _, _ = stats.linregress(x, highs.fillna(method='ffill'))
        low_slope, low_intercept, _, _, _ = stats.linregress(x, lows.fillna(method='ffill'))

        if high_slope < 0 and low_slope > 0:
            pattern_type = 'symmetrical_triangle'
        elif abs(high_slope) < 0.001 and low_slope > 0:
            pattern_type = 'ascending_triangle'
        elif high_slope < 0 and abs(low_slope) < 0.001:
            pattern_type = 'descending_triangle'
        else:
            return None

        return {
            'pattern': pattern_type,
            'upper_trendline': {'slope': float(high_slope), 'intercept': float(high_intercept)},
            'lower_trendline': {'slope': float(low_slope), 'intercept': float(low_intercept)},
            'apex': float((high_intercept - low_intercept) / (low_slope - high_slope)) if low_slope != high_slope else None
        }

class MarketRegimeDetector:
    """Identify current market regime"""

    @staticmethod
    def identify_regime(volatility: float, trend_strength: float, volume_ratio: float) -> str:
        """Classify market regime based on multiple factors"""
        if volatility > 0.03:  # High volatility threshold (3% daily)
            if trend_strength > 25:
                return 'volatile_trending'
            else:
                return 'volatile_ranging'
        elif trend_strength > 30:
            if volume_ratio > 1.2:
                return 'strong_trend_high_volume'
            else:
                return 'strong_trend_normal_volume'
        elif trend_strength < 15:
            return 'ranging_market'
        else:
            return 'weak_trend'

    @staticmethod
    def calculate_regime_metrics(prices: pd.Series, volume: pd.Series, 
                                 adx: pd.Series) -> Dict[str, Any]:
        """Calculate metrics for regime identification"""
        returns = prices.pct_change()
        volatility = returns.std()

        # Trend strength from ADX
        trend_strength = float(adx.iloc[-1]) if not adx.empty else 0

        # Volume ratio (recent vs average)
        recent_volume = volume.tail(5).mean()
        avg_volume = volume.tail(20).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1

        regime = MarketRegimeDetector.identify_regime(volatility, trend_strength, volume_ratio)

        return {
            'regime': regime,
            'volatility': float(volatility),
            'trend_strength': float(trend_strength),
            'volume_ratio': float(volume_ratio)
        }

class RiskAnalyzer:
    """Risk metrics and analysis"""

    @staticmethod
    def calculate_risk_metrics(prices: pd.Series, predicted_price: float, 
                              confidence: float) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        returns = prices.pct_change().dropna()

        # Sharpe Ratio (annualized)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()

        # Risk-adjusted prediction range
        current_price = float(prices.iloc[-1])
        price_std = prices.std()
        prediction_range = {
            'optimistic': predicted_price + price_std * (1 - confidence),
            'base': predicted_price,
            'pessimistic': predicted_price - price_std * (1 - confidence)
        }

        return {
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'prediction_range': prediction_range,
            'risk_score': abs(max_drawdown) * (1 - confidence)  # Higher score = higher risk
        }

class EnhancedAIAnalyzer:
    """Enhanced AI Analyzer with advanced trend detection and multi-indicator analysis"""

    def __init__(self, debug: Optional[bool] = None):
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT5_MODEL", "gpt-5-nano")
        self.gpt = _OpenAIWrapper(self.openai_key, self.model_name, debug=self.debug)
        self._last_current_price: Optional[float] = None

        # Initialize helper classes
        self.indicators = TechnicalIndicators()
        self.patterns = PatternRecognition()
        self.regime_detector = MarketRegimeDetector()
        self.risk_analyzer = RiskAnalyzer()

    def generate_comprehensive_analysis(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
        target_datetime: Optional[datetime] = None,
        asset_name: str = "BTCUSD",
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis with enhanced features"""

        if not self.gpt or not self.gpt.client:
            return {"status": "error", "error": "GPT client not initialized"}

        self._last_current_price = float(current_price)

        try:
            # Enhance indicators with additional calculations
            indicators_3m = self._enhance_indicators(data_3m, indicators_3m)
            indicators_1w = self._enhance_indicators(data_1w, indicators_1w)

            # Prepare analysis data with enhanced features
            analysis_data = self._prepare_enhanced_analysis_data(
                data_3m=data_3m,
                data_1w=data_1w,
                indicators_3m=indicators_3m,
                indicators_1w=indicators_1w,
                current_price=current_price,
                target_datetime=target_datetime,
                asset_name=asset_name,
            )

            target_ts_fallback = analysis_data.get("target_time", "")

            if analysis_data.get("prep_status") == "insufficient_data":
                msg = "; ".join(analysis_data.get("prep_notes", []))
                text_summary, text_pred = self._compose_text_when_insufficient(msg, target_ts_fallback)
                return {
                    "status": "insufficient_data",
                    "model_json": {"status": "insufficient_data", "notes": analysis_data.get("prep_notes", [])},
                    "probabilities": self._default_probs(),
                    "technical_summary": text_summary,
                    "price_prediction": text_pred,
                    "timestamp": datetime.now().isoformat(),
                }

            # Generate AI analysis with enhanced context
            raw = self._generate_enhanced_technical_analysis(analysis_data)

            # Parse outputs
            json_text, narrative_text = self._split_dual_output(raw)
            parsed_json = self._parse_json_response(json_text, self._last_current_price or current_price)

            # Extract sections from narrative
            sections = self._parse_comprehensive_response(narrative_text) if narrative_text else {}

            # Build probabilities with risk adjustment
            if parsed_json.get("status") == "ok":
                probs = self._extract_probabilities_from_json(parsed_json, self._last_current_price or current_price)

                # Adjust probabilities based on risk metrics
                risk_metrics = analysis_data.get("risk_metrics", {})
                if risk_metrics.get("risk_score", 0) > 0.5:
                    # High risk - reduce confidence
                    probs["confidence_fraction"] *= 0.8
                    probs["confidence_pct"] = probs["confidence_fraction"] * 100
            else:
                probs = self._extract_probabilities(sections.get("price_prediction", "") if sections else "")

            # Get text blocks
            tech_md = sections.get("technical_summary") if sections else None
            pred_md = sections.get("price_prediction") if sections else None

            # Synthesize if missing
            if not tech_md or not pred_md:
                synth_tech, synth_pred = self._compose_text_from_model_json(parsed_json, current_price)
                tech_md = tech_md or synth_tech
                pred_md = pred_md or synth_pred

            status = parsed_json.get("status", "ok") if isinstance(parsed_json, dict) else "error"
            if status not in ("ok", "insufficient_data"):
                status = "insufficient_data"

            # Ensure proper target formatting
            if status != "ok" and ("Target:" in pred_md) and not re.search(r"`[^`]+`", pred_md):
                pred_md = f"**Target:** `{target_ts_fallback}`\n\n_No price prediction due to insufficient data._"

            return {
                "status": status,
                "model_json": parsed_json,
                "probabilities": probs,
                "technical_summary": tech_md,
                "price_prediction": pred_md,
                "timestamp": datetime.now().isoformat(),
                "enhanced_metrics": {
                    "patterns": analysis_data.get("patterns", {}),
                    "regime": analysis_data.get("regime", {}),
                    "risk_metrics": analysis_data.get("risk_metrics", {}),
                    "trend_analysis": analysis_data.get("trend_analysis", {})
                }
            }

        except Exception as e:
            st.error(f"Error generating AI analysis: {e}")
            target_ts = target_datetime.isoformat() if target_datetime else ""
            tech, pred = self._compose_text_when_insufficient(str(e), target_ts)
            return {
                "status": "error",
                "error": str(e),
                "probabilities": self._default_probs(),
                "technical_summary": tech,
                "price_prediction": pred,
            }

    def _enhance_indicators(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Add additional technical indicators"""
        if data is None or data.empty:
            return indicators

        enhanced = indicators.copy() if indicators else {}

        try:
            # Add Stochastic
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                stoch = self.indicators.calculate_stochastic(
                    data['High'], data['Low'], data['Close']
                )
                enhanced.update(stoch)

            # Add ATR
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                enhanced['ATR'] = self.indicators.calculate_atr(
                    data['High'], data['Low'], data['Close']
                )

            # Add OBV
            if 'Close' in data.columns and 'Volume' in data.columns:
                enhanced['OBV'] = self.indicators.calculate_obv(
                    data['Close'], data['Volume']
                )

            # Add ADX for trend strength
            if all(col in data.columns for col in ['High', 'Low', 'Close']):
                adx_data = self.indicators.calculate_adx(
                    data['High'], data['Low'], data['Close']
                )
                enhanced.update(adx_data)

        except Exception as e:
            st.warning(f"Error enhancing indicators: {e}")

        return enhanced

    def _prepare_enhanced_analysis_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
        target_datetime: Optional[datetime],
        asset_name: str,
    ) -> Dict[str, Any]:
        """Prepare analysis data with enhanced features"""

        # Get base analysis data
        base_data = self._prepare_analysis_data(
            data_3m, data_1w, indicators_3m, indicators_1w,
            current_price, target_datetime, asset_name
        )

        if base_data.get("prep_status") == "insufficient_data":
            return base_data

        try:
            # Add pattern detection
            patterns = {}
            if data_3m is not None and not data_3m.empty and 'Close' in data_3m.columns:
                patterns['3m'] = {
                    'double_top': self.patterns.detect_double_top(data_3m['Close']),
                    'double_bottom': self.patterns.detect_double_bottom(data_3m['Close']),
                    'triangle': self.patterns.detect_triangle_pattern(data_3m['Close'])
                }

            if data_1w is not None and not data_1w.empty and 'Close' in data_1w.columns:
                patterns['1w'] = {
                    'double_top': self.patterns.detect_double_top(data_1w['Close']),
                    'double_bottom': self.patterns.detect_double_bottom(data_1w['Close']),
                    'triangle': self.patterns.detect_triangle_pattern(data_1w['Close'])
                }

            base_data['patterns'] = patterns

            # Add trend analysis
            trend_analysis = self._analyze_trends(data_3m, data_1w, indicators_3m, indicators_1w)
            base_data['trend_analysis'] = trend_analysis

            # Add market regime detection
            if 'ADX' in indicators_3m and data_3m is not None and not data_3m.empty:
                regime = self.regime_detector.calculate_regime_metrics(
                    data_3m['Close'], data_3m.get('Volume', pd.Series()),
                    indicators_3m['ADX']
                )
                base_data['regime'] = regime

            # Add Fibonacci levels
            if data_3m is not None and not data_3m.empty:
                high_3m = float(data_3m['High'].max())
                low_3m = float(data_3m['Low'].min())
                base_data['fibonacci_3m'] = self.indicators.calculate_fibonacci_levels(high_3m, low_3m)

            if data_1w is not None and not data_1w.empty:
                high_1w = float(data_1w['High'].max())
                low_1w = float(data_1w['Low'].min())
                base_data['fibonacci_1w'] = self.indicators.calculate_fibonacci_levels(high_1w, low_1w)

            # Add correlation analysis
            if indicators_3m and len(indicators_3m) > 1:
                correlation = self._analyze_indicator_correlation(indicators_3m)
                base_data['correlation_3m'] = correlation

            # Add risk metrics (will use predicted price later)
            if data_3m is not None and not data_3m.empty:
                risk_metrics = self.risk_analyzer.calculate_risk_metrics(
                    data_3m['Close'], current_price, 0.7  # Default confidence
                )
                base_data['risk_metrics'] = risk_metrics

        except Exception as e:
            st.warning(f"Error in enhanced analysis data preparation: {e}")

        return base_data

    def _analyze_trends(self, data_3m: pd.DataFrame, data_1w: pd.DataFrame,
                       indicators_3m: Dict[str, pd.Series], indicators_1w: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analyze trends using linear regression and other methods"""
        trends = {}

        def calculate_trend(prices: pd.Series, label: str) -> Dict[str, Any]:
            if prices is None or len(prices) < 2:
                return {}

            x = np.arange(len(prices))
            y = prices.values

            # Remove NaN values
            mask = ~np.isnan(y)
            if mask.sum() < 2:
                return {}

            x_clean = x[mask]
            y_clean = y[mask]

            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)

            # Trend strength (0-100)
            trend_strength = min(abs(r_value) * 100, 100)

            # Trend direction
            if slope > 0:
                direction = 'bullish' if trend_strength > 30 else 'weakly_bullish'
            else:
                direction = 'bearish' if trend_strength > 30 else 'weakly_bearish'

            return {
                'slope': float(slope),
                'strength': float(trend_strength),
                'direction': direction,
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value)
            }

        # Analyze price trends
        if data_3m is not None and not data_3m.empty and 'Close' in data_3m.columns:
            trends['price_3m'] = calculate_trend(data_3m['Close'], '3m_price')

        if data_1w is not None and not data_1w.empty and 'Close' in data_1w.columns:
            trends['price_1w'] = calculate_trend(data_1w['Close'], '1w_price')

        # Analyze RSI trends
        if indicators_3m and 'RSI' in indicators_3m:
            trends['rsi_3m'] = calculate_trend(indicators_3m['RSI'].dropna(), '3m_rsi')

        if indicators_1w and 'RSI' in indicators_1w:
            trends['rsi_1w'] = calculate_trend(indicators_1w['RSI'].dropna(), '1w_rsi')

        # Analyze volume trends
        if data_3m is not None and 'Volume' in data_3m.columns:
            trends['volume_3m'] = calculate_trend(data_3m['Volume'], '3m_volume')

        if data_1w is not None and 'Volume' in data_1w.columns:
            trends['volume_1w'] = calculate_trend(data_1w['Volume'], '1w_volume')

        return trends

    def _analyze_indicator_correlation(self, indicators: Dict[str, pd.Series]) -> Dict[str, float]:
        """Analyze correlation between indicators"""
        correlations = {}

        try:
            # Create DataFrame from indicators
            df_indicators = pd.DataFrame()
            for name, series in indicators.items():
                if isinstance(series, pd.Series) and not series.empty:
                    df_indicators[name] = series

            if len(df_indicators.columns) > 1:
                corr_matrix = df_indicators.corr()

                # Extract key correlations
                if 'RSI' in corr_matrix.columns and 'MACD' in corr_matrix.columns:
                    correlations['rsi_macd'] = float(corr_matrix.loc['RSI', 'MACD'])

                if 'Volume' in corr_matrix.columns and 'Close' in corr_matrix.columns:
                    correlations['volume_price'] = float(corr_matrix.loc['Volume', 'Close'])

                # Overall correlation strength (average absolute correlation)
                mask = np.triu(np.ones_like(corr_matrix), k=1)
                masked_corr = corr_matrix.where(mask.astype(bool))
                correlations['avg_correlation'] = float(masked_corr.abs().mean().mean())

        except Exception as e:
            st.warning(f"Error calculating correlations: {e}")

        return correlations

    def _generate_enhanced_technical_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Generate technical analysis with enhanced prompting"""
        asset_name = analysis_data.get("asset_name", "Asset")

        # Build enhanced prompt with additional context
        system_msg, user_msg = self._build_enhanced_messages(analysis_data, asset_name)

        if self.debug:
            print("\n" + "=" * 80)
            print("🤖 ENHANCED PROMPT SENT TO MODEL")
            print("=" * 80)
            print("\n📋 SYSTEM MESSAGE:\n" + system_msg["content"])
            print("\n📊 USER MESSAGE (partial):\n" + user_msg["content"][:2000] + "...")
            print("\n" + "=" * 80)

        raw = self.gpt.generate(system_msg["content"], user_msg["content"]) or ""

        if not raw.strip():
            return json.dumps({"status": "insufficient_data", "notes": ["empty_model_response"]})

        return raw

    def _build_enhanced_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build enhanced messages with additional context"""

        system_content = (
            "You are an advanced technical analyst with expertise in multi-timeframe analysis, pattern recognition, and risk assessment. "
            "Analyze ALL provided data including: price arrays, enhanced indicators (RSI, MACD, BB, EMA, Stochastic, ATR, OBV, ADX), "
            "detected patterns, trend analysis, market regime, Fibonacci levels, correlations, and risk metrics. "
            "Compare 1-week and 3-month timeframes comprehensively, identifying confirmations and divergences. "
            "Use the trend strength metrics and pattern detection results to support your analysis. "
            "Consider the market regime when making predictions. "
            "All claims must be quantitatively verifiable from the provided data. "
            "Forbidden: news, fundamentals, macro events, or any external information not in the data. "
            "If data is insufficient, return status='insufficient_data' with specific notes."
        )

        # Prepare comprehensive data summary
        data_3m = analysis_data.get("data_3m", {})
        data_1w = analysis_data.get("data_1w", {})

        # Enhanced output schema
        output_schema = {
            "status": "ok or insufficient_data",
            "asset": asset_name,
            "as_of": analysis_data.get("current_time"),
            "target_ts": analysis_data.get("target_time"),
            "predicted_price": "number or null",
            "p_up": "float 0..1",
            "p_down": "float 0..1",
            "conf_overall": "float 0..1",
            "conf_price": "float 0..1",
            "expected_pct_move": "percent",
            "critical_levels": {
                "bullish_above": "number",
                "bearish_below": "number",
                "support_levels": ["list of support levels"],
                "resistance_levels": ["list of resistance levels"]
            },
            "evidence": [
                {
                    "type": "indicator/pattern/trend/regime",
                    "timeframe": "3m/1w",
                    "value": "specific value",
                    "note": "interpretation"
                }
            ],
            "risk_assessment": {
                "risk_level": "low/medium/high",
                "key_risks": ["list of identified risks"],
                "stop_loss": "suggested stop loss level",
                "take_profit": ["TP1", "TP2", "TP3"]
            },
            "notes": ["any warnings or caveats"]
        }

        # Enhanced narrative template
        narrative_template = f"""
        [TECHNICAL_ANALYSIS_START]
        **COMPREHENSIVE TECHNICAL ANALYSIS**
        **Current Price: ${analysis_data.get('current_price'):,.2f}**

        **1. MULTI-TIMEFRAME OVERVIEW**
        - 3-Month Trend Analysis: <slope, strength, direction from trend_analysis>
        - 1-Week Trend Analysis: <slope, strength, direction from trend_analysis>
        - Timeframe Alignment: <convergence or divergence between timeframes>
        - Market Regime: <from regime detection>

        **2. ENHANCED TECHNICAL INDICATORS**
        - RSI Analysis: <levels, trends, divergences from both timeframes>
        - MACD Analysis: <crossovers, histogram trends, momentum>
        - Bollinger Bands: <width, position, squeeze detection>
        - Stochastic: <overbought/oversold conditions>
        - ADX: <trend strength quantification>
        - OBV: <volume-price relationship>
        - ATR: <volatility assessment>

        **3. PATTERN RECOGNITION**
        - Detected Patterns: <from pattern detection algorithms>
        - Support/Resistance: <from Fibonacci and price structure>
        - Chart Formation: <triangles, channels, etc.>

        **4. CORRELATION & DIVERGENCE ANALYSIS**
        - Indicator Correlations: <from correlation analysis>
        - Price-Volume Divergence: <if detected>
        - Multi-Timeframe Divergences: <between 1w and 3m>

        **5. RISK ASSESSMENT**
        - Current Risk Level: <from risk metrics>
        - Sharpe Ratio: <value and interpretation>
        - Maximum Drawdown: <historical context>
        - VaR (95%): <downside risk>

        **6. TRADING RECOMMENDATION**
        - Overall Bias: **BULLISH/BEARISH/NEUTRAL**
        - Entry Zone: <specific levels>
        - Stop Loss: <risk-based calculation>
        - Take Profit Targets: <based on Fibonacci and resistance>
        - Risk/Reward Ratio: <calculated value>

        [TECHNICAL_ANALYSIS_END]

        [PRICE_PREDICTION_START]
        **PREDICTED PRICE: ${asset_name} will be at $<PRICE> on {analysis_data.get('target_time')}**

        ⏰ **ENHANCED TIME-BASED ANALYSIS:**
        - Target Timestamp: {analysis_data.get('target_time')}
        - Hours Until Target: {analysis_data.get('hours_until_target', 0):.1f}
        - Trend Projection: <based on regression analysis>
        - Momentum Assessment: <from ADX and MACD>
        - Volatility Forecast: <from ATR trends>

        **PROBABILITY DISTRIBUTION:**
        1. **Probability HIGHER than ${analysis_data.get('current_price'):,.2f}: <X>%**
        2. **Probability LOWER than ${analysis_data.get('current_price'):,.2f}: <Y>%**
        3. **Overall Analysis Confidence: <Z>%**
        4. **Price Prediction Confidence: <W>%**
        5. **Expected % Move: <±M>%**

        **RISK-ADJUSTED PREDICTION RANGE:**
        - Optimistic Target: $<amount> (<X>% probability)
        - Base Case Target: $<amount> (<Y>% probability)
        - Pessimistic Target: $<amount> (<Z>% probability)

        **KEY TECHNICAL DRIVERS:**
        - Primary Driver: <most significant factor>
        - Secondary Driver: <supporting factor>
        - Risk Factor: <main risk to prediction>

        **CRITICAL LEVELS TO WATCH:**
        - Immediate Resistance: $<level>
        - Immediate Support: $<level>
        - Breakout Level: $<level>
        - Breakdown Level: $<level>

        [PRICE_PREDICTION_END]
        """.strip()

        # Compile all enhanced data
        user_content = f"""
        ASSET: {asset_name}
        CURRENT_PRICE: {analysis_data.get('current_price')}
        TARGET_TIME: {analysis_data.get('target_time')}
        AS_OF: {analysis_data.get('current_time')}

        DATA WINDOWS:
        - 3m range: {data_3m.get('start_date','N/A')} → {data_3m.get('end_date','N/A')}
        - 1w range: {data_1w.get('start_date','N/A')} → {data_1w.get('end_date','N/A')}

        TREND ANALYSIS:
        {json.dumps(analysis_data.get('trend_analysis', {}), indent=2)}

        PATTERN DETECTION:
        {json.dumps(analysis_data.get('patterns', {}), indent=2)}

        MARKET REGIME:
        {json.dumps(analysis_data.get('regime', {}), indent=2)}

        FIBONACCI LEVELS:
        3M: {json.dumps(analysis_data.get('fibonacci_3m', {}), indent=2)}
        1W: {json.dumps(analysis_data.get('fibonacci_1w', {}), indent=2)}

        CORRELATIONS:
        {json.dumps(analysis_data.get('correlation_3m', {}), indent=2)}

        RISK METRICS:
        {json.dumps(analysis_data.get('risk_metrics', {}), indent=2)}

        INDICATOR SNAPSHOT:
        {json.dumps(analysis_data.get('indicators', {}), indent=2)}

        FEATURES:
        {json.dumps(analysis_data.get('features', {}), indent=2)}

        ENHANCED ARRAYS:
        {json.dumps(analysis_data.get('enhanced_chart_data', {}), indent=2)}

        INSTRUCTIONS:
        1. Analyze ALL provided metrics, especially trend analysis and pattern detection results
        2. Weight predictions based on market regime and risk metrics
        3. Use correlation data to validate signals
        4. Return BOTH a JSON object and narrative sections as specified

        JSON FORMAT:
        ```json
        {json.dumps(output_schema, indent=2)}
        ```

        NARRATIVE FORMAT:
        {narrative_template}
        """

        return ({"role": "system", "content": system_content}, {"role": "user", "content": user_content})

    # Include all the helper methods from the original script
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has datetime index"""
        if df is None or df.empty:
            return df

        df = df.copy()

        try:
            if getattr(df.index, "dtype", None) is not None and getattr(df.index, "dtype", object).kind in ("i", "f"):
                df.reset_index(drop=True, inplace=True)
                date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
                if date_cols:
                    df.index = pd.to_datetime(df[date_cols[0]], errors="coerce")
                    with contextlib.suppress(Exception):
                        df.drop(columns=[date_cols[0]], inplace=True)
                else:
                    df.index = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
        except Exception:
            pass

        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            df.index = pd.to_datetime(df.index, errors="coerce")

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df

    def _coerce_ohlcv_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert OHLCV columns to numeric"""
        if df is None or df.empty:
            return df

        df = df.copy()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        return df

    def _limit_to_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Limit DataFrame to specified number of days"""
        if df is None or df.empty:
            return df

        df = self._ensure_datetime_index(df)
        try:
            end = pd.Timestamp(df.index.max()) if pd.notna(df.index.max()) else pd.Timestamp("now")
        except Exception:
            end = pd.Timestamp("now")

        start = end - pd.Timedelta(days=days)
        return df.loc[df.index >= start]

    def _annualization_sqrt(self, index: pd.Index) -> float:
        """Calculate annualization factor"""
        try:
            if len(index) < 2:
                return 1.0

            dt = index[1] - index[0]
            seconds = float(getattr(dt, "total_seconds", lambda: float(dt))())
            if seconds <= 0:
                return 1.0

            periods_per_year = (365 * 24 * 3600) / seconds
            return float(np.sqrt(periods_per_year))
        except Exception:
            return 1.0

    def _infer_index_step(self, index: pd.Index) -> Optional[pd.Timedelta]:
        """Infer time step from index"""
        if index is None or len(index) < 2:
            return None

        try:
            diffs = pd.Series(index[1:]).reset_index(drop=True) - pd.Series(index[:-1]).reset_index(drop=True)
            diffs = diffs[diffs > pd.Timedelta(0)]
            if diffs.empty:
                return None

            return pd.to_timedelta(np.median(diffs.values))
        except Exception:
            return None

    def _determine_timezone(self, *indexes: Optional[pd.Index]):
        """Determine timezone from indexes"""
        for idx in indexes:
            if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
                return idx.tz
        return pytz.timezone("US/Eastern")

    def _prepare_analysis_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
        target_datetime: Optional[datetime],
        asset_name: str,
    ) -> Dict[str, Any]:
        """Prepare analysis data for AI processing"""
        try:
            # Basic data preparation
            analysis_data = {
                "asset": asset_name or "BTCUSD",
                "current_price": current_price,
                "target_datetime": target_datetime.isoformat() if target_datetime else None,
                "data_3m_length": len(data_3m) if data_3m is not None else 0,
                "data_1w_length": len(data_1w) if data_1w is not None else 0,
                "indicators_3m": indicators_3m or {},
                "indicators_1w": indicators_1w or {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Add latest indicator values
            if indicators_3m:
                analysis_data["latest_indicators_3m"] = {
                    k: float(v.iloc[-1]) if len(v) > 0 and pd.notna(v.iloc[-1]) else None
                    for k, v in indicators_3m.items()
                }
            
            if indicators_1w:
                analysis_data["latest_indicators_1w"] = {
                    k: float(v.iloc[-1]) if len(v) > 0 and pd.notna(v.iloc[-1]) else None
                    for k, v in indicators_1w.items()
                }
                
            return analysis_data
            
        except Exception as e:
            st.warning(f"Error preparing analysis data: {e}")
            return {
                "asset": asset_name or "BTCUSD",
                "current_price": current_price,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _compose_text_when_insufficient(self, error_msg: str, target_ts: str) -> Tuple[str, str]:
        """Compose fallback text when analysis fails"""
        technical_summary = f"""**TECHNICAL ANALYSIS UNAVAILABLE**

Due to insufficient data or technical issues, a comprehensive analysis cannot be generated at this time.

**Error Details:** {error_msg}

Please try again later or check data availability."""

        price_prediction = f"""**PRICE PREDICTION UNAVAILABLE**

**Target:** `{target_ts}`

Unable to generate price prediction due to data limitations or technical issues.

**Confidence:** Low - Analysis not available"""
        
        return technical_summary, price_prediction

    def _default_probs(self) -> Dict[str, float]:
        """Return default probability values"""
        return {
            "higher_fraction": 0.5,
            "lower_fraction": 0.5,
            "confidence_fraction": 0.3,
            "higher_pct": 50.0,
            "lower_pct": 50.0,
            "confidence_pct": 30.0,
            "predicted_price": 0.0,
            "price_confidence_pct": 30.0,
            "move_percentage": 0.0
        }

    def _compose_text_from_model_json(self, parsed_json: Dict[str, Any], current_price: float) -> Tuple[str, str]:
        """Compose text from parsed JSON response"""
        try:
            if not isinstance(parsed_json, dict):
                return self._compose_text_when_insufficient("Invalid JSON format", "")
            
            model_json = parsed_json.get("model_json", {})
            if not model_json:
                return self._compose_text_when_insufficient("No model data available", "")
            
            # Extract basic info
            predicted_price = model_json.get("predicted_price", current_price)
            target_ts = model_json.get("target_ts", "")
            
            # Create technical summary
            technical_summary = "**TECHNICAL ANALYSIS**\n\n"
            technical_summary += f"**Current Price:** ${current_price:,.2f}\n\n"
            
            if "evidence" in model_json:
                technical_summary += "**Key Technical Indicators:**\n"
                for evidence in model_json["evidence"]:
                    note = evidence.get("note", "")
                    if note:
                        technical_summary += f"• {note}\n"
            
            # Create price prediction
            price_prediction = f"**PREDICTED PRICE:** ${predicted_price:,.2f}\n\n"
            price_prediction += f"**Target:** `{target_ts}`\n\n"
            
            p_up = model_json.get("p_up", 0.5)
            p_down = model_json.get("p_down", 0.5)
            price_prediction += f"**Probability Assessment:**\n"
            price_prediction += f"• Higher: {p_up*100:.0f}%\n"
            price_prediction += f"• Lower: {p_down*100:.0f}%\n"
            
            return technical_summary, price_prediction
            
        except Exception as e:
            return self._compose_text_when_insufficient(f"Error composing text: {str(e)}", "")

# Usage example
if __name__ == "__main__":
    analyzer = EnhancedAIAnalyzer(debug=True)
    # Your data loading and analysis call here
