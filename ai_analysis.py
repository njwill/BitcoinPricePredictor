import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Dict, Any, Optional
from anthropic import Anthropic


class AIAnalyzer:
    """
    Handles AI-powered analysis using Anthropic Claude.

    Expects:
      - data_3m, data_1w: pandas.DataFrame with columns ['Open','High','Low','Close','Volume'].
        Index should be datetime-like (will be coerced if needed) and sorted ascending.
        These should represent the "3-month" and "1-week" datasets respectively; if they
        contain more than those windows, this class will trim them internally.
        Optional: df.attrs['display_from_index'] to indicate a trimmed display window (for RECENT arrays only).
      - indicators_3m, indicators_1w: dict-like of pandas.Series for technical indicators
        (e.g., 'RSI','MACD','MACD_Signal','BB_Upper','BB_Lower','BB_Middle','EMA_20','SMA_50','SMA_200').

    Config via environment:
      - ANTHROPIC_API_KEY: required
      - ANTHROPIC_MODEL: optional (default: "claude-sonnet-4-20250514")
      - AI_ANALYZER_DEBUG: "1" to enable verbose Streamlit debug boxes
    """

    def __init__(self, debug: Optional[bool] = None):
        # Debug flag (UI messages gated to avoid spamming users)
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        # Initialize Anthropic (Claude) for technical analysis
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

        if not self.anthropic_key:
            st.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
            self.claude_client = None
        else:
            self.claude_client = Anthropic(api_key=self.anthropic_key)

    # ---------- public API ----------

    def generate_comprehensive_analysis(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
        target_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI analysis including technical analysis and price prediction.

        Args:
            data_3m: 3-month Bitcoin price data (likely daily bars or similar)
            data_1w: 1-week Bitcoin price data (likely hourly bars)
            indicators_3m: Technical indicators for 3-month data
            indicators_1w: Technical indicators for 1-week data
            current_price: Current Bitcoin price (float)
            target_datetime: Optional custom target datetime for prediction (ET). Defaults to next Friday 4PM ET.

        Returns:
            Dict with keys: technical_summary, price_prediction, probabilities, timestamp
        """
        if not self.claude_client:
            return {"error": "Claude client not initialized"}

        try:
            analysis_data = self._prepare_analysis_data(
                data_3m, data_1w, indicators_3m, indicators_1w, current_price, target_datetime
            )

            comprehensive_response = self._generate_technical_analysis_claude(analysis_data)
            parsed = self._parse_comprehensive_response(comprehensive_response)
            probs = self._extract_probabilities(parsed.get("price_prediction", ""))

            return {
                "technical_summary": parsed.get("technical_summary", "Technical analysis not available"),
                "price_prediction": parsed.get("price_prediction", "Price prediction not available"),
                "probabilities": probs,  # contains both fractions and percents
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            st.error(f"Error generating AI analysis: {str(e)}")
            return {"error": str(e)}

    # ---------- helpers ----------

    def _dbg(self, level: str, msg: str):
        """Conditional Streamlit UI debug logs."""
        if not self.debug:
            return
        fn = getattr(st, level, None)
        (fn or st.write)(msg)

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerce index to datetime if needed and sort ascending."""
        if df.empty:
            return df
        if not hasattr(df.index, "inferred_type") or "date" not in str(df.index.inferred_type):
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df

    def _coerce_ohlcv_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV are numeric (handle strings with commas)."""
        if df.empty:
            return df
        df = df.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        return df

    def _limit_to_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Return the slice of df covering the last `days` days."""
        if df.empty:
            return df
        df = self._ensure_datetime_index(df)
        end = df.index.max()
        start = end - pd.Timedelta(days=days)
        return df.loc[df.index >= start]

    def _annualization_sqrt(self, index: pd.Index) -> float:
        """
        Compute sqrt of periods-per-year from index spacing.
        For crypto (24/7), using actual spacing is reasonable.
        """
        try:
            if len(index) < 2:
                return 1.0
            dt = (index[1] - index[0])
            # Handle pandas Timestamps / Timedelta
            if hasattr(dt, "total_seconds"):
                seconds = dt.total_seconds()
            else:
                seconds = float(dt)  # fallback
            if seconds <= 0:
                return 1.0
            periods_per_year = (365 * 24 * 3600) / seconds
            return float(np.sqrt(periods_per_year))
        except Exception:
            return 1.0

    def _prepare_analysis_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
        target_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Prepare and summarize data for AI analysis."""
        try:
            # Ensure datetime indexes & ascending order, and coerce to numeric
            data_3m = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_3m))
            data_1w = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_1w))

            # Constrain to the intended time windows (safety if callers pass extra history)
            window_3m = self._limit_to_days(data_3m, 92)  # ~3 months
            window_1w = self._limit_to_days(data_1w, 7)   # 1 week

            eastern_tz = pytz.timezone("US/Eastern")
            current_time = datetime.now(eastern_tz)

            # Use custom target or default to next Friday 4PM ET
            if target_datetime:
                if target_datetime.tzinfo is None:
                    prediction_target = eastern_tz.localize(target_datetime)
                else:
                    prediction_target = target_datetime.astimezone(eastern_tz)
            else:
                prediction_target = self._get_next_friday_4pm(current_time)

            actual_current_price = float(current_price)

            # Basic checks
            if window_3m.empty or window_1w.empty:
                self._dbg("warning", "One of the analysis datasets is empty after time-window trimming.")
                return {}

            # Sanity: warn if provided current price is far from latest 1w close
            latest_close_1w = float(window_1w["Close"].iloc[-1])
            if latest_close_1w > 0:
                rel_diff = abs(latest_close_1w - actual_current_price) / latest_close_1w
                if rel_diff > 0.02:  # >2%
                    self._dbg(
                        "warning",
                        f"⚠️ current_price {actual_current_price:,.2f} differs from latest 1w close "
                        f"{latest_close_1w:,.2f} by {rel_diff*100:.1f}%."
                    )

            start_price_3m = float(window_3m["Close"].iloc[0])
            start_price_1w = float(window_1w["Close"].iloc[0])

            # Volatility (annualized) using detected frequency
            ann_sqrt_3m = self._annualization_sqrt(window_3m.index)
            ann_sqrt_1w = self._annualization_sqrt(window_1w.index)

            data_3m_summary = {
                "period": "3 months",
                "current_price": actual_current_price,
                "start_price_3m": start_price_3m,
                "high_3m": float(window_3m["High"].max()),
                "low_3m": float(window_3m["Low"].min()),
                "price_change_3m": float((actual_current_price - start_price_3m) / start_price_3m * 100.0),
                "volatility_3m": float(window_3m["Close"].pct_change().std() * ann_sqrt_3m * 100.0),
                "avg_volume_3m": float(window_3m["Volume"].mean()),
                "start_date": str(window_3m.index[0]),
                "end_date": str(window_3m.index[-1]),
            }

            data_1w_summary = {
                "period": "1 week",
                "start_price_1w": start_price_1w,
                "high_1w": float(window_1w["High"].max()),
                "low_1w": float(window_1w["Low"].min()),
                "price_change_1w": float((actual_current_price - start_price_1w) / start_price_1w * 100.0),
                "volatility_1w": float(window_1w["Close"].pct_change().std() * ann_sqrt_1w * 100.0),
                "avg_volume_1w": float(window_1w["Volume"].mean()),
                "start_date": str(window_1w.index[0]),
                "end_date": str(window_1w.index[-1]),
            }

            self._dbg(
                "warning",
                f"📊 DATA SUMMARY TO CLAUDE: 3M High=${data_3m_summary['high_3m']:,.2f}, "
                f"1W High=${data_1w_summary['high_1w']:,.2f}"
            )

            # Indicator summary must consider the actual current price
            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w, actual_current_price)

            # Prepare enhanced chart data using the WINDOWED frames
            enhanced_data = self._prepare_enhanced_chart_data(
                window_3m, window_1w, indicators_3m, indicators_1w
            )

            # FIX: Override summary with CORRECT values from enhanced data to eliminate inconsistency
            data_3m_summary['high_3m'] = enhanced_data['3m_data']['period_highs_lows']['period_high']
            data_3m_summary['low_3m'] = enhanced_data['3m_data']['period_highs_lows']['period_low']
            data_1w_summary['high_1w'] = enhanced_data['1w_data']['period_highs_lows']['period_high']
            data_1w_summary['low_1w'] = enhanced_data['1w_data']['period_highs_lows']['period_low']

            analysis_data = {
                "current_time": current_time.isoformat(),
                "target_time": prediction_target.isoformat(),
                "hours_until_target": (prediction_target - current_time).total_seconds() / 3600.0,
                "data_3m": data_3m_summary,
                "data_1w": data_1w_summary,
                "indicators": indicators_summary,
                "enhanced_chart_data": enhanced_data,
                "current_price": actual_current_price,
                "target_datetime_formatted": prediction_target.strftime("%A %B %d, %Y at %I:%M %p ET"),
            }

            self._dbg(
                "success",
                f"📋 AFTER STORING: 3M period_high="
                f"{analysis_data['enhanced_chart_data']['3m_data']['period_highs_lows']['period_high']:,.2f} "
                f"(should match summary ~{data_3m_summary['high_3m']:,.2f})"
            )

            return analysis_data

        except Exception as e:
            st.error(f"Error preparing analysis data: {str(e)}")
            return {}

    def _prepare_enhanced_chart_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive chart data with full arrays for deep analysis.

        NOTE: `data_3m` and `data_1w` are expected to be the WINDOWED datasets
              (i.e., already trimmed to ~3 months and 1 week respectively).
        """
        try:
            enhanced: Dict[str, Any] = {}

            # Always compute "period" stats from the full WINDOW (not the display slice)
            full_3m = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_3m))
            full_1w = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_1w))

            full_3m_high = float(full_3m["High"].max())
            full_3m_low = float(full_3m["Low"].min())
            full_1w_high = float(full_1w["High"].max())
            full_1w_low = float(full_1w["Low"].min())
            
            # Debug: Show the actual calculated values
            self._dbg("error", f"🚨 CALCULATED BEFORE STORAGE: 3M High=${full_3m_high:,.2f}, 3M Low=${full_3m_low:,.2f}")
            self._dbg("error", f"🚨 CALCULATED BEFORE STORAGE: 1W High=${full_1w_high:,.2f}, 1W Low=${full_1w_low:,.2f}")

            # Optional display trimming for the RECENT arrays only
            display_from_3m = getattr(data_3m, "attrs", {}).get("display_from_index", 0)
            display_from_1w = getattr(data_1w, "attrs", {}).get("display_from_index", 0)

            if display_from_3m > 0 and display_from_3m < len(full_3m):
                recent_3m = full_3m.iloc[display_from_3m:]
            else:
                recent_3m = full_3m

            if display_from_1w > 0 and display_from_1w < len(full_1w):
                recent_1w = full_1w.iloc[display_from_1w:]
            else:
                recent_1w = full_1w

            # Get "tails" to keep arrays compact in prompt
            tail_3m = recent_3m.tail(50)
            tail_1w = recent_1w.tail(30)

            self._dbg(
                "error",
                f"🚨 WINDOWED STATS: 3M High=${full_3m_high:,.2f}, 1W High=${full_1w_high:,.2f}"
            )

            # 3-MONTH ENHANCED DATA
            enhanced["3m_data"] = {
                "timeframe": "3-month",
                "full_range": f"{full_3m.index[0].strftime('%B %d')} to {full_3m.index[-1].strftime('%B %d, %Y')}",
                "data_range": f"{tail_3m.index[0].strftime('%B %d')} to {tail_3m.index[-1].strftime('%B %d, %Y')}"
                if not tail_3m.empty else "N/A",
                "period_highs_lows": {
                    "period_high": full_3m_high,
                    "period_low": full_3m_low,
                    "recent_high": float(recent_3m["High"].max()) if not recent_3m.empty else None,
                    "recent_low": float(recent_3m["Low"].min()) if not recent_3m.empty else None,
                },
                "recent_prices": {
                    "dates": [d.strftime("%Y-%m-%d %H:%M") for d in tail_3m.index],
                    "open": tail_3m["Open"].round(2).tolist(),
                    "high": tail_3m["High"].round(2).tolist(),
                    "low": tail_3m["Low"].round(2).tolist(),
                    "close": tail_3m["Close"].round(2).tolist(),
                    "volume": tail_3m["Volume"].round(0).tolist(),
                },
                "indicators": {},
            }

            self._dbg("warning", f"🔧 ENHANCED 3M STORED: {enhanced['3m_data']['period_highs_lows']['period_high']:,.2f}")

            # Fill 3M indicator arrays (last 50)
            for indicator in [
                "RSI",
                "MACD",
                "MACD_Signal",
                "MACD_Histogram",
                "BB_Upper",
                "BB_Lower",
                "BB_Middle",
                "EMA_20",
                "SMA_50",
                "SMA_200",
            ]:
                if indicator in indicators_3m:
                    values = indicators_3m[indicator].dropna().tail(50)
                    enhanced["3m_data"]["indicators"][indicator] = values.round(4).tolist()

            # 1-WEEK ENHANCED DATA
            enhanced["1w_data"] = {
                "timeframe": "1-week",
                "full_range": f"{full_1w.index[0].strftime('%B %d')} to {full_1w.index[-1].strftime('%B %d, %Y')}",
                "data_range": f"{tail_1w.index[0].strftime('%B %d')} to {tail_1w.index[-1].strftime('%B %d, %Y')}"
                if not tail_1w.empty else "N/A",
                "period_highs_lows": {
                    "period_high": full_1w_high,
                    "period_low": full_1w_low,
                    "recent_high": float(recent_1w["High"].max()) if not recent_1w.empty else None,
                    "recent_low": float(recent_1w["Low"].min()) if not recent_1w.empty else None,
                },
                "recent_prices": {
                    "dates": [d.strftime("%Y-%m-%d %H:%M") for d in tail_1w.index],
                    "open": tail_1w["Open"].round(2).tolist(),
                    "high": tail_1w["High"].round(2).tolist(),
                    "low": tail_1w["Low"].round(2).tolist(),
                    "close": tail_1w["Close"].round(2).tolist(),
                    "volume": tail_1w["Volume"].round(0).tolist(),
                },
                "indicators": {},
            }

            self._dbg("warning", f"🔧 ENHANCED 1W STORED: {enhanced['1w_data']['period_highs_lows']['period_high']:,.2f}")

            # Fill 1W indicator arrays (last 30)
            for indicator in [
                "RSI",
                "MACD",
                "MACD_Signal",
                "MACD_Histogram",
                "BB_Upper",
                "BB_Lower",
                "BB_Middle",
                "EMA_20",
                "SMA_50",
                "SMA_200",
            ]:
                if indicator in indicators_1w:
                    values = indicators_1w[indicator].dropna().tail(30)
                    enhanced["1w_data"]["indicators"][indicator] = values.round(4).tolist()

            # Volume analysis on display-trimmed windows (but still derived from WINDOWED data)
            enhanced["volume_analysis"] = {
                "3m_avg_volume": float(full_3m["Volume"].tail(50).mean()),
                "3m_volume_trend": "increasing"
                if full_3m["Volume"].tail(10).mean() > full_3m["Volume"].tail(50).mean()
                else "decreasing",
                "1w_avg_volume": float(full_1w["Volume"].tail(30).mean()),
                "1w_volume_trend": "increasing"
                if full_1w["Volume"].tail(5).mean() > full_1w["Volume"].tail(30).mean()
                else "decreasing",
            }

            self._dbg(
                "error",
                f"🚀 ABOUT TO RETURN: 3M={enhanced['3m_data']['period_highs_lows']['period_high']:,.2f}, "
                f"1W={enhanced['1w_data']['period_highs_lows']['period_high']:,.2f}"
            )

            return enhanced

        except Exception as e:
            st.warning(f"Error preparing enhanced chart data: {str(e)}")
            return {}

    def _summarize_indicators(
        self,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
    ) -> Dict[str, Any]:
        """Summarize technical indicators for AI analysis, using actual current_price."""
        summary: Dict[str, Any] = {"current_price": float(current_price)}

        try:
            # RSI summary
            if "RSI" in indicators_3m and "RSI" in indicators_1w:
                rsi_3m_last = indicators_3m["RSI"].iloc[-1]
                rsi_1w_last = indicators_1w["RSI"].iloc[-1]
                summary["RSI"] = {
                    "3m_current": float(rsi_3m_last) if not np.isnan(rsi_3m_last) else None,
                    "1w_current": float(rsi_1w_last) if not np.isnan(rsi_1w_last) else None,
                }

            # MACD summary
            if "MACD" in indicators_3m and "MACD_Signal" in indicators_3m:
                macd_3m = indicators_3m["MACD"].iloc[-1]
                signal_3m = indicators_3m["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_3m) and not np.isnan(signal_3m):
                    summary["MACD_3m"] = {
                        "macd": float(macd_3m),
                        "signal": float(signal_3m),
                        "crossover": "bullish" if macd_3m > signal_3m else "bearish",
                    }

            if "MACD" in indicators_1w and "MACD_Signal" in indicators_1w:
                macd_1w = indicators_1w["MACD"].iloc[-1]
                signal_1w = indicators_1w["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_1w) and not np.isnan(signal_1w):
                    summary["MACD_1w"] = {
                        "macd": float(macd_1w),
                        "signal": float(signal_1w),
                        "crossover": "bullish" if macd_1w > signal_1w else "bearish",
                    }

            # Bollinger Bands summary
            for timeframe, indicators in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if all(k in indicators for k in ["BB_Upper", "BB_Lower", "BB_Middle"]):
                    upper = indicators["BB_Upper"].iloc[-1]
                    lower = indicators["BB_Lower"].iloc[-1]
                    middle = indicators["BB_Middle"].iloc[-1]
                    if not any(np.isnan([upper, lower, middle])):
                        cp = summary["current_price"]
                        summary[f"BB_{timeframe}"] = {
                            "upper": float(upper),
                            "lower": float(lower),
                            "middle": float(middle),
                            "position": "above_upper" if cp > upper else ("below_lower" if cp < lower else "within_bands"),
                        }

            # EMA summary
            for timeframe, indicators in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if "EMA_20" in indicators:
                    ema_20 = indicators["EMA_20"].iloc[-1]
                    if not np.isnan(ema_20):
                        cp = summary["current_price"]
                        summary[f"EMA_20_{timeframe}"] = {
                            "value": float(ema_20),
                            "trend": "bullish" if cp > ema_20 else "bearish",
                        }

        except Exception as e:
            st.warning(f"Error summarizing indicators: {str(e)}")

        return summary

    def _get_next_friday_4pm(self, current_time: datetime) -> datetime:
        """Calculate the next Friday 4:00 PM Eastern Time (today if before 4pm ET)."""
        # Friday = 4; modulo handles going forward within the week
        days_ahead = (4 - current_time.weekday()) % 7
        # If it's Friday and after/equal 4pm, jump to next week
        if days_ahead == 0 and current_time.hour >= 16:
            days_ahead = 7
        target = current_time + timedelta(days=days_ahead)
        return target.replace(hour=16, minute=0, second=0, microsecond=0)

    def _generate_technical_analysis_claude(self, analysis_data: Dict[str, Any]) -> str:
        """Generate technical analysis and price prediction using Claude."""
        try:
            if not self.claude_client:
                return "Error: Claude client not initialized"

            current_price = analysis_data.get("current_price", 0.0)
            data_3m = analysis_data.get("data_3m", {})
            data_1w = analysis_data.get("data_1w", {})
            target_datetime_formatted = analysis_data.get("target_datetime_formatted", "Friday 4PM ET")

            current_date = datetime.now().strftime("%B %d, %Y")
            start_candidates = [d for d in [data_3m.get("start_date"), analysis_data.get("data_1w", {}).get("start_date")] if d]
            end_candidates = [d for d in [data_3m.get("end_date"), analysis_data.get("data_1w", {}).get("end_date")] if d]
            start_date = min(start_candidates) if start_candidates else "N/A"
            end_date = max(end_candidates) if end_candidates else "N/A"
            enhanced_data = analysis_data.get("enhanced_chart_data", {})

            # Helpful debug hints (gated)
            if "enhanced_chart_data" in analysis_data and analysis_data["enhanced_chart_data"].get("3m_data"):
                claude_3m_high = analysis_data["enhanced_chart_data"]["3m_data"]["period_highs_lows"]["period_high"]
                claude_3m_low = analysis_data["enhanced_chart_data"]["3m_data"]["period_highs_lows"]["period_low"]
                self._dbg("success", f"🔍 3M FULL PERIOD: High=${claude_3m_high:,.2f}, Low=${claude_3m_low:,.2f}")

            if "enhanced_chart_data" in analysis_data and analysis_data["enhanced_chart_data"].get("1w_data"):
                claude_1w_high = analysis_data["enhanced_chart_data"]["1w_data"]["period_highs_lows"]["period_high"]
                claude_1w_low = analysis_data["enhanced_chart_data"]["1w_data"]["period_highs_lows"]["period_low"]
                self._dbg("success", f"🔍 1W FULL PERIOD: High=${claude_1w_high:,.2f}, Low=${claude_1w_low:,.2f}")

            self._dbg("success", f"🔍 CURRENT PRICE PARAMETER: ${current_price:,.2f}")
            
            # Debug: Show exactly what values are being sent to Claude
            self._dbg("error", f"🔍 SENDING TO CLAUDE - 3M HIGH: ${data_3m.get('high_3m', float('nan')):,.2f}")
            self._dbg("error", f"🔍 SENDING TO CLAUDE - 3M LOW: ${data_3m.get('low_3m', float('nan')):,.2f}")
            self._dbg("error", f"🔍 SENDING TO CLAUDE - 1W HIGH: ${data_1w.get('high_1w', float('nan')):,.2f}")
            self._dbg("error", f"🔍 SENDING TO CLAUDE - 1W LOW: ${data_1w.get('low_1w', float('nan')):,.2f}")

            comprehensive_prompt = f"""
            CRITICAL: Today is {current_date}. The data provided covers ONLY {start_date} through {end_date}.
            DO NOT REFERENCE ANY DATES OUTSIDE THIS RANGE.

            Bitcoin's current price is ${current_price:,.2f}.
            Always use ${current_price:,.2f} when referring to Bitcoin's current price.

            🚨🚨🚨 CRITICAL DATA REFERENCE - FOLLOW EXACTLY 🚨🚨🚨
            
            THESE ARE THE ONLY VALID PERIOD HIGH AND LOW VALUES:
            
            FOR 3-MONTH ANALYSIS:
            When you mention "3M period high" or "3-month high" → ONLY use: ${data_3m.get('high_3m', float('nan')):,.2f}
            When you mention "3M period low" or "3-month low" → ONLY use: ${data_3m.get('low_3m', float('nan')):,.2f}
            
            FOR 1-WEEK ANALYSIS:
            When you mention "1W period high" or "1-week high" → ONLY use: ${data_1w.get('high_1w', float('nan')):,.2f}
            When you mention "1W period low" or "1-week low" → ONLY use: ${data_1w.get('low_1w', float('nan')):,.2f}
            
            VALIDATION CHECK:
            - Is ${data_3m.get('high_3m', float('nan')):,.2f} > ${data_3m.get('low_3m', float('nan')):,.2f}? (3M high > 3M low) ✓
            - Is ${data_1w.get('high_1w', float('nan')):,.2f} > ${data_1w.get('low_1w', float('nan')):,.2f}? (1W high > 1W low) ✓
            
            NEVER SWAP OR CONFUSE THESE VALUES. NEVER USE ANY OTHER NUMBERS FOR PERIOD HIGHS/LOWS.

            PRICE PERFORMANCE:
            • 3-month change: {analysis_data.get('data_3m', {}).get('price_change_3m', 0):+.2f}%
            • 1-week change: {analysis_data.get('data_1w', {}).get('price_change_1w', 0):+.2f}%

            TECHNICAL INDICATORS SUMMARY:
            {json.dumps(analysis_data.get('indicators', {}), indent=2)}

            Provide analysis in three sections:

            [TECHNICAL_ANALYSIS_START]
            **COMPREHENSIVE TECHNICAL ANALYSIS**

            **Current Price: ${current_price:,.2f}**

            **1. MULTI-TIMEFRAME OVERVIEW**
            - 3-Month Chart Analysis: [Trend, key levels, patterns]
            - 1-Week Chart Analysis: [Trend, key levels, patterns]
            - Timeframe Alignment: [How they agree or conflict]

            **2. TECHNICAL INDICATORS ANALYSIS**
            - RSI Analysis: [3M vs 1W RSI, overbought/oversold, divergences]
            - MACD Analysis: [Signal line crossovers, histogram, divergences]
            - Bollinger Bands: [Position relative to bands, squeeze/expansion]
            - EMA Analysis: [20-period trends, price above/below]

            **3. ADVANCED PATTERN ANALYSIS**
            - Chart Patterns: [Triangles, head & shoulders, flags, wedges]
            - Candlestick Patterns: [Recent significant patterns]
            - Support/Resistance: [Key levels identified from price action]

            **4. DIVERGENCE ANALYSIS**
            - Price vs RSI Divergences: [Bullish or bearish divergences]
            - Price vs MACD Divergences: [Hidden or regular divergences]
            - Volume Divergences: [If volume data conflicts with price]

            **5. FAILURE SWING ANALYSIS**
            - RSI Failure Swings: [Bullish/bearish failure swings in RSI]
            - MACD Failures: [Failed breakouts or breakdowns]
            - Price Action Failures: [Failed support/resistance tests]

            **6. TRENDLINE & STRUCTURE**
            - Primary Trendlines: [Major ascending/descending lines]
            - Support Levels: [Key horizontal and trend support]
            - Resistance Levels: [Key horizontal and trend resistance]
            - Market Structure: [Higher highs/lows, lower highs/lows]

            **7. TRADING RECOMMENDATION**
            - Overall Bias: **[BULLISH/BEARISH/NEUTRAL]**
            - Entry Strategy: [Specific entry points and conditions]
            - Stop Loss: [Risk management levels]
            - Target Levels: [Profit-taking areas]
            [TECHNICAL_ANALYSIS_END]

            [PRICE_PREDICTION_START]
            **PRICE PREDICTION for {target_datetime_formatted}**

            Based on the comprehensive technical analysis above:

            1. **Probability HIGHER than ${current_price:,.2f}: [X]%**
            2. **Probability LOWER than ${current_price:,.2f}: [Y]%**
            3. **Confidence Level: [Z]%**

            **Key Technical Factors Supporting This Assessment:**
            - [List 3-5 specific technical reasons for the prediction]

            **Potential Price Targets:**
            - Upside Target 1: $[amount] (reasoning)
            - Upside Target 2: $[amount] (reasoning)
            - Downside Target 1: $[amount] (reasoning)
            - Downside Target 2: $[amount] (reasoning)

            **Critical Levels to Watch:**
            - Bullish above: $[level]
            - Bearish below: $[level]

            [PRICE_PREDICTION_END]

            STRICT ANALYSIS RULES:
            - ONLY analyze data from {start_date} to {end_date}
            - Use ONLY the actual dates provided in the chart data arrays
            - Probabilities must sum to 100% (internal instruction - do not print this)
            """

            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=2500,
                temperature=0.3,
                system="You are a professional Bitcoin analyst providing comprehensive, consistent analysis across technical, predictive, and market perspectives.",
                messages=[{"role": "user", "content": comprehensive_prompt}],
            )

            # Anthropic python SDK returns content items; pick first text block if available
            content = response.content
            ai_response = content[0].text if content and hasattr(content[0], "text") else str(content[0])

            return ai_response

        except Exception as e:
            return f"Error generating technical analysis: {str(e)}"

    def _parse_comprehensive_response(self, response: str) -> Dict[str, str]:
        """Parse Anthropic response into sections."""
        try:
            sections: Dict[str, str] = {}

            tech_start = response.find("[TECHNICAL_ANALYSIS_START]")
            tech_end = response.find("[TECHNICAL_ANALYSIS_END]")
            if tech_start != -1 and tech_end != -1:
                sections["technical_summary"] = response[tech_start + len("[TECHNICAL_ANALYSIS_START]") : tech_end].strip()

            pred_start = response.find("[PRICE_PREDICTION_START]")
            pred_end = response.find("[PRICE_PREDICTION_END]")
            if pred_start != -1 and pred_end != -1:
                sections["price_prediction"] = response[pred_start + len("[PRICE_PREDICTION_START]") : pred_end].strip()

            if not sections:
                # fallback parsing
                lines = response.split("\n")
                current_section = None
                current_content = []
                for line in lines:
                    line_lower = line.lower().strip()
                    if "technical analysis" in line_lower:
                        if current_section and current_content:
                            sections[current_section] = "\n".join(current_content).strip()
                        current_section = "technical_summary"
                        current_content = []
                    elif "price prediction" in line_lower or "friday" in line_lower:
                        if current_section and current_content:
                            sections[current_section] = "\n".join(current_content).strip()
                        current_section = "price_prediction"
                        current_content = []
                    elif current_section:
                        current_content.append(line)
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content).strip()

            return sections or {"technical_summary": response, "price_prediction": ""}

        except Exception:
            return {"technical_summary": response, "price_prediction": "Unable to parse prediction section"}

    def _extract_probabilities(self, prediction_text: str) -> Dict[str, Any]:
        """Extract probability percentages from prediction text. Returns both fractions and percents."""
        probs = {"higher_fraction": 0.5, "lower_fraction": 0.5, "confidence_fraction": 0.5,
                 "higher_pct": 50.0, "lower_pct": 50.0, "confidence_pct": 50.0}
        try:
            import re

            patterns = {
                "higher": [
                    r"(\d+)%?\s*(?:probability|chance|likelihood).*?(?:higher|up|increase)",
                    r"(?:higher|up|increase).*?(\d+)%",
                    r"HIGHER.*?(\d+)%",
                    r"(\d+)%.*?higher",
                ],
                "lower": [
                    r"(\d+)%?\s*(?:probability|chance|likelihood).*?(?:lower|down|decrease)",
                    r"(?:lower|down|decrease).*?(\d+)%",
                    r"LOWER.*?(\d+)%",
                    r"(\d+)%.*?lower",
                ],
                "confidence": [
                    r"confidence.*?(\d+)%",
                    r"(\d+)%.*?confidence",
                    r"confident.*?(\d+)%",
                ],
            }

            text_lower = prediction_text.lower()

            for key in ["higher", "lower", "confidence"]:
                for pat in patterns[key]:
                    m = re.findall(pat, text_lower, flags=re.IGNORECASE)
                    if m:
                        val = float(m[0])
                        if key == "higher":
                            probs["higher_pct"] = val
                        elif key == "lower":
                            probs["lower_pct"] = val
                        else:
                            probs["confidence_pct"] = val
                        break

            # Normalize higher/lower to sum to 100
            total = probs["higher_pct"] + probs["lower_pct"]
            if total > 0:
                probs["higher_pct"] = probs["higher_pct"] * 100.0 / total
                probs["lower_pct"] = probs["lower_pct"] * 100.0 / total

            # Fractions 0..1
            probs["higher_fraction"] = probs["higher_pct"] / 100.0
            probs["lower_fraction"] = probs["lower_pct"] / 100.0
            probs["confidence_fraction"] = probs["confidence_pct"] / 100.0

        except Exception as e:
            self._dbg("warning", f"Error extracting probabilities: {str(e)}")

        return probs
