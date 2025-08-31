# ai_analysis_fixed.py
# Robust rewrite of your AIAnalyzer to avoid breakages, add safer parsing, and
# make OpenAI calls more resilient across SDK versions.
# Enhanced with recommended improvements: stronger data validation, auto-compute missing indicators (using pandas_ta),
# advanced features (e.g., trend slopes), few-shot prompting, structured output validation (jsonschema),
# enhanced logging, and optional data fetching.

from __future__ import annotations
import os
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import logging
from scipy import stats  # For outlier detection
import jsonschema  # For JSON schema validation

# For auto-computing indicators (assuming pandas_ta is installed; pip install pandas_ta)
try:
    import pandas_ta as ta
except ImportError:
    ta = None  # Fallback if not available

# For optional data fetching (assuming yfinance is installed; pip install yfinance)
try:
    import yfinance as yf
except ImportError:
    yf = None

# --- Streamlit-safe import (falls back to console logger) ---
try:  # noqa: SIM105
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _DummySt:
        def write(self, *a, **k):
            print(*a)
        def warning(self, *a, **k):
            print("[warning]", *a)
        def error(self, *a, **k):
            print("[error]", *a)
        def info(self, *a, **k):
            print("[info]", *a)
    st = _DummySt()  # type: ignore

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- OpenAI client wrapper (tolerant to API/SDK differences) ---
try:
    from openai import OpenAI  # Official SDK (>=1.0)
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

class _OpenAIWrapper:
    def __init__(self, api_key: Optional[str], model: str, debug: bool = False):
        self.model = model
        self.debug = debug
        self.client = None
        if api_key and OpenAI is not None:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:  # pragma: no cover
                st.warning(f"Could not initialize OpenAI client: {e}")
                logger.warning(f"OpenAI init error: {e}")
                self.client = None
        else:
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY.")
                logger.error("Missing OpenAI API key.")
            if OpenAI is None:
                st.error("openai SDK not available in environment.")
                logger.error("OpenAI SDK missing.")

    def _extract_text_from_responses(self, resp: Any) -> str:
        # ... (unchanged, keeping original extraction logic)
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
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
        except Exception:  # pragma: no cover
            pass
        try:
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        return content
        except Exception:  # pragma: no cover
            pass
        return ""

    def generate(self, system_msg: str, user_msg: str) -> str:
        if not self.client:
            return ""
        # Add temperature=0 for determinism
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,  # Added for determinism
            )
            text = self._extract_text_from_responses(resp)
            if text:
                return text
        except Exception as e:
            if self.debug:
                st.warning(f"Responses API failed: {e}")
                logger.warning(f"Responses API error: {e}")
        try:
            chat = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,  # Added
            )
            return self._extract_text_from_responses(chat)
        except Exception as e:
            if self.debug:
                st.error(f"Chat Completions failed: {e}")
                logger.error(f"Chat Completions error: {e}")
            return ""

class AIAnalyzer:
    """
    AIAnalyzer — strict, data-only technical analysis + narrative + point forecast.
    Outputs expected:
      - 'technical_summary' (markdown, from [TECHNICAL_ANALYSIS_*] block)
      - 'price_prediction' (markdown, from [PRICE_PREDICTION_*] block)
      - 'probabilities' (numbers for gauges)
      - 'model_json' (parsed JSON block)
      - 'status' ('ok' | 'insufficient_data' | 'error')
    """
    def __init__(self, debug: Optional[bool] = None, min_bars_3m: int = 60, min_bars_1w: int = 5):
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT5_MODEL", "gpt-5-nano")  # Configurable
        self.gpt = _OpenAIWrapper(self.openai_key, self.model_name, debug=self.debug)
        self._last_current_price: Optional[float] = None
        self.min_bars_3m = min_bars_3m  # Added for validation
        self.min_bars_1w = min_bars_1w
        # JSON schema for validation
        self.output_schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["ok", "insufficient_data"]},
                "asset": {"type": "string"},
                "as_of": {"type": "string"},
                "target_ts": {"type": "string"},
                "predicted_price": {"type": ["number", "null"]},
                "p_up": {"type": "number"},
                "p_down": {"type": "number"},
                "conf_overall": {"type": "number"},
                "conf_price": {"type": "number"},
                "expected_pct_move": {"type": ["number", "null"]},
                "critical_levels": {
                    "type": "object",
                    "properties": {
                        "bullish_above": {"type": ["number", "null"]},
                        "bearish_below": {"type": ["number", "null"]}
                    }
                },
                "evidence": {"type": "array"},
                "notes": {"type": "array"}
            },
            "required": ["status"]
        }

    # Added: Optional data fetching method
    def fetch_data(self, asset: str = "BTC-USD", period_3m: str = "3mo", period_1w: str = "1wk", interval: str = "1d") -> Tuple[pd.DataFrame, pd.DataFrame]:
        if yf is None:
            logger.error("yfinance not available for data fetching.")
            raise ImportError("yfinance required for data fetching.")
        data_3m = yf.download(asset, period=period_3m, interval=interval)
        data_1w = yf.download(asset, period=period_1w, interval=interval)
        return data_3m, data_1w

    # ---------- public API ----------
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
        if not self.gpt or not self.gpt.client:
            return {"status": "error", "error": "GPT client not initialized"}
        self._last_current_price = float(current_price)
        try:
            # Auto-compute missing indicators if pandas_ta available
            if ta is not None:
                indicators_3m = self._auto_compute_indicators(data_3m, indicators_3m)
                indicators_1w = self._auto_compute_indicators(data_1w, indicators_1w)

            analysis_data = self._prepare_analysis_data(
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
            # Ask for BOTH: JSON block + narrative section blocks
            raw = self._generate_technical_analysis_gpt5(analysis_data)
            # Split/parse outputs
            json_text, narrative_text = self._split_dual_output(raw)
            parsed_json = self._parse_json_response(json_text, self._last_current_price or current_price)
            # Extract rich sections (like older script)
            sections = self._parse_comprehensive_response(narrative_text) if narrative_text else {}
            # Build probabilities
            if parsed_json.get("status") == "ok":
                probs = self._extract_probabilities_from_json(parsed_json, self._last_current_price or current_price)
            else:
                probs = self._extract_probabilities(sections.get("price_prediction", "") if sections else "")
            # Text blocks to show in UI
            tech_md = sections.get("technical_summary") if sections else None
            pred_md = sections.get("price_prediction") if sections else None
            # If model omitted narrative, synthesize from JSON so UI still looks good
            if not tech_md or not pred_md:
                synth_tech, synth_pred = self._compose_text_from_model_json(parsed_json, current_price)
                tech_md = tech_md or synth_tech
                pred_md = pred_md or synth_pred
            # Honor JSON status; map other to insufficient
            status = parsed_json.get("status", "ok") if isinstance(parsed_json, dict) else "error"
            if status not in ("ok", "insufficient_data"):
                status = "insufficient_data"
            # Ensure we never show empty target in insufficient state
            if status != "ok" and ("Target:" in pred_md) and not re.search(r"`[^`]+`", pred_md):
                pred_md = f"**Target:** `{target_ts_fallback}`\n\n_No price prediction due to insufficient data._"
            return {
                "status": status,
                "model_json": parsed_json,
                "probabilities": probs,
                "technical_summary": tech_md,
                "price_prediction": pred_md,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            st.error(f"Error generating AI analysis: {e}")
            logger.error(f"Analysis error: {e}", exc_info=True)
            target_ts = target_datetime.isoformat() if target_datetime else ""
            tech, pred = self._compose_text_when_insufficient(str(e), target_ts)
            return {
                "status": "error",
                "error": str(e),
                "probabilities": self._default_probs(),
                "technical_summary": tech,
                "price_prediction": pred,
            }

    # Added: Auto-compute missing indicators
    def _auto_compute_indicators(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        if df.empty or 'Close' not in df.columns:
            return indicators
        required = ["RSI", "MACD", "MACD_Signal", "MACD_Histogram", "BB_Upper", "BB_Lower", "BB_Middle", "EMA_20", "SMA_50", "SMA_200"]
        for ind in required:
            if ind not in indicators or indicators[ind].empty:
                if ind == "RSI":
                    indicators[ind] = ta.rsi(df['Close'])
                elif ind.startswith("MACD"):
                    macd = ta.macd(df['Close'])
                    indicators["MACD"] = macd["MACD_12_26_9"]
                    indicators["MACD_Signal"] = macd["MACDs_12_26_9"]
                    indicators["MACD_Histogram"] = macd["MACDh_12_26_9"]
                elif ind.startswith("BB"):
                    bb = ta.bbands(df['Close'])
                    indicators["BB_Upper"] = bb["BBU_5_2.0"]
                    indicators["BB_Lower"] = bb["BBL_5_2.0"]
                    indicators["BB_Middle"] = bb["BBM_5_2.0"]
                elif ind == "EMA_20":
                    indicators[ind] = ta.ema(df['Close'], length=20)
                elif ind == "SMA_50":
                    indicators[ind] = ta.sma(df['Close'], length=50)
                elif ind == "SMA_200":
                    indicators[ind] = ta.sma(df['Close'], length=200)
        return indicators

    # ---------- helpers: debug ----------
    def _dbg(self, level: str, msg: str) -> None:
        if not self.debug:
            return
        fn = getattr(st, level, None)
        (fn or st.write)(msg)
        logger.log(getattr(logging, level.upper(), logging.INFO), msg)

    # ---------- helpers: dataframe hygiene ----------
    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (unchanged)
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
        # Added: Outlier detection for volume
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
                if col == "Volume":
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = z_scores > 3
                    if outliers.any():
                        logger.warning(f"Outliers detected in {col}: {outliers.sum()} values > 3 std devs.")
                        # Optional: Cap outliers
                        # df.loc[outliers, col] = df[col].median() * 3
        return df

    # ... (other helpers unchanged for brevity: _limit_to_days, _annualization_sqrt, _infer_index_step, _determine_timezone)

    # ---------- core prep ----------
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
        prep_notes = []
        try:
            data_3m = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_3m))
            data_1w = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_1w))
            window_3m = self._limit_to_days(data_3m, 92) if data_3m is not None else pd.DataFrame()
            window_1w = self._limit_to_days(data_1w, 7) if data_1w is not None else pd.DataFrame()
            # Added: Stronger validation
            if len(window_3m) < self.min_bars_3m:
                prep_notes.append(f"3m data too short ({len(window_3m)} < {self.min_bars_3m})")
            if len(window_1w) < self.min_bars_1w:
                prep_notes.append(f"1w data too short ({len(window_1w)} < {self.min_bars_1w})")
            if prep_notes:
                return {"prep_status": "insufficient_data", "prep_notes": prep_notes}
            if (window_3m is None or window_3m.empty) and (window_1w is None or window_1w.empty):
                return {"prep_status": "insufficient_data", "prep_notes": ["no_price_data_after_trimming"]}
            # Added: Validate indicator alignment
            for tf, data, inds in [("3m", window_3m, indicators_3m), ("1w", window_1w, indicators_1w)]:
                for key, ser in inds.items():
                    if not ser.index.equals(data.index):
                        prep_notes.append(f"{tf} {key} index mismatch with price data")
            if prep_notes:
                return {"prep_status": "insufficient_data", "prep_notes": prep_notes}
            # ... (rest of _prepare_analysis_data unchanged, including summaries, features, etc.)
            tz = self._determine_timezone(
                window_1w.index if not window_1w.empty else None,
                window_3m.index if not window_3m.empty else None,
            )
            current_time = datetime.now(tz)
            if target_datetime:
                target = target_datetime.astimezone(tz) if target_datetime.tzinfo else tz.localize(target_datetime)
            else:
                base_df = window_1w if not window_1w.empty else window_3m
                step = self._infer_index_step(base_df.index)
                if step is None:
                    return {"prep_status": "insufficient_data", "prep_notes": ["no_target_and_cannot_infer_step"]}
                last_val = base_df.index[-1] if pd.notna(base_df.index[-1]) else pd.Timestamp("now")
                target = pd.Timestamp(last_val) + step
                target = target.tz_convert(tz) if target.tzinfo else target.tz_localize(tz)
            if (target - current_time).total_seconds() < 0:
                return {"prep_status": "insufficient_data", "prep_notes": ["target_before_current_time"]}
            actual_current_price = float(current_price)
            if not window_1w.empty and "Close" in window_1w.columns:
                latest_close_1w = float(window_1w["Close"].iloc[-1])
                if latest_close_1w > 0:
                    rel_diff = abs(latest_close_1w - actual_current_price) / latest_close_1w
                    if rel_diff > 0.02:
                        self._dbg(
                            "warning",
                            f"⚠️ current_price {actual_current_price:,.2f} differs from latest 1w close "
                            f"{latest_close_1w:,.2f} by {rel_diff*100:.1f}%.",
                        )
            def _summary(df: pd.DataFrame, label: str) -> Dict[str, Any]:
                if df is None or df.empty:
                    return {}
                start_price = float(df["Close"].iloc[0])
                ann_sqrt = self._annualization_sqrt(df.index)
                return {
                    "period": label,
                    "start_price": start_price,
                    "high": float(df["High"].max()) if "High" in df.columns else None,
                    "low": float(df["Low"].min()) if "Low" in df.columns else None,
                    "price_change_pct": float((actual_current_price - start_price) / max(1e-9, start_price) * 100.0),
                    "volatility_ann_pct": float(df["Close"].pct_change().std() * ann_sqrt * 100.0),
                    "avg_volume": float(df["Volume"].mean()) if "Volume" in df.columns else None,
                    "start_date": str(df.index[0]),
                    "end_date": str(df.index[-1]),
                }
            data_3m_summary = _summary(window_3m, "3 months") if not window_3m.empty else {}
            data_1w_summary = _summary(window_1w, "1 week") if not window_1w.empty else {}
            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w, actual_current_price)
            enhanced_data = self._prepare_enhanced_chart_data(window_3m, window_1w, indicators_3m, indicators_1w)
            if enhanced_data.get("3m_data", {}).get("period_highs_lows") and data_3m_summary:
                data_3m_summary["high"] = enhanced_data["3m_data"]["period_highs_lows"].get("period_high")
                data_3m_summary["low"] = enhanced_data["3m_data"]["period_highs_lows"].get("period_low")
            if enhanced_data.get("1w_data", {}).get("period_highs_lows") and data_1w_summary:
                data_1w_summary["high"] = enhanced_data["1w_data"]["period_highs_lows"].get("period_high")
                data_1w_summary["low"] = enhanced_data["1w_data"]["period_highs_lows"].get("period_low")
            features = self._compute_features(window_3m, window_1w, indicators_3m, indicators_1w)
            return {
                "asset_name": asset_name,
                "current_time": current_time.isoformat(),
                "target_time": target.isoformat(),
                "hours_until_target": (target - current_time).total_seconds() / 3600.0,
                "data_3m": data_3m_summary,
                "data_1w": data_1w_summary,
                "indicators": indicators_summary,
                "enhanced_chart_data": enhanced_data,
                "features": features,
                "current_price": actual_current_price,
                "prep_status": "ok",
                "prep_notes": prep_notes,
            }
        except Exception as e:
            logger.error(f"Prep error: {e}", exc_info=True)
            return {"prep_status": "insufficient_data", "prep_notes": [str(e)]}

    def _compute_features(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        ind_3m: Dict[str, pd.Series],
        ind_1w: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        def last_n_returns(df: pd.DataFrame, n: int) -> Optional[float]:
            try:
                return float(df["Close"].pct_change().tail(n).sum())
            except Exception:
                return None
        def bb_width(ind: Dict[str, pd.Series]) -> Optional[float]:
            try:
                if "BB_Upper" in ind and "BB_Lower" in ind:
                    return float(ind["BB_Upper"].iloc[-1] - ind["BB_Lower"].iloc[-1])
            except Exception:
                pass
            return None
        def ema_slope(ind: Dict[str, pd.Series], k: int = 5) -> Optional[float]:
            try:
                if "EMA_20" in ind and len(ind["EMA_20"]) > k:
                    return float(ind["EMA_20"].iloc[-1] - ind["EMA_20"].iloc[-k - 1])
            except Exception:
                pass
            return None
        # Added: Advanced features - price trend slope
        def price_trend_slope(df: pd.DataFrame, tail_n: int) -> Optional[float]:
            try:
                tail = df["Close"].tail(tail_n).values
                x = np.arange(len(tail))
                slope, _, _, _, _ = stats.linregress(x, tail)
                return float(slope)
            except Exception:
                return None
        if data_1w is not None and not data_1w.empty:
            feats["ret_1w_last5bars"] = last_n_returns(data_1w, 5)
            feats["vol_ann_1w_pct"] = float(
                data_1w["Close"].pct_change().std() * self._annualization_sqrt(data_1w.index) * 100.0
            )
            feats["price_slope_1w"] = price_trend_slope(data_1w, 30)
        if ind_1w:
            feats["bb_width_1w"] = bb_width(ind_1w)
            feats["ema20_slope_1w"] = ema_slope(ind_1w, 5)
            feats["rsi_last_1w"] = (
                float(ind_1w["RSI"].iloc[-1]) if "RSI" in ind_1w and not np.isnan(ind_1w["RSI"].iloc[-1]) else None
            )
        if data_3m is not None and not data_3m.empty:
            feats["ret_3m_last5bars"] = last_n_returns(data_3m, 5)
            feats["vol_ann_3m_pct"] = float(
                data_3m["Close"].pct_change().std() * self._annualization_sqrt(data_3m.index) * 100.0
            )
            feats["price_slope_3m"] = price_trend_slope(data_3m, 50)
        if ind_3m:
            feats["bb_width_3m"] = bb_width(ind_3m)
            feats["ema20_slope_3m"] = ema_slope(ind_3m, 5)
            feats["rsi_last_3m"] = (
                float(ind_3m["RSI"].iloc[-1]) if "RSI" in ind_3m and not np.isnan(ind_3m["RSI"].iloc[-1]) else None
            )
        return feats

    # ... (other methods like _prepare_enhanced_chart_data, _summarize_indicators unchanged for brevity)

    # ---------- prompt + model call ----------
    def _build_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        system_content = (
            "You are a deterministic technical analyst. Use ONLY the structured arrays provided in the ENHANCED ARRAYS, FEATURES, INDICATOR SNAPSHOT, and VOLUME_ANALYSIS sections. "
            "Analyze the FULL arrays (e.g., entire RSI, MACD, BB, EMA histories) to derive trends, slopes, peaks/troughs, divergences, and patterns. "
            "Explicitly compare 1-week (1w) and 3-month (3m) arrays for each indicator, noting alignments, conflicts, and how short-term momentum influences longer-term trends. "
            "Forbidden: news, macro, on-chain, session effects (market open/close), weekdays, seasonality, or inferences beyond given timestamps and data. "
            "All claims must be quantitatively verifiable from the arrays (e.g., compute averages, slopes, or crossovers directly). "
            "If any required arrays are missing, empty, or insufficient for full analysis (e.g., no full RSI array for divergence detection), output status='insufficient_data' with detailed 'notes'. "
            "Prioritize depth: synthesize across indicators and timeframes for the most comprehensive, evidence-based analysis possible. "
            "Avoid overconfidence; base confidences on data variance (e.g., high vol_ann_pct lowers conf_price)."
        )
        # Added: Few-shot example
        few_shot = """
Example Input (fictional):
FEATURES: {"price_slope_3m": 100, "rsi_last_3m": 60}
ENHANCED ARRAYS: {"3m_data": {"indicators": {"RSI": [50, 55, 60]}}}

Example Output JSON:
{"status": "ok", "p_up": 0.6, "p_down": 0.4, ...}

Example Narrative:
[TECHNICAL_ANALYSIS_START]
**COMPREHENSIVE TECHNICAL ANALYSIS**
- RSI Analysis: 3m RSI trending up from 50 to 60...
[TECHNICAL_ANALYSIS_END]
"""
        system_content += "\n" + few_shot

        # ... (rest of _build_messages unchanged, including user_content, narrative_template)

        return ({"role": "system", "content": system_content}, {"role": "user", "content": user_content})

    # ... (other methods like _generate_technical_analysis_gpt5 unchanged)

    def _parse_json_response(self, response_text: str, current_price: float) -> Dict[str, Any]:
        if not response_text:
            return {"status": "insufficient_data", "notes": ["no_json_block_found"]}
        try:
            data = json.loads(response_text)
            if not isinstance(data, dict):
                return {"status": "insufficient_data", "notes": ["non_dict_json"]}
            # Added: Schema validation
            jsonschema.validate(instance=data, schema=self.output_schema)
            # Normalize status
            st_raw = str(data.get("status", "insufficient_data")).lower().strip()
            data["status"] = "ok" if st_raw == "ok" else "insufficient_data"
            # ... (rest unchanged)
            return data
        except jsonschema.ValidationError as ve:
            logger.warning(f"JSON schema validation error: {ve}")
            return {"status": "insufficient_data", "notes": [f"schema_error:{ve.message}"]}
        except Exception as e:
            logger.warning(f"JSON parse error: {e}")
            return {"status": "insufficient_data", "notes": [f"json_parse_error:{e}"]}

    # ... (remaining methods like _parse_comprehensive_response, _extract_probabilities_from_json, etc., unchanged)

# Optional: import contextlib for safe suppress above
import contextlib  # keep at end to avoid masking top imports in some environments