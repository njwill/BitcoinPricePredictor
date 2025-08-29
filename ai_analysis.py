# ai_analysis.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
import re


class AIAnalyzer:
    """
    AIAnalyzer — strict, data-only technical analysis + narrative + point forecast.

    Outputs your app expects:
      - 'technical_summary' (markdown, from [TECHNICAL_ANALYSIS_*] block)
      - 'price_prediction' (markdown, from [PRICE_PREDICTION_*] block)
      - 'probabilities' (numbers for your gauges)
      - 'model_json' (parsed JSON block)
      - 'status' ('ok' | 'insufficient_data' | 'error')
    """

    def __init__(self, debug: Optional[bool] = None):
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT5_MODEL", "gpt-5")

        if not self.openai_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            self.gpt5_client = None
        else:
            self.gpt5_client = OpenAI(api_key=self.openai_key)

        self._last_current_price: Optional[float] = None

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
        if not self.gpt5_client:
            return {"status": "error", "error": "GPT-5 client not initialized"}

        self._last_current_price = float(current_price)

        try:
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

            # Extract rich sections (like your old script)
            sections = self._parse_comprehensive_response(narrative_text) if narrative_text else {}

            # Build probabilities
            if parsed_json.get("status") == "ok":
                probs = self._extract_probabilities_from_json(parsed_json, self._last_current_price or current_price)
            else:
                probs = self._extract_probabilities(sections.get("price_prediction", "") if sections else "")

            # Text blocks to show in UI
            tech_md = sections.get("technical_summary")
            pred_md = sections.get("price_prediction")

            # If model omitted narrative, synthesize from JSON so UI still looks good
            if not tech_md or not pred_md:
                synth_tech, synth_pred = self._compose_text_from_model_json(parsed_json, current_price)
                tech_md = tech_md or synth_tech
                pred_md = pred_md or synth_pred

            # If the JSON says insufficient but narrative exists, honor JSON for status
            status = parsed_json.get("status", "ok") if isinstance(parsed_json, dict) else "error"
            if status != "ok" and status != "insufficient_data":
                status = "insufficient_data"

            # Ensure we never show empty target
            if status != "ok" and ("`" in pred_md and "Target:" in pred_md and "``" in pred_md):
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
            st.error(f"Error generating AI analysis: {str(e)}")
            target_ts = target_datetime.isoformat() if target_datetime else ""
            tech, pred = self._compose_text_when_insufficient(str(e), target_ts)
            return {"status": "error", "error": str(e), "probabilities": self._default_probs(),
                    "technical_summary": tech, "price_prediction": pred}

    # ---------- helpers: debug ----------

    def _dbg(self, level: str, msg: str):
        if not self.debug:
            return
        fn = getattr(st, level, None)
        (fn or st.write)(msg)

    # ---------- helpers: dataframe hygiene ----------

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()

        # If data_fetcher reset index and left a 'Datetime' column
        if df.index.dtype.kind in ['i', 'f']:
            df.reset_index(drop=True, inplace=True)
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if date_cols:
                df.index = pd.to_datetime(df[date_cols[0]])
                df.drop(columns=[date_cols[0]], inplace=True)
            else:
                df.index = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

        if not hasattr(df.index, "inferred_type") or "date" not in str(df.index.inferred_type):
            df.index = pd.to_datetime(df.index)

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df

    def _coerce_ohlcv_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        return df

    def _limit_to_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = self._ensure_datetime_index(df)
        end = pd.Timestamp(df.index.max())
        start = end - pd.Timedelta(days=days)
        return df.loc[df.index >= start]

    def _annualization_sqrt(self, index: pd.Index) -> float:
        try:
            if len(index) < 2:
                return 1.0
            dt = (index[1] - index[0])
            seconds = dt.total_seconds() if hasattr(dt, "total_seconds") else float(dt)
            if seconds <= 0:
                return 1.0
            periods_per_year = (365 * 24 * 3600) / seconds
            return float(np.sqrt(periods_per_year))
        except Exception:
            return 1.0

    def _infer_index_step(self, index: pd.Index) -> Optional[pd.Timedelta]:
        if index is None or len(index) < 2:
            return None
        diffs = pd.Series(index[1:]) - pd.Series(index[:-1])
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return None
        return pd.to_timedelta(np.median(diffs.values))

    def _determine_timezone(self, *indexes: pd.Index) -> pytz.BaseTzInfo:
        for idx in indexes:
            if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
                return idx.tz
        return pytz.timezone("US/Eastern")

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
        try:
            data_3m = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_3m))
            data_1w = self._coerce_ohlcv_numeric(self._ensure_datetime_index(data_1w))

            window_3m = self._limit_to_days(data_3m, 92) if data_3m is not None else pd.DataFrame()
            window_1w = self._limit_to_days(data_1w, 7) if data_1w is not None else pd.DataFrame()

            if (window_3m is None or window_3m.empty) and (window_1w is None or window_1w.empty):
                return {"prep_status": "insufficient_data", "prep_notes": ["no_price_data_after_trimming"]}

            tz = self._determine_timezone(window_1w.index if not window_1w.empty else None,
                                          window_3m.index if not window_3m.empty else None)
            current_time = datetime.now(tz)

            # Target handling
            if target_datetime:
                target = target_datetime.astimezone(tz) if target_datetime.tzinfo else tz.localize(target_datetime)
            else:
                base_df = window_1w if not window_1w.empty else window_3m
                step = self._infer_index_step(base_df.index)
                if step is None:
                    return {"prep_status": "insufficient_data", "prep_notes": ["no_target_and_cannot_infer_step"]}
                target = pd.Timestamp(base_df.index[-1]) + step
                target = (target.tz_convert(tz) if target.tzinfo else tz.localize(target.to_pydatetime()))

            if (target - current_time).total_seconds() < 0:
                return {"prep_status": "insufficient_data", "prep_notes": ["target_before_current_time"]}

            actual_current_price = float(current_price)

            # Optional debug about current price mismatch
            if not window_1w.empty:
                latest_close_1w = float(window_1w["Close"].iloc[-1])
                if latest_close_1w > 0:
                    rel_diff = abs(latest_close_1w - actual_current_price) / latest_close_1w
                    if rel_diff > 0.02:
                        self._dbg("warning", f"⚠️ current_price {actual_current_price:,.2f} differs from latest 1w close "
                                             f"{latest_close_1w:,.2f} by {rel_diff*100:.1f}%.")

            def _summary(df: pd.DataFrame, label: str) -> Dict[str, Any]:
                if df is None or df.empty:
                    return {}
                start_price = float(df["Close"].iloc[0])
                ann_sqrt = self._annualization_sqrt(df.index)
                return {
                    "period": label,
                    "start_price": start_price,
                    "high": float(df["High"].max()),
                    "low": float(df["Low"].min()),
                    "price_change_pct": float((actual_current_price - start_price) / max(1e-9, start_price) * 100.0),
                    "volatility_ann_pct": float(df["Close"].pct_change().std() * ann_sqrt * 100.0),
                    "avg_volume": float(df["Volume"].mean()),
                    "start_date": str(df.index[0]),
                    "end_date": str(df.index[-1]),
                }

            data_3m_summary = _summary(window_3m, "3 months") if not window_3m.empty else {}
            data_1w_summary = _summary(window_1w, "1 week") if not window_1w.empty else {}

            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w, actual_current_price)
            enhanced_data = self._prepare_enhanced_chart_data(window_3m, window_1w, indicators_3m, indicators_1w)

            if enhanced_data.get("3m_data", {}).get("period_highs_lows"):
                data_3m_summary["high"] = enhanced_data["3m_data"]["period_highs_lows"]["period_high"]
                data_3m_summary["low"] = enhanced_data["3m_data"]["period_highs_lows"]["period_low"]
            if enhanced_data.get("1w_data", {}).get("period_highs_lows"):
                data_1w_summary["high"] = enhanced_data["1w_data"]["period_highs_lows"]["period_high"]
                data_1w_summary["low"] = enhanced_data["1w_data"]["period_highs_lows"]["period_low"]

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
            }

        except Exception as e:
            st.error(f"Error preparing analysis data: {str(e)}")
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
                    return float(ind["EMA_20"].iloc[-1] - ind["EMA_20"].iloc[-k-1])
            except Exception:
                pass
            return None

        if data_1w is not None and not data_1w.empty:
            feats["ret_1w_last5bars"] = last_n_returns(data_1w, 5)
            feats["vol_ann_1w_pct"] = float(data_1w["Close"].pct_change().std() * self._annualization_sqrt(data_1w.index) * 100.0)
        if ind_1w:
            feats["bb_width_1w"] = bb_width(ind_1w)
            feats["ema20_slope_1w"] = ema_slope(ind_1w, 5)
            feats["rsi_last_1w"] = float(ind_1w["RSI"].iloc[-1]) if "RSI" in ind_1w and not np.isnan(ind_1w["RSI"].iloc[-1]) else None

        if data_3m is not None and not data_3m.empty:
            feats["ret_3m_last5bars"] = last_n_returns(data_3m, 5)
            feats["vol_ann_3m_pct"] = float(data_3m["Close"].pct_change().std() * self._annualization_sqrt(data_3m.index) * 100.0)
        if ind_3m:
            feats["bb_width_3m"] = bb_width(ind_3m)
            feats["ema20_slope_3m"] = ema_slope(ind_3m, 5)
            feats["rsi_last_3m"] = float(ind_3m["RSI"].iloc[-1]) if "RSI" in ind_3m and not np.isnan(ind_3m["RSI"].iloc[-1]) else None

        return feats

    # ---------- enhanced chart data ----------

    def _prepare_enhanced_chart_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
    ) -> Dict[str, Any]:
        try:
            enhanced: Dict[str, Any] = {}

            full_3m = data_3m if data_3m is not None else pd.DataFrame()
            full_1w = data_1w if data_1w is not None else pd.DataFrame()

            full_3m_high = float(full_3m["High"].max()) if not full_3m.empty else None
            full_3m_low = float(full_3m["Low"].min()) if not full_3m.empty else None
            full_1w_high = float(full_1w["High"].max()) if not full_1w.empty else None
            full_1w_low = float(full_1w["Low"].min()) if not full_1w.empty else None

            display_from_3m = getattr(data_3m, "attrs", {}).get("display_from_index", 0) if data_3m is not None else 0
            display_from_1w = getattr(data_1w, "attrs", {}).get("display_from_index", 0) if data_1w is not None else 0

            recent_3m = full_3m.iloc[display_from_3m:] if not full_3m.empty else full_3m
            recent_1w = full_1w.iloc[display_from_1w:] if not full_1w.empty else full_1w

            tail_3m = recent_3m.tail(50) if not recent_3m.empty else recent_3m
            tail_1w = recent_1w.tail(30) if not recent_1w.empty else recent_1w

            enhanced["3m_data"] = {
                "timeframe": "3-month",
                "full_range": (
                    f"{pd.Timestamp(full_3m.index[0]).strftime('%B %d')} to "
                    f"{pd.Timestamp(full_3m.index[-1]).strftime('%B %d, %Y')}"
                ) if not full_3m.empty else "N/A",
                "data_range": (
                    f"{pd.Timestamp(tail_3m.index[0]).strftime('%B %d')} to "
                    f"{pd.Timestamp(tail_3m.index[-1]).strftime('%B %d, %Y')}"
                ) if not tail_3m.empty else "N/A",
                "period_highs_lows": {
                    "period_high": full_3m_high,
                    "period_low": full_3m_low,
                    "recent_high": float(recent_3m["High"].max()) if not recent_3m.empty else None,
                    "recent_low": float(recent_3m["Low"].min()) if not recent_3m.empty else None,
                },
                "recent_prices": (
                    {
                        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d %H:%M") for d in tail_3m.index],
                        "open": tail_3m["Open"].round(2).tolist(),
                        "high": tail_3m["High"].round(2).tolist(),
                        "low": tail_3m["Low"].round(2).tolist(),
                        "close": tail_3m["Close"].round(2).tolist(),
                        "volume": tail_3m["Volume"].round(0).tolist(),
                    } if not tail_3m.empty else {}
                ),
                "indicators": {},
            }

            for indicator in [
                "RSI","MACD","MACD_Signal","MACD_Histogram",
                "BB_Upper","BB_Lower","BB_Middle","EMA_20","SMA_50","SMA_200",
            ]:
                if indicators_3m and indicator in indicators_3m:
                    values = indicators_3m[indicator].dropna().tail(50)
                    enhanced["3m_data"]["indicators"][indicator] = values.round(4).tolist()

            enhanced["1w_data"] = {
                "timeframe": "1-week",
                "full_range": (
                    f"{pd.Timestamp(full_1w.index[0]).strftime('%B %d')} to "
                    f"{pd.Timestamp(full_1w.index[-1]).strftime('%B %d, %Y')}"
                ) if not full_1w.empty else "N/A",
                "data_range": (
                    f"{pd.Timestamp(tail_1w.index[0]).strftime('%B %d')} to "
                    f"{pd.Timestamp(tail_1w.index[-1]).strftime('%B %d, %Y')}"
                ) if not tail_1w.empty else "N/A",
                "period_highs_lows": {
                    "period_high": full_1w_high,
                    "period_low": full_1w_low,
                    "recent_high": float(recent_1w["High"].max()) if not recent_1w.empty else None,
                    "recent_low": float(recent_1w["Low"].min()) if not recent_1w.empty else None,
                },
                "recent_prices": (
                    {
                        "dates": [pd.Timestamp(d).strftime("%Y-%m-%d %H:%M") for d in tail_1w.index],
                        "open": tail_1w["Open"].round(2).tolist(),
                        "high": tail_1w["High"].round(2).tolist(),
                        "low": tail_1w["Low"].round(2).tolist(),
                        "close": tail_1w["Close"].round(2).tolist(),
                        "volume": tail_1w["Volume"].round(0).tolist(),
                    } if not tail_1w.empty else {}
                ),
                "indicators": {},
            }

            for indicator in [
                "RSI","MACD","MACD_Signal","MACD_Histogram",
                "BB_Upper","BB_Lower","BB_Middle","EMA_20","SMA_50","SMA_200",
            ]:
                if indicators_1w and indicator in indicators_1w:
                    values = indicators_1w[indicator].dropna().tail(30)
                    enhanced["1w_data"]["indicators"][indicator] = values.round(4).tolist()

            v50 = full_3m["Volume"].tail(50).mean() if not full_3m.empty else None
            v10 = full_3m["Volume"].tail(10).mean() if not full_3m.empty else None
            v30 = full_1w["Volume"].tail(30).mean() if not full_1w.empty else None
            v5 = full_1w["Volume"].tail(5).mean() if not full_1w.empty else None

            enhanced["volume_analysis"] = {
                "3m_avg_volume_tail50": float(v50) if v50 is not None else None,
                "3m_volume_trend": ("increasing" if (v10 is not None and v50 is not None and v10 > v50) else
                                    ("decreasing" if (v10 is not None and v50 is not None and v10 < v50) else None)),
                "1w_avg_volume_tail30": float(v30) if v30 is not None else None,
                "1w_volume_trend": ("increasing" if (v5 is not None and v30 is not None and v5 > v30) else
                                    ("decreasing" if (v5 is not None and v30 is not None and v5 < v30) else None)),
            }

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
        summary: Dict[str, Any] = {"current_price": float(current_price)}
        try:
            rsi_3m_last = indicators_3m["RSI"].iloc[-1] if indicators_3m and "RSI" in indicators_3m and not indicators_3m["RSI"].empty else np.nan
            rsi_1w_last = indicators_1w["RSI"].iloc[-1] if indicators_1w and "RSI" in indicators_1w and not indicators_1w["RSI"].empty else np.nan
            summary["RSI"] = {
                "3m_current": float(rsi_3m_last) if not np.isnan(rsi_3m_last) else None,
                "1w_current": float(rsi_1w_last) if not np.isnan(rsi_1w_last) else None,
            }

            if indicators_3m and "MACD" in indicators_3m and "MACD_Signal" in indicators_3m:
                macd_3m = indicators_3m["MACD"].iloc[-1]
                signal_3m = indicators_3m["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_3m) and not np.isnan(signal_3m):
                    summary["MACD_3m"] = {"macd": float(macd_3m), "signal": float(signal_3m),
                                          "crossover": "bullish" if macd_3m > signal_3m else "bearish"}

            if indicators_1w and "MACD" in indicators_1w and "MACD_Signal" in indicators_1w:
                macd_1w = indicators_1w["MACD"].iloc[-1]
                signal_1w = indicators_1w["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_1w) and not np.isnan(signal_1w):
                    summary["MACD_1w"] = {"macd": float(macd_1w), "signal": float(signal_1w),
                                          "crossover": "bullish" if macd_1w > signal_1w else "bearish"}

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and all(k in inds for k in ["BB_Upper", "BB_Lower", "BB_Middle"]):
                    upper = inds["BB_Upper"].iloc[-1]; lower = inds["BB_Lower"].iloc[-1]; middle = inds["BB_Middle"].iloc[-1]
                    if not any(np.isnan([upper, lower, middle])):
                        cp = summary["current_price"]
                        summary[f"BB_{timeframe}"] = {
                            "upper": float(upper), "lower": float(lower), "middle": float(middle),
                            "position": "above_upper" if cp > upper else ("below_lower" if cp < lower else "within_bands"),
                        }

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and "EMA_20" in inds:
                    ema_20 = inds["EMA_20"].iloc[-1]
                    if not np.isnan(ema_20):
                        cp = summary["current_price"]
                        summary[f"EMA_20_{timeframe}"] = {"value": float(ema_20),
                                                          "trend": "bullish" if cp > ema_20 else "bearish"}

        except Exception as e:
            st.warning(f"Error summarizing indicators: {str(e)}")

        return summary

    # ---------- prompt + model call ----------

    def _build_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        system_content = (
            "You are a deterministic technical analyst. Use ONLY the structured arrays provided. "
            "Forbidden: news, macro, on-chain, session effects (market open/close), weekdays, seasonality, etc. "
            "Do not infer beyond given timestamps. If required arrays are missing/empty, output status='insufficient_data'."
        )

        data_3m = analysis_data.get("data_3m", {})
        data_1w = analysis_data.get("data_1w", {})

        # JSON schema
        output_schema = {
            "status": "ok or insufficient_data",
            "asset": asset_name,
            "as_of": analysis_data.get("current_time"),
            "target_ts": analysis_data.get("target_time"),
            "predicted_price": "number or null",
            "p_up": "float 0..1",
            "p_down": "float 0..1 (p_up + p_down = 1)",
            "conf_overall": "float 0..1",
            "conf_price": "float 0..1",
            "expected_pct_move": "signed float percent (optional; compute from predicted/current if omitted)",
            "critical_levels": {"bullish_above": "number or null", "bearish_below": "number or null"},
            "evidence": [
                {"type":"rsi|macd|bb|ema|price|volume|structure",
                 "timeframe":"3m|1w","ts":"YYYY-mm-dd HH:MM","value":"number or object","note":"short factual note"}
            ],
            "notes": ["if status=insufficient_data, list what's missing"]
        }

        # Narrative section template (your original)
        narrative_template = """
[TECHNICAL_ANALYSIS_START]
**COMPREHENSIVE TECHNICAL ANALYSIS**

**Current Price: ${current_price:,.2f}**

**1. MULTI-TIMEFRAME OVERVIEW**
- 3-Month Chart Analysis: <facts only from arrays>
- 1-Week Chart Analysis: <facts only from arrays>
- Timeframe Alignment: <consistency or conflict>

**2. TECHNICAL INDICATORS ANALYSIS**
- RSI Analysis: <levels/divergences>
- MACD Analysis: <crossovers/histogram>
- Bollinger Bands: <within/upper/lower, squeeze/expansion>
- EMA Analysis: <price vs EMA_20>

**3. ADVANCED PATTERN ANALYSIS**
- Patterns & Candles: <if evident in arrays; else 'none observed'>
- Support/Resistance: <explicit levels from highs/lows>

**4. DIVERGENCE & FAILURE SWINGS**
- RSI/MACD divergences and failure swings: <only if visible>
- Volume divergences: <only if visible>

**5. TRENDLINE & STRUCTURE**
- Higher highs/lows or lower highs/lows
- Key levels

**6. TRADING RECOMMENDATION**
- Overall Bias: **BULLISH/BEARISH/NEUTRAL**
- Entry/Stop/Targets: <levels based on arrays>
[TECHNICAL_ANALYSIS_END]

[PRICE_PREDICTION_START]
**PREDICTED PRICE: I predict {asset} will be at $<PRICE> on {target_formatted}**

⏰ **DATA-BASED TIME ANALYSIS:**
- Exact Target: {target_ts}
- Momentum Direction: <from RSI/MACD arrays>
- Trend Strength: <from EMA/price action>
- Volume Analysis: <from volume arrays>
- Expected Path: <data-grounded path>

1. **Probability HIGHER than ${current_price:,.2f}: <X>%**
2. **Probability LOWER than ${current_price:,.2f}: <Y>%**
3. **Overall Analysis Confidence: <Z>%**
4. **Price Prediction Confidence: <W>%**
5. **Expected % Move: <±M>%**

**Key Technical Factors from the Actual Data:**
- Factor 1…
- Factor 2…
- Factor 3…

**Price Targets Based on Chart Analysis:**
- Upside Target 1: $<amount>
- Upside Target 2: $<amount>
- Downside Target 1: $<amount>
- Downside Target 2: $<amount>

**Critical Levels from the Data:**
- Bullish above: $<level>
- Bearish below: $<level>
[PRICE_PREDICTION_END]
""".strip()

        user_content = f"""
ASSET: {asset_name}

DATA WINDOWS (do not infer beyond them):
- 3m index range: {data_3m.get('start_date','N/A')} → {data_3m.get('end_date','N/A')}
- 1w index range: {data_1w.get('start_date','N/A')} → {data_1w.get('end_date','N/A')}

TARGET (timestamp string; do not alter): {analysis_data.get('target_time')}
AS_OF (timestamp string; do not alter): {analysis_data.get('current_time')}
CURRENT_PRICE (use exactly this number): {analysis_data.get('current_price')}

INDICATOR SNAPSHOT:
{json.dumps(analysis_data.get('indicators', {}), indent=2)}

FEATURES:
{json.dumps(analysis_data.get('features', {}), indent=2)}

ENHANCED ARRAYS:
{json.dumps(analysis_data.get('enhanced_chart_data', {}), indent=2)}

STRICT RULES:
- Use ONLY these arrays. Do NOT use session effects, news, or any outside info.
- If required arrays are empty/missing, return status='insufficient_data' with 'notes'.
- All levels/claims must be verifiable from these arrays.

YOU MUST RETURN **TWO** BLOCKS, IN THIS ORDER:

1) A VALID JSON object, in a fenced code block like:
```json
{json.dumps(output_schema, indent=2)}
Fill all applicable fields with numbers (not strings) where appropriate.

A narrative block using EXACTLY the following section markers and structure,
filling in concrete values from the arrays. This is the display copy for the app:

{narrative_template.format(
current_price=analysis_data.get('current_price'),
asset=asset_name,
target_formatted=analysis_data.get('target_time'),
target_ts=analysis_data.get('target_time')
)}
""".strip()

        return ({"role": "system", "content": system_content},
                {"role": "user", "content": user_content})

    def _generate_technical_analysis_gpt5(self, analysis_data: Dict[str, Any]) -> str:
        """
        Call the model WITHOUT temperature/top_p/reasoning/text params.
        Ask for JSON + narrative blocks. Return raw text.
        """
        if not self.gpt5_client:
            return '{"status":"insufficient_data","notes":["no_client"]}'

        asset_name = analysis_data.get("asset_name", "Asset")
        system_msg, user_msg = self._build_messages(analysis_data, asset_name)

        # DEBUG: Print the full prompt being sent to OpenAI
        print("\n" + "="*80)
        print("🤖 FULL PROMPT BEING SENT TO OPENAI/CHATGPT")
        print("="*80)
        print("\n📋 SYSTEM MESSAGE:")
        print("-" * 40)
        print(system_msg["content"])
        print("\n📊 USER MESSAGE (WITH ALL DATA POINTS):")
        print("-" * 40)
        print(user_msg["content"])
        print("\n" + "="*80)
        print("END OF PROMPT")
        print("="*80 + "\n")

        # Responses API
        try:
            resp = self.gpt5_client.responses.create(
                model=self.model_name,
                input=[system_msg, user_msg],
            )
            return resp.output_text  # raw text with both blocks
        except Exception as e:
            self._dbg("warning", f"Responses API failed, will try Chat Completions. Error: {e}")

        # Fallback: Chat Completions (text)
        try:
            chat = self.gpt5_client.chat.completions.create(
                model=self.model_name,
                messages=[system_msg, user_msg],
            )
            return chat.choices[0].message.content
        except Exception as e:
            return json.dumps({"status": "insufficient_data", "notes": [f"model_error:{str(e)}"]})

    # ---------- parsing + probabilities + text composition ----------

    def _split_dual_output(self, raw: str) -> Tuple[str, str]:
        """Extract a JSON fenced block (```json ... ```) and the rest (narrative)."""
        if not raw:
            return "", ""

        # Prefer fenced ```json blocks
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            json_block = m.group(1).strip()
            narrative = raw[:m.start()] + raw[m.end():]
            return json_block, narrative.strip()

        # Fallback: first JSON object in text (greedy braces balance heuristic)
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw[first_brace:last_brace+1]
            # Quick sanity: try to loads
            try:
                json.loads(candidate)
                narrative = raw[:first_brace] + raw[last_brace+1:]
                return candidate.strip(), narrative.strip()
            except Exception:
                pass

        # Nothing extractable; return raw as narrative
        return "", raw.strip()

    def _parse_json_response(self, response_text: str, current_price: float) -> Dict[str, Any]:
        if not response_text:
            return {"status": "insufficient_data", "notes": ["no_json_block_found"]}

        try:
            data = json.loads(response_text)
            if not isinstance(data, dict):
                return {"status": "insufficient_data", "notes": ["non_dict_json"]}

            data["status"] = data.get("status") if data.get("status") in ("ok","insufficient_data") else "insufficient_data"

            # Normalize probabilities
            p_up = float(self._safe_num(data.get("p_up"), 0.5))
            p_down = float(self._safe_num(data.get("p_down"), 0.5))
            p_up = max(0.0, min(1.0, p_up))
            p_down = max(0.0, min(1.0, p_down))
            total = p_up + p_down
            if total == 0:
                p_up, p_down = 0.5, 0.5
            else:
                p_up, p_down = p_up/total, p_down/total
            data["p_up"], data["p_down"] = p_up, p_down

            # Compute expected % move if we have predicted_price
            cp = float(current_price) if current_price is not None else None
            pred = self._safe_num(data.get("predicted_price"), None)
            if pred is not None and cp and cp > 0:
                data["expected_pct_move"] = (float(pred) - cp) / cp * 100.0

            for key in ("conf_overall", "conf_price"):
                if key in data and data[key] is not None:
                    data[key] = max(0.0, min(1.0, float(self._safe_num(data[key], 0.5))))

            return data
        except Exception as e:
            return {"status": "insufficient_data", "notes": [f"json_parse_error:{e}"]}

    def _parse_comprehensive_response(self, response: str) -> Dict[str, str]:
        """Parse narrative into sections using your original markers."""
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
                # fallback parsing (very lenient)
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
                    elif "price prediction" in line_lower:
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

    def _extract_probabilities_from_json(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("status") != "ok":
            return self._default_probs()

        p_up = float(self._safe_num(data.get("p_up"), 0.5))
        p_down = float(self._safe_num(data.get("p_down"), 0.5))
        conf_overall = float(self._safe_num(data.get("conf_overall"), 0.5))
        conf_price = float(self._safe_num(data.get("conf_price"), 0.5))
        pred = self._safe_num(data.get("predicted_price"), None)

        move_pct = 0.0
        cp = float(current_price) if current_price else None
        if pred is not None and cp and cp > 0:
            move_pct = (float(pred) - cp) / cp * 100.0

        return {
            "higher_fraction": p_up,
            "lower_fraction": p_down,
            "confidence_fraction": conf_overall,
            "higher_pct": p_up * 100.0,
            "lower_pct": p_down * 100.0,
            "confidence_pct": conf_overall * 100.0,
            "predicted_price": float(pred) if pred is not None else None,
            "price_confidence_pct": conf_price * 100.0,
            "move_percentage": float(move_pct),
        }

    def _default_probs(self) -> Dict[str, Any]:
        return {
            "higher_fraction": 0.5, "lower_fraction": 0.5, "confidence_fraction": 0.5,
            "higher_pct": 50.0, "lower_pct": 50.0, "confidence_pct": 50.0,
            "predicted_price": None, "price_confidence_pct": 50.0, "move_percentage": 0.0
        }

    def _safe_num(self, x: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    # ---------- text composition + legacy regex fallback ----------

    def _compose_text_from_model_json(self, data: Dict[str, Any], current_price: float) -> Tuple[str, str]:
        if not isinstance(data, dict):
            return ("Analysis not available", "Prediction not available")

        status = data.get("status", "insufficient_data")
        as_of = data.get("as_of", "")
        target_ts = data.get("target_ts", "")
        pred = data.get("predicted_price", None)
        p_up = data.get("p_up", 0.5); p_down = data.get("p_down", 0.5)
        expected = data.get("expected_pct_move", None)
        crit = data.get("critical_levels", {}) or {}
        bull = crit.get("bullish_above", None)
        bear = crit.get("bearish_below", None)

        if status != "ok":
            notes = data.get("notes", [])
            msg = "; ".join([str(n) for n in notes]) if notes else "insufficient data"
            return (
                f"**Status:** _insufficient data_\n\n**Notes:** {msg}",
                f"**Target:** `{target_ts}`\n\n_No price prediction due to insufficient data._"
            )

        bullets: List[str] = []
        if bull is not None:
            bullets.append(f"- **Bullish above:** ${bull:,.0f}")
        if bear is not None:
            bullets.append(f"- **Bearish below:** ${bear:,.0f}")
        if expected is not None:
            bullets.append(f"- **Expected move:** {expected:+.2f}% vs current (${current_price:,.0f})")
        ev = data.get("evidence", []) or []
        for e in ev[:6]:
            t = e.get("type","fact"); tf = e.get("timeframe",""); ts = e.get("ts",""); note = e.get("note","")
            bullets.append(f"- **{t.upper()} {tf} @ {ts}:** {note}")

        tech_md = f"**As of:** `{as_of}`\n\n" + ("\n".join(bullets) if bullets else "_No additional evidence provided._")
        price_line = f"**Predicted price at `{target_ts}`:** " + (f"**${pred:,.0f}**" if pred is not None else "unavailable")
        pred_md = f"{price_line}\n\n- **P(higher)**: {p_up*100:.0f}%   - **P(lower)**: {p_down*100:.0f}%   - **AI confidence**: {data.get('conf_overall',0.5)*100:.0f}%"
        return tech_md, pred_md

    def _compose_text_when_insufficient(self, reason: str, target_ts: str) -> Tuple[str, str]:
        tech = f"**Status:** _insufficient data_\n\n**Notes:** {reason or 'missing inputs'}"
        pred = f"**Target:** `{target_ts}`\n\n_No price prediction due to insufficient data._"
        return tech, pred

    def _extract_probabilities(self, prediction_text: str) -> Dict[str, Any]:
        """Legacy regex extraction from narrative price section."""
        probs = {"higher_fraction": 0.5, "lower_fraction": 0.5, "confidence_fraction": 0.5,
                 "higher_pct": 50.0, "lower_pct": 50.0, "confidence_pct": 50.0, "predicted_price": None,
                 "price_confidence_pct": 50.0, "move_percentage": 0.0}
        try:
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
                    r"overall.*?confidence.*?(\d+)%",
                    r"analysis.*?confidence.*?(\d+)%",
                    r"confidence.*?(\d+)%",
                    r"(\d+)%.*?confidence",
                    r"confident.*?(\d+)%",
                ],
                "price_confidence": [
                    r"price.*?confidence.*?(\d+)%",
                    r"price.*?prediction.*?confidence.*?(\d+)%",
                    r"target.*?confidence.*?(\d+)%",
                ],
                "move_percentage": [
                    r"([+-]?\d+\.?\d*)%.*?move",
                    r"move.*?([+-]?\d+\.?\d*)%",
                    r"expected.*?([+-]?\d+\.?\d*)%",
                    r"change.*?([+-]?\d+\.?\d*)%",
                ],
                "predicted_price": [
                    r"predict.*?\$([\d,]+)(?:\.\d+)?",
                    r"predicted price.*?\$([\d,]+)(?:\.\d+)?",
                    r"will be.*?\$([\d,]+)(?:\.\d+)?",
                    r"target.*?\$([\d,]+)(?:\.\d+)?",
                    r"bitcoin.*?\$([\d,]+)(?:\.\d+)?",
                ],
            }

            text_lower = prediction_text.lower()

            for key in ["higher", "lower", "confidence", "price_confidence", "move_percentage", "predicted_price"]:
                for pat in patterns[key]:
                    m = re.findall(pat, text_lower, flags=re.IGNORECASE)
                    if m:
                        if key == "predicted_price":
                            price_str = m[0].replace(',', '')
                            probs["predicted_price"] = float(price_str)
                        else:
                            val = float(m[0])
                            if key == "higher":
                                probs["higher_pct"] = val
                            elif key == "lower":
                                probs["lower_pct"] = val
                            elif key == "confidence":
                                probs["confidence_pct"] = val
                            elif key == "price_confidence":
                                probs["price_confidence_pct"] = val
                            elif key == "move_percentage":
                                probs["move_percentage"] = val
                        break

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