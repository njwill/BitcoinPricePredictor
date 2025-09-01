from __future__ import annotations

# ai_analysis_fixed.py — upgraded
# - Full rewrite with robust JSON parsing (last fenced block), safer prep,
#   richer engineered features, dynamic tails, divergence-ready pivots,
#   SMA cross, ATR/OBV, volume thresholds, and target/bars metadata.

import os
import json
import re
import contextlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd
import pytz

# --- Import OpenAI client from separate module ---
from openai_client import OpenAIWrapper

# --- Streamlit-safe import (falls back to console logger) ---
try:  # noqa: SIM105
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    class _DummySt:
        def write(self, *a, **k): print(*a)
        def warning(self, *a, **k): print("[warning]", *a)
        def error(self, *a, **k): print("[error]", *a)
        def info(self, *a, **k): print("[info]", *a)
    st = _DummySt()  # type: ignore


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

    def __init__(self, debug: Optional[bool] = None):
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)

        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT5_MODEL", "gpt-5")
        self.gpt = OpenAIWrapper(self.openai_key, self.model_name, debug=self.debug)
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
        if not self.gpt or not self.gpt.client:
            return {"status": "error", "error": "GPT client not initialized"}

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

            # Validate/adjust predicted price using volatility bands and horizon
            parsed_json = self._validate_and_adjust_prediction_block(
                parsed_json,
                current_price=self._last_current_price or current_price,
                bars_to_target=analysis_data.get("bars_to_target"),
                atr_pct=analysis_data.get("features", {}).get("atr_pct_1w") or analysis_data.get("features", {}).get("atr_pct_3m"),
                sigma=analysis_data.get("features", {}).get("ret_sigma_base"),
            )

            # Extract narrative sections
            sections = self._parse_comprehensive_response(narrative_text) if narrative_text else {}

            # Build probabilities
            if parsed_json.get("status") == "ok":
                probs = self._extract_probabilities_from_json(parsed_json, self._last_current_price or current_price)
            else:
                # Improved fallback: heuristic probabilities based on features (better than regex-only)
                probs = self._heuristic_probabilities(analysis_data)
                # Try to enhance with regex if narrative provided specific numbers
                if sections:
                    p_from_text = self._extract_probabilities(sections.get("price_prediction", ""))
                    # Prefer explicit narrative percentages if present
                    if any(k in p_from_text and p_from_text[k] != probs.get(k) for k in ("higher_pct", "lower_pct")):
                        probs.update(p_from_text)

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
            target_ts = target_datetime.isoformat() if target_datetime else ""
            tech, pred = self._compose_text_when_insufficient(str(e), target_ts)
            return {
                "status": "error",
                "error": str(e),
                "probabilities": self._default_probs(),
                "technical_summary": tech,
                "price_prediction": pred,
            }

    # ---------- helpers: debug ----------

    def _dbg(self, level: str, msg: str) -> None:
        if not self.debug:
            return
        fn = getattr(st, level, None)
        (fn or st.write)(msg)

    # ---------- helpers: dataframe hygiene ----------

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()

        # If numeric index and there is a timestamp-like column, use it
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
            # Force to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            df.index = pd.to_datetime(df.index, errors="coerce")

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        return df

    def _coerce_ohlcv_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = df.copy()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        return df

    def _limit_to_days(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        df = self._ensure_datetime_index(df)
        try:
            end = pd.Timestamp(df.index.max()) if pd.notna(df.index.max()) else pd.Timestamp("now", tz=pytz.UTC)
        except Exception:
            end = pd.Timestamp("now", tz=pytz.UTC)
        start = end - pd.Timedelta(days=days)
        return df.loc[df.index >= start]

    def _annualization_sqrt(self, index: pd.Index) -> float:
        # Use median step for robustness
        try:
            if index is None or len(index) < 2:
                return 1.0
            step = self._infer_index_step(index) or pd.Timedelta(days=1)
            seconds = float(step.total_seconds() or 86400)
            periods_per_year = (365 * 24 * 3600) / seconds
            return float(np.sqrt(periods_per_year))
        except Exception:
            return 1.0

    def _infer_index_step(self, index: pd.Index) -> Optional[pd.Timedelta]:
        if index is None or len(index) < 2:
            return None
        try:
            diffs = pd.Series(index[1:]).reset_index(drop=True) - pd.Series(index[:-1]).reset_index(drop=True)
            diffs = diffs[diffs > pd.Timedelta(0)]
            if diffs.empty:
                return None
            # Median step is robust to outliers
            return pd.to_timedelta(np.median(diffs.values))
        except Exception:
            return None

    def _determine_timezone(self, *indexes: Optional[pd.Index]):
        for idx in indexes:
            if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
                return idx.tz
        # Default to UTC for crypto
        return pytz.UTC

    # ---------- series compression and packing helpers ----------

    def _compress_series(self, s: pd.Series, max_points: int, keep_tail: int) -> pd.Series:
        """Downsample series uniformly while preserving the last keep_tail points."""
        s = s.dropna()
        n = len(s)
        if n <= max_points:
            return s
        keep_tail = min(keep_tail, n)
        tail = s.iloc[-keep_tail:]
        head_n = max_points - keep_tail
        if head_n <= 0:
            return tail
        # Evenly pick head_n points from the head portion
        head = s.iloc[:-keep_tail]
        idx = np.linspace(0, len(head) - 1, num=head_n, dtype=int)
        sampled_head = head.iloc[idx]
        return pd.concat([sampled_head, tail])

    def _pack_series(self, s: pd.Series, max_points: int, keep_tail: int, round_to: int = 4) -> Dict[str, Any]:
        s = s.dropna()
        s_comp = self._compress_series(s, max_points=max_points, keep_tail=keep_tail)
        return {
            "dates": [self._safe_format_datetime(d) for d in s_comp.index],
            "values": s_comp.round(round_to).astype(float).tolist(),
        }

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

            tz = self._determine_timezone(
                window_1w.index if not window_1w.empty else None,
                window_3m.index if not window_3m.empty else None,
            )
            current_time = datetime.now(tz)

            # Target handling and step inference (use 1w if available as base for horizon)
            base_df = window_1w if not window_1w.empty else window_3m
            step = self._infer_index_step(base_df.index)
            if target_datetime:
                target = target_datetime.astimezone(tz) if target_datetime.tzinfo else tz.localize(target_datetime)
            else:
                if step is None:
                    return {"prep_status": "insufficient_data", "prep_notes": ["no_target_and_cannot_infer_step"]}
                last_val = base_df.index[-1] if pd.notna(base_df.index[-1]) else pd.Timestamp("now", tz=tz)
                target = pd.Timestamp(last_val) + step
                target = target.tz_convert(tz) if target.tzinfo else target.tz_localize(tz)

            if (target - current_time).total_seconds() < 0:
                return {"prep_status": "insufficient_data", "prep_notes": ["target_before_current_time"]}

            # Compute bars_to_target
            step_seconds = float((step or pd.Timedelta(hours=1)).total_seconds() or 3600.0)
            last_ts = base_df.index[-1]
            delta_sec = max(0.0, (target - last_ts).total_seconds())
            bars_to_target = int(math.ceil(delta_sec / step_seconds)) if step_seconds > 0 else 1

            actual_current_price = float(current_price)

            # Optional debug about current price mismatch vs last 1w close
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
                    "price_change_pct": float((actual_current_price / max(1e-9, start_price) - 1.0) * 100.0),
                    "volatility_ann_pct": float(df["Close"].pct_change().std() * ann_sqrt * 100.0),
                    "avg_volume": float(df["Volume"].mean()) if "Volume" in df.columns else None,
                    "start_date": str(df.index[0]),
                    "end_date": str(df.index[-1]),
                }

            data_3m_summary = _summary(window_3m, "3 months") if not window_3m.empty else {}
            data_1w_summary = _summary(window_1w, "1 week") if not window_1w.empty else {}

            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w, actual_current_price)

            enhanced_data = self._prepare_enhanced_chart_data(
                window_3m, window_1w, indicators_3m, indicators_1w, max_points_3m=350, keep_tail_3m=120,
                max_points_1w=220, keep_tail_1w=60
            )

            # override highs/lows with period data if present
            if enhanced_data.get("3m_data", {}).get("period_highs_lows") and data_3m_summary:
                data_3m_summary["high"] = enhanced_data["3m_data"]["period_highs_lows"].get("period_high")
                data_3m_summary["low"] = enhanced_data["3m_data"]["period_highs_lows"].get("period_low")
            if enhanced_data.get("1w_data", {}).get("period_highs_lows") and data_1w_summary:
                data_1w_summary["high"] = enhanced_data["1w_data"]["period_highs_lows"].get("period_high")
                data_1w_summary["low"] = enhanced_data["1w_data"]["period_highs_lows"].get("period_low")

            # Richer features (uses aligned series)
            features = self._compute_features(
                window_3m, window_1w, indicators_3m, indicators_1w, bars_to_target=bars_to_target
            )

            availability_flags = self._availability_flags(indicators_3m, indicators_1w, window_3m, window_1w)

            # Baseline forecast scaffold (for horizon)
            baseline = self._baseline_forecast(window_1w if not window_1w.empty else window_3m, bars_to_target, actual_current_price)
            features.update(baseline)

            return {
                "asset_name": asset_name,
                "current_time": current_time.isoformat(),
                "target_time": target.isoformat(),
                "hours_until_target": (target - current_time).total_seconds() / 3600.0,
                "step_seconds": step_seconds,
                "bars_to_target": bars_to_target,
                "data_3m": data_3m_summary,
                "data_1w": data_1w_summary,
                "indicators": indicators_summary,
                "enhanced_chart_data": enhanced_data,
                "features": features,
                "availability_flags": availability_flags,
                "current_price": actual_current_price,
            }

        except Exception as e:
            st.error(f"Error preparing analysis data: {e}")
            return {"prep_status": "insufficient_data", "prep_notes": [str(e)]}

    # ---------- feature engineering ----------

    def _compute_features(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        ind_3m: Dict[str, pd.Series],
        ind_1w: Dict[str, pd.Series],
        bars_to_target: int,
    ) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}

        def compounded_return(df: pd.DataFrame, n: int) -> Optional[float]:
            try:
                r = df["Close"].pct_change().tail(n)
                return float((r + 1.0).prod() - 1.0)
            except Exception:
                return None

        def bb_width_series(inds: Dict[str, pd.Series]) -> Optional[pd.Series]:
            try:
                if inds and all(k in inds for k in ("BB_Upper", "BB_Lower", "BB_Middle")):
                    upper = inds["BB_Upper"]
                    lower = inds["BB_Lower"]
                    middle = inds["BB_Middle"]
                    s = (upper - lower) / middle.replace(0, np.nan)
                    return s.dropna()
            except Exception:
                return None
            return None

        def lr_slope(y: pd.Series, n: int) -> Optional[float]:
            try:
                y = y.dropna().tail(n)
                if len(y) < 3:
                    return None
                x = np.arange(len(y))
                b1, b0 = np.polyfit(x, y.values, 1)
                return float(b1)
            except Exception:
                return None

        def percentile_of_last(s: pd.Series, lookback: int) -> Optional[float]:
            try:
                s = s.dropna().tail(lookback)
                if len(s) < 5:
                    return None
                last = s.iloc[-1]
                rank = (s <= last).mean()
                return float(rank * 100.0)
            except Exception:
                return None

        def rsi_stats(inds: Dict[str, pd.Series], n_short: int = 10, n_long: int = 20) -> Dict[str, Optional[float]]:
            out = {"rsi_last": None, "rsi_slope_short": None, "rsi_mean_short": None, "rsi_mean_long": None, "time_over_50": None}
            try:
                if not inds or "RSI" not in inds or inds["RSI"].empty:
                    return out
                r = inds["RSI"].dropna()
                out["rsi_last"] = float(r.iloc[-1]) if len(r) else None
                out["rsi_slope_short"] = lr_slope(r, n_short)
                out["rsi_mean_short"] = float(r.tail(n_short).mean()) if len(r) >= n_short else (float(r.mean()) if len(r) else None)
                out["rsi_mean_long"] = float(r.tail(n_long).mean()) if len(r) >= n_long else (float(r.mean()) if len(r) else None)
                over50 = (r > 50).astype(int).tail(n_long)
                out["time_over_50"] = float(over50.mean()*100.0) if len(over50) else None
                return out
            except Exception:
                return out

        def ema_distance_and_duration(df: pd.DataFrame, inds: Dict[str, pd.Series], ema_key: str = "EMA_20") -> Dict[str, Optional[float]]:
            out = {"price_above_ema": None, "pct_distance": None, "ema_slope": None, "streak_bars_above": None}
            try:
                if df is None or df.empty or not inds or ema_key not in inds or inds[ema_key].empty:
                    return out
                # align on common index
                ema = inds[ema_key].dropna()
                close = df["Close"].dropna()
                common = ema.index.intersection(close.index)
                if len(common) < 5:
                    return out
                ema = ema.loc[common]
                close = close.loc[common]
                last_close = float(close.iloc[-1])
                last_ema = float(ema.iloc[-1])
                out["price_above_ema"] = 1.0 if last_close > last_ema else 0.0
                out["pct_distance"] = float((last_close - last_ema) / max(1e-9, last_ema) * 100.0)
                out["ema_slope"] = lr_slope(ema, 10)
                # streak
                above = (close > ema).astype(int)
                streak = 0
                for v in reversed(above.values):
                    if v == 1:
                        streak += 1
                    else:
                        break
                out["streak_bars_above"] = float(streak)
                return out
            except Exception:
                return out

        def percent_b(df: pd.DataFrame, inds: Dict[str, pd.Series]) -> Dict[str, Optional[float]]:
            out = {"pct_b_last": None, "pct_b_slope": None}
            try:
                if df is None or df.empty or not inds:
                    return out
                if not all(k in inds for k in ("BB_Upper", "BB_Lower")):
                    return out
                upper = inds["BB_Upper"].dropna()
                lower = inds["BB_Lower"].dropna()
                close = df["Close"].dropna()
                common = upper.index.intersection(lower.index).intersection(close.index)
                if len(common) < 5:
                    return out
                upper = upper.loc[common]
                lower = lower.loc[common]
                close = close.loc[common]
                rng = (upper - lower).replace(0, np.nan)
                pb = (close - lower) / rng
                pb = pb.clip(lower=0.0, upper=1.0).dropna()
                out["pct_b_last"] = float(pb.iloc[-1]) if len(pb) else None
                out["pct_b_slope"] = lr_slope(pb, 10)
                return out
            except Exception:
                return out

        def macd_hist_slope(inds: Dict[str, pd.Series]) -> Optional[float]:
            try:
                if inds and "MACD_Histogram" in inds and not inds["MACD_Histogram"].empty:
                    return lr_slope(inds["MACD_Histogram"], 12)
            except Exception:
                return None
            return None

        def obv_slope(df: pd.DataFrame) -> Optional[float]:
            try:
                if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
                    return None
                close = df["Close"].dropna()
                vol = df["Volume"].reindex(close.index).fillna(0)
                ret = close.diff()
                obv = (np.sign(ret.fillna(0)) * vol).cumsum()
                return lr_slope(obv, 20)
            except Exception:
                return None

        def relative_volume(df: pd.DataFrame, n_ref: int = 20) -> Optional[float]:
            try:
                if df is None or df.empty or "Volume" not in df.columns:
                    return None
                vol = df["Volume"].dropna()
                if len(vol) < n_ref + 1:
                    return None
                rv = float(vol.iloc[-1] / max(1e-9, vol.tail(n_ref).mean()))
                return rv
            except Exception:
                return None

        def atr_series(df: pd.DataFrame, n: int = 14) -> Optional[pd.Series]:
            try:
                if df is None or df.empty:
                    return None
                h = df["High"]
                l = df["Low"]
                c = df["Close"]
                prev_c = c.shift(1)
                tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
                atr = tr.rolling(n, min_periods=3).mean()
                return atr.dropna()
            except Exception:
                return None

        # 1w features
        if data_1w is not None and not data_1w.empty:
            feats["ret_1w_last5bars_compound"] = compounded_return(data_1w, 5)
            feats["vol_ann_1w_pct"] = float(
                data_1w["Close"].pct_change().std() * self._annualization_sqrt(data_1w.index) * 100.0
            )
            feats["rel_volume_1w_last_vs_mean20"] = relative_volume(data_1w, 20)
            feats["obv_slope_1w"] = obv_slope(data_1w)
            atr_1w = atr_series(data_1w, 14)
            if atr_1w is not None and len(atr_1w) > 0:
                feats["atr_1w"] = float(atr_1w.iloc[-1])
                last_close = float(data_1w["Close"].iloc[-1])
                feats["atr_pct_1w"] = float(atr_1w.iloc[-1] / max(1e-9, last_close) * 100.0)
                feats["atr_slope_1w"] = lr_slope(atr_1w, 10)

        # 1w indicator features
        if ind_1w:
            bw_1w = bb_width_series(ind_1w)
            if bw_1w is not None and len(bw_1w):
                feats["bb_width_last_1w_norm_pct"] = float(bw_1w.iloc[-1] * 100.0)
                feats["bb_width_slope_1w"] = lr_slope(bw_1w, 10)
            feats["macd_hist_slope_1w"] = macd_hist_slope(ind_1w)
            feats.update({f"{k}_1w": v for k, v in rsi_stats(ind_1w).items()})
            feats.update({f"{k}_1w": v for k, v in percent_b(data_1w, ind_1w).items()})
            feats.update({f"{k}_1w": v for k, v in ema_distance_and_duration(data_1w, ind_1w).items()})

        # 3m features
        if data_3m is not None and not data_3m.empty:
            feats["ret_3m_last5bars_compound"] = compounded_return(data_3m, 5)
            feats["vol_ann_3m_pct"] = float(
                data_3m["Close"].pct_change().std() * self._annualization_sqrt(data_3m.index) * 100.0
            )
            feats["rel_volume_3m_last_vs_mean20"] = relative_volume(data_3m, 20)
            feats["obv_slope_3m"] = obv_slope(data_3m)
            atr_3m = atr_series(data_3m, 14)
            if atr_3m is not None and len(atr_3m) > 0:
                feats["atr_3m"] = float(atr_3m.iloc[-1])
                last_close = float(data_3m["Close"].iloc[-1])
                feats["atr_pct_3m"] = float(atr_3m.iloc[-1] / max(1e-9, last_close) * 100.0)
                feats["atr_slope_3m"] = lr_slope(atr_3m, 10)

        if ind_3m:
            bw_3m = bb_width_series(ind_3m)
            if bw_3m is not None and len(bw_3m):
                feats["bb_width_last_3m_norm_pct"] = float(bw_3m.iloc[-1] * 100.0)
                feats["bb_width_slope_3m"] = lr_slope(bw_3m, 20)
                feats["bb_width_percentile_3m"] = percentile_of_last(bw_3m, 120)
            feats["macd_hist_slope_3m"] = macd_hist_slope(ind_3m)
            feats.update({f"{k}_3m": v for k, v in rsi_stats(ind_3m).items()})
            feats.update({f"{k}_3m": v for k, v in percent_b(data_3m, ind_3m).items()})
            feats.update({f"{k}_3m": v for k, v in ema_distance_and_duration(data_3m, ind_3m).items()})

        # Price structure via simple pivot detection (fractal pivots)
        feats.update(self._structure_features(data_1w, prefix="struct_1w"))
        feats.update(self._structure_features(data_3m, prefix="struct_3m"))

        # Base return stats used by baseline forecast and validation
        base_df = data_1w if data_1w is not None and not data_1w.empty else data_3m
        if base_df is not None and not base_df.empty:
            r = base_df["Close"].pct_change().dropna()
            feats["ret_mu_base"] = float(r.tail(100).mean()) if len(r) else None
            feats["ret_sigma_base"] = float(r.tail(100).std()) if len(r) else None
            feats["bars_to_target"] = float(bars_to_target)

        return feats

    def _structure_features(self, df: pd.DataFrame, prefix: str, left: int = 2, right: int = 2) -> Dict[str, Any]:
        """Detect recent swing highs/lows and classify HH/HL or LH/LL."""
        out: Dict[str, Any] = {
            f"{prefix}_hh_hl_trend": None,
            f"{prefix}_last_pivots": [],
            f"{prefix}_key_levels": {},
        }
        try:
            if df is None or df.empty:
                return out
            high = df["High"].values
            low = df["Low"].values
            idx = list(df.index)
            n = len(df)
            pivots = []  # (ts, 'H'/'L', price)

            # Simple fractal pivot definition
            for i in range(left, n - right):
                window_h = high[i - left:i + right + 1]
                window_l = low[i - left:i + right + 1]
                if high[i] == np.max(window_h):
                    pivots.append((idx[i], "H", float(high[i])))
                if low[i] == np.min(window_l):
                    pivots.append((idx[i], "L", float(low[i])))

            pivots = pivots[-8:]  # keep the last few
            out[f"{prefix}_last_pivots"] = [{"ts": self._safe_format_datetime(t), "type": tp, "price": p} for t, tp, p in pivots]

            # Determine HH/HL vs LH/LL from last pairs
            highs = [(t, p) for t, tp, p in pivots if tp == "H"]
            lows = [(t, p) for t, tp, p in pivots if tp == "L"]
            trend = None
            if len(highs) >= 2 and len(lows) >= 2:
                hh = highs[-1][1] > highs[-2][1]
                hl = lows[-1][1] > lows[-2][1]
                if hh and hl:
                    trend = "HH_HL"
                elif (not hh) and (not hl):
                    trend = "LH_LL"
                else:
                    trend = "mixed"
            out[f"{prefix}_hh_hl_trend"] = trend

            # Key levels from last swing high/low
            key_levels = {}
            if highs:
                key_levels["resistance"] = highs[-1][1]
            if lows:
                key_levels["support"] = lows[-1][1]
            out[f"{prefix}_key_levels"] = key_levels
            return out
        except Exception:
            return out

    # ---------- enhanced chart data with timestamps ----------

    def _prepare_enhanced_chart_data(
        self,
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        max_points_3m: int = 350,
        keep_tail_3m: int = 120,
        max_points_1w: int = 220,
        keep_tail_1w: int = 60,
    ) -> Dict[str, Any]:
        try:
            enhanced: Dict[str, Any] = {}

            full_3m = data_3m if data_3m is not None else pd.DataFrame()
            full_1w = data_1w if data_1w is not None else pd.DataFrame()

            full_3m_high = float(full_3m["High"].max()) if (not full_3m.empty and "High" in full_3m.columns) else None
            full_3m_low = float(full_3m["Low"].min()) if (not full_3m.empty and "Low" in full_3m.columns) else None
            full_1w_high = float(full_1w["High"].max()) if (not full_1w.empty and "High" in full_1w.columns) else None
            full_1w_low = float(full_1w["Low"].min()) if (not full_1w.empty and "Low" in full_1w.columns) else None

            # Display windows (same as before, but with compression)
            recent_3m = full_3m
            recent_1w = full_1w

            # compressed recent slices for display arrays
            tail_3m_idx = self._compress_series(pd.Series(recent_3m.index, index=recent_3m.index), max_points_3m, keep_tail_3m).index
            tail_1w_idx = self._compress_series(pd.Series(recent_1w.index, index=recent_1w.index), max_points_1w, keep_tail_1w).index

            tail_3m = recent_3m.loc[tail_3m_idx] if not recent_3m.empty else recent_3m
            tail_1w = recent_1w.loc[tail_1w_idx] if not recent_1w.empty else recent_1w

            enhanced["3m_data"] = {
                "timeframe": "3-month",
                "full_range": (
                    self._safe_format_daterange(full_3m.index[0], full_3m.index[-1]) if not full_3m.empty else "N/A"
                ),
                "data_range": (
                    self._safe_format_daterange(tail_3m.index[0], tail_3m.index[-1]) if not tail_3m.empty else "N/A"
                ),
                "period_highs_lows": {
                    "period_high": full_3m_high,
                    "period_low": full_3m_low,
                    "recent_high": float(recent_3m["High"].max()) if (not recent_3m.empty and "High" in recent_3m.columns) else None,
                    "recent_low": float(recent_3m["Low"].min()) if (not recent_3m.empty and "Low" in recent_3m.columns) else None,
                },
                "recent_prices": (
                    {
                        "dates": [self._safe_format_datetime(d) for d in tail_3m.index],
                        "open": tail_3m.get("Open", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_3m.empty else [],
                        "high": tail_3m.get("High", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_3m.empty else [],
                        "low": tail_3m.get("Low", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_3m.empty else [],
                        "close": tail_3m.get("Close", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_3m.empty else [],
                        "volume": tail_3m.get("Volume", pd.Series(dtype=float)).round(0).astype(float).tolist() if not tail_3m.empty else [],
                    }
                    if not tail_3m.empty
                    else {}
                ),
                "indicators": {},
            }

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
                "MFI",
            ]:
                if indicators_3m and indicator in indicators_3m:
                    enhanced["3m_data"]["indicators"][indicator] = self._pack_series(
                        indicators_3m[indicator], max_points=max_points_3m, keep_tail=keep_tail_3m, round_to=4
                    )

            enhanced["1w_data"] = {
                "timeframe": "1-week",
                "full_range": (
                    self._safe_format_daterange(full_1w.index[0], full_1w.index[-1]) if not full_1w.empty else "N/A"
                ),
                "data_range": (
                    self._safe_format_daterange(tail_1w.index[0], tail_1w.index[-1]) if not tail_1w.empty else "N/A"
                ),
                "period_highs_lows": {
                    "period_high": full_1w_high,
                    "period_low": full_1w_low,
                    "recent_high": float(recent_1w["High"].max()) if (not recent_1w.empty and "High" in recent_1w.columns) else None,
                    "recent_low": float(recent_1w["Low"].min()) if (not recent_1w.empty and "Low" in recent_1w.columns) else None,
                },
                "recent_prices": (
                    {
                        "dates": [self._safe_format_datetime(d) for d in tail_1w.index],
                        "open": tail_1w.get("Open", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_1w.empty else [],
                        "high": tail_1w.get("High", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_1w.empty else [],
                        "low": tail_1w.get("Low", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_1w.empty else [],
                        "close": tail_1w.get("Close", pd.Series(dtype=float)).round(2).astype(float).tolist() if not tail_1w.empty else [],
                        "volume": tail_1w.get("Volume", pd.Series(dtype=float)).round(0).astype(float).tolist() if not tail_1w.empty else [],
                    }
                    if not tail_1w.empty
                    else {}
                ),
                "indicators": {},
            }

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
                "MFI",
            ]:
                if indicators_1w and indicator in indicators_1w:
                    enhanced["1w_data"]["indicators"][indicator] = self._pack_series(
                        indicators_1w[indicator], max_points=max_points_1w, keep_tail=keep_tail_1w, round_to=4
                    )

            # Volume trend flags (unchanged logic)
            v50 = full_3m.get("Volume", pd.Series(dtype=float)).tail(50).mean() if not full_3m.empty else None
            v10 = full_3m.get("Volume", pd.Series(dtype=float)).tail(10).mean() if not full_3m.empty else None
            v30 = full_1w.get("Volume", pd.Series(dtype=float)).tail(30).mean() if not full_1w.empty else None
            v5 = full_1w.get("Volume", pd.Series(dtype=float)).tail(5).mean() if not full_1w.empty else None

            enhanced["volume_analysis"] = {
                "3m_avg_volume_tail50": float(v50) if v50 is not None else None,
                "3m_volume_trend": (
                    "increasing" if (v10 is not None and v50 is not None and v10 > v50) else (
                        "decreasing" if (v10 is not None and v50 is not None and v10 < v50) else None
                    )
                ),
                "1w_avg_volume_tail30": float(v30) if v30 is not None else None,
                "1w_volume_trend": (
                    "increasing" if (v5 is not None and v30 is not None and v5 > v30) else (
                        "decreasing" if (v5 is not None and v30 is not None and v5 < v30) else None
                    )
                ),
            }

            return enhanced

        except Exception as e:
            st.warning(f"Error preparing enhanced chart data: {e}")
            return {}

    def _summarize_indicators(
        self,
        indicators_3m: Dict[str, pd.Series],
        indicators_1w: Dict[str, pd.Series],
        current_price: float,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"current_price": float(current_price)}
        try:
            rsi_3m_last = (
                indicators_3m["RSI"].dropna().iloc[-1]
                if indicators_3m and "RSI" in indicators_3m and not indicators_3m["RSI"].empty
                else np.nan
            )
            rsi_1w_last = (
                indicators_1w["RSI"].dropna().iloc[-1]
                if indicators_1w and "RSI" in indicators_1w and not indicators_1w["RSI"].empty
                else np.nan
            )
            summary["RSI"] = {
                "3m_current": float(rsi_3m_last) if not np.isnan(rsi_3m_last) else None,
                "1w_current": float(rsi_1w_last) if not np.isnan(rsi_1w_last) else None,
            }

            if indicators_3m and "MACD" in indicators_3m and "MACD_Signal" in indicators_3m:
                macd_3m = indicators_3m["MACD"].dropna().iloc[-1]
                signal_3m = indicators_3m["MACD_Signal"].dropna().iloc[-1]
                if not np.isnan(macd_3m) and not np.isnan(signal_3m):
                    summary["MACD_3m"] = {
                        "macd": float(macd_3m),
                        "signal": float(signal_3m),
                        "crossover": "bullish" if macd_3m > signal_3m else "bearish",
                    }

            if indicators_1w and "MACD" in indicators_1w and "MACD_Signal" in indicators_1w:
                macd_1w = indicators_1w["MACD"].dropna().iloc[-1]
                signal_1w = indicators_1w["MACD_Signal"].dropna().iloc[-1]
                if not np.isnan(macd_1w) and not np.isnan(signal_1w):
                    summary["MACD_1w"] = {
                        "macd": float(macd_1w),
                        "signal": float(signal_1w),
                        "crossover": "bullish" if macd_1w > signal_1w else "bearish",
                    }

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and all(k in inds for k in ["BB_Upper", "BB_Lower", "BB_Middle"]):
                    upper = inds["BB_Upper"].dropna().iloc[-1]
                    lower = inds["BB_Lower"].dropna().iloc[-1]
                    middle = inds["BB_Middle"].dropna().iloc[-1]
                    if not any(pd.isna([upper, lower, middle])):
                        cp = summary["current_price"]
                        summary[f"BB_{timeframe}"] = {
                            "upper": float(upper),
                            "lower": float(lower),
                            "middle": float(middle),
                            "position": "above_upper"
                            if cp > upper
                            else ("below_lower" if cp < lower else "within_bands"),
                        }

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and "EMA_20" in inds:
                    ema_20 = inds["EMA_20"].dropna().iloc[-1]
                    if not np.isnan(ema_20):
                        cp = summary["current_price"]
                        summary[f"EMA_20_{timeframe}"] = {
                            "value": float(ema_20),
                            "trend": "bullish" if cp > ema_20 else "bearish",
                        }

        except Exception as e:
            st.warning(f"Error summarizing indicators: {e}")

        return summary

    def _availability_flags(
        self,
        ind_3m: Dict[str, pd.Series],
        ind_1w: Dict[str, pd.Series],
        data_3m: pd.DataFrame,
        data_1w: pd.DataFrame,
    ) -> Dict[str, bool]:
        def has(ind: Dict[str, pd.Series], k: str) -> bool:
            return bool(ind and k in ind and not ind[k].dropna().empty)

        return {
            "has_rsi_3m": has(ind_3m, "RSI"),
            "has_rsi_1w": has(ind_1w, "RSI"),
            "has_macd_3m": has(ind_3m, "MACD") and has(ind_3m, "MACD_Signal"),
            "has_macd_1w": has(ind_1w, "MACD") and has(ind_1w, "MACD_Signal"),
            "has_bb_3m": has(ind_3m, "BB_Upper") and has(ind_3m, "BB_Lower") and has(ind_3m, "BB_Middle"),
            "has_bb_1w": has(ind_1w, "BB_Upper") and has(ind_1w, "BB_Lower") and has(ind_1w, "BB_Middle"),
            "has_ema20_3m": has(ind_3m, "EMA_20"),
            "has_ema20_1w": has(ind_1w, "EMA_20"),
            "has_ohlcv_3m": bool(data_3m is not None and not data_3m.empty),
            "has_ohlcv_1w": bool(data_1w is not None and not data_1w.empty),
        }

    # ---------- baseline forecast (scaffold and fallback) ----------

    def _baseline_forecast(self, df: pd.DataFrame, bars_to_target: int, current_price: float) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            if df is None or df.empty or "Close" not in df.columns:
                return out
            r = df["Close"].pct_change().dropna()
            if len(r) < 10:
                return out
            mu = float(r.tail(100).mean())
            sigma = float(r.tail(100).std())
            # Compound expected move using arithmetic mean as approximation
            exp_pct = float((1.0 + mu) ** bars_to_target - 1.0) if bars_to_target > 0 else 0.0
            # 1-sigma band over horizon (sqrt time)
            vol_pct = float(sigma * math.sqrt(max(1, bars_to_target)))
            baseline_price = float(current_price * (1.0 + exp_pct))
            upper_1s = float(current_price * (1.0 + exp_pct + vol_pct))
            lower_1s = float(current_price * (1.0 + exp_pct - vol_pct))
            out.update({
                "baseline_expected_pct_to_target": exp_pct * 100.0,
                "baseline_sigma_pct_sqrt_bars": vol_pct * 100.0,
                "baseline_predicted_price": baseline_price,
                "baseline_upper_1sigma": upper_1s,
                "baseline_lower_1sigma": lower_1s,
            })
            return out
        except Exception:
            return out

    # ---------- prompt + model call ----------

    def _build_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        system_content = (
            "You are a deterministic technical analyst. Use ONLY the structured arrays provided in the ENHANCED ARRAYS, FEATURES, "
            "INDICATOR SNAPSHOT, AVAILABILITY_FLAGS, and VOLUME_ANALYSIS sections. "
            "Analyze the FULL arrays (use provided dated arrays; do not infer beyond). "
            "Explicitly compare 1-week (1w) and 3-month (3m) arrays for each indicator, noting alignments, conflicts, "
            "and how short-term momentum influences longer-term trends. "
            "Forbidden: news, macro, on-chain, session effects, weekdays, seasonality, or inferences beyond given timestamps and data. "
            "All claims must be quantitatively verifiable from the arrays (compute averages, slopes, crossovers, divergences with timestamps). "
            "If any required arrays are missing/insufficient, output status='insufficient_data' with detailed 'notes'. "
            "Prioritize depth: synthesize across indicators and timeframes for comprehensive, evidence-based analysis."
        )

        data_3m = analysis_data.get("data_3m", {})
        data_1w = analysis_data.get("data_1w", {})

        # JSON schema (example; must return a VALID JSON object)
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
                {
                    "type": "rsi|macd|bb|ema|price|volume|structure|volatility",
                    "timeframe": "3m|1w",
                    "ts": "YYYY-MM-DD HH:MM" or "recent",
                    "value": "number or object",
                    "note": "short factual note with numbers and timestamps",
                }
            ],
            "notes": ["if status=insufficient_data, list what's missing; else optional warnings"],
        }

        # Narrative section template (unchanged structure; model fills real values)
        narrative_template = f"""
        [TECHNICAL_ANALYSIS_START]
        **COMPREHENSIVE TECHNICAL ANALYSIS**

        **Current Price: ${{analysis_data.get('current_price'):,.2f}}**

        **1. MULTI-TIMEFRAME OVERVIEW**
        - 3-Month Chart Analysis: <facts from full 3m arrays>
        - 1-Week Chart Analysis: <facts from full 1w arrays>
        - Timeframe Alignment: <how 1w aligns/conflicts with 3m>

        **2. TECHNICAL INDICATORS ANALYSIS**
        - RSI Analysis: <levels, trends, divergences; compare 1w vs 3m>
        - MACD Analysis: <histogram slopes, crossovers; compare timeframes>
        - Bollinger Bands: <width trends, position; compare>
        - EMA Analysis: <price vs EMA_20 trends/slopes; compare>

        **3. ADVANCED PATTERN ANALYSIS**
        - Patterns & Candles: <from recent_prices and highs/lows>
        - Support/Resistance: <explicit levels>

        **4. DIVERGENCE & FAILURE SWINGS**
        - RSI/MACD divergences and failure swings: <scan full arrays>
        - Volume divergences: <from volume_analysis + recent_prices>

        **5. TRENDLINE & STRUCTURE**
        - Higher highs/lows or lower highs/lows: <across timeframes>
        - Key levels: <derived from BB, highs/lows>

        **6. TRADING RECOMMENDATION**
        - Overall Bias: **BULLISH/BEARISH/NEUTRAL**
        - Entry/Stop/Targets: <levels from arrays>
        [TECHNICAL_ANALYSIS_END]

        [SIMPLE_EXPLANATION_START]
        **📚 SIMPLE MAN'S EXPLANATION**

        **What These Indicators Actually Mean:**

        **🔄 RSI (Relative Strength Index):**
        - **What it is:** Measures if Bitcoin is "overbought" (too expensive) or "oversold" (too cheap)
        - **How to read it:** 0-30 = oversold (good time to buy), 70-100 = overbought (might fall soon), 30-70 = neutral
        - **Current reading:** <explain current RSI levels and what they mean for regular people>

        **📈 MACD (Moving Average Convergence Divergence):**
        - **What it is:** Shows the relationship between two moving averages to spot trend changes
        - **How to read it:** When MACD line crosses above signal line = bullish (price might go up), below = bearish (price might go down)
        - **Current reading:** <explain current MACD situation in simple terms>

        **🎯 Bollinger Bands:**
        - **What it is:** Shows if Bitcoin is trading in a "normal" price range or breaking out
        - **How to read it:** Price touching upper band = might be overbought, touching lower band = might be oversold, squeezing bands = big move coming
        - **Current reading:** <explain where price is relative to the bands>

        **📊 EMA (Exponential Moving Average):**
        - **What it is:** The average price over recent periods, giving more weight to recent prices
        - **How to read it:** Price above EMA = uptrend, below EMA = downtrend, EMA slope shows trend strength
        - **Current reading:** <explain current EMA situation>

        **📦 Volume Analysis:**
        - **What it is:** How much Bitcoin is being traded
        - **How to read it:** High volume + price increase = strong move, low volume = weak move, volume confirms trends
        - **Current reading:** <explain current volume trends>

        **Why This Matters:** <Tie all indicators together to explain the overall market sentiment and what it means for someone considering buying/selling Bitcoin>
        [SIMPLE_EXPLANATION_END]

        [PRICE_PREDICTION_START]
        **PREDICTED PRICE: I predict {asset_name} will be at $<PRICE> on {analysis_data.get('target_time')}**

        ⏰ **DATA-BASED TIME ANALYSIS:**
        - Exact Target: {analysis_data.get('target_time')}
        - Momentum Direction: <from full RSI/MACD arrays>
        - Trend Strength: <from EMA slopes, hist averages>
        - Volume Analysis: <trends>
        - Expected Path: <data-grounded path>

        1. **Probability HIGHER than ${{analysis_data.get('current_price'):,.2f}}: <X>%**
        2. **Probability LOWER than ${{analysis_data.get('current_price'):,.2f}}: <Y>%**
        3. **Overall Analysis Confidence: <Z>%**
        4. **Price Prediction Confidence: <W>%**
        5. **Expected % Move: <±M>%**

        **Key Technical Factors from the Actual Data:**
        - Factor 1: <example>
        - Factor 2: <example>
        - Factor 3: <example>

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
        STEP_SECONDS: {analysis_data.get('step_seconds')}
        BARS_TO_TARGET: {analysis_data.get('bars_to_target')}
        CURRENT_PRICE (use exactly this number): {analysis_data.get('current_price')}

        AVAILABILITY_FLAGS:
        {json.dumps(analysis_data.get('availability_flags', {}), indent=2)}

        INDICATOR SNAPSHOT:
        {json.dumps(analysis_data.get('indicators', {}), indent=2)}

        FEATURES:
        {json.dumps(analysis_data.get('features', {}), indent=2)}

        ENHANCED ARRAYS:
        {json.dumps(analysis_data.get('enhanced_chart_data', {}), indent=2)}

        STRICT RULES:
        - Use ONLY these arrays for all analysis. Do NOT use session effects, news, or any outside info.
        - Use the dated arrays to make timestamp-referenced claims (e.g., divergences with specific candles).
        - For trend analysis: Scan the arrays to compute metrics (RSI averages over 10/20, MACD histogram slopes, BB width trends/percentiles, EMA slopes/distance, %b), and verify with recent_prices.dates.
        - Mandatory comparisons: For RSI, MACD, BB, EMA, compare 1w vs 3m values/trends and assess timeframe alignment.
        - Divergences/Failures: Only claim if verifiable in data with timestamps.
        - Volume: Use volume_analysis, relative volume features, and OBV slopes for trends/divergences.
        - Predictions: Derive predicted_price from array-derived trends. p_up + p_down = 1. Scale expectation to the horizon (BARS_TO_TARGET).
        - If arrays are empty/missing, return status='insufficient_data' with 'notes' listing specifics.

        YOU MUST RETURN **TWO** BLOCKS, IN THIS ORDER:

        1) A VALID JSON object, in a fenced code block like:
        ```json
        {json.dumps(output_schema, indent=2)}
        ```
        2) A narrative block using EXACTLY the following section markers and structure (fill in concrete values):

        {narrative_template}
        """.strip()
        return ({"role": "system", "content": system_content}, {"role": "user", "content": user_content})

    def _generate_technical_analysis_gpt5(self, analysis_data: Dict[str, Any]) -> str:
        """Call the model asking for JSON + narrative blocks. Return raw text."""
        asset_name = analysis_data.get("asset_name", "Asset")
        system_msg, user_msg = self._build_messages(analysis_data, asset_name)

        # DEBUG: Show full prompt only if debug enabled
        if self.debug:
            print("\n" + "=" * 80)
            print("🤖 FULL PROMPT SENT TO MODEL (debug)")
            print("=" * 80)
            print("\n📋 SYSTEM MESSAGE:\n" + (system_msg["content"] or ""))
            print("\n📊 USER MESSAGE (DATA):\n" + (user_msg["content"] or ""))
            print("\n" + "=" * 80)

        raw = self.gpt.generate(system_msg["content"], user_msg["content"]) or ""
        if not raw.strip():
            return json.dumps({"status": "insufficient_data", "notes": ["empty_model_response"]})
        return raw

    # ---------- parsing + probabilities + text composition ----------

    def _split_dual_output(self, raw: str) -> Tuple[str, str]:
        """Extract a JSON fenced block (```json ... ```) and the rest (narrative)."""
        if not raw:
            return "", ""

        # Prefer fenced ```json blocks
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if m:
            json_block = m.group(1).strip()
            narrative = (raw[: m.start()] + raw[m.end() :]).strip()
            return json_block, narrative

        # Fallback: first JSON object in text (greedy braces balance heuristic)
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw[first_brace : last_brace + 1]
            try:
                json.loads(candidate)
                narrative = (raw[:first_brace] + raw[last_brace + 1 :]).strip()
                return candidate.strip(), narrative
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

            # Normalize status
            st_raw = str(data.get("status", "insufficient_data")).lower().strip()
            data["status"] = "ok" if st_raw == "ok" else "insufficient_data"

            # Normalize probabilities
            p_up = self._clip01(self._safe_num(data.get("p_up"), 0.5))
            p_down = self._clip01(self._safe_num(data.get("p_down"), 0.5))
            total = p_up + p_down
            if total <= 0:
                p_up, p_down = 0.5, 0.5
            else:
                p_up, p_down = p_up / total, p_down / total
            data["p_up"], data["p_down"] = p_up, p_down

            # Confidences
            data["conf_overall"] = self._clip01(self._safe_num(data.get("conf_overall"), 0.5))
            data["conf_price"] = self._clip01(self._safe_num(data.get("conf_price"), 0.5))

            # Expected % move (compute if needed)
            cp = float(current_price) if current_price is not None else None
            pred = self._safe_num(data.get("predicted_price"), None)
            if pred is not None and cp and cp > 0:
                data["expected_pct_move"] = (float(pred) - cp) / cp * 100.0

            # Critical levels always present as object
            data["critical_levels"] = data.get("critical_levels") or {"bullish_above": None, "bearish_below": None}

            return data
        except Exception as e:
            return {"status": "insufficient_data", "notes": [f"json_parse_error:{e}"]}

    def _validate_and_adjust_prediction_block(
        self,
        data: Dict[str, Any],
        current_price: float,
        bars_to_target: Optional[int],
        atr_pct: Optional[float],
        sigma: Optional[float],
        k_atr: float = 6.0,
        z_sigma: float = 3.0,
    ) -> Dict[str, Any]:
        """Clip or warn about extreme predicted_price using ATR%/sigma bands."""
        try:
            if not isinstance(data, dict):
                return data
            pred = self._safe_num(data.get("predicted_price"), None)
            if pred is None or current_price is None or current_price <= 0:
                return data

            bars = int(bars_to_target or 1)
            # Build a volatility guardrail in percent
            guard_pct = 0.0
            if atr_pct is not None:
                guard_pct = max(guard_pct, float(atr_pct) * math.sqrt(max(1, bars)) * k_atr / 100.0)
            if sigma is not None:
                guard_pct = max(guard_pct, float(sigma) * math.sqrt(max(1, bars)) * z_sigma)

            if guard_pct <= 0.0:
                return data

            upper = current_price * (1.0 + guard_pct)
            lower = current_price * (1.0 - guard_pct)

            adjusted = float(np.clip(pred, lower, upper))
            if adjusted != pred:
                notes = data.get("notes", []) or []
                notes.append(f"predicted_price_clipped_to_vol_band[{lower:,.0f},{upper:,.0f}]")
                data["notes"] = notes
                data["predicted_price"] = adjusted
                # recompute expected pct move
                data["expected_pct_move"] = (adjusted - current_price) / current_price * 100.0

            return data
        except Exception:
            return data

    def _parse_comprehensive_response(self, response: str) -> Dict[str, str]:
        """Parse narrative into sections using explicit markers."""
        try:
            sections: Dict[str, str] = {}
            tech_start = response.find("[TECHNICAL_ANALYSIS_START]")
            tech_end = response.find("[TECHNICAL_ANALYSIS_END]")
            if tech_start != -1 and tech_end != -1 and tech_end > tech_start:
                sections["technical_summary"] = response[tech_start + len("[TECHNICAL_ANALYSIS_START]") : tech_end].strip()

            simple_start = response.find("[SIMPLE_EXPLANATION_START]")
            simple_end = response.find("[SIMPLE_EXPLANATION_END]")
            if simple_start != -1 and simple_end != -1 and simple_end > simple_start:
                sections["simple_explanation"] = response[simple_start + len("[SIMPLE_EXPLANATION_START]") : simple_end].strip()

            pred_start = response.find("[PRICE_PREDICTION_START]")
            pred_end = response.find("[PRICE_PREDICTION_END]")
            if pred_start != -1 and pred_end != -1 and pred_end > pred_start:
                sections["price_prediction"] = response[pred_start + len("[PRICE_PREDICTION_START]") : pred_end].strip()

            if not sections:
                # Very lenient fallback based on headings
                lines = response.splitlines()
                current_section = None
                buf: List[str] = []
                for line in lines:
                    ll = line.lower().strip()
                    if "technical analysis" in ll and not sections.get("technical_summary"):
                        if current_section and buf:
                            sections[current_section] = "\n".join(buf).strip()
                        current_section = "technical_summary"
                        buf = []
                    elif "price prediction" in ll and not sections.get("price_prediction"):
                        if current_section and buf:
                            sections[current_section] = "\n".join(buf).strip()
                        current_section = "price_prediction"
                        buf = []
                    elif current_section:
                        buf.append(line)
                if current_section and buf:
                    sections[current_section] = "\n".join(buf).strip()

            return sections or {"technical_summary": response, "price_prediction": ""}
        except Exception:
            return {"technical_summary": response, "price_prediction": "Unable to parse prediction section"}

    def _extract_probabilities_from_json(self, data: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        if not isinstance(data, dict) or data.get("status") != "ok":
            return self._default_probs()

        p_up = self._clip01(self._safe_num(data.get("p_up"), 0.5))
        p_down = self._clip01(self._safe_num(data.get("p_down"), 0.5))
        conf_overall = self._clip01(self._safe_num(data.get("conf_overall"), 0.5))
        conf_price = self._clip01(self._safe_num(data.get("conf_price"), 0.5))

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

    def _heuristic_probabilities(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Heuristic fallback mapping features to probabilities."""
        feats = analysis_data.get("features", {}) or {}
        cp = float(analysis_data.get("current_price") or 0.0)
        # Score components (weights are heuristic)
        score = 0.0
        w = {
            "ema_slope": 1.0,
            "price_above_ema": 1.0,
            "macd_hist": 1.0,
            "rsi": 1.0,
            "pct_b": 0.7,
            "structure": 1.0,
            "volume": 0.5,
            "volatility": 0.3,
        }
        # 1w signals get more weight for near-term horizon
        ema_slope = (feats.get("ema_slope_1w") or feats.get("ema_slope_3m") or 0.0)
        price_above = feats.get("price_above_ema_1w")
        macd_slope = (feats.get("macd_hist_slope_1w") or feats.get("macd_hist_slope_3m") or 0.0)
        rsi_last = (feats.get("rsi_last_1w") or feats.get("rsi_last_3m") or 50.0)
        pct_b_last = (feats.get("pct_b_last_1w") or feats.get("pct_b_last_3m") or 0.5)
        struct_trend = feats.get("struct_1w_hh_hl_trend") or feats.get("struct_3m_hh_hl_trend") or "mixed"
        rel_vol = (feats.get("rel_volume_1w_last_vs_mean20") or feats.get("rel_volume_3m_last_vs_mean20") or 1.0)
        atr_slope = (feats.get("atr_slope_1w") or feats.get("atr_slope_3m") or 0.0)

        score += w["ema_slope"] * (1.0 if (ema_slope is not None and ema_slope > 0) else -1.0 if (ema_slope is not None and ema_slope < 0) else 0.0)
        score += w["price_above_ema"] * (1.0 if price_above == 1.0 else -1.0 if price_above == 0.0 else 0.0)
        score += w["macd_hist"] * (1.0 if (macd_slope is not None and macd_slope > 0) else -1.0 if (macd_slope is not None and macd_slope < 0) else 0.0)
        score += w["rsi"] * (1.0 if (rsi_last is not None and rsi_last > 50) else -1.0 if (rsi_last is not None and rsi_last < 50) else 0.0)
        score += w["pct_b"] * (1.0 if (pct_b_last is not None and pct_b_last > 0.5) else -1.0 if (pct_b_last is not None and pct_b_last < 0.5) else 0.0)
        score += w["structure"] * (1.0 if struct_trend == "HH_HL" else -1.0 if struct_trend == "LH_LL" else 0.0)
        score += w["volume"] * (1.0 if (rel_vol is not None and rel_vol > 1.05) else -1.0 if (rel_vol is not None and rel_vol < 0.95) else 0.0)
        score += w["volatility"] * (1.0 if (atr_slope is not None and atr_slope > 0) else 0.0)

        # Map score -> p_up via sigmoid
        p_up = 1.0 / (1.0 + math.exp(-0.8 * score))
        p_down = 1.0 - p_up
        conf_overall = 0.4 + 0.2 * abs(score) / 5.0
        conf_overall = float(max(0.2, min(0.85, conf_overall)))

        # Baseline expected move
        move_pct = analysis_data.get("features", {}).get("baseline_expected_pct_to_target", 0.0) or 0.0

        return {
            "higher_fraction": float(p_up),
            "lower_fraction": float(p_down),
            "confidence_fraction": float(conf_overall),
            "higher_pct": float(p_up * 100.0),
            "lower_pct": float(p_down * 100.0),
            "confidence_pct": float(conf_overall * 100.0),
            "predicted_price": None,
            "price_confidence_pct": float(conf_overall * 100.0),
            "move_percentage": float(move_pct),
        }

    def _default_probs(self) -> Dict[str, Any]:
        return {
            "higher_fraction": 0.5,
            "lower_fraction": 0.5,
            "confidence_fraction": 0.5,
            "higher_pct": 50.0,
            "lower_pct": 50.0,
            "confidence_pct": 50.0,
            "predicted_price": None,
            "price_confidence_pct": 50.0,
            "move_percentage": 0.0,
        }

    def _safe_format_datetime(self, dt) -> str:
        """Safely format a datetime value, handling NaT and edge cases."""
        try:
            if pd.isna(dt):
                return "N/A"
            ts = pd.Timestamp(dt)
            if pd.isna(ts) or str(ts) == "NaT":
                return "N/A"
            if ts.tzinfo is None:
                ts = ts.tz_localize(pytz.UTC)
            return ts.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "N/A"

    def _safe_format_daterange(self, start_dt, end_dt) -> str:
        try:
            if pd.isna(start_dt) or pd.isna(end_dt):
                return "N/A"
            start_ts = pd.Timestamp(start_dt)
            end_ts = pd.Timestamp(end_dt)
            if pd.isna(start_ts) or pd.isna(end_ts) or str(start_ts) == "NaT" or str(end_ts) == "NaT":
                return "N/A"
            return f"{start_ts.strftime('%B %d')} to {end_ts.strftime('%B %d, %Y')}"
        except Exception:
            return "N/A"

    def _safe_num(self, x: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if x is None:
                return default
            return float(x)
        except Exception:
            return default

    def _clip01(self, v: Optional[float]) -> float:
        try:
            if v is None:
                return 0.5
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.5

    # ---------- text composition + legacy regex fallback ----------

    def _compose_text_from_model_json(self, data: Dict[str, Any], current_price: float) -> Tuple[str, str]:
        if not isinstance(data, dict):
            return ("Analysis not available", "Prediction not available")

        status = data.get("status", "insufficient_data")
        as_of = data.get("as_of", "")
        target_ts = data.get("target_ts", "")
        pred = self._safe_num(data.get("predicted_price"), None)
        p_up = float(data.get("p_up", 0.5))
        p_down = float(data.get("p_down", 0.5))
        expected = data.get("expected_pct_move", None)
        crit = data.get("critical_levels", {}) or {}
        bull = crit.get("bullish_above", None)
        bear = crit.get("bearish_below", None)

        if status != "ok":
            notes = data.get("notes", [])
            msg = "; ".join([str(n) for n in notes]) if notes else "insufficient data"
            return (
                f"Status: insufficient data\n\nNotes: {msg}",
                f"**Target:** `{target_ts}`\n\n_No price prediction due to insufficient data._",
            )

        bullets: List[str] = []
        if bull is not None:
            bullets.append(f"- Bullish above: ${bull:,.0f}")
        if bear is not None:
            bullets.append(f"- Bearish below: ${bear:,.0f}")
        if expected is not None:
            bullets.append(f"- Expected move: {expected:+.2f}% vs current (${current_price:,.0f})")

        ev = data.get("evidence", []) or []
        for e in ev[:6]:
            t = e.get("type", "fact")
            tf = e.get("timeframe", "")
            ts = e.get("ts", "")
            note = e.get("note", "")
            bullets.append(f"- {t.upper()} {tf} @ {ts}: {note}")

        tech_md = f"As of: {as_of}\n\n" + ("\n".join(bullets) if bullets else "No additional evidence provided.")
        price_line = f"Predicted price at {target_ts}: " + (f"${pred:,.0f}" if pred is not None else "unavailable")
        pred_md = (
            f"{price_line}\n\n- P(higher): {p_up*100:.0f}%   - P(lower): {p_down*100:.0f}%   "
            f"- AI confidence: {float(data.get('conf_overall', 0.5))*100:.0f}%"
        )
        return tech_md, pred_md

    def _compose_text_when_insufficient(self, reason: str, target_ts: str) -> Tuple[str, str]:
        tech = f"Status: insufficient data\n\nNotes: {reason or 'missing inputs'}"
        pred = f"**Target:** `{target_ts}`\n\n_No price prediction due to insufficient data._"
        return tech, pred

    def _extract_probabilities(self, prediction_text: str) -> Dict[str, Any]:
        """Legacy regex extraction from narrative price section (best-effort)."""
        probs = {
            "higher_fraction": 0.5,
            "lower_fraction": 0.5,
            "confidence_fraction": 0.5,
            "higher_pct": 50.0,
            "lower_pct": 50.0,
            "confidence_pct": 50.0,
            "predicted_price": None,
            "price_confidence_pct": 50.0,
            "move_percentage": 0.0,
        }
        try:
            text = prediction_text

            patterns = {
                "higher": [
                    r"(\d+(?:\.\d+)?)%\s*(?:probability|chance|likelihood).*?(?:higher|up|increase)",
                    r"(?:higher|up|increase).*?(\d+(?:\.\d+)?)%",
                    r"HIGHER.*?(\d+(?:\.\d+)?)%",
                    r"(\d+(?:\.\d+)?)%.*?higher",
                ],
                "lower": [
                    r"(\d+(?:\.\d+)?)%\s*(?:probability|chance|likelihood).*?(?:lower|down|decrease)",
                    r"(?:lower|down|decrease).*?(\d+(?:\.\d+)?)%",
                    r"LOWER.*?(\d+(?:\.\d+)?)%",
                    r"(\d+(?:\.\d+)?)%.*?lower",
                ],
                "confidence": [
                    r"overall.*?confidence.*?(\d+(?:\.\d+)?)%",
                    r"analysis.*?confidence.*?(\d+(?:\.\d+)?)%",
                    r"confidence.*?(\d+(?:\.\d+)?)%",
                ],
                "price_confidence": [
                    r"price.*?confidence.*?(\d+(?:\.\d+)?)%",
                    r"price.*?prediction.*?confidence.*?(\d+(?:\.\d+)?)%",
                    r"target.*?confidence.*?(\d+(?:\.\d+)?)%",
                ],
                "move_percentage": [
                    r"([+-]?\d+(?:\.\d+)?)%.*?move",
                    r"expected.*?([+-]?\d+(?:\.\d+)?)%",
                    r"change.*?([+-]?\d+(?:\.\d+)?)%",
                ],
                "predicted_price": [
                    r"predict.*?\$([\d,]+(?:\.\d+)?)",
                    r"predicted price.*?\$([\d,]+(?:\.\d+)?)",
                    r"will be.*?\$([\d,]+(?:\.\d+)?)",
                    r"target.*?\$([\d,]+(?:\.\d+)?)",
                ],
            }

            for key in [
                "higher",
                "lower",
                "confidence",
                "price_confidence",
                "move_percentage",
                "predicted_price",
            ]:
                for pat in patterns[key]:
                    m = re.findall(pat, text, flags=re.IGNORECASE | re.DOTALL)
                    if m:
                        if key == "predicted_price":
                            price_str = str(m[0]).replace(",", "")
                            try:
                                probs["predicted_price"] = float(price_str)
                            except Exception:
                                pass
                        else:
                            try:
                                val = float(m[0])
                            except Exception:
                                continue
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
            self._dbg("warning", f"Error extracting probabilities: {e}")
        return probs
