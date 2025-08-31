# ai_analysis_fixed.py
# Robust rewrite of your AIAnalyzer to avoid breakages, add safer parsing,
# richer deterministic features, fuller arrays with timestamps, and a baseline forecast.

from __future__ import annotations

import os
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import pytz

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
                self.client = None
        else:
            if not api_key:
                st.error("OpenAI API key not found. Set OPENAI_API_KEY.")
            if OpenAI is None:
                st.error("openai SDK not available in environment.")

    def _extract_text_from_responses(self, resp: Any) -> str:
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
        except Exception:
            pass
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
        # 1) Try Responses API
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
        # 2) Fallback: Chat Completions
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


# =========================
#  Core Analyzer
# =========================
class AIAnalyzer:
    """
    AIAnalyzer — strict, data-first technical analysis + narrative + point forecast.

    Outputs expected:
      - 'technical_summary' (markdown, from [TECHNICAL_ANALYSIS_*] block)
      - 'price_prediction' (markdown, from [PRICE_PREDICTION_*] block)
      - 'probabilities' (numbers for gauges)
      - 'model_json' (parsed JSON block)
      - 'status' ('ok' | 'insufficient_data' | 'error')
    """

    # Limits to keep tokens sane while preserving full windows
    MAX_POINTS_PRICE = 360     # per timeframe after downsampling
    MAX_POINTS_INDIC = 360     # per indicator per timeframe

    def __init__(self, debug: Optional[bool] = None):
        if debug is None:
            self.debug = os.getenv("AI_ANALYZER_DEBUG", "0") == "1"
        else:
            self.debug = bool(debug)
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.model_name = os.getenv("GPT5_MODEL", "gpt-5-nano")
        self.gpt = _OpenAIWrapper(self.openai_key, self.model_name, debug=self.debug)
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

            # Extract rich sections
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
            end = pd.Timestamp(df.index.max()) if pd.notna(df.index.max()) else pd.Timestamp("now")
        except Exception:
            end = pd.Timestamp("now")
        start = end - pd.Timedelta(days=days)
        return df.loc[df.index >= start]

    def _infer_index_step(self, index: pd.Index) -> Optional[pd.Timedelta]:
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

    def _annualization_sqrt(self, index: pd.Index) -> float:
        """Use median step to annualize robustly."""
        step = self._infer_index_step(index)
        if not step:
            return 1.0
        secs = step.total_seconds()
        if secs <= 0:
            return 1.0
        periods_per_year = (365 * 24 * 3600) / secs
        return float(np.sqrt(periods_per_year))

    def _determine_timezone(self, *indexes: Optional[pd.Index]):
        for idx in indexes:
            if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
                return idx.tz
        return pytz.timezone("US/Eastern")

    # ---------- simple downsampling to cap tokens while keeping full window ----------
    def _downsample_df(self, df: pd.DataFrame, max_points: int) -> pd.DataFrame:
        if df is None or df.empty or len(df) <= max_points:
            return df
        idx = np.linspace(0, len(df) - 1, num=max_points).round().astype(int)
        return df.iloc[idx]

    def _downsample_series(self, s: pd.Series, max_points: int) -> pd.Series:
        s = s.dropna()
        if s.empty or len(s) <= max_points:
            return s
        idx = np.linspace(0, len(s) - 1, num=max_points).round().astype(int)
        return s.iloc[idx]

    # ---------- packing helpers ----------
    def _safe_format_datetime(self, dt) -> str:
        try:
            if pd.isna(dt):
                return "N/A"
            ts = pd.Timestamp(dt)
            if pd.isna(ts) or str(ts) == "NaT":
                return "N/A"
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

    def _pack_series(self, s: pd.Series, max_points: int) -> Dict[str, list]:
        s = self._downsample_series(s, max_points)
        s = s.dropna()
        return {
            "dates": [self._safe_format_datetime(t) for t in s.index],
            "values": [float(x) for x in s.round(6).tolist()],
        }

    # ---------- deterministic stats ----------
    def _lin_slope_and_r2(self, s: pd.Series, k: Optional[int] = None) -> Tuple[Optional[float], Optional[float]]:
        try:
            y = s.dropna()
            if k is not None:
                y = y.tail(k)
            if len(y) < 3:
                return None, None
            x = np.arange(len(y), dtype=float)
            m, b = np.polyfit(x, y.values.astype(float), 1)
            # R^2
            y_hat = m * x + b
            ss_res = np.sum((y.values - y_hat) ** 2)
            ss_tot = np.sum((y.values - y.values.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
            return float(m), (float(r2) if r2 is not None else None)
        except Exception:
            return None, None

    def _percentile_of_value(self, s: pd.Series, v: Optional[float]) -> Optional[float]:
        try:
            s = s.dropna()
            if s.empty or v is None or np.isnan(v):
                return None
            return float((s.values <= v).mean() * 100.0)
        except Exception:
            return None

    def _macd_cross_metrics(self, macd: pd.Series, signal: pd.Series) -> Dict[str, Optional[float]]:
        try:
            macd, signal = macd.dropna(), signal.dropna()
            n = min(len(macd), len(signal))
            if n < 3:
                return {"bars_since_cross": None, "last_cross": None}
            m = macd.tail(n).values
            s = signal.tail(n).values
            diff = m - s
            signs = np.sign(diff)
            # find last sign change (exclude tail)
            changes = np.where(np.diff(signs) != 0)[0]
            if len(changes) == 0:
                return {"bars_since_cross": None, "last_cross": None}
            last_change_idx = changes[-1]  # index before the cross bar
            bars_since = (n - 1) - (last_change_idx + 1)
            last_cross = "bullish" if diff[last_change_idx + 1] > 0 else "bearish"
            return {"bars_since_cross": float(bars_since), "last_cross": last_cross}
        except Exception:
            return {"bars_since_cross": None, "last_cross": None}

    def _hist_sign_flips(self, hist: pd.Series, k: int = 50) -> Optional[int]:
        try:
            h = hist.dropna().tail(k).values
            if len(h) < 3:
                return None
            signs = np.sign(h)
            flips = int(np.sum(np.diff(signs) != 0))
            return flips
        except Exception:
            return None

    def _price_ema_distance_bps(self, price: pd.Series, ema: pd.Series) -> Optional[float]:
        try:
            p = float(price.dropna().iloc[-1])
            e = float(ema.dropna().iloc[-1])
            if e == 0:
                return None
            return float((p - e) / e * 1e4)  # basis points
        except Exception:
            return None

    # ---------- swing structure & levels ----------
    def _swing_points(self, close: pd.Series, lookback: int = 2) -> Dict[str, List[int]]:
        # naive local extrema: a point greater/less than neighbors within lookback
        s = close.dropna()
        idxs = np.arange(len(s))
        peaks, troughs = [], []
        for i in range(lookback, len(s) - lookback):
            window = s.iloc[i - lookback:i + lookback + 1]
            c = s.iloc[i]
            if c == window.max() and (window.values.argmax() == lookback):
                peaks.append(i)
            if c == window.min() and (window.values.argmin() == lookback):
                troughs.append(i)
        return {"peaks": peaks, "troughs": troughs}

    def _hh_hl_counts(self, close: pd.Series, swing: Dict[str, List[int]]) -> Dict[str, Optional[int]]:
        try:
            s = close.dropna()
            peaks = swing.get("peaks", [])
            troughs = swing.get("troughs", [])
            hh = sum(s.iloc[peaks[i]] > s.iloc[peaks[i - 1]] for i in range(1, len(peaks))) if len(peaks) >= 2 else 0
            hl = sum(s.iloc[troughs[i]] > s.iloc[troughs[i - 1]] for i in range(1, len(troughs))) if len(troughs) >= 2 else 0
            lh = sum(s.iloc[peaks[i]] < s.iloc[peaks[i - 1]] for i in range(1, len(peaks))) if len(peaks) >= 2 else 0
            ll = sum(s.iloc[troughs[i]] < s.iloc[troughs[i - 1]] for i in range(1, len(troughs))) if len(troughs) >= 2 else 0
            return {"HH": int(hh), "HL": int(hl), "LH": int(lh), "LL": int(ll)}
        except Exception:
            return {"HH": None, "HL": None, "LH": None, "LL": None}

    def _support_resistance(self, close: pd.Series, swing: Dict[str, List[int]], top_n: int = 3) -> Dict[str, List[float]]:
        try:
            s = close.dropna()
            peaks = swing.get("peaks", [])
            troughs = swing.get("troughs", [])
            res = sorted([float(s.iloc[i]) for i in peaks], reverse=True)[:top_n]
            sup = sorted([float(s.iloc[i]) for i in troughs])[:top_n]
            return {"resistance": res, "support": sup}
        except Exception:
            return {"resistance": [], "support": []}

    # ---------- baseline (deterministic) forecast ----------
    def _baseline_forecast(
        self, prices: pd.Series, hours_until_target: float, step_seconds: Optional[float]
    ) -> Dict[str, Optional[float]]:
        try:
            c = prices.dropna()
            if len(c) < 3:
                return {"pred": None, "lower": None, "upper": None, "exp_move_pct": None}
            # log returns per bar
            r = np.log(c / c.shift(1)).dropna()
            if r.empty:
                return {"pred": None, "lower": None, "upper": None, "exp_move_pct": None}
            mu = float(r.mean())
            sig = float(r.std())
            if step_seconds is None or step_seconds <= 0:
                bars = 1.0
            else:
                bars = max(1.0, hours_until_target * 3600.0 / step_seconds)
            s0 = float(c.iloc[-1])
            # GBM expected value over bars (no shock term; mean of distribution)
            pred = s0 * float(np.exp(mu * bars))
            # Expected one-sigma move magnitude over bars
            exp_move_pct = float(sig * np.sqrt(bars) * 100.0)
            lower = float(pred * np.exp(-sig * np.sqrt(bars)))
            upper = float(pred * np.exp(sig * np.sqrt(bars)))
            return {"pred": pred, "lower": lower, "upper": upper, "exp_move_pct": exp_move_pct}
        except Exception:
            return {"pred": None, "lower": None, "upper": None, "exp_move_pct": None}

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

            notes: List[str] = []
            if (window_3m is None or window_3m.empty) and (window_1w is None or window_1w.empty):
                return {"prep_status": "insufficient_data", "prep_notes": ["no_price_data_after_trimming"]}

            tz = self._determine_timezone(
                window_1w.index if not window_1w.empty else None,
                window_3m.index if not window_3m.empty else None,
            )
            current_time = datetime.now(tz)

            # Target handling
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

            # Warn if current price diverges from last close
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

            # Data sufficiency checks for indicators
            def _have(inds: Dict[str, pd.Series], key: str) -> bool:
                return inds is not None and key in inds and not inds[key].dropna().empty

            required = ["RSI", "MACD", "MACD_Signal", "MACD_Histogram", "BB_Upper", "BB_Lower", "BB_Middle", "EMA_20"]
            for tf, inds in (("3m", indicators_3m), ("1w", indicators_1w)):
                missing = [k for k in required if not _have(inds, k)]
                if missing:
                    notes.append(f"{tf}_missing:{','.join(missing)}")

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

            data_3m_summary = _summary(window_3m, "3m") if not window_3m.empty else {}
            data_1w_summary = _summary(window_1w, "1w") if not window_1w.empty else {}

            # Enhanced arrays (full windows, downsampled)
            enhanced_data = self._prepare_enhanced_chart_data(window_3m, window_1w, indicators_3m, indicators_1w)

            # Adjust highs/lows from enhanced (full range)
            if enhanced_data.get("3m", {}).get("period_highs_lows") and data_3m_summary:
                data_3m_summary["high"] = enhanced_data["3m"]["period_highs_lows"].get("period_high")
                data_3m_summary["low"] = enhanced_data["3m"]["period_highs_lows"].get("period_low")
            if enhanced_data.get("1w", {}).get("period_highs_lows") and data_1w_summary:
                data_1w_summary["high"] = enhanced_data["1w"]["period_highs_lows"].get("period_high")
                data_1w_summary["low"] = enhanced_data["1w"]["period_highs_lows"].get("period_low")

            # Deterministic features & structure
            features = self._compute_features(window_3m, window_1w, indicators_3m, indicators_1w)

            # Baseline forecast from the *denser* timeframe (prefer 1w, else 3m)
            base_df = window_1w if not window_1w.empty else window_3m
            step = self._infer_index_step(base_df.index) if base_df is not None and not base_df.empty else None
            step_secs = step.total_seconds() if step is not None else None
            baseline = self._baseline_forecast(
                base_df["Close"] if (base_df is not None and "Close" in base_df.columns) else pd.Series(dtype=float),
                hours_until_target=(pd.Timestamp(target) - pd.Timestamp(current_time)).total_seconds() / 3600.0,
                step_seconds=step_secs,
            )

            return {
                "asset_name": asset_name,
                "current_time": current_time.isoformat(),
                "target_time": target.isoformat(),
                "hours_until_target": (target - current_time).total_seconds() / 3600.0,
                "data_3m": data_3m_summary,
                "data_1w": data_1w_summary,
                "indicators": self._summarize_indicators(indicators_3m, indicators_1w, actual_current_price),
                "enhanced_chart_data": enhanced_data,
                "features": features,
                "current_price": actual_current_price,
                "baseline": baseline,
                "data_notes": notes,
            }

        except Exception as e:
            st.error(f"Error preparing analysis data: {e}")
            return {"prep_status": "insufficient_data", "prep_notes": [str(e)]}

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

            def build_timeframe_block(df: pd.DataFrame, inds: Dict[str, pd.Series], label: str) -> Dict[str, Any]:
                full = df if df is not None else pd.DataFrame()
                full = full.copy()
                if not full.empty:
                    # Downsample prices but keep full window coverage
                    full_ds = self._downsample_df(full, self.MAX_POINTS_PRICE)
                    bar_step = self._infer_index_step(full.index)
                    bar_step_secs = float(bar_step.total_seconds()) if bar_step is not None else None
                    block = {
                        "timeframe": label,
                        "bars": int(len(full.index)),
                        "bar_step_seconds": bar_step_secs,
                        "full_range": self._safe_format_daterange(full.index[0], full.index[-1]),
                        "period_highs_lows": {
                            "period_high": float(full["High"].max()) if "High" in full.columns else None,
                            "period_low": float(full["Low"].min()) if "Low" in full.columns else None,
                            "recent_high": float(full.tail(max(1, len(full)//3))["High"].max()) if "High" in full.columns else None,
                            "recent_low": float(full.tail(max(1, len(full)//3))["Low"].min()) if "Low" in full.columns else None,
                        },
                        "prices": {
                            "dates": [self._safe_format_datetime(d) for d in full_ds.index],
                            "open": full_ds.get("Open", pd.Series(dtype=float)).round(4).astype(float).tolist(),
                            "high": full_ds.get("High", pd.Series(dtype=float)).round(4).astype(float).tolist(),
                            "low": full_ds.get("Low", pd.Series(dtype=float)).round(4).astype(float).tolist(),
                            "close": full_ds.get("Close", pd.Series(dtype=float)).round(4).astype(float).tolist(),
                            "volume": full_ds.get("Volume", pd.Series(dtype=float)).round(0).astype(float).tolist(),
                        },
                        "indicators": {},
                    }
                else:
                    block = {
                        "timeframe": label,
                        "bars": 0,
                        "bar_step_seconds": None,
                        "full_range": "N/A",
                        "period_highs_lows": {"period_high": None, "period_low": None, "recent_high": None, "recent_low": None},
                        "prices": {"dates": [], "open": [], "high": [], "low": [], "close": [], "volume": []},
                        "indicators": {},
                    }

                for indicator in [
                    "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
                    "BB_Upper", "BB_Lower", "BB_Middle",
                    "EMA_20", "SMA_50", "SMA_200",
                ]:
                    if inds and indicator in inds:
                        block["indicators"][indicator] = self._pack_series(inds[indicator], self.MAX_POINTS_INDIC)

                # Volume percentiles
                if not full.empty and "Volume" in full.columns:
                    last_vol = float(full["Volume"].iloc[-1])
                    vol_pct = self._percentile_of_value(full["Volume"], last_vol)
                else:
                    vol_pct = None
                block["volume_percentile"] = vol_pct
                return block

            enhanced["3m"] = build_timeframe_block(data_3m, indicators_3m, "3m")
            enhanced["1w"] = build_timeframe_block(data_1w, indicators_1w, "1w")
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
                indicators_3m["RSI"].iloc[-1]
                if indicators_3m and "RSI" in indicators_3m and not indicators_3m["RSI"].empty
                else np.nan
            )
            rsi_1w_last = (
                indicators_1w["RSI"].iloc[-1]
                if indicators_1w and "RSI" in indicators_1w and not indicators_1w["RSI"].empty
                else np.nan
            )
            summary["RSI"] = {
                "3m_current": float(rsi_3m_last) if not np.isnan(rsi_3m_last) else None,
                "1w_current": float(rsi_1w_last) if not np.isnan(rsi_1w_last) else None,
            }

            if indicators_3m and "MACD" in indicators_3m and "MACD_Signal" in indicators_3m:
                macd_3m = indicators_3m["MACD"].iloc[-1]
                signal_3m = indicators_3m["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_3m) and not np.isnan(signal_3m):
                    summary["MACD_3m"] = {
                        "macd": float(macd_3m),
                        "signal": float(signal_3m),
                        "crossover": "bullish" if macd_3m > signal_3m else "bearish",
                    }

            if indicators_1w and "MACD" in indicators_1w and "MACD_Signal" in indicators_1w:
                macd_1w = indicators_1w["MACD"].iloc[-1]
                signal_1w = indicators_1w["MACD_Signal"].iloc[-1]
                if not np.isnan(macd_1w) and not np.isnan(signal_1w):
                    summary["MACD_1w"] = {
                        "macd": float(macd_1w),
                        "signal": float(signal_1w),
                        "crossover": "bullish" if macd_1w > signal_1w else "bearish",
                    }

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and all(k in inds for k in ["BB_Upper", "BB_Lower", "BB_Middle"]):
                    upper = inds["BB_Upper"].iloc[-1]
                    lower = inds["BB_Lower"].iloc[-1]
                    middle = inds["BB_Middle"].iloc[-1]
                    if not any(pd.isna([upper, lower, middle])):
                        cp = summary["current_price"]
                        summary[f"BB_{timeframe}"] = {
                            "upper": float(upper),
                            "lower": float(lower),
                            "middle": float(middle),
                            "position": "above_upper" if cp > upper else ("below_lower" if cp < lower else "within_bands"),
                        }

            for timeframe, inds in [("3m", indicators_3m), ("1w", indicators_1w)]:
                if inds and "EMA_20" in inds:
                    ema_20 = inds["EMA_20"].iloc[-1]
                    if not np.isnan(ema_20):
                        cp = summary["current_price"]
                        summary[f"EMA_20_{timeframe}"] = {
                            "value": float(ema_20),
                            "trend": "bullish" if cp > ema_20 else "bearish",
                        }
        except Exception as e:
            st.warning(f"Error summarizing indicators: {e}")
        return summary

    # ---------- deterministic features & structure ----------
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

        def vol_ann(df: pd.DataFrame) -> Optional[float]:
            try:
                return float(df["Close"].pct_change().std() * self._annualization_sqrt(df.index) * 100.0)
            except Exception:
                return None

        # Per-timeframe helper
        def tf_features(df: pd.DataFrame, ind: Dict[str, pd.Series], prefix: str) -> Dict[str, Any]:
            f: Dict[str, Any] = {}
            if df is not None and not df.empty:
                f[f"ret_{prefix}_last5bars"] = last_n_returns(df, 5)
                f[f"vol_ann_{prefix}_pct"] = vol_ann(df)

                # Slopes/R² of Close & RSI & MACD_Histogram
                m_close, r2_close = self._lin_slope_and_r2(df["Close"], k=None)
                f[f"close_slope_{prefix}"] = m_close
                f[f"close_r2_{prefix}"] = r2_close

            if ind:
                # EMA distance & slope
                if "EMA_20" in ind:
                    f[f"ema20_slope_{prefix}"], _ = self._lin_slope_and_r2(ind["EMA_20"], k=30)
                    if df is not None and not df.empty:
                        f[f"price_ema20_dist_bps_{prefix}"] = self._price_ema_distance_bps(df["Close"], ind["EMA_20"])

                # RSI
                if "RSI" in ind:
                    rsi_last = float(ind["RSI"].dropna().iloc[-1]) if not ind["RSI"].dropna().empty else None
                    f[f"rsi_last_{prefix}"] = rsi_last
                    f[f"rsi_slope_{prefix}_20"], f[f"rsi_r2_{prefix}_20"] = self._lin_slope_and_r2(ind["RSI"], k=20)
                    f[f"rsi_pctile_{prefix}"] = self._percentile_of_value(ind["RSI"], rsi_last)

                # MACD / Signal / Histogram
                if "MACD" in ind and "MACD_Signal" in ind:
                    mc = ind["MACD"]
                    sg = ind["MACD_Signal"]
                    cross = self._macd_cross_metrics(mc, sg)
                    f[f"macd_bars_since_cross_{prefix}"] = cross.get("bars_since_cross")
                    f[f"macd_last_cross_{prefix}"] = cross.get("last_cross")
                if "MACD_Histogram" in ind:
                    mh = ind["MACD_Histogram"]
                    f[f"macd_hist_slope_{prefix}_20"], f[f"macd_hist_r2_{prefix}_20"] = self._lin_slope_and_r2(mh, k=20)
                    f[f"macd_hist_flips_{prefix}_50"] = self._hist_sign_flips(mh, k=50)

                # BB width
                if "BB_Upper" in ind and "BB_Lower" in ind:
                    try:
                        bbw = (ind["BB_Upper"].iloc[-1] - ind["BB_Lower"].iloc[-1])
                        f[f"bb_width_{prefix}"] = float(bbw)
                    except Exception:
                        f[f"bb_width_{prefix}"] = None

            # Swing structure & SR levels
            if df is not None and not df.empty:
                swing = self._swing_points(df["Close"], lookback=2)
                f.update({f"{k}_{prefix}": v for k, v in self._hh_hl_counts(df["Close"], swing).items()})
                sr = self._support_resistance(df["Close"], swing, top_n=3)
                f[f"sr_resistance_{prefix}"] = sr["resistance"]
                f[f"sr_support_{prefix}"] = sr["support"]

            return f

        feats.update(tf_features(data_1w, ind_1w, "1w"))
        feats.update(tf_features(data_3m, ind_3m, "3m"))
        return feats

    # ---------- prompt + model call ----------
    def _build_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        # Pull a few values for the template tables (filled by us so the output is already richer)
        tf3m = (analysis_data.get("enhanced_chart_data") or {}).get("3m", {}) or {}
        tf1w = (analysis_data.get("enhanced_chart_data") or {}).get("1w", {}) or {}

        bars_3m = tf3m.get("bars", 0)
        bars_1w = tf1w.get("bars", 0)
        step_3m = tf3m.get("bar_step_seconds")
        step_1w = tf1w.get("bar_step_seconds")
        range_3m = tf3m.get("full_range", "N/A")
        range_1w = tf1w.get("full_range", "N/A")

        baseline = analysis_data.get("baseline", {}) or {}
        baseline_pred = baseline.get("pred")
        baseline_lower = baseline.get("lower")
        baseline_upper = baseline.get("upper")
        baseline_exp = baseline.get("exp_move_pct")

        system_content = (
            "You are a deterministic technical analyst. Use ONLY the structured arrays, deterministic FEATURES, "
            "and baseline forecast provided. You must produce a thorough, quantitative narrative with tables "
            "and at least 12 numbered evidence bullets. Explicitly compare 1w vs 3m for RSI, MACD, BB, and EMA. "
            "Forbidden: news, macro, seasonality, or any external info. Every claim must be grounded in the data. "
            "Do NOT leave placeholders; no angle brackets like <value> may appear in the final output. "
            "Use exact numeric values with units where relevant. Keep the narrative between 200 and 800 words."
        )

        data_3m = analysis_data.get("data_3m", {})
        data_1w = analysis_data.get("data_1w", {})

        # JSON schema (unchanged apart from baseline fields already populated)
        output_schema = {
            "status": "ok or insufficient_data",
            "asset": asset_name,
            "as_of": analysis_data.get("current_time"),
            "target_ts": analysis_data.get("target_time"),
            "predicted_price": "number or null",
            "baseline_pred": baseline_pred,
            "baseline_band": {"lower": baseline_lower, "upper": baseline_upper, "exp_move_pct": baseline_exp},
            "p_up": "float 0..1",
            "p_down": "float 0..1 (p_up + p_down = 1)",
            "conf_overall": "float 0..1",
            "conf_price": "float 0..1",
            "expected_pct_move": "signed float percent (optional; compute if omitted)",
            "critical_levels": {"bullish_above": "number or null", "bearish_below": "number or null"},
            "evidence": [
                {
                    "type": "rsi|macd|bb|ema|price|volume|structure",
                    "timeframe": "3m|1w",
                    "ts": "YYYY-MM-DD HH:MM" or "recent",
                    "value": "number or object",
                    "note": "short factual note",
                }
            ],
            "notes": ["if status=insufficient_data, list what's missing; else optional warnings"],
        }

        # === Narrative template with hard requirements for content density ===
        # We pre-fill the coverage table and baseline numbers so the output is not sparse even if the model is lazy.
        narrative_template = f"""
[TECHNICAL_ANALYSIS_START]
### Methodology & Coverage
- **As-of:** {analysis_data.get('current_time')}  |  **Target:** {analysis_data.get('target_time')}  |  **Horizon:** ~{analysis_data.get('hours_until_target', 0):.2f}h

| Timeframe | Bars | Step (s) | Range |
|---|---:|---:|---|
| 1w | {bars_1w} | {step_1w} | {range_1w} |
| 3m | {bars_3m} | {step_3m} | {range_3m} |

### Multi-Timeframe Overview (≥3 bullets; quantify with % or slopes)
1. ...
2. ...
3. ...

### Indicator Tables (fill with precise numbers; no placeholders)

**RSI (levels, slopes, R², percentile)**
| TF | Last | Slope(20) | R²(20) | Percentile(%) | Note |
|---|---:|---:|---:|---:|---|
| 1w |  |  |  |  |  |
| 3m |  |  |  |  |  |

**MACD (values, histogram slope & flips, cross timing)**
| TF | MACD | Signal | Hist Slope(20) | R²(20) | BarsSinceCross | LastCross | Flips(50) |
|---|---:|---:|---:|---:|---:|---|---:|
| 1w |  |  |  |  |  |  |  |
| 3m |  |  |  |  |  |  |  |

**BB & EMA (width, price–EMA distance)**
| TF | BB Width | Price vs EMA20 (bps) | EMA20 Slope(30) | Position vs Bands |
|---|---:|---:|---:|---|
| 1w |  |  |  |  |
| 3m |  |  |  |  |

### Structure & Levels
- Swing structure counts **(HH/HL/LH/LL)** by timeframe and **Top-3 Support/Resistance** with exact prices.

### Volume
- Volume percentile per timeframe and any price/volume divergences.

### Alignment & Conflicts (≥3 bullets)
1. ...
2. ...
3. ...

### Baseline Check
- Baseline: **{baseline_pred}** (±{baseline_exp}%). Band: **[{baseline_lower}, {baseline_upper}]**. State whether you **agree or adjust**, and quantify by how much based on data.

### Quantified Evidence (≥12 bullets across RSI, MACD, BB, EMA, Price, Volume, Structure)
1. ...
2. ...
3. ...
4. ...
5. ...
6. ...
7. ...
8. ...
9. ...
10. ...
11. ...
12. ...

### Trading View
- Overall Bias (**BULLISH/BEARISH/NEUTRAL**) with exact numeric justification.
- Key levels (bullish-above / bearish-below) and why.

[TECHNICAL_ANALYSIS_END]

[PRICE_PREDICTION_START]
**PREDICTED PRICE:** I predict {asset_name} will be at **$<PRICE>** on {analysis_data.get('target_time')}.
- Horizon: ~{analysis_data.get('hours_until_target', 0):.2f}h
- Momentum & Trend: cite specific slopes/cross timings.
- Expected Move (from realized vol/features): include number; consider baseline {baseline_exp}% for calibration.

1. **Probability HIGHER than ${analysis_data.get('current_price'):,.2f}: <X>%**
2. **Probability LOWER than ${analysis_data.get('current_price'):,.2f}: <Y>%**
3. **Overall Analysis Confidence: <Z>%**
4. **Price Prediction Confidence: <W>%**
5. **Expected % Move: <±M>%**

**Scenario Targets**
- **Bull case (if above bullish level):** two numbered targets with data justification.
- **Bear case (if below bearish level):** two numbered targets with data justification.

**Critical Levels**
- **Bullish above:** $<level>
- **Bearish below:** $<level>
[PRICE_PREDICTION_END]
""".strip()

        # Strong output rules to stop the model from leaving placeholders
        hard_rules = (
            "OUTPUT RULES:\n"
            "- Replace ALL '...' and ALL angle-bracket placeholders with real numbers/words from the data; no '<>' allowed.\n"
            "- Every table cell must be filled; if unknown, compute from arrays/features or state 'N/A' explicitly.\n"
            "- Provide at least 12 evidence bullets. Provide at least 3 bullets in 'Alignment & Conflicts' and 3 in 'Overview'.\n"
            "- Use units: %, bps, bars, or USD as appropriate.\n"
        )

        user_content = f"""
ASSET: {asset_name}

DATA WINDOWS (strict):
- 3m index range: {data_3m.get('start_date','N/A')} → {data_3m.get('end_date','N/A')}
- 1w index range: {data_1w.get('start_date','N/A')} → {data_1w.get('end_date','N/A')}

TARGET: {analysis_data.get('target_time')}
AS_OF: {analysis_data.get('current_time')}
CURRENT_PRICE: {analysis_data.get('current_price')}

DATA_NOTES:
{json.dumps(analysis_data.get('data_notes', []), indent=2)}

BASELINE_FORECAST:
{json.dumps(baseline, indent=2)}

INDICATOR SNAPSHOT:
{json.dumps(analysis_data.get('indicators', {}), indent=2)}

FEATURES (deterministic metrics):
{json.dumps(analysis_data.get('features', {}), indent=2)}

ENHANCED ARRAYS (timestamps + values; includes bar_step_seconds & bars):
{json.dumps(analysis_data.get('enhanced_chart_data', {}), indent=2)}

STRICT RULES:
- Use ONLY these arrays/features; no external info.
- Quantify slopes (m), r², percentiles, cross timings, flips, and EMA distance (bps).
- Mandatory comparisons: for RSI/MACD/BB/EMA, compare 1w vs 3m and state alignment.
- Structure: use HH/HL/LH/LL counts and SR levels provided (or compute from arrays).
- Volume: use volume_percentile and price-volume behavior from arrays.
- Baseline: anchor on it; you may adjust with evidence. If you deviate, explain why.
- Predictions: give a numeric predicted_price; ensure p_up + p_down = 1.
- If arrays are empty/missing, return status='insufficient_data' with specifics.

{hard_rules}

            YOU MUST RETURN **TWO** BLOCKS, IN THIS ORDER:
            1) A VALID JSON object, in a fenced code block like:
            ```json
            {json.dumps(output_schema, indent=2)}
            ```
            2) A narrative block using EXACTLY the markers and structure above.
            """.strip()

        return ({"role": "system", "content": system_content}, {"role": "user", "content": user_content})

    def _generate_technical_analysis_gpt5(self, analysis_data: Dict[str, Any]) -> str:
        asset_name = analysis_data.get("asset_name", "Asset")
        system_msg, user_msg = self._build_messages(analysis_data, asset_name)
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
        if not raw:
            return "", ""
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL | re.IGNORECASE)
        if m:
            json_block = m.group(1).strip()
            narrative = (raw[: m.start()] + raw[m.end() :]).strip()
            return json_block, narrative
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
        return "", raw.strip()

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

    def _parse_json_response(self, response_text: str, current_price: float) -> Dict[str, Any]:
        if not response_text:
            return {"status": "insufficient_data", "notes": ["no_json_block_found"]}
        try:
            data = json.loads(response_text)
            if not isinstance(data, dict):
                return {"status": "insufficient_data", "notes": ["non_dict_json"]}

            st_raw = str(data.get("status", "insufficient_data")).lower().strip()
            data["status"] = "ok" if st_raw == "ok" else "insufficient_data"

            p_up = self._clip01(self._safe_num(data.get("p_up"), 0.5))
            p_down = self._clip01(self._safe_num(data.get("p_down"), 0.5))
            total = p_up + p_down
            if total <= 0:
                p_up, p_down = 0.5, 0.5
            else:
                p_up, p_down = p_up / total, p_down / total
            data["p_up"], data["p_down"] = p_up, p_down

            data["conf_overall"] = self._clip01(self._safe_num(data.get("conf_overall"), 0.5))
            data["conf_price"] = self._clip01(self._safe_num(data.get("conf_price"), 0.5))

            cp = float(current_price) if current_price is not None else None
            pred = self._safe_num(data.get("predicted_price"), None)
            if pred is not None and cp and cp > 0:
                data["expected_pct_move"] = (float(pred) - cp) / cp * 100.0

            data["critical_levels"] = data.get("critical_levels") or {"bullish_above": None, "bearish_below": None}
            return data
        except Exception as e:
            return {"status": "insufficient_data", "notes": [f"json_parse_error:{e}"]}

    def _parse_comprehensive_response(self, response: str) -> Dict[str, str]:
        try:
            sections: Dict[str, str] = {}
            tech_start = response.find("[TECHNICAL_ANALYSIS_START]")
            tech_end = response.find("[TECHNICAL_ANALYSIS_END]")
            if tech_start != -1 and tech_end != -1 and tech_end > tech_start:
                sections["technical_summary"] = response[tech_start + len("[TECHNICAL_ANALYSIS_START]") : tech_end].strip()
            pred_start = response.find("[PRICE_PREDICTION_START]")
            pred_end = response.find("[PRICE_PREDICTION_END]")
            if pred_start != -1 and pred_end != -1 and pred_end > pred_start:
                sections["price_prediction"] = response[pred_start + len("[PRICE_PREDICTION_START]") : pred_end].strip()

            if not sections:
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
            t = e.get("type", "fact"); tf = e.get("timeframe", ""); ts = e.get("ts", ""); note = e.get("note", "")
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
            for key in ["higher", "lower", "confidence", "price_confidence", "move_percentage", "predicted_price"]:
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
            probs["higher_fraction"] = probs["higher_pct"] / 100.0
            probs["lower_fraction"] = probs["lower_pct"] / 100.0
            probs["confidence_fraction"] = probs["confidence_pct"] / 100.0
        except Exception as e:
            self._dbg("warning", f"Error extracting probabilities: {e}")
        return probs


# Optional: import contextlib for safe suppress above
import contextlib  # keep at end to avoid masking top imports in some environments
