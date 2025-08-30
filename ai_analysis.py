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
                try:
                    max_val = df.index.max()
                    if pd.isna(max_val):
                        end = pd.Timestamp('now')
                    else:
                        end = pd.Timestamp(max_val)
                except Exception:
                    end = pd.Timestamp('now')
                start = end - pd.Timedelta(days=days)
                mask = df.index >= start
                return df.loc[mask]

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

            def _determine_timezone(self, *indexes: Optional[pd.Index]):
                for idx in indexes:
                    if idx is not None and isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
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
                        if target_datetime.tzinfo:
                            target = target_datetime.astimezone(tz)
                        else:
                            target = target_datetime.replace(tzinfo=tz)
                    else:
                        base_df = window_1w if not window_1w.empty else window_3m
                        step = self._infer_index_step(base_df.index)
                        if step is None:
                            return {"prep_status": "insufficient_data", "prep_notes": ["no_target_and_cannot_infer_step"]}
                        try:
                            last_val = base_df.index[-1]
                            if pd.isna(last_val):
                                target_ts = pd.Timestamp('now')
                            else:
                                target_ts = pd.Timestamp(last_val)
                            target = target_ts + step
                        except Exception:
                            target = pd.Timestamp('now') + step
                        if target.tzinfo:
                            target = target.tz_convert(tz)
                        else:
                            if hasattr(tz, 'localize'):
                                target = tz.localize(target.to_pydatetime())
                            else:
                                target = target.tz_localize(tz)

                    if (target - current_time).total_seconds() < 0:
                        return {"prep_status": "insufficient_data", "prep_notes": ["target_before_current_time"]}

                    actual_current_price = float(current_price)

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

                    return {
                        "asset_name": asset_name,
                        "current_time": current_time.isoformat(),
                        "target_time": target.isoformat(),
                        "hours_until_target": (target - current_time).total_seconds() / 3600.0,
                        "data_3m": data_3m_summary,
                        "data_1w": data_1w_summary,
                        "indicators": indicators_summary,
                        "enhanced_chart_data": enhanced_data,
                        "features": enhanced_data.get("features", {}),
                        "current_price": actual_current_price,
                    }

                except Exception as e:
                    st.error(f"Error preparing analysis data: {str(e)}")
                    return {"prep_status": "insufficient_data", "prep_notes": [str(e)]}

            # ---------- message and model call helpers ----------

            def _build_messages(self, analysis_data: Dict[str, Any], asset_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
                system_content = (
                    "You are a deterministic technical analyst. Use ONLY the structured arrays provided. "
                    "Forbidden: news, macro, on-chain, session effects (market open/close), weekdays, seasonality, etc. "
                    "Do not infer beyond given timestamps. If required arrays are missing/empty, output status='insufficient_data'."
                )

                data_3m = analysis_data.get("data_3m", {})
                data_1w = analysis_data.get("data_1w", {})

                # JSON schema for comparison
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

