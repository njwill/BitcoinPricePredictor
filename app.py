import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from data_fetcher import BitcoinDataFetcher
from chart_generator import ChartGenerator
from technical_analysis import TechnicalAnalyzer
from ai_analysis import AIAnalyzer
from scheduler import ScheduleManager
from utils import format_currency, get_eastern_time, calculate_time_until_update, should_update_analysis, save_analysis_cache, load_analysis_cache

# Configure page
st.set_page_config(
    page_title="Bitcoin Analysis Dashboard",
    page_icon="₿",
    layout="wide"
)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

def main():
    # Add custom CSS for consistent fonts
    st.markdown("""
    <style>
    /* Force consistent font family throughout */
    .stApp, .stApp * {
        font-family: "Source Sans Pro", sans-serif !important;
    }
    
    /* Specifically target markdown and text elements */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
        font-family: "Source Sans Pro", sans-serif !important;
    }
    
    /* Technical indicators table styling */
    .stDataFrame, .stDataFrame * {
        font-family: "Source Sans Pro", sans-serif !important;
    }
    
    /* Disable KaTeX rendering for certain symbols */
    .katex {
        display: none !important;
    }
    
    /* Hide the entire sidebar */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Hide sidebar toggle controls */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    </style>
    

    """, unsafe_allow_html=True)
    
    # Header
    st.title("₿ Bitcoin Analysis Dashboard")
    st.markdown("### Automated Weekly Bitcoin Chart Analysis & Probability Assessments")
    
    # Settings moved to main content area
    eastern_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern_tz)
    next_update = calculate_time_until_update(current_time)
    
    # Show timing info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"⏰ Next Update: {next_update}")
    with col2:
        if 'last_update' in st.session_state and st.session_state.last_update:
            last_update_str = st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"📊 Last Analysis: {last_update_str} ET")
        else:
            st.info("📊 Last Analysis: Not yet performed")
    with col3:
        st.info(f"🕐 Current ET: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Analysis settings
    st.subheader("⚙️ Settings")
    col1, col2 = st.columns(2)
    with col1:
        show_indicators = st.checkbox("Show Technical Indicators", value=True)
    with col2:
        show_volume = st.checkbox("Show Volume", value=True)

    # Initialize components
    data_fetcher = BitcoinDataFetcher()
    chart_generator = ChartGenerator()
    technical_analyzer = TechnicalAnalyzer()
    ai_analyzer = AIAnalyzer()
    
    # Main content area
    try:
        # Check if we need to load from file cache or update data
        cached_analysis, cache_timestamp = load_analysis_cache()
        
        # Determine if we should use cached data or fetch new data
        if cached_analysis and cache_timestamp and not should_update_analysis(cache_timestamp):
            # Use cached data
            btc_3m = pd.DataFrame(cached_analysis['btc_3m'])
            btc_1w = pd.DataFrame(cached_analysis['btc_1w'])
            indicators_3m = cached_analysis['indicators_3m']
            indicators_1w = cached_analysis['indicators_1w']
            analysis = cached_analysis['analysis']
            
            # Update session state with cached timestamp
            st.session_state.last_update = cache_timestamp
            
            cache_time_str = cache_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            st.success(f"📊 Using cached analysis data from {cache_time_str} ET")
        else:
            # Fetch new data
            with st.spinner("📈 Fetching Bitcoin data..."):
                btc_3m = data_fetcher.get_bitcoin_data(period='3mo')
                btc_1w = data_fetcher.get_bitcoin_data(period='1wk')
            
            if btc_3m.empty or btc_1w.empty:
                st.error("❌ Failed to fetch Bitcoin data. Please try again later.")
                return
            
            # Calculate technical indicators
            with st.spinner("🔍 Calculating technical indicators..."):
                indicators_3m = technical_analyzer.calculate_all_indicators(btc_3m)
                indicators_1w = technical_analyzer.calculate_all_indicators(btc_1w)
            
            # Generate AI analysis
            with st.spinner("🤖 Generating AI analysis..."):
                analysis_key = f"analysis_{datetime.now().strftime('%Y%m%d')}"
                
                if analysis_key not in st.session_state.analysis_cache:
                    current_price = btc_1w['Close'].iloc[-1]
                    analysis = ai_analyzer.generate_comprehensive_analysis(
                        data_3m=btc_3m,
                        data_1w=btc_1w,
                        indicators_3m=indicators_3m,
                        indicators_1w=indicators_1w,
                        current_price=current_price
                    )
                    st.session_state.analysis_cache[analysis_key] = analysis
                else:
                    analysis = st.session_state.analysis_cache[analysis_key]
            
            # Save to file cache
            current_time = get_eastern_time()
            
            # Convert DataFrames to serializable format
            btc_3m_data = btc_3m.reset_index().to_dict('records')
            btc_1w_data = btc_1w.reset_index().to_dict('records')
            
            # Convert indicators to serializable format
            indicators_3m_data = {}
            for key, value in indicators_3m.items():
                if isinstance(value, pd.Series):
                    indicators_3m_data[key] = value.tolist()
                else:
                    indicators_3m_data[key] = value
            
            indicators_1w_data = {}
            for key, value in indicators_1w.items():
                if isinstance(value, pd.Series):
                    indicators_1w_data[key] = value.tolist()
                else:
                    indicators_1w_data[key] = value
            
            cache_data = {
                'btc_3m': btc_3m_data,
                'btc_1w': btc_1w_data,
                'indicators_3m': indicators_3m_data,
                'indicators_1w': indicators_1w_data,
                'analysis': analysis
            }
            save_analysis_cache(cache_data)
            
            # Update session state timestamp
            st.session_state.last_update = current_time
        
        if btc_3m.empty or btc_1w.empty:
            st.error("❌ Failed to fetch Bitcoin data. Please try again later.")
            return
        
        # Current price and basic stats
        current_price = btc_1w['Close'].iloc[-1]
        price_change_24h = btc_1w['Close'].iloc[-1] - btc_1w['Close'].iloc[-2]
        price_change_pct = (price_change_24h / btc_1w['Close'].iloc[-2]) * 100
        
        # Display current price metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Current Price",
                format_currency(current_price),
                f"{price_change_pct:+.2f}%"
            )
        
        with col2:
            weekly_high = btc_1w['High'].max()
            st.metric("Weekly High", format_currency(weekly_high))
        
        with col3:
            weekly_low = btc_1w['Low'].min()
            st.metric("Weekly Low", format_currency(weekly_low))
        
        with col4:
            volume_24h = btc_1w['Volume'].iloc[-1]
            st.metric("24h Volume", f"{volume_24h:,.0f}")
        
        st.divider()
        
        # Convert DataFrames back to proper format if loaded from cache
        if isinstance(btc_3m, list):
            btc_3m = pd.DataFrame(btc_3m)
            # Convert date strings back to datetime index
            if 'Date' in btc_3m.columns:
                btc_3m['Date'] = pd.to_datetime(btc_3m['Date'])
                btc_3m.set_index('Date', inplace=True)
            elif btc_3m.index.dtype == 'object':
                btc_3m.index = pd.to_datetime(btc_3m.index)
        
        if isinstance(btc_1w, list):
            btc_1w = pd.DataFrame(btc_1w)
            # Convert date strings back to datetime index
            if 'Date' in btc_1w.columns:
                btc_1w['Date'] = pd.to_datetime(btc_1w['Date'])
                btc_1w.set_index('Date', inplace=True)
            elif btc_1w.index.dtype == 'object':
                btc_1w.index = pd.to_datetime(btc_1w.index)
        
        # Ensure proper numeric data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in btc_3m.columns:
                btc_3m[col] = pd.to_numeric(btc_3m[col], errors='coerce')
            if col in btc_1w.columns:
                btc_1w[col] = pd.to_numeric(btc_1w[col], errors='coerce')
        
        # Convert indicators back to proper format if they're dictionaries
        if isinstance(indicators_3m, dict):
            for key, value in indicators_3m.items():
                if isinstance(value, (list, str)):
                    if isinstance(value, str):
                        # Skip string representations, recalculate indicators instead
                        continue
                    indicators_3m[key] = pd.Series(value, index=btc_3m.index)
        
        if isinstance(indicators_1w, dict):
            for key, value in indicators_1w.items():
                if isinstance(value, (list, str)):
                    if isinstance(value, str):
                        # Skip string representations, recalculate indicators instead  
                        continue
                    indicators_1w[key] = pd.Series(value, index=btc_1w.index)
        
        # If indicators are corrupted, recalculate them
        if (indicators_3m and any(isinstance(v, str) for v in indicators_3m.values())) or not indicators_3m:
            with st.spinner("🔍 Recalculating 3-month technical indicators..."):
                indicators_3m = technical_analyzer.calculate_all_indicators(btc_3m)
        
        if (indicators_1w and any(isinstance(v, str) for v in indicators_1w.values())) or not indicators_1w:
            with st.spinner("🔍 Recalculating 1-week technical indicators..."):
                indicators_1w = technical_analyzer.calculate_all_indicators(btc_1w)
        

        
        # Chart Generation
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.subheader("📈 3-Month Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                fig_3m = chart_generator.create_comprehensive_chart(
                    btc_3m, 
                    indicators_3m, 
                    title="Bitcoin - 3 Month Analysis",
                    show_indicators=show_indicators,
                    show_volume=show_volume
                )
                st.plotly_chart(fig_3m, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        
        with col2:
            st.subheader("📊 1-Week Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                fig_1w = chart_generator.create_comprehensive_chart(
                    btc_1w, 
                    indicators_1w, 
                    title="Bitcoin - 1 Week Analysis",
                    show_indicators=show_indicators,
                    show_volume=show_volume
                )
                st.plotly_chart(fig_1w, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        
        st.divider()
        
        # AI Analysis Section
        st.header("🤖 AI-Powered Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display AI Analysis Results
        if analysis:
            col1, col2 = st.columns([2, 1], gap="medium")
            
            with col1:
                st.subheader("📝 Technical Analysis Summary")
                st.markdown(analysis.get('technical_summary', 'Analysis not available'))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("🎯 Friday 4PM ET Price Prediction")
                st.markdown(analysis.get('price_prediction', 'Prediction not available'))
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader("📰 Market Sentiment & Key Events")
                st.markdown(analysis.get('market_sentiment', 'Sentiment analysis not available'))
            
            with col2:
                st.subheader("📊 Probability Assessment")
                
                # Display probabilities if available
                probabilities = analysis.get('probabilities', {})
                if probabilities:
                    higher_prob = probabilities.get('higher', 0)
                    lower_prob = probabilities.get('lower', 0)
                    confidence = probabilities.get('confidence', 0)
                    
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = higher_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Higher by Friday (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Confidence metrics
                    st.metric("Higher Probability", f"{higher_prob:.1%}")
                    st.metric("Lower Probability", f"{lower_prob:.1%}")
                    st.metric("AI Confidence", f"{confidence:.1%}")
                    
                    # Confidence indicator
                    if confidence >= 0.75:
                        st.success(f"✅ High confidence analysis (≥75%)")
                    else:
                        st.warning(f"⚠️ Lower confidence analysis (<75%)")
        
        # Technical Indicators Summary Table
        st.divider()
        st.subheader("📋 Technical Indicators Summary")
        
        # Create indicators comparison table with sentiment analysis
        indicators_summary = []
        
        for timeframe, indicators, data in [("1 Week", indicators_1w, btc_1w), ("3 Month", indicators_3m, btc_3m)]:
            if indicators:
                rsi_current = indicators.get('RSI', pd.Series()).iloc[-1] if 'RSI' in indicators else None
                macd_current = indicators.get('MACD', pd.Series()).iloc[-1] if 'MACD' in indicators else None
                macd_signal = indicators.get('MACD_Signal', pd.Series()).iloc[-1] if 'MACD_Signal' in indicators else None
                
                # RSI sentiment
                rsi_sentiment = "N/A"
                if rsi_current is not None:
                    if rsi_current < 30:
                        rsi_sentiment = f"{rsi_current:.1f} - Oversold/Bullish"
                    elif rsi_current > 70:
                        rsi_sentiment = f"{rsi_current:.1f} - Overbought/Bearish"
                    else:
                        rsi_sentiment = f"{rsi_current:.1f} - Neutral"
                
                # MACD sentiment
                macd_sentiment = "N/A"
                if macd_current is not None and macd_signal is not None:
                    if macd_current > macd_signal:
                        macd_sentiment = f"{macd_current:.2f} - Bullish"
                    else:
                        macd_sentiment = f"{macd_current:.2f} - Bearish"
                
                # Bollinger Bands sentiment
                bb_upper = indicators.get('BB_Upper', pd.Series()).iloc[-1] if 'BB_Upper' in indicators else None
                bb_lower = indicators.get('BB_Lower', pd.Series()).iloc[-1] if 'BB_Lower' in indicators else None
                current_price = data['Close'].iloc[-1]
                
                bb_sentiment = "N/A"
                if bb_upper is not None and bb_lower is not None:
                    if current_price > bb_upper:
                        bb_sentiment = "Above Upper - Overbought/Bearish"
                    elif current_price < bb_lower:
                        bb_sentiment = "Below Lower - Oversold/Bullish"
                    else:
                        bb_sentiment = "Between Bands - Neutral"
                
                indicators_summary.append({
                    'Timeframe': timeframe,
                    'RSI': rsi_sentiment,
                    'MACD': macd_sentiment,
                    'Bollinger Position': bb_sentiment,
                    'EMA Trend': "Bullish" if current_price > indicators.get('EMA_20', pd.Series()).iloc[-1] else "Bearish"
                })
        
        if indicators_summary:
            df_indicators = pd.DataFrame(indicators_summary)
            st.dataframe(df_indicators, use_container_width=True)
        
        # Update timestamp
        st.session_state.last_update = current_time
        
        # Footer with last update info
        st.divider()
        st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Data source: Yahoo Finance | AI: OpenAI GPT-4o")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
    
    # Manual refresh section at the bottom
    st.divider()
    st.subheader("🔄 Manual Refresh")
    st.markdown("Use this section to manually refresh the analysis data when needed.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        refresh_password = st.text_input("Enter password to refresh:", type="password", key="refresh_pwd")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("🔄 Refresh Analysis", type="primary"):
            if refresh_password == "bitcoin2025":
                # Clear all caches including file cache
                st.session_state.data_cache = {}
                st.session_state.analysis_cache = {}
                try:
                    import os
                    cache_file = "bitcoin_analysis_cache.json"
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                except:
                    pass
                st.success("Analysis refreshed successfully!")
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")

if __name__ == "__main__":
    main()
