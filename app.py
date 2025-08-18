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
from utils import format_currency, get_eastern_time, calculate_time_until_update

# Configure page
st.set_page_config(
    page_title="Bitcoin Analysis Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
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
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("₿ Bitcoin Analysis Dashboard")
    st.markdown("### Automated Weekly Bitcoin Chart Analysis & Probability Assessments")
    
    # Sidebar for controls and info
    with st.sidebar:
        st.header("📊 Analysis Controls")
        
        # Password protected refresh button
        st.subheader("🔄 Manual Refresh")
        refresh_password = st.text_input("Enter password to refresh:", type="password", key="refresh_pwd")
        if st.button("🔄 Refresh Analysis", type="primary"):
            if refresh_password == "bitcoin2025":
                st.session_state.data_cache = {}
                st.session_state.analysis_cache = {}
                st.success("Analysis refreshed successfully!")
                st.rerun()
            else:
                st.error("Incorrect password. Access denied.")
        
        # Show next update time
        eastern_tz = pytz.timezone('US/Eastern')
        current_time = datetime.now(eastern_tz)
        next_update = calculate_time_until_update(current_time)
        
        st.info(f"⏰ Next Update: {next_update}")
        
        # Show last analysis time
        if 'last_update' in st.session_state and st.session_state.last_update:
            last_update_str = st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')
            st.info(f"📊 Last Analysis: {last_update_str} ET")
        else:
            st.info("📊 Last Analysis: Not yet performed")
        
        # Show current time
        st.write(f"**Current ET:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Analysis settings
        st.subheader("⚙️ Settings")
        show_indicators = st.checkbox("Show Technical Indicators", value=True)
        show_volume = st.checkbox("Show Volume", value=True)

    # Initialize components
    data_fetcher = BitcoinDataFetcher()
    chart_generator = ChartGenerator()
    technical_analyzer = TechnicalAnalyzer()
    ai_analyzer = AIAnalyzer()
    
    # Main content area
    try:
        # Data fetching with caching
        with st.spinner("📈 Fetching Bitcoin data..."):
            if 'btc_3m' not in st.session_state.data_cache:
                btc_3m = data_fetcher.get_bitcoin_data(period='3mo')
                btc_1w = data_fetcher.get_bitcoin_data(period='1wk')
                st.session_state.data_cache['btc_3m'] = btc_3m
                st.session_state.data_cache['btc_1w'] = btc_1w
            else:
                btc_3m = st.session_state.data_cache['btc_3m']
                btc_1w = st.session_state.data_cache['btc_1w']
        
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
        
        # Technical Analysis
        with st.spinner("🔍 Calculating technical indicators..."):
            indicators_3m = technical_analyzer.calculate_all_indicators(btc_3m)
            indicators_1w = technical_analyzer.calculate_all_indicators(btc_1w)
        
        # Chart Generation
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.subheader("📈 3-Month Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            fig_3m = chart_generator.create_comprehensive_chart(
                btc_3m, 
                indicators_3m, 
                title="Bitcoin - 3 Month Analysis",
                show_indicators=show_indicators,
                show_volume=show_volume
            )
            st.plotly_chart(fig_3m, use_container_width=True)
        
        with col2:
            st.subheader("📊 1-Week Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            fig_1w = chart_generator.create_comprehensive_chart(
                btc_1w, 
                indicators_1w, 
                title="Bitcoin - 1 Week Analysis",
                show_indicators=show_indicators,
                show_volume=show_volume
            )
            st.plotly_chart(fig_1w, use_container_width=True)
        
        st.divider()
        
        # AI Analysis Section
        st.header("🤖 AI-Powered Analysis")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Generate AI analysis with caching
        analysis_key = f"analysis_{current_time.strftime('%Y-%m-%d')}"
        
        if analysis_key not in st.session_state.analysis_cache:
            with st.spinner("🧠 Generating AI analysis..."):
                try:
                    analysis = ai_analyzer.generate_comprehensive_analysis(
                        btc_3m, btc_1w, indicators_3m, indicators_1w, current_price
                    )
                    st.session_state.analysis_cache[analysis_key] = analysis
                except Exception as e:
                    st.error(f"❌ AI Analysis failed: {str(e)}")
                    return
        else:
            analysis = st.session_state.analysis_cache[analysis_key]
        
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

if __name__ == "__main__":
    main()
