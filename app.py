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
from utils import format_currency, get_eastern_time, calculate_time_until_update, should_update_analysis, save_analysis_cache, load_analysis_cache, save_prediction, load_predictions_history, update_prediction_accuracy

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
    # Apply theme-based styling
    theme_mode = "dark" if st.session_state.get('dark_mode', False) else "light"
    
    # Add custom CSS for consistent fonts but preserve icons
    st.markdown(f"""
    <style>
    /* Completely hide Streamlit header to remove whitespace */
    .stApp > header {{
        display: none !important;
    }}
    
    /* Target the specific header element you found */
    .stAppHeader {{
        display: none !important;
    }}
    
    .stApp {{
        padding-top: 0rem !important;
        {'''
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
        ''' if theme_mode == 'dark' else '''
        background-color: #FFFFFF !important;
        color: #262730 !important;
        '''}
    }}
    
    /* Remove all top padding/margins from main content */
    .block-container {{
        padding-top: 1rem !important;
        margin-top: 0rem !important;
    }}
    
    /* Position dark mode button properly without header interference */
    [data-testid="column"] {{
        position: relative;
        z-index: 1000;
    }}
    
    /* Target only text elements, leave icons untouched */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stMetric, .stMetric *,
    .stDataFrame, .stDataFrame *,
    h1, h2, h3, h4, h5, h6, p, span {{
        font-family: "Source Sans Pro", sans-serif !important;
        {f'color: #FAFAFA !important;' if theme_mode == 'dark' else 'color: #262730 !important;'}
    }}
    
    /* Theme-based chart backgrounds */
    .js-plotly-plot .plotly .modebar {{
        {f'background-color: #262730 !important;' if theme_mode == 'dark' else 'background-color: #FFFFFF !important;'}
    }}
    
    /* Disable KaTeX rendering for certain symbols */
    .katex {{
        display: none !important;
    }}
    
    /* Hide the entire sidebar */
    [data-testid="stSidebar"] {{
        display: none !important;
    }}
    
    /* Hide sidebar toggle controls */
    [data-testid="collapsedControl"] {{
        display: none !important;
    }}
    
    /* Hide the entire Streamlit menu completely */
    [data-testid="stMainMenu"] {{
        display: none !important;
    }}
    
    [data-testid="stToolbar"] {{
        display: none !important;
    }}
    
    /* Position theme toggle in top right */
    [data-testid="column"]:last-child button[kind="secondary"] {{
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 999;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Theme-aware button styling */
    button[kind="secondary"] {{
        {f'''
        background-color: #262730 !important;
        color: #FAFAFA !important;
        border: 1px solid #4A4A4A !important;
        ''' if theme_mode == 'dark' else '''
        background-color: #FFFFFF !important;
        color: #262730 !important;
        border: 1px solid #D0D0D0 !important;
        '''}
    }}
    
    button[kind="secondary"]:hover {{
        {f'''
        background-color: #3A3A3A !important;
        border: 1px solid #6A6A6A !important;
        ''' if theme_mode == 'dark' else '''
        background-color: #F8F8F8 !important;
        border: 1px solid #B0B0B0 !important;
        '''}
    }}
    
    </style>
    

    """, unsafe_allow_html=True)
    
    # Custom theme toggle using Streamlit button
    col_spacer, col_theme = st.columns([6, 1])
    with col_theme:
        # Initialize theme state
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = False
        
        theme_label = "Dark" if not st.session_state.dark_mode else "Light"
        if st.button(f"Mode: {theme_label}", type="secondary", help="Toggle between light and dark theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    # Header
    st.title("₿itcoin Analysis Dashboard")
    st.markdown("### Advanced Bitcoin Chart Analysis & Probability Assessments")
    
    # Settings moved to main content area
    eastern_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern_tz)
    next_update = calculate_time_until_update(current_time)
    
    # Prediction Target Selection
    st.divider()
    st.subheader("🎯 Prediction Target Selection")
    st.markdown("Choose when you want the Bitcoin price prediction for:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Date selection (default to upcoming Friday)
        days_until_friday = (4 - current_time.weekday()) % 7
        if days_until_friday == 0:  # If today is Friday
            if current_time.hour >= 16:  # If it's Friday after 4PM
                days_until_friday = 7  # Next Friday
            # If it's Friday before 4PM, keep days_until_friday = 0 (today)
        default_date = (current_time + timedelta(days=days_until_friday)).date()
        
        selected_date = st.date_input(
            "Target Date",
            value=default_date,
            min_value=current_time.date(),
            max_value=(current_time + timedelta(days=14)).date(),
            help="Select the date for your price prediction (up to 2 weeks out)"
        )
    
    with col2:
        # Time selection (default to 4:00 PM)
        selected_time = st.time_input(
            "Target Time (ET)",
            value=datetime.strptime("16:00", "%H:%M").time(),
            help="Select the time for your price prediction in Eastern Time"
        )
    
    # Combine date and time
    target_datetime = datetime.combine(selected_date, selected_time)
    target_datetime = eastern_tz.localize(target_datetime)
    
    # Display the selected target
    target_formatted = target_datetime.strftime('%A, %B %d, %Y at %I:%M %p ET')
    st.info(f"📊 Prediction target: **{target_formatted}**")
    
    # Analyze button
    st.write("")  # Add some space
    analyze_button = st.button("🚀 **Analyze Bitcoin**", type="primary", use_container_width=True)
    
    if not analyze_button:
        # Show instructions when not analyzing
        st.markdown("""
        ### 📋 Ready to Analyze Bitcoin?
        
        **Instructions:**
        1. Select your target **date** and **time** above  
        2. Click the **"Analyze Bitcoin"** button
        3. Get comprehensive technical analysis with AI insights
        
        The analysis will include:
        - 🔍 **Multi-timeframe technical analysis** (3-month and 1-week charts)
        - 📊 **Advanced indicators** (RSI, MACD, Bollinger Bands, EMAs)
        - 🎯 **Price prediction** with probability assessment
        - 📈 **Interactive charts** with detailed visualizations
        """)
        return  # Exit early, don't run analysis

    # Initialize components (only when analyze button is pressed)
    data_fetcher = BitcoinDataFetcher()
    chart_generator = ChartGenerator()
    technical_analyzer = TechnicalAnalyzer()
    ai_analyzer = AIAnalyzer()
    
    # Main content area (only runs when analyze button is pressed)
    try:
        # Fetch fresh data when user requests analysis
        with st.spinner("📈 Fetching fresh Bitcoin data..."):
            btc_3m = data_fetcher.get_bitcoin_data(period='3mo')
            btc_1w = data_fetcher.get_bitcoin_data(period='1wk')
        
        if btc_3m.empty or btc_1w.empty:
            st.error("❌ Failed to fetch Bitcoin data. Please try again later.")
            return
        
        # Calculate technical indicators
        with st.spinner("🔍 Calculating technical indicators..."):
            indicators_3m = technical_analyzer.calculate_all_indicators(btc_3m)
            indicators_1w = technical_analyzer.calculate_all_indicators(btc_1w)
        
        # Generate fresh AI analysis every time
        with st.spinner("🤖 Generating fresh AI analysis..."):
            current_price = btc_1w['Close'].iloc[-1]
            analysis = ai_analyzer.generate_comprehensive_analysis(
                data_3m=btc_3m,
                data_1w=btc_1w,
                indicators_3m=indicators_3m,
                indicators_1w=indicators_1w,
                current_price=current_price,
                target_datetime=target_datetime
            )
        
        # Update session state timestamp
        current_time = get_eastern_time()
        st.session_state.last_update = current_time
        
        # Save prediction to history (if analysis contains prediction data)
        if analysis and isinstance(analysis, dict) and 'probabilities' in analysis:
            try:
                probabilities = analysis['probabilities']
                if isinstance(probabilities, dict):
                    prediction_data = {
                        'target_datetime': target_datetime.isoformat(),
                        'current_price': float(current_price),
                        'predicted_price': probabilities.get('predicted_price'),
                        'probability_higher': probabilities.get('higher_fraction', 0) * 100,
                        'probability_lower': probabilities.get('lower_fraction', 0) * 100,
                        'confidence_level': probabilities.get('confidence_level', 0),
                        'technical_summary': analysis.get('technical_summary', ''),
                        'prediction_reasoning': analysis.get('price_prediction', '')
                    }
                    save_prediction(prediction_data)
            except Exception as e:
                st.warning(f"Note: Could not save prediction to history: {str(e)}")
        
        # Update any past predictions with current price if their target time has passed
        update_prediction_accuracy(float(current_price))
        
        # Display fresh analysis message
        current_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        if analysis and isinstance(analysis, dict) and 'probabilities' in analysis:
            probabilities = analysis['probabilities']
            if isinstance(probabilities, dict):
                higher_prob = probabilities.get('higher_fraction', 0)
                lower_prob = probabilities.get('lower_fraction', 0)
                predicted_price = probabilities.get('predicted_price', None)
            else:
                higher_prob = lower_prob = 0
                predicted_price = None
            
            # Determine direction and probability
            if higher_prob > lower_prob:
                direction = "higher"
                probability = higher_prob
            else:
                direction = "lower" 
                probability = lower_prob
            
            # Determine recommendation based on probability and direction
            if probability >= 0.7 and direction == "higher":
                recommendation = "**Buy**"
            elif probability >= 0.7 and direction == "lower":
                recommendation = "**Sell**"
            else:
                recommendation = "**Hold**"
            
            # Create base message with prediction
            predicted_price_text = f" Predicted price: ${predicted_price:,.0f}." if predicted_price else ""
            analysis_message = f"Based on fresh analysis at {current_time_str} ET, Bitcoin has a {probability:.0%} chance of being {direction} by {target_formatted}.{predicted_price_text} Recommendation: {recommendation}"
            st.info(f"📊 {analysis_message}")
        else:
            st.success(f"📊 Fresh analysis completed at {current_time_str} ET")
        
        # Current price and basic stats
        current_price = btc_1w['Close'].iloc[-1]
        price_change_24h = btc_1w['Close'].iloc[-1] - btc_1w['Close'].iloc[-2]
        price_change_pct = (price_change_24h / btc_1w['Close'].iloc[-2]) * 100
        
        # Display current price metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Price at Last Analysis",
                format_currency(current_price)
            )
        
        with col2:
            weekly_high = btc_1w['High'].max()
            st.metric("High Last 7 Days", format_currency(weekly_high))
        
        with col3:
            weekly_low = btc_1w['Low'].min()
            st.metric("Low Last 7 Days", format_currency(weekly_low))
        
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
        

        
        # Default chart settings
        show_indicators = True
        show_volume = True
        
        # Chart Generation
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.subheader("📈 3-Month Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                display_from_3m = getattr(btc_3m, 'attrs', {}).get('display_from_index', None)
                fig_3m = chart_generator.create_comprehensive_chart(
                    btc_3m, 
                    indicators_3m, 
                    title="Bitcoin - 3 Month Analysis",
                    show_indicators=show_indicators,
                    show_volume=show_volume,
                    theme=theme_mode,
                    display_from_index=display_from_3m
                )
                st.plotly_chart(fig_3m, use_container_width=True)
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        
        with col2:
            st.subheader("📊 1-Week Bitcoin Chart")
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                display_from_1w = getattr(btc_1w, 'attrs', {}).get('display_from_index', None)
                fig_1w = chart_generator.create_comprehensive_chart(
                    btc_1w, 
                    indicators_1w, 
                    title="Bitcoin - 1 Week Analysis",
                    show_indicators=show_indicators,
                    show_volume=show_volume,
                    theme=theme_mode,
                    display_from_index=display_from_1w
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
                technical_summary = analysis.get('technical_summary', 'Analysis not available')
                # Clean and properly format the text
                if isinstance(technical_summary, str) and technical_summary.strip():
                    # Remove any problematic characters and ensure proper formatting
                    cleaned_summary = technical_summary.replace('\\n', '\n').strip()
                    
                    # Display the full content using st.write which handles long content better
                    st.write(cleaned_summary)
                else:
                    st.write("Analysis not available")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"🎯 {target_formatted} Price Prediction")
                price_prediction = analysis.get('price_prediction', 'Prediction not available')
                if isinstance(price_prediction, str) and price_prediction.strip():
                    cleaned_prediction = price_prediction.replace('\\n', '\n').strip()
                    
                    # Display the full content using st.write which handles long content better
                    st.write(cleaned_prediction)
                else:
                    st.write("Prediction not available")
            
            with col2:
                st.subheader("📊 Probability Assessment")
                
                # Display probabilities if available
                probabilities = analysis.get('probabilities', {}) if isinstance(analysis, dict) else {}
                if probabilities and isinstance(probabilities, dict):
                    higher_prob = probabilities.get('higher_fraction', 0)
                    lower_prob = probabilities.get('lower_fraction', 0)
                    confidence = probabilities.get('confidence_fraction', 0)
                    
                    # Probability gauge with theme colors
                    gauge_colors = {
                        'bgcolor': '#0E1117' if theme_mode == 'dark' else '#FFFFFF',
                        'bordercolor': '#FAFAFA' if theme_mode == 'dark' else '#262730',
                        'font_color': '#FAFAFA' if theme_mode == 'dark' else '#262730'
                    }
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = higher_prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"Higher by {target_datetime.strftime('%A %I:%M %p')} (%)", 'font': {'color': gauge_colors['font_color']}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickcolor': gauge_colors['font_color']},
                            'bar': {'color': "darkgreen"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgray"},
                                {'range': [25, 50], 'color': "gray"},
                                {'range': [50, 75], 'color': "lightgreen"},
                                {'range': [75, 100], 'color': "green"}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(
                        height=300,
                        paper_bgcolor=gauge_colors['bgcolor'],
                        plot_bgcolor=gauge_colors['bgcolor'],
                        font=dict(color=gauge_colors['font_color'])
                    )
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
            st.dataframe(df_indicators, use_container_width=True, hide_index=True)
        
        # Prediction History Section
        st.divider()
        st.subheader("📊 Prediction History")
        
        predictions = load_predictions_history()
        if predictions:
            # Show only the 10 most recent predictions
            recent_predictions = predictions[-10:]
            
            prediction_data = []
            for pred in reversed(recent_predictions):  # Show newest first
                prediction_time = pred.get('prediction_timestamp', '')
                target_time = pred.get('target_datetime', '')
                predicted_price = pred.get('predicted_price')
                current_price_at_pred = pred.get('current_price_at_prediction')
                actual_price = pred.get('actual_price')
                prob_higher = pred.get('probability_higher', 0)
                prob_lower = pred.get('probability_lower', 0)
                
                try:
                    pred_time_formatted = datetime.fromisoformat(prediction_time).strftime('%Y-%m-%d %H:%M')
                    target_time_formatted = datetime.fromisoformat(target_time).strftime('%Y-%m-%d %H:%M')
                except:
                    pred_time_formatted = prediction_time
                    target_time_formatted = target_time
                
                # Calculate accuracy if we have actual price
                accuracy_text = "Pending"
                accuracy_color = "🟡"
                if actual_price is not None and predicted_price is not None:
                    error_pct = abs(actual_price - predicted_price) / predicted_price * 100
                    if error_pct <= 5:
                        accuracy_text = f"✅ Very Good ({error_pct:.1f}% error)"
                        accuracy_color = "🟢"
                    elif error_pct <= 10:
                        accuracy_text = f"✅ Good ({error_pct:.1f}% error)"
                        accuracy_color = "🟢"
                    elif error_pct <= 20:
                        accuracy_text = f"⚠️ Fair ({error_pct:.1f}% error)"
                        accuracy_color = "🟡"
                    else:
                        accuracy_text = f"❌ Poor ({error_pct:.1f}% error)"
                        accuracy_color = "🔴"
                
                prediction_data.append({
                    'Prediction Made': pred_time_formatted,
                    'Target Time': target_time_formatted,
                    'Price at Prediction': f"${current_price_at_pred:,.0f}" if current_price_at_pred else "N/A",
                    'Predicted Price': f"${predicted_price:,.0f}" if predicted_price else "N/A",
                    'Actual Price': f"${actual_price:,.0f}" if actual_price else "Pending",
                    'Direction': f"↗️ {prob_higher:.0f}% higher / ↘️ {prob_lower:.0f}% lower",
                    'Accuracy': accuracy_text
                })
            
            if prediction_data:
                df_predictions = pd.DataFrame(prediction_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                
                # Calculate and display accuracy stats
                completed_predictions = [p for p in recent_predictions if p.get('actual_price') is not None]
                if completed_predictions:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Predictions", len(recent_predictions))
                    
                    with col2:
                        st.metric("Completed", len(completed_predictions))
                    
                    with col3:
                        good_predictions = 0
                        for pred in completed_predictions:
                            if pred.get('predicted_price') and pred.get('actual_price'):
                                error_pct = abs(pred['actual_price'] - pred['predicted_price']) / pred['predicted_price'] * 100
                                if error_pct <= 10:
                                    good_predictions += 1
                        
                        accuracy_rate = (good_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("Accuracy Rate (≤10% error)", f"{accuracy_rate:.0f}%")
            else:
                st.info("No predictions to display yet.")
        else:
            st.info("No prediction history available. Make your first prediction above!")
        
        # Update timestamp
        st.session_state.last_update = current_time
        
        # Footer with last update info
        st.divider()
        st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Data source: Yahoo Finance | AI: GPT-5 Nano")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
    

if __name__ == "__main__":
    main()
