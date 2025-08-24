import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our custom modules
from data_fetcher import BitcoinDataFetcher
from chart_generator import ChartGenerator
from technical_analysis import TechnicalAnalyzer
from ai_analysis import AIAnalyzer
from database import analysis_db
from social_media_generator import social_media_generator
from utils import format_currency, get_eastern_time, calculate_time_until_update, should_update_analysis, save_analysis_cache, load_analysis_cache

# Configure page
st.set_page_config(
    page_title="theBTCcourse.com - Bitcoin Price Prediction Tool",
    page_icon="https://www.thebtccourse.com/wp-content/uploads/2023/02/thebtccourse-favicon.png",
    layout="wide"
)

# Initialize session state
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

def get_current_domain():
    """Get the current domain - returns empty string to use relative URLs"""
    # Use relative URLs so links work on any domain (Replit, production, localhost, etc.)
    return ""

def load_stored_analysis(analysis_hash: str):
    """Load and display a stored analysis by hash - use same format as main page"""
    
    try:
        # Load the stored analysis data
        result = analysis_db.load_analysis_by_hash(analysis_hash)
        if not result:
            st.error(f"Analysis with hash '{analysis_hash}' not found.")
            st.markdown("---")
            if st.button("← Return to main page", type="secondary"):
                st.query_params.clear()
                st.rerun()
            return
        
        prediction_data, btc_3m, btc_1w, indicators_3m, indicators_1w = result
        
    except Exception as e:
        st.error(f"Error loading analysis: {str(e)}")
        st.markdown(f"Analysis with hash '{analysis_hash}' not found.")
        st.markdown("---")
        if st.button("← Return to main page", type="secondary"):
            st.query_params.clear()
            st.rerun()
        return
    
    # Add same header styling and navigation as main page
    st.markdown("""
    <style>
    /* Hide problematic Streamlit class */
    .st-emotion-cache-13892zc {
        display: none !important;
    }
    
    /* Remove default Streamlit top spacing */
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Remove any extra top margins */
    .main .block-container {
        padding-top: 0rem !important;
    }
    
    /* Ensure no top spacing on main content */
    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem !important;
    }
    
    /* Header navigation bar */
    .header-nav {
        background-color: #FFFFFF;
        border-bottom: 2px solid #F7931A;
        padding: 8px 0;
        margin: -1rem -1rem 1rem -1rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(247, 147, 26, 0.1);
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        gap: 2rem;
    }
    
    .nav-logo {
        height: 35px;
        width: auto;
        transition: opacity 0.2s ease;
    }
    
    .nav-logo:hover {
        opacity: 0.8;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .nav-link {
        color: #F7931A !important;
        text-decoration: none !important;
        font-weight: normal !important;
        font-size: 0.9rem;
        padding: 6px 12px;
        border-radius: 4px;
        transition: all 0.2s ease;
        white-space: nowrap !important;
    }
    
    .nav-link:hover {
        background-color: #F7931A;
        color: #FFFFFF !important;
        text-decoration: none !important;
        transform: translateY(-1px);
    }
    
    /* Bitcoin symbol styling - official Bitcoin orange */
    h1:first-letter {
        color: #F7931A !important;
    }
    
    /* Mobile responsive styling */
    @media (max-width: 768px) {
        .nav-container {
            padding: 0 1rem;
            gap: 1rem;
        }
        
        .nav-logo {
            height: 28px;
        }
        
        .nav-links {
            gap: 1rem;
        }
        
        .nav-link {
            font-size: 0.8rem !important;
            padding: 4px 8px !important;
        }
    }
    
    @media (max-width: 480px) {
        .nav-container {
            padding: 0 0.5rem;
            gap: 0.5rem;
        }
        
        .nav-logo {
            height: 25px;
        }
        
        .nav-links {
            gap: 0.5rem;
        }
        
        .nav-link {
            font-size: 0.75rem !important;
            padding: 3px 6px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Header navigation bar
    st.markdown("""
    <div class="header-nav">
        <div class="nav-container">
            <a href="https://www.thebtccourse.com" target="_blank">
                <img src="https://www.thebtccourse.com/wp-content/uploads/2023/02/theBTCcourse-logo.png" alt="theBTCcourse Logo" class="nav-logo">
            </a>
            <div class="nav-links">
                <a href="https://www.thebtccourse.com" target="_blank" class="nav-link">↩️ Return Home</a>
                <a href="https://www.thebtccourse.com/support-me/" target="_blank" class="nav-link">💝 Support Me!</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Title for stored analysis page
    st.title("₿itcoin Analysis Dashboard {Stored Analysis}")
    st.markdown("### Advanced Bitcoin Chart Analysis & Probability Assessments")
    
    # Add return to main page button at the top
    if st.button("← Return to Main Page", type="secondary", key="top_return_button"):
        st.query_params.clear()
        st.rerun()
    
    # Show analysis details in same format as fresh analysis
    if prediction_data:
        eastern_tz = pytz.timezone('US/Eastern')
        try:
            pred_time = datetime.fromisoformat(prediction_data.get('prediction_timestamp', ''))
            if pred_time.tzinfo is not None:
                pred_time = pred_time.astimezone(eastern_tz)
            pred_time_str = pred_time.strftime('%Y-%m-%d %H:%M:%S ET')
        except:
            pred_time_str = prediction_data.get('prediction_timestamp', 'Unknown')
        
        # Format target time for display
        try:
            target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
            if target_time.tzinfo is not None:
                target_time = target_time.astimezone(eastern_tz)
            target_formatted = target_time.strftime('%A, %B %d, %Y at %I:%M %p ET')
        except:
            target_formatted = prediction_data.get('target_datetime', 'Unknown')
        
        # Get probability values
        higher_prob = prediction_data.get('probability_higher', 0)
        lower_prob = prediction_data.get('probability_lower', 0)
        predicted_price = prediction_data.get('predicted_price')
        
        # Determine direction and probability
        if higher_prob > lower_prob:
            direction = "higher"
            probability = higher_prob
        else:
            direction = "lower"
            probability = lower_prob
        
        # Determine recommendation based on probability and direction
        if probability >= 70 and direction == "higher":
            recommendation = "**Buy**"
        elif probability >= 70 and direction == "lower":
            recommendation = "**Sell**"
        else:
            recommendation = "**HODL!**"
        
        # Create detailed analysis message similar to main page
        predicted_price_text = f" Predicted price: **${predicted_price:,.0f}**." if predicted_price else ""
        analysis_message = f"Based on historical analysis pulled {pred_time_str}, Bitcoin had a **{probability:.0f}% chance of being {direction}** by {target_formatted}.{predicted_price_text} Recommendation: {recommendation}"
        st.info(f"📊 {analysis_message}")
        
        # Show support message like main page
        st.markdown("""
        <div style="padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724; margin: 16px 0;">
            🎯 <strong>Enjoying this tool?</strong> It costs me about $0.05 per analysis and I want to keep it free, so <a href='https://www.thebtccourse.com/support-me/' target='_blank' style='color: #155724;'>showing some support</a> would be awesome!
        </div>
        """, unsafe_allow_html=True)
        
        # Display current price metrics in same format as main page
        if not btc_1w.empty:
            current_price = btc_1w['Close'].iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Price at Analysis Time",
                    f"${current_price:,.0f}"
                )
            
            with col2:
                available_data_points = min(168, len(btc_1w))
                last_7_days = btc_1w.tail(available_data_points)
                clean_high_data = last_7_days['High'].dropna()
                weekly_high = clean_high_data.max() if not clean_high_data.empty else current_price
                st.metric("High Last 7 Days", f"${weekly_high:,.0f}")
            
            with col3:
                clean_low_data = last_7_days['Low'].dropna()
                weekly_low = clean_low_data.min() if not clean_low_data.empty else current_price
                st.metric("Low Last 7 Days", f"${weekly_low:,.0f}")
        
        st.divider()
    
    # Initialize chart generator and technical analyzer
    chart_generator = ChartGenerator()
    technical_analyzer = TechnicalAnalyzer()
    
    # Default chart settings like main page
    show_indicators = True
    show_volume = True
    
    # Calculate correct display_from_index values for stored data
    # 3-month chart: show last 90 days if we have more data
    display_from_3m = None
    if len(btc_3m) > 90:
        display_from_3m = len(btc_3m) - 90
    
    # 1-week chart: show last 168 hours (7 days) if we have more data  
    display_from_1w = None
    if len(btc_1w) > 168:
        display_from_1w = len(btc_1w) - 168
    
    # Chart Generation - Same 2-column layout as main page
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
                show_volume=show_volume,
                theme="light",
                display_from_index=display_from_3m
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
                show_volume=show_volume,
                theme="light",
                display_from_index=display_from_1w
            )
            st.plotly_chart(fig_1w, use_container_width=True)
        except Exception as e:
            st.error(f"Chart error: {str(e)}")
    
    st.divider()
    
    # AI Analysis Section - Same format as main page
    st.header("🤖 AI-Powered Analysis")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display AI Analysis Results - same format as main page
    if prediction_data.get('full_ai_analysis'):
        # Parse the stored analysis 
        full_analysis = prediction_data['full_ai_analysis']
        
        # Handle different storage formats
        if isinstance(full_analysis, dict):
            # Extract clean content from dictionary
            technical_summary = full_analysis.get('technical_summary', 'Analysis not available')
            price_prediction = full_analysis.get('price_prediction', 'Prediction not available')
        elif isinstance(full_analysis, str):
            # Check if it's a string representation of a dict
            if full_analysis.startswith("{'technical_summary'"):
                # Try to parse the dictionary string
                try:
                    import ast
                    parsed_dict = ast.literal_eval(full_analysis)
                    technical_summary = parsed_dict.get('technical_summary', 'Analysis not available')
                    price_prediction = parsed_dict.get('price_prediction', 'Prediction not available')
                except:
                    # If parsing fails, treat as plain text
                    technical_summary = full_analysis
                    price_prediction = 'Prediction not available'
            else:
                # Plain text format
                technical_summary = full_analysis
                price_prediction = 'Prediction not available'
        else:
            technical_summary = 'Analysis not available'
            price_prediction = 'Prediction not available'
        
        col1, col2 = st.columns([2, 1], gap="medium")
        
        with col1:
            st.subheader("📝 Technical Analysis Summary")
            # Clean and properly format the text same as main page
            if isinstance(technical_summary, str) and technical_summary.strip():
                # Remove any problematic characters and ensure proper formatting
                cleaned_summary = technical_summary.replace('\\n', '\n').strip()
                
                # Use st.markdown but escape potential math symbols to avoid KaTeX rendering
                # Replace common math triggers while preserving markdown formatting
                escaped_summary = cleaned_summary.replace('$', '\\$').replace('_', '\\_')
                st.markdown(escaped_summary)
            else:
                st.write("Analysis not available")
            
            st.markdown("<br>", unsafe_allow_html=True)
            # Show target date if available
            target_formatted = "Target Date"
            if prediction_data.get('target_datetime'):
                try:
                    target_dt = datetime.fromisoformat(prediction_data['target_datetime'])
                    eastern_tz = pytz.timezone('US/Eastern')
                    if target_dt.tzinfo is not None:
                        target_dt = target_dt.astimezone(eastern_tz)
                    target_formatted = target_dt.strftime('%a %b %d, %Y at %I:%M %p ET')
                except:
                    target_formatted = prediction_data['target_datetime']
            
            st.subheader(f"🎯 {target_formatted} Price Prediction")
            if isinstance(price_prediction, str) and price_prediction.strip():
                cleaned_prediction = price_prediction.replace('\\n', '\n').strip()
                # Escape potential math symbols while preserving markdown formatting
                escaped_prediction = cleaned_prediction.replace('$', '\\$').replace('_', '\\_')
                st.markdown(escaped_prediction)
            else:
                st.write("Prediction not available")
        
        with col2:
            st.subheader("📊 Probability Assessment")
            
            # Display probabilities if available (stored as 0-100 integers, convert to 0-1 fractions)
            higher_prob_raw = prediction_data.get('probability_higher', 0)
            lower_prob_raw = prediction_data.get('probability_lower', 0) 
            confidence_raw = prediction_data.get('confidence_level', 0)
            
            # Convert percentages to fractions for display
            higher_prob = higher_prob_raw / 100.0 if higher_prob_raw is not None else 0
            lower_prob = lower_prob_raw / 100.0 if lower_prob_raw is not None else 0
            confidence = confidence_raw / 100.0 if confidence_raw is not None else 0
            
            # Probability gauge with theme colors - same as main page
            gauge_colors = {
                'bgcolor': '#FFFFFF',
                'bordercolor': '#262730',
                'font_color': '#262730'
            }
            
            try:
                target_dt = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
                gauge_title = f"Higher by {target_dt.strftime('%A %I:%M %p')} (%)"
            except:
                gauge_title = "Higher Probability (%)"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = higher_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': gauge_title, 'font': {'color': gauge_colors['font_color']}},
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
    
    
    # Return to main page same as main page
    st.markdown("---")
    if st.button("← Return to Main Page", type="secondary"):
        st.query_params.clear()
        st.rerun()
    
    # Add footer
    display_footer()

def display_footer():
    """Display footer with timestamp, data source, AI model, and GitHub link"""
    current_time = get_eastern_time()
    st.divider()
    st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Data source: Yahoo Finance | AI: GPT-5 | [GitHub](https://github.com/njwill/BitcoinPricePredictor)")

def main():
    # Check for analysis hash in URL parameters
    query_params = st.query_params
    if 'analysis' in query_params:
        analysis_hash = query_params['analysis']
        load_stored_analysis(analysis_hash)
        return
    
    # Add custom CSS for consistent fonts and light theme styling
    st.markdown("""
    <style>
    /* Completely hide Streamlit header to remove whitespace */
    .stApp > header {
        display: none !important;
    }
    
    /* Target the specific header element you found */
    .stAppHeader {
        display: none !important;
    }
    
    .stApp {
        padding-top: 0rem !important;
        background-color: #FFFFFF !important;
        color: #262730 !important;
    }
    
    /* Remove all top padding/margins from main content */
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0rem !important;
    }
    
    [data-testid="column"] {
        position: relative;
        z-index: 1000;
    }
    
    /* Target only text elements, leave icons untouched */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stMetric, .stMetric *,
    .stDataFrame, .stDataFrame *,
    h1, h2, h3, h4, h5, h6, p, span {
        font-family: "Source Sans Pro", sans-serif !important;
        color: #262730 !important;
    }
    
    /* Chart backgrounds */
    .js-plotly-plot .plotly .modebar {
        background-color: #FFFFFF !important;
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
    
    /* Hide the entire Streamlit menu completely */
    [data-testid="stMainMenu"] {
        display: none !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Bitcoin symbol styling - official Bitcoin orange */
    h1:first-letter {
        color: #F7931A !important;
    }
    
    /* Header navigation bar */
    .header-nav {
        background-color: #FFFFFF;
        border-bottom: 2px solid #F7931A;
        padding: 8px 0;
        margin: -1rem -1rem 1rem -1rem;
        position: sticky;
        top: 0;
        z-index: 1000;
        box-shadow: 0 2px 4px rgba(247, 147, 26, 0.1);
    }
    
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
        gap: 2rem;
    }
    
    .nav-logo {
        height: 35px;
        width: auto;
        transition: opacity 0.2s ease;
    }
    
    .nav-logo:hover {
        opacity: 0.8;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    /* Mobile responsive styling */
    @media (max-width: 768px) {
        .nav-container {
            padding: 0 1rem;
            gap: 1rem;
        }
        
        .nav-logo {
            height: 28px;
        }
        
        .nav-links {
            gap: 1rem;
        }
        
        .nav-link {
            font-size: 0.8rem !important;
            padding: 4px 8px !important;
        }
    }
    
    @media (max-width: 480px) {
        .nav-container {
            padding: 0 0.5rem;
            gap: 0.5rem;
        }
        
        .nav-logo {
            height: 25px;
        }
        
        .nav-links {
            gap: 0.5rem;
        }
        
        .nav-link {
            font-size: 0.75rem !important;
            padding: 3px 6px !important;
        }
    }
    
    .nav-link {
        color: #F7931A !important;
        text-decoration: none !important;
        font-weight: normal !important;
        font-size: 0.9rem;
        padding: 6px 12px;
        border-radius: 4px;
        transition: all 0.2s ease;
        white-space: nowrap !important;
    }
    
    .nav-link:hover {
        background-color: #F7931A;
        color: #FFFFFF !important;
        text-decoration: none !important;
        transform: translateY(-1px);
    }
    
    /* Support popup styling */
    .support-popup {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #F7931A, #FF8C00);
        color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 8px 25px rgba(247, 147, 26, 0.3);
        max-width: 350px;
        z-index: 10000;
        font-family: "Source Sans Pro", sans-serif;
        display: none;
        animation: slideInUp 0.5s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(50px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .support-popup-close {
        position: absolute;
        top: 8px;
        right: 12px;
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.2s;
    }
    
    .support-popup-close:hover {
        opacity: 1;
    }
    
    .support-popup-content {
        margin-bottom: 15px;
        font-size: 14px;
        line-height: 1.4;
    }
    
    .support-popup-link {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        color: white !important;
        text-decoration: none !important;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 500;
        transition: background 0.2s;
    }
    
    .support-popup-link:hover {
        background: rgba(255, 255, 255, 0.3);
        color: white !important;
        text-decoration: none !important;
    }
    
    /* Mobile responsive for popup */
    @media (max-width: 480px) {
        .support-popup {
            bottom: 10px;
            right: 10px;
            left: 10px;
            max-width: none;
            padding: 15px;
        }
    }
    
    </style>
    

    """, unsafe_allow_html=True)
    

    # Header navigation bar
    st.markdown("""
    <div class="header-nav">
        <div class="nav-container">
            <a href="https://www.thebtccourse.com" target="_blank">
                <img src="https://www.thebtccourse.com/wp-content/uploads/2023/02/theBTCcourse-logo.png" alt="theBTCcourse Logo" class="nav-logo">
            </a>
            <div class="nav-links">
                <a href="https://www.thebtccourse.com" target="_blank" class="nav-link">↩️ Return Home</a>
                <a href="https://www.thebtccourse.com/support-me/" target="_blank" class="nav-link">💝 Support Me!</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("₿itcoin Analysis Dashboard")
    st.markdown("### Advanced Bitcoin Chart Analysis & Probability Assessments")
    
    # Instructions section at the top
    st.markdown("""
    **Instructions:**
    1. Select your target **date** and **time** below, 3-10 days is best  
    2. Click the **"Analyze Bitcoin"** button
    3. Get comprehensive technical analysis with AI insights
    
    The analysis will include:
    - 🔍 **Multi-timeframe technical analysis** (3-month and 1-week charts)
    - 📊 **Advanced indicators** (RSI, MACD, Bollinger Bands, EMAs)
    - 🎯 **Price prediction** with probability assessment
    - 📈 **Interactive charts** with detailed visualizations
    """)
    
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
    
    # Check if target datetime is in the past
    is_future_datetime = target_datetime > current_time
    
    # Show validation message if target is in the past
    if not is_future_datetime:
        st.error("⚠️ Target date and time must be in the future. Please select a later time.")
    
    # Analyze button (directly under target selection)
    st.write("")  # Add some space
    st.markdown("*By clicking \"Analyze Bitcoin\" you're agreeing this is not financial advice, pure entertainment purposes only*")
    analyze_button = st.button("🚀 **Analyze Bitcoin**", type="primary", use_container_width=True, disabled=not is_future_datetime)
    
    
    # Show Prediction History ONLY on front page (when not analyzing)
    if not analyze_button:
        # Prediction History Section (visible on front page only)
        st.divider()
        
        # Automatically update expired predictions from database with historical prices
        try:
            analysis_db.update_analysis_accuracy()  # Uses historical prices automatically
        except:
            pass  # Continue even if we can't update accuracy
        
        predictions = analysis_db.get_recent_analyses(limit=1000)  # Get all predictions
        
        if predictions:
            # Pagination setup
            predictions_per_page = 50
            total_predictions = len(predictions)
            total_pages = max(1, (total_predictions + predictions_per_page - 1) // predictions_per_page)
            
            # Initialize page number in session state
            if 'prediction_page' not in st.session_state:
                st.session_state.prediction_page = 1
            
            # Page navigation
            if total_pages > 1:
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button("⬅️ Previous", disabled=(st.session_state.prediction_page <= 1)):
                        st.session_state.prediction_page = max(1, st.session_state.prediction_page - 1)
                        st.rerun()
                
                with col2:
                    if st.button("⏮️ First", disabled=(st.session_state.prediction_page <= 1)):
                        st.session_state.prediction_page = 1
                        st.rerun()
                
                with col3:
                    st.write(f"**Page {st.session_state.prediction_page} of {total_pages}**")
                
                with col4:
                    if st.button("⏭️ Last", disabled=(st.session_state.prediction_page >= total_pages)):
                        st.session_state.prediction_page = total_pages
                        st.rerun()
                
                with col5:
                    if st.button("Next ➡️", disabled=(st.session_state.prediction_page >= total_pages)):
                        st.session_state.prediction_page = min(total_pages, st.session_state.prediction_page + 1)
                        st.rerun()
            
            # Calculate which predictions to show
            start_idx = (st.session_state.prediction_page - 1) * predictions_per_page
            
            # Get predictions for current page from database
            try:
                predictions = analysis_db.get_recent_analyses(limit=predictions_per_page, offset=start_idx)
            except Exception as e:
                st.error(f"Error loading predictions: {str(e)}")
                predictions = []
            
            prediction_data = []
            for pred in predictions:
                prediction_time = pred.get('prediction_timestamp', '')
                target_time = pred.get('target_datetime', '')
                predicted_price = pred.get('predicted_price')
                current_price_at_pred = pred.get('current_price_at_prediction')
                actual_price = pred.get('actual_price')
                prob_higher = pred.get('probability_higher', 0)
                prob_lower = pred.get('probability_lower', 0)
                
                try:
                    # Convert to Eastern Time for display
                    eastern_tz = pytz.timezone('US/Eastern')
                    pred_dt = datetime.fromisoformat(prediction_time)
                    target_dt = datetime.fromisoformat(target_time)
                    
                    # If timezone-aware, convert to Eastern; if naive, assume it's already Eastern
                    if pred_dt.tzinfo is not None:
                        pred_dt = pred_dt.astimezone(eastern_tz)
                    if target_dt.tzinfo is not None:
                        target_dt = target_dt.astimezone(eastern_tz)
                    
                    pred_time_formatted = pred_dt.strftime('%Y-%m-%d %H:%M ET')
                    target_time_formatted = target_dt.strftime('%Y-%m-%d %H:%M ET')
                except:
                    pred_time_formatted = prediction_time
                    target_time_formatted = target_time
                
                # Calculate accuracy if we have actual price
                accuracy_text = "Pending"
                if actual_price is not None and predicted_price is not None:
                    error_pct = abs(actual_price - predicted_price) / predicted_price * 100
                    if error_pct <= 1.5:
                        accuracy_text = f"✅ Very Good ({error_pct:.1f}% error)"
                    elif error_pct <= 3.0:
                        accuracy_text = f"✅ Good ({error_pct:.1f}% error)"
                    elif error_pct <= 5.0:
                        accuracy_text = f"⚠️ Fair ({error_pct:.1f}% error)"
                    else:
                        accuracy_text = f"❌ Poor ({error_pct:.1f}% error)"
                
                # Add hash link for complete analysis view
                analysis_hash = pred.get('analysis_hash', '')
                if analysis_hash:
                    view_link = f"?analysis={analysis_hash}"
                else:
                    view_link = ""
                
                # Calculate direction confidence (the higher probability)
                direction_confidence = max(prob_higher, prob_lower)
                
                # Check if direction was correct
                direction_correct = "Pending"
                if actual_price is not None and current_price_at_pred is not None and predicted_price is not None:
                    predicted_direction = "up" if predicted_price > current_price_at_pred else "down"
                    actual_direction = "up" if actual_price > current_price_at_pred else "down"
                    if predicted_direction == actual_direction:
                        direction_correct = "✅ Correct"
                    else:
                        direction_correct = "❌ Wrong"
                
                prediction_data.append({
                    'Prediction Made': pred_time_formatted,
                    'Target Time': target_time_formatted,
                    'Price at Prediction': f"${current_price_at_pred:,.0f}" if current_price_at_pred else "N/A",
                    'Predicted Price': f"${predicted_price:,.0f}" if predicted_price else "N/A",
                    'Actual Price': f"${actual_price:,.0f}" if actual_price else "Pending",
                    'Accuracy': accuracy_text,
                    'Direction': f"↗️ {prob_higher:.0f}% higher / ↘️ {prob_lower:.0f}% lower",
                    'Direction Confidence %': f"{direction_confidence:.0f}%",
                    'Direction Correct': direction_correct,
                    'Full Analysis': view_link
                })
            
            if prediction_data:
                # Prediction Dashboard Analytics (above the table)
                completed_predictions = [p for p in predictions if p.get('actual_price') is not None]
                
                # Filter completed predictions to only include 3-10 day predictions for dashboard metrics
                dashboard_predictions = []
                for pred in completed_predictions:
                    try:
                        # Parse timestamps
                        pred_timestamp = datetime.fromisoformat(pred.get('prediction_timestamp', ''))
                        target_timestamp = datetime.fromisoformat(pred.get('target_datetime', ''))
                        
                        # Calculate days between prediction and target
                        if pred_timestamp.tzinfo is None:
                            pred_timestamp = eastern_tz.localize(pred_timestamp)
                        if target_timestamp.tzinfo is None:
                            target_timestamp = eastern_tz.localize(target_timestamp)
                        
                        days_ahead = (target_timestamp - pred_timestamp).days
                        
                        # Only include predictions made 3-10 days in advance
                        if 3 <= days_ahead <= 10:
                            dashboard_predictions.append(pred)
                    except:
                        # Skip predictions with invalid timestamps
                        continue
                
                # Always show dashboard (even if no eligible predictions)
                st.subheader("📈 Prediction Performance Dashboard")
                st.caption("📅 Showing metrics for predictions made 3-10 days in advance only")
                
                # Calculate comprehensive metrics
                errors = []
                direction_correct = 0
                very_good_predictions = 0
                good_predictions = 0
                fair_predictions = 0
                poor_predictions = 0
                
                for pred in dashboard_predictions:
                        if pred.get('predicted_price') and pred.get('actual_price'):
                            predicted = pred['predicted_price']
                            actual = pred['actual_price']
                            current_at_pred = pred.get('current_price_at_prediction', predicted)
                            
                            # Calculate error percentage
                            error_pct = abs(actual - predicted) / predicted * 100
                            errors.append(error_pct)
                            
                            # Check direction accuracy
                            predicted_direction = "up" if predicted > current_at_pred else "down"
                            actual_direction = "up" if actual > current_at_pred else "down"
                            if predicted_direction == actual_direction:
                                direction_correct += 1
                            
                            # Categorize accuracy
                            if error_pct <= 1.5:
                                very_good_predictions += 1
                            elif error_pct <= 3.0:
                                good_predictions += 1
                            elif error_pct <= 5.0:
                                fair_predictions += 1
                            else:
                                poor_predictions += 1
                    
                # Main dashboard metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                
                with col2:
                    st.metric("Eligible (3-10 days)", len(dashboard_predictions))
                
                with col3:
                    accuracy_rate = ((very_good_predictions + good_predictions) / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("Price Accuracy Rate (≤3%)", f"{accuracy_rate:.0f}%")
                
                with col4:
                    avg_error = sum(errors) / len(errors) if errors else 0
                    st.metric("Avg Error", f"{avg_error:.1f}%")
                
                with col5:
                    direction_accuracy = (direction_correct / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("Direction Accuracy", f"{direction_accuracy:.0f}%")
                    
                # Detailed accuracy breakdown
                st.divider()
                st.subheader("🎯 Price Accuracy Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    very_good_rate = (very_good_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("✅ Very Good (≤1.5%)", f"{very_good_rate:.0f}%", help=f"{very_good_predictions} predictions")
                
                with col2:
                    good_rate = (good_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("✅ Good (1.5-3%)", f"{good_rate:.0f}%", help=f"{good_predictions} predictions")
                
                with col3:
                    fair_rate = (fair_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("⚠️ Fair (3-5%)", f"{fair_rate:.0f}%", help=f"{fair_predictions} predictions")
                
                with col4:
                    poor_rate = (poor_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("❌ Poor (>5%)", f"{poor_rate:.0f}%", help=f"{poor_predictions} predictions")
                
                st.divider()
                st.subheader("📋 Detailed Prediction History")
                
                df_predictions = pd.DataFrame(prediction_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True, 
                           column_config={
                               "Full Analysis": st.column_config.LinkColumn(
                                   "Full Analysis",
                                   help="Click to view the complete analysis with charts and indicators",
                                   validate="^https://.*",
                                   max_chars=100,
                                   display_text="📊 View Full Analysis"
                               )
                           })
                
                if completed_predictions:
                    st.caption(f"Showing {len(prediction_data)} of {total_predictions} predictions")
            else:
                st.info("No predictions to display on this page.")
        else:
            st.info("No prediction history available. Make your first prediction above!")
        
        # Don't return here - let footer display at end
    else:
        # If analyze button was clicked, proceed with analysis (prediction history will only show at bottom)

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
            with st.spinner("🤖 Generating fresh AI analysis... usually takes a couple minutes..."):
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
            
            # Format target for later use
            target_formatted = target_datetime.strftime('%A, %B %d, %Y at %I:%M %p ET')
            
            # Save prediction to history (if analysis contains prediction data)
            if analysis and isinstance(analysis, dict) and 'probabilities' in analysis:
                try:
                    probabilities = analysis['probabilities']
                    if isinstance(probabilities, dict):
                        prediction_data = {
                        'prediction_timestamp': get_eastern_time().isoformat(),
                        'target_datetime': target_datetime.isoformat(),
                        'current_price': float(current_price),
                        'predicted_price': probabilities.get('predicted_price'),
                        'probability_higher': probabilities.get('higher_fraction', 0) * 100,
                        'probability_lower': probabilities.get('lower_fraction', 0) * 100,
                        'confidence_level': probabilities.get('confidence_pct', 0),
                        'technical_summary': analysis.get('technical_summary', ''),
                        'prediction_reasoning': analysis.get('price_prediction', '')
                    }
                    
                    # Save complete analysis to database and get hash
                    full_ai_text = str(analysis)  # Convert entire analysis to string
                    analysis_hash = analysis_db.save_complete_analysis(
                        prediction_data, btc_3m, btc_1w, indicators_3m, indicators_1w, full_ai_text
                    )
                    
                    # Generate social media tweet text for sharing
                    try:
                        domain = get_current_domain()
                        tweet_text = social_media_generator.generate_shareable_text(
                            prediction_data, analysis_hash, domain
                        )
                        st.session_state.social_media_text = tweet_text
                    except Exception as e:
                        print(f"Error generating social media text: {e}")
                        st.session_state.social_media_text = None
                    
                    # Analysis already saved to database above
                    
                        # Store hash in session state for later display
                        st.session_state.analysis_hash = analysis_hash
                except Exception as e:
                    st.warning(f"Note: Could not save prediction to history: {str(e)}")
            
            # Update any past predictions with current price if their target time has passed
            try:
                analysis_db.update_analysis_accuracy()  # Uses historical prices automatically
            except:
                pass  # Continue even if we can't update accuracy
            
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
                recommendation = "**HODL!**"
            
            # Create base message with prediction
            predicted_price_text = f" Predicted price: **${predicted_price:,.0f}**." if predicted_price else ""
            analysis_message = f"Based on fresh analysis at {current_time_str} ET, Bitcoin has a **{probability:.0%} chance of being {direction}** by {target_formatted}.{predicted_price_text} Recommendation: {recommendation}"
            st.info(f"📊 {analysis_message}")
        else:
            st.success(f"📊 Fresh analysis completed at {current_time_str} ET")
        
        # Show support message immediately after analysis
        st.markdown("""
        <div style="padding: 12px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px; color: #155724; margin: 16px 0;">
            🎯 <strong>Enjoying this tool?</strong> It costs me about $0.05 per analysis and I want to keep it free, so <a href='https://www.thebtccourse.com/support-me/' target='_blank' style='color: #155724;'>showing some support</a> would be awesome!
        </div>
        """, unsafe_allow_html=True)
        
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
            # Get only the last 7 days for accurate weekly high/low
            # Simply take the last 168 hours (7 days * 24 hours) of actual data
            # Handle potential data gaps by taking available data in last 7 days
            available_data_points = min(168, len(btc_1w))  # Don't exceed available data
            last_7_days = btc_1w.tail(available_data_points)
            
            # Clean data to remove any NaN values that might affect calculations
            clean_high_data = last_7_days['High'].dropna()
            weekly_high = clean_high_data.max() if not clean_high_data.empty else current_price
            st.metric("High Last 7 Days", format_currency(weekly_high))
        
        with col3:
            clean_low_data = last_7_days['Low'].dropna()
            weekly_low = clean_low_data.min() if not clean_low_data.empty else current_price
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
                    theme="light",
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
                    theme="light",
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
                        'bgcolor': '#FFFFFF',
                        'bordercolor': '#262730',
                        'font_color': '#262730'
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
        
        
        # Display share link if analysis was saved
        if 'analysis_hash' in st.session_state and st.session_state.analysis_hash:
            st.divider()
            st.subheader("🔗 Save & Share This Analysis")
            domain = get_current_domain()
            full_share_url = f"{domain}?analysis={st.session_state.analysis_hash}"
            st.success("Analysis saved! Share this link to recall the complete analysis:")
            st.markdown(f'<a href="{full_share_url}" style="color: #FF6B35; text-decoration: none;">📋 Open this analysis link</a>', unsafe_allow_html=True)
            
            # Display social media tweet text if generated
            if 'social_media_text' in st.session_state and st.session_state.social_media_text:
                st.markdown("### 🐦 Share This Analysis")
                try:
                    # Display the tweet text in a code block for easy copying
                    st.code(st.session_state.social_media_text, language=None)
                    
                    # Provide copy button functionality
                    st.success("📋 Copy the text above to share on X or other social media!")
                        
                except Exception as e:
                    st.warning(f"Could not display social media text: {e}")
        
        # Update timestamp
        st.session_state.last_update = current_time
        
        # Prediction History Section (moved to bottom after analysis)
        st.divider()
        
        # Get predictions from database and automatically update expired ones
        try:
            # Always check and update expired predictions when viewing the page
            # We'll fetch fresh Bitcoin data for accuracy updates
            analysis_db.update_analysis_accuracy()  # Now fetches historical prices automatically
            
            # Get total count for pagination
            total_predictions = analysis_db.get_total_analyses_count()
        except Exception as e:
            st.warning(f"Database connection issue: {str(e)}")
            total_predictions = 0
        
        if total_predictions > 0:
            
            # Pagination setup
            predictions_per_page = 50
            total_pages = max(1, (total_predictions + predictions_per_page - 1) // predictions_per_page)
            
            # Initialize page number in session state
            if 'prediction_page' not in st.session_state:
                st.session_state.prediction_page = 1
            
            # Page navigation
            if total_pages > 1:
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button("⬅️ Previous", disabled=(st.session_state.prediction_page <= 1)):
                        st.session_state.prediction_page = max(1, st.session_state.prediction_page - 1)
                        st.rerun()
                
                with col2:
                    if st.button("⏮️ First", disabled=(st.session_state.prediction_page <= 1)):
                        st.session_state.prediction_page = 1
                        st.rerun()
                
                with col3:
                    st.write(f"**Page {st.session_state.prediction_page} of {total_pages}**")
                
                with col4:
                    if st.button("⏭️ Last", disabled=(st.session_state.prediction_page >= total_pages)):
                        st.session_state.prediction_page = total_pages
                        st.rerun()
                
                with col5:
                    if st.button("Next ➡️", disabled=(st.session_state.prediction_page >= total_pages)):
                        st.session_state.prediction_page = min(total_pages, st.session_state.prediction_page + 1)
                        st.rerun()
            
            # Calculate which predictions to show
            start_idx = (st.session_state.prediction_page - 1) * predictions_per_page
            
            # Get predictions for current page from database
            try:
                predictions = analysis_db.get_recent_analyses(limit=predictions_per_page, offset=start_idx)
            except Exception as e:
                st.error(f"Error loading predictions: {str(e)}")
                predictions = []
            
            prediction_data = []
            for pred in predictions:
                prediction_time = pred.get('prediction_timestamp', '')
                target_time = pred.get('target_datetime', '')
                predicted_price = pred.get('predicted_price')
                current_price_at_pred = pred.get('current_price_at_prediction')
                actual_price = pred.get('actual_price')
                prob_higher = pred.get('probability_higher', 0)
                prob_lower = pred.get('probability_lower', 0)
                
                try:
                    # Convert to Eastern Time for display
                    eastern_tz = pytz.timezone('US/Eastern')
                    pred_dt = datetime.fromisoformat(prediction_time)
                    target_dt = datetime.fromisoformat(target_time)
                    
                    # If timezone-aware, convert to Eastern; if naive, assume it's already Eastern
                    if pred_dt.tzinfo is not None:
                        pred_dt = pred_dt.astimezone(eastern_tz)
                    if target_dt.tzinfo is not None:
                        target_dt = target_dt.astimezone(eastern_tz)
                    
                    pred_time_formatted = pred_dt.strftime('%Y-%m-%d %H:%M ET')
                    target_time_formatted = target_dt.strftime('%Y-%m-%d %H:%M ET')
                except:
                    pred_time_formatted = prediction_time
                    target_time_formatted = target_time
                
                # Calculate accuracy if we have actual price
                accuracy_text = "Pending"
                if actual_price is not None and predicted_price is not None:
                    error_pct = abs(actual_price - predicted_price) / predicted_price * 100
                    if error_pct <= 1.5:
                        accuracy_text = f"✅ Very Good ({error_pct:.1f}% error)"
                    elif error_pct <= 3.0:
                        accuracy_text = f"✅ Good ({error_pct:.1f}% error)"
                    elif error_pct <= 5.0:
                        accuracy_text = f"⚠️ Fair ({error_pct:.1f}% error)"
                    else:
                        accuracy_text = f"❌ Poor ({error_pct:.1f}% error)"
                
                # Add hash link for complete analysis view
                analysis_hash = pred.get('analysis_hash', '')
                if analysis_hash:
                    view_link = f"?analysis={analysis_hash}"
                else:
                    view_link = ""
                
                # Calculate direction confidence (the higher probability)
                direction_confidence = max(prob_higher, prob_lower)
                
                # Check if direction was correct
                direction_correct = "Pending"
                if actual_price is not None and current_price_at_pred is not None and predicted_price is not None:
                    predicted_direction = "up" if predicted_price > current_price_at_pred else "down"
                    actual_direction = "up" if actual_price > current_price_at_pred else "down"
                    if predicted_direction == actual_direction:
                        direction_correct = "✅ Correct"
                    else:
                        direction_correct = "❌ Wrong"
                
                prediction_data.append({
                    'Prediction Made': pred_time_formatted,
                    'Target Time': target_time_formatted,
                    'Price at Prediction': f"${current_price_at_pred:,.0f}" if current_price_at_pred else "N/A",
                    'Predicted Price': f"${predicted_price:,.0f}" if predicted_price else "N/A",
                    'Actual Price': f"${actual_price:,.0f}" if actual_price else "Pending",
                    'Accuracy': accuracy_text,
                    'Direction': f"↗️ {prob_higher:.0f}% higher / ↘️ {prob_lower:.0f}% lower",
                    'Direction Confidence %': f"{direction_confidence:.0f}%",
                    'Direction Correct': direction_correct,
                    'Full Analysis': view_link
                })
            
            if prediction_data:
                # Prediction Dashboard Analytics (above the table)
                completed_predictions = [p for p in predictions if p.get('actual_price') is not None]
                
                # Filter completed predictions to only include 3-10 day predictions for dashboard metrics
                dashboard_predictions = []
                for pred in completed_predictions:
                    try:
                        # Parse timestamps
                        pred_timestamp = datetime.fromisoformat(pred.get('prediction_timestamp', ''))
                        target_timestamp = datetime.fromisoformat(pred.get('target_datetime', ''))
                        
                        # Calculate days between prediction and target
                        if pred_timestamp.tzinfo is None:
                            pred_timestamp = eastern_tz.localize(pred_timestamp)
                        if target_timestamp.tzinfo is None:
                            target_timestamp = eastern_tz.localize(target_timestamp)
                        
                        days_ahead = (target_timestamp - pred_timestamp).days
                        
                        # Only include predictions made 3-10 days in advance
                        if 3 <= days_ahead <= 10:
                            dashboard_predictions.append(pred)
                    except:
                        # Skip predictions with invalid timestamps
                        continue
                
                # Always show dashboard (even if no eligible predictions)
                st.subheader("📈 Prediction Performance Dashboard")
                st.caption("📅 Showing metrics for predictions made 3-10 days in advance only")
                
                # Calculate comprehensive metrics
                errors = []
                direction_correct = 0
                very_good_predictions = 0
                good_predictions = 0
                fair_predictions = 0
                poor_predictions = 0
                
                for pred in dashboard_predictions:
                        if pred.get('predicted_price') and pred.get('actual_price'):
                            predicted = pred['predicted_price']
                            actual = pred['actual_price']
                            current_at_pred = pred.get('current_price_at_prediction', predicted)
                            
                            # Calculate error percentage
                            error_pct = abs(actual - predicted) / predicted * 100
                            errors.append(error_pct)
                            
                            # Check direction accuracy
                            predicted_direction = "up" if predicted > current_at_pred else "down"
                            actual_direction = "up" if actual > current_at_pred else "down"
                            if predicted_direction == actual_direction:
                                direction_correct += 1
                            
                            # Categorize accuracy
                            if error_pct <= 1.5:
                                very_good_predictions += 1
                            elif error_pct <= 3.0:
                                good_predictions += 1
                            elif error_pct <= 5.0:
                                fair_predictions += 1
                            else:
                                poor_predictions += 1
                    
                # Main dashboard metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Total Predictions", len(predictions))
                
                with col2:
                    st.metric("Eligible (3-10 days)", len(dashboard_predictions))
                
                with col3:
                    accuracy_rate = ((very_good_predictions + good_predictions) / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("Price Accuracy Rate (≤3%)", f"{accuracy_rate:.0f}%")
                
                with col4:
                    avg_error = sum(errors) / len(errors) if errors else 0
                    st.metric("Avg Error", f"{avg_error:.1f}%")
                
                with col5:
                    direction_accuracy = (direction_correct / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("Direction Accuracy", f"{direction_accuracy:.0f}%")
                    
                # Detailed accuracy breakdown
                st.divider()
                st.subheader("🎯 Price Accuracy Breakdown")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    very_good_rate = (very_good_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("✅ Very Good (≤1.5%)", f"{very_good_rate:.0f}%", help=f"{very_good_predictions} predictions")
                
                with col2:
                    good_rate = (good_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("✅ Good (1.5-3%)", f"{good_rate:.0f}%", help=f"{good_predictions} predictions")
                
                with col3:
                    fair_rate = (fair_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("⚠️ Fair (3-5%)", f"{fair_rate:.0f}%", help=f"{fair_predictions} predictions")
                
                with col4:
                    poor_rate = (poor_predictions / len(dashboard_predictions)) * 100 if dashboard_predictions else 0
                    st.metric("❌ Poor (>5%)", f"{poor_rate:.0f}%", help=f"{poor_predictions} predictions")
                
                st.divider()
                st.subheader("📋 Detailed Prediction History")
                
                df_predictions = pd.DataFrame(prediction_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True, 
                           column_config={
                               "Full Analysis": st.column_config.LinkColumn(
                                   "Full Analysis",
                                   help="Click to view the complete analysis with charts and indicators",
                                   validate="^https://.*",
                                   max_chars=100,
                                   display_text="📊 View Full Analysis"
                               )
                           })
                
                if completed_predictions:
                    st.caption(f"Showing {len(prediction_data)} of {total_predictions} predictions")
            else:
                st.info("No predictions to display on this page.")
        
        except Exception as e:
            st.error(f"❌ An error occurred during analysis: {str(e)}")
            st.exception(e)
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
    
    # Always show footer at the very end, regardless of any logic above
    display_footer()

if __name__ == "__main__":
    main()
