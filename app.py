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
    
    # Analyze button (directly under target selection)
    st.write("")  # Add some space
    st.markdown("*By clicking \"Analyze Bitcoin\" you're agreeing this is not financial advice, pure entertainment purposes only*")
    analyze_button = st.button("🚀 **Analyze Bitcoin**", type="primary", use_container_width=True)
    
    # Check if we should show support message after analysis
    if 'analysis_completed_time' in st.session_state and st.session_state.analysis_completed_time:
        import time
        time_since_analysis = time.time() - st.session_state.analysis_completed_time
        if time_since_analysis >= 7 and time_since_analysis <= 60:  # Show for up to 60 seconds
            st.info("🎯 **Enjoying this tool?** It costs me about $0.05 per analysis and I want to keep it free, so [showing some support](https://www.thebtccourse.com/support-me/) would be awesome!")
            # Clear the flag after showing once
            if st.button("✕ Close", key="close_support"):
                st.session_state.analysis_completed_time = None
                st.rerun()
    
    # Show Prediction History ONLY on front page (when not analyzing)
    if not analyze_button:
        # Prediction History Section (visible on front page only)
        st.divider()
        
        predictions = load_predictions_history()
        if predictions:
            # Update any past predictions with current price if their target time has passed
            try:
                # Get current Bitcoin price for accuracy updates
                data_fetcher = BitcoinDataFetcher()
                btc_data = data_fetcher.get_bitcoin_data(period='1d')
                if not btc_data.empty:
                    current_btc_price = btc_data['Close'].iloc[-1]
                    update_prediction_accuracy(float(current_btc_price))
                    # Reload predictions after potential updates
                    predictions = load_predictions_history()
            except:
                pass  # Continue even if we can't update accuracy
            
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
            end_idx = min(start_idx + predictions_per_page, total_predictions)
            
            # Get predictions for current page (newest first)
            page_predictions = list(reversed(predictions))[start_idx:end_idx]
            
            prediction_data = []
            for pred in page_predictions:
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
                # Prediction Dashboard Analytics (above the table)
                completed_predictions = [p for p in predictions if p.get('actual_price') is not None]
                if completed_predictions:
                    st.subheader("📈 Prediction Performance Dashboard")
                    
                    # Calculate comprehensive metrics
                    errors = []
                    direction_correct = 0
                    very_good_predictions = 0
                    good_predictions = 0
                    fair_predictions = 0
                    poor_predictions = 0
                    
                    for pred in completed_predictions:
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
                        st.metric("Completed", len(completed_predictions))
                    
                    with col3:
                        accuracy_rate = ((very_good_predictions + good_predictions) / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("Accuracy Rate (≤3%)", f"{accuracy_rate:.0f}%")
                    
                    with col4:
                        direction_accuracy = (direction_correct / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("Direction Accuracy", f"{direction_accuracy:.0f}%")
                    
                    with col5:
                        avg_error = sum(errors) / len(errors) if errors else 0
                        st.metric("Avg Error", f"{avg_error:.1f}%")
                    
                    # Detailed accuracy breakdown
                    st.divider()
                    st.subheader("🎯 Accuracy Breakdown")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        very_good_rate = (very_good_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("✅ Very Good (≤1.5%)", f"{very_good_rate:.0f}%", help=f"{very_good_predictions} predictions")
                    
                    with col2:
                        good_rate = (good_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("✅ Good (1.5-3%)", f"{good_rate:.0f}%", help=f"{good_predictions} predictions")
                    
                    with col3:
                        fair_rate = (fair_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("⚠️ Fair (3-5%)", f"{fair_rate:.0f}%", help=f"{fair_predictions} predictions")
                    
                    with col4:
                        poor_rate = (poor_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("❌ Poor (>5%)", f"{poor_rate:.0f}%", help=f"{poor_predictions} predictions")
                    

                    
                    st.divider()
                    st.subheader("📋 Detailed Prediction History")
                
                df_predictions = pd.DataFrame(prediction_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                
                if completed_predictions:
                    st.caption(f"Showing {len(prediction_data)} of {total_predictions} predictions")
            else:
                st.info("No predictions to display on this page.")
        else:
            st.info("No prediction history available. Make your first prediction above!")
        
        return  # Exit early, don't run analysis
    
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
        
        # Display the selected target (only after analysis starts)
        target_formatted = target_datetime.strftime('%A, %B %d, %Y at %I:%M %p ET')
        st.info(f"📊 Prediction target: **{target_formatted}**")
        
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
            predicted_price_text = f" Predicted price: **${predicted_price:,.0f}**." if predicted_price else ""
            analysis_message = f"Based on fresh analysis at {current_time_str} ET, Bitcoin has a {probability:.0%} chance of being {direction} by {target_formatted}.{predicted_price_text} Recommendation: {recommendation}"
            st.info(f"📊 {analysis_message}")
        else:
            st.success(f"📊 Fresh analysis completed at {current_time_str} ET")
            
            # Set flag to trigger support message
            if 'analysis_completed_time' not in st.session_state:
                st.session_state.analysis_completed_time = None
                
            # Record when analysis completed
            st.session_state.analysis_completed_time = time.time()
            
            # Auto-refresh after 7 seconds to show support message
            st.markdown("""
            <script>
            setTimeout(function() {
                window.location.reload();
            }, 7000);
            </script>
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
        
        
        # Update timestamp
        st.session_state.last_update = current_time
        
        # Prediction History Section (moved to bottom after analysis)
        st.divider()
        
        predictions = load_predictions_history()
        if predictions:
            # Update any past predictions with current price if their target time has passed
            try:
                current_btc_price = btc_1w['Close'].iloc[-1]  # Use already fetched data
                update_prediction_accuracy(float(current_btc_price))
                # Reload predictions after potential updates
                predictions = load_predictions_history()
            except:
                pass  # Continue even if we can't update accuracy
            
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
            end_idx = min(start_idx + predictions_per_page, total_predictions)
            
            # Get predictions for current page (newest first)
            page_predictions = list(reversed(predictions))[start_idx:end_idx]
            
            prediction_data = []
            for pred in page_predictions:
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
                # Prediction Dashboard Analytics (above the table)
                completed_predictions = [p for p in predictions if p.get('actual_price') is not None]
                if completed_predictions:
                    st.subheader("📈 Prediction Performance Dashboard")
                    
                    # Calculate comprehensive metrics
                    errors = []
                    direction_correct = 0
                    very_good_predictions = 0
                    good_predictions = 0
                    fair_predictions = 0
                    poor_predictions = 0
                    
                    for pred in completed_predictions:
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
                        st.metric("Completed", len(completed_predictions))
                    
                    with col3:
                        accuracy_rate = ((very_good_predictions + good_predictions) / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("Accuracy Rate (≤3%)", f"{accuracy_rate:.0f}%")
                    
                    with col4:
                        direction_accuracy = (direction_correct / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("Direction Accuracy", f"{direction_accuracy:.0f}%")
                    
                    with col5:
                        avg_error = sum(errors) / len(errors) if errors else 0
                        st.metric("Avg Error", f"{avg_error:.1f}%")
                    
                    # Detailed accuracy breakdown
                    st.divider()
                    st.subheader("🎯 Accuracy Breakdown")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        very_good_rate = (very_good_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("✅ Very Good (≤1.5%)", f"{very_good_rate:.0f}%", help=f"{very_good_predictions} predictions")
                    
                    with col2:
                        good_rate = (good_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("✅ Good (1.5-3%)", f"{good_rate:.0f}%", help=f"{good_predictions} predictions")
                    
                    with col3:
                        fair_rate = (fair_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("⚠️ Fair (3-5%)", f"{fair_rate:.0f}%", help=f"{fair_predictions} predictions")
                    
                    with col4:
                        poor_rate = (poor_predictions / len(completed_predictions)) * 100 if completed_predictions else 0
                        st.metric("❌ Poor (>5%)", f"{poor_rate:.0f}%", help=f"{poor_predictions} predictions")
                    

                    
                    st.divider()
                    st.subheader("📋 Detailed Prediction History")
                
                df_predictions = pd.DataFrame(prediction_data)
                st.dataframe(df_predictions, use_container_width=True, hide_index=True)
                
                if completed_predictions:
                    st.caption(f"Showing {len(prediction_data)} of {total_predictions} predictions")
            else:
                st.info("No predictions to display on this page.")
        
        # Footer with last update info
        st.divider()
        st.caption(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET | Data source: Yahoo Finance | AI: GPT-5")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)
    

if __name__ == "__main__":
    main()
