import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Union, Optional

def format_currency(amount: float, currency: str = "USD") -> str:
    """
    Format currency amounts for display
    
    Args:
        amount: Numeric amount to format
        currency: Currency symbol (default: USD)
        
    Returns:
        str: Formatted currency string
    """
    try:
        if pd.isna(amount) or np.isnan(amount):
            return "N/A"
        
        if currency == "USD":
            return f"${amount:,.2f}"
        elif currency == "BTC":
            return f"₿{amount:.8f}"
        else:
            return f"{amount:,.2f} {currency}"
            
    except (ValueError, TypeError):
        return "N/A"

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    Format percentage values for display
    
    Args:
        value: Percentage value (as decimal, e.g., 0.05 for 5%)
        decimal_places: Number of decimal places
        
    Returns:
        str: Formatted percentage string
    """
    try:
        if pd.isna(value) or np.isnan(value):
            return "N/A"
        
        percentage = value * 100
        return f"{percentage:+.{decimal_places}f}%"
        
    except (ValueError, TypeError):
        return "N/A"

def format_large_number(number: float, suffix: bool = True) -> str:
    """
    Format large numbers with appropriate suffixes (K, M, B, T)
    
    Args:
        number: Number to format
        suffix: Whether to add suffix (K, M, B, T)
        
    Returns:
        str: Formatted number string
    """
    try:
        if pd.isna(number) or np.isnan(number):
            return "N/A"
        
        if not suffix:
            return f"{number:,.0f}"
        
        abs_number = abs(number)
        
        if abs_number >= 1_000_000_000_000:
            return f"{number/1_000_000_000_000:.1f}T"
        elif abs_number >= 1_000_000_000:
            return f"{number/1_000_000_000:.1f}B"
        elif abs_number >= 1_000_000:
            return f"{number/1_000_000:.1f}M"
        elif abs_number >= 1_000:
            return f"{number/1_000:.1f}K"
        else:
            return f"{number:.1f}"
            
    except (ValueError, TypeError):
        return "N/A"

def get_eastern_time() -> datetime:
    """
    Get current time in Eastern timezone
    
    Returns:
        datetime: Current Eastern time
    """
    eastern_tz = pytz.timezone('US/Eastern')
    return datetime.now(eastern_tz)

def calculate_time_until_update(current_time: datetime) -> str:
    """
    Calculate and format time until next Monday 9:30 AM ET update
    
    Args:
        current_time: Current time in Eastern timezone
        
    Returns:
        str: Formatted time until next update
    """
    try:
        # Calculate next Monday 9:30 AM
        days_until_monday = (7 - current_time.weekday()) % 7
        
        if days_until_monday == 0:  # Today is Monday
            if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                # Update hasn't happened yet today
                next_update = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            else:
                # Update already happened, next Monday
                next_update = current_time + timedelta(days=7)
                next_update = next_update.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            # Calculate next Monday
            next_update = current_time + timedelta(days=days_until_monday)
            next_update = next_update.replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Calculate time difference
        time_diff = next_update - current_time
        
        # Format the difference
        days = time_diff.days
        hours, remainder = divmod(time_diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        if days > 0:
            return f"Monday {next_update.strftime('%m/%d')} at 9:30 AM ET (in {days}d {hours}h {minutes}m)"
        elif hours > 0:
            return f"Today at 9:30 AM ET (in {hours}h {minutes}m)" if days_until_monday == 0 else f"Monday at 9:30 AM ET (in {hours}h {minutes}m)"
        else:
            return f"Monday at 9:30 AM ET (in {minutes}m)"
            
    except Exception as e:
        return f"Error calculating update time: {str(e)}"

def validate_price_data(data: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate Bitcoin price data for completeness and sanity
    
    Args:
        data: DataFrame with Bitcoin price data
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        if data.empty:
            return False, "No data available"
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for reasonable price ranges
        min_price = data['Close'].min()
        max_price = data['Close'].max()
        
        if min_price < 100 or max_price > 10_000_000:
            return False, f"Prices outside reasonable range: ${min_price:,.2f} - ${max_price:,.2f}"
        
        # Check for missing values
        null_counts = data[required_columns].isnull().sum()
        total_nulls = null_counts.sum()
        
        if total_nulls > 0:
            return False, f"Found {total_nulls} missing values in price data"
        
        # Check for negative values where they shouldn't exist
        if (data[['Open', 'High', 'Low', 'Close']] < 0).any().any():
            return False, "Found negative price values"
        
        if (data['Volume'] < 0).any():
            return False, "Found negative volume values"
        
        # Check for logical consistency (High >= Low, etc.)
        if (data['High'] < data['Low']).any():
            return False, "Found High prices lower than Low prices"
        
        if (data['High'] < data['Close']).any() or (data['High'] < data['Open']).any():
            return False, "Found High prices lower than Open/Close prices"
        
        if (data['Low'] > data['Close']).any() or (data['Low'] > data['Open']).any():
            return False, "Found Low prices higher than Open/Close prices"
        
        return True, "Data validation passed"
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

def calculate_price_metrics(data: pd.DataFrame) -> dict:
    """
    Calculate various price metrics from Bitcoin data
    
    Args:
        data: DataFrame with Bitcoin price data
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    try:
        if data.empty:
            return {}
        
        metrics = {}
        
        # Basic price metrics
        metrics['current_price'] = float(data['Close'].iloc[-1])
        metrics['price_change'] = float(data['Close'].iloc[-1] - data['Close'].iloc[0])
        metrics['price_change_pct'] = float((metrics['price_change'] / data['Close'].iloc[0]) * 100)
        
        # High/Low metrics
        metrics['period_high'] = float(data['High'].max())
        metrics['period_low'] = float(data['Low'].min())
        metrics['high_low_range'] = metrics['period_high'] - metrics['period_low']
        
        # Volatility metrics
        daily_returns = data['Close'].pct_change().dropna()
        metrics['volatility_daily'] = float(daily_returns.std())
        metrics['volatility_annualized'] = float(daily_returns.std() * np.sqrt(365))
        
        # Volume metrics
        metrics['avg_volume'] = float(data['Volume'].mean())
        metrics['total_volume'] = float(data['Volume'].sum())
        metrics['volume_trend'] = float((data['Volume'].iloc[-5:].mean() / data['Volume'].iloc[:5].mean() - 1) * 100) if len(data) >= 10 else 0
        
        # Recent performance
        if len(data) >= 7:
            metrics['7d_change'] = float((data['Close'].iloc[-1] / data['Close'].iloc[-7] - 1) * 100)
        
        if len(data) >= 30:
            metrics['30d_change'] = float((data['Close'].iloc[-1] / data['Close'].iloc[-30] - 1) * 100)
        
        # Distance from highs/lows
        metrics['distance_from_high'] = float((metrics['current_price'] / metrics['period_high'] - 1) * 100)
        metrics['distance_from_low'] = float((metrics['current_price'] / metrics['period_low'] - 1) * 100)
        
        return metrics
        
    except Exception as e:
        st.warning(f"Error calculating price metrics: {str(e)}")
        return {}

def format_time_remaining(target_time: datetime, current_time: datetime = None) -> str:
    """
    Format time remaining until target time
    
    Args:
        target_time: Target datetime
        current_time: Current datetime (default: now in Eastern)
        
    Returns:
        str: Formatted time remaining string
    """
    try:
        if current_time is None:
            current_time = get_eastern_time()
        
        time_diff = target_time - current_time
        
        if time_diff.total_seconds() <= 0:
            return "Time has passed"
        
        days = time_diff.days
        hours, remainder = divmod(time_diff.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        
        if not parts:
            return f"{seconds} seconds"
        
        return ", ".join(parts)
        
    except Exception as e:
        return f"Error calculating time: {str(e)}"

def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero
    
    Args:
        numerator: Number to divide
        denominator: Number to divide by
        default: Default value if division by zero
        
    Returns:
        float: Result of division or default value
    """
    try:
        if denominator == 0 or pd.isna(denominator) or np.isnan(denominator):
            return default
        
        result = numerator / denominator
        
        if pd.isna(result) or np.isnan(result) or np.isinf(result):
            return default
        
        return float(result)
        
    except (ValueError, TypeError, ZeroDivisionError):
        return default

def create_error_message(error: Exception, context: str = "operation") -> str:
    """
    Create user-friendly error messages
    
    Args:
        error: Exception object
        context: Context where error occurred
        
    Returns:
        str: Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Common error patterns and user-friendly messages
    if "API" in error_msg.upper() or "request" in error_msg.lower():
        return f"❌ Network error during {context}. Please check your connection and try again."
    elif "key" in error_msg.lower() and ("api" in error_msg.lower() or "token" in error_msg.lower()):
        return f"❌ Authentication error during {context}. Please check your API credentials."
    elif "timeout" in error_msg.lower():
        return f"❌ Request timed out during {context}. Please try again later."
    elif "rate limit" in error_msg.lower():
        return f"❌ Rate limit exceeded during {context}. Please wait and try again."
    else:
        return f"❌ Error during {context}: {error_type} - {error_msg}"
