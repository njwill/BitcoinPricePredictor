import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st

# Using GPT-4.1 for enhanced data analysis and coding capabilities
# GPT-4.1 released April 2025 with 21.4% improvement over GPT-4o
import anthropic
from anthropic import Anthropic

class AIAnalyzer:
    """Handles AI-powered analysis using Anthropic Claude"""
    
    def __init__(self):
        # Initialize Anthropic (Claude) for technical analysis
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not self.anthropic_key:
            st.error("Anthropic API key not found. Please set ANTHROPIC_API_KEY environment variable.")
            self.claude_client = None
        else:
            self.claude_client = Anthropic(api_key=self.anthropic_key)
        
        # Only using Claude for technical analysis now
    
    def generate_comprehensive_analysis(self, data_3m, data_1w, indicators_3m, indicators_1w, current_price, target_datetime=None):
        """
        Generate comprehensive AI analysis including technical analysis, predictions, and market sentiment
        
        Args:
            data_3m: 3-month Bitcoin price data
            data_1w: 1-week Bitcoin price data  
            indicators_3m: Technical indicators for 3-month data
            indicators_1w: Technical indicators for 1-week data
            current_price: Current Bitcoin price
            target_datetime: Custom target datetime for prediction (defaults to next Friday 4PM ET)
            
        Returns:
            Dictionary with analysis results
        """
        if not self.claude_client:
            return {"error": "Claude client not initialized"}
        
        try:
            # Prepare data summary for AI analysis
            analysis_data = self._prepare_analysis_data(data_3m, data_1w, indicators_3m, indicators_1w, current_price, target_datetime)
            
            # Generate comprehensive technical analysis with Claude
            comprehensive_response = self._generate_technical_analysis_claude(analysis_data)
            
            # Parse the combined response
            parsed_analysis = self._parse_comprehensive_response(comprehensive_response)
            
            # Extract probabilities from prediction section
            probabilities = self._extract_probabilities(parsed_analysis.get('price_prediction', ''))
            
            return {
                'technical_summary': parsed_analysis.get('technical_summary', 'Technical analysis not available'),
                'price_prediction': parsed_analysis.get('price_prediction', 'Price prediction not available'),
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error generating AI analysis: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_analysis_data(self, data_3m, data_1w, indicators_3m, indicators_1w, current_price, target_datetime=None):
        """Prepare and summarize data for AI analysis"""
        try:
            eastern_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern_tz)
            
            # Use custom target datetime if provided, otherwise default to next Friday 4PM
            if target_datetime:
                # Ensure target datetime is in Eastern timezone
                if target_datetime.tzinfo is None:
                    target_datetime = eastern_tz.localize(target_datetime)
                elif target_datetime.tzinfo != eastern_tz:
                    target_datetime = target_datetime.astimezone(eastern_tz)
                prediction_target = target_datetime
            else:
                prediction_target = self._get_next_friday_4pm(current_time)
            
            # Use FULL datasets for accurate period calculations, not trimmed display data
            # The display trimming is only for charts, not for AI analysis
            
            # Use the current_price parameter (from 1-week hourly data) as it's more recent
            # The 3-month data uses daily intervals, so it might be less current than hourly 1-week data
            actual_current_price = float(current_price)
            
            
            # 3-month data summary - use display_from_index for accurate start price
            display_from_3m = getattr(data_3m, 'attrs', {}).get('display_from_index', 0)
            
            if display_from_3m > 0:
                # Use the price from 3 months ago (not 6 months ago)
                start_price_3m = float(data_3m['Close'].iloc[display_from_3m])
            else:
                # Fallback if no display index is set
                start_price_3m = float(data_3m['Close'].iloc[0])
            
            data_3m_summary = {
                'period': '3 months',
                'current_price': actual_current_price,
                'start_price_3m': start_price_3m,
                'high_3m': float(data_3m['High'].max()),
                'low_3m': float(data_3m['Low'].min()),
                'price_change_3m': float((actual_current_price - start_price_3m) / start_price_3m * 100),
                'volatility_3m': float(data_3m['Close'].pct_change().std() * np.sqrt(252) * 100),
                'avg_volume_3m': float(data_3m['Volume'].mean()),
                'start_date': str(data_3m.index[0]),
                'end_date': str(data_3m.index[-1])
            }
            
            # 1-week data summary - use display_from_index for accurate start price
            display_from_1w = getattr(data_1w, 'attrs', {}).get('display_from_index', 0)
            
            if display_from_1w > 0:
                # Use the price from 1 week ago (not 2 weeks ago)
                start_price_1w = float(data_1w['Close'].iloc[display_from_1w])
            else:
                # Fallback if no display index is set
                start_price_1w = float(data_1w['Close'].iloc[0])
            
            data_1w_summary = {
                'period': '1 week',
                'start_price_1w': start_price_1w,
                'high_1w': float(data_1w['High'].max()),
                'low_1w': float(data_1w['Low'].min()),
                'price_change_1w': float((actual_current_price - start_price_1w) / start_price_1w * 100),
                'volatility_1w': float(data_1w['Close'].pct_change().std() * np.sqrt(365) * 100),
                'avg_volume_1w': float(data_1w['Volume'].mean()),
                'start_date': str(data_1w.index[0]),
                'end_date': str(data_1w.index[-1])
            }
            
            # Technical indicators summary (old simple format)
            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w)
            
            # ENHANCED: Add full chart data for deep analysis
            enhanced_data = self._prepare_enhanced_chart_data(data_3m, data_1w, indicators_3m, indicators_1w)
            
            return {
                'current_time': current_time.isoformat(),
                'target_time': prediction_target.isoformat(),
                'hours_until_target': (prediction_target - current_time).total_seconds() / 3600,
                'data_3m': data_3m_summary,
                'data_1w': data_1w_summary,
                'indicators': indicators_summary,
                'enhanced_chart_data': enhanced_data,  # NEW: Full chart data
                'current_price': actual_current_price,
                'target_datetime_formatted': prediction_target.strftime('%A %B %d, %Y at %I:%M %p ET')
            }
            
        except Exception as e:
            st.error(f"Error preparing analysis data: {str(e)}")
            return {}
    
    def _prepare_enhanced_chart_data(self, data_3m, data_1w, indicators_3m, indicators_1w):
        """Prepare comprehensive chart data with full arrays for deep analysis"""
        try:
            enhanced = {}
            
            # FIX: Use only the DISPLAY portion of data, not the full extended fetch
            # Check if data has display range markers
            display_from_3m = getattr(data_3m, 'attrs', {}).get('display_from_index', 0)
            display_from_1w = getattr(data_1w, 'attrs', {}).get('display_from_index', 0)
            
            # Get the correct data ranges for analysis
            # FIX: Check bounds before slicing
            if display_from_3m > 0 and display_from_3m < len(data_3m):
                analysis_data_3m = data_3m.iloc[display_from_3m:]
            elif display_from_3m > 0:
                # If display_from_index is out of bounds, use all data (already trimmed)
                analysis_data_3m = data_3m
            else:
                analysis_data_3m = data_3m
                
            if display_from_1w > 0 and display_from_1w < len(data_1w):
                analysis_data_1w = data_1w.iloc[display_from_1w:]
            elif display_from_1w > 0:
                # If display_from_index is out of bounds, use all data (already trimmed)
                analysis_data_1w = data_1w
            else:
                analysis_data_1w = data_1w
            
            # Get recent data points from the CORRECT display ranges
            recent_3m = analysis_data_3m.tail(50)
            recent_1w = analysis_data_1w.tail(30)
            
            # Calculate highs/lows from CORRECT display data ranges
            full_3m_high = float(analysis_data_3m['High'].max())
            full_3m_low = float(analysis_data_3m['Low'].min()) 
            full_1w_high = float(analysis_data_1w['High'].max())
            full_1w_low = float(analysis_data_1w['Low'].min())
            
            # DEBUG: Show corrected data info
            st.error(f"🚨 3M: Original Shape={data_3m.shape}, Display from index={display_from_3m}, Final Shape={analysis_data_3m.shape}")
            st.error(f"🚨 3M FINAL HIGHS: Top 5 = {analysis_data_3m['High'].nlargest(5).tolist()}")
            st.error(f"🚨 1W: Original Shape={data_1w.shape}, Display from index={display_from_1w}, Final Shape={analysis_data_1w.shape}")
            st.error(f"🚨 1W FINAL HIGHS: Top 5 = {analysis_data_1w['High'].nlargest(5).tolist()}")
            st.error(f"🚨 CALCULATED VALUES: 3M High=${full_3m_high:,.0f}, 1W High=${full_1w_high:,.0f}")
            
            # Ensure ALL indexes are datetime to prevent strftime errors
            try:
                # Convert analysis dataset indexes
                if not hasattr(analysis_data_3m.index[0], 'strftime'):
                    analysis_data_3m.index = pd.to_datetime(analysis_data_3m.index)
                if not hasattr(analysis_data_1w.index[0], 'strftime'):
                    analysis_data_1w.index = pd.to_datetime(analysis_data_1w.index)
                    
                # Convert recent dataset indexes
                if not hasattr(recent_3m.index[0], 'strftime'):
                    recent_3m.index = pd.to_datetime(recent_3m.index)
                if not hasattr(recent_1w.index[0], 'strftime'):
                    recent_1w.index = pd.to_datetime(recent_1w.index)
                    
            except (IndexError, AttributeError, TypeError) as e:
                # Handle empty dataframes or other issues
                st.warning(f"Issue with date formatting in chart data: {str(e)}")
                return {}
            
            # 3-MONTH ENHANCED DATA
            enhanced['3m_data'] = {
                'timeframe': '3-month',
                'full_range': f"{analysis_data_3m.index[0].strftime('%B %d')} to {analysis_data_3m.index[-1].strftime('%B %d, %Y')}",
                'data_range': f"{recent_3m.index[0].strftime('%B %d')} to {recent_3m.index[-1].strftime('%B %d, %Y')}",
                'period_highs_lows': {
                    'period_high': full_3m_high,
                    'period_low': full_3m_low,
                    'recent_high': float(recent_3m['High'].max()),
                    'recent_low': float(recent_3m['Low'].min())
                },
                'recent_prices': {
                    'dates': [d.strftime('%Y-%m-%d %H:%M') for d in recent_3m.index],
                    'open': recent_3m['Open'].round(2).tolist(),
                    'high': recent_3m['High'].round(2).tolist(), 
                    'low': recent_3m['Low'].round(2).tolist(),
                    'close': recent_3m['Close'].round(2).tolist(),
                    'volume': recent_3m['Volume'].round(0).tolist()
                },
                'indicators': {}
            }
            
            # DEBUG: Verify what was actually stored in enhanced data
            st.warning(f"🔧 ENHANCED 3M STORED: {enhanced['3m_data']['period_highs_lows']['period_high']:,.0f}")
            
            # Add full indicator arrays for 3M
            for indicator in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'EMA_20', 'SMA_50', 'SMA_200']:
                if indicator in indicators_3m:
                    values = indicators_3m[indicator].tail(50).dropna()
                    enhanced['3m_data']['indicators'][indicator] = values.round(4).tolist()
            
            # 1-WEEK ENHANCED DATA  
            enhanced['1w_data'] = {
                'timeframe': '1-week',
                'full_range': f"{analysis_data_1w.index[0].strftime('%B %d')} to {analysis_data_1w.index[-1].strftime('%B %d, %Y')}",
                'data_range': f"{recent_1w.index[0].strftime('%B %d')} to {recent_1w.index[-1].strftime('%B %d, %Y')}",
                'period_highs_lows': {
                    'period_high': full_1w_high,
                    'period_low': full_1w_low,
                    'recent_high': float(recent_1w['High'].max()),
                    'recent_low': float(recent_1w['Low'].min())
                },
                'recent_prices': {
                    'dates': [d.strftime('%Y-%m-%d %H:%M') for d in recent_1w.index],
                    'open': recent_1w['Open'].round(2).tolist(),
                    'high': recent_1w['High'].round(2).tolist(),
                    'low': recent_1w['Low'].round(2).tolist(), 
                    'close': recent_1w['Close'].round(2).tolist(),
                    'volume': recent_1w['Volume'].round(0).tolist()
                },
                'indicators': {}
            }
            
            # DEBUG: Verify what was actually stored in enhanced data
            st.warning(f"🔧 ENHANCED 1W STORED: {enhanced['1w_data']['period_highs_lows']['period_high']:,.0f}")
            
            # Add full indicator arrays for 1W
            for indicator in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle', 'EMA_20', 'SMA_50', 'SMA_200']:
                if indicator in indicators_1w:
                    values = indicators_1w[indicator].tail(30).dropna()
                    enhanced['1w_data']['indicators'][indicator] = values.round(4).tolist()
            
            # VOLUME ANALYSIS
            enhanced['volume_analysis'] = {
                '3m_avg_volume': float(data_3m['Volume'].tail(50).mean()),
                '3m_volume_trend': 'increasing' if data_3m['Volume'].tail(10).mean() > data_3m['Volume'].tail(50).mean() else 'decreasing',
                '1w_avg_volume': float(data_1w['Volume'].tail(30).mean()),
                '1w_volume_trend': 'increasing' if data_1w['Volume'].tail(5).mean() > data_1w['Volume'].tail(30).mean() else 'decreasing'
            }
            
            return enhanced
            
        except Exception as e:
            st.warning(f"Error preparing enhanced chart data: {str(e)}")
            return {}
    
    def _summarize_indicators(self, indicators_3m, indicators_1w):
        """Summarize technical indicators for AI analysis"""
        summary = {}
        
        try:
            # RSI summary
            if 'RSI' in indicators_3m and 'RSI' in indicators_1w:
                summary['RSI'] = {
                    '3m_current': float(indicators_3m['RSI'].iloc[-1]) if not np.isnan(indicators_3m['RSI'].iloc[-1]) else None,
                    '1w_current': float(indicators_1w['RSI'].iloc[-1]) if not np.isnan(indicators_1w['RSI'].iloc[-1]) else None
                }
            
            # MACD summary
            if 'MACD' in indicators_3m and 'MACD_Signal' in indicators_3m:
                macd_3m = indicators_3m['MACD'].iloc[-1]
                signal_3m = indicators_3m['MACD_Signal'].iloc[-1]
                summary['MACD_3m'] = {
                    'macd': float(macd_3m) if not np.isnan(macd_3m) else None,
                    'signal': float(signal_3m) if not np.isnan(signal_3m) else None,
                    'crossover': 'bullish' if macd_3m > signal_3m else 'bearish'
                }
            
            if 'MACD' in indicators_1w and 'MACD_Signal' in indicators_1w:
                macd_1w = indicators_1w['MACD'].iloc[-1]
                signal_1w = indicators_1w['MACD_Signal'].iloc[-1]
                summary['MACD_1w'] = {
                    'macd': float(macd_1w) if not np.isnan(macd_1w) else None,
                    'signal': float(signal_1w) if not np.isnan(signal_1w) else None,
                    'crossover': 'bullish' if macd_1w > signal_1w else 'bearish'
                }
            
            # Bollinger Bands summary
            for timeframe, indicators in [('3m', indicators_3m), ('1w', indicators_1w)]:
                if all(key in indicators for key in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                    upper = indicators['BB_Upper'].iloc[-1]
                    lower = indicators['BB_Lower'].iloc[-1]
                    middle = indicators['BB_Middle'].iloc[-1]
                    
                    if not any(np.isnan([upper, lower, middle])):
                        summary[f'BB_{timeframe}'] = {
                            'upper': float(upper),
                            'lower': float(lower),
                            'middle': float(middle),
                            'position': 'above_upper' if summary.get('current_price', 0) > upper else 'below_lower' if summary.get('current_price', 0) < lower else 'within_bands'
                        }
            
            # EMA summary
            for timeframe, indicators in [('3m', indicators_3m), ('1w', indicators_1w)]:
                if 'EMA_20' in indicators:
                    ema_20 = indicators['EMA_20'].iloc[-1]
                    if not np.isnan(ema_20):
                        summary[f'EMA_20_{timeframe}'] = {
                            'value': float(ema_20),
                            'trend': 'bullish' if summary.get('current_price', 0) > ema_20 else 'bearish'
                        }
            
        except Exception as e:
            st.warning(f"Error summarizing indicators: {str(e)}")
        
        return summary
    
    def _generate_technical_analysis(self, analysis_data):
        """Generate technical analysis using AI"""
        try:
            if not self.client:
                return "Error: OpenAI client not initialized"
            # Extract key price points for clarity
            current_price = analysis_data.get('current_price', 0)
            data_3m = analysis_data.get('data_3m', {})
            data_1w = analysis_data.get('data_1w', {})
            
            prompt = f"""
            You are an expert Bitcoin technical analyst. Analyze the provided Bitcoin price data with EXTREME ACCURACY.
            
            === CRITICAL PRICE INFORMATION ===
            CURRENT BITCOIN PRICE: ${current_price:,.2f}
            
            3-MONTH TIMEFRAME (Period: {data_3m.get('start_date', 'N/A')} to {data_3m.get('end_date', 'N/A')}):
            - Current Price: ${current_price:,.2f}
            - Start Price: ${data_3m.get('start_price_3m', 0):,.2f} 
            - Period High: ${data_3m.get('high_3m', 0):,.2f}
            - Period Low: ${data_3m.get('low_3m', 0):,.2f}
            - Price Change: {data_3m.get('price_change_3m', 0):+.2f}%
            
            1-WEEK TIMEFRAME (Period: {data_1w.get('start_date', 'N/A')} to {data_1w.get('end_date', 'N/A')}):
            - Current Price: ${current_price:,.2f}
            - Start Price: ${data_1w.get('start_price_1w', 0):,.2f}
            - Period High: ${data_1w.get('high_1w', 0):,.2f}
            - Period Low: ${data_1w.get('low_1w', 0):,.2f}
            - Price Change: {data_1w.get('price_change_1w', 0):+.2f}%
            
            Technical Indicators:
            {json.dumps(analysis_data.get('indicators', {}), indent=2)}
            
            CRITICAL FORMATTING REQUIREMENTS:
            1. ALL PRICES must include $ sign (e.g., $115,287.50, not 115287.50)
            2. Price logic must be correct: 
               - If current > start = "increased/rose FROM start TO current"
               - If current < start = "decreased/fell FROM start TO current"
            3. Support = price levels BELOW ${current_price:,.2f}
            4. Resistance = price levels ABOVE ${current_price:,.2f}
            5. Make Buy/Sell/Hold recommendations **BOLD** like **Buy** or **Sell** or **Hold**
            
            Provide analysis covering:
            1. Price action analysis (use correct FROM/TO direction)
            2. Technical indicator interpretation  
            3. Accurate support/resistance levels
            4. Market structure and trend analysis
            5. Key technical patterns
            
            Double-check all price values and logic before responding. Keep analysis 200-300 words.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a professional Bitcoin technical analyst with expertise in chart analysis and technical indicators."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating technical analysis: {str(e)}"
    
    def _generate_price_prediction(self, analysis_data):
        """Generate price prediction with probabilities"""
        try:
            if not self.client:
                return "Error: OpenAI client not initialized"
            # Extract key data for clarity
            current_price = analysis_data.get('current_price', 0)
            hours_until_target = analysis_data.get('hours_until_target', 0)
            data_3m = analysis_data.get('data_3m', {})
            data_1w = analysis_data.get('data_1w', {})
            
            prompt = f"""
            Calculate probability of Bitcoin price movement by Friday at 4:00 PM Eastern Time.
            
            === CURRENT BITCOIN PRICE ===
            ${current_price:,.2f}
            
            Time until Friday 4PM ET: {hours_until_target:.1f} hours
            
            === PRICE MOVEMENTS ===
            3-Month: ${data_3m.get('start_price_3m', 0):,.2f} → ${current_price:,.2f} ({data_3m.get('price_change_3m', 0):+.2f}%)
            1-Week: ${data_1w.get('start_price_1w', 0):,.2f} → ${current_price:,.2f} ({data_1w.get('price_change_1w', 0):+.2f}%)
            
            === PRICE RANGES ===
            3-Month High: ${data_3m.get('high_3m', 0):,.2f}
            3-Month Low: ${data_3m.get('low_3m', 0):,.2f}
            1-Week High: ${data_1w.get('high_1w', 0):,.2f}
            1-Week Low: ${data_1w.get('low_1w', 0):,.2f}
            
            Technical Indicators:
            {json.dumps(analysis_data.get('indicators', {}), indent=2)}
            
            FORMATTING REQUIREMENTS:
            - ALL prices must include $ sign
            - Make trading recommendations **BOLD** like **Buy**, **Sell**, **Hold**
            - Be consistent with probability percentages across sections
            
            Provide EXACTLY these sections:
            1. Probability of Price Being HIGHER by Friday 4PM ET: [X]%
            2. Probability of Price Being LOWER by Friday 4PM ET: [Y]%
            3. Confidence Level: [Z]%
            4. Key technical factors supporting assessment
            5. Potential price targets (with $ signs)
            6. Trading recommendation: **Buy**//**Sell**//**Hold**
            
            CRITICAL: X + Y must equal 100%. Use the SAME percentages in all analysis sections.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a quantitative Bitcoin analyst specializing in probability-based price predictions using technical analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating price prediction: {str(e)}"
    
    
    def _extract_probabilities(self, prediction_text):
        """Extract probability percentages from prediction text"""
        probabilities = {'higher': 0.5, 'lower': 0.5, 'confidence': 0.5}
        
        try:
            # Simple regex patterns to extract percentages
            import re
            
            # Look for patterns like "60% higher", "probability of 65%", etc.
            higher_patterns = [
                r'(\d+)%?\s*(?:probability|chance|likelihood).*?(?:higher|up|increase)',
                r'(?:higher|up|increase).*?(\d+)%',
                r'HIGHER.*?(\d+)%',
                r'(\d+)%.*?higher'
            ]
            
            lower_patterns = [
                r'(\d+)%?\s*(?:probability|chance|likelihood).*?(?:lower|down|decrease)',
                r'(?:lower|down|decrease).*?(\d+)%',
                r'LOWER.*?(\d+)%',
                r'(\d+)%.*?lower'
            ]
            
            confidence_patterns = [
                r'confidence.*?(\d+)%',
                r'(\d+)%.*?confidence',
                r'confident.*?(\d+)%'
            ]
            
            text_lower = prediction_text.lower()
            
            # Extract higher probability
            for pattern in higher_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    probabilities['higher'] = float(matches[0]) / 100
                    break
            
            # Extract lower probability
            for pattern in lower_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    probabilities['lower'] = float(matches[0]) / 100
                    break
            
            # Extract confidence
            for pattern in confidence_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    probabilities['confidence'] = float(matches[0]) / 100
                    break
            
            # Ensure probabilities sum to 1
            total = probabilities['higher'] + probabilities['lower']
            if total > 0:
                probabilities['higher'] = probabilities['higher'] / total
                probabilities['lower'] = probabilities['lower'] / total
            
        except Exception as e:
            st.warning(f"Error extracting probabilities: {str(e)}")
        
        return probabilities
    
    def _get_next_friday_4pm(self, current_time):
        """Calculate next Friday 4PM Eastern Time"""
        # Find next Friday
        days_ahead = 4 - current_time.weekday()  # Friday is 4
        if days_ahead <= 0 or (days_ahead == 0 and current_time.hour >= 16):
            days_ahead += 7
        
        next_friday = current_time + timedelta(days=days_ahead)
        friday_4pm = next_friday.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return friday_4pm
    
    def _generate_technical_analysis_claude(self, analysis_data):
        """Generate technical analysis and price prediction using Claude"""
        try:
            if not self.claude_client:
                return "Error: Claude client not initialized"
            
            # Extract key data for clarity
            current_price = analysis_data.get('current_price', 0)
            hours_until_target = analysis_data.get('hours_until_target', 0)
            data_3m = analysis_data.get('data_3m', {})
            data_1w = analysis_data.get('data_1w', {})
            target_datetime_formatted = analysis_data.get('target_datetime_formatted', 'Friday 4PM ET')
            
            
            # Get current date and actual data range dynamically
            from datetime import datetime
            current_date = datetime.now().strftime('%B %d, %Y')
            start_date = data_3m.get('start_date', 'N/A')
            end_date = data_3m.get('end_date', 'N/A')
            
            # DEBUG: Show actual price data being sent to Claude
            enhanced_data = analysis_data.get('enhanced_chart_data', {})
            if enhanced_data.get('3m_data') and enhanced_data['3m_data'].get('period_highs_lows'):
                highs_lows_3m = enhanced_data['3m_data']['period_highs_lows']
                st.success(f"🔍 3M FULL PERIOD: High=${highs_lows_3m.get('period_high', 0):,.0f}, Low=${highs_lows_3m.get('period_low', 0):,.0f}")
            
            if enhanced_data.get('1w_data') and enhanced_data['1w_data'].get('period_highs_lows'):
                highs_lows_1w = enhanced_data['1w_data']['period_highs_lows']
                st.success(f"🔍 1W FULL PERIOD: High=${highs_lows_1w.get('period_high', 0):,.0f}, Low=${highs_lows_1w.get('period_low', 0):,.0f}")
            
            st.success(f"🔍 CURRENT PRICE PARAMETER: ${current_price:,.2f}")
            
            comprehensive_prompt = f"""
            CRITICAL: Today is {current_date}. The data provided covers ONLY {start_date} through {end_date}.
            
            DO NOT REFERENCE ANY DATES OUTSIDE THIS RANGE. There is NO December data - do not mention December.
            
            Bitcoin's current price is ${current_price:,.2f}.
            
            Always use ${current_price:,.2f} when referring to Bitcoin's current price.
            
            DATA RANGE VALIDATION:
            • Start: {start_date}  
            • End: {end_date}
            • ONLY analyze data within this range
            
            COMPREHENSIVE CHART DATA:
            {json.dumps(analysis_data.get('enhanced_chart_data', {}), indent=2)}
            
            SUMMARY INDICATORS:
            {json.dumps(analysis_data.get('indicators', {}), indent=2)}
            
            PERFORMANCE SUMMARY:
            • 3-month change: {data_3m.get('price_change_3m', 0):+.2f}%
            • 1-week change: {data_1w.get('price_change_1w', 0):+.2f}%
            
            Provide analysis in three sections:
            
            [TECHNICAL_ANALYSIS_START]
            **COMPREHENSIVE TECHNICAL ANALYSIS**
            
            **Current Price: ${current_price:,.2f}**
            
            **1. MULTI-TIMEFRAME OVERVIEW**
            - 3-Month Chart Analysis: [Trend, key levels, patterns]
            - 1-Week Chart Analysis: [Trend, key levels, patterns] 
            - Timeframe Alignment: [How they agree or conflict]
            
            **2. TECHNICAL INDICATORS ANALYSIS**
            - RSI Analysis: [3M vs 1W RSI, overbought/oversold, divergences]
            - MACD Analysis: [Signal line crossovers, histogram, divergences]
            - Bollinger Bands: [Position relative to bands, squeeze/expansion]
            - EMA Analysis: [20-period trends, price above/below]
            
            **3. ADVANCED PATTERN ANALYSIS**
            - Chart Patterns: [Triangles, head & shoulders, flags, wedges]
            - Candlestick Patterns: [Recent significant patterns]
            - Support/Resistance: [Key levels identified from price action]
            
            **4. DIVERGENCE ANALYSIS**
            - Price vs RSI Divergences: [Bullish or bearish divergences]
            - Price vs MACD Divergences: [Hidden or regular divergences]
            - Volume Divergences: [If volume data conflicts with price]
            
            **5. FAILURE SWING ANALYSIS**
            - RSI Failure Swings: [Bullish/bearish failure swings in RSI]
            - MACD Failures: [Failed breakouts or breakdowns]
            - Price Action Failures: [Failed support/resistance tests]
            
            **6. TRENDLINE & STRUCTURE**
            - Primary Trendlines: [Major ascending/descending lines]
            - Support Levels: [Key horizontal and trend support]
            - Resistance Levels: [Key horizontal and trend resistance]
            - Market Structure: [Higher highs/lows, lower highs/lows]
            
            **7. TRADING RECOMMENDATION**
            - Overall Bias: **[BULLISH/BEARISH/NEUTRAL]**
            - Entry Strategy: [Specific entry points and conditions]
            - Stop Loss: [Risk management levels]
            - Target Levels: [Profit-taking areas]
            [TECHNICAL_ANALYSIS_END]
            
            [PRICE_PREDICTION_START]
            **PRICE PREDICTION for {target_datetime_formatted}**
            
            Based on the comprehensive technical analysis above:
            
            1. **Probability HIGHER than ${current_price:,.2f}: [X]%**
            2. **Probability LOWER than ${current_price:,.2f}: [Y]%**
            3. **Confidence Level: [Z]%**
            
            **Key Technical Factors Supporting This Assessment:**
            - [List 3-5 specific technical reasons for the prediction]
            
            **Potential Price Targets:**
            - Upside Target 1: $[amount] (reasoning)
            - Upside Target 2: $[amount] (reasoning)
            - Downside Target 1: $[amount] (reasoning)
            - Downside Target 2: $[amount] (reasoning)
            
            **Critical Levels to Watch:**
            - Bullish above: $[level]
            - Bearish below: $[level]
            
            [PRICE_PREDICTION_END]
            
            STRICT ANALYSIS RULES:
            - ONLY analyze data from {start_date} to {end_date}
            - NEVER mention dates outside this actual data range
            - NEVER reference December, November, or any months not in the provided data
            - Use ONLY the actual dates provided in the chart data arrays
            - ALWAYS reference actual dates (like "May 24" or "August 15"), NEVER array indices (like "132-181")
            - Probabilities must sum to 100% (internal instruction - do not print this)
            
            ANALYZE THE COMPREHENSIVE CHART DATA ABOVE, including full price arrays and indicator series. Use this data for:
            
            1. **Pattern Recognition**: Use the full price arrays to identify chart patterns
            2. **Divergence Analysis**: Compare price movements vs full RSI/MACD arrays for divergences  
            3. **Failure Swing Analysis**: Examine full indicator arrays for failure swings
            4. **Trendline Analysis**: Use the OHLC arrays to identify support/resistance trendlines
            5. **Volume Confirmation**: Analyze volume patterns with price movements
            6. **Multi-timeframe Comparison**: Compare 3M arrays vs 1W arrays for alignment/conflict
            7. **Candlestick Patterns**: Use OHLC data for candlestick pattern analysis
            
            REFERENCE ONLY ACTUAL CHART DATA - no external knowledge about Bitcoin price history.
            """
            
            
            # The newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229".
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                temperature=0.3,
                system="You are a professional Bitcoin analyst providing comprehensive, consistent analysis across technical, predictive, and market perspectives.",
                messages=[
                    {"role": "user", "content": comprehensive_prompt}
                ]
            )
            
            ai_response = response.content[0].text if response.content and hasattr(response.content[0], 'text') else str(response.content[0])
            
            
            return ai_response
            
        except Exception as e:
            return f"Error generating technical analysis: {str(e)}"
    
    
    def _parse_comprehensive_response(self, response):
        """Parse the comprehensive response into separate sections"""
        try:
            sections = {}
            
            # Extract Technical Analysis
            tech_start = response.find("[TECHNICAL_ANALYSIS_START]")
            tech_end = response.find("[TECHNICAL_ANALYSIS_END]")
            if tech_start != -1 and tech_end != -1:
                tech_content = response[tech_start + len("[TECHNICAL_ANALYSIS_START]"):tech_end].strip()
                sections['technical_summary'] = tech_content
            
            # Extract Price Prediction
            pred_start = response.find("[PRICE_PREDICTION_START]")
            pred_end = response.find("[PRICE_PREDICTION_END]")
            if pred_start != -1 and pred_end != -1:
                pred_content = response[pred_start + len("[PRICE_PREDICTION_START]"):pred_end].strip()
                sections['price_prediction'] = pred_content
            
            # Market sentiment section removed - focusing only on technical analysis
            
            # If structured parsing fails, try to split by common headers
            if not sections:
                # Fallback parsing method
                lines = response.split('\n')
                current_section = None
                current_content = []
                
                for line in lines:
                    line_lower = line.lower().strip()
                    if 'technical analysis' in line_lower:
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = 'technical_summary'
                        current_content = []
                    elif 'price prediction' in line_lower or 'friday' in line_lower:
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = 'price_prediction'
                        current_content = []
                    # Market sentiment parsing removed
                    elif current_section:
                        current_content.append(line)
                
                # Add the last section
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content).strip()
            
            return sections
            
        except Exception as e:
            # Return the full response if parsing fails
            return {
                'technical_summary': response,
                'price_prediction': 'Unable to parse prediction section',
            }
