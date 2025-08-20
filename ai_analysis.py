import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st

# Using GPT-4.1 for enhanced data analysis and coding capabilities
# GPT-4.1 released April 2025 with 21.4% improvement over GPT-4o
from openai import OpenAI

class AIAnalyzer:
    """Handles AI-powered analysis using OpenAI GPT-4.1"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
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
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        try:
            # Prepare data summary for AI analysis
            analysis_data = self._prepare_analysis_data(data_3m, data_1w, indicators_3m, indicators_1w, current_price, target_datetime)
            
            # Generate comprehensive analysis in single query for consistency
            comprehensive_response = self._generate_unified_analysis(analysis_data)
            
            # Parse the comprehensive response
            parsed_analysis = self._parse_comprehensive_response(comprehensive_response)
            
            # Extract probabilities from prediction section
            probabilities = self._extract_probabilities(parsed_analysis.get('price_prediction', ''))
            
            return {
                'technical_summary': parsed_analysis.get('technical_summary', 'Technical analysis not available'),
                'price_prediction': parsed_analysis.get('price_prediction', 'Price prediction not available'),
                'market_sentiment': parsed_analysis.get('market_sentiment', 'Market sentiment not available'),
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
            
            # Get actual current price from the latest data point of FULL 3-month data
            actual_current_price = float(data_3m['Close'].iloc[-1])
            
            # Validate that current_price parameter matches data
            if abs(current_price - actual_current_price) > 1:  # Allow small differences
                st.warning(f"Price mismatch detected: parameter={current_price}, data={actual_current_price}")
                actual_current_price = current_price  # Use the parameter value
            
            
            # 3-month data summary (use FULL dataset for accurate 3-month period)
            start_price_3m = float(data_3m['Close'].iloc[0])
            
            # Debug information to help identify the data issue
            if abs(start_price_3m - actual_current_price) < 100:  # If they're too close, something's wrong
                st.warning(f"Debug: Possible data ordering issue - Start: ${start_price_3m:,.2f}, Current: ${actual_current_price:,.2f}")
                st.warning(f"Debug: Data range {data_3m.index[0]} to {data_3m.index[-1]}")
            
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
            
            # 1-week data summary (use FULL dataset for accurate 1-week period)
            start_price_1w = float(data_1w['Close'].iloc[0])
            
            # Debug information for 1-week data too
            if abs(start_price_1w - actual_current_price) < 100:
                st.warning(f"Debug 1W: Possible data ordering issue - Start: ${start_price_1w:,.2f}, Current: ${actual_current_price:,.2f}")
                st.warning(f"Debug 1W: Data range {data_1w.index[0]} to {data_1w.index[-1]}")
            
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
            
            # Technical indicators summary
            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w)
            
            return {
                'current_time': current_time.isoformat(),
                'target_time': prediction_target.isoformat(),
                'hours_until_target': (prediction_target - current_time).total_seconds() / 3600,
                'data_3m': data_3m_summary,
                'data_1w': data_1w_summary,
                'indicators': indicators_summary,
                'current_price': actual_current_price,
                'target_datetime_formatted': prediction_target.strftime('%A %B %d, %Y at %I:%M %p ET')
            }
            
        except Exception as e:
            st.error(f"Error preparing analysis data: {str(e)}")
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
    
    def _generate_market_sentiment(self, analysis_data):
        """Generate market sentiment and key events analysis"""
        try:
            if not self.client:
                return "Error: OpenAI client not initialized"
            current_date = datetime.now().strftime("%Y-%m-%d")
            
            prompt = f"""
            Analyze the general cryptocurrency market sentiment for the upcoming week and identify key events that may impact Bitcoin's price.
            
            Current Date: {current_date}
            Current Bitcoin Price: ${analysis_data.get('current_price', 0):,.2f}
            
            Please provide:
            1. General market sentiment analysis for the upcoming week
            2. Key scheduled events that typically impact Bitcoin (Fed meetings, economic data releases, options/futures expiry)
            3. Seasonal or cyclical patterns relevant to this time period
            4. Potential external factors that could influence Bitcoin price movement
            5. Risk factors to monitor
            
            Focus on factual, recurring events and general market dynamics rather than speculation. Keep analysis concise (200-250 words).
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market analyst with expertise in macroeconomic factors and market sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.4
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating market sentiment: {str(e)}"
    
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
    
    def _generate_unified_analysis(self, analysis_data):
        """Generate comprehensive analysis in a single API call for consistency"""
        try:
            if not self.client:
                return "Error: OpenAI client not initialized"
            
            # Extract key data for clarity
            current_price = analysis_data.get('current_price', 0)
            hours_until_target = analysis_data.get('hours_until_target', 0)
            data_3m = analysis_data.get('data_3m', {})
            data_1w = analysis_data.get('data_1w', {})
            target_datetime_formatted = analysis_data.get('target_datetime_formatted', 'Friday 4PM ET')
            
            comprehensive_prompt = f"""
            You are a comprehensive Bitcoin analyst providing consistent analysis across technical, predictive, and market sentiment perspectives.
            
            === CRITICAL BITCOIN DATA ===
            CURRENT BITCOIN PRICE: ${current_price:,.2f}
            Time until {target_datetime_formatted}: {hours_until_target:.1f} hours
            
            3-MONTH TIMEFRAME (Period: {data_3m.get('start_date', 'N/A')} to {data_3m.get('end_date', 'N/A')}):
            - Current Price: ${current_price:,.2f}
            - Start Price: ${data_3m.get('start_price_3m', 0):,.2f} 
            - Period High: ${data_3m.get('high_3m', 0):,.2f}
            - Period Low: ${data_3m.get('low_3m', 0):,.2f}
            - Price Change: {data_3m.get('price_change_3m', 0):+.2f}%
            - Volatility: {data_3m.get('volatility_3m', 0):.1f}%
            
            1-WEEK TIMEFRAME (Period: {data_1w.get('start_date', 'N/A')} to {data_1w.get('end_date', 'N/A')}):
            - Current Price: ${current_price:,.2f}
            - Start Price: ${data_1w.get('start_price_1w', 0):,.2f}
            - Period High: ${data_1w.get('high_1w', 0):,.2f}
            - Period Low: ${data_1w.get('low_1w', 0):,.2f}
            - Price Change: {data_1w.get('price_change_1w', 0):+.2f}%
            - Volatility: {data_1w.get('volatility_1w', 0):.1f}%
            
            Technical Indicators:
            {json.dumps(analysis_data.get('indicators', {}), indent=2)}
            
            === RESPONSE FORMAT REQUIREMENTS ===
            Provide a comprehensive, CONSISTENT analysis divided into exactly these three sections with clear headers:
            
            [TECHNICAL_ANALYSIS_START]
            Technical Analysis Summary:
            - ALL PRICES must include $ sign (e.g., $115,287.50)
            - Price logic must be correct: if current > start = "increased/rose FROM start TO current"
            - Support = price levels BELOW ${current_price:,.2f}
            - Resistance = price levels ABOVE ${current_price:,.2f}
            - Make Buy/Sell/Hold recommendations **BOLD**
            - Include price action analysis, technical indicator interpretation, support/resistance levels
            - Keep analysis 200-300 words
            [TECHNICAL_ANALYSIS_END]
            
            [PRICE_PREDICTION_START]
            {target_datetime_formatted} Price Prediction:
            Based on the SAME technical analysis above, provide:
            1. Probability of Price Being HIGHER by {target_datetime_formatted}: [X]%
            2. Probability of Price Being LOWER by {target_datetime_formatted}: [Y]%
            3. Confidence Level: [Z]%
            4. Key technical factors supporting assessment (must align with technical analysis)
            5. Potential price targets (with $ signs)
            6. Trading recommendation: **Buy**/**Sell**/**Hold** (must be consistent with technical analysis)
            
            CRITICAL: X + Y must equal 100%. Use consistent reasoning with technical analysis section.
            [PRICE_PREDICTION_END]
            
            [MARKET_SENTIMENT_START]
            Market Sentiment & Key Events:
            Analyze general market sentiment and key events that may impact Bitcoin, considering the technical picture:
            1. General market sentiment for the upcoming week
            2. Key scheduled events (Fed meetings, economic data, options/futures expiry)
            3. Seasonal or cyclical patterns relevant to this time period
            4. External factors that could influence price movement
            5. Risk factors to monitor
            
            Keep analysis concise (200-250 words) and ensure it complements the technical analysis.
            [MARKET_SENTIMENT_END]
            
            === CONSISTENCY REQUIREMENTS ===
            - All three sections must tell the same story about Bitcoin's outlook
            - Recommendations across sections must align (**Buy**/**Sell**/**Hold**)
            - Price targets and probabilities must be consistent with technical analysis
            - Use the SAME fundamental assessment throughout all sections
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a professional Bitcoin analyst providing comprehensive, consistent analysis across technical, predictive, and market perspectives."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                max_tokens=2500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating unified analysis: {str(e)}"
    
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
            
            # Extract Market Sentiment
            sent_start = response.find("[MARKET_SENTIMENT_START]")
            sent_end = response.find("[MARKET_SENTIMENT_END]")
            if sent_start != -1 and sent_end != -1:
                sent_content = response[sent_start + len("[MARKET_SENTIMENT_START]"):sent_end].strip()
                sections['market_sentiment'] = sent_content
            
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
                    elif 'market sentiment' in line_lower or 'key events' in line_lower:
                        if current_section and current_content:
                            sections[current_section] = '\n'.join(current_content).strip()
                        current_section = 'market_sentiment'
                        current_content = []
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
                'market_sentiment': 'Unable to parse sentiment section'
            }
