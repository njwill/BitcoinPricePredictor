import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import streamlit as st

# the newest OpenAI model is "gpt-5" which was released August 7, 2024.
# Updated per user request to use GPT-5
from openai import OpenAI

class AIAnalyzer:
    """Handles AI-powered analysis using OpenAI GPT-5"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_comprehensive_analysis(self, data_3m, data_1w, indicators_3m, indicators_1w, current_price):
        """
        Generate comprehensive AI analysis including technical analysis, predictions, and market sentiment
        
        Args:
            data_3m: 3-month Bitcoin price data
            data_1w: 1-week Bitcoin price data  
            indicators_3m: Technical indicators for 3-month data
            indicators_1w: Technical indicators for 1-week data
            current_price: Current Bitcoin price
            
        Returns:
            Dictionary with analysis results
        """
        if not self.client:
            return {"error": "OpenAI client not initialized"}
        
        try:
            # Prepare data summary for AI analysis
            analysis_data = self._prepare_analysis_data(data_3m, data_1w, indicators_3m, indicators_1w, current_price)
            
            # Generate technical analysis
            technical_analysis = self._generate_technical_analysis(analysis_data)
            
            # Generate price prediction
            price_prediction = self._generate_price_prediction(analysis_data)
            
            # Generate market sentiment analysis
            market_sentiment = self._generate_market_sentiment(analysis_data)
            
            # Extract probabilities from prediction
            probabilities = self._extract_probabilities(price_prediction)
            
            return {
                'technical_summary': technical_analysis,
                'price_prediction': price_prediction,
                'market_sentiment': market_sentiment,
                'probabilities': probabilities,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.error(f"Error generating AI analysis: {str(e)}")
            return {"error": str(e)}
    
    def _prepare_analysis_data(self, data_3m, data_1w, indicators_3m, indicators_1w, current_price):
        """Prepare and summarize data for AI analysis"""
        try:
            eastern_tz = pytz.timezone('US/Eastern')
            current_time = datetime.now(eastern_tz)
            friday_4pm = self._get_next_friday_4pm(current_time)
            
            # 3-month data summary
            data_3m_summary = {
                'period': '3 months',
                'current_price': current_price,
                'high_3m': float(data_3m['High'].max()),
                'low_3m': float(data_3m['Low'].min()),
                'price_change_3m': float((current_price - data_3m['Close'].iloc[0]) / data_3m['Close'].iloc[0] * 100),
                'volatility_3m': float(data_3m['Close'].pct_change().std() * np.sqrt(252) * 100),
                'avg_volume_3m': float(data_3m['Volume'].mean())
            }
            
            # 1-week data summary
            data_1w_summary = {
                'period': '1 week',
                'high_1w': float(data_1w['High'].max()),
                'low_1w': float(data_1w['Low'].min()),
                'price_change_1w': float((current_price - data_1w['Close'].iloc[0]) / data_1w['Close'].iloc[0] * 100),
                'volatility_1w': float(data_1w['Close'].pct_change().std() * np.sqrt(365) * 100),
                'avg_volume_1w': float(data_1w['Volume'].mean())
            }
            
            # Technical indicators summary
            indicators_summary = self._summarize_indicators(indicators_3m, indicators_1w)
            
            return {
                'current_time': current_time.isoformat(),
                'target_time': friday_4pm.isoformat(),
                'hours_until_target': (friday_4pm - current_time).total_seconds() / 3600,
                'data_3m': data_3m_summary,
                'data_1w': data_1w_summary,
                'indicators': indicators_summary,
                'current_price': current_price
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
            prompt = f"""
            You are an expert Bitcoin technical analyst. Analyze the following Bitcoin price data and technical indicators to provide a comprehensive technical analysis.
            
            Data Summary:
            {json.dumps(analysis_data, indent=2)}
            
            Provide a detailed technical analysis covering:
            1. Price action analysis for both 3-month and 1-week timeframes
            2. Technical indicator interpretation (RSI, MACD, Bollinger Bands, EMA)
            3. Support and resistance levels
            4. Current market structure and trend analysis
            5. Key technical patterns or formations
            
            Keep the analysis professional, concise (200-300 words), and focused solely on technical aspects of the charts provided.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a professional Bitcoin technical analyst with expertise in chart analysis and technical indicators."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=400,
                reasoning_effort="minimal"
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating technical analysis: {str(e)}"
    
    def _generate_price_prediction(self, analysis_data):
        """Generate price prediction with probabilities"""
        try:
            prompt = f"""
            Based solely on the Bitcoin chart analysis and technical indicators provided, calculate the probability of Bitcoin being higher or lower by Friday at 4:00 PM Eastern Time.
            
            Current Analysis Data:
            {json.dumps(analysis_data, indent=2)}
            
            Time until target: {analysis_data.get('hours_until_target', 0):.1f} hours
            
            Provide:
            1. Probability assessment for price being HIGHER by Friday 4PM ET (0-100%)
            2. Probability assessment for price being LOWER by Friday 4PM ET (0-100%)
            3. Your confidence level in this analysis (0-100%)
            4. Key technical factors supporting your probability assessment
            5. Potential price targets or ranges
            
            Base your analysis ONLY on the technical chart data provided. Be specific with probability percentages and explain your reasoning.
            
            Format your response to include clear probability statements.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a quantitative Bitcoin analyst specializing in probability-based price predictions using technical analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=500,
                reasoning_effort="minimal"
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating price prediction: {str(e)}"
    
    def _generate_market_sentiment(self, analysis_data):
        """Generate market sentiment and key events analysis"""
        try:
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
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency market analyst with expertise in macroeconomic factors and market sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=400,
                reasoning_effort="minimal"
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
