import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pytz
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from io import BytesIO

class SocialMediaTextGenerator:
    def __init__(self):
        pass  # No image properties needed anymore
    
    def generate_tweet_text(self, prediction_data, analysis_url=None):
        """Generate tweetable text with key metrics"""
        try:
            # Get prediction data
            higher_prob = prediction_data.get('probability_higher', 0)
            lower_prob = prediction_data.get('probability_lower', 0)
            predicted_price = prediction_data.get('predicted_price')
            confidence_pct = prediction_data.get('confidence_level', 0)
            price_confidence_pct = prediction_data.get('price_confidence_level', 0)
            move_percentage = prediction_data.get('move_percentage', 0)
            
            # Determine direction and recommendation
            if higher_prob > lower_prob:
                direction = "HIGHER"
                probability = higher_prob
                arrow = "📈"
            else:
                direction = "LOWER"
                probability = lower_prob
                arrow = "📉"
            
            # Determine recommendation based on probability and direction
            if probability >= 70 and direction == "HIGHER":
                recommendation = "🟢 BUY"
            elif probability >= 70 and direction == "LOWER":
                recommendation = "🔴 SELL"
            else:
                recommendation = "🟡 HODL"
            
            # Target date
            eastern_tz = pytz.timezone('US/Eastern')
            try:
                target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
                if target_time.tzinfo is not None:
                    target_time = target_time.astimezone(eastern_tz)
                target_formatted = target_time.strftime('%b %d, %Y')
            except:
                target_formatted = "Unknown Date"
            
            # Build tweet text
            tweet_parts = []
            
            # Main prediction
            if predicted_price:
                tweet_parts.append(f"🔮 #Bitcoin Prediction: ${predicted_price:,.0f} by {target_formatted}")
            
            # Direction and probability with move %
            direction_text = f"{arrow} {probability:.0f}% chance {direction}"
            if move_percentage != 0:
                move_sign = "+" if move_percentage > 0 else ""
                direction_text += f" ({move_sign}{move_percentage:.1f}%)"
            tweet_parts.append(direction_text)
            
            # Recommendation
            tweet_parts.append(f"Signal: {recommendation}")
            
            # Confidence levels
            if confidence_pct > 0:
                tweet_parts.append(f"📊 AI Confidence: {confidence_pct:.0f}%")
            
            if price_confidence_pct > 0:
                tweet_parts.append(f"🎯 Price Target Confidence: {price_confidence_pct:.0f}%")
            
            # Add analysis link if provided
            if analysis_url:
                tweet_parts.append(f"📈 Full Analysis: {analysis_url}")
            
            return "\n".join(tweet_parts)
            
        except Exception as e:
            return f"Bitcoin prediction analysis failed: {e}"
    
    def generate_shareable_text(self, prediction_data, analysis_hash, domain=""):
        """Generate shareable tweet text for a prediction"""
        try:
            # Generate the full analysis URL with complete domain
            if domain and not domain.endswith('/'):
                domain += '/'
            analysis_url = f"{domain}?analysis={analysis_hash}" if domain else f"?analysis={analysis_hash}"
            
            # Generate the tweet text
            tweet_text = self.generate_tweet_text(prediction_data, analysis_url)
            
            return tweet_text
        except Exception as e:
            print(f"Error generating shareable text: {e}")
            return f"Bitcoin analysis complete - check the full analysis for details."

# Create a global instance
social_media_generator = SocialMediaTextGenerator()