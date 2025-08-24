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
            
            # Target date and time
            eastern_tz = pytz.timezone('US/Eastern')
            try:
                target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
                if target_time.tzinfo is not None:
                    target_time = target_time.astimezone(eastern_tz)
                target_formatted = target_time.strftime('%b %d, %Y')
                target_time_formatted = target_time.strftime('%I:%M %p ET')
            except:
                target_formatted = "Unknown Date"
                target_time_formatted = "Unknown Time"
            
            # Build tweet text with new format
            tweet_parts = []
            
            # Opening line
            tweet_parts.append("I just ran some advanced #Bitcoin TA and it says...")
            
            # Main prediction with time
            if predicted_price:
                tweet_parts.append(f"🔮 ${predicted_price:,.0f} by {target_formatted} ({target_time_formatted})")
            
            # Direction and probability - specify it will be HIGHER/LOWER than target date/time
            direction_text = f"{arrow} {probability:.0f}% chance it will be {direction} than {target_formatted} {target_time_formatted}"
            tweet_parts.append(direction_text)
            
            # Confidence levels
            if confidence_pct > 0:
                tweet_parts.append(f"📊 Confidence Level: {confidence_pct:.0f}%")
            
            # Signal after confidence
            tweet_parts.append(f"🚦 Signal: {recommendation}")
            
            # Add analysis link if provided
            if analysis_url:
                tweet_parts.append(f"📈 Full Analysis: {analysis_url}")
            
            return "\n".join(tweet_parts)
            
        except Exception as e:
            return f"Bitcoin prediction analysis failed: {e}"
    
    def generate_shareable_text(self, prediction_data, analysis_hash, domain=""):
        """Generate shareable tweet text for a prediction"""
        try:
            # Hardcode the analysis URL to the specific domain
            analysis_url = f"https://predict.thebtccourse.com/?analysis={analysis_hash}"
            
            # Generate the tweet text
            tweet_text = self.generate_tweet_text(prediction_data, analysis_url)
            
            return tweet_text
        except Exception as e:
            print(f"Error generating shareable text: {e}")
            return f"Bitcoin analysis complete - check the full analysis for details."

# Create a global instance
social_media_generator = SocialMediaTextGenerator()