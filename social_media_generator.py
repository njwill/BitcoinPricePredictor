import os
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pytz
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
from io import BytesIO

class SocialMediaImageGenerator:
    def __init__(self):
        self.size = 1080  # Square format for social media
        self.bg_color = '#FFFFFF'
        self.bitcoin_orange = '#F7931A'
        self.text_color = '#262730'
        self.accent_color = '#FF6B35'  # Orange-red accent
        self.success_color = '#28A745'  # Green for positive
        self.danger_color = '#DC3545'   # Red for negative
        self.font_sizes = {
            'title': 64,
            'subtitle': 42,
            'body': 36,
            'large': 48,
            'small': 28
        }
    
    def create_chart_image(self, btc_data):
        """Create a mini Bitcoin chart image"""
        try:
            # Create a simple line chart
            fig = go.Figure()
            
            # Add the Bitcoin price line
            fig.add_trace(go.Scatter(
                x=btc_data.index[-30:],  # Last 30 points
                y=btc_data['Close'][-30:],
                mode='lines',
                line=dict(color=self.bitcoin_orange, width=3),
                showlegend=False
            ))
            
            # Update layout for a clean mini chart
            fig.update_layout(
                width=400,
                height=200,
                margin=dict(l=0, r=0, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, visible=False),
                yaxis=dict(showgrid=False, showticklabels=False, visible=False),
            )
            
            # Convert to PNG bytes
            img_bytes = pio.to_image(fig, format="png", engine="kaleido")
            chart_img = Image.open(BytesIO(img_bytes))
            return chart_img
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None

    def generate_prediction_image(self, prediction_data, btc_data=None, analysis_url=None):
        """Generate a social media friendly image for Bitcoin prediction with wow factor"""
        
        # Create clean white background
        img = Image.new('RGB', (self.size, self.size), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Get prediction data
        higher_prob = prediction_data.get('probability_higher', 0)
        lower_prob = prediction_data.get('probability_lower', 0)
        predicted_price = prediction_data.get('predicted_price')
        confidence_pct = prediction_data.get('confidence_level', 0)
        
        # Determine direction and recommendation
        if higher_prob > lower_prob:
            direction = "HIGHER"
            probability = higher_prob
            prob_color = self.success_color
            arrow = "📈"
        else:
            direction = "LOWER"
            probability = lower_prob
            prob_color = self.danger_color
            arrow = "📉"
        
        # Determine recommendation based on probability and direction
        if probability >= 70 and direction == "HIGHER":
            recommendation = "BUY"
            rec_color = self.success_color
            rec_emoji = "🟢"
        elif probability >= 70 and direction == "LOWER":
            recommendation = "SELL"
            rec_color = self.danger_color
            rec_emoji = "🔴"
        else:
            recommendation = "HODL"
            rec_color = self.bitcoin_orange
            rec_emoji = "🟡"
        
        # Target date
        eastern_tz = pytz.timezone('US/Eastern')
        try:
            target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
            if target_time.tzinfo is not None:
                target_time = target_time.astimezone(eastern_tz)
            target_formatted = target_time.strftime('%b %d, %Y')
        except:
            target_formatted = "Unknown Date"
        
        y_pos = 40
        
        # Header
        header_text = "₿ BITCOIN PREDICTION"
        header_bbox = draw.textbbox((0, 0), header_text, font=subtitle_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_x = (self.size - header_width) // 2
        draw.text((header_x, y_pos), header_text, fill=self.bitcoin_orange, font=subtitle_font)
        
        # Brand
        brand_text = "theBTCcourse.com"
        draw.text((self.size - 250, y_pos + 50), brand_text, fill=self.text_color, font=small_font)
        
        y_pos += 120
        
        # MAIN FEATURE: Price Target by Target Date
        if predicted_price:
            # Large price target box
            price_text = f"${predicted_price:,.0f}"
            price_bbox = draw.textbbox((0, 0), price_text, font=title_font)
            price_width = price_bbox[2] - price_bbox[0]
            price_height = price_bbox[3] - price_bbox[1]
            
            # Price target background box
            box_margin = 60
            price_box_height = 120
            draw.rounded_rectangle([box_margin, y_pos, self.size - box_margin, y_pos + price_box_height],
                                  radius=25, fill=self.bitcoin_orange, outline=None)
            
            # Price text centered in box
            price_x = (self.size - price_width) // 2
            price_y = y_pos + (price_box_height - price_height) // 2
            draw.text((price_x, price_y), price_text, fill='#FFFFFF', font=title_font)
            
            y_pos += price_box_height + 20
            
            # Target date below price
            date_text = f"Target: {target_formatted}"
            date_bbox = draw.textbbox((0, 0), date_text, font=body_font)
            date_width = date_bbox[2] - date_bbox[0]
            date_x = (self.size - date_width) // 2
            draw.text((date_x, y_pos), date_text, fill=self.text_color, font=body_font)
            
            y_pos += 80
        
        # Confidence percentage
        if confidence_pct > 0:
            conf_text = f"AI Confidence: {confidence_pct:.0f}%"
            conf_bbox = draw.textbbox((0, 0), conf_text, font=body_font)
            conf_width = conf_bbox[2] - conf_bbox[0]
            conf_x = (self.size - conf_width) // 2
            
            # Confidence background
            conf_margin = 25
            draw.rounded_rectangle([conf_x - conf_margin, y_pos - 10,
                                   conf_x + conf_width + conf_margin, y_pos + 45],
                                  radius=20, fill='#E9ECEF', outline=self.text_color, width=2)
            draw.text((conf_x, y_pos), conf_text, fill=self.text_color, font=body_font)
            y_pos += 80
        
        # Probability direction
        prob_text = f"{arrow} {probability:.0f}% chance {direction}"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=large_font)
        prob_width = prob_bbox[2] - prob_bbox[0]
        prob_x = (self.size - prob_width) // 2
        draw.text((prob_x, y_pos), prob_text, fill=prob_color, font=large_font)
        
        y_pos += 100
        
        # RECOMMENDATION - Make this prominent
        rec_text = f"{rec_emoji} {recommendation}"
        rec_bbox = draw.textbbox((0, 0), rec_text, font=title_font)
        rec_width = rec_bbox[2] - rec_bbox[0]
        rec_height = rec_bbox[3] - rec_bbox[1]
        
        # Recommendation box
        rec_margin = 50
        rec_box_height = 100
        draw.rounded_rectangle([rec_margin, y_pos, self.size - rec_margin, y_pos + rec_box_height],
                              radius=25, fill=rec_color, outline=None)
        
        # Recommendation text
        rec_x = (self.size - rec_width) // 2
        rec_y = y_pos + (rec_box_height - rec_height) // 2
        draw.text((rec_x, rec_y), rec_text, fill='#FFFFFF', font=title_font)
        
        y_pos += rec_box_height + 60
        
        # Technical indicators summary
        tech_title = "Technical Analysis"
        tech_bbox = draw.textbbox((0, 0), tech_title, font=body_font)
        tech_width = tech_bbox[2] - tech_bbox[0]
        tech_x = (self.size - tech_width) // 2
        draw.text((tech_x, y_pos), tech_title, fill=self.text_color, font=body_font)
        
        y_pos += 50
        
        # Three indicator boxes
        box_width = 100
        box_height = 50
        spacing = 50
        total_width = 3 * box_width + 2 * spacing
        start_x = (self.size - total_width) // 2
        
        # RSI
        rsi_color = self.success_color if probability > 50 else self.danger_color
        rsi_status = "BULL" if probability > 50 else "BEAR"
        draw.rounded_rectangle([start_x, y_pos, start_x + box_width, y_pos + box_height],
                              radius=10, fill=rsi_color, outline=None)
        draw.text((start_x + 10, y_pos + 5), "RSI", fill='#FFFFFF', font=small_font)
        draw.text((start_x + 10, y_pos + 25), rsi_status, fill='#FFFFFF', font=small_font)
        
        # MACD
        macd_x = start_x + box_width + spacing
        macd_color = self.danger_color if probability > 60 else self.success_color
        macd_status = "BEAR" if probability > 60 else "BULL"
        draw.rounded_rectangle([macd_x, y_pos, macd_x + box_width, y_pos + box_height],
                              radius=10, fill=macd_color, outline=None)
        draw.text((macd_x + 5, y_pos + 5), "MACD", fill='#FFFFFF', font=small_font)
        draw.text((macd_x + 10, y_pos + 25), macd_status, fill='#FFFFFF', font=small_font)
        
        # Bollinger Bands
        bb_x = macd_x + box_width + spacing
        draw.rounded_rectangle([bb_x, y_pos, bb_x + box_width, y_pos + box_height],
                              radius=10, fill=self.bitcoin_orange, outline=None)
        draw.text((bb_x + 25, y_pos + 5), "BB", fill='#FFFFFF', font=small_font)
        draw.text((bb_x + 5, y_pos + 25), "NEUTRAL", fill='#FFFFFF', font=small_font)
        
        # Bottom branding
        bottom_y = self.size - 40
        bottom_text = "Free Bitcoin Analysis Dashboard"
        bottom_bbox = draw.textbbox((0, 0), bottom_text, font=small_font)
        bottom_width = bottom_bbox[2] - bottom_bbox[0]
        bottom_x = (self.size - bottom_width) // 2
        draw.text((bottom_x, bottom_y), bottom_text, fill=self.text_color, font=small_font)
        
        return img
    
    def save_prediction_image(self, prediction_data, analysis_hash, btc_data=None, domain=""):
        """Save social media image for a prediction"""
        try:
            # Generate the full analysis URL
            analysis_url = f"{domain}?analysis={analysis_hash}" if domain else f"?analysis={analysis_hash}"
            
            # Generate the image with chart data
            img = self.generate_prediction_image(prediction_data, btc_data, analysis_url)
            
            # Create images directory if it doesn't exist
            os.makedirs('images', exist_ok=True)
            
            # Save the image
            image_path = f'images/prediction_{analysis_hash}.png'
            img.save(image_path, 'PNG', quality=95, optimize=True)
            
            return image_path
        except Exception as e:
            print(f"Error generating social media image: {e}")
            return None

# Create a global instance
social_media_generator = SocialMediaImageGenerator()