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
        
        # Create image with dynamic gradient background
        img = Image.new('RGB', (self.size, self.size), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Create dynamic gradient background based on prediction
        higher_prob = prediction_data.get('probability_higher', 0)
        lower_prob = prediction_data.get('probability_lower', 0)
        
        # Determine colors based on prediction
        if higher_prob > lower_prob:
            gradient_color = (40, 167, 69)  # Green tint for bullish
        else:
            gradient_color = (220, 53, 69)   # Red tint for bearish
        
        # Create diagonal gradient
        for y in range(self.size):
            for x in range(self.size):
                # Create diagonal pattern
                distance = (x + y) / (2 * self.size)
                alpha = 0.03 * distance
                
                r = int(255 * (1 - alpha) + gradient_color[0] * alpha)
                g = int(255 * (1 - alpha) + gradient_color[1] * alpha)
                b = int(255 * (1 - alpha) + gradient_color[2] * alpha)
                
                if y % 4 == 0:  # Only draw every 4th line for performance
                    draw.line([(x, y), (x, y)], fill=(r, g, b))
        
        # Load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 68)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 44)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 52)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        y_pos = 50
        
        # Clean header without overlapping boxes
        header_text = "₿ BITCOIN PREDICTION"
        header_bbox = draw.textbbox((0, 0), header_text, font=subtitle_font)
        header_width = header_bbox[2] - header_bbox[0]
        header_x = (self.size - header_width) // 2
        
        # Draw header with glow effect
        for offset in range(3, 0, -1):
            alpha = 50 + (3-offset) * 30
            glow_color = f'#{self.bitcoin_orange[1:]}{alpha:02x}'[:-2] + f'{alpha:02x}'
            draw.text((header_x + offset, y_pos + offset), header_text, fill='#00000030', font=subtitle_font)
        draw.text((header_x, y_pos), header_text, fill=self.bitcoin_orange, font=subtitle_font)
        
        # Brand on the side
        brand_text = "theBTCcourse.com"
        draw.text((self.size - 280, y_pos + 50), brand_text, fill=self.text_color, font=small_font)
        
        y_pos += 110
        
        # Get prediction data
        predicted_price = prediction_data.get('predicted_price')
        
        if higher_prob > lower_prob:
            direction = "HIGHER"
            probability = higher_prob
            prob_color = self.success_color
            arrow = "🚀"
            trend_emoji = "📈"
        else:
            direction = "LOWER"
            probability = lower_prob
            prob_color = self.danger_color
            arrow = "📉" 
            trend_emoji = "⚡"
        
        # Target date
        eastern_tz = pytz.timezone('US/Eastern')
        try:
            target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
            if target_time.tzinfo is not None:
                target_time = target_time.astimezone(eastern_tz)
            target_formatted = target_time.strftime('%b %d, %Y')
        except:
            target_formatted = "Unknown Date"
        
        # Main prediction section with modern design
        # Large percentage in a circle
        circle_radius = 80
        circle_center_x = self.size // 2
        circle_center_y = y_pos + circle_radius + 20
        
        # Draw main circle with gradient border
        for thickness in range(8, 0, -1):
            circle_color = prob_color if thickness <= 4 else '#FFFFFF'
            draw.ellipse([circle_center_x - circle_radius - thickness, circle_center_y - circle_radius - thickness,
                         circle_center_x + circle_radius + thickness, circle_center_y + circle_radius + thickness],
                        fill=None, outline=circle_color, width=2)
        
        # Fill circle
        draw.ellipse([circle_center_x - circle_radius, circle_center_y - circle_radius,
                     circle_center_x + circle_radius, circle_center_y + circle_radius],
                    fill='#FFFFFF', outline=prob_color, width=6)
        
        # Percentage text in circle
        prob_text = f"{probability:.0f}%"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=title_font)
        prob_width = prob_bbox[2] - prob_bbox[0]
        prob_height = prob_bbox[3] - prob_bbox[1]
        prob_x = circle_center_x - prob_width // 2
        prob_y = circle_center_y - prob_height // 2
        draw.text((prob_x, prob_y), prob_text, fill=prob_color, font=title_font)
        
        y_pos = circle_center_y + circle_radius + 40
        
        # Direction with cool styling
        direction_text = f"{arrow} {direction} {trend_emoji}"
        direction_bbox = draw.textbbox((0, 0), direction_text, font=large_font)
        direction_width = direction_bbox[2] - direction_bbox[0]
        direction_x = (self.size - direction_width) // 2
        
        # Draw direction with shadow
        draw.text((direction_x + 2, y_pos + 2), direction_text, fill='#00000040', font=large_font)
        draw.text((direction_x, y_pos), direction_text, fill=prob_color, font=large_font)
        
        y_pos += 80
        
        # Target date in stylish box
        date_text = f"Target: {target_formatted}"
        date_bbox = draw.textbbox((0, 0), date_text, font=body_font)
        date_width = date_bbox[2] - date_bbox[0]
        date_x = (self.size - date_width) // 2
        
        # Date background
        date_bg_margin = 20
        draw.rounded_rectangle([date_x - date_bg_margin, y_pos - 10, 
                               date_x + date_width + date_bg_margin, y_pos + 35],
                              radius=20, fill='#F8F9FA', outline=self.text_color, width=2)
        draw.text((date_x, y_pos), date_text, fill=self.text_color, font=body_font)
        
        y_pos += 70
        
        # Predicted price in bitcoin orange box
        if predicted_price:
            price_text = f"Price Target: ${predicted_price:,.0f}"
            price_bbox = draw.textbbox((0, 0), price_text, font=body_font)
            price_width = price_bbox[2] - price_bbox[0]
            price_x = (self.size - price_width) // 2
            
            price_bg_margin = 25
            draw.rounded_rectangle([price_x - price_bg_margin, y_pos - 10,
                                   price_x + price_width + price_bg_margin, y_pos + 40],
                                  radius=20, fill=self.bitcoin_orange, outline=None)
            draw.text((price_x, y_pos), price_text, fill='#FFFFFF', font=body_font)
            y_pos += 80
        
        # Technical indicators section (fake data for visual appeal)
        tech_y = y_pos + 20
        tech_title = "📊 TECHNICAL SIGNALS"
        tech_title_bbox = draw.textbbox((0, 0), tech_title, font=body_font)
        tech_title_width = tech_title_bbox[2] - tech_title_bbox[0]
        tech_title_x = (self.size - tech_title_width) // 2
        draw.text((tech_title_x, tech_y), tech_title, fill=self.text_color, font=body_font)
        
        # Create three indicator boxes
        indicator_y = tech_y + 50
        box_width = 120
        box_height = 60
        spacing = 40
        total_width = 3 * box_width + 2 * spacing
        start_x = (self.size - total_width) // 2
        
        # RSI
        rsi_color = self.success_color if probability > 50 else self.danger_color
        rsi_status = "BULLISH" if probability > 50 else "BEARISH"
        draw.rounded_rectangle([start_x, indicator_y, start_x + box_width, indicator_y + box_height],
                              radius=15, fill=rsi_color, outline=None)
        draw.text((start_x + 15, indicator_y + 10), "RSI", fill='#FFFFFF', font=small_font)
        draw.text((start_x + 10, indicator_y + 30), rsi_status[:4], fill='#FFFFFF', font=small_font)
        
        # MACD  
        macd_x = start_x + box_width + spacing
        macd_color = self.danger_color if probability > 50 else self.success_color
        macd_status = "BEAR" if probability > 50 else "BULL"
        draw.rounded_rectangle([macd_x, indicator_y, macd_x + box_width, indicator_y + box_height],
                              radius=15, fill=macd_color, outline=None)
        draw.text((macd_x + 10, indicator_y + 10), "MACD", fill='#FFFFFF', font=small_font)
        draw.text((macd_x + 15, indicator_y + 30), macd_status, fill='#FFFFFF', font=small_font)
        
        # BB
        bb_x = macd_x + box_width + spacing
        bb_color = self.bitcoin_orange
        draw.rounded_rectangle([bb_x, indicator_y, bb_x + box_width, indicator_y + box_height],
                              radius=15, fill=bb_color, outline=None)
        draw.text((bb_x + 25, indicator_y + 10), "BB", fill='#FFFFFF', font=small_font)
        draw.text((bb_x + 10, indicator_y + 30), "SQUEEZE", fill='#FFFFFF', font=small_font)
        
        # Bottom branding with modern style
        bottom_y = self.size - 50
        bottom_text = "🎯 Free Bitcoin Analysis • AI Powered"
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