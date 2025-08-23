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
        
        # Create image with gradient background
        img = Image.new('RGB', (self.size, self.size), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # Create gradient background
        for y in range(self.size):
            # Subtle gradient from white to very light orange
            ratio = y / self.size
            r = int(255 * (1 - ratio * 0.05))
            g = int(255 * (1 - ratio * 0.02))
            b = int(255 * (1 - ratio * 0.1))
            color = (r, g, b)
            draw.line([(0, y), (self.size, y)], fill=color)
        
        # Load fonts with better fallback
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_sizes['title'])
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_sizes['subtitle'])
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_sizes['body'])
            large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_sizes['large'])
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_sizes['small'])
        except:
            # Use PIL default font with different sizes
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default() 
            body_font = ImageFont.load_default()
            large_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Add header section with better spacing
        y_pos = 40
        
        # Bitcoin symbol with shadow effect
        bitcoin_text = "₿"
        bitcoin_bbox = draw.textbbox((0, 0), bitcoin_text, font=title_font)
        bitcoin_width = bitcoin_bbox[2] - bitcoin_bbox[0]
        bitcoin_x = 60
        
        # Draw shadow first
        draw.text((bitcoin_x + 2, y_pos + 2), bitcoin_text, fill='#00000020', font=title_font)
        # Draw main text
        draw.text((bitcoin_x, y_pos), bitcoin_text, fill=self.bitcoin_orange, font=title_font)
        
        # Title next to Bitcoin symbol
        title_text = "Bitcoin Analysis"
        title_x = bitcoin_x + bitcoin_width + 20
        draw.text((title_x + 1, y_pos + 1), title_text, fill='#00000020', font=subtitle_font)  # Shadow
        draw.text((title_x, y_pos), title_text, fill=self.text_color, font=subtitle_font)
        
        # Website branding on the right
        brand_text = "theBTCcourse.com"
        brand_bbox = draw.textbbox((0, 0), brand_text, font=small_font)
        brand_width = brand_bbox[2] - brand_bbox[0]
        brand_x = self.size - brand_width - 60
        draw.text((brand_x, y_pos + 15), brand_text, fill=self.bitcoin_orange, font=small_font)
        
        y_pos += 120
        
        # Add chart if data available
        if btc_data is not None:
            chart_img = self.create_chart_image(btc_data)
            if chart_img:
                # Resize and center the chart
                chart_resized = chart_img.resize((500, 150))
                chart_x = (self.size - 500) // 2
                img.paste(chart_resized, (chart_x, y_pos), chart_resized if chart_resized.mode == 'RGBA' else None)
                y_pos += 180
        
        # Extract and format prediction data
        eastern_tz = pytz.timezone('US/Eastern')
        try:
            target_time = datetime.fromisoformat(prediction_data.get('target_datetime', ''))
            if target_time.tzinfo is not None:
                target_time = target_time.astimezone(eastern_tz)
            target_formatted = target_time.strftime('%b %d, %Y')
        except:
            target_formatted = "Unknown Date"
        
        # Get probability values
        higher_prob = prediction_data.get('probability_higher', 0)
        lower_prob = prediction_data.get('probability_lower', 0)
        predicted_price = prediction_data.get('predicted_price')
        
        # Determine direction and probability
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
        
        # Create prominent prediction box
        box_height = 200
        box_margin = 80
        box_y = y_pos
        
        # Draw rounded rectangle background
        draw.rounded_rectangle([box_margin, box_y, self.size - box_margin, box_y + box_height], 
                              radius=20, fill='#FFFFFF', outline=prob_color, width=4)
        
        # Probability percentage (large and bold)
        prob_text = f"{probability:.0f}%"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=title_font)
        prob_width = prob_bbox[2] - prob_bbox[0]
        prob_x = (self.size - prob_width) // 2
        draw.text((prob_x, box_y + 30), prob_text, fill=prob_color, font=title_font)
        
        # Direction with arrow
        direction_text = f"{arrow} {direction}"
        direction_bbox = draw.textbbox((0, 0), direction_text, font=large_font)
        direction_width = direction_bbox[2] - direction_bbox[0]
        direction_x = (self.size - direction_width) // 2
        draw.text((direction_x, box_y + 100), direction_text, fill=prob_color, font=large_font)
        
        # Target date
        date_text = f"by {target_formatted}"
        date_bbox = draw.textbbox((0, 0), date_text, font=body_font)
        date_width = date_bbox[2] - date_bbox[0]
        date_x = (self.size - date_width) // 2
        draw.text((date_x, box_y + 150), date_text, fill=self.text_color, font=body_font)
        
        y_pos += box_height + 60
        
        # Add predicted price if available (in a separate accent box)
        if predicted_price:
            price_box_height = 80
            draw.rounded_rectangle([box_margin + 40, y_pos, self.size - box_margin - 40, y_pos + price_box_height],
                                  radius=15, fill=self.bitcoin_orange, outline=None)
            
            price_text = f"Target: ${predicted_price:,.0f}"
            price_bbox = draw.textbbox((0, 0), price_text, font=body_font)
            price_width = price_bbox[2] - price_bbox[0]
            price_x = (self.size - price_width) // 2
            draw.text((price_x, y_pos + 25), price_text, fill='#FFFFFF', font=body_font)
            y_pos += price_box_height + 40
        
        # Bottom branding with style
        bottom_text = "🔮 Free Bitcoin Analysis Dashboard"
        bottom_bbox = draw.textbbox((0, 0), bottom_text, font=small_font)
        bottom_width = bottom_bbox[2] - bottom_bbox[0]
        bottom_x = (self.size - bottom_width) // 2
        bottom_y = self.size - 60
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