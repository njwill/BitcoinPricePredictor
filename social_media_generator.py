import os
from PIL import Image, ImageDraw, ImageFont
import qrcode
from qrcode import QRCode
from datetime import datetime
import pytz

class SocialMediaImageGenerator:
    def __init__(self):
        self.size = 1080  # Square format for social media
        self.bg_color = '#FFFFFF'
        self.bitcoin_orange = '#F7931A'
        self.text_color = '#262730'
        self.font_sizes = {
            'title': 72,
            'subtitle': 48,
            'body': 36,
            'small': 28
        }
    
    def generate_prediction_image(self, prediction_data, analysis_url):
        """Generate a social media friendly image for Bitcoin prediction"""
        
        # Create image
        img = Image.new('RGB', (self.size, self.size), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to load custom fonts, fallback to default if not available
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.font_sizes['title'])
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_sizes['subtitle'])
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_sizes['body'])
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.font_sizes['small'])
        except:
            # Fallback to default fonts
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            body_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        # Current Y position for text
        y_pos = 60
        
        # Add Bitcoin symbol and title
        title_text = "₿ Bitcoin Analysis"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.size - title_width) // 2
        draw.text((title_x, y_pos), title_text, fill=self.bitcoin_orange, font=title_font)
        y_pos += 100
        
        # Add website branding
        brand_text = "theBTCcourse.com"
        brand_bbox = draw.textbbox((0, 0), brand_text, font=subtitle_font)
        brand_width = brand_bbox[2] - brand_bbox[0]
        brand_x = (self.size - brand_width) // 2
        draw.text((brand_x, y_pos), brand_text, fill=self.text_color, font=subtitle_font)
        y_pos += 80
        
        # Add divider line
        line_margin = 200
        draw.line([(line_margin, y_pos), (self.size - line_margin, y_pos)], fill=self.bitcoin_orange, width=3)
        y_pos += 60
        
        # Extract prediction data
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
            prob_color = '#28a745'  # Green for higher
        else:
            direction = "LOWER"
            probability = lower_prob
            prob_color = '#dc3545'  # Red for lower
        
        # Add prediction text
        prob_text = f"{probability:.0f}% chance of being"
        prob_bbox = draw.textbbox((0, 0), prob_text, font=body_font)
        prob_width = prob_bbox[2] - prob_bbox[0]
        prob_x = (self.size - prob_width) // 2
        draw.text((prob_x, y_pos), prob_text, fill=self.text_color, font=body_font)
        y_pos += 50
        
        # Add direction (HIGHER/LOWER) in color
        direction_bbox = draw.textbbox((0, 0), direction, font=title_font)
        direction_width = direction_bbox[2] - direction_bbox[0]
        direction_x = (self.size - direction_width) // 2
        draw.text((direction_x, y_pos), direction, fill=prob_color, font=title_font)
        y_pos += 100
        
        # Add target date
        date_text = f"by {target_formatted}"
        date_bbox = draw.textbbox((0, 0), date_text, font=body_font)
        date_width = date_bbox[2] - date_bbox[0]
        date_x = (self.size - date_width) // 2
        draw.text((date_x, y_pos), date_text, fill=self.text_color, font=body_font)
        y_pos += 60
        
        # Add predicted price if available
        if predicted_price:
            price_text = f"Predicted: ${predicted_price:,.0f}"
            price_bbox = draw.textbbox((0, 0), price_text, font=body_font)
            price_width = price_bbox[2] - price_bbox[0]
            price_x = (self.size - price_width) // 2
            draw.text((price_x, y_pos), price_text, fill=self.text_color, font=body_font)
            y_pos += 80
        
        # Add QR code for analysis URL (smaller, in corner)
        if analysis_url:
            qr = QRCode(version=1, box_size=4, border=1)
            qr.add_data(analysis_url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color='black', back_color='white')
            qr_size = 120
            qr_img = qr_img.resize((qr_size, qr_size))
            
            # Position QR code in bottom right
            qr_x = self.size - qr_size - 40
            qr_y = self.size - qr_size - 40
            img.paste(qr_img, (qr_x, qr_y))
            
            # Add "Scan for details" text near QR code
            qr_text = "Scan for full analysis"
            qr_text_bbox = draw.textbbox((0, 0), qr_text, font=small_font)
            qr_text_width = qr_text_bbox[2] - qr_text_bbox[0]
            qr_text_x = qr_x + (qr_size - qr_text_width) // 2
            qr_text_y = qr_y - 35
            draw.text((qr_text_x, qr_text_y), qr_text, fill=self.text_color, font=small_font)
        
        # Add bottom branding
        bottom_text = "Free Bitcoin Analysis Dashboard"
        bottom_bbox = draw.textbbox((0, 0), bottom_text, font=small_font)
        bottom_width = bottom_bbox[2] - bottom_bbox[0]
        bottom_x = (self.size - bottom_width) // 2
        bottom_y = self.size - 100
        draw.text((bottom_x, bottom_y), bottom_text, fill=self.text_color, font=small_font)
        
        return img
    
    def save_prediction_image(self, prediction_data, analysis_hash, domain=""):
        """Save social media image for a prediction"""
        try:
            # Generate the full analysis URL
            analysis_url = f"{domain}?analysis={analysis_hash}" if domain else f"?analysis={analysis_hash}"
            
            # Generate the image
            img = self.generate_prediction_image(prediction_data, analysis_url)
            
            # Create images directory if it doesn't exist
            os.makedirs('images', exist_ok=True)
            
            # Save the image
            image_path = f'images/prediction_{analysis_hash}.png'
            img.save(image_path, 'PNG', quality=95)
            
            return image_path
        except Exception as e:
            print(f"Error generating social media image: {e}")
            return None

# Create a global instance
social_media_generator = SocialMediaImageGenerator()