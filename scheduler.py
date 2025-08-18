import schedule
import time
import threading
from datetime import datetime, timedelta
import pytz
import streamlit as st
from typing import Callable, Optional

class ScheduleManager:
    """Handles scheduling of automated updates for the Bitcoin analysis"""
    
    def __init__(self):
        self.eastern_tz = pytz.timezone('US/Eastern')
        self.scheduler_thread = None
        self.is_running = False
        
    def schedule_weekly_update(self, update_function: Callable):
        """
        Schedule the weekly update for Monday at 9:30 AM Eastern Time
        
        Args:
            update_function: Function to call for updates
        """
        try:
            # Clear any existing schedules
            schedule.clear()
            
            # Schedule for Monday at 9:30 AM Eastern
            schedule.every().monday.at("09:30").do(self._run_update, update_function)
            
            st.info("✅ Weekly update scheduled for Monday at 9:30 AM Eastern Time")
            
        except Exception as e:
            st.error(f"Error scheduling updates: {str(e)}")
    
    def _run_update(self, update_function: Callable):
        """Execute the scheduled update"""
        try:
            current_time = datetime.now(self.eastern_tz)
            st.info(f"🔄 Running scheduled update at {current_time.strftime('%Y-%m-%d %H:%M:%S')} ET")
            
            # Execute the update function
            update_function()
            
            st.success(f"✅ Scheduled update completed successfully")
            
        except Exception as e:
            st.error(f"Error during scheduled update: {str(e)}")
    
    def start_scheduler(self):
        """Start the background scheduler thread"""
        if not self.is_running:
            self.is_running = True
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            st.info("🚀 Background scheduler started")
    
    def stop_scheduler(self):
        """Stop the background scheduler"""
        self.is_running = False
        schedule.clear()
        st.info("⏹️ Background scheduler stopped")
    
    def _run_scheduler(self):
        """Run the scheduler in a background thread"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                st.error(f"Scheduler error: {str(e)}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def get_next_update_time(self) -> Optional[datetime]:
        """
        Get the next scheduled update time
        
        Returns:
            datetime: Next update time in Eastern timezone
        """
        try:
            current_time = datetime.now(self.eastern_tz)
            
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
            
            return next_update
            
        except Exception as e:
            st.error(f"Error calculating next update time: {str(e)}")
            return None
    
    def time_until_next_update(self) -> Optional[timedelta]:
        """
        Calculate time remaining until next update
        
        Returns:
            timedelta: Time remaining until next update
        """
        try:
            next_update = self.get_next_update_time()
            if next_update:
                current_time = datetime.now(self.eastern_tz)
                return next_update - current_time
            return None
            
        except Exception as e:
            st.error(f"Error calculating time until update: {str(e)}")
            return None
    
    def should_update_now(self) -> bool:
        """
        Check if an update should be triggered now
        
        Returns:
            bool: True if update should run
        """
        try:
            current_time = datetime.now(self.eastern_tz)
            
            # Check if it's Monday between 9:30 and 10:00 AM
            if current_time.weekday() == 0:  # Monday
                if current_time.hour == 9 and current_time.minute >= 30:
                    return True
                elif current_time.hour == 10 and current_time.minute == 0:
                    return True
            
            return False
            
        except Exception as e:
            st.error(f"Error checking update timing: {str(e)}")
            return False
    
    def format_next_update_display(self) -> str:
        """
        Format next update time for display
        
        Returns:
            str: Formatted next update time
        """
        try:
            next_update = self.get_next_update_time()
            time_until = self.time_until_next_update()
            
            if next_update and time_until:
                next_update_str = next_update.strftime('%A, %B %d at %I:%M %p ET')
                
                days = time_until.days
                hours, remainder = divmod(time_until.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                
                if days > 0:
                    time_until_str = f"{days}d {hours}h {minutes}m"
                elif hours > 0:
                    time_until_str = f"{hours}h {minutes}m"
                else:
                    time_until_str = f"{minutes}m"
                
                return f"{next_update_str} (in {time_until_str})"
            
            return "Update time not available"
            
        except Exception as e:
            return f"Error calculating update time: {str(e)}"
    
    def get_update_history(self) -> list:
        """
        Get history of recent updates (placeholder for future implementation)
        
        Returns:
            list: List of recent update timestamps
        """
        # This would be implemented with a database or file storage
        # For now, return empty list
        return []
    
    def manual_trigger_available(self) -> bool:
        """
        Check if manual trigger is available (e.g., not too recently triggered)
        
        Returns:
            bool: True if manual trigger is allowed
        """
        try:
            # Allow manual trigger if last automated update was more than 1 hour ago
            # This is a simplified check - in production, you'd track actual update times
            current_time = datetime.now(self.eastern_tz)
            
            # For now, always allow manual trigger
            # In production, implement proper rate limiting
            return True
            
        except Exception as e:
            st.error(f"Error checking manual trigger availability: {str(e)}")
            return False
