import psycopg2
import json
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import streamlit as st
from typing import Dict, Optional, Tuple
import yfinance as yf
import pytz

class AnalysisDatabase:
    """Handles database operations for storing and retrieving complete Bitcoin analyses"""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.ensure_tables_exist()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def ensure_tables_exist(self):
        """Create database tables if they don't exist"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Create analyses table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS bitcoin_analyses (
                            id SERIAL PRIMARY KEY,
                            analysis_hash VARCHAR(64) UNIQUE NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE NOT NULL,
                            target_datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                            current_price_at_prediction DECIMAL(12,2) NOT NULL,
                            predicted_price DECIMAL(12,2),
                            probability_higher INTEGER,
                            probability_lower INTEGER,
                            confidence_level INTEGER,
                            technical_summary TEXT,
                            prediction_reasoning TEXT,
                            full_ai_analysis TEXT,
                            actual_price DECIMAL(12,2),
                            accuracy_calculated BOOLEAN DEFAULT FALSE,
                            chart_config JSONB,
                            btc_3m_data JSONB NOT NULL,
                            btc_1w_data JSONB NOT NULL,
                            indicators_3m JSONB NOT NULL,
                            indicators_1w JSONB NOT NULL
                        )
                    """)
                    
                    # Create index on hash for fast lookups
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_analysis_hash 
                        ON bitcoin_analyses(analysis_hash)
                    """)
                    
                    # Create index on created_at for chronological queries
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_analysis_created_at 
                        ON bitcoin_analyses(created_at DESC)
                    """)
                    
                    conn.commit()
        except Exception as e:
            st.error(f"Database initialization error: {str(e)}")
    
    def generate_analysis_hash(self, prediction_data: dict, btc_3m: pd.DataFrame, btc_1w: pd.DataFrame) -> str:
        """Generate unique hash for analysis based on data and timestamp"""
        # Create a string that uniquely identifies this analysis
        hash_input = f"{prediction_data.get('prediction_timestamp')}"
        hash_input += f"{prediction_data.get('target_datetime')}"
        hash_input += f"{prediction_data.get('current_price')}"
        hash_input += f"{len(btc_3m)}_{len(btc_1w)}"  # Data length as additional uniqueness
        
        # Generate SHA-256 hash and take first 12 characters for URL friendliness
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    def serialize_dataframe(self, df: pd.DataFrame) -> dict:
        """Convert DataFrame to JSON-serializable format"""
        if df.empty:
            return {}
        
        # Convert to dictionary with proper handling of datetime index and values
        # Handle datetime index
        if hasattr(df.index, 'strftime'):
            # Ensure timezone-aware datetime index and consistent format
            if df.index.tz is None:
                # If naive, assume UTC for consistency
                df_index = df.index.tz_localize('UTC')
            else:
                df_index = df.index
            index_list = df_index.strftime('%Y-%m-%d %H:%M:%S%z').tolist()
        else:
            index_list = [str(x) for x in df.index.tolist()]  # Convert to strings
        
        # Convert data to records and handle any remaining timestamp issues
        data_records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                value = row[col]
                # Convert timestamps and other non-serializable types to strings
                if pd.isna(value):
                    record[col] = None
                elif hasattr(value, 'isoformat'):  # datetime-like objects
                    record[col] = value.isoformat() if value.isoformat else str(value)
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value)  # Convert numpy types to Python types
                else:
                    record[col] = value
            data_records.append(record)
        
        result = {
            'data': data_records,
            'index': index_list,
            'columns': df.columns.tolist()
        }
        return result
    
    def serialize_indicators(self, indicators: Dict) -> dict:
        """Convert indicators dictionary to JSON-serializable format"""
        serialized = {}
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                # Convert Series to list, handling NaN values
                serialized[key] = value.fillna(value=0).tolist()  # Fill NaN with 0
            elif isinstance(value, np.ndarray):
                # Convert numpy array to list, handling NaN values  
                series_val = pd.Series(value)
                serialized[key] = series_val.fillna(value=0).tolist()  # Fill NaN with 0
            else:
                serialized[key] = value
        return serialized
    
    def deserialize_dataframe(self, data: dict) -> pd.DataFrame:
        """Convert stored JSON back to DataFrame"""
        if not data or 'data' not in data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data['data'])
        if 'index' in data and data['index']:
            try:
                # Try to parse dates with explicit format, fallback to automatic parsing
                df.index = pd.to_datetime(data['index'], format='%Y-%m-%d %H:%M:%S%z', errors='coerce')
                # If that fails, try without timezone
                if df.index.isna().any():
                    df.index = pd.to_datetime(data['index'], errors='coerce')
            except Exception as e:
                print(f"Date parsing error: {e}")
                # Fallback: create a simple numeric index
                df.index = range(len(df))
        return df
    
    def deserialize_indicators(self, data: dict, df_index) -> Dict:
        """Convert stored JSON back to indicators dictionary with proper pandas Series"""
        if not data:
            return {}
        
        indicators = {}
        for key, value in data.items():
            if isinstance(value, list):
                # Convert back to pandas Series with the dataframe index
                indicators[key] = pd.Series(value, index=df_index)
            else:
                indicators[key] = value
        return indicators
    
    def save_complete_analysis(self, prediction_data: dict, btc_3m: pd.DataFrame, btc_1w: pd.DataFrame, 
                             indicators_3m: Dict, indicators_1w: Dict, full_ai_analysis: str = "") -> str:
        """Save complete analysis data and return the hash"""
        try:
            # Generate unique hash
            analysis_hash = self.generate_analysis_hash(prediction_data, btc_3m, btc_1w)
            
            # Serialize data for storage
            btc_3m_serialized = self.serialize_dataframe(btc_3m)
            btc_1w_serialized = self.serialize_dataframe(btc_1w)
            indicators_3m_serialized = self.serialize_indicators(indicators_3m)
            indicators_1w_serialized = self.serialize_indicators(indicators_1w)
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO bitcoin_analyses (
                            analysis_hash, created_at, target_datetime, current_price_at_prediction,
                            predicted_price, probability_higher, probability_lower, confidence_level,
                            technical_summary, prediction_reasoning, full_ai_analysis,
                            btc_3m_data, btc_1w_data, indicators_3m, indicators_1w
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (analysis_hash) DO UPDATE SET
                            full_ai_analysis = EXCLUDED.full_ai_analysis
                    """, (
                        analysis_hash,
                        prediction_data.get('prediction_timestamp'),
                        prediction_data.get('target_datetime'),
                        prediction_data.get('current_price'),
                        prediction_data.get('predicted_price'),
                        prediction_data.get('probability_higher'),
                        prediction_data.get('probability_lower'),
                        prediction_data.get('confidence_level', 0),
                        prediction_data.get('technical_summary', '')[:1000],  # Store more than before
                        prediction_data.get('prediction_reasoning', '')[:1000],
                        full_ai_analysis,  # Store complete AI analysis
                        json.dumps(btc_3m_serialized),  # Need json.dumps() for saving
                        json.dumps(btc_1w_serialized),
                        json.dumps(indicators_3m_serialized),
                        json.dumps(indicators_1w_serialized)
                    ))
                    conn.commit()
            
            return analysis_hash
            
        except Exception as e:
            st.error(f"Error saving analysis: {str(e)}")
            return None
    
    def load_analysis_by_hash(self, analysis_hash: str) -> Optional[Tuple[dict, pd.DataFrame, pd.DataFrame, Dict, Dict]]:
        """Load complete analysis data by hash"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT analysis_hash, created_at, target_datetime, current_price_at_prediction,
                               predicted_price, probability_higher, probability_lower, confidence_level,
                               technical_summary, prediction_reasoning, full_ai_analysis,
                               btc_3m_data, btc_1w_data, indicators_3m, indicators_1w,
                               actual_price, accuracy_calculated
                        FROM bitcoin_analyses 
                        WHERE analysis_hash = %s
                    """, (analysis_hash,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    # Extract data
                    (hash_val, created_at, target_datetime, current_price, predicted_price,
                     prob_higher, prob_lower, confidence, tech_summary, reasoning, full_ai,
                     btc_3m_json, btc_1w_json, indicators_3m_json, indicators_1w_json,
                     actual_price, accuracy_calculated) = row
                    
                    # Deserialize DataFrames (PostgreSQL JSONB returns dicts directly, no json.loads needed)
                    btc_3m = self.deserialize_dataframe(btc_3m_json)
                    btc_1w = self.deserialize_dataframe(btc_1w_json)
                    
                    # Deserialize indicators with proper DataFrame indices
                    indicators_3m = self.deserialize_indicators(indicators_3m_json, btc_3m.index)
                    indicators_1w = self.deserialize_indicators(indicators_1w_json, btc_1w.index)
                    
                    # Prepare prediction data
                    prediction_data = {
                        'analysis_hash': hash_val,
                        'prediction_timestamp': created_at.isoformat(),
                        'target_datetime': target_datetime.isoformat(),
                        'current_price': float(current_price),
                        'predicted_price': float(predicted_price) if predicted_price else None,
                        'probability_higher': prob_higher,
                        'probability_lower': prob_lower,
                        'confidence_level': confidence,
                        'technical_summary': tech_summary,
                        'prediction_reasoning': reasoning,
                        'full_ai_analysis': full_ai,
                        'actual_price': float(actual_price) if actual_price else None,
                        'accuracy_calculated': accuracy_calculated
                    }
                    
                    return prediction_data, btc_3m, btc_1w, indicators_3m, indicators_1w
                    
        except Exception as e:
            st.error(f"Error loading analysis: {str(e)}")
            return None
    
    def get_recent_analyses(self, limit: int = 50, offset: int = 0) -> list:
        """Get recent analyses for display in history"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT analysis_hash, created_at, target_datetime, current_price_at_prediction,
                               predicted_price, probability_higher, probability_lower, confidence_level,
                               technical_summary, prediction_reasoning, actual_price, accuracy_calculated
                        FROM bitcoin_analyses 
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                    
                    rows = cur.fetchall()
                    analyses = []
                    
                    for row in rows:
                        analyses.append({
                            'analysis_hash': row[0],
                            'prediction_timestamp': row[1].isoformat(),
                            'target_datetime': row[2].isoformat(),
                            'current_price_at_prediction': float(row[3]),
                            'predicted_price': float(row[4]) if row[4] else None,
                            'probability_higher': row[5],
                            'probability_lower': row[6],
                            'confidence_level': row[7],
                            'technical_summary': row[8],
                            'prediction_reasoning': row[9],
                            'actual_price': float(row[10]) if row[10] else None,
                            'accuracy_calculated': row[11]
                        })
                    
                    return analyses
                    
        except Exception as e:
            st.error(f"Error loading recent analyses: {str(e)}")
            return []
    
    def get_total_analyses_count(self) -> int:
        """Get total number of analyses"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM bitcoin_analyses")
                    return cur.fetchone()[0]
        except Exception as e:
            st.error(f"Error counting analyses: {str(e)}")
            return 0
    
    def get_historical_btc_price(self, target_datetime):
        """Get Bitcoin price at a specific historical datetime"""
        try:
            # Convert to UTC for yfinance
            if target_datetime.tzinfo is not None:
                target_utc = target_datetime.astimezone(pytz.UTC)
            else:
                # Assume Eastern Time if no timezone
                eastern = pytz.timezone('US/Eastern')
                target_et = eastern.localize(target_datetime)
                target_utc = target_et.astimezone(pytz.UTC)
            
            # Get date range - fetch a few days around the target to ensure we get data
            start_date = target_utc.date() - timedelta(days=2)
            end_date = target_utc.date() + timedelta(days=2)
            
            # Fetch Bitcoin data
            btc = yf.Ticker("BTC-USD")
            hist_data = btc.history(start=start_date, end=end_date, interval="1h")
            
            if hist_data.empty:
                print(f"No historical data available for {target_utc}")
                return None
            
            # Find the closest time to our target
            target_timestamp = target_utc.replace(second=0, microsecond=0)
            
            # Convert index to UTC timezone-aware if not already
            if hist_data.index.tz is None:
                hist_data.index = hist_data.index.tz_localize('UTC')
            elif hist_data.index.tz != pytz.UTC:
                hist_data.index = hist_data.index.tz_convert('UTC')
            
            # Find closest timestamp
            time_diffs = abs(hist_data.index - target_timestamp)
            closest_idx = time_diffs.argmin()
            
            # Get the closest price
            closest_price = hist_data.iloc[closest_idx]['Close']
            closest_time = hist_data.index[closest_idx]
            
            print(f"Target time: {target_timestamp}, Closest data: {closest_time}, Price: ${closest_price:.2f}")
            return float(closest_price)
            
        except Exception as e:
            print(f"Error fetching historical price for {target_datetime}: {e}")
            return None

    def update_analysis_accuracy(self, current_btc_price: float = None):
        """Update accuracy for analyses whose target time has passed using historical prices"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    # Get all pending predictions that need accuracy updates
                    cur.execute("""
                        SELECT id, analysis_hash, target_datetime
                        FROM bitcoin_analyses 
                        WHERE target_datetime <= NOW() AND accuracy_calculated = FALSE
                        ORDER BY target_datetime
                    """)
                    
                    pending_analyses = cur.fetchall()
                    
                    if not pending_analyses:
                        return  # Nothing to update
                    
                    print(f"Updating accuracy for {len(pending_analyses)} expired predictions...")
                    
                    # Update each prediction with its historical price
                    for analysis_id, analysis_hash, target_datetime in pending_analyses:
                        # Get historical price at the target time
                        historical_price = self.get_historical_btc_price(target_datetime)
                        
                        if historical_price is not None:
                            # Update this specific prediction
                            cur.execute("""
                                UPDATE bitcoin_analyses 
                                SET actual_price = %s, accuracy_calculated = TRUE
                                WHERE id = %s
                            """, (historical_price, analysis_id))
                            
                            print(f"Updated prediction {analysis_hash[:8]} - Target: {target_datetime}, Historical Price: ${historical_price:.2f}")
                        else:
                            print(f"Could not fetch historical price for prediction {analysis_hash[:8]} at {target_datetime}")
                    
                    conn.commit()
                    print(f"Accuracy update complete!")
                    
        except Exception as e:
            print(f"Error updating accuracy: {str(e)}")
            st.error(f"Error updating accuracy: {str(e)}")

# Global instance
analysis_db = AnalysisDatabase()