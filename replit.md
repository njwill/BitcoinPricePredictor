# Bitcoin Analysis Dashboard

## Overview

This is a comprehensive Bitcoin analysis dashboard built with Streamlit that provides automated technical analysis, AI-powered market insights, and interactive visualizations. The application fetches real-time Bitcoin price data, calculates technical indicators, and generates intelligent analysis using OpenAI's GPT-4o model. The system is designed to run automated weekly updates and provide probability assessments for Bitcoin price movements.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

### 2025-08-18 UI/UX Improvements
- Added "Last Analysis" time display in sidebar alongside "Next Update"
- Removed AI Confidence Threshold slider (fixed at 75%)
- Improved chart spacing by increasing vertical spacing from 0.05 to 0.08
- Enhanced chart height and margins for better visibility
- Added consistent font formatting for AI analysis text with proper line spacing
- Fixed data type conversion issues in technical analysis calculations

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar for controls
- **State Management**: Session state for caching data and analysis results
- **Visualization**: Plotly for interactive charts and technical indicator displays
- **User Interface**: Clean, professional dashboard with Bitcoin-themed styling

### Backend Architecture
- **Modular Design**: Separated into specialized modules for different responsibilities:
  - `data_fetcher.py`: Handles Bitcoin data retrieval
  - `technical_analysis.py`: Calculates technical indicators using TA-Lib
  - `ai_analysis.py`: AI-powered analysis using OpenAI GPT-4o
  - `chart_generator.py`: Creates interactive Plotly visualizations
  - `scheduler.py`: Manages automated updates
  - `utils.py`: Common utility functions
- **Asynchronous Processing**: Support for async operations for better performance
- **Caching Strategy**: Session-based caching to minimize API calls and improve responsiveness

### Data Processing Pipeline
- **Data Source**: Yahoo Finance API via yfinance library
- **Time Periods**: Multiple timeframes (3-month and 1-week data) for comprehensive analysis
- **Technical Indicators**: Full suite including moving averages, Bollinger Bands, RSI, MACD, and Stochastic Oscillator
- **Data Validation**: Robust error handling and data quality checks

### Analysis Engine
- **Technical Analysis**: TA-Lib library for professional-grade technical indicator calculations
- **AI Analysis**: OpenAI GPT-4o integration for intelligent market analysis and predictions
- **Multi-timeframe Analysis**: Combines short-term (1 week) and medium-term (3 month) perspectives
- **Probability Assessments**: AI-generated probability scores for market movements

### Scheduling System
- **Automated Updates**: Weekly scheduled updates every Monday at 9:30 AM Eastern Time
- **Timezone Handling**: Proper Eastern Time zone management using pytz
- **Background Processing**: Threaded scheduler for non-blocking operations
- **Manual Override**: User-triggered refresh capability

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework for dashboard interface
- **Plotly**: Interactive charting and visualization library
- **Pandas & NumPy**: Data manipulation and numerical computations
- **yfinance**: Yahoo Finance API client for Bitcoin price data
- **TA-Lib**: Technical analysis library for financial indicators

### AI and Machine Learning
- **OpenAI**: GPT-4o model integration for intelligent analysis
- **API Requirements**: OpenAI API key required via environment variable

### Time and Scheduling
- **pytz**: Timezone handling for Eastern Time scheduling
- **schedule**: Task scheduling library for automated updates
- **datetime**: Built-in Python datetime handling

### Data Sources
- **Yahoo Finance**: Primary data source for Bitcoin price data (BTC-USD ticker)
- **Real-time Data**: Hourly updates for short-term analysis, daily for longer periods

### Configuration
- **Environment Variables**: OpenAI API key stored securely
- **Error Handling**: Comprehensive error management for API failures and data issues
- **Fallback Mechanisms**: Graceful degradation when external services are unavailable