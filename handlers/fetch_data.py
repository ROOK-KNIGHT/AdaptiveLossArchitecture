#!/usr/bin/env python3
"""
Enhanced data fetching script with integrated preprocessing pipeline
Supports both raw data fetching and full preprocessing with technical indicators
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from historical_data_handler import HistoricalDataHandler
import time

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the unified preprocessor
from src.data_processing.unified_preprocessing import UnifiedPreprocessor

def fetch_and_save_recent_data(symbol, days_lookback=365, process_indicators=False):
    """
    Fetch recent data for the specified symbol and optionally process with technical indicators
    
    Args:
        symbol (str): Stock symbol to fetch data for
        days_lookback (int): Number of days to fetch (default: 365)
        process_indicators (bool): Whether to calculate technical indicators (default: False)
    """
    print(f"Fetching recent {days_lookback} days of data for {symbol}...")
    
    try:
        if process_indicators:
            # Use the unified preprocessor for full processing
            print("🔄 Using Unified Preprocessor with live data...")
            processor = UnifiedPreprocessor(
                symbol=symbol,
                days_lookback=days_lookback,
                target_horizon=1,
                test_size=0.2
            )
            
            # Get fully processed data
            processed_data = processor.get_standardized_data()
            
            print(f"\n✅ Data processing completed!")
            print(f"📊 Features: {processed_data['total_features']}")
            print(f"📈 Training samples: {len(processed_data['X_train'])}")
            print(f"📉 Test samples: {len(processed_data['X_test'])}")
            
            return processed_data
            
        else:
            # Just fetch raw data
            print("📥 Fetching raw data only...")
            data_handler = HistoricalDataHandler()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_lookback)
            
            # Convert to milliseconds since epoch (required by Schwab API)
            start_date_ms = int(start_date.timestamp() * 1000)
            end_date_ms = int(end_date.timestamp() * 1000)
            
            print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Fetch historical data using daily frequency
            data = data_handler.fetch_historical_data(
                symbol=symbol,
                periodType="day",
                period=1,
                frequencyType="daily",
                freq=1,
                startDate=start_date_ms,
                endDate=end_date_ms,
                needExtendedHoursData=False
            )
            
            if not data or not data.get('candles'):
                print(f"No data received for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reorder columns
            df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert datetime strings to datetime objects
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/{symbol}_recent_data_{timestamp}.csv"
            
            # Save to CSV
            df.to_csv(filename, index=False)
            
            print(f"\n✅ Data successfully saved to: {filename}")
            print(f"📊 Total records: {len(df)}")
            print(f"📅 Date range in data: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            
            # Display first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Display last few rows
            print("\nLast 5 rows:")
            print(df.tail())
            
            # Display basic statistics
            print(f"\nBasic Statistics for {symbol}:")
            print(f"💰 Average Close Price: ${df['close'].mean():.2f}")
            print(f"📈 Highest Close Price: ${df['close'].max():.2f}")
            print(f"📉 Lowest Close Price: ${df['close'].min():.2f}")
            print(f"📊 Average Daily Volume: {df['volume'].mean():,.0f}")
            
            return filename
        
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {str(e)}")
        return None

def fetch_and_save_10_year_data(symbol):
    """
    Legacy function: Fetch 10 years of daily data for the specified symbol and save as CSV
    
    Args:
        symbol (str): Stock symbol to fetch data for
    """
    return fetch_and_save_recent_data(symbol, days_lookback=10*365, process_indicators=False)

def main():
    """Main function with enhanced options"""
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Enhanced data fetcher with preprocessing capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch raw data only
  python fetch_data.py AAPL
  python fetch_data.py NVDA --days 180
  
  # Fetch and process with technical indicators
  python fetch_data.py TSLA --process
  python fetch_data.py NVDA --process --days 500
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol to fetch data for (e.g., AAPL, NVDA, TSLA)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Number of days to fetch (default: 365)'
    )
    
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process data with technical indicators and prepare for ML models'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert ticker to uppercase for consistency
    ticker = args.ticker.upper()
    
    print("=" * 80)
    if args.process:
        print("ENHANCED DATA PROCESSOR - LIVE DATA + TECHNICAL INDICATORS")
    else:
        print("BASIC DATA FETCHER - RAW DATA ONLY")
    print("=" * 80)
    
    # Fetch data for the specified ticker
    result = fetch_and_save_recent_data(
        symbol=ticker,
        days_lookback=args.days,
        process_indicators=args.process
    )
    
    if result:
        if args.process:
            print(f"\n🎯 Data processing completed successfully!")
            print(f"📊 Ready for ML model training with {result['total_features']} features")
        else:
            print(f"\n✅ Successfully saved raw data to: {result}")
    else:
        print("\n❌ Failed to fetch and process data")

if __name__ == "__main__":
    main()
