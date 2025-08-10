"""
Unified Preprocessing Pipeline for Comprehensive ML Model Library
Integrates all technical indicators from technical_indicators_calculator.py
Supports dual-target prediction (returns + prices) with standardized preprocessing

Usage:
    # Direct usage with live data
    from src.data_processing.unified_preprocessing import UnifiedPreprocessor
    processor = UnifiedPreprocessor('NVDA', days_lookback=365)
    data = processor.get_standardized_data()
    
    # Via command line interface
    python handlers/fetch_data.py NVDA --process --days 365
    
Features:
    - Live data fetching from Schwab API
    - 60+ technical indicators from technical_indicators_calculator.py
    - Dual-target creation (returns and prices)
    - Standardized preprocessing for all models
    - Temporal splitting (no data leakage)
    - Feature categorization for analysis
    - Automatic data saving and caching
"""

import pandas as pd
import numpy as np
import sys
import os
import yaml
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, f_regression, 
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP for advanced feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_indicators_config():
    """Load technical indicators configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'technical_indicators_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Technical indicators configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load global configuration
CONFIG = load_config()
INDICATORS_CONFIG = load_indicators_config()

# Import all technical indicator functions
from .technical_indicators_calculator import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_bollinger_bands,
    calculate_macd, calculate_stochastic, calculate_williams_r, calculate_atr,
    calculate_cci, calculate_momentum, calculate_roc, calculate_obv,
    calculate_adl, calculate_volatility, calculate_support_resistance,
    calculate_vwap, calculate_mfi, calculate_parabolic_sar,
    TechnicalIndicatorsCalculator
)

# Import handlers for real-time data fetching
from handlers.historical_data_handler import HistoricalDataHandler
from handlers.connection_manager import ensure_valid_tokens

class UnifiedPreprocessor:
    """
    Unified preprocessing pipeline that ensures all models use identical data preparation
    """
    
    def __init__(self, symbol: str, target_horizon: int = None, test_size: float = None, 
                 lag_periods: List[int] = None, days_lookback: int = None):
        """
        Initialize the unified preprocessor with live data fetching
        
        Args:
            symbol: Stock symbol for live data fetching (e.g., 'NVDA', 'AAPL')
            target_horizon: Number of periods ahead to predict (uses config default if None)
            test_size: Proportion of data for testing (uses config default if None)
            lag_periods: List of lag periods for features (uses config default if None)
            days_lookback: Number of days to fetch for live data (uses config default if None)
        """
        # Load config values with fallbacks
        data_config = CONFIG['data_processing']
        
        self.symbol = symbol.upper()
        self.target_horizon = target_horizon if target_horizon is not None else data_config['target_horizon']
        self.test_size = test_size if test_size is not None else data_config['test_size']
        self.lag_periods = lag_periods if lag_periods is not None else data_config['lag_periods']
        self.days_lookback = days_lookback if days_lookback is not None else data_config['days_lookback']
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = None
        self.scaler_X = None
        self.scaler_y_returns = None
        self.scaler_y_prices = None
        self.imputer = None
        
        # Initialize data handler for live data
        self.data_handler = HistoricalDataHandler()
        
        # Feature categories for analysis
        self.feature_categories = {
            'trend': ['SMA', 'EMA', 'BB_Middle', 'VWAP', 'PSAR'],
            'momentum': ['RSI', 'Stoch', 'Williams', 'CCI', 'MFI', 'Momentum', 'ROC'],
            'volume': ['volume', 'OBV', 'ADL', 'Volume'],
            'volatility': ['ATR', 'Volatility', 'BB_Width', 'BB_Upper', 'BB_Lower', 'Band'],
            'signal': ['MACD', 'BB_Position', 'Signal'],
            'price': ['Price_Change', 'Support', 'Resistance'],
            'advanced_volatility': ['Exceedance', 'Distance', 'Position_In_Range', 'Stability']
        }
        
        # Enhanced feature selection parameters
        self.feature_selection_method = 'hybrid'  # 'mutual_info', 'shap', 'rfe', 'permutation', 'hybrid'
        self.n_features_to_select = 20
        self.nan_threshold = 0.1
        self.diversity_threshold = 0.8
        self.dynamic_adjustment = True
        self.shap_weight = 0.3
        self.feature_selection_history = []
    
    def fetch_recent_data(self, symbol: str, days_lookback: int = 365) -> pd.DataFrame:
        """
        Fetch recent data from Schwab API
        
        Args:
            symbol: Stock symbol to fetch (e.g., 'NVDA', 'AAPL')
            days_lookback: Number of days to fetch (default: 365)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Fetching recent {days_lookback} days of data for {symbol} from Schwab API...")
        
        try:
            # Ensure we have valid tokens
            tokens = ensure_valid_tokens()
            if not tokens:
                raise Exception("Failed to get valid API tokens")
            
            # Use period-based approach like in the reference file to avoid timestamp issues
            print(f"Using period-based API call to fetch {days_lookback} days of data...")
            
            # Use the exact same parameters as the working reference file
            # From divergence_backtest.py: periodType="year", period=1, frequencyType="daily", freq=1
            period_type = "year"
            period = 1
            frequency_type = "daily"
            frequency = 1
            
            print(f"Using periodType={period_type}, period={period}, frequencyType={frequency_type}, freq={frequency}")
            
            # Fetch historical data using the same approach as the reference
            data = self.data_handler.fetch_historical_data(
                symbol=symbol,
                periodType=period_type,
                period=period,
                frequencyType=frequency_type,
                freq=frequency,
                needExtendedHoursData=False
            )
            
            if not data or not data.get('candles'):
                raise Exception(f"No data received for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Reorder columns to match expected format
            df = df[['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert datetime strings to datetime objects
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            print(f"✓ Fetched {len(df)} records for {symbol}")
            print(f"Date range in data: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
            
            # Basic data validation
            if len(df) < 50:
                print(f"⚠️ Warning: Only {len(df)} records fetched, may not be sufficient for technical indicators")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {str(e)}")
            raise
    
    def save_fetched_data(self, df: pd.DataFrame, symbol: str) -> str:
        """
        Save fetched data to CSV file
        
        Args:
            df: DataFrame with fetched data
            symbol: Stock symbol
            
        Returns:
            Path to saved file
        """
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"data/{symbol}_data.csv"
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        print(f"✓ Data saved to: {filename}")
        return filename
    
    def load_data(self) -> pd.DataFrame:
        """Load and basic preprocessing of raw data"""
        print(f"Loading data from: {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        print(f"Loaded dataset shape: {df.shape}")
        
        # Convert datetime and sort
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        
        # Basic data cleaning
        initial_rows = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        final_rows = len(df)
        
        if initial_rows != final_rows:
            print(f"Removed {initial_rows - final_rows} rows with missing OHLCV data")
        
        self.raw_data = df
        return df
    
    def calculate_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators based on configuration settings
        """
        print("Calculating technical indicators based on configuration...")
        print(f"📋 Using configuration from: config/technical_indicators_config.yaml")
        
        # Make a copy to avoid modifying original
        df_indicators = df.copy()
        
        # Check global settings
        global_settings = INDICATORS_CONFIG.get('global_settings', {})
        
        # Track enabled indicators
        enabled_indicators = []
        
        # 1. Trend Indicators
        if global_settings.get('enable_trend_indicators', False):
            print("  • Trend Indicators...")
            trend_indicators = INDICATORS_CONFIG.get('trend_indicators', {})
            
            # Simple Moving Averages
            for indicator_name, config in trend_indicators.items():
                if config.get('enabled', False) and config.get('function') == 'calculate_sma':
                    window = config.get('parameters', {}).get('window', 20)
                    col_name = f"SMA_{window}"
                    df_indicators[col_name] = calculate_sma(df['close'], window)
                    enabled_indicators.append(col_name)
            
            # Exponential Moving Averages
            for indicator_name, config in trend_indicators.items():
                if config.get('enabled', False) and config.get('function') == 'calculate_ema':
                    window = config.get('parameters', {}).get('window', 12)
                    col_name = f"EMA_{window}"
                    df_indicators[col_name] = calculate_ema(df['close'], window)
                    enabled_indicators.append(col_name)
            
            # VWAP
            if trend_indicators.get('vwap', {}).get('enabled', False):
                df_indicators['VWAP'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
                enabled_indicators.append('VWAP')
            
            # Parabolic SAR
            if trend_indicators.get('parabolic_sar', {}).get('enabled', False):
                df_indicators['PSAR'] = calculate_parabolic_sar(df['high'], df['low'], df['close'])
                enabled_indicators.append('PSAR')
        
        # 2. Momentum Indicators
        if global_settings.get('enable_momentum_indicators', False):
            print("  • Momentum Indicators...")
            momentum_indicators = INDICATORS_CONFIG.get('momentum_indicators', {})
            
            # RSI
            if momentum_indicators.get('rsi_14', {}).get('enabled', False):
                df_indicators['RSI_14'] = calculate_rsi(df['close'], 14)
                enabled_indicators.append('RSI_14')
            
            # Stochastic Oscillator
            if momentum_indicators.get('stochastic_k', {}).get('enabled', False):
                stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
                df_indicators['Stoch_K'] = stoch_k
                enabled_indicators.append('Stoch_K')
            
            if momentum_indicators.get('stochastic_d', {}).get('enabled', False):
                if 'Stoch_K' not in df_indicators.columns:
                    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
                    df_indicators['Stoch_K'] = stoch_k
                df_indicators['Stoch_D'] = stoch_d
                enabled_indicators.append('Stoch_D')
            
            # Williams %R
            if momentum_indicators.get('williams_r', {}).get('enabled', False):
                df_indicators['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
                enabled_indicators.append('Williams_R')
            
            # CCI
            if momentum_indicators.get('cci', {}).get('enabled', False):
                df_indicators['CCI'] = calculate_cci(df['high'], df['low'], df['close'])
                enabled_indicators.append('CCI')
            
            # MFI
            if momentum_indicators.get('mfi', {}).get('enabled', False):
                df_indicators['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
                enabled_indicators.append('MFI')
            
            # Momentum
            if momentum_indicators.get('momentum_10', {}).get('enabled', False):
                df_indicators['Momentum_10'] = calculate_momentum(df['close'], 10)
                enabled_indicators.append('Momentum_10')
            
            # Rate of Change
            if momentum_indicators.get('roc_12', {}).get('enabled', False):
                df_indicators['ROC_12'] = calculate_roc(df['close'], 12)
                enabled_indicators.append('ROC_12')
        
        # 3. Volume Indicators
        if global_settings.get('enable_volume_indicators', False):
            print("  • Volume Indicators...")
            volume_indicators = INDICATORS_CONFIG.get('volume_indicators', {})
            
            # OBV
            if volume_indicators.get('obv', {}).get('enabled', False):
                df_indicators['OBV'] = calculate_obv(df['close'], df['volume'])
                enabled_indicators.append('OBV')
            
            # ADL
            if volume_indicators.get('adl', {}).get('enabled', False):
                df_indicators['ADL'] = calculate_adl(df['high'], df['low'], df['close'], df['volume'])
                enabled_indicators.append('ADL')
            
            # Volume SMA
            if volume_indicators.get('volume_sma_20', {}).get('enabled', False):
                df_indicators['Volume_SMA_20'] = calculate_sma(df['volume'], 20)
                enabled_indicators.append('Volume_SMA_20')
            
            # Volume Ratio
            if volume_indicators.get('volume_ratio', {}).get('enabled', False):
                if 'Volume_SMA_20' not in df_indicators.columns:
                    df_indicators['Volume_SMA_20'] = calculate_sma(df['volume'], 20)
                df_indicators['Volume_Ratio'] = df['volume'] / df_indicators['Volume_SMA_20']
                enabled_indicators.append('Volume_Ratio')
        
        # 4. Volatility Indicators
        if global_settings.get('enable_volatility_indicators', False):
            print("  • Volatility Indicators...")
            volatility_indicators = INDICATORS_CONFIG.get('volatility_indicators', {})
            
            # ATR
            if volatility_indicators.get('atr_14', {}).get('enabled', False):
                df_indicators['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
                enabled_indicators.append('ATR')
            
            # Historical Volatility
            if volatility_indicators.get('volatility_20', {}).get('enabled', False):
                df_indicators['Volatility_20'] = calculate_volatility(df['close'], 20)
                enabled_indicators.append('Volatility_20')
            
            # Bollinger Bands
            bb_enabled = any(volatility_indicators.get(bb_key, {}).get('enabled', False) 
                           for bb_key in ['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position'])
            
            if bb_enabled:
                bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(df['close'], 20, 2)
                
                if volatility_indicators.get('bb_upper', {}).get('enabled', False):
                    df_indicators['BB_Upper'] = bb_upper
                    enabled_indicators.append('BB_Upper')
                
                if volatility_indicators.get('bb_lower', {}).get('enabled', False):
                    df_indicators['BB_Lower'] = bb_lower
                    enabled_indicators.append('BB_Lower')
                
                if volatility_indicators.get('bb_middle', {}).get('enabled', False):
                    df_indicators['BB_Middle'] = bb_middle
                    enabled_indicators.append('BB_Middle')
                
                if volatility_indicators.get('bb_width', {}).get('enabled', False):
                    df_indicators['BB_Width'] = bb_upper - bb_lower
                    enabled_indicators.append('BB_Width')
                
                if volatility_indicators.get('bb_position', {}).get('enabled', False):
                    df_indicators['BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
                    enabled_indicators.append('BB_Position')
        
        # 5. Signal Indicators
        if global_settings.get('enable_signal_indicators', False):
            print("  • Signal Indicators...")
            signal_indicators = INDICATORS_CONFIG.get('signal_indicators', {})
            
            # MACD
            macd_enabled = any(signal_indicators.get(macd_key, {}).get('enabled', False) 
                             for macd_key in ['macd_line', 'macd_signal', 'macd_histogram'])
            
            if macd_enabled:
                macd_line, signal_line, histogram = calculate_macd(df['close'])
                
                if signal_indicators.get('macd_line', {}).get('enabled', False):
                    df_indicators['MACD'] = macd_line
                    enabled_indicators.append('MACD')
                
                if signal_indicators.get('macd_signal', {}).get('enabled', False):
                    df_indicators['MACD_Signal'] = signal_line
                    enabled_indicators.append('MACD_Signal')
                
                if signal_indicators.get('macd_histogram', {}).get('enabled', False):
                    df_indicators['MACD_Histogram'] = histogram
                    enabled_indicators.append('MACD_Histogram')
        
        # 6. Advanced Indicators
        if global_settings.get('enable_advanced_indicators', False):
            print("  • Advanced Indicators...")
            advanced_indicators = INDICATORS_CONFIG.get('advanced_indicators', {})
            
            # Support and Resistance
            if advanced_indicators.get('support_20', {}).get('enabled', False) or advanced_indicators.get('resistance_20', {}).get('enabled', False):
                support, resistance = calculate_support_resistance(df['close'], 20)
                
                if advanced_indicators.get('support_20', {}).get('enabled', False):
                    df_indicators['Support_20'] = support
                    enabled_indicators.append('Support_20')
                
                if advanced_indicators.get('resistance_20', {}).get('enabled', False):
                    df_indicators['Resistance_20'] = resistance
                    enabled_indicators.append('Resistance_20')
            
            # Price Change Indicators
            if advanced_indicators.get('price_change', {}).get('enabled', False):
                df_indicators['Price_Change'] = df['close'].diff()
                enabled_indicators.append('Price_Change')
            
            if advanced_indicators.get('price_change_pct', {}).get('enabled', False):
                df_indicators['Price_Change_Pct'] = df['close'].pct_change() * 100
                enabled_indicators.append('Price_Change_Pct')
            
            # Advanced Volatility Metrics
            volatility_metrics_enabled = any(advanced_indicators.get(vol_key, {}).get('enabled', False) 
                                           for vol_key in ['volatility_high_band', 'volatility_low_band', 
                                                         'high_exceedance', 'low_exceedance', 
                                                         'position_in_range_pct', 'distance_to_high_pct', 
                                                         'distance_to_low_pct'])
            
            if volatility_metrics_enabled:
                print("    • Advanced Volatility Metrics...")
                df_indicators = self._calculate_advanced_volatility_metrics(df_indicators, advanced_indicators)
                
                # Add enabled volatility metrics to the list
                volatility_metric_names = [
                    'Volatility_High_Band', 'Volatility_Low_Band', 'High_Exceedance', 'Low_Exceedance',
                    'Distance_To_High_Pct', 'Distance_To_Low_Pct', 'Position_In_Range_Pct',
                    'Band_Range', 'Band_Midpoint', 'High_Band_Stability', 'Low_Band_Stability',
                    'Signal_Long', 'Signal_Short', 'Price_Above_High_Band', 'Price_Below_Low_Band',
                    'Band_Width_Normalized', 'High_Band_Distance_Normalized', 'Low_Band_Distance_Normalized',
                    'High_Band_SMA_5', 'Low_Band_SMA_5', 'Band_Range_SMA_5', 'High_Band_ROC_5', 'Low_Band_ROC_5'
                ]
                enabled_indicators.extend(volatility_metric_names)
        
        print(f"  ✓ Calculated {len(enabled_indicators)} enabled technical indicators")
        print(f"  Enabled indicators: {', '.join(enabled_indicators[:10])}{'...' if len(enabled_indicators) > 10 else ''}")
        
        return df_indicators
    
    def _calculate_advanced_volatility_metrics(self, df: pd.DataFrame, advanced_indicators: dict = None) -> pd.DataFrame:
        """
        Calculate advanced volatility metrics using TechnicalIndicatorsCalculator
        """
        # Initialize the calculator
        calculator = TechnicalIndicatorsCalculator()
        
        # Set datetime as index temporarily
        df_temp = df.set_index('datetime')
        
        # Initialize new columns
        volatility_columns = [
            'Volatility_High_Band', 'Volatility_Low_Band', 'High_Exceedance', 'Low_Exceedance',
            'Distance_To_High_Pct', 'Distance_To_Low_Pct', 'Position_In_Range_Pct',
            'Band_Range', 'Band_Midpoint', 'High_Band_Stability', 'Low_Band_Stability',
            'Signal_Long', 'Signal_Short', 'Price_Above_High_Band', 'Price_Below_Low_Band',
            'Band_Width_Normalized', 'High_Band_Distance_Normalized', 'Low_Band_Distance_Normalized'
        ]
        
        for col in volatility_columns:
            if col in ['High_Band_Stability', 'Low_Band_Stability', 'Signal_Long', 'Signal_Short', 
                      'Price_Above_High_Band', 'Price_Below_Low_Band']:
                df[col] = 0  # Binary columns
            else:
                df[col] = np.nan  # Continuous columns
        
        # Calculate volatility metrics for each row (starting from row 20)
        print("    Processing advanced volatility metrics...")
        for i in range(20, len(df)):
            if i % 500 == 0:
                print(f"    Processing row {i}/{len(df)}")
            
            # Get subset of data up to current row
            subset_df = df_temp.iloc[:i+1].copy()
            
            # Calculate volatility metrics WITHOUT saving to CSV (save_to_csv=False)
            vol_metrics = calculator.calculate_volatility_metrics(subset_df, save_to_csv=False)
            
            if vol_metrics:
                # Store volatility band values
                df.iloc[i, df.columns.get_loc('Volatility_High_Band')] = vol_metrics['high_band']
                df.iloc[i, df.columns.get_loc('Volatility_Low_Band')] = vol_metrics['low_band']
                df.iloc[i, df.columns.get_loc('High_Exceedance')] = vol_metrics['high_exceedance']
                df.iloc[i, df.columns.get_loc('Low_Exceedance')] = vol_metrics['low_exceedance']
                df.iloc[i, df.columns.get_loc('Distance_To_High_Pct')] = vol_metrics['distance_to_high']
                df.iloc[i, df.columns.get_loc('Distance_To_Low_Pct')] = vol_metrics['distance_to_low']
                df.iloc[i, df.columns.get_loc('Position_In_Range_Pct')] = vol_metrics['position_in_range']
                
                # Calculate additional features
                band_range = vol_metrics['high_band'] - vol_metrics['low_band']
                band_midpoint = vol_metrics['low_band'] + (band_range / 2)
                df.iloc[i, df.columns.get_loc('Band_Range')] = band_range
                df.iloc[i, df.columns.get_loc('Band_Midpoint')] = band_midpoint
                
                # Binary features
                current_price = vol_metrics['current_price']
                df.iloc[i, df.columns.get_loc('Price_Above_High_Band')] = 1 if current_price > vol_metrics['high_band'] else 0
                df.iloc[i, df.columns.get_loc('Price_Below_Low_Band')] = 1 if current_price < vol_metrics['low_band'] else 0
                
                # Signal encoding
                if vol_metrics['signal_direction'] == 'LONG':
                    df.iloc[i, df.columns.get_loc('Signal_Long')] = 1
                elif vol_metrics['signal_direction'] == 'SHORT':
                    df.iloc[i, df.columns.get_loc('Signal_Short')] = 1
                
                # Normalized features
                df.iloc[i, df.columns.get_loc('Band_Width_Normalized')] = band_range / current_price
                df.iloc[i, df.columns.get_loc('High_Band_Distance_Normalized')] = (vol_metrics['high_band'] - current_price) / current_price
                df.iloc[i, df.columns.get_loc('Low_Band_Distance_Normalized')] = (current_price - vol_metrics['low_band']) / current_price
                
                # Band stability (simplified for performance)
                if i >= 39:  # Need at least 20 periods
                    high_band_values = df['Volatility_High_Band'].iloc[i-19:i+1].dropna().tolist()
                    low_band_values = df['Volatility_Low_Band'].iloc[i-19:i+1].dropna().tolist()
                    
                    if len(high_band_values) >= 10:
                        # Simplified stability check (coefficient of variation < 0.02)
                        high_cv = np.std(high_band_values) / np.mean(high_band_values) if np.mean(high_band_values) > 0 else 1
                        df.iloc[i, df.columns.get_loc('High_Band_Stability')] = 1 if high_cv < 0.02 else 0
                    
                    if len(low_band_values) >= 10:
                        low_cv = np.std(low_band_values) / np.mean(low_band_values) if np.mean(low_band_values) > 0 else 1
                        df.iloc[i, df.columns.get_loc('Low_Band_Stability')] = 1 if low_cv < 0.02 else 0
        
        # Add moving averages of volatility metrics
        df['High_Band_SMA_5'] = calculate_sma(df['Volatility_High_Band'], 5)
        df['Low_Band_SMA_5'] = calculate_sma(df['Volatility_Low_Band'], 5)
        df['Band_Range_SMA_5'] = calculate_sma(df['Band_Range'], 5)
        df['High_Band_ROC_5'] = calculate_roc(df['Volatility_High_Band'], 5)
        df['Low_Band_ROC_5'] = calculate_roc(df['Volatility_Low_Band'], 5)
        
        return df
    
    def create_dual_targets(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Create dual targets: returns and prices
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (returns_target, prices_target)
        """
        print(f"Creating dual targets with {self.target_horizon}-day horizon...")
        
        # Returns target: log returns
        returns_target = np.log(df['close'].shift(-self.target_horizon) / df['close'])
        
        # Prices target: future prices
        prices_target = df['close'].shift(-self.target_horizon)
        
        print(f"Returns target range: {returns_target.min():.4f} to {returns_target.max():.4f}")
        print(f"Prices target range: ${prices_target.min():.2f} to ${prices_target.max():.2f}")
        
        return returns_target, prices_target
    
    def create_lagged_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Create lagged features for true forecasting (no nowcasting)
        
        Args:
            df: DataFrame with technical indicators
            feature_columns: List of feature columns to lag
            
        Returns:
            DataFrame with lagged features
        """
        print(f"Creating lagged features for TRUE FORECASTING...")
        print(f"  Lag periods: {self.lag_periods}")
        
        lagged_df = df[['datetime']].copy()
        
        # Create lagged features for each lag period
        for lag in self.lag_periods:
            print(f"  Creating lag-{lag} features...")
            for col in feature_columns:
                lagged_col_name = f"{col}_lag{lag}"
                lagged_df[lagged_col_name] = df[col].shift(lag)
        
        # Add targets (future values)
        lagged_df['target_returns'] = df['target_returns']
        lagged_df['target_prices'] = df['target_prices']
        
        print(f"  ✓ Created {len(feature_columns) * len(self.lag_periods)} lagged features")
        
        return lagged_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with proper imputation
        
        Args:
            df: DataFrame with potential NaN values
            
        Returns:
            DataFrame with imputed values
        """
        print("Handling missing values with imputation...")
        
        # Separate feature columns from datetime and targets
        feature_columns = [col for col in df.columns 
                          if col not in ['datetime', 'target_returns', 'target_prices']]
        
        # Initialize imputer (median is robust to outliers)
        self.imputer = SimpleImputer(strategy='median')
        
        # Fit imputer on features only
        df_features = df[feature_columns].copy()
        
        # Check for columns with all NaN values
        all_nan_cols = df_features.columns[df_features.isna().all()].tolist()
        if all_nan_cols:
            print(f"  Removing columns with all NaN values: {all_nan_cols}")
            df_features = df_features.drop(columns=all_nan_cols)
            feature_columns = [col for col in feature_columns if col not in all_nan_cols]
        
        # Impute missing values
        if len(feature_columns) > 0:
            df_features_imputed = pd.DataFrame(
                self.imputer.fit_transform(df_features),
                columns=feature_columns,
                index=df_features.index
            )
            
            # Combine with datetime and targets
            result_df = pd.concat([
                df[['datetime']],
                df_features_imputed,
                df[['target_returns', 'target_prices']]
            ], axis=1)
            
            # Report imputation statistics
            nan_counts_before = df_features.isna().sum().sum()
            nan_counts_after = df_features_imputed.isna().sum().sum()
            print(f"  ✓ Imputed {nan_counts_before} missing values (now: {nan_counts_after})")
            
            return result_df
        else:
            print("  ⚠️ No valid feature columns found after NaN removal")
            return df
    
    def _calculate_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate mutual information scores for features"""
        print("    Calculating mutual information scores...")
        mi_scores = mutual_info_regression(X, y, random_state=42)
        return dict(zip(X.columns, mi_scores))
    
    def _calculate_shap_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate SHAP-based feature importance"""
        if not SHAP_AVAILABLE:
            print("    SHAP not available, falling back to mutual information")
            return self._calculate_mutual_information(X, y)
        
        try:
            print("    Calculating SHAP importance scores...")
            # Use a simple model for SHAP calculation
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Calculate mean absolute SHAP values as importance
            shap_importance = np.mean(np.abs(shap_values), axis=0)
            return dict(zip(X.columns, shap_importance))
        
        except Exception as e:
            print(f"    SHAP calculation failed: {e}, falling back to mutual information")
            return self._calculate_mutual_information(X, y)
    
    def _calculate_rfe_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate RFE-based feature importance"""
        print("    Calculating RFE importance scores...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe = RFE(model, n_features_to_select=min(self.n_features_to_select, X.shape[1]))
        rfe.fit(X, y)
        
        # Convert ranking to importance (lower rank = higher importance)
        importance = 1.0 / rfe.ranking_
        return dict(zip(X.columns, importance))
    
    def _calculate_permutation_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Calculate permutation-based feature importance"""
        print("    Calculating permutation importance scores...")
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X, y)
        
        perm_importance = permutation_importance(model, X, y, random_state=42)
        return dict(zip(X.columns, perm_importance.importances_mean))
    
    def _remove_correlated_features(self, X: pd.DataFrame, importance_scores: Dict[str, float]) -> List[str]:
        """Remove highly correlated features, keeping the most important ones"""
        print(f"    Removing correlated features (threshold: {self.diversity_threshold})")
        
        try:
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            
            # Handle NaN values in correlation matrix
            corr_matrix = corr_matrix.fillna(0)
            
            # Ensure correlation matrix is symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            
            # Create distance matrix for clustering
            distance_matrix = 1 - corr_matrix
            
            # Ensure distance matrix is valid for clustering
            np.fill_diagonal(distance_matrix.values, 0)  # Diagonal should be 0
            
            # Convert to condensed distance matrix for linkage
            try:
                condensed_dist = squareform(distance_matrix, checks=False)
                
                # Perform hierarchical clustering
                linkage_matrix = linkage(condensed_dist, method='average')
                
                # Get clusters based on diversity threshold
                clusters = fcluster(linkage_matrix, 1 - self.diversity_threshold, criterion='distance')
                
                # Select best feature from each cluster
                selected_features = []
                for cluster_id in np.unique(clusters):
                    cluster_features = X.columns[clusters == cluster_id].tolist()
                    
                    # Select feature with highest importance in this cluster
                    cluster_importance = {f: importance_scores.get(f, 0) for f in cluster_features}
                    best_feature = max(cluster_importance, key=cluster_importance.get)
                    selected_features.append(best_feature)
                
                print(f"    Reduced from {len(X.columns)} to {len(selected_features)} diverse features")
                return selected_features
                
            except Exception as clustering_error:
                print(f"    Clustering failed: {clustering_error}, using simple correlation-based selection")
                # Fallback to simple correlation-based selection
                return self._simple_correlation_filter(X, importance_scores)
                
        except Exception as e:
            print(f"    Correlation analysis failed: {e}, returning top features by importance")
            # Fallback to top features by importance
            sorted_features = sorted(importance_scores.keys(), 
                                   key=lambda x: importance_scores[x], 
                                   reverse=True)
            return sorted_features[:min(len(sorted_features), self.n_features_to_select)]
    
    def _simple_correlation_filter(self, X: pd.DataFrame, importance_scores: Dict[str, float]) -> List[str]:
        """Simple correlation-based feature filtering as fallback"""
        selected_features = []
        remaining_features = list(X.columns)
        
        # Sort features by importance
        sorted_features = sorted(remaining_features, 
                               key=lambda x: importance_scores.get(x, 0), 
                               reverse=True)
        
        for feature in sorted_features:
            if len(selected_features) >= self.n_features_to_select:
                break
                
            # Check correlation with already selected features
            is_diverse = True
            for selected_feature in selected_features:
                try:
                    corr = X[feature].corr(X[selected_feature])
                    if abs(corr) > self.diversity_threshold:
                        is_diverse = False
                        break
                except:
                    # If correlation calculation fails, assume diverse
                    continue
            
            if is_diverse:
                selected_features.append(feature)
        
        # If we don't have enough diverse features, fill with top importance
        if len(selected_features) < self.n_features_to_select:
            for feature in sorted_features:
                if feature not in selected_features:
                    selected_features.append(feature)
                    if len(selected_features) >= self.n_features_to_select:
                        break
        
        return selected_features
    
    def _dynamic_nan_threshold_adjustment(self, X: pd.DataFrame) -> float:
        """Dynamically adjust NaN threshold based on data characteristics"""
        if not self.dynamic_adjustment:
            return self.nan_threshold
        
        # Calculate overall data completeness
        overall_completeness = 1 - X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        
        # Adjust threshold based on data completeness
        if overall_completeness > 0.9:
            # High completeness: be more strict
            adjusted_threshold = self.nan_threshold * 0.5
        elif overall_completeness < 0.7:
            # Low completeness: be more lenient
            adjusted_threshold = self.nan_threshold * 1.5
        else:
            adjusted_threshold = self.nan_threshold
        
        # Ensure threshold stays within reasonable bounds
        adjusted_threshold = max(0.05, min(0.3, adjusted_threshold))
        
        if adjusted_threshold != self.nan_threshold:
            print(f"    Dynamically adjusted NaN threshold: {self.nan_threshold} → {adjusted_threshold}")
        
        return adjusted_threshold
    
    def _dynamic_feature_count_adjustment(self, X: pd.DataFrame, y: pd.Series) -> int:
        """Dynamically adjust number of features based on data characteristics"""
        if not self.dynamic_adjustment:
            return self.n_features_to_select
        
        # Base adjustment on data size and dimensionality
        n_samples, n_features_total = X.shape
        
        # Rule of thumb: roughly 10-20 samples per feature
        max_features_by_samples = max(5, n_samples // 15)
        
        # Don't select more than 50% of available features
        max_features_by_total = max(5, n_features_total // 2)
        
        # Take the minimum of constraints and original request
        adjusted_n_features = min(self.n_features_to_select, max_features_by_samples, max_features_by_total)
        
        if adjusted_n_features != self.n_features_to_select:
            print(f"    Dynamically adjusted feature count: {self.n_features_to_select} → {adjusted_n_features}")
        
        return adjusted_n_features
    
    def _enhanced_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Enhanced feature selection using non-linear methods and dynamic adjustments
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            List of selected feature names
        """
        print(f"\n🎯 Enhanced Feature Selection ({self.feature_selection_method} method)")
        
        # Dynamic parameter adjustments
        adjusted_nan_threshold = self._dynamic_nan_threshold_adjustment(X)
        adjusted_n_features = self._dynamic_feature_count_adjustment(X, y)
        
        # Remove features with too many NaN values
        nan_ratios = X.isnull().sum() / len(X)
        valid_features = nan_ratios[nan_ratios <= adjusted_nan_threshold].index.tolist()
        X_valid = X[valid_features]
        
        print(f"  Removed {len(X.columns) - len(valid_features)} features due to NaN threshold")
        
        if len(valid_features) == 0:
            raise ValueError("No features remain after NaN filtering")
        
        # Calculate importance scores based on method
        if self.feature_selection_method == 'mutual_info':
            importance_scores = self._calculate_mutual_information(X_valid, y)
        elif self.feature_selection_method == 'shap':
            importance_scores = self._calculate_shap_importance(X_valid, y)
        elif self.feature_selection_method == 'rfe':
            importance_scores = self._calculate_rfe_importance(X_valid, y)
        elif self.feature_selection_method == 'permutation':
            importance_scores = self._calculate_permutation_importance(X_valid, y)
        elif self.feature_selection_method == 'hybrid':
            print("    Using hybrid approach combining multiple methods...")
            # Combine multiple methods
            mi_scores = self._calculate_mutual_information(X_valid, y)
            shap_scores = self._calculate_shap_importance(X_valid, y)
            perm_scores = self._calculate_permutation_importance(X_valid, y)
            
            # Normalize scores
            mi_norm = {k: v / max(mi_scores.values()) if max(mi_scores.values()) > 0 else 0 
                      for k, v in mi_scores.items()}
            shap_norm = {k: v / max(shap_scores.values()) if max(shap_scores.values()) > 0 else 0 
                        for k, v in shap_scores.items()}
            perm_norm = {k: v / max(perm_scores.values()) if max(perm_scores.values()) > 0 else 0 
                        for k, v in perm_scores.items()}
            
            # Combine scores with weights
            importance_scores = {}
            for feature in X_valid.columns:
                importance_scores[feature] = (
                    0.4 * mi_norm.get(feature, 0) +
                    self.shap_weight * shap_norm.get(feature, 0) +
                    (0.6 - self.shap_weight) * perm_norm.get(feature, 0)
                )
        else:
            # Default to mutual information
            importance_scores = self._calculate_mutual_information(X_valid, y)
        
        # Remove highly correlated features
        diverse_features = self._remove_correlated_features(X_valid, importance_scores)
        
        # Select top features from diverse set
        diverse_importance = {f: importance_scores[f] for f in diverse_features}
        selected_features = sorted(diverse_importance.keys(), 
                                 key=lambda x: diverse_importance[x], 
                                 reverse=True)[:adjusted_n_features]
        
        # Store feature selection history
        self.feature_selection_history.append({
            'timestamp': datetime.now(),
            'method': self.feature_selection_method,
            'original_features': len(X.columns),
            'valid_features': len(valid_features),
            'diverse_features': len(diverse_features),
            'selected_features': len(selected_features),
            'nan_threshold': adjusted_nan_threshold,
            'selected_feature_names': selected_features
        })
        
        print(f"  ✓ Selected {len(selected_features)} features using {self.feature_selection_method} method")
        print(f"  Top 5 selected features:")
        for i, feature in enumerate(selected_features[:5], 1):
            score = importance_scores.get(feature, 0)
            print(f"    {i}. {feature}: {score:.4f}")
        
        return selected_features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame]:
        """
        Prepare and select features for modeling with enhanced feature selection
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            Tuple of (selected_feature_names, cleaned_dataframe)
        """
        print("Preparing features with ENHANCED FEATURE SELECTION...")
        
        # Exclude non-feature columns
        exclude_columns = [
            'symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume',
            'target_returns', 'target_prices'
        ]
        
        # Get all potential feature columns (technical indicators)
        base_feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Remove columns with too many NaN values (>70% for base features)
        valid_base_features = []
        for col in base_feature_columns:
            nan_pct = df[col].isna().sum() / len(df)
            if nan_pct < 0.7:  # More lenient for base features
                valid_base_features.append(col)
            else:
                print(f"  Excluding {col}: {nan_pct*100:.1f}% missing values")
        
        print(f"Selected {len(valid_base_features)} base features from {len(base_feature_columns)} candidates")
        
        # Create lagged features for true forecasting
        df_with_lags = self.create_lagged_features(df, valid_base_features)
        
        # Handle missing values with imputation
        df_clean = self.handle_missing_values(df_with_lags)
        
        # Get all lagged feature columns
        all_lagged_features = [col for col in df_clean.columns 
                              if col not in ['datetime', 'target_returns', 'target_prices']]
        
        print(f"Created {len(all_lagged_features)} lagged features")
        
        # Apply enhanced feature selection on returns target
        X_all = df_clean[all_lagged_features]
        y_returns = df_clean['target_returns']
        
        # Remove rows with NaN targets for feature selection
        valid_mask = ~y_returns.isna()
        X_valid = X_all[valid_mask]
        y_valid = y_returns[valid_mask]
        
        # Enhanced feature selection
        selected_features = self._enhanced_feature_selection(X_valid, y_valid)
        
        # Categorize selected features for analysis
        self._categorize_lagged_features(selected_features)
        
        self.feature_columns = selected_features
        return selected_features, df_clean
    
    def _categorize_features(self, features: List[str]) -> None:
        """Categorize features for analysis"""
        categorized = {category: [] for category in self.feature_categories.keys()}
        categorized['other'] = []
        
        for feature in features:
            feature_lower = feature.lower()
            categorized_flag = False
            
            for category, keywords in self.feature_categories.items():
                if any(keyword.lower() in feature_lower for keyword in keywords):
                    categorized[category].append(feature)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(feature)
        
        print("\nFeature Categories:")
        for category, feature_list in categorized.items():
            if feature_list:
                print(f"  {category}: {len(feature_list)} features")
    
    def _categorize_lagged_features(self, features: List[str]) -> None:
        """Categorize lagged features for analysis"""
        categorized = {category: [] for category in self.feature_categories.keys()}
        categorized['other'] = []
        
        for feature in features:
            feature_lower = feature.lower()
            categorized_flag = False
            
            # Remove lag suffix for categorization
            base_feature = feature_lower.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '')
            
            for category, keywords in self.feature_categories.items():
                if any(keyword.lower() in base_feature for keyword in keywords):
                    categorized[category].append(feature)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(feature)
        
        print("\nLagged Feature Categories:")
        for category, feature_list in categorized.items():
            if feature_list:
                print(f"  {category}: {len(feature_list)} features")
    
    def temporal_split(self, X: pd.DataFrame, y_returns: pd.Series, y_prices: pd.Series) -> Dict:
        """
        Perform temporal split to avoid data leakage
        
        Args:
            X: Features DataFrame
            y_returns: Returns target
            y_prices: Prices target
            
        Returns:
            Dictionary with train/test splits for both targets
        """
        print(f"\nPerforming temporal split ({(1-self.test_size)*100:.0f}%/{self.test_size*100:.0f}%)...")
        
        # Remove rows with NaN targets
        valid_mask = ~(y_returns.isna() | y_prices.isna())
        X_clean = X[valid_mask]
        y_returns_clean = y_returns[valid_mask]
        y_prices_clean = y_prices[valid_mask]
        
        print(f"Valid samples after removing NaN targets: {len(X_clean)}")
        
        # Temporal split
        split_idx = int((1 - self.test_size) * len(X_clean))
        
        # Split features
        X_train = X_clean.iloc[:split_idx]
        X_test = X_clean.iloc[split_idx:]
        
        # Split returns target
        y_returns_train = y_returns_clean.iloc[:split_idx]
        y_returns_test = y_returns_clean.iloc[split_idx:]
        
        # Split prices target
        y_prices_train = y_prices_clean.iloc[:split_idx]
        y_prices_test = y_prices_clean.iloc[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Distribution analysis
        print(f"\nDistribution Analysis:")
        print(f"Returns - Train mean: {y_returns_train.mean():.6f}, Test mean: {y_returns_test.mean():.6f}")
        print(f"Prices - Train mean: ${y_prices_train.mean():.2f}, Test mean: ${y_prices_test.mean():.2f}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_returns_train': y_returns_train,
            'y_returns_test': y_returns_test,
            'y_prices_train': y_prices_train,
            'y_prices_test': y_prices_test,
            'train_test_returns_diff': y_returns_test.mean() - y_returns_train.mean(),
            'train_test_prices_diff': y_prices_test.mean() - y_prices_train.mean()
        }
    
    def scale_data(self, data_splits: Dict) -> Dict:
        """
        Scale features and targets using robust scaling
        
        Args:
            data_splits: Dictionary from temporal_split
            
        Returns:
            Dictionary with scaled data and scalers
        """
        print("\nScaling features and targets...")
        
        # Initialize scalers
        self.scaler_X = RobustScaler()  # More robust to outliers
        self.scaler_y_returns = StandardScaler()
        self.scaler_y_prices = StandardScaler()
        
        # Scale features (fit only on training data)
        X_train_scaled = self.scaler_X.fit_transform(data_splits['X_train'])
        X_test_scaled = self.scaler_X.transform(data_splits['X_test'])
        
        # Scale returns target
        y_returns_train_scaled = self.scaler_y_returns.fit_transform(
            data_splits['y_returns_train'].values.reshape(-1, 1)
        ).flatten()
        y_returns_test_scaled = self.scaler_y_returns.transform(
            data_splits['y_returns_test'].values.reshape(-1, 1)
        ).flatten()
        
        # Scale prices target
        y_prices_train_scaled = self.scaler_y_prices.fit_transform(
            data_splits['y_prices_train'].values.reshape(-1, 1)
        ).flatten()
        y_prices_test_scaled = self.scaler_y_prices.transform(
            data_splits['y_prices_test'].values.reshape(-1, 1)
        ).flatten()
        
        print("✓ Scaling completed")
        
        # Create scaled data dictionary
        scaled_data = {
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'y_returns_train_scaled': y_returns_train_scaled,
            'y_returns_test_scaled': y_returns_test_scaled,
            'y_prices_train_scaled': y_prices_train_scaled,
            'y_prices_test_scaled': y_prices_test_scaled,
            'scaler_X': self.scaler_X,
            'scaler_y_returns': self.scaler_y_returns,
            'scaler_y_prices': self.scaler_y_prices,
            'feature_names': data_splits['X_train'].columns.tolist(),
            'train_test_returns_diff': data_splits['train_test_returns_diff'],
            'train_test_prices_diff': data_splits['train_test_prices_diff']
        }
        
        # Add original unscaled data
        scaled_data.update(data_splits)
        
        return scaled_data
    
    def get_standardized_data(self) -> Dict:
        """
        Main method to get standardized data for all models using live data
        
        Returns:
            Dictionary with all processed data, scalers, and metadata
        """
        print("="*80)
        print("UNIFIED PREPROCESSING PIPELINE - LIVE DATA")
        print("="*80)
        print("Comprehensive technical indicators + dual-target preparation")
        print()
        
        # 1. Fetch live data from Schwab API
        print(f"🔄 LIVE DATA MODE: Fetching recent data for {self.symbol}")
        df = self.fetch_recent_data(self.symbol, self.days_lookback)
        
        # Save the fetched data
        saved_path = self.save_fetched_data(df, self.symbol)
        print(f"💾 Live data saved to: {saved_path}")
        
        # 2. Calculate all technical indicators
        df_with_indicators = self.calculate_all_technical_indicators(df)
        
        # Save complete indicators DataFrame to comprehensive_indicators directory
        self._save_complete_indicators_to_csv(df_with_indicators)
        
        # 3. Create dual targets
        returns_target, prices_target = self.create_dual_targets(df_with_indicators)
        df_with_indicators['target_returns'] = returns_target
        df_with_indicators['target_prices'] = prices_target
        
        # 4. Prepare features with enhanced selection (returns selected_features and cleaned dataframe)
        selected_features, df_clean = self.prepare_features(df_with_indicators)
        
        # 5. Extract features and targets from cleaned dataframe
        X = df_clean[selected_features]
        y_returns = df_clean['target_returns']
        y_prices = df_clean['target_prices']
        
        # 6. Temporal split
        data_splits = self.temporal_split(X, y_returns, y_prices)
        
        # 7. Scale data
        final_data = self.scale_data(data_splits)
        
        # 8. Add metadata
        final_data.update({
            'total_features': len(selected_features),
            'total_samples': len(df_with_indicators),
            'valid_samples': len(data_splits['X_train']) + len(data_splits['X_test']),
            'symbol': self.symbol,
            'saved_data_path': saved_path,
            'target_horizon': self.target_horizon,
            'test_size': self.test_size,
            'feature_selection_method': self.feature_selection_method,
            'feature_selection_history': self.feature_selection_history,
            'complete_indicators_saved': True
        })
        
        self.processed_data = final_data
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        print(f"✓ Total features: {final_data['total_features']}")
        print(f"✓ Valid samples: {final_data['valid_samples']}")
        print(f"✓ Training samples: {len(final_data['X_train'])}")
        print(f"✓ Test samples: {len(final_data['X_test'])}")
        print(f"✓ Dual targets: Returns + Prices")
        print(f"✓ Forecast horizon: {self.target_horizon} day(s)")
        print("✓ Complete indicators saved to comprehensive_indicators/")
        print("✓ Ready for model training!")
        print("="*80)
        
        return final_data
    
    def _save_complete_indicators_to_csv(self, df_with_indicators):
        """Save the complete DataFrame with all indicators to comprehensive CSV - matches neural model input"""
        try:
            # Create comprehensive indicators directory
            base_dir = "/Users/isaac/AdaptiveLossArchitecture/data/results"
            comprehensive_dir = os.path.join(base_dir, 'comprehensive_indicators')
            os.makedirs(comprehensive_dir, exist_ok=True)
            
            # Generate filename with symbol

            filename = f"unified_preprocessing_{self.symbol}.csv"
            filepath = os.path.join(comprehensive_dir, filename)
            
            # Save the complete DataFrame - this matches exactly what neural models see
            df_with_indicators.to_csv(filepath, index=False)
            
            print(f"✓ Complete indicators DataFrame saved to: {filepath}")
            print(f"  Rows: {len(df_with_indicators)}, Columns: {len(df_with_indicators.columns)}")
            
            
        except Exception as e:
            print(f"❌ Error saving complete DataFrame to CSV: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def get_feature_importance_analysis(self) -> Dict:
        """
        Analyze feature importance and correlations
        
        Returns:
            Dictionary with feature analysis
        """
        if self.processed_data is None:
            raise ValueError("Must call get_standardized_data() first")
        
        print("\nAnalyzing feature importance...")
        
        # Get correlation with targets
        X_train = self.processed_data['X_train']
        y_returns_train = self.processed_data['y_returns_train']
        y_prices_train = self.processed_data['y_prices_train']
        
        # Calculate correlations
        returns_corr = X_train.corrwith(y_returns_train).abs().sort_values(ascending=False)
        prices_corr = X_train.corrwith(y_prices_train).abs().sort_values(ascending=False)
        
        # Top features for each target
        top_returns_features = returns_corr.head(20)
        top_prices_features = prices_corr.head(20)
        
        print(f"\nTop 10 features for Returns prediction:")
        for i, (feature, corr) in enumerate(top_returns_features.head(10).items(), 1):
            print(f"  {i:2d}. {feature}: {corr:.4f}")
        
        print(f"\nTop 10 features for Prices prediction:")
        for i, (feature, corr) in enumerate(top_prices_features.head(10).items(), 1):
            print(f"  {i:2d}. {feature}: {corr:.4f}")
        
        return {
            'returns_correlations': returns_corr,
            'prices_correlations': prices_corr,
            'top_returns_features': top_returns_features.head(20).index.tolist(),
            'top_prices_features': top_prices_features.head(20).index.tolist()
        }
    
    def get_enhanced_feature_selection_analysis(self) -> Dict:
        """
        Get detailed analysis of the enhanced feature selection process
        
        Returns:
            Dictionary with feature selection analysis
        """
        if not self.feature_selection_history:
            raise ValueError("No feature selection history available. Run get_standardized_data() first.")
        
        latest_selection = self.feature_selection_history[-1]
        
        print("\n" + "="*80)
        print("ENHANCED FEATURE SELECTION ANALYSIS")
        print("="*80)
        
        print(f"Method Used: {latest_selection['method']}")
        print(f"Original Features: {latest_selection['original_features']}")
        print(f"Valid Features (after NaN filtering): {latest_selection['valid_features']}")
        print(f"Diverse Features (after correlation filtering): {latest_selection['diverse_features']}")
        print(f"Final Selected Features: {latest_selection['selected_features']}")
        print(f"NaN Threshold Used: {latest_selection['nan_threshold']:.3f}")
        
        print(f"\nSelected Features:")
        for i, feature in enumerate(latest_selection['selected_feature_names'], 1):
            print(f"  {i:2d}. {feature}")
        
        # Categorize selected features
        categorized = {category: [] for category in self.feature_categories.keys()}
        categorized['other'] = []
        
        for feature in latest_selection['selected_feature_names']:
            feature_lower = feature.lower()
            base_feature = feature_lower.replace('_lag1', '').replace('_lag2', '').replace('_lag3', '')
            categorized_flag = False
            
            for category, keywords in self.feature_categories.items():
                if any(keyword.lower() in base_feature for keyword in keywords):
                    categorized[category].append(feature)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                categorized['other'].append(feature)
        
        print(f"\nFeature Categories in Selection:")
        for category, features in categorized.items():
            if features:
                print(f"  {category}: {len(features)} features")
                for feature in features[:3]:  # Show first 3
                    print(f"    - {feature}")
                if len(features) > 3:
                    print(f"    ... and {len(features) - 3} more")
        
        return {
            'selection_method': latest_selection['method'],
            'selection_stats': {
                'original_features': latest_selection['original_features'],
                'valid_features': latest_selection['valid_features'],
                'diverse_features': latest_selection['diverse_features'],
                'selected_features': latest_selection['selected_features']
            },
            'selected_feature_names': latest_selection['selected_feature_names'],
            'feature_categories': categorized,
            'nan_threshold_used': latest_selection['nan_threshold']
        }

def main():
    """Example usage of the unified preprocessor"""
    
    # Example with NVDA data
    data_path = "data/NVDA_with_indicators_20250725_231651.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists or update the path.")
        return
    
    # Initialize preprocessor
    processor = UnifiedPreprocessor(data_path, target_horizon=1, test_size=0.2)
    
    # Get standardized data
    data = processor.get_standardized_data()
    
    # Analyze feature importance
    feature_analysis = processor.get_feature_importance_analysis()
    
    print(f"\n🎯 Preprocessing completed successfully!")
    print(f"📊 Data ready for {data['total_features']} features across {data['valid_samples']} samples")
    print(f"🔄 Dual targets: Returns and Prices")
    print(f"⚖️ Scalers fitted on training data only (no data leakage)")

if __name__ == "__main__":
    main()
