import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import os

class TechnicalIndicatorsCalculator:
    def __init__(self, strategy='reversal', direction='both'):
        self.strategy = strategy
        self.direction = direction
        self.last_api_minute = None
        self.orders_placed = 0
        self.scale_in_count = 0
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def calculate_band_stability(self, band_values: list) -> bool:
        """Calculate if a volatility band has been stable over the period"""
        self.logger.info("\n" + "="*80)
        self.logger.info("BAND STABILITY ANALYSIS")
        self.logger.info("="*80)
        
        # Log initial band values with timestamps
        self.logger.info("\nHistorical Band Values:")
        self.logger.info("-"*40)
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        for i, value in enumerate(band_values):
            # Calculate approximate time for each bar (5 min intervals)
            bar_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
            self.logger.info(f"Bar {i} ({bar_time.strftime('%H:%M:%S')}): ${value:.2f}")
        
        # Calculate and log statistical measures
        mean_value = sum(band_values) / len(band_values)
        std_dev = (sum((x - mean_value) ** 2 for x in band_values) / len(band_values)) ** 0.5
        max_value = max(band_values)
        min_value = min(band_values)
        total_range = max_value - min_value
        
        self.logger.info("\nStatistical Analysis:")
        self.logger.info("-"*40)
        self.logger.info(f"Mean Value: ${mean_value:.2f}")
        self.logger.info(f"Standard Deviation: ${std_dev:.2f}")
        self.logger.info(f"Range: ${total_range:.2f} (${min_value:.2f} - ${max_value:.2f})")
        
        # Calculate percentage changes for the band relative to itself
        self.logger.info("\nBar-to-Bar Changes:")
        self.logger.info("-"*40)
        
        band_changes = []
        for i in range(1, len(band_values)):
            prev_value = band_values[i-1]
            curr_value = band_values[i]
            pct_change = abs((curr_value - prev_value) / prev_value)
            band_changes.append(pct_change)
            
            # Calculate time for these bars
            prev_time = current_time - timedelta(minutes=5*(len(band_values)-i))
            curr_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
            
            # Log each change with details and direction
            change_direction = "▲" if curr_value > prev_value else "▼" if curr_value < prev_value else "="
            self.logger.info(f"\n{prev_time.strftime('%H:%M:%S')} → {curr_time.strftime('%H:%M:%S')}:")
            self.logger.info(f"  Previous: ${prev_value:.2f}")
            self.logger.info(f"  Current:  ${curr_value:.2f} {change_direction}")
            self.logger.info(f"  Change:   {pct_change*100:.3f}% ({change_direction})")
        
        # Define stability threshold (0.1% change per bar)
        stability_threshold = 0.0019
        self.logger.info(f"\nStability Analysis:")
        self.logger.info("-"*40)
        self.logger.info(f"Threshold: {stability_threshold*100:.2f}% maximum change allowed per bar")
        
        # Calculate average change
        avg_change = sum(band_changes) / len(band_changes) if band_changes else 0
        self.logger.info(f"Average Change: {avg_change*100:.3f}%")
        
        # Check each change against threshold
        unstable_changes = [
            (i+1, change) 
            for i, change in enumerate(band_changes) 
            if change >= stability_threshold
        ]
        
        if unstable_changes:
            self.logger.info("\n⚠️ UNSTABLE CHANGES DETECTED:")
            for bar, change in unstable_changes:
                bar_time = current_time - timedelta(minutes=5*(len(band_values)-bar))
                self.logger.info(f"  Bar {bar} ({bar_time.strftime('%H:%M:%S')}): {change*100:.3f}% change")
                self.logger.info(f"  Exceeds {stability_threshold*100:.2f}% threshold by {(change-stability_threshold)*100:.3f}%")
            self.logger.info("\n❌ RESULT: Band is NOT stable")
            return False
        else:
            self.logger.info("\nAll changes within stability threshold:")
            for i, change in enumerate(band_changes):
                bar_time = current_time - timedelta(minutes=5*(len(band_values)-i-1))
                self.logger.info(f"  Bar {i+1} ({bar_time.strftime('%H:%M:%S')}): {change*100:.3f}% ✓")
            self.logger.info("\n✅ RESULT: Band is stable")
            return True

    def calculate_volatility_metrics(self, df):
        """Calculate volatility metrics using the previous bar's data"""
        if len(df) < 20:  # Need minimum data for meaningful analysis
            return None
            
        # Use previous bar's close for calculations
        prev_bar = df.iloc[-2]  # Previous bar
        lookback = min(2000, len(df)-2)  # Use last 20 bars excluding current bar
        
        # Get current system time for accurate second tracking
        current_time = datetime.now(pytz.timezone('America/Los_Angeles'))
        api_minute = df.index[-1].minute
        
        # Only update last_api_minute, counters are now reset after order fill confirmation
        if api_minute != self.last_api_minute:
            self.last_api_minute = api_minute
            
            # Log counter state for debugging
            if self.orders_placed > 0 or self.scale_in_count > 0:
                self.logger.info(f"Current counter state - orders_placed: {self.orders_placed}, scale_in_count: {self.scale_in_count}")
        
        # Calculate rolling means and standard deviations up to previous bar
        highside_vol = df['high'] - df['close']
        lowside_vol = df['low'] - df['close']
        mean_highside = highside_vol.iloc[:-1].rolling(window=lookback).mean().iloc[-1]
        mean_lowside = lowside_vol.iloc[:-1].rolling(window=lookback).mean().iloc[-1]
        std_highside = highside_vol.iloc[:-1].rolling(window=lookback).std().iloc[-1]
        std_lowside = lowside_vol.iloc[:-1].rolling(window=lookback).std().iloc[-1]
        
        # Calculate volatility bands based on previous bar's close
        high_side_limit = prev_bar['close'] + (std_highside + mean_highside)
        low_side_limit = prev_bar['close'] - (std_lowside - mean_lowside)
        
        # Calculate exceedances for current bar
        current_bar = df.iloc[-1]
        high_exceedance = current_bar['high'] - high_side_limit if current_bar['high'] > high_side_limit else 0
        low_exceedance = low_side_limit - current_bar['low'] if current_bar['low'] < low_side_limit else 0
        
        # Calculate relative distances and levels
        current_price = current_bar['close']
        band_range = high_side_limit - low_side_limit
        band_midpoint = low_side_limit + (band_range / 2)
        
        # Calculate distance from each band as percentage
        distance_to_high = ((high_side_limit - current_price) / band_range) * 100
        distance_to_low = ((current_price - low_side_limit) / band_range) * 100
        
        # Calculate position within band range as percentage (0% = at lower band, 100% = at upper band)
        position_in_range = ((current_price - low_side_limit) / band_range) * 100
        
        # Check for trading signals using system time
        seconds = current_time.second
        near_minute_end = seconds >= 55 and seconds <= 59
        
        trading_signal = None
        signal_direction = None
        
        if near_minute_end:
            self.logger.info(f"Evaluating trading signals at {current_time.strftime('%H:%M:%S')}")
            self.logger.info(f"Current Price: ${current_price:.2f} | Position in Range: {position_in_range:.1f}%")
            self.logger.info(f"Bands: High=${high_side_limit:.2f} | Low=${low_side_limit:.2f}")
            
            # Check if we're past 12:30 PM PST cutoff
            current_hour = current_time.hour
            current_minute = current_time.minute
            after_cutoff = (current_hour > 12) or (current_hour == 12 and current_minute >= 30)
            
            if after_cutoff:
                self.logger.info("No signal generated - After 12:30 PM PST cutoff")
            
            # Only generate trading signals before cutoff time
            if not after_cutoff:
                # Strategy determines the signal direction based on price position in the band
                if self.strategy == 'reversal':
                    self.logger.info("Evaluating reversal strategy conditions")
                    # Reversal strategy: SHORT at top of range, LONG at bottom of range
                    if position_in_range >= 99 and self.direction in ['short', 'both']:
                        trading_signal = f"SHORT Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "SHORT"
                        self.logger.info(f"Reversal SHORT signal triggered: Price at {position_in_range:.1f}% of range")
                    elif position_in_range <= 1 and self.direction in ['long', 'both']:
                        trading_signal = f"LONG Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "LONG"
                        self.logger.info(f"Reversal LONG signal triggered: Price at {position_in_range:.1f}% of range")
                    else:
                        self.logger.info(f"No reversal signal: Price at {position_in_range:.1f}% of range")
                else:  # momentum strategy
                    self.logger.info("Evaluating momentum strategy conditions")
                    # Momentum strategy: LONG at top of range, SHORT at bottom of range
                    if position_in_range >= 99 and self.direction in ['long', 'both']:
                        trading_signal = f"LONG Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "LONG"
                        self.logger.info(f"Momentum LONG signal triggered: Price at {position_in_range:.1f}% of range")
                    elif position_in_range <= 1 and self.direction in ['short', 'both']:
                        trading_signal = f"SHORT Signal @ {current_time.strftime('%H:%M:%S')}"
                        signal_direction = "SHORT"
                        self.logger.info(f"Momentum SHORT signal triggered: Price at {position_in_range:.1f}% of range")
                    else:
                        self.logger.info(f"No momentum signal: Price at {position_in_range:.1f}% of range")
        
        return {
            'high_band': high_side_limit,
            'low_band': low_side_limit,
            'high_exceedance': high_exceedance,
            'low_exceedance': low_exceedance,
            'trading_signal': trading_signal,
            'signal_direction': signal_direction,
            'distance_to_high': distance_to_high,
            'distance_to_low': distance_to_low,
            'position_in_range': position_in_range,
            'current_price': current_price
        }

def calculate_sma(data, window):
    """Calculate Simple Moving Average"""
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    """Calculate Exponential Moving Average"""
    return data.ewm(span=window).mean()

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band, sma

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_williams_r(high, low, close, window=14):
    """Calculate Williams %R"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

def calculate_atr(high, low, close, window=14):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cci(high, low, close, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma_tp) / (0.015 * mad)
    return cci

def calculate_momentum(data, window=10):
    """Calculate Momentum"""
    return data.diff(window)

def calculate_roc(data, window=12):
    """Calculate Rate of Change"""
    return ((data - data.shift(window)) / data.shift(window)) * 100

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = np.where(close > close.shift(1), volume, 
                   np.where(close < close.shift(1), -volume, 0))
    return pd.Series(obv, index=close.index).cumsum()

def calculate_adl(high, low, close, volume):
    """Calculate Accumulation/Distribution Line"""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)  # Handle division by zero
    mfv = clv * volume
    adl = mfv.cumsum()
    return adl

def calculate_volatility(data, window=20):
    """Calculate Historical Volatility (annualized)"""
    returns = data.pct_change()
    volatility = returns.rolling(window=window).std() * np.sqrt(252)
    return volatility

def calculate_support_resistance(data, window=20):
    """Calculate Support and Resistance levels"""
    support = data.rolling(window=window).min()
    resistance = data.rolling(window=window).max()
    return support, resistance

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()

def calculate_mfi(high, low, close, volume, window=14):
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=window).sum()
    negative_mf = negative_flow.rolling(window=window).sum()
    
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

def calculate_parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Calculate Parabolic SAR"""
    length = len(close)
    psar = np.zeros(length)
    trend = np.zeros(length)
    af = np.zeros(length)
    ep = np.zeros(length)
    
    # Initialize
    psar[0] = low.iloc[0]
    trend[0] = 1  # 1 for uptrend, -1 for downtrend
    af[0] = af_start
    ep[0] = high.iloc[0]
    
    for i in range(1, length):
        if trend[i-1] == 1:  # Uptrend
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            if low.iloc[i] <= psar[i]:
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low.iloc[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high.iloc[i] > ep[i-1]:
                    ep[i] = high.iloc[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            psar[i] = psar[i-1] - af[i-1] * (psar[i-1] - ep[i-1])
            
            if high.iloc[i] >= psar[i]:
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high.iloc[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low.iloc[i] < ep[i-1]:
                    ep[i] = low.iloc[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    return pd.Series(psar, index=close.index)

def main():
    # Load the historical data
    data_file = 'data/historical/NVDA_10_year_daily_data_20250725_180003.csv'
    
    print("Loading historical data...")
    df = pd.read_csv(data_file)
    
    # Convert datetime column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Calculate technical indicators
    print("Calculating technical indicators...")
    
    # 1. Simple Moving Averages
    df['SMA_5'] = calculate_sma(df['close'], 5)
    df['SMA_10'] = calculate_sma(df['close'], 10)
    df['SMA_20'] = calculate_sma(df['close'], 20)
    df['SMA_50'] = calculate_sma(df['close'], 50)
    df['SMA_200'] = calculate_sma(df['close'], 200)
    
    # 2. Exponential Moving Averages
    df['EMA_12'] = calculate_ema(df['close'], 12)
    df['EMA_26'] = calculate_ema(df['close'], 26)
    df['EMA_50'] = calculate_ema(df['close'], 50)
    
    # 3. RSI
    df['RSI_14'] = calculate_rsi(df['close'], 14)
    
    # 4. Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(df['close'], 20, 2)
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BB_Middle'] = bb_middle
    df['BB_Width'] = bb_upper - bb_lower
    df['BB_Position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    
    # 5. MACD
    macd_line, signal_line, histogram = calculate_macd(df['close'])
    df['MACD'] = macd_line
    df['MACD_Signal'] = signal_line
    df['MACD_Histogram'] = histogram
    
    # 6. Stochastic Oscillator
    stoch_k, stoch_d = calculate_stochastic(df['high'], df['low'], df['close'])
    df['Stoch_K'] = stoch_k
    df['Stoch_D'] = stoch_d
    
    # 7. Williams %R
    df['Williams_R'] = calculate_williams_r(df['high'], df['low'], df['close'])
    
    # 8. Average True Range
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # 9. Commodity Channel Index
    df['CCI'] = calculate_cci(df['high'], df['low'], df['close'])
    
    # 10. Momentum
    df['Momentum_10'] = calculate_momentum(df['close'], 10)
    
    # 11. Rate of Change
    df['ROC_12'] = calculate_roc(df['close'], 12)
    
    # 12. On-Balance Volume
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    
    # 13. Accumulation/Distribution Line
    df['ADL'] = calculate_adl(df['high'], df['low'], df['close'], df['volume'])
    
    # 14. Historical Volatility
    df['Volatility_20'] = calculate_volatility(df['close'], 20)
    
    # 15. Support and Resistance
    support, resistance = calculate_support_resistance(df['close'], 20)
    df['Support_20'] = support
    df['Resistance_20'] = resistance
    
    # 16. VWAP
    df['VWAP'] = calculate_vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # 17. Money Flow Index
    df['MFI'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
    
    # 18. Parabolic SAR
    df['PSAR'] = calculate_parabolic_sar(df['high'], df['low'], df['close'])
    
    # 19. Price change and returns
    df['Price_Change'] = df['close'].diff()
    df['Price_Change_Pct'] = df['close'].pct_change() * 100
    
    # 20. Volume indicators
    df['Volume_SMA_20'] = calculate_sma(df['volume'], 20)
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']
    
    # 21. Advanced Volatility Metrics (ML-ready format)
    print("Calculating advanced volatility metrics for ML training...")
    
    # Initialize the calculator
    calculator = TechnicalIndicatorsCalculator()
    
    # Set datetime as index temporarily for volatility calculations
    df_temp = df.set_index('datetime')
    
    # Initialize new columns with proper data types for ML
    df['Volatility_High_Band'] = np.nan
    df['Volatility_Low_Band'] = np.nan
    df['High_Exceedance'] = 0.0  # Initialize as float for ML
    df['Low_Exceedance'] = 0.0   # Initialize as float for ML
    df['Distance_To_High_Pct'] = np.nan
    df['Distance_To_Low_Pct'] = np.nan
    df['Position_In_Range_Pct'] = np.nan
    df['Band_Range'] = np.nan
    df['Band_Midpoint'] = np.nan
    df['High_Band_Stability'] = 0  # Binary: 1 for stable, 0 for unstable
    df['Low_Band_Stability'] = 0   # Binary: 1 for stable, 0 for unstable
    df['Signal_Long'] = 0          # Binary: 1 for long signal, 0 otherwise
    df['Signal_Short'] = 0         # Binary: 1 for short signal, 0 otherwise
    df['Price_Above_High_Band'] = 0  # Binary feature for ML
    df['Price_Below_Low_Band'] = 0   # Binary feature for ML
    
    # Calculate volatility metrics for each row (starting from row 20 to have enough data)
    print("Processing volatility metrics...")
    for i in range(20, len(df)):
        if i % 500 == 0:  # Progress indicator
            print(f"Processing row {i}/{len(df)}")
            
        # Get subset of data up to current row
        subset_df = df_temp.iloc[:i+1].copy()
        
        # Calculate volatility metrics
        vol_metrics = calculator.calculate_volatility_metrics(subset_df)
        
        if vol_metrics:
            # Store volatility band values
            df.iloc[i, df.columns.get_loc('Volatility_High_Band')] = vol_metrics['high_band']
            df.iloc[i, df.columns.get_loc('Volatility_Low_Band')] = vol_metrics['low_band']
            df.iloc[i, df.columns.get_loc('High_Exceedance')] = vol_metrics['high_exceedance']
            df.iloc[i, df.columns.get_loc('Low_Exceedance')] = vol_metrics['low_exceedance']
            df.iloc[i, df.columns.get_loc('Distance_To_High_Pct')] = vol_metrics['distance_to_high']
            df.iloc[i, df.columns.get_loc('Distance_To_Low_Pct')] = vol_metrics['distance_to_low']
            df.iloc[i, df.columns.get_loc('Position_In_Range_Pct')] = vol_metrics['position_in_range']
            
            # Calculate additional ML-friendly features
            band_range = vol_metrics['high_band'] - vol_metrics['low_band']
            band_midpoint = vol_metrics['low_band'] + (band_range / 2)
            df.iloc[i, df.columns.get_loc('Band_Range')] = band_range
            df.iloc[i, df.columns.get_loc('Band_Midpoint')] = band_midpoint
            
            # Binary features for ML algorithms
            current_price = vol_metrics['current_price']
            df.iloc[i, df.columns.get_loc('Price_Above_High_Band')] = 1 if current_price > vol_metrics['high_band'] else 0
            df.iloc[i, df.columns.get_loc('Price_Below_Low_Band')] = 1 if current_price < vol_metrics['low_band'] else 0
            
            # Signal encoding for ML (binary features)
            if vol_metrics['signal_direction'] == 'LONG':
                df.iloc[i, df.columns.get_loc('Signal_Long')] = 1
            elif vol_metrics['signal_direction'] == 'SHORT':
                df.iloc[i, df.columns.get_loc('Signal_Short')] = 1
            
            # Calculate band stability for last 20 periods if we have enough data
            if i >= 39:  # Need at least 20 periods of band data
                high_band_values = df['Volatility_High_Band'].iloc[i-19:i+1].dropna().tolist()
                low_band_values = df['Volatility_Low_Band'].iloc[i-19:i+1].dropna().tolist()
                
                if len(high_band_values) >= 10:  # Need at least 10 values for stability calculation
                    # Calculate stability for high band (returns boolean, convert to binary)
                    high_stability = calculator.calculate_band_stability(high_band_values)
                    df.iloc[i, df.columns.get_loc('High_Band_Stability')] = 1 if high_stability else 0
                    
                if len(low_band_values) >= 10:  # Need at least 10 values for stability calculation
                    # Calculate stability for low band (returns boolean, convert to binary)
                    low_stability = calculator.calculate_band_stability(low_band_values)
                    df.iloc[i, df.columns.get_loc('Low_Band_Stability')] = 1 if low_stability else 0
    
    # Add additional ML-ready derived features
    print("Adding derived features for ML training...")
    
    # Volatility band ratios and normalized features
    df['Band_Width_Normalized'] = df['Band_Range'] / df['close']  # Normalized by price
    df['High_Band_Distance_Normalized'] = (df['Volatility_High_Band'] - df['close']) / df['close']
    df['Low_Band_Distance_Normalized'] = (df['close'] - df['Volatility_Low_Band']) / df['close']
    
    # Moving averages of volatility metrics for trend analysis
    df['High_Band_SMA_5'] = calculate_sma(df['Volatility_High_Band'], 5)
    df['Low_Band_SMA_5'] = calculate_sma(df['Volatility_Low_Band'], 5)
    df['Band_Range_SMA_5'] = calculate_sma(df['Band_Range'], 5)
    
    # Volatility band momentum (rate of change)
    df['High_Band_ROC_5'] = calculate_roc(df['Volatility_High_Band'], 5)
    df['Low_Band_ROC_5'] = calculate_roc(df['Volatility_Low_Band'], 5)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/NVDA_with_indicators_{timestamp}.csv'
    
    # Save the enhanced dataset
    print(f"Saving enhanced dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Complete! Enhanced dataset saved with {len(df.columns)} columns")
    print(f"Original columns: {['symbol', 'datetime', 'open', 'high', 'low', 'close', 'volume']}")
    print(f"Added indicators: {len(df.columns) - 7}")
    
    # Display ML-ready features summary
    print("\n🤖 ML-Ready Features Summary:")
    
    # Categorize features for ML training
    binary_features = [
        'High_Band_Stability', 'Low_Band_Stability', 'Signal_Long', 'Signal_Short',
        'Price_Above_High_Band', 'Price_Below_Low_Band'
    ]
    
    continuous_features = [col for col in df.columns if col not in 
                          ['symbol', 'datetime'] + binary_features]
    
    print(f"\n📊 Feature Categories:")
    print(f"• Continuous Features: {len(continuous_features)} (suitable for regression, neural networks)")
    print(f"• Binary Features: {len(binary_features)} (suitable for classification, decision trees)")
    
    print(f"\n🎯 New Volatility Metrics Added:")
    volatility_features = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                          ['volatility', 'band', 'exceedance', 'signal', 'stability', 'distance', 'position', 'range'])]
    
    for feature in volatility_features:
        if feature in df.columns and df[feature].notna().sum() > 0:
            if feature in binary_features:
                signal_count = df[feature].sum()
                print(f"• {feature}: Binary ({signal_count} positive signals / {len(df)} total)")
            else:
                print(f"• {feature}: Continuous (Min={df[feature].min():.4f}, Max={df[feature].max():.4f}, Mean={df[feature].mean():.4f})")
    
    print(f"\n📈 Data Quality Check:")
    print(f"• Total rows: {len(df)}")
    print(f"• Rows with complete volatility data: {df['Volatility_High_Band'].notna().sum()}")
    print(f"• Missing data percentage: {(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    # ML Training Recommendations
    print(f"\n💡 ML Training Recommendations:")
    print("• Use rows 40+ for training (complete volatility band stability data)")
    print("• Binary features are ready for classification algorithms")
    print("• Continuous features may benefit from normalization/standardization")
    print("• Consider time-series cross-validation due to temporal nature")
    print("• Features include both technical indicators and advanced volatility metrics")
    
    print(f"\n💾 File saved: {output_file}")
    print(f"📈 Total rows: {len(df)}")
    print(f"📊 Total columns: {len(df.columns)}")
    
    return output_file

if __name__ == "__main__":
    output_file = main()
