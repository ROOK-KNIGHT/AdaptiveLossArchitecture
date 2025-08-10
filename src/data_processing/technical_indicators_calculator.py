import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import os

class TechnicalIndicatorsCalculator:
    def __init__(self, strategy='momentum', direction='long'):
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
            self.logger.setLevel(logging.DEBUG)

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

    def calculate_volatility_metrics_for_dataframe(self, df, lookback=20, save_to_csv=True):
        """Calculate volatility metrics for entire DataFrame using rolling window approach"""
        if len(df) < lookback:
            self.logger.warning(f"DataFrame has {len(df)} rows, need at least {lookback} for volatility calculations")
            return df
        
        # Initialize result lists
        high_bands = []
        low_bands = []
        high_exceedances = []
        low_exceedances = []
        distances_to_high = []
        distances_to_low = []
        positions_in_range = []
        close_above_high_band = []
        close_below_low_band = []
        close_vs_high_band = []
        close_vs_low_band = []
        momentum_signals = []
        band_ranges = []
        band_midpoints = []
        
        # Get timestamps for CSV logging
        if hasattr(df.index, 'dtype') and ('datetime' in str(df.index.dtype) or 'Timestamp' in str(type(df.index[0]))):
            timestamps = df.index
        elif 'datetime' in df.columns:
            timestamps = df['datetime']
        else:
            # Create dummy timestamps
            timestamps = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
        
        self.logger.info(f"Calculating volatility metrics for {len(df)} rows with lookback={lookback}")
        
        for i in range(len(df)):
            if i < lookback:
                # For early rows, use NaN values
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                band_ranges.append(np.nan)
                band_midpoints.append(np.nan)
                continue
            
            # Get the lookback window (excluding current bar for calculation)
            window_end = i  # Current bar index
            window_start = max(0, window_end - lookback)
            window_df = df.iloc[window_start:window_end]  # Exclude current bar
            
            if len(window_df) == 0:
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                band_ranges.append(np.nan)
                band_midpoints.append(np.nan)
                continue
            
            # Use previous bar's close for calculations (if available)
            if i > 0:
                prev_close = df.iloc[i-1]['close']
            else:
                prev_close = df.iloc[i]['close']
            
            # Calculate volatility components
            highside_vol = window_df['high'] - window_df['close']
            lowside_vol = window_df['low'] - window_df['close']
            
            mean_highside = highside_vol.mean()
            mean_lowside = lowside_vol.mean()
            std_highside = highside_vol.std()
            std_lowside = lowside_vol.std()
            
            # Handle NaN values
            if pd.isna(mean_highside) or pd.isna(mean_lowside) or pd.isna(std_highside) or pd.isna(std_lowside):
                high_bands.append(np.nan)
                low_bands.append(np.nan)
                high_exceedances.append(np.nan)
                low_exceedances.append(np.nan)
                distances_to_high.append(np.nan)
                distances_to_low.append(np.nan)
                positions_in_range.append(np.nan)
                close_above_high_band.append(np.nan)
                close_below_low_band.append(np.nan)
                close_vs_high_band.append(np.nan)
                close_vs_low_band.append(np.nan)
                momentum_signals.append(np.nan)
                band_ranges.append(np.nan)
                band_midpoints.append(np.nan)
                continue
            
            # Calculate volatility bands based on previous bar's close
            high_side_limit = prev_close + (std_highside + mean_highside)
            low_side_limit = prev_close - (std_lowside - mean_lowside)
            
            # Current bar data
            current_bar = df.iloc[i]
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_close = current_bar['close']
            
            # Calculate exceedances for current bar
            high_exceedance = max(0, current_high - high_side_limit)
            low_exceedance = max(0, low_side_limit - current_low)
            
            # Calculate close price comparisons with bands
            is_close_above_high = current_close > high_side_limit
            is_close_below_low = current_close < low_side_limit
            close_vs_high_diff = current_close - high_side_limit  # Positive if above, negative if below
            close_vs_low_diff = current_close - low_side_limit    # Positive if above, negative if below
            
            # Calculate relative distances and levels
            band_range = high_side_limit - low_side_limit
            band_midpoint = low_side_limit + (band_range / 2)
            
            if band_range > 0:
                # Calculate distance from each band as percentage
                distance_to_high = ((high_side_limit - current_close) / band_range) * 100
                distance_to_low = ((current_close - low_side_limit) / band_range) * 100
                
                # Calculate position within band range as percentage (0% = at lower band, 100% = at upper band)
                position_in_range = ((current_close - low_side_limit) / band_range) * 100
                
                # Calculate momentum signals based on position in range
                # Momentum strategy: LONG at top of range (>=99%), SHORT at bottom of range (<=1%)
                if position_in_range >= 99:
                    momentum_signal = "LONG"
                elif position_in_range <= 1:
                    momentum_signal = "SHORT"
                else:
                    momentum_signal = None
            else:
                distance_to_high = np.nan
                distance_to_low = np.nan
                position_in_range = np.nan
                momentum_signal = None
            
            # Store results
            high_bands.append(high_side_limit)
            low_bands.append(low_side_limit)
            high_exceedances.append(high_exceedance)
            low_exceedances.append(low_exceedance)
            distances_to_high.append(distance_to_high)
            distances_to_low.append(distance_to_low)
            positions_in_range.append(position_in_range)
            close_above_high_band.append(1 if is_close_above_high else 0)
            close_below_low_band.append(1 if is_close_below_low else 0)
            close_vs_high_band.append(close_vs_high_diff)
            close_vs_low_band.append(close_vs_low_diff)
            momentum_signals.append(momentum_signal)
            band_ranges.append(band_range)
            band_midpoints.append(band_midpoint)
            
            # Log progress every 100 rows and last row
            if i % 100 == 0 or i == len(df) - 1:
                self.logger.info(f"Processed {i+1}/{len(df)} rows")
        
        # Add results to DataFrame
        df_result = df.copy()
        df_result['Volatility_High_Band'] = high_bands
        df_result['Volatility_Low_Band'] = low_bands
        df_result['High_Exceedance'] = high_exceedances
        df_result['Low_Exceedance'] = low_exceedances
        df_result['Distance_To_High_Pct'] = distances_to_high
        df_result['Distance_To_Low_Pct'] = distances_to_low
        df_result['Position_In_Range_Pct'] = positions_in_range
        df_result['Close_Above_High_Band'] = close_above_high_band
        df_result['Close_Below_Low_Band'] = close_below_low_band
        df_result['Close_Vs_High_Band'] = close_vs_high_band
        df_result['Close_Vs_Low_Band'] = close_vs_low_band
        df_result['Momentum_Signal'] = momentum_signals
        df_result['Band_Range'] = band_ranges
        df_result['Band_Midpoint'] = band_midpoints
        
        # Note: CSV saving is now handled by the unified_preprocessing.py pipeline
        # Individual indicator CSV files are still saved via _save_technical_indicators_to_csv
        
        self.logger.info(f"✓ Volatility metrics calculated for all {len(df)} rows")
        return df_result
    

    def calculate_volatility_metrics(self, df, save_to_csv=True):
        """Legacy method for backward compatibility - calls the new DataFrame method"""
        df_with_metrics = self.calculate_volatility_metrics_for_dataframe(df)
        
        # Return the last row's metrics in the original format
        if len(df_with_metrics) > 0:
            last_row = df_with_metrics.iloc[-1]
            
            # Save to CSV if requested
            if save_to_csv:
                # Get timestamp from the data
                if hasattr(df.index, 'dtype') and ('datetime' in str(df.index.dtype) or 'Timestamp' in str(type(df.index[0]))):
                    current_time = df.index[-1]
                elif 'datetime' in df.columns:
                    current_time = df['datetime'].iloc[-1]
                else:
                    current_time = pd.Timestamp.now()
                
                # Convert pandas Timestamp to datetime if needed
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                
                # Prepare metrics for CSV saving
                metrics = {
                    'timestamp': current_time,
                    'current_price': last_row.get('close', np.nan),
                    'prev_close': df.iloc[-2]['close'] if len(df) > 1 else last_row.get('close', np.nan),
                    'high_side_limit': last_row.get('Volatility_High_Band', np.nan),
                    'low_side_limit': last_row.get('Volatility_Low_Band', np.nan),
                    'band_range': last_row.get('Band_Range', np.nan),
                    'band_midpoint': last_row.get('Band_Midpoint', np.nan),
                    'high_exceedance': last_row.get('High_Exceedance', 0),
                    'low_exceedance': last_row.get('Low_Exceedance', 0),
                    'distance_to_high': last_row.get('Distance_To_High_Pct', np.nan),
                    'distance_to_low': last_row.get('Distance_To_Low_Pct', np.nan),
                    'position_in_range': last_row.get('Position_In_Range_Pct', np.nan),
                    'mean_highside': np.nan,  # Not available in new method
                    'mean_lowside': np.nan,   # Not available in new method
                    'std_highside': np.nan,   # Not available in new method
                    'std_lowside': np.nan,    # Not available in new method
                    'strategy': self.strategy,
                    'direction': self.direction,
                    'signal_direction': last_row.get('Momentum_Signal', None),
                    'after_cutoff': False  # Default value
                }
                
                # Save to CSV
                self._save_technical_indicators_to_csv(df, metrics)
            
            return {
                'high_band': last_row.get('Volatility_High_Band', np.nan),
                'low_band': last_row.get('Volatility_Low_Band', np.nan),
                'high_exceedance': last_row.get('High_Exceedance', 0),
                'low_exceedance': last_row.get('Low_Exceedance', 0),
                'trading_signal': None,
                'signal_direction': last_row.get('Momentum_Signal', None),
                'distance_to_high': last_row.get('Distance_To_High_Pct', np.nan),
                'distance_to_low': last_row.get('Distance_To_Low_Pct', np.nan),
                'position_in_range': last_row.get('Position_In_Range_Pct', np.nan),
                'current_price': last_row.get('close', np.nan)
            }
        return None

    def _save_technical_indicators_to_csv(self, df, metrics):
        """Save comprehensive technical indicator data to CSV files organized by indicator type"""
        try:
            # Create directory structure for organized CSV storage
            base_dir = "/Users/isaac/AdaptiveLossArchitecture/data/results"
            
            # Create subdirectories for different indicator categories
            directories = {
                'volatility': os.path.join(base_dir, 'volatility_indicators'),
                'bands': os.path.join(base_dir, 'band_indicators'), 
                'signals': os.path.join(base_dir, 'signal_indicators'),
                'price_analysis': os.path.join(base_dir, 'price_analysis'),
                'comprehensive': os.path.join(base_dir, 'comprehensive_indicators')
            }
            
            # Create directories if they don't exist
            for dir_path in directories.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Get current bar data
            current_bar = df.iloc[-1]
            prev_bar = df.iloc[-2] if len(df) > 1 else current_bar
            
            # Safe get function for metrics
            def safe_get(key, default=np.nan):
                return metrics.get(key, default) if metrics.get(key) is not None else default
            
            # 1. Volatility Indicators CSV
            volatility_data = {
                'timestamp': metrics['timestamp'],
                'current_price': metrics['current_price'],
                'prev_close': metrics['prev_close'],
                'price_change': metrics['current_price'] - metrics['prev_close'],
                'price_change_pct': ((metrics['current_price'] - metrics['prev_close']) / metrics['prev_close']) * 100 if metrics['prev_close'] != 0 else 0,
                'high_volatility': current_bar['high'] - current_bar['low'],
                'high_exceedance': safe_get('high_exceedance', 0),
                'low_exceedance': safe_get('low_exceedance', 0),
                'total_exceedance': safe_get('high_exceedance', 0) + safe_get('low_exceedance', 0)
            }
            
            self._append_to_csv(os.path.join(directories['volatility'], 'volatility_metrics.csv'), volatility_data)
            
            # 2. Band Indicators CSV
            band_data = {
                'timestamp': metrics['timestamp'],
                'current_price': metrics['current_price'],
                'high_band': safe_get('high_side_limit'),
                'low_band': safe_get('low_side_limit'),
                'band_midpoint': safe_get('band_midpoint'),
                'band_range': safe_get('band_range'),
                'band_width_pct': (safe_get('band_range') / metrics['current_price']) * 100 if metrics['current_price'] != 0 and not pd.isna(safe_get('band_range')) else np.nan,
                'distance_to_high_band': safe_get('distance_to_high'),
                'distance_to_low_band': safe_get('distance_to_low'),
                'position_in_range_pct': safe_get('position_in_range'),
                'above_midpoint': 1 if not pd.isna(safe_get('band_midpoint')) and metrics['current_price'] > safe_get('band_midpoint') else 0,
                'near_high_band': 1 if not pd.isna(safe_get('position_in_range')) and safe_get('position_in_range') >= 90 else 0,
                'near_low_band': 1 if not pd.isna(safe_get('position_in_range')) and safe_get('position_in_range') <= 10 else 0
            }
            
            self._append_to_csv(os.path.join(directories['bands'], 'band_metrics.csv'), band_data)
            
            # 3. Signal Indicators CSV
            signal_data = {
                'timestamp': metrics['timestamp'],
                'current_price': metrics['current_price'],
                'position_in_range_pct': safe_get('position_in_range'),
                'strategy': safe_get('strategy', 'momentum'),
                'direction_setting': safe_get('direction', 'long'),
                'signal_direction': safe_get('signal_direction', 'None'),
                'long_signal': 1 if safe_get('signal_direction') == 'LONG' else 0,
                'short_signal': 1 if safe_get('signal_direction') == 'SHORT' else 0
            }
            
            self._append_to_csv(os.path.join(directories['signals'], 'signal_metrics.csv'), signal_data)
            
            # 4. Price Analysis CSV
            price_data = {
                'timestamp': metrics['timestamp'],
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar.get('volume', 0),
                'prev_close': metrics['prev_close'],
                'intraday_range': current_bar['high'] - current_bar['low'],
                'intraday_range_pct': ((current_bar['high'] - current_bar['low']) / current_bar['close']) * 100 if current_bar['close'] != 0 else 0,
                'gap_from_prev': current_bar['open'] - metrics['prev_close'],
                'gap_pct': ((current_bar['open'] - metrics['prev_close']) / metrics['prev_close']) * 100 if metrics['prev_close'] != 0 else 0,
                'body_size': abs(current_bar['close'] - current_bar['open']),
                'upper_wick': current_bar['high'] - max(current_bar['open'], current_bar['close']),
                'lower_wick': min(current_bar['open'], current_bar['close']) - current_bar['low'],
                'bullish_candle': 1 if current_bar['close'] > current_bar['open'] else 0
            }
            
            self._append_to_csv(os.path.join(directories['price_analysis'], 'price_analysis.csv'), price_data)
            
            # 5. Comprehensive Indicators CSV (all data in one file) - This matches what neural models see
            comprehensive_data = {**volatility_data, **band_data, **signal_data, **price_data}
            # Remove duplicate timestamp columns
            comprehensive_data = {k: v for k, v in comprehensive_data.items() if not (k.startswith('timestamp') and k != 'timestamp')}
            
            self._append_to_csv(os.path.join(directories['comprehensive'], 'all_indicators.csv'), comprehensive_data)
            
            self.logger.info(f"Technical indicators saved to CSV files in {base_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving technical indicators to CSV: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _save_technical_indicators_to_csv_row(self, timestamp, metrics):
        """Save individual row technical indicator data to CSV files"""
        try:
            # Create directory structure for organized CSV storage
            base_dir = "/Users/isaac/AdaptiveLossArchitecture/data/results"
            
            # Create subdirectories for different indicator categories
            directories = {
                'volatility': os.path.join(base_dir, 'volatility_indicators'),
                'bands': os.path.join(base_dir, 'band_indicators'), 
                'signals': os.path.join(base_dir, 'signal_indicators'),
                'price_analysis': os.path.join(base_dir, 'price_analysis'),
                'comprehensive': os.path.join(base_dir, 'comprehensive_indicators')
            }
            
            # Create directories if they don't exist
            for dir_path in directories.values():
                os.makedirs(dir_path, exist_ok=True)
            
            # Get current bar data
            current_bar = metrics['current_bar']
            
            # 1. Volatility Indicators CSV
            volatility_data = {
                'timestamp': timestamp,
                'current_price': metrics['current_price'],
                'prev_close': metrics['prev_close'],
                'price_change': metrics['current_price'] - metrics['prev_close'],
                'price_change_pct': ((metrics['current_price'] - metrics['prev_close']) / metrics['prev_close']) * 100,
                'high_volatility': current_bar['high'] - current_bar['low'],
                'high_exceedance': metrics['high_exceedance'],
                'low_exceedance': metrics['low_exceedance'],
                'total_exceedance': metrics['high_exceedance'] + metrics['low_exceedance']
            }
            
            self._append_to_csv(os.path.join(directories['volatility'], 'volatility_metrics.csv'), volatility_data)
            
            # 2. Band Indicators CSV
            band_data = {
                'timestamp': timestamp,
                'current_price': metrics['current_price'],
                'high_band': metrics['high_side_limit'],
                'low_band': metrics['low_side_limit'],
                'band_midpoint': metrics['band_midpoint'],
                'band_range': metrics['band_range'],
                'band_width_pct': (metrics['band_range'] / metrics['current_price']) * 100,
                'distance_to_high_band': metrics['distance_to_high'],
                'distance_to_low_band': metrics['distance_to_low'],
                'position_in_range_pct': metrics['position_in_range'],
                'above_midpoint': 1 if metrics['current_price'] > metrics['band_midpoint'] else 0,
                'near_high_band': 1 if metrics['position_in_range'] >= 90 else 0,
                'near_low_band': 1 if metrics['position_in_range'] <= 10 else 0
            }
            
            self._append_to_csv(os.path.join(directories['bands'], 'band_metrics.csv'), band_data)
            
            # 3. Signal Indicators CSV
            signal_data = {
                'timestamp': timestamp,
                'current_price': metrics['current_price'],
                'position_in_range_pct': metrics['position_in_range'],
                'momentum_signal': metrics['momentum_signal'] if metrics['momentum_signal'] else 'None',
                'long_signal': 1 if metrics['momentum_signal'] == 'LONG' else 0,
                'short_signal': 1 if metrics['momentum_signal'] == 'SHORT' else 0
            }
            
            self._append_to_csv(os.path.join(directories['signals'], 'signal_metrics.csv'), signal_data)
            
            # 4. Price Analysis CSV
            price_data = {
                'timestamp': timestamp,
                'open': current_bar['open'],
                'high': current_bar['high'],
                'low': current_bar['low'],
                'close': current_bar['close'],
                'volume': current_bar['volume'] if 'volume' in current_bar else 0,
                'prev_close': metrics['prev_close'],
                'intraday_range': current_bar['high'] - current_bar['low'],
                'intraday_range_pct': ((current_bar['high'] - current_bar['low']) / current_bar['close']) * 100,
                'gap_from_prev': current_bar['open'] - metrics['prev_close'],
                'gap_pct': ((current_bar['open'] - metrics['prev_close']) / metrics['prev_close']) * 100,
                'body_size': abs(current_bar['close'] - current_bar['open']),
                'upper_wick': current_bar['high'] - max(current_bar['open'], current_bar['close']),
                'lower_wick': min(current_bar['open'], current_bar['close']) - current_bar['low'],
                'bullish_candle': 1 if current_bar['close'] > current_bar['open'] else 0
            }
            
            self._append_to_csv(os.path.join(directories['price_analysis'], 'price_analysis.csv'), price_data)
            
            # 5. Comprehensive Indicators CSV (all data in one file)
            comprehensive_data = {**volatility_data, **band_data, **signal_data, **price_data}
            # Remove duplicate timestamp columns
            comprehensive_data = {k: v for k, v in comprehensive_data.items() if not (k.startswith('timestamp') and k != 'timestamp')}
            
            self._append_to_csv(os.path.join(directories['comprehensive'], 'all_indicators.csv'), comprehensive_data)
            
        except Exception as e:
            self.logger.error(f"Error saving technical indicators row to CSV: {str(e)}")

    def _append_to_csv(self, filepath, data):
        """Append data to CSV file, creating headers if file doesn't exist"""
        try:
            file_exists = os.path.isfile(filepath)
            
            # Convert data to DataFrame, ensuring timestamp is preserved correctly
            df_new = pd.DataFrame([data])
            
            # Debug: Log the timestamp being saved
            if 'timestamp' in data:
                self.logger.debug(f"Saving timestamp to CSV: {data['timestamp']} (type: {type(data['timestamp'])})")
            
            if file_exists:
                # Append to existing file
                df_new.to_csv(filepath, mode='a', header=False, index=False, date_format='%Y-%m-%d %H:%M:%S')
            else:
                # Create new file with headers
                df_new.to_csv(filepath, mode='w', header=True, index=False, date_format='%Y-%m-%d %H:%M:%S')
                
        except Exception as e:
            self.logger.error(f"Error appending to CSV {filepath}: {str(e)}")

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
    output_file = f'data/NVDA_with_indicators.csv'
    
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
