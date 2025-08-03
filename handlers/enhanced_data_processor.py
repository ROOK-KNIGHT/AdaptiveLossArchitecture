#!/usr/bin/env python3
"""
Enhanced Data Processor - Unified interface for fetching and processing stock data
Integrates live data fetching with comprehensive technical analysis
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.unified_preprocessing import UnifiedPreprocessor
from historical_data_handler import HistoricalDataHandler

class EnhancedDataProcessor:
    """
    Enhanced data processor that combines live data fetching with comprehensive preprocessing
    """
    
    def __init__(self):
        """Initialize the enhanced data processor"""
        self.data_handler = HistoricalDataHandler()
    
    def process_symbol_live(self, symbol: str, days_lookback: int = 365, 
                           target_horizon: int = 1, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Process a symbol with live data fetching and full technical analysis
        
        Args:
            symbol: Stock symbol to process (e.g., 'NVDA', 'AAPL')
            days_lookback: Number of days to fetch (default: 365)
            target_horizon: Number of periods ahead to predict (default: 1)
            test_size: Proportion of data for testing (default: 0.2)
            
        Returns:
            Dictionary with processed data, scalers, and metadata
        """
        print("="*80)
