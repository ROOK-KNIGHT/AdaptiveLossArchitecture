"""
Show Actual Prediction Values vs Actual Values
Displays the actual prediction numbers for returns and converted prices

Usage:
    python3 src/data_processing/show_prediction_values.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

def load_model_predictions():
    """Load predictions from the Enhanced Adaptive Predictor"""
    try:
        with open('data/results/json/enhanced_adaptive_predictor_results.json', 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("❌ Enhanced Adaptive Predictor results not found")
        return None

def convert_returns_to_prices(return_predictions, return_actuals, starting_price=150.0):
    """Convert log returns to actual price predictions"""
    
    # Convert log returns to prices
    predicted_prices = [starting_price]
    actual_prices = [starting_price]
    
    for i in range(len(return_predictions)):
