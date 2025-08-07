"""
Enhanced Adaptive Stock Price Predictor with Technical Indicators
Specialized for pre-processed technical indicator datasets

Usage:
    python3 src/models/enhanced_adaptive_predictor.py
    
Features:
    - Uses comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Covariance-based adaptive loss function with technical indicator categories
    - Intelligent feature selection from 35+ technical indicators
    - Enhanced analysis with trading-relevant insights
    - No API dependencies - works with pre-processed CSV data
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
import sys
import os
import yaml
from datetime import datetime
warnings.filterwarnings('ignore')

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

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import unified preprocessing and targets
from src.data_processing.unified_preprocessing import UnifiedPreprocessor
from src.data_processing.unified_targets import UnifiedTargetManager

# Import dead neuron monitoring system
from src.utils.dead_neuron_monitor import DeadNeuronMonitor, integrate_with_training_loop, monitor_enhanced_adaptive_predictor

# Define evaluation functions (same as other models)
def comprehensive_model_evaluation(model, X_test, y_test, scaler_y, model_name, train_test_mean_diff=0):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    
    # Convert back to original scale
    predictions_orig = scaler_y.inverse_transform(predictions.numpy())
    y_test_orig = scaler_y.inverse_transform(y_test.numpy())
    
    # Basic metrics
    mae = np.mean(np.abs(predictions_orig - y_test_orig))
    mse = np.mean((predictions_orig - y_test_orig) ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(predictions_orig - y_test_orig)
    
    # Bias-corrected MAE
    bias_corrected_mae = mae  # For now, same as MAE
    
    # Directional accuracy (same calculation as other models)
    predicted_direction = np.sign(predictions_orig.flatten())
    actual_direction = np.sign(y_test_orig.flatten())
    directional_accuracy = np.mean(actual_direction == predicted_direction)
    
    return {
        'mae': mae,
        'mse': mse, 
        'rmse': rmse,
        'bias': bias,
        'bias_corrected_mae': bias_corrected_mae,
        'predictions': predictions_orig,
        'actuals': y_test_orig,
        'bias_metrics': {'bias': bias, 'corrected_mae': bias_corrected_mae},
        'directional_accuracy': directional_accuracy
    }

def compare_models_comprehensive(adaptive_results, standard_results):
    print(f"\nModel Comparison:")
    print(f"Adaptive MAE: ${adaptive_results['mae']:.4f}")
    print(f"Standard MAE: ${standard_results['mae']:.4f}")

class SimplifiedAdaptiveLossFunction(nn.Module):
    """
    Simplified Adaptive Loss Function with Ablation Capabilities
    Focus on covariance-based learning with optional penalty terms for ablation study
    """
    
    def __init__(self, feature_names: List[str], feature_categories: Dict[str, str] = None, 
                 initial_weights: Dict[str, float] = None, 
                 enable_temporal_penalty: bool = None,
                 enable_volatility_penalty: bool = None,
                 enable_smoothness_penalty: bool = None):
        super(SimplifiedAdaptiveLossFunction, self).__init__()
        self.feature_names = feature_names
        self.mse_loss = nn.MSELoss()
        
        # Load config values with fallbacks
        adaptive_config = CONFIG['adaptive_loss']
        
        # Ablation flags from config
        self.enable_temporal_penalty = (enable_temporal_penalty if enable_temporal_penalty is not None 
                                       else adaptive_config['penalties']['temporal']['enabled'])
        self.enable_volatility_penalty = (enable_volatility_penalty if enable_volatility_penalty is not None 
                                         else adaptive_config['penalties']['volatility']['enabled'])
        self.enable_smoothness_penalty = (enable_smoothness_penalty if enable_smoothness_penalty is not None 
                                         else adaptive_config['penalties']['smoothness']['enabled'])
        
        # Categorize technical indicators
        self.feature_categories = feature_categories or self._auto_categorize_features(feature_names)
        
        # Initialize weights with config-based approach
        if initial_weights is None:
            self.weights = self._initialize_category_weights()
        else:
            self.weights = initial_weights.copy()
        
        # Weight constraints from config
        self.min_weight = adaptive_config['min_weight']
        self.max_weight = adaptive_config['max_weight']
        self.learning_rate_factor = adaptive_config['learning_rate_factor']
        self.weight_update_interval = adaptive_config['weight_update_interval']
        self.weight_update_adjustment = adaptive_config['weight_update_adjustment']
        
        # Penalty weights from config
        self.temporal_weight = (adaptive_config['penalties']['temporal']['weight'] 
                               if self.enable_temporal_penalty else 0.0)
        self.volatility_weight = (adaptive_config['penalties']['volatility']['weight'] 
                                 if self.enable_volatility_penalty else 0.0)
        self.smoothness_weight = (adaptive_config['penalties']['smoothness']['weight'] 
                                 if self.enable_smoothness_penalty else 0.0)
        
        # Tracking
        self.covariance_history = {name: [] for name in feature_names}
        self.weight_history = {name: [] for name in feature_names}
        
    def _auto_categorize_features(self, feature_names: List[str]) -> Dict[str, str]:
        """Automatically categorize technical indicators"""
        categories = {}
        
        for name in feature_names:
            name_lower = name.lower()
            if any(x in name_lower for x in ['sma', 'ema', 'ma_', 'bb_middle', 'vwap', 'psar']):
                categories[name] = 'trend'
            elif any(x in name_lower for x in ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'momentum', 'roc']):
                categories[name] = 'momentum'
            elif any(x in name_lower for x in ['volume', 'obv', 'adl']):
                categories[name] = 'volume'
            elif any(x in name_lower for x in ['atr', 'volatility', 'bb_width', 'bb_upper', 'bb_lower']):
                categories[name] = 'volatility'
            elif any(x in name_lower for x in ['macd', 'bb_position']):
                categories[name] = 'signal'
            else:
                categories[name] = 'other'
                
        return categories
    
    def _initialize_category_weights(self) -> Dict[str, float]:
        """Initialize weights based on technical indicator categories from config"""
        weights = {}
        category_base_weights = CONFIG['adaptive_loss']['category_base_weights']
        
        for name in self.feature_names:
            category = self.feature_categories[name]
            weights[name] = category_base_weights.get(category, category_base_weights.get('other', 0.05))
            
        return weights
    
    def compute_covariance_loss(self, features: torch.Tensor, target: torch.Tensor, 
                               feature_idx: int) -> torch.Tensor:
        """Compute covariance-based loss term for a specific technical indicator"""
        feature_col = features[:, feature_idx]
        
        # Center the data (subtract mean)
        feature_centered = feature_col - torch.mean(feature_col)
        target_centered = target.squeeze() - torch.mean(target.squeeze())
        
        # Compute covariance
        covariance = torch.mean(feature_centered * target_centered)
        
        # Return absolute covariance as loss term
        return torch.abs(covariance)
    
    def compute_momentum_consistency_loss(self, predictions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Compute momentum consistency penalty for momentum indicators"""
        if len(predictions) < 3:
            return torch.tensor(0.0)
        
        # Find momentum indicator indices
        momentum_indices = [i for i, name in enumerate(self.feature_names) 
                          if self.feature_categories[name] == 'momentum']
        
        if not momentum_indices:
            return torch.tensor(0.0)
        
        # Calculate momentum consistency
        pred_momentum = torch.diff(predictions.squeeze())
        momentum_loss = torch.tensor(0.0)
        
        for idx in momentum_indices:
            indicator_momentum = torch.diff(features[:, idx])
            if len(indicator_momentum) == len(pred_momentum):
                # Penalize when prediction momentum contradicts indicator momentum
                momentum_alignment = torch.mean(torch.abs(
                    torch.sign(pred_momentum) - torch.sign(indicator_momentum)
                ))
                momentum_loss += momentum_alignment
        
        return momentum_loss / len(momentum_indices) if momentum_indices else torch.tensor(0.0)
    
    def compute_temporal_consistency_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency penalty"""
        if len(predictions) < 2:
            return torch.tensor(0.0)
        
        pred_diffs = torch.diff(predictions.squeeze())
        temporal_loss = torch.mean(torch.abs(pred_diffs))
        return temporal_loss
    
    def compute_volatility_penalty(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute volatility penalty"""
        if len(predictions) < 3:
            return torch.tensor(0.0)
        
        pred_volatility = torch.std(torch.diff(predictions.squeeze()))
        target_volatility = torch.std(torch.diff(target.squeeze()))
        volatility_loss = torch.abs(pred_volatility - target_volatility)
        return volatility_loss
    
    def compute_smoothness_penalty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute smoothness penalty using second derivatives"""
        if len(predictions) < 3:
            return torch.tensor(0.0)
        
        pred_squeeze = predictions.squeeze()
        first_diff = torch.diff(pred_squeeze)
        second_diff = torch.diff(first_diff)
        smoothness_loss = torch.mean(torch.abs(second_diff))
        return smoothness_loss

    def forward(self, predictions: torch.Tensor, target: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Enhanced forward pass with technical indicator specific penalties"""
        # Base MSE loss
        mse_loss = self.mse_loss(predictions, target)
        
        # Covariance losses for each technical indicator
        correlation_losses = {}
        total_correlation_loss = 0.0
        category_losses = {cat: 0.0 for cat in set(self.feature_categories.values())}
        
        for i, feature_name in enumerate(self.feature_names):
            cov_loss = self.compute_covariance_loss(features, target, i)
            correlation_losses[feature_name] = cov_loss.item()
            total_correlation_loss += self.weights[feature_name] * cov_loss
            
            # Track category performance
            category = self.feature_categories[feature_name]
            category_losses[category] += cov_loss.item()
        
        # Enhanced penalty terms
        temporal_loss = self.compute_temporal_consistency_loss(predictions)
        volatility_loss = self.compute_volatility_penalty(predictions, target)
        smoothness_loss = self.compute_smoothness_penalty(predictions)
        momentum_loss = self.compute_momentum_consistency_loss(predictions, features)
        
        # Simplified total loss (focus on covariance + optional penalties)
        total_loss = (mse_loss + 
                     total_correlation_loss + 
                     self.temporal_weight * temporal_loss +
                     self.volatility_weight * volatility_loss +
                     self.smoothness_weight * smoothness_loss)
        
        # Store metrics for analysis
        correlation_losses['temporal_consistency'] = temporal_loss.item()
        correlation_losses['volatility_penalty'] = volatility_loss.item()
        correlation_losses['smoothness_penalty'] = smoothness_loss.item()
        
        # Store covariances for weight updates
        for feature_name, cov_val in correlation_losses.items():
            if feature_name in self.covariance_history:
                self.covariance_history[feature_name].append(cov_val)
        
        return total_loss, correlation_losses
    
    def update_weights(self, epoch: int):
        """Simplified weight update based on covariance performance"""
        if epoch % self.weight_update_interval != 0 or epoch == 0:  # Update based on config interval
            return
        
        print(f"\nUpdating weights at epoch {epoch}:")
        
        for feature_name in self.feature_names:
            if len(self.covariance_history[feature_name]) > 5:
                # Get recent covariances
                recent_covs = self.covariance_history[feature_name][-5:]
                avg_cov = np.mean(recent_covs)
                
                # Simple sigmoid normalization
                normalized_cov = 1 / (1 + np.exp(-avg_cov * 10))
                
                # Update weight using config values
                old_weight = self.weights[feature_name]
                adjustment = self.learning_rate_factor * normalized_cov * self.weight_update_adjustment
                new_weight = old_weight + adjustment
                
                # Apply constraints
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                
                self.weights[feature_name] = new_weight
                self.weight_history[feature_name].append(new_weight)
                
                print(f"  {feature_name}: {old_weight:.4f} -> {new_weight:.4f} (cov: {avg_cov:.4f})")

class StockPricePredictor(nn.Module):
    """Enhanced neural network for stock price prediction with technical indicators"""
    
    def __init__(self, input_size: int, hidden_size: int = None):
        super(StockPricePredictor, self).__init__()
        
        # Use config values with fallback
        if hidden_size is None:
            hidden_size = CONFIG['model']['hidden_size']
        
        dropout_rates = CONFIG['model']['dropout_rates']
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rates['layer1']),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rates['layer2']),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rates['layer3']),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def get_data_from_unified_preprocessing(symbol: str = 'NVDA', days_lookback: int = 365, 
                                      target_horizon: int = 1, test_size: float = 0.2) -> Dict:
    """
    Get standardized data directly from unified preprocessing pipeline
    
    Args:
        symbol: Stock symbol to fetch (e.g., 'NVDA', 'AAPL')
        days_lookback: Number of days to fetch for live data (default: 365)
        target_horizon: Number of periods ahead to predict (default: 1 day)
        test_size: Proportion of data for testing (default: 0.2)
        
    Returns:
        Dictionary with all processed data, scalers, and metadata from unified preprocessing
    """
    print(f"🔄 Getting standardized data from unified preprocessing pipeline...")
    print(f"Symbol: {symbol}, Days: {days_lookback}, Horizon: {target_horizon}")
    
    # Initialize unified preprocessor with live data fetching
    processor = UnifiedPreprocessor(
        symbol=symbol,
        target_horizon=target_horizon,
        test_size=test_size,
        lag_periods=[1, 2, 3],  # Use same lag periods as original
        days_lookback=days_lookback
    )
    
    # Get standardized data (this will fetch live data and process it)
    standardized_data = processor.get_standardized_data()
    
    print(f"✅ Received standardized data with {standardized_data['total_features']} features")
    print(f"   Training samples: {len(standardized_data['X_train'])}")
    print(f"   Test samples: {len(standardized_data['X_test'])}")
    print(f"   Dual targets: Returns + Prices")
    
    return standardized_data

def select_best_features(df: pd.DataFrame, target_col: str = 'target_return', 
                        max_features: int = 15, enforce_diversity: bool = True) -> List[str]:
    """
    Intelligently select the best technical indicators based on correlation and diversity
    Uses technical indicators configuration for category definitions
    
    Args:
        df: DataFrame with technical indicators
        target_col: Target column name (now defaults to target_return)
        max_features: Maximum number of features to select
        enforce_diversity: Force minimum per category for trading edge
    """
    print(f"\nSelecting best {max_features} technical indicators for RETURN PREDICTION...")
    print(f"📋 Using technical indicators configuration from: config/technical_indicators_config.yaml")
    
    # Calculate correlations with target
    correlations = df.corr()[target_col].abs().sort_values(ascending=False)
    
    # Remove target and reference columns from correlations
    exclude_cols = [target_col, 'target_price', 'current_close', 'datetime']
    for col in exclude_cols:
        if col in correlations.index:
            correlations = correlations.drop(col)
    
    print("Top correlated indicators with RETURNS:")
    for i, (indicator, corr) in enumerate(correlations.head(20).items(), 1):
        print(f"  {i:2d}. {indicator}: {corr:.4f}")
    
    # Get category definitions from technical indicators config
    categories = {}
    
    # Extract category keywords from all indicator sections
    for section_name in ['trend_indicators', 'momentum_indicators', 'volume_indicators', 
                        'volatility_indicators', 'signal_indicators', 'advanced_indicators', 'derived_indicators']:
        if section_name in INDICATORS_CONFIG:
            section = INDICATORS_CONFIG[section_name]
            for indicator_name, indicator_config in section.items():
                if indicator_config.get('enabled', False):
                    category = indicator_config.get('category', 'other')
                    if category not in categories:
                        categories[category] = []
                    # Add the indicator name as a keyword for categorization
                    categories[category].append(indicator_name.lower())
    
    # Add traditional keyword-based categorization as fallback
    traditional_categories = {
        'trend': ['sma', 'ema', 'ma_', 'bb_middle', 'vwap', 'psar'],
        'momentum': ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'momentum', 'roc'],
        'volume': ['volume', 'obv', 'adl'],
        'volatility': ['atr', 'volatility', 'bb_width', 'bb_upper', 'bb_lower'],
        'signal': ['macd', 'bb_position']
    }
    
    # Merge traditional categories with config-based categories
    for cat, keywords in traditional_categories.items():
        if cat not in categories:
            categories[cat] = []
        categories[cat].extend(keywords)
    
    print(f"\n📊 Available indicator categories from config:")
    for cat, keywords in categories.items():
        print(f"  {cat}: {len(keywords)} keywords")
    
    # Enhanced category-aware selection for trading edge
    selected_features = []
    
    if enforce_diversity:
        # FORCE minimum per category for trading signals
        min_per_category = max(1, max_features // (len(categories) + 1))  # +1 for 'other'
        print(f"\nENFORCING diversity: minimum {min_per_category} per category")
        
        category_counts = {cat: 0 for cat in categories.keys()}
        category_counts['other'] = 0
        
        # First pass: ensure minimum per category
        for cat in categories.keys():
            cat_indicators = []
            for indicator, corr in correlations.items():
                if any(keyword in indicator.lower() for keyword in categories[cat]):
                    cat_indicators.append((indicator, corr))
            
            # Sort by correlation and take top for this category
            cat_indicators.sort(key=lambda x: x[1], reverse=True)
            for indicator, corr in cat_indicators[:min_per_category]:
                if len(selected_features) < max_features:
                    selected_features.append(indicator)
                    category_counts[cat] += 1
        
        # Second pass: fill remaining slots with best correlations
        for indicator, corr in correlations.items():
            if len(selected_features) >= max_features:
                break
            if indicator not in selected_features:
                selected_features.append(indicator)
                
                # Update category count
                indicator_category = 'other'
                for cat, keywords in categories.items():
                    if any(keyword in indicator.lower() for keyword in keywords):
                        indicator_category = cat
                        break
                category_counts[indicator_category] += 1
    else:
        # Original selection logic
        category_counts = {cat: 0 for cat in categories.keys()}
        category_counts['other'] = 0
        max_per_category = max_features // len(categories) + 1
        
        for indicator, corr in correlations.items():
            if len(selected_features) >= max_features:
                break
                
            # Determine category
            indicator_category = 'other'
            for cat, keywords in categories.items():
                if any(keyword in indicator.lower() for keyword in keywords):
                    indicator_category = cat
                    break
            
            # Add if category not full or if correlation is very high
            if (category_counts.get(indicator_category, 0) < max_per_category or 
                corr > 0.7 or len(selected_features) < max_features // 2):
                selected_features.append(indicator)
                category_counts[indicator_category] += 1
    
    print(f"\nSelected {len(selected_features)} diverse technical indicators:")
    for i, feature in enumerate(selected_features, 1):
        corr = correlations[feature] if feature in correlations else 0.0
        print(f"  {i:2d}. {feature}: {corr:.4f}")
    
    print(f"\nCategory distribution:")
    for cat, count in category_counts.items():
        if count > 0:
            print(f"  {cat}: {count} indicators")
    
    return selected_features

def train_model(model, train_loader, loss_function, optimizer, epochs: int, 
                model_name: str, enable_dead_neuron_monitoring: bool = True) -> Dict[str, List[float]]:
    """Train model and return training history with optional dead neuron monitoring"""
    
    # Initialize dead neuron monitoring if enabled
    dead_neuron_monitor = None
    if enable_dead_neuron_monitoring:
        print(f"\n🔍 Initializing dead neuron monitoring for {model_name}...")
        dead_neuron_monitor = monitor_enhanced_adaptive_predictor(
            model, optimizer,
            threshold=0.01,
            monitoring_frequency=5,
            verbose=True,
            save_results=True
        )
    
    # Initialize history
    if hasattr(loss_function, 'feature_names'):
        all_loss_names = (loss_function.feature_names + 
                         ['temporal_consistency', 'volatility_penalty', 'smoothness_penalty', 'momentum_consistency'])
    else:
        all_loss_names = []
    
    history = {
        'loss': [],
        'mse_loss': [],
        'correlation_losses': {name: [] for name in all_loss_names},
        'dead_neuron_monitoring': dead_neuron_monitor is not None
    }
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_corr_losses = {name: 0.0 for name in all_loss_names}
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            
            predictions = model(batch_features)
            
            if hasattr(loss_function, 'feature_names'):
                # Enhanced adaptive loss function
                loss, corr_losses = loss_function(predictions, batch_targets, batch_features)
                
                # Track all correlation losses
                for name, val in corr_losses.items():
                    if name in epoch_corr_losses:
                        epoch_corr_losses[name] += val
                
                # Also compute MSE for comparison
                mse_loss = nn.MSELoss()(predictions, batch_targets)
                epoch_mse += mse_loss.item()
                
            else:
                # Standard loss function
                loss = loss_function(predictions, batch_targets)
                epoch_mse += loss.item()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Update dead neuron monitoring
            if dead_neuron_monitor:
                dead_neuron_monitor.update_batch_count()
                dead_neuron_monitor.track_learning_rate(optimizer)
        
        # Average losses over batches
        avg_loss = epoch_loss / len(train_loader)
        avg_mse = epoch_mse / len(train_loader)
        
        history['loss'].append(avg_loss)
        history['mse_loss'].append(avg_mse)
        
        # Average correlation losses
        for name in epoch_corr_losses.keys():
            avg_corr_loss = epoch_corr_losses[name] / len(train_loader)
            history['correlation_losses'][name].append(avg_corr_loss)
        
        # Update weights for adaptive loss
        if hasattr(loss_function, 'update_weights'):
            loss_function.update_weights(epoch)
        
        # Update dead neuron monitoring
        if dead_neuron_monitor:
            dead_neuron_monitor.update_epoch_count()
        
        # Print progress using config interval
        progress_interval = CONFIG['training']['progress_report_interval']
        if epoch % progress_interval == 0 or epoch == epochs - 1:
            if hasattr(loss_function, 'feature_names'):
                temporal_loss = history['correlation_losses'].get('temporal_consistency', [0])[-1]
                volatility_loss = history['correlation_losses'].get('volatility_penalty', [0])[-1]
                smoothness_loss = history['correlation_losses'].get('smoothness_penalty', [0])[-1]
                momentum_loss = history['correlation_losses'].get('momentum_consistency', [0])[-1]
                print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.6f}, MSE = {avg_mse:.6f}")
                print(f"    Penalties - Temporal: {temporal_loss:.6f}, Volatility: {volatility_loss:.6f}, "
                      f"Smoothness: {smoothness_loss:.6f}, Momentum: {momentum_loss:.6f}")
            else:
                print(f"{model_name} - Epoch {epoch:3d}: Loss = {avg_loss:.6f}, MSE = {avg_mse:.6f}")
    
    # Cleanup dead neuron monitoring and get final report
    if dead_neuron_monitor:
        print(f"\n🔍 Dead Neuron Monitoring Final Report:")
        final_report = dead_neuron_monitor.get_summary_report()
        print(f"   Overall Health: {final_report['overall_health'].upper()}")
        print(f"   Critical Layers: {sum(1 for layer in final_report['layer_analysis'].values() if layer.get('status') == 'critical')}")
        print(f"   Warning Layers: {sum(1 for layer in final_report['layer_analysis'].values() if layer.get('status') == 'warning')}")
        print(f"   Healthy Layers: {sum(1 for layer in final_report['layer_analysis'].values() if layer.get('status') == 'healthy')}")
        
        if final_report['recommendations']:
            print(f"   Recommendations:")
            for rec in final_report['recommendations']:
                print(f"     • {rec}")
        
        # Store monitoring results in history
        history['dead_neuron_report'] = final_report
        
        # Cleanup
        dead_neuron_monitor.cleanup()
    
    return history

def analyze_feature_importance(adaptive_loss, selected_features):
    """Analyze and display feature importance from adaptive loss function"""
    print("\n" + "="*80)
    print("TECHNICAL INDICATOR IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Final weights
    final_weights = [(name, weight) for name, weight in adaptive_loss.weights.items()]
    final_weights.sort(key=lambda x: x[1], reverse=True)
    
    print("\nFinal Technical Indicator Weights (Importance Ranking):")
    for i, (indicator, weight) in enumerate(final_weights, 1):
        category = adaptive_loss.feature_categories[indicator]
        print(f"  {i:2d}. {indicator:20s} ({category:10s}): {weight:.6f}")
    
    # Category performance
    print(f"\nCategory Performance Summary:")
    category_weights = {}
    for indicator, weight in adaptive_loss.weights.items():
        category = adaptive_loss.feature_categories[indicator]
        category_weights[category] = category_weights.get(category, 0) + weight
    
    sorted_categories = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)
    for category, total_weight in sorted_categories:
        avg_weight = total_weight / sum(1 for cat in adaptive_loss.feature_categories.values() if cat == category)
        print(f"  {category:12s}: Total Weight = {total_weight:.4f}, Avg Weight = {avg_weight:.4f}")
    
    # Weight evolution for top indicators
    print(f"\nTop 5 Indicator Weight Evolution:")
    for indicator, _ in final_weights[:5]:
        if adaptive_loss.weight_history[indicator]:
            initial_weight = adaptive_loss.weight_history[indicator][0] if adaptive_loss.weight_history[indicator] else adaptive_loss.weights[indicator]
            final_weight = adaptive_loss.weights[indicator]
            change = ((final_weight - initial_weight) / initial_weight * 100) if initial_weight > 0 else 0
            print(f"  {indicator:20s}: {initial_weight:.6f} -> {final_weight:.6f} ({change:+.1f}%)")

def main(symbol: str = None, days_lookback: int = None, target_horizon: int = None):
    """Main function for enhanced adaptive predictor with technical indicators"""
    
    # Load config values with fallbacks
    if symbol is None:
        symbol = 'NVDA'  # Default symbol
    if days_lookback is None:
        days_lookback = CONFIG['data_processing']['days_lookback']
    if target_horizon is None:
        target_horizon = CONFIG['data_processing']['target_horizon']
    
    test_size = CONFIG['data_processing']['test_size']
    
    print("=" * 80)
    print("ENHANCED ADAPTIVE STOCK PRICE PREDICTOR WITH UNIFIED TARGETS")
    print("=" * 80)
    print("Features: Live data fetching + 60+ technical indicators + adaptive learning")
    print(f"Symbol: {symbol}, Days: {days_lookback}, Horizon: {target_horizon}\n")
    print(f"📋 Using configuration from: config/model_config.yaml")
    
    # Initialize unified target manager for validation
    print("🎯 Initializing unified target manager...")
    target_manager = UnifiedTargetManager(
        target_horizon=target_horizon,
        primary_target='returns',
        secondary_targets=['prices', 'direction']
    )
    
    # Get target configuration for this model
    model_target_config = target_manager.create_target_config_for_model('adaptive')
    
    # Validate target consistency
    target_manager.validate_target_consistency(model_target_config, "Enhanced Adaptive Predictor")
    
    # 1. Get standardized data from unified preprocessing pipeline
    print("🔄 Step 1: Getting data from unified preprocessing pipeline...")
    try:
        data = get_data_from_unified_preprocessing(
            symbol=symbol,
            days_lookback=days_lookback,
            target_horizon=target_horizon,
            test_size=test_size
        )
    except Exception as e:
        print(f"❌ Error getting data from unified preprocessing: {str(e)}")
        return
    
    # 2. Extract data components (already preprocessed and scaled)
    print("\n📊 Step 2: Extracting preprocessed data components...")
    
    # Get scaled training and test data (for RETURNS prediction)
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_returns_train_scaled = data['y_returns_train_scaled']
    y_returns_test_scaled = data['y_returns_test_scaled']
    
    # Get feature names and scalers
    feature_names = data['feature_names']
    scaler_y_returns = data['scaler_y_returns']
    train_test_returns_diff = data['train_test_returns_diff']
    
    print(f"✅ Data extracted successfully:")
    print(f"   Features: {len(feature_names)} (lagged technical indicators)")
    print(f"   Training samples: {len(X_train_scaled)}")
    print(f"   Test samples: {len(X_test_scaled)}")
    print(f"   Target: Returns (log returns)")
    print(f"   Distribution shift: {train_test_returns_diff:.6f}")
    
    # 3. Feature selection from preprocessed features
    print(f"\n🎯 Step 3: Intelligent feature selection...")
    
    # Create a temporary DataFrame for feature selection
    temp_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    temp_df['target_return'] = y_returns_train_scaled
    
    # Get feature selection config
    feature_config = CONFIG['feature_selection']
    
    # Select best features (use scaled data for correlation analysis)
    selected_features = select_best_features(
        temp_df, 
        target_col='target_return', 
        max_features=feature_config['max_features'], 
        enforce_diversity=feature_config['enforce_category_diversity']
    )
    
    # Get indices of selected features
    feature_indices = [feature_names.index(feat) for feat in selected_features]
    
    # Extract selected features from scaled data
    X_train_selected = X_train_scaled[:, feature_indices]
    X_test_selected = X_test_scaled[:, feature_indices]
    
    print(f"✅ Selected {len(selected_features)} features from {len(feature_names)} available")
    
    # 4. Convert to PyTorch tensors
    print(f"\n🔧 Step 4: Converting to PyTorch tensors...")
    X_train_tensor = torch.FloatTensor(X_train_selected)
    y_train_tensor = torch.FloatTensor(y_returns_train_scaled.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test_selected)
    y_test_tensor = torch.FloatTensor(y_returns_test_scaled.reshape(-1, 1))
    
    # Get training config
    training_config = CONFIG['training']
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=training_config['shuffle']
    )
    
    print(f"✅ PyTorch tensors created successfully")
    
    # 5. Initialize adaptive model and loss function
    print(f"\n🤖 Step 5: Initializing enhanced adaptive model...")
    
    # Enhanced adaptive model with capacity for selected technical indicators
    adaptive_model = StockPricePredictor(input_size=len(selected_features))
    
    # Adaptive loss function with selected features (using config defaults)
    adaptive_loss = SimplifiedAdaptiveLossFunction(feature_names=selected_features)
    
    # Optimizer using config values
    adaptive_optimizer = optim.Adam(
        adaptive_model.parameters(), 
        lr=training_config['learning_rate'], 
        weight_decay=training_config['weight_decay']
    )
    
    print(f"✅ Enhanced adaptive model initialized with {len(selected_features)} input features")
    
    # 6. Train adaptive model
    print(f"\n🏋️ Step 6: Training enhanced adaptive model...")
    epochs = training_config['epochs']
    
    print(f"\nTraining Enhanced Adaptive Model:")
    adaptive_history = train_model(
        adaptive_model, train_loader, adaptive_loss, adaptive_optimizer, 
        epochs, "Enhanced Adaptive"
    )
    
    # 7. Model Evaluation
    print(f"\n📈 Step 7: Enhanced adaptive model evaluation...")
    
    # Evaluate adaptive model
    adaptive_results = comprehensive_model_evaluation(
        adaptive_model, X_test_tensor, y_test_tensor, scaler_y_returns, 
        "Enhanced Adaptive Model", train_test_mean_diff=train_test_returns_diff
    )
    
    # 9. Technical Indicator Analysis
    print(f"\n🔍 Step 9: Feature importance analysis...")
    analyze_feature_importance(adaptive_loss, selected_features)
    
    # 8. Final Summary
    print("\n" + "="*80)
    print("ENHANCED ADAPTIVE PREDICTOR SUMMARY")
    print("="*80)
    
    # Extract metrics safely
    adaptive_mae = adaptive_results.get('bias_corrected_mae', adaptive_results.get('mae', 0))
    adaptive_bias = adaptive_results.get('bias_metrics', {}).get('bias', 0)
    adaptive_dir = adaptive_results.get('directional_accuracy', 0)
    
    print(f"\n📊 Model Performance:")
    print(f"  Enhanced Adaptive MAE: {adaptive_mae:.6f}")
    print(f"  Model Bias: {adaptive_bias:.6f}")
    print(f"  Directional Accuracy: {adaptive_dir:.2%}")
    
    # Top performing technical indicators
    final_weights = [(name, weight) for name, weight in adaptive_loss.weights.items()]
    final_weights.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 Top 5 Most Important Technical Indicators:")
    for i, (indicator, weight) in enumerate(final_weights[:5], 1):
        category = adaptive_loss.feature_categories[indicator]
        print(f"  {i}. {indicator} ({category}): {weight:.6f}")
    
    print(f"\n✅ Enhanced adaptive predictor analysis complete!")
    print(f"📊 Used {len(selected_features)} features from {data['total_features']} available")
    print(f"🔄 Live data from unified preprocessing: {symbol} ({days_lookback} days)")
    print(f"🎯 Target horizon: {target_horizon} day(s)")
    
    # Key insights
    print(f"\n🔍 Key Insights:")
    if len(final_weights) >= 2:
        print(f"  • Top indicators: {final_weights[0][0]} (weight: {final_weights[0][1]:.6f})")
        print(f"                   {final_weights[1][0]} (weight: {final_weights[1][1]:.6f})")
    
    trend_count = sum(1 for name, _ in final_weights if adaptive_loss.feature_categories[name] == 'trend')
    print(f"  • Trend indicators: {trend_count}/{len(final_weights)} selected features")
    print(f"  • Data source: Live {symbol} data via unified preprocessing")
    print(f"  • TRUE FORECASTING: {target_horizon}-day ahead prediction using lagged indicators")
    print(f"  • Enhanced adaptive model with {len(selected_features)} selected features")
    
    # Save comprehensive results to JSON
    results_data = {
        'model_name': 'Enhanced Adaptive Predictor',
        'model_type': 'neural_network_adaptive',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'symbol': symbol,
            'days_lookback': days_lookback,
            'target_horizon': target_horizon,
            'test_size': test_size,
            'epochs': epochs,
            'selected_features_count': len(selected_features),
            'total_features_available': data['total_features']
        },
        'performance_metrics': {
            'mae': adaptive_results.get('mae', 0),
            'rmse': adaptive_results.get('rmse', 0),
            'bias': adaptive_results.get('bias_metrics', {}).get('bias', 0),
            'bias_corrected_mae': adaptive_results.get('bias_corrected_mae', adaptive_results.get('mae', 0)),
            'directional_accuracy': adaptive_results.get('directional_accuracy', 0),
            'mse': adaptive_results.get('mse', 0)
        },
        'feature_analysis': {
            'selected_features': selected_features,
            'feature_weights': dict(adaptive_loss.weights),
            'feature_categories': dict(adaptive_loss.feature_categories),
            'top_5_features': [(name, weight) for name, weight in sorted(adaptive_loss.weights.items(), key=lambda x: x[1], reverse=True)[:5]]
        },
        'training_history': {
            'loss_history': adaptive_history['loss'],
            'mse_history': adaptive_history['mse_loss'],
            'final_loss': adaptive_history['loss'][-1] if adaptive_history['loss'] else 0,
            'epochs_trained': len(adaptive_history['loss'])
        },
        'data_summary': {
            'training_samples': len(X_train_scaled),
            'test_samples': len(X_test_scaled),
            'train_test_distribution_shift': train_test_returns_diff
        },
        'predictions': {
            'test_predictions': adaptive_results.get('predictions', []).tolist() if hasattr(adaptive_results.get('predictions', []), 'tolist') else list(adaptive_results.get('predictions', [])),
            'test_actuals': adaptive_results.get('actuals', []).tolist() if hasattr(adaptive_results.get('actuals', []), 'tolist') else list(adaptive_results.get('actuals', []))
        }
    }
    
    # Save to JSON file in organized directory structure (overwrite previous results)
    import json
    
    # Ensure the results directory exists
    results_dir = os.path.join('data', 'results', 'json')
    os.makedirs(results_dir, exist_ok=True)
    
    json_filename = os.path.join(results_dir, "enhanced_adaptive_predictor_results.json")
    
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"✅ Results saved to JSON: {json_filename}")
    
    return results_data

if __name__ == "__main__":
    main()
