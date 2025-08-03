"""
Unified Target Configuration System
Ensures all models use consistent prediction targets and objectives

This module provides a centralized way to define and manage prediction targets
across all models in the AdaptiveLossArchitecture project.

Usage:
    from src.data_processing.unified_targets import UnifiedTargetManager
    
    # Initialize with default config
    target_manager = UnifiedTargetManager()
    
    # Get target configuration
    config = target_manager.get_target_config()
    
    # Create targets from data
    targets = target_manager.create_targets(df)
    
    # Validate target consistency
    target_manager.validate_target_consistency(model_targets)

Features:
    - Centralized target definition
    - Multiple target types (returns, prices, volatility, direction)
    - Consistent horizon and scaling
    - Target validation and verification
    - Configuration management
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'model_config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load global configuration
CONFIG = load_config()

class UnifiedTargetManager:
    """
    Unified Target Manager for consistent prediction objectives across all models
    """
    
    def __init__(self, target_horizon: int = None, primary_target: str = None, 
                 secondary_targets: List[str] = None):
        """
        Initialize the unified target manager
        
        Args:
            target_horizon: Number of periods ahead to predict (uses config default if None)
            primary_target: Primary target type ('returns', 'prices', 'volatility', 'direction')
            secondary_targets: List of secondary targets to also compute
        """
        # Load config values with fallbacks
        data_config = CONFIG['data_processing']
        target_config = CONFIG.get('unified_targets', {
            'primary_target': 'returns',
            'secondary_targets': ['prices', 'direction'],
            'target_horizon': 1,
            'returns_type': 'log_returns',
            'volatility_window': 20,
            'direction_threshold': 0.001,
            'price_scaling': 'standard',
            'validation_enabled': True
        })
        
        self.target_horizon = target_horizon if target_horizon is not None else target_config['target_horizon']
        self.primary_target = primary_target if primary_target is not None else target_config['primary_target']
        self.secondary_targets = secondary_targets if secondary_targets is not None else target_config['secondary_targets']
        
        # Target configuration
        self.returns_type = target_config['returns_type']  # 'log_returns', 'simple_returns', 'pct_change'
        self.volatility_window = target_config['volatility_window']
        self.direction_threshold = target_config['direction_threshold']
        self.price_scaling = target_config['price_scaling']
        self.validation_enabled = target_config['validation_enabled']
        
        # Target definitions
        self.target_definitions = {
            'returns': {
                'description': 'Future returns prediction',
                'type': 'continuous',
                'scaling': 'standard',
                'primary_metric': 'mae',
                'secondary_metrics': ['mse', 'directional_accuracy', 'bias']
            },
            'prices': {
                'description': 'Future price prediction',
                'type': 'continuous', 
                'scaling': self.price_scaling,
                'primary_metric': 'mae',
                'secondary_metrics': ['mse', 'mape', 'bias']
            },
            'volatility': {
                'description': 'Future volatility prediction',
                'type': 'continuous',
                'scaling': 'robust',
                'primary_metric': 'mae',
                'secondary_metrics': ['mse', 'directional_accuracy']
            },
            'direction': {
                'description': 'Price direction prediction (up/down)',
                'type': 'binary',
                'scaling': None,
                'primary_metric': 'accuracy',
                'secondary_metrics': ['precision', 'recall', 'f1']
            }
        }
        
        # Validation history
        self.validation_history = []
        
        print(f"🎯 Unified Target Manager initialized:")
        print(f"   Primary target: {self.primary_target}")
        print(f"   Secondary targets: {self.secondary_targets}")
        print(f"   Target horizon: {self.target_horizon} period(s)")
        print(f"   Returns type: {self.returns_type}")
    
    def get_target_config(self) -> Dict[str, Any]:
        """
        Get the complete target configuration
        
        Returns:
            Dictionary with target configuration
        """
        return {
            'target_horizon': self.target_horizon,
            'primary_target': self.primary_target,
            'secondary_targets': self.secondary_targets,
            'returns_type': self.returns_type,
            'volatility_window': self.volatility_window,
            'direction_threshold': self.direction_threshold,
            'price_scaling': self.price_scaling,
            'target_definitions': self.target_definitions,
            'validation_enabled': self.validation_enabled
        }
    
    def create_returns_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create returns target based on configuration
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with returns target
        """
        if self.returns_type == 'log_returns':
            # Log returns (preferred for modeling)
            returns = np.log(df['close'].shift(-self.target_horizon) / df['close'])
        elif self.returns_type == 'simple_returns':
            # Simple returns
            returns = (df['close'].shift(-self.target_horizon) / df['close']) - 1
        elif self.returns_type == 'pct_change':
            # Percentage change
            returns = df['close'].pct_change(periods=self.target_horizon).shift(-self.target_horizon)
        else:
            raise ValueError(f"Unknown returns type: {self.returns_type}")
        
        return returns
    
    def create_prices_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create prices target
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with prices target
        """
        return df['close'].shift(-self.target_horizon)
    
    def create_volatility_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create volatility target
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with volatility target
        """
        # Calculate rolling volatility
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=self.volatility_window).std()
        
        # Future volatility target
        return volatility.shift(-self.target_horizon)
    
    def create_direction_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create direction target (binary: up=1, down=0)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with direction target
        """
        # Calculate future price change
        price_change = df['close'].shift(-self.target_horizon) - df['close']
        
        # Convert to binary direction (1 for up, 0 for down)
        # Use threshold to avoid noise around zero
        direction = (price_change > self.direction_threshold).astype(int)
        
        return direction
    
    def create_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Create all configured targets
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with all targets
        """
        print(f"Creating unified targets with {self.target_horizon}-period horizon...")
        
        targets = {}
        
        # Always create primary target
        if self.primary_target == 'returns':
            targets['target_returns'] = self.create_returns_target(df)
        elif self.primary_target == 'prices':
            targets['target_prices'] = self.create_prices_target(df)
        elif self.primary_target == 'volatility':
            targets['target_volatility'] = self.create_volatility_target(df)
        elif self.primary_target == 'direction':
            targets['target_direction'] = self.create_direction_target(df)
        else:
            raise ValueError(f"Unknown primary target: {self.primary_target}")
        
        # Create secondary targets
        for target_type in self.secondary_targets:
            if target_type == 'returns' and 'target_returns' not in targets:
                targets['target_returns'] = self.create_returns_target(df)
            elif target_type == 'prices' and 'target_prices' not in targets:
                targets['target_prices'] = self.create_prices_target(df)
            elif target_type == 'volatility' and 'target_volatility' not in targets:
                targets['target_volatility'] = self.create_volatility_target(df)
            elif target_type == 'direction' and 'target_direction' not in targets:
                targets['target_direction'] = self.create_direction_target(df)
        
        # Print target statistics
        print(f"✓ Created {len(targets)} unified targets:")
        for target_name, target_series in targets.items():
            valid_count = target_series.notna().sum()
            if target_series.dtype in ['int64', 'bool']:
                # Binary target
                positive_rate = target_series.mean() if valid_count > 0 else 0
                print(f"   {target_name}: {valid_count} valid samples, positive rate: {positive_rate:.2%}")
            else:
                # Continuous target
                mean_val = target_series.mean() if valid_count > 0 else 0
                std_val = target_series.std() if valid_count > 0 else 0
                print(f"   {target_name}: {valid_count} valid samples, mean: {mean_val:.6f}, std: {std_val:.6f}")
        
        return targets
    
    def get_target_metrics(self, target_type: str) -> Dict[str, str]:
        """
        Get appropriate metrics for a target type
        
        Args:
            target_type: Type of target ('returns', 'prices', 'volatility', 'direction')
            
        Returns:
            Dictionary with primary and secondary metrics
        """
        if target_type in self.target_definitions:
            return {
                'primary_metric': self.target_definitions[target_type]['primary_metric'],
                'secondary_metrics': self.target_definitions[target_type]['secondary_metrics']
            }
        else:
            # Default metrics
            return {
                'primary_metric': 'mae',
                'secondary_metrics': ['mse', 'bias']
            }
    
    def validate_target_consistency(self, model_targets: Dict[str, Any], 
                                  model_name: str = "Unknown") -> bool:
        """
        Validate that model targets are consistent with unified configuration
        
        Args:
            model_targets: Dictionary with model's target configuration
            model_name: Name of the model for logging
            
        Returns:
            Boolean indicating if targets are consistent
        """
        if not self.validation_enabled:
            return True
        
        print(f"\n🔍 Validating target consistency for {model_name}...")
        
        validation_results = {
            'model_name': model_name,
            'timestamp': datetime.now(),
            'consistent': True,
            'issues': []
        }
        
        # Check target horizon
        model_horizon = model_targets.get('target_horizon', None)
        if model_horizon != self.target_horizon:
            validation_results['consistent'] = False
            validation_results['issues'].append(
                f"Target horizon mismatch: expected {self.target_horizon}, got {model_horizon}"
            )
        
        # Check primary target
        model_primary = model_targets.get('primary_target', None)
        if model_primary != self.primary_target:
            validation_results['consistent'] = False
            validation_results['issues'].append(
                f"Primary target mismatch: expected {self.primary_target}, got {model_primary}"
            )
        
        # Check returns type if applicable
        if 'returns_type' in model_targets:
            model_returns_type = model_targets['returns_type']
            if model_returns_type != self.returns_type:
                validation_results['consistent'] = False
                validation_results['issues'].append(
                    f"Returns type mismatch: expected {self.returns_type}, got {model_returns_type}"
                )
        
        # Check target names
        expected_targets = set(['target_' + self.primary_target] + 
                             ['target_' + t for t in self.secondary_targets])
        model_target_names = set(model_targets.get('target_names', []))
        
        missing_targets = expected_targets - model_target_names
        if missing_targets:
            validation_results['consistent'] = False
            validation_results['issues'].append(
                f"Missing targets: {missing_targets}"
            )
        
        # Store validation results
        self.validation_history.append(validation_results)
        
        # Print results
        if validation_results['consistent']:
            print(f"✅ {model_name} targets are consistent with unified configuration")
        else:
            print(f"❌ {model_name} targets have consistency issues:")
            for issue in validation_results['issues']:
                print(f"   • {issue}")
        
        return validation_results['consistent']
    
    def get_unified_target_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the unified target configuration
        
        Returns:
            Dictionary with target summary
        """
        return {
            'configuration': self.get_target_config(),
            'target_definitions': self.target_definitions,
            'validation_history': self.validation_history,
            'total_validations': len(self.validation_history),
            'consistent_models': sum(1 for v in self.validation_history if v['consistent']),
            'inconsistent_models': sum(1 for v in self.validation_history if not v['consistent'])
        }
    
    def create_target_config_for_model(self, model_type: str = "generic") -> Dict[str, Any]:
        """
        Create a target configuration dictionary for a specific model
        
        Args:
            model_type: Type of model ('adaptive', 'comparison', 'evaluation')
            
        Returns:
            Dictionary with target configuration for the model
        """
        base_config = {
            'target_horizon': self.target_horizon,
            'primary_target': self.primary_target,
            'secondary_targets': self.secondary_targets,
            'returns_type': self.returns_type,
            'target_names': ['target_' + self.primary_target] + ['target_' + t for t in self.secondary_targets]
        }
        
        # Add model-specific configurations
        if model_type == 'adaptive':
            base_config.update({
                'focus_target': 'target_returns',  # Adaptive models focus on returns
                'primary_metrics': self.get_target_metrics('returns'),
                'loss_function_target': 'returns'
            })
        elif model_type == 'comparison':
            base_config.update({
                'comparison_targets': ['target_returns', 'target_prices'],
                'primary_metrics': self.get_target_metrics(self.primary_target),
                'comparison_metrics': ['mae', 'mse', 'directional_accuracy', 'bias']
            })
        elif model_type == 'evaluation':
            base_config.update({
                'evaluation_targets': ['target_' + self.primary_target] + ['target_' + t for t in self.secondary_targets],
                'comprehensive_metrics': True,
                'all_target_metrics': {target: self.get_target_metrics(target) 
                                     for target in [self.primary_target] + self.secondary_targets}
            })
        
        return base_config
    
    def print_unified_target_info(self):
        """Print comprehensive information about unified targets"""
        print("\n" + "="*80)
        print("UNIFIED TARGET CONFIGURATION")
        print("="*80)
        
        print(f"Primary Target: {self.primary_target}")
        print(f"Secondary Targets: {', '.join(self.secondary_targets)}")
        print(f"Target Horizon: {self.target_horizon} period(s)")
        print(f"Returns Type: {self.returns_type}")
        
        if 'volatility' in [self.primary_target] + self.secondary_targets:
            print(f"Volatility Window: {self.volatility_window}")
        
        if 'direction' in [self.primary_target] + self.secondary_targets:
            print(f"Direction Threshold: {self.direction_threshold}")
        
        print(f"\nTarget Definitions:")
        for target_type, definition in self.target_definitions.items():
            if target_type in [self.primary_target] + self.secondary_targets:
                print(f"  {target_type}:")
                print(f"    Description: {definition['description']}")
                print(f"    Type: {definition['type']}")
                print(f"    Primary Metric: {definition['primary_metric']}")
                print(f"    Secondary Metrics: {', '.join(definition['secondary_metrics'])}")
        
        if self.validation_history:
            print(f"\nValidation History:")
            print(f"  Total Validations: {len(self.validation_history)}")
            consistent_count = sum(1 for v in self.validation_history if v['consistent'])
            print(f"  Consistent Models: {consistent_count}/{len(self.validation_history)}")
        
        print("="*80)

def create_unified_target_config_file():
    """Create a unified target configuration file for easy reference"""
    
    # Initialize target manager with defaults
    target_manager = UnifiedTargetManager()
    
    # Create configurations for different model types
    configs = {
        'adaptive_model_config': target_manager.create_target_config_for_model('adaptive'),
        'comparison_model_config': target_manager.create_target_config_for_model('comparison'),
        'evaluation_model_config': target_manager.create_target_config_for_model('evaluation'),
        'unified_target_summary': target_manager.get_unified_target_summary()
    }
    
    # Save to YAML file
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'unified_targets_config.yaml')
    
    with open(config_path, 'w') as f:
        yaml.dump(configs, f, default_flow_style=False, indent=2)
    
    print(f"✅ Unified target configuration saved to: {config_path}")
    return config_path

def main():
    """Example usage of the unified target manager"""
    
    print("🎯 Unified Target Manager - Example Usage")
    print("="*60)
    
    # Initialize target manager
    target_manager = UnifiedTargetManager()
    
    # Print configuration info
    target_manager.print_unified_target_info()
    
    # Create sample data for demonstration
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'datetime': dates,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.02),  # Random walk
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Create targets
    targets = target_manager.create_targets(sample_data)
    
    # Example model validation
    model_config = {
        'target_horizon': 1,
        'primary_target': 'returns',
        'returns_type': 'log_returns',
        'target_names': ['target_returns', 'target_prices', 'target_direction']
    }
    
    is_consistent = target_manager.validate_target_consistency(model_config, "Example Model")
    
    # Create unified config file
    config_file = create_unified_target_config_file()
    
    print(f"\n✅ Unified Target Manager demonstration completed!")
    print(f"📁 Configuration file created: {config_file}")

if __name__ == "__main__":
    main()
