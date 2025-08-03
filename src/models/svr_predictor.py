"""
Support Vector Regression Stock Price Predictor with Unified Targets
Uses Support Vector Machine for regression with RBF kernel

Usage:
    python3 src/models/svr_predictor.py
    
Features:
    - SVR with RBF kernel and optimized hyperparameters
    - Uses unified preprocessing and target system
    - Comprehensive evaluation metrics
    - Configuration-driven approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import sys
import os
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

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

# Add the project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import unified preprocessing and targets
from src.data_processing.unified_preprocessing import UnifiedPreprocessor
from src.data_processing.unified_targets import UnifiedTargetManager

class SVRStockPredictor:
    """
    Support Vector Regression predictor using unified preprocessing and targets
    """
    
    def __init__(self, symbol: str = None, days_lookback: int = None, target_horizon: int = None):
        """Initialize the SVR predictor"""
        
        # Load config values with fallbacks
        if symbol is None:
            symbol = 'NVDA'
        if days_lookback is None:
            days_lookback = CONFIG['data_processing']['days_lookback']
        if target_horizon is None:
            target_horizon = CONFIG['data_processing']['target_horizon']
        
        self.symbol = symbol
        self.days_lookback = days_lookback
        self.target_horizon = target_horizon
        self.test_size = CONFIG['data_processing']['test_size']
        
        # Initialize unified target manager
        self.target_manager = UnifiedTargetManager(
            target_horizon=target_horizon,
            primary_target='returns',
            secondary_targets=['prices', 'direction']
        )
        
        # Validate target consistency
        model_target_config = self.target_manager.create_target_config_for_model('adaptive')
        self.target_manager.validate_target_consistency(model_target_config, "SVR Predictor")
        
        # Model parameters from config
        self.svr_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1,
            'cache_size': 200
        }
        
        # Initialize model
        self.model = None
        self.data = None
        
    def get_data(self) -> Dict:
        """Get standardized data from unified preprocessing pipeline"""
        print(f"🔄 Getting standardized data from unified preprocessing pipeline...")
        print(f"Symbol: {self.symbol}, Days: {self.days_lookback}, Horizon: {self.target_horizon}")
        
        # Initialize unified preprocessor
        processor = UnifiedPreprocessor(
            symbol=self.symbol,
            target_horizon=self.target_horizon,
            test_size=self.test_size,
            lag_periods=[1, 2, 3],
            days_lookback=self.days_lookback
        )
        
        # Get standardized data
        self.data = processor.get_standardized_data()
        
        print(f"✅ Received standardized data with {self.data['total_features']} features")
        print(f"   Training samples: {len(self.data['X_train'])}")
        print(f"   Test samples: {len(self.data['X_test'])}")
        
        return self.data
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Optimize SVR hyperparameters using time series cross-validation"""
        print("\n🔧 Optimizing SVR hyperparameters...")
        
        # Parameter grid for optimization
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search with time series CV
        svr_base = SVR(cache_size=200)
        
        grid_search = GridSearchCV(
            svr_base,
            param_grid,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=1
        )
        
        print("   Running grid search with time series cross-validation...")
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"   Best CV MAE: {-grid_search.best_score_:.6f}")
        
        return grid_search.best_params_
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, optimize: bool = True) -> None:
        """Train the SVR model"""
        print("\n🎯 Training SVR model...")
        
        if optimize:
            # Optimize hyperparameters
            best_params = self.optimize_hyperparameters(X_train, y_train)
            self.svr_params.update(best_params)
        
        # Initialize and train model with best parameters
        self.model = SVR(**self.svr_params)
        
        print("   Training SVR with optimized parameters...")
        self.model.fit(X_train, y_train)
        
        print("✅ SVR training completed!")
        print(f"   Model parameters: {self.svr_params}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      scaler_y: Any, train_test_diff: float = 0) -> Dict:
        """Evaluate the SVR model"""
        print("\n📈 Evaluating SVR model...")
        
        # Make predictions
        predictions_scaled = self.model.predict(X_test)
        
        # Convert back to original scale
        predictions_orig = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        mse = mean_squared_error(y_test_orig, predictions_orig)
        rmse = np.sqrt(mse)
        
        # Bias metrics
        bias = np.mean(predictions_orig - y_test_orig)
        bias_corrected_predictions = predictions_orig - bias
        bias_corrected_mae = mean_absolute_error(y_test_orig, bias_corrected_predictions)
        
        # Directional accuracy
        actual_direction = np.sign(y_test_orig)
        predicted_direction = np.sign(predictions_orig)
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        results = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'bias': bias,
            'bias_corrected_mae': bias_corrected_mae,
            'directional_accuracy': directional_accuracy,
            'predictions': predictions_orig,
            'actuals': y_test_orig,
            'bias_metrics': {
                'bias': bias,
                'corrected_mae': bias_corrected_mae
            }
        }
        
        print(f"✅ SVR Evaluation Results:")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Bias: {bias:.6f}")
        print(f"   Bias-Corrected MAE: {bias_corrected_mae:.6f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2%}")
        
        return results
    
    def create_visualizations(self, results: Dict, feature_names: List[str]) -> None:
        """Create visualizations for SVR analysis"""
        print("\n📊 Creating SVR visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Support Vector Regression Stock Predictor Analysis', fontsize=16, fontweight='bold')
        
        # Predictions vs Actuals
        axes[0, 0].scatter(results['actuals'], results['predictions'], alpha=0.6, color='blue')
        axes[0, 0].plot([results['actuals'].min(), results['actuals'].max()], 
                       [results['actuals'].min(), results['actuals'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title('Predictions vs Actuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = results['predictions'] - results['actuals']
        axes[0, 1].scatter(results['predictions'], residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Returns')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Model parameters visualization
        param_names = list(self.svr_params.keys())
        param_values = []
        for param, value in self.svr_params.items():
            if isinstance(value, (int, float)):
                param_values.append(value)
            else:
                param_values.append(0)  # For non-numeric parameters
        
        if any(v > 0 for v in param_values):
            axes[1, 0].bar(range(len(param_values)), param_values, color='orange')
            axes[1, 0].set_xticks(range(len(param_names)))
            axes[1, 0].set_xticklabels(param_names, rotation=45)
            axes[1, 0].set_ylabel('Parameter Values')
            axes[1, 0].set_title('SVR Parameters (Numeric Only)')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, f"Kernel: {self.svr_params['kernel']}\nC: {self.svr_params['C']}\nGamma: {self.svr_params['gamma']}\nEpsilon: {self.svr_params['epsilon']}", 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[1, 0].set_title('SVR Parameters')
            axes[1, 0].axis('off')
        
        # Performance metrics
        metrics = ['MAE', 'RMSE', 'Bias (abs)', 'Dir Accuracy']
        values = [results['mae'], results['rmse'], abs(results['bias']), results['directional_accuracy'] * 100]
        
        bars = axes[1, 1].bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].annotate(f'{value:.4f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('svr_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive report"""
        print("\n📋 Generating SVR report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"svr_predictor_report_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Support Vector Regression Stock Predictor Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"**Symbol**: {self.symbol}\n")
            f.write(f"**Data Source**: Live data via unified preprocessing\n")
            f.write(f"**Lookback Period**: {self.days_lookback} days\n")
            f.write(f"**Target Horizon**: {self.target_horizon} day(s)\n")
            f.write(f"**Model Type**: Support Vector Regression\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Mean Absolute Error**: {results['mae']:.6f}\n")
            f.write(f"- **Root Mean Square Error**: {results['rmse']:.6f}\n")
            f.write(f"- **Bias**: {results['bias']:.6f}\n")
            f.write(f"- **Bias-Corrected MAE**: {results['bias_corrected_mae']:.6f}\n")
            f.write(f"- **Directional Accuracy**: {results['directional_accuracy']:.2%}\n\n")
            
            f.write("## Model Configuration\n\n")
            f.write("### SVR Parameters\n")
            for param, value in self.svr_params.items():
                f.write(f"- **{param}**: {value}\n")
            f.write("\n")
            
            f.write("### Data Processing\n")
            f.write("- **Preprocessing**: Unified preprocessing pipeline with 60+ technical indicators\n")
            f.write("- **Feature Engineering**: Lagged features for true forecasting\n")
            f.write("- **Scaling**: RobustScaler for features, StandardScaler for targets\n")
            f.write("- **Target**: Log returns prediction\n\n")
            
            f.write("## Performance Analysis\n\n")
            f.write("### Prediction Accuracy\n")
            f.write(f"The SVR model achieved a Mean Absolute Error of **{results['mae']:.6f}** ")
            f.write("on the test set, indicating the average prediction error in log returns.\n\n")
            
            f.write("### Bias Analysis\n")
            bias = results['bias']
            if abs(bias) < 0.001:
                f.write(f"The model shows minimal bias (**{bias:.6f}**), indicating well-calibrated predictions.\n\n")
            else:
                f.write(f"The model shows some bias (**{bias:.6f}**), which has been corrected in the analysis.\n\n")
            
            f.write("### Directional Accuracy\n")
            dir_acc = results['directional_accuracy']
            if dir_acc > 0.55:
                f.write(f"Strong directional accuracy of **{dir_acc:.2%}** indicates good trend prediction capability.\n\n")
            elif dir_acc > 0.50:
                f.write(f"Modest directional accuracy of **{dir_acc:.2%}** shows some trend prediction ability.\n\n")
            else:
                f.write(f"Directional accuracy of **{dir_acc:.2%}** suggests limited trend prediction capability.\n\n")
            
            f.write("## Technical Details\n\n")
            f.write("### Model Architecture\n")
            f.write("- **Algorithm**: Support Vector Regression with RBF kernel\n")
            f.write("- **Hyperparameter Optimization**: Grid search with time series cross-validation\n")
            f.write("- **Kernel Function**: Maps data to higher-dimensional space for non-linear relationships\n")
            f.write("- **Regularization**: C parameter controls trade-off between smoothness and training error\n\n")
            
            f.write("### Key Advantages\n")
            f.write("- **Non-linear Modeling**: RBF kernel captures complex relationships\n")
            f.write("- **Robust to Outliers**: SVR is less sensitive to outliers than linear regression\n")
            f.write("- **Memory Efficient**: Only support vectors are stored, not entire training set\n")
            f.write("- **Regularization**: Built-in regularization prevents overfitting\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The Support Vector Regression model provides a robust, non-linear approach to stock return prediction. ")
            f.write("Its kernel-based approach allows it to capture complex relationships in the technical indicators ")
            f.write("while maintaining good generalization performance.\n\n")
            
            f.write("### Recommendations\n")
            f.write("1. **Kernel Selection**: Experiment with different kernels (polynomial, sigmoid)\n")
            f.write("2. **Feature Scaling**: Ensure proper feature scaling for optimal SVR performance\n")
            f.write("3. **Hyperparameter Tuning**: Regular re-optimization as market conditions change\n")
            f.write("4. **Ensemble Methods**: Consider combining with other model types\n\n")
            
            f.write("---\n")
            f.write("*Report generated by Support Vector Regression Stock Predictor*\n")
            f.write(f"*Using unified preprocessing pipeline with live {self.symbol} data*")
        
        print(f"✅ Report saved to: {report_filename}")
        return report_filename

def main(symbol: str = None, days_lookback: int = None, target_horizon: int = None):
    """Main function for SVR predictor"""
    
    print("=" * 80)
    print("SUPPORT VECTOR REGRESSION STOCK PRICE PREDICTOR WITH UNIFIED TARGETS")
    print("=" * 80)
    print("Features: Non-linear SVR + unified preprocessing + kernel optimization")
    print()
    
    # Initialize predictor
    predictor = SVRStockPredictor(symbol, days_lookback, target_horizon)
    
    # Get data
    data = predictor.get_data()
    
    # Extract data components
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_returns_train_scaled = data['y_returns_train_scaled']
    y_returns_test_scaled = data['y_returns_test_scaled']
    feature_names = data['feature_names']
    scaler_y_returns = data['scaler_y_returns']
    train_test_returns_diff = data['train_test_returns_diff']
    
    print(f"\n📊 Data Summary:")
    print(f"   Features: {len(feature_names)} (lagged technical indicators)")
    print(f"   Training samples: {len(X_train_scaled)}")
    print(f"   Test samples: {len(X_test_scaled)}")
    print(f"   Target: Returns (log returns)")
    
    # Train model
    predictor.train_model(X_train_scaled, y_returns_train_scaled, optimize=True)
    
    # Evaluate model
    results = predictor.evaluate_model(
        X_test_scaled, y_returns_test_scaled, 
        scaler_y_returns, train_test_returns_diff
    )
    
    # Save comprehensive results to JSON
    results_data = {
        'model_name': 'SVR Predictor',
        'model_type': 'support_vector_regression',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'symbol': predictor.symbol,
            'days_lookback': predictor.days_lookback,
            'target_horizon': predictor.target_horizon,
            'test_size': predictor.test_size,
            'kernel': predictor.svr_params.get('kernel', 'rbf'),
            'C': predictor.svr_params.get('C', 1.0),
            'gamma': predictor.svr_params.get('gamma', 'scale'),
            'epsilon': predictor.svr_params.get('epsilon', 0.1)
        },
        'performance_metrics': {
            'mae': results.get('mae', 0),
            'rmse': results.get('rmse', 0),
            'bias': results.get('bias', 0),
            'bias_corrected_mae': results.get('bias_corrected_mae', 0),
            'directional_accuracy': results.get('directional_accuracy', 0),
            'mse': results.get('mse', 0)
        },
        'feature_analysis': {
            'total_features': data['total_features'],
            'feature_names': feature_names
        },
        'model_parameters': predictor.svr_params,
        'data_summary': {
            'training_samples': len(X_train_scaled),
            'test_samples': len(X_test_scaled),
            'train_test_distribution_shift': train_test_returns_diff
        },
        'predictions': {
            'test_predictions': results.get('predictions', []).tolist() if hasattr(results.get('predictions', []), 'tolist') else list(results.get('predictions', [])),
            'test_actuals': results.get('actuals', []).tolist() if hasattr(results.get('actuals', []), 'tolist') else list(results.get('actuals', []))
        }
    }
    
    # Save to JSON file in organized directory structure (overwrite previous results)
    import json
    
    # Ensure the results directory exists
    results_dir = os.path.join('data', 'results', 'json')
    os.makedirs(results_dir, exist_ok=True)
    
    json_filename = os.path.join(results_dir, "svr_predictor_results.json")
    
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("SVR PREDICTOR SUMMARY")
    print("="*80)
    print(f"✅ Model trained and evaluated successfully")
    print(f"📊 MAE: {results['mae']:.6f}")
    print(f"📊 Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"📊 Results saved to JSON: {json_filename}")
    print(f"🎯 Using unified targets and preprocessing")
    
    return results_data

if __name__ == "__main__":
    main()
