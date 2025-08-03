"""
Linear Regression Stock Price Predictor with Unified Targets
Uses regularized linear regression (Ridge, Lasso, ElasticNet) for stock prediction

Usage:
    python3 src/models/linear_regression_predictor.py
    
Features:
    - Multiple regularization techniques (Ridge, Lasso, ElasticNet)
    - Uses unified preprocessing and target system
    - Feature coefficient analysis
    - Comprehensive evaluation metrics
    - Configuration-driven approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

class LinearRegressionStockPredictor:
    """
    Linear Regression predictor using unified preprocessing and targets
    """
    
    def __init__(self, symbol: str = None, days_lookback: int = None, target_horizon: int = None):
        """Initialize the Linear Regression predictor"""
        
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
        self.target_manager.validate_target_consistency(model_target_config, "Linear Regression Predictor")
        
        # Model selection and parameters
        self.model_type = 'ridge'  # Options: 'linear', 'ridge', 'lasso', 'elasticnet'
        self.model_params = {
            'ridge': {'alpha': 1.0, 'random_state': CONFIG['environment']['random_seed']},
            'lasso': {'alpha': 1.0, 'random_state': CONFIG['environment']['random_seed']},
            'elasticnet': {'alpha': 1.0, 'l1_ratio': 0.5, 'random_state': CONFIG['environment']['random_seed']},
            'linear': {}
        }
        
        # Initialize models
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.coefficients = None
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
        """Optimize hyperparameters for different regularization techniques"""
        print("\n🔧 Optimizing Linear Regression hyperparameters...")
        
        best_models = {}
        
        # Ridge Regression
        print("   Optimizing Ridge Regression...")
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        tscv = TimeSeriesSplit(n_splits=3)
        
        ridge_grid = GridSearchCV(
            Ridge(random_state=CONFIG['environment']['random_seed']),
            ridge_params,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        ridge_grid.fit(X_train, y_train)
        best_models['ridge'] = {
            'model': ridge_grid.best_estimator_,
            'score': -ridge_grid.best_score_,
            'params': ridge_grid.best_params_
        }
        
        # Lasso Regression
        print("   Optimizing Lasso Regression...")
        lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}
        
        lasso_grid = GridSearchCV(
            Lasso(random_state=CONFIG['environment']['random_seed']),
            lasso_params,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        lasso_grid.fit(X_train, y_train)
        best_models['lasso'] = {
            'model': lasso_grid.best_estimator_,
            'score': -lasso_grid.best_score_,
            'params': lasso_grid.best_params_
        }
        
        # ElasticNet Regression
        print("   Optimizing ElasticNet Regression...")
        elasticnet_params = {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        
        elasticnet_grid = GridSearchCV(
            ElasticNet(random_state=CONFIG['environment']['random_seed']),
            elasticnet_params,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        elasticnet_grid.fit(X_train, y_train)
        best_models['elasticnet'] = {
            'model': elasticnet_grid.best_estimator_,
            'score': -elasticnet_grid.best_score_,
            'params': elasticnet_grid.best_params_
        }
        
        # Linear Regression (no regularization)
        print("   Training Linear Regression...")
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_train)
        linear_mae = mean_absolute_error(y_train, linear_pred)
        best_models['linear'] = {
            'model': linear_model,
            'score': linear_mae,
            'params': {}
        }
        
        # Find best model
        best_model_name = min(best_models.keys(), key=lambda k: best_models[k]['score'])
        self.best_model = best_models[best_model_name]['model']
        self.best_model_name = best_model_name
        self.models = best_models
        
        print(f"✅ Best model: {best_model_name.upper()}")
        print(f"   Best CV MAE: {best_models[best_model_name]['score']:.6f}")
        print(f"   Best parameters: {best_models[best_model_name]['params']}")
        
        return best_models[best_model_name]['params']
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, optimize: bool = True) -> None:
        """Train the Linear Regression model"""
        print("\n📏 Training Linear Regression models...")
        
        if optimize:
            # Optimize hyperparameters and select best model
            self.optimize_hyperparameters(X_train, y_train)
        else:
            # Use default Ridge regression
            self.best_model = Ridge(**self.model_params['ridge'])
            self.best_model.fit(X_train, y_train)
            self.best_model_name = 'ridge'
        
        # Extract coefficients
        if hasattr(self.best_model, 'coef_'):
            self.coefficients = self.best_model.coef_
        
        print("✅ Linear Regression training completed!")
        print(f"   Selected model: {self.best_model_name.upper()}")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, 
                      scaler_y: Any, train_test_diff: float = 0) -> Dict:
        """Evaluate the Linear Regression model"""
        print("\n📈 Evaluating Linear Regression model...")
        
        # Make predictions
        predictions_scaled = self.best_model.predict(X_test)
        
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
            'model_type': self.best_model_name,
            'bias_metrics': {
                'bias': bias,
                'corrected_mae': bias_corrected_mae
            }
        }
        
        print(f"✅ Linear Regression Evaluation Results:")
        print(f"   Selected Model: {self.best_model_name.upper()}")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Bias: {bias:.6f}")
        print(f"   Bias-Corrected MAE: {bias_corrected_mae:.6f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2%}")
        
        return results
    
    def analyze_coefficients(self, feature_names: List[str]) -> None:
        """Analyze and display model coefficients"""
        print(f"\n🔍 Analyzing {self.best_model_name.upper()} Coefficients...")
        
        if self.coefficients is None:
            print("❌ Model must be trained first")
            return
        
        # Create coefficients DataFrame
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"\nTop 15 Most Important Coefficients (by absolute value):")
        for i, (_, row) in enumerate(coef_df.head(15).iterrows(), 1):
            direction = "↑" if row['coefficient'] > 0 else "↓"
            print(f"  {i:2d}. {row['feature']:30s}: {row['coefficient']:+.6f} {direction}")
        
        # Categorize coefficients
        categories = {
            'trend': ['sma', 'ema', 'ma_', 'bb_middle', 'vwap', 'psar'],
            'momentum': ['rsi', 'stoch', 'williams', 'cci', 'mfi', 'momentum', 'roc'],
            'volume': ['volume', 'obv', 'adl'],
            'volatility': ['atr', 'volatility', 'bb_width', 'bb_upper', 'bb_lower'],
            'signal': ['macd', 'bb_position']
        }
        
        category_importance = {cat: 0.0 for cat in categories.keys()}
        category_importance['other'] = 0.0
        
        for _, row in coef_df.iterrows():
            feature_lower = row['feature'].lower()
            categorized = False
            
            for cat, keywords in categories.items():
                if any(keyword in feature_lower for keyword in keywords):
                    category_importance[cat] += abs(row['coefficient'])
                    categorized = True
                    break
            
            if not categorized:
                category_importance['other'] += abs(row['coefficient'])
        
        print(f"\nCoefficient Importance by Category (absolute values):")
        sorted_categories = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        for category, importance in sorted_categories:
            if importance > 0:
                print(f"  {category:12s}: {importance:.6f}")
    
    def create_visualizations(self, results: Dict, feature_names: List[str]) -> None:
        """Create visualizations for Linear Regression analysis"""
        print("\n📊 Creating Linear Regression visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Linear Regression ({results["model_type"].upper()}) Stock Predictor Analysis', 
                    fontsize=16, fontweight='bold')
        
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
        
        # Coefficients (top 15 by absolute value)
        if self.coefficients is not None:
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': self.coefficients
            }).sort_values('coefficient', key=abs, ascending=False).head(15)
            
            colors = ['red' if c < 0 else 'blue' for c in coef_df['coefficient']]
            axes[1, 0].barh(range(len(coef_df)), coef_df['coefficient'], color=colors, alpha=0.7)
            axes[1, 0].set_yticks(range(len(coef_df)))
            axes[1, 0].set_yticklabels([f.split('_')[0] for f in coef_df['feature']], fontsize=8)
            axes[1, 0].set_xlabel('Coefficient Value')
            axes[1, 0].set_title('Top 15 Coefficients')
            axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Model comparison (if multiple models were trained)
        if hasattr(self, 'models') and len(self.models) > 1:
            model_names = list(self.models.keys())
            model_scores = [self.models[name]['score'] for name in model_names]
            
            bars = axes[1, 1].bar(model_names, model_scores, 
                                 color=['gold' if name == self.best_model_name else 'lightblue' 
                                       for name in model_names])
            axes[1, 1].set_ylabel('Cross-Validation MAE')
            axes[1, 1].set_title('Model Comparison')
            axes[1, 1].grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, score in zip(bars, model_scores):
                height = bar.get_height()
                axes[1, 1].annotate(f'{score:.4f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),
                                   textcoords="offset points",
                                   ha='center', va='bottom', fontsize=10)
        else:
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
        plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    

def main(symbol: str = None, days_lookback: int = None, target_horizon: int = None):
    """Main function for Linear Regression predictor"""
    
    print("=" * 80)
    print("LINEAR REGRESSION STOCK PRICE PREDICTOR WITH UNIFIED TARGETS")
    print("=" * 80)
    print("Features: Regularized linear models + unified preprocessing + coefficient analysis")
    print()
    
    # Initialize predictor
    predictor = LinearRegressionStockPredictor(symbol, days_lookback, target_horizon)
    
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
    
    # Analyze coefficients
    predictor.analyze_coefficients(feature_names)
    
    # Skip visualizations - save data to JSON instead
    # predictor.create_visualizations(results, feature_names)
    
    print("\n" + "="*80)
    print("LINEAR REGRESSION PREDICTOR SUMMARY")
    print("="*80)
    print(f"✅ Model trained and evaluated successfully")
    print(f"📊 Selected Model: {results['model_type'].upper()}")
    print(f"📊 MAE: {results['mae']:.6f}")
    print(f"📊 Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"🎯 Using unified targets and preprocessing")
    
    # Save comprehensive results to JSON
    results_data = {
        'model_name': 'Linear Regression Predictor',
        'model_type': f'linear_regression_{predictor.best_model_name}',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'symbol': predictor.symbol,
            'days_lookback': predictor.days_lookback,
            'target_horizon': predictor.target_horizon,
            'test_size': predictor.test_size,
            'selected_model': predictor.best_model_name,
            'model_parameters': predictor.models[predictor.best_model_name]['params'] if hasattr(predictor, 'models') and predictor.best_model_name in predictor.models else {}
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
            'feature_names': feature_names,
            'coefficients': predictor.coefficients.tolist() if predictor.coefficients is not None else [],
            'top_10_coefficients': [(feature_names[i], coef) for i, coef in enumerate(predictor.coefficients) if predictor.coefficients is not None][:10] if predictor.coefficients is not None else []
        },
        'model_comparison': {
            'all_models_tested': list(predictor.models.keys()) if hasattr(predictor, 'models') else [predictor.best_model_name],
            'selected_model': predictor.best_model_name,
            'model_scores': {name: info['score'] for name, info in predictor.models.items()} if hasattr(predictor, 'models') else {}
        },
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
    
    json_filename = os.path.join(results_dir, "linear_regression_predictor_results.json")
    
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"📊 Results saved to JSON: {json_filename}")
    
    return results_data

if __name__ == "__main__":
    main()
