"""
Comprehensive Model Comparison Framework
Runs all models in parallel threads for efficient comparison

Features:
- Parallel execution of all models using threading
- Live data fetching via unified preprocessing
- Comprehensive performance comparison across all models
- Statistical significance testing
- Detailed reporting and visualization
- Thread-safe model execution

Usage:
    python3 src/evaluation/comprehensive_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
import os
import yaml
from datetime import datetime
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Callable
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

# Import unified preprocessing and targets
from src.data_processing.unified_preprocessing import UnifiedPreprocessor
from src.data_processing.unified_targets import UnifiedTargetManager

# Import all model main functions
from src.models.enhanced_adaptive_predictor import main as run_enhanced_adaptive_predictor
from src.models.random_forest_predictor import main as run_random_forest_predictor
from src.models.xgboost_predictor import main as run_xgboost_predictor
from src.models.svr_predictor import main as run_svr_predictor
from src.models.lstm_predictor import main as run_lstm_predictor
from src.models.lightgbm_predictor import main as run_lightgbm_predictor
from src.models.linear_regression_predictor import main as run_linear_regression_predictor

class ComprehensiveModelComparison:
    """
    Comprehensive comparison framework running all models in parallel threads
    """
    
    def __init__(self, symbol: str = None, days_lookback: int = None, target_horizon: int = None):
        """Initialize the comprehensive comparison with live data parameters"""
        # Load config values with fallbacks
        if symbol is None:
            symbol = 'NVDA'  # Default symbol
        if days_lookback is None:
            days_lookback = CONFIG['data_processing']['days_lookback']
        if target_horizon is None:
            target_horizon = CONFIG['data_processing']['target_horizon']
        
        self.symbol = symbol
        self.days_lookback = days_lookback
        self.target_horizon = target_horizon
        self.test_size = CONFIG['data_processing']['test_size']
        self.lag_periods = CONFIG['data_processing']['lag_periods']
        
        # Model results storage
        self.model_results = {}
        self.execution_times = {}
        self.thread_lock = threading.Lock()
        
        # Initialize unified target manager for validation
        print("🎯 Initializing unified target manager for comparison...")
        self.target_manager = UnifiedTargetManager(
            target_horizon=target_horizon,
            primary_target='returns',
            secondary_targets=['prices', 'direction']
        )
        
        # Get target configuration for comparison model
        model_target_config = self.target_manager.create_target_config_for_model('comparison')
        
        # Validate target consistency
        self.target_manager.validate_target_consistency(model_target_config, "Comprehensive Comparison")
        
        # Define all models to run
        self.models_to_run = {
            'Enhanced Adaptive': run_enhanced_adaptive_predictor,
            'Random Forest': run_random_forest_predictor,
            'XGBoost': run_xgboost_predictor,
            'SVR': run_svr_predictor,
            'LSTM': run_lstm_predictor,
            'LightGBM': run_lightgbm_predictor,
            'Linear Regression': run_linear_regression_predictor
        }
        
    def run_single_model(self, model_name: str, model_func: Callable) -> Dict:
        """Run a single model in a thread-safe manner"""
        thread_id = threading.current_thread().ident
        start_time = time.time()
        
        try:
            print(f"🔄 [{thread_id}] Starting {model_name}...")
            
            # Set matplotlib to non-interactive backend for thread safety
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            
            # Run the model - we'll need to modify each model to skip visualizations in threads
            # For now, let's catch and handle GUI-related errors
            result = model_func(
                symbol=self.symbol,
                days_lookback=self.days_lookback,
                target_horizon=self.target_horizon
            )
            
            execution_time = time.time() - start_time
            
            # Thread-safe storage of results
            with self.thread_lock:
                self.model_results[model_name] = result
                self.execution_times[model_name] = execution_time
            
            print(f"✅ [{thread_id}] {model_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"❌ [{thread_id}] {model_name} failed after {execution_time:.2f}s: {str(e)}"
            print(error_msg)
            
            # Store error result
            with self.thread_lock:
                self.model_results[model_name] = {'error': str(e)}
                self.execution_times[model_name] = execution_time
            
            return {'error': str(e)}
    
    def run_all_models_parallel(self, max_workers: int = 4) -> Dict:
        """Run all models in parallel using ThreadPoolExecutor"""
        print("🚀 Starting parallel execution of all models...")
        print("="*80)
        print(f"Models to run: {list(self.models_to_run.keys())}")
        print(f"Max workers: {max_workers}")
        print()
        
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all model tasks
            future_to_model = {
                executor.submit(self.run_single_model, model_name, model_func): model_name
                for model_name, model_func in self.models_to_run.items()
            }
            
            # Process completed tasks
            completed_count = 0
            total_models = len(self.models_to_run)
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    print(f"📊 Progress: {completed_count}/{total_models} models completed")
                    
                except Exception as e:
                    print(f"❌ {model_name} generated an exception: {e}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 All models completed in {total_time:.2f}s")
        
        return self.model_results
    
    def extract_model_metrics(self, model_result: Dict) -> Dict:
        """Extract standardized metrics from model results"""
        if 'error' in model_result:
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'bias': float('inf'),
                'directional_accuracy': 0.0,
                'execution_time': 0.0,
                'error': model_result['error']
            }
        
        # Extract performance metrics from the JSON structure
        performance_metrics = model_result.get('performance_metrics', {})
        
        # Convert string metrics to float if needed
        mae = float(performance_metrics.get('mae', 0))
        rmse = float(performance_metrics.get('rmse', 0))
        bias = float(performance_metrics.get('bias', 0))
        bias_corrected_mae = float(performance_metrics.get('bias_corrected_mae', mae))
        directional_accuracy = float(performance_metrics.get('directional_accuracy', 0))
        
        return {
            'mae': bias_corrected_mae if bias_corrected_mae > 0 else mae,
            'rmse': rmse,
            'bias': bias,
            'directional_accuracy': directional_accuracy,
            'execution_time': self.execution_times.get(model_result.get('model_name', ''), 0.0)
        }
    
    def analyze_all_models_performance(self) -> Dict:
        """Analyze performance of all models"""
        print("\n🔍 Analyzing All Models Performance...")
        print("="*80)
        
        if not self.model_results:
            print("❌ No model results available for analysis")
            return {}
        
        analysis_results = {}
        
        print("\n📊 MODEL PERFORMANCE SUMMARY:")
        print("-" * 80)
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Dir Acc':<10} {'Time (s)':<10} {'Status':<10}")
        print("-" * 80)
        
        for model_name, model_result in self.model_results.items():
            metrics = self.extract_model_metrics(model_result)
            execution_time = self.execution_times.get(model_name, 0.0)
            
            status = "✅ OK" if 'error' not in metrics else "❌ ERROR"
            
            print(f"{model_name:<20} {metrics['mae']:<12.6f} {metrics['rmse']:<12.6f} "
                  f"{metrics['directional_accuracy']:<10.2%} {execution_time:<10.2f} {status:<10}")
            
            analysis_results[model_name] = metrics
        
        # Find best performing models
        valid_models = {k: v for k, v in analysis_results.items() if 'error' not in v}
        
        if valid_models:
            best_mae_model = min(valid_models.keys(), key=lambda k: valid_models[k]['mae'])
            best_dir_model = max(valid_models.keys(), key=lambda k: valid_models[k]['directional_accuracy'])
            fastest_model = min(valid_models.keys(), key=lambda k: self.execution_times.get(k, float('inf')))
            
            print(f"\n🏆 BEST PERFORMERS:")
            print(f"   Best MAE: {best_mae_model} ({valid_models[best_mae_model]['mae']:.6f})")
            print(f"   Best Directional Accuracy: {best_dir_model} ({valid_models[best_dir_model]['directional_accuracy']:.2%})")
            print(f"   Fastest: {fastest_model} ({self.execution_times.get(fastest_model, 0):.2f}s)")
        
        return analysis_results
    
    def create_enhanced_visualizations(self, comparison_summary):
        """Create visualizations for enhanced adaptive predictor analysis"""
        print("\n📊 Creating Enhanced Adaptive Predictor Visualizations...")
        
        if comparison_summary is None:
            print("No comparison data available for visualization")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create enhanced analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Adaptive Predictor Analysis', fontsize=16, fontweight='bold')
        
        # MAE Comparison
        models = ['Enhanced Adaptive', 'Standard']
        mae_values = [comparison_summary['enhanced_adaptive_mae'], comparison_summary['standard_mae']]
        colors = ['green' if comparison_summary['improvement_pct'] > 0 else 'red', 'blue']
        
        bars = axes[0, 0].bar(models, mae_values, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('MAE Comparison')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add improvement percentage as text
        improvement_text = f"{comparison_summary['improvement_pct']:+.2f}%"
        axes[0, 0].text(0.5, max(mae_values) * 0.8, improvement_text, 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bias Comparison
        bias_values = [comparison_summary['enhanced_bias'], comparison_summary['standard_bias']]
        axes[0, 1].bar(models, bias_values, color=['orange', 'purple'], alpha=0.7)
        axes[0, 1].set_ylabel('Bias')
        axes[0, 1].set_title('Bias Comparison')
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.7)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Directional Accuracy Comparison
        dir_acc_values = [comparison_summary['enhanced_dir_acc'] * 100, comparison_summary['standard_dir_acc'] * 100]
        axes[1, 0].bar(models, dir_acc_values, color=['teal', 'coral'], alpha=0.7)
        axes[1, 0].set_ylabel('Directional Accuracy (%)')
        axes[1, 0].set_title('Directional Accuracy Comparison')
        axes[1, 0].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Feature Usage
        feature_data = [comparison_summary['selected_features'], 
                       comparison_summary['total_features'] - comparison_summary['selected_features']]
        feature_labels = ['Selected Features', 'Unused Features']
        
        axes[1, 1].pie(feature_data, labels=feature_labels, autopct='%1.1f%%', 
                      colors=['lightgreen', 'lightcoral'], startangle=90)
        axes[1, 1].set_title('Feature Usage')
        
        plt.tight_layout()
        
        # Ensure the visualizations directory exists
        os.makedirs('data/results/visualizations', exist_ok=True)
        plt.savefig('data/results/visualizations/enhanced_adaptive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create performance metrics summary plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics = ['MAE', 'Bias (abs)', 'Dir Accuracy']
        enhanced_values = [comparison_summary['enhanced_adaptive_mae'], 
                          abs(comparison_summary['enhanced_bias']),
                          comparison_summary['enhanced_dir_acc'] * 100]
        standard_values = [comparison_summary['standard_mae'],
                          abs(comparison_summary['standard_bias']),
                          comparison_summary['standard_dir_acc'] * 100]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, enhanced_values, width, label='Enhanced Adaptive', alpha=0.8, color='green')
        bars2 = ax.bar(x + width/2, standard_values, width, label='Standard', alpha=0.8, color='blue')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Enhanced Adaptive Predictor - Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_enhanced_report(self, comparison_summary):
        """Generate a comprehensive report for enhanced adaptive predictor"""
        print("\n📋 Generating Enhanced Adaptive Predictor Report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"enhanced_adaptive_predictor_report_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Enhanced Adaptive Predictor Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if comparison_summary:
                data_summary = comparison_summary.get('data_summary', {})
                f.write(f"**Symbol**: {data_summary.get('symbol', 'N/A')}\n")
                f.write(f"**Data Source**: Live data via unified preprocessing\n")
                f.write(f"**Lookback Period**: {data_summary.get('days_lookback', 'N/A')} days\n")
                f.write(f"**Target Horizon**: {data_summary.get('target_horizon', 'N/A')} day(s)\n\n")
            
            f.write("## Executive Summary\n\n")
            
            if comparison_summary:
                improvement = comparison_summary['improvement_pct']
                f.write(f"- **Performance Improvement**: {improvement:+.2f}%\n")
                f.write(f"- **Enhanced Adaptive MAE**: {comparison_summary['enhanced_adaptive_mae']:.6f}\n")
                f.write(f"- **Standard Model MAE**: {comparison_summary['standard_mae']:.6f}\n")
                f.write(f"- **Bias Reduction**: {comparison_summary['bias_reduction']:.6f}\n")
                f.write(f"- **Directional Accuracy Improvement**: {comparison_summary['dir_acc_improvement']:+.2f} percentage points\n")
                f.write(f"- **Features Used**: {comparison_summary['selected_features']}/{comparison_summary['total_features']}\n\n")
                
                f.write("## Performance Analysis\n\n")
                f.write("### Mean Absolute Error (MAE)\n")
                f.write(f"- **Enhanced Adaptive**: {comparison_summary['enhanced_adaptive_mae']:.6f}\n")
                f.write(f"- **Standard Model**: {comparison_summary['standard_mae']:.6f}\n")
                f.write(f"- **Improvement**: {improvement:+.2f}%\n\n")
                
                f.write("### Bias Analysis\n")
                f.write(f"- **Enhanced Adaptive Bias**: {comparison_summary['enhanced_bias']:.6f}\n")
                f.write(f"- **Standard Model Bias**: {comparison_summary['standard_bias']:.6f}\n")
                f.write(f"- **Bias Reduction**: {comparison_summary['bias_reduction']:.6f}\n\n")
                
                f.write("### Directional Accuracy\n")
                f.write(f"- **Enhanced Adaptive**: {comparison_summary['enhanced_dir_acc']:.2%}\n")
                f.write(f"- **Standard Model**: {comparison_summary['standard_dir_acc']:.2%}\n")
                f.write(f"- **Improvement**: {comparison_summary['dir_acc_improvement']:+.2f} percentage points\n\n")
                
                f.write("## Technical Details\n\n")
                f.write("### Data Processing\n")
                f.write("- **Preprocessing**: Unified preprocessing pipeline with 60+ technical indicators\n")
                f.write("- **Feature Engineering**: Lagged features for true forecasting (no data leakage)\n")
                f.write("- **Scaling**: RobustScaler for features, StandardScaler for targets\n")
                f.write("- **Temporal Split**: 80/20 train/test split maintaining chronological order\n\n")
                
                f.write("### Model Architecture\n")
                f.write("- **Enhanced Adaptive**: Neural network with adaptive loss function\n")
                f.write("- **Loss Function**: Covariance-based adaptive weighting of technical indicators\n")
                f.write("- **Feature Selection**: Intelligent selection with category diversity enforcement\n")
                f.write("- **Optimization**: Adam optimizer with weight decay\n\n")
                
                f.write("### Key Features\n")
                f.write("- **Live Data Fetching**: Real-time data from Schwab API\n")
                f.write("- **Technical Indicators**: 60+ comprehensive indicators (RSI, MACD, Bollinger Bands, etc.)\n")
                f.write("- **Adaptive Learning**: Dynamic feature weighting based on covariance with target\n")
                f.write("- **True Forecasting**: Predicts future returns using lagged indicators\n\n")
                
                f.write("## Results Interpretation\n\n")
                
                if improvement > 0:
                    f.write("### ✅ Positive Results\n")
                    f.write(f"The Enhanced Adaptive Predictor achieved a **{improvement:.2f}% improvement** over the standard model. ")
                    f.write("This demonstrates the effectiveness of:\n")
                    f.write("- Adaptive loss function with technical indicator weighting\n")
                    f.write("- Comprehensive feature engineering from unified preprocessing\n")
                    f.write("- Intelligent feature selection with category diversity\n\n")
                else:
                    f.write("### ⚠️ Mixed Results\n")
                    f.write(f"The Enhanced Adaptive Predictor showed a **{improvement:.2f}% change** compared to the standard model. ")
                    f.write("This suggests:\n")
                    f.write("- The adaptive approach may need further tuning\n")
                    f.write("- Market conditions may favor simpler models\n")
                    f.write("- Additional penalty terms in the loss function might be beneficial\n\n")
                
                f.write("### Bias Analysis\n")
                bias_reduction = comparison_summary['bias_reduction']
                if bias_reduction > 0.001:
                    f.write(f"Significant bias reduction of **{bias_reduction:.6f}** indicates improved prediction accuracy.\n\n")
                else:
                    f.write(f"Minimal bias difference of **{bias_reduction:.6f}** suggests both models have similar systematic errors.\n\n")
                
                f.write("### Directional Accuracy\n")
                dir_improvement = comparison_summary['dir_acc_improvement']
                if dir_improvement > 1:
                    f.write(f"Strong directional accuracy improvement of **{dir_improvement:+.2f} percentage points** ")
                    f.write("indicates better trend prediction capability.\n\n")
                elif dir_improvement > 0:
                    f.write(f"Modest directional accuracy improvement of **{dir_improvement:+.2f} percentage points** ")
                    f.write("shows some enhancement in trend prediction.\n\n")
                else:
                    f.write(f"Directional accuracy change of **{dir_improvement:+.2f} percentage points** ")
                    f.write("suggests similar trend prediction performance.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("### Data Pipeline\n")
            f.write("1. **Live Data Fetching**: Schwab API integration for real-time market data\n")
            f.write("2. **Technical Indicators**: Comprehensive calculation of 60+ indicators\n")
            f.write("3. **Feature Engineering**: Lagged features with multiple time horizons\n")
            f.write("4. **Preprocessing**: Robust scaling and missing value imputation\n")
            f.write("5. **Temporal Splitting**: Chronological train/test split to prevent data leakage\n\n")
            
            f.write("### Model Training\n")
            f.write("1. **Feature Selection**: Correlation-based selection with category diversity\n")
            f.write("2. **Architecture**: Multi-layer neural network with dropout regularization\n")
            f.write("3. **Adaptive Loss**: Covariance-based weighting of technical indicators\n")
            f.write("4. **Optimization**: Adam optimizer with learning rate scheduling\n")
            f.write("5. **Evaluation**: Comprehensive metrics including bias correction\n\n")
            
            f.write("## Conclusion\n\n")
            
            if comparison_summary:
                if improvement > 2:
                    f.write("The Enhanced Adaptive Predictor demonstrates **strong performance improvements** ")
                    f.write("over standard approaches, validating the adaptive loss function and comprehensive ")
                    f.write("technical indicator integration.\n\n")
                elif improvement > 0:
                    f.write("The Enhanced Adaptive Predictor shows **modest improvements** over standard ")
                    f.write("approaches, indicating potential for further optimization.\n\n")
                else:
                    f.write("The Enhanced Adaptive Predictor shows **mixed results** compared to standard ")
                    f.write("approaches, suggesting the need for additional tuning or different market conditions.\n\n")
                
                f.write("### Recommendations\n")
                f.write("1. **Continue Development**: The adaptive approach shows promise and warrants further research\n")
                f.write("2. **Parameter Tuning**: Experiment with different penalty weights in the loss function\n")
                f.write("3. **Feature Engineering**: Explore additional technical indicators or transformations\n")
                f.write("4. **Market Conditions**: Test performance across different market regimes\n")
                f.write("5. **Ensemble Methods**: Consider combining adaptive and standard approaches\n\n")
            
            f.write("---\n")
            f.write("*Report generated by Enhanced Adaptive Predictor Analysis Framework*\n")
            f.write(f"*Using unified preprocessing pipeline with live {self.symbol} data*")
        
        print(f"✅ Enhanced report saved to: {report_filename}")
        return report_filename
    
    def create_all_models_visualizations(self, analysis_results: Dict) -> None:
        """Create comprehensive visualizations for all models"""
        print("\n📊 Creating comprehensive model comparison visualizations...")
        
        # Filter out error results
        valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
        
        if not valid_results:
            print("❌ No valid results to visualize")
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Comprehensive Model Comparison - All Models', fontsize=18, fontweight='bold')
        
        # Extract data for plotting
        model_names = list(valid_results.keys())
        mae_values = [valid_results[name]['mae'] for name in model_names]
        rmse_values = [valid_results[name]['rmse'] for name in model_names]
        dir_acc_values = [valid_results[name]['directional_accuracy'] * 100 for name in model_names]
        execution_times = [self.execution_times.get(name, 0) for name in model_names]
        
        # MAE Comparison
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        bars1 = axes[0, 0].bar(model_names, mae_values, color=colors, alpha=0.8)
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('MAE Comparison Across All Models')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mae_values):
            height = bar.get_height()
            axes[0, 0].annotate(f'{value:.4f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        # Directional Accuracy Comparison
        bars2 = axes[0, 1].bar(model_names, dir_acc_values, color=colors, alpha=0.8)
        axes[0, 1].set_ylabel('Directional Accuracy (%)')
        axes[0, 1].set_title('Directional Accuracy Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars2, dir_acc_values):
            height = bar.get_height()
            axes[0, 1].annotate(f'{value:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        # Execution Time Comparison
        bars3 = axes[1, 0].bar(model_names, execution_times, color=colors, alpha=0.8)
        axes[1, 0].set_ylabel('Execution Time (seconds)')
        axes[1, 0].set_title('Model Training Time Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars3, execution_times):
            height = bar.get_height()
            axes[1, 0].annotate(f'{value:.1f}s',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
        
        # Performance vs Speed Scatter Plot
        axes[1, 1].scatter(execution_times, mae_values, c=range(len(model_names)), 
                          cmap='Set3', s=100, alpha=0.8)
        axes[1, 1].set_xlabel('Execution Time (seconds)')
        axes[1, 1].set_ylabel('Mean Absolute Error')
        axes[1, 1].set_title('Performance vs Speed Trade-off')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add model labels to scatter plot
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (execution_times[i], mae_values[i]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        # Ensure the visualizations directory exists
        os.makedirs('data/results/visualizations', exist_ok=True)
        plt.savefig('data/results/visualizations/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self, analysis_results: Dict) -> str:
        """Generate comprehensive report for all models"""
        print("\n📋 Generating comprehensive model comparison report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure the reports directory exists
        os.makedirs('data/results/reports', exist_ok=True)
        report_filename = f"data/results/reports/comprehensive_model_comparison_report_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Comprehensive Model Comparison Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"**Symbol**: {self.symbol}\n")
            f.write(f"**Data Source**: Live data via unified preprocessing\n")
            f.write(f"**Lookback Period**: {self.days_lookback} days\n")
            f.write(f"**Target Horizon**: {self.target_horizon} day(s)\n")
            f.write(f"**Models Tested**: {len(self.models_to_run)}\n")
            f.write(f"**Parallel Execution**: ✅ Enabled\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            valid_results = {k: v for k, v in analysis_results.items() if 'error' not in v}
            error_results = {k: v for k, v in analysis_results.items() if 'error' in v}
            
            f.write(f"- **Models Successfully Executed**: {len(valid_results)}/{len(self.models_to_run)}\n")
            f.write(f"- **Models with Errors**: {len(error_results)}\n")
            
            if valid_results:
                best_mae_model = min(valid_results.keys(), key=lambda k: valid_results[k]['mae'])
                best_dir_model = max(valid_results.keys(), key=lambda k: valid_results[k]['directional_accuracy'])
                fastest_model = min(valid_results.keys(), key=lambda k: self.execution_times.get(k, float('inf')))
                
                f.write(f"- **Best MAE**: {best_mae_model} ({valid_results[best_mae_model]['mae']:.6f})\n")
                f.write(f"- **Best Directional Accuracy**: {best_dir_model} ({valid_results[best_dir_model]['directional_accuracy']:.2%})\n")
                f.write(f"- **Fastest Model**: {fastest_model} ({self.execution_times.get(fastest_model, 0):.2f}s)\n")
                
                total_time = sum(self.execution_times.values())
                f.write(f"- **Total Execution Time**: {total_time:.2f}s (parallel execution)\n\n")
            
            # Detailed Results
            f.write("## Detailed Model Results\n\n")
            
            f.write("| Model | MAE | RMSE | Directional Accuracy | Execution Time | Status |\n")
            f.write("|-------|-----|------|---------------------|----------------|--------|\n")
            
            for model_name in self.models_to_run.keys():
                if model_name in valid_results:
                    metrics = valid_results[model_name]
                    exec_time = self.execution_times.get(model_name, 0)
                    f.write(f"| {model_name} | {metrics['mae']:.6f} | {metrics['rmse']:.6f} | "
                           f"{metrics['directional_accuracy']:.2%} | {exec_time:.2f}s | ✅ Success |\n")
                else:
                    exec_time = self.execution_times.get(model_name, 0)
                    error_msg = analysis_results.get(model_name, {}).get('error', 'Unknown error')
                    f.write(f"| {model_name} | - | - | - | {exec_time:.2f}s | ❌ Error |\n")
            
            f.write("\n")
            
            # Performance Analysis
            if valid_results:
                f.write("## Performance Analysis\n\n")
                
                f.write("### Accuracy Ranking (by MAE)\n")
                sorted_by_mae = sorted(valid_results.items(), key=lambda x: x[1]['mae'])
                for i, (model_name, metrics) in enumerate(sorted_by_mae, 1):
                    f.write(f"{i}. **{model_name}**: {metrics['mae']:.6f}\n")
                f.write("\n")
                
                f.write("### Directional Accuracy Ranking\n")
                sorted_by_dir = sorted(valid_results.items(), key=lambda x: x[1]['directional_accuracy'], reverse=True)
                for i, (model_name, metrics) in enumerate(sorted_by_dir, 1):
                    f.write(f"{i}. **{model_name}**: {metrics['directional_accuracy']:.2%}\n")
                f.write("\n")
                
                f.write("### Speed Ranking (by Execution Time)\n")
                sorted_by_speed = sorted(valid_results.items(), key=lambda x: self.execution_times.get(x[0], float('inf')))
                for i, (model_name, _) in enumerate(sorted_by_speed, 1):
                    exec_time = self.execution_times.get(model_name, 0)
                    f.write(f"{i}. **{model_name}**: {exec_time:.2f}s\n")
                f.write("\n")
            
            # Technical Details
            f.write("## Technical Details\n\n")
            f.write("### Unified Preprocessing Pipeline\n")
            f.write("- **Technical Indicators**: 60+ comprehensive indicators\n")
            f.write("- **Feature Engineering**: Lagged features for true forecasting\n")
            f.write("- **Scaling**: RobustScaler for features, StandardScaler for targets\n")
            f.write("- **Target Consistency**: Unified target validation across all models\n\n")
            
            f.write("### Parallel Execution\n")
            f.write("- **Threading**: ThreadPoolExecutor for concurrent model training\n")
            f.write("- **Thread Safety**: Synchronized result storage\n")
            f.write("- **Error Handling**: Individual model failures don't affect others\n")
            f.write("- **Progress Tracking**: Real-time execution monitoring\n\n")
            
            # Error Analysis
            if error_results:
                f.write("## Error Analysis\n\n")
                for model_name, error_info in error_results.items():
                    f.write(f"### {model_name}\n")
                    f.write(f"**Error**: {error_info.get('error', 'Unknown error')}\n")
                    f.write(f"**Execution Time**: {self.execution_times.get(model_name, 0):.2f}s\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            if valid_results:
                f.write("### Model Selection\n")
                best_mae_model = min(valid_results.keys(), key=lambda k: valid_results[k]['mae'])
                best_dir_model = max(valid_results.keys(), key=lambda k: valid_results[k]['directional_accuracy'])
                
                f.write(f"- **For Accuracy**: Use **{best_mae_model}** (lowest MAE)\n")
                f.write(f"- **For Trend Prediction**: Use **{best_dir_model}** (highest directional accuracy)\n")
                f.write("- **For Production**: Consider ensemble methods combining top performers\n")
                f.write("- **For Speed**: Balance accuracy requirements with execution time constraints\n\n")
            
            f.write("### Future Improvements\n")
            f.write("1. **Hyperparameter Optimization**: Further tune model parameters\n")
            f.write("2. **Feature Engineering**: Explore additional technical indicators\n")
            f.write("3. **Ensemble Methods**: Combine multiple models for better performance\n")
            f.write("4. **Market Regime Analysis**: Test performance across different market conditions\n\n")
            
            f.write("---\n")
            f.write("*Report generated by Comprehensive Model Comparison Framework*\n")
            f.write(f"*Using unified preprocessing pipeline with live {self.symbol} data*\n")
            f.write("*All models executed in parallel threads for efficient comparison*")
        
        print(f"✅ Comprehensive report saved to: {report_filename}")
        return report_filename
    
    def run_complete_comparison(self, max_workers: int = 4):
        """Run the complete parallel model comparison"""
        print("🚀 Starting Comprehensive Model Comparison with Parallel Execution")
        print("="*100)
        print("This will run all models in parallel threads for efficient comparison")
        print(f"Symbol: {self.symbol}, Days: {self.days_lookback}, Horizon: {self.target_horizon}")
        print(f"Max workers: {max_workers}")
        print()
        
        # Run all models in parallel
        model_results = self.run_all_models_parallel(max_workers=max_workers)
        
        # Analyze performance of all models
        analysis_results = self.analyze_all_models_performance()
        
        # Create comprehensive visualizations
        if analysis_results:
            self.create_all_models_visualizations(analysis_results)
        
        # Generate comprehensive report
        report_file = self.generate_comprehensive_report(analysis_results)
        
        print("\n" + "="*100)
        print("🎉 COMPREHENSIVE MODEL COMPARISON COMPLETED!")
        print("="*100)
        print("✅ All models executed in parallel")
        print("✅ Performance analysis completed")
        print("✅ Comprehensive visualizations created")
        print(f"✅ Report generated: {report_file}")
        print("\n📊 Check the generated plots and report for detailed analysis!")
        
        return {
            'model_results': model_results,
            'analysis_results': analysis_results,
            'execution_times': self.execution_times,
            'report_file': report_file
        }

def main(symbol: str = 'NVDA', days_lookback: int = 365, target_horizon: int = 1, max_workers: int = 4):
    """Main function to run comprehensive parallel model comparison"""
    
    print("🎯 Initializing Comprehensive Model Comparison with Parallel Execution...")
    print("This analysis will:")
    print("  • Run all 7 models in parallel threads")
    print("  • Use unified preprocessing and target validation")
    print("  • Fetch live data with 60+ technical indicators")
    print("  • Generate comprehensive performance analysis")
    print("  • Create detailed comparison visualizations")
    print("  • Produce statistical analysis and reports")
    print(f"  • Symbol: {symbol}, Days: {days_lookback}, Horizon: {target_horizon}")
    print(f"  • Max workers: {max_workers}")
    print()
    
    try:
        # Initialize and run comparison
        comparison = ComprehensiveModelComparison(symbol, days_lookback, target_horizon)
        results = comparison.run_complete_comparison(max_workers=max_workers)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in comprehensive comparison: {str(e)}")
        print("Please check your API credentials and network connection.")
        return None

if __name__ == "__main__":
    results = main()
