"""
Comprehensive Model Performance Visualization Script
Reads JSON results from all models and creates detailed performance comparisons

Usage:
    python3 src/data_processing/visualize_model_performance.py
    
Features:
    - Performance metrics comparison across all models
    - Feature importance analysis and comparison
    - Prediction accuracy visualizations
    - Model timing and efficiency analysis
    - Interactive plots with detailed insights
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelPerformanceVisualizer:
    """
    Comprehensive model performance visualization and analysis
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.models_data = {}
        self.performance_df = None
        self.feature_importance_data = {}
        
    def load_model_results(self, pattern: str = "data/results/json/*_results.json") -> None:
        """Load all model results from JSON files"""
        print("🔄 Loading model results from JSON files...")
        
        json_files = glob.glob(pattern)
        if not json_files:
            raise FileNotFoundError("No JSON result files found matching pattern")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                model_name = data.get('model_name', 'Unknown Model')
                self.models_data[model_name] = data
                print(f"   ✅ Loaded: {model_name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {file_path}: {e}")
        
        print(f"✅ Loaded {len(self.models_data)} model results")
    
    def prepare_performance_dataframe(self) -> pd.DataFrame:
        """Prepare a comprehensive performance comparison DataFrame"""
        print("📊 Preparing performance comparison data...")
        
        performance_data = []
        
        for model_name, data in self.models_data.items():
            metrics = data.get('performance_metrics', {})
            config = data.get('configuration', {})
            
            # Convert string metrics to float if needed
            mae = float(metrics.get('mae', 0))
            rmse = float(metrics.get('rmse', 0))
            bias = float(metrics.get('bias', 0))
            bias_corrected_mae = float(metrics.get('bias_corrected_mae', 0))
            directional_accuracy = float(metrics.get('directional_accuracy', 0))
            mse = float(metrics.get('mse', 0))
            
            # Extract model type and timing if available
            model_type = data.get('model_type', 'unknown')
            
            performance_data.append({
                'Model': model_name,
                'Model_Type': model_type,
                'MAE': mae,
                'RMSE': rmse,
                'MSE': mse,
                'Bias': bias,
                'Bias_Corrected_MAE': bias_corrected_mae,
                'Directional_Accuracy': directional_accuracy * 100,  # Convert to percentage
                'Symbol': config.get('symbol', 'NVDA'),
                'Target_Horizon': config.get('target_horizon', 1),
                'Test_Size': config.get('test_size', 0.2)
            })
        
        self.performance_df = pd.DataFrame(performance_data)
        return self.performance_df
    
    def create_performance_overview(self) -> None:
        """Create comprehensive performance overview plots"""
        print("📈 Creating performance overview visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. MAE Comparison
        mae_data = self.performance_df.sort_values('MAE')
        bars1 = axes[0, 0].bar(range(len(mae_data)), mae_data['MAE'], 
                              color=sns.color_palette("viridis", len(mae_data)))
        axes[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].set_xticks(range(len(mae_data)))
        axes[0, 0].set_xticklabels([name.replace(' Predictor', '') for name in mae_data['Model']], 
                                  rotation=45, ha='right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars1, mae_data['MAE'])):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 2. RMSE Comparison
        rmse_data = self.performance_df.sort_values('RMSE')
        bars2 = axes[0, 1].bar(range(len(rmse_data)), rmse_data['RMSE'],
                              color=sns.color_palette("plasma", len(rmse_data)))
        axes[0, 1].set_title('Root Mean Square Error (RMSE)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xticks(range(len(rmse_data)))
        axes[0, 1].set_xticklabels([name.replace(' Predictor', '') for name in rmse_data['Model']], 
                                  rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars2, rmse_data['RMSE'])):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                           f'{value:.4f}', ha='center', va='bottom', fontsize=9)
        
        # 3. Directional Accuracy
        dir_data = self.performance_df.sort_values('Directional_Accuracy', ascending=False)
        bars3 = axes[0, 2].bar(range(len(dir_data)), dir_data['Directional_Accuracy'],
                              color=sns.color_palette("crest", len(dir_data)))
        axes[0, 2].set_title('Directional Accuracy (%)', fontweight='bold')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].set_xticks(range(len(dir_data)))
        axes[0, 2].set_xticklabels([name.replace(' Predictor', '') for name in dir_data['Model']], 
                                  rotation=45, ha='right')
        axes[0, 2].grid(axis='y', alpha=0.3)
        axes[0, 2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random (50%)')
        axes[0, 2].legend()
        
        for i, (bar, value) in enumerate(zip(bars3, dir_data['Directional_Accuracy'])):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Bias Analysis
        bias_data = self.performance_df.copy()
        bias_data['Abs_Bias'] = abs(bias_data['Bias'])
        bias_data = bias_data.sort_values('Abs_Bias')
        
        colors = ['green' if b >= 0 else 'red' for b in bias_data['Bias']]
        bars4 = axes[1, 0].bar(range(len(bias_data)), bias_data['Bias'], color=colors, alpha=0.7)
        axes[1, 0].set_title('Model Bias (Positive = Overestimate)', fontweight='bold')
        axes[1, 0].set_ylabel('Bias')
        axes[1, 0].set_xticks(range(len(bias_data)))
        axes[1, 0].set_xticklabels([name.replace(' Predictor', '') for name in bias_data['Model']], 
                                  rotation=45, ha='right')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars4, bias_data['Bias'])):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, 
                           value + (0.001 if value >= 0 else -0.001),
                           f'{value:.4f}', ha='center', 
                           va='bottom' if value >= 0 else 'top', fontsize=9)
        
        # 5. MAE vs Directional Accuracy Scatter
        axes[1, 1].scatter(self.performance_df['MAE'], self.performance_df['Directional_Accuracy'],
                          s=100, alpha=0.7, c=range(len(self.performance_df)), cmap='viridis')
        axes[1, 1].set_xlabel('Mean Absolute Error (MAE)')
        axes[1, 1].set_ylabel('Directional Accuracy (%)')
        axes[1, 1].set_title('MAE vs Directional Accuracy', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add model labels
        for i, row in self.performance_df.iterrows():
            axes[1, 1].annotate(row['Model'].replace(' Predictor', ''), 
                               (row['MAE'], row['Directional_Accuracy']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Performance Ranking Heatmap
        ranking_data = self.performance_df.copy()
        ranking_data['MAE_Rank'] = ranking_data['MAE'].rank()
        ranking_data['RMSE_Rank'] = ranking_data['RMSE'].rank()
        ranking_data['Dir_Acc_Rank'] = ranking_data['Directional_Accuracy'].rank(ascending=False)
        ranking_data['Bias_Rank'] = abs(ranking_data['Bias']).rank()
        
        heatmap_data = ranking_data[['Model', 'MAE_Rank', 'RMSE_Rank', 'Dir_Acc_Rank', 'Bias_Rank']].set_index('Model')
        heatmap_data.columns = ['MAE', 'RMSE', 'Dir Acc', 'Bias']
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', ax=axes[1, 2], 
                   cbar_kws={'label': 'Rank (1=Best)'}, fmt='.0f')
        axes[1, 2].set_title('Performance Ranking Heatmap', fontweight='bold')
        axes[1, 2].set_ylabel('')
        
        plt.tight_layout()
        plt.savefig('data/results/visualizations/comprehensive_model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_analysis(self) -> None:
        """Create feature importance comparison across models"""
        print("🔍 Creating feature importance analysis...")
        
        # Extract feature importance data
        feature_data = {}
        
        for model_name, data in self.models_data.items():
            feature_analysis = data.get('feature_analysis', {})
            
            if 'feature_importance' in feature_analysis and feature_analysis['feature_importance']:
                # For tree-based models with feature_importance arrays
                feature_names = feature_analysis.get('feature_names', [])
                importance_values = feature_analysis.get('feature_importance', [])
                
                if len(feature_names) == len(importance_values):
                    feature_data[model_name] = dict(zip(feature_names, importance_values))
            
            elif 'feature_weights' in feature_analysis:
                # For adaptive model with feature weights
                feature_data[model_name] = feature_analysis['feature_weights']
            
            elif 'coefficients' in feature_analysis and feature_analysis['coefficients']:
                # For linear models with coefficients
                feature_names = feature_analysis.get('feature_names', [])
                coefficients = feature_analysis.get('coefficients', [])
                
                if len(feature_names) == len(coefficients):
                    # Use absolute values for comparison
                    feature_data[model_name] = {name: abs(coef) for name, coef in zip(feature_names, coefficients)}
            
            elif 'top_10_features' in feature_analysis:
                # For models with top features list
                top_features = feature_analysis.get('top_10_features', [])
                if top_features and len(top_features[0]) == 2:
                    feature_data[model_name] = {name: importance for name, importance in top_features}
        
        if not feature_data:
            print("⚠️ No feature importance data found in models")
            return
        
        # Create feature importance comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Feature Importance Analysis Across Models', fontsize=16, fontweight='bold')
        
        # 1. Top 10 features for each model
        model_names = list(feature_data.keys())
        n_models = len(model_names)
        
        if n_models > 0:
            # Get top 10 features for first model as example
            first_model = model_names[0]
            top_features = sorted(feature_data[first_model].items(), key=lambda x: x[1], reverse=True)[:10]
            
            feature_names = [f[0] for f in top_features]
            importance_values = [f[1] for f in top_features]
            
            bars = axes[0, 0].barh(range(len(feature_names)), importance_values, color='skyblue')
            axes[0, 0].set_yticks(range(len(feature_names)))
            axes[0, 0].set_yticklabels([name.replace('_lag', '_L') for name in feature_names])
            axes[0, 0].set_xlabel('Importance')
            axes[0, 0].set_title(f'Top 10 Features - {first_model}', fontweight='bold')
            axes[0, 0].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, importance_values):
                axes[0, 0].text(value + max(importance_values) * 0.01, bar.get_y() + bar.get_height()/2,
                               f'{value:.3f}', va='center', fontsize=9)
        
        # 2. Feature category analysis (if available)
        category_data = {}
        for model_name, data in self.models_data.items():
            feature_analysis = data.get('feature_analysis', {})
            if 'feature_categories' in feature_analysis:
                categories = feature_analysis['feature_categories']
                category_counts = {}
                for feature, category in categories.items():
                    category_counts[category] = category_counts.get(category, 0) + 1
                category_data[model_name] = category_counts
        
        if category_data:
            # Create category comparison
            all_categories = set()
            for counts in category_data.values():
                all_categories.update(counts.keys())
            
            category_matrix = []
            model_labels = []
            for model_name, counts in category_data.items():
                model_labels.append(model_name.replace(' Predictor', ''))
                row = [counts.get(cat, 0) for cat in sorted(all_categories)]
                category_matrix.append(row)
            
            category_df = pd.DataFrame(category_matrix, 
                                     index=model_labels, 
                                     columns=sorted(all_categories))
            
            sns.heatmap(category_df, annot=True, cmap='Blues', ax=axes[0, 1], fmt='d')
            axes[0, 1].set_title('Feature Categories by Model', fontweight='bold')
            axes[0, 1].set_ylabel('Models')
        
        # 3. Common important features across models
        if len(feature_data) > 1:
            # Find features that appear in multiple models
            all_features = set()
            for features in feature_data.values():
                all_features.update(features.keys())
            
            # Calculate average importance for common features
            common_features = {}
            for feature in all_features:
                importances = []
                for model_features in feature_data.values():
                    if feature in model_features:
                        importances.append(model_features[feature])
                
                if len(importances) > 1:  # Feature appears in multiple models
                    common_features[feature] = {
                        'avg_importance': np.mean(importances),
                        'std_importance': np.std(importances),
                        'model_count': len(importances)
                    }
            
            if common_features:
                # Plot common features
                sorted_common = sorted(common_features.items(), 
                                     key=lambda x: x[1]['avg_importance'], reverse=True)[:15]
                
                feature_names = [f[0] for f in sorted_common]
                avg_importances = [f[1]['avg_importance'] for f in sorted_common]
                std_importances = [f[1]['std_importance'] for f in sorted_common]
                
                bars = axes[1, 0].barh(range(len(feature_names)), avg_importances, 
                                      xerr=std_importances, color='lightcoral', alpha=0.7)
                axes[1, 0].set_yticks(range(len(feature_names)))
                axes[1, 0].set_yticklabels([name.replace('_lag', '_L') for name in feature_names])
                axes[1, 0].set_xlabel('Average Importance')
                axes[1, 0].set_title('Common Important Features (Avg ± Std)', fontweight='bold')
                axes[1, 0].grid(axis='x', alpha=0.3)
        
        # 4. Model-specific feature importance comparison
        if len(feature_data) >= 2:
            # Compare top features between two models
            model1, model2 = list(feature_data.keys())[:2]
            
            # Get common features
            common_features = set(feature_data[model1].keys()) & set(feature_data[model2].keys())
            
            if common_features:
                model1_imp = [feature_data[model1][f] for f in common_features]
                model2_imp = [feature_data[model2][f] for f in common_features]
                
                axes[1, 1].scatter(model1_imp, model2_imp, alpha=0.7, s=60)
                axes[1, 1].set_xlabel(f'{model1.replace(" Predictor", "")} Importance')
                axes[1, 1].set_ylabel(f'{model2.replace(" Predictor", "")} Importance')
                axes[1, 1].set_title('Feature Importance Correlation', fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Add diagonal line
                max_val = max(max(model1_imp), max(model2_imp))
                axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
                
                # Add correlation coefficient
                correlation = np.corrcoef(model1_imp, model2_imp)[0, 1]
                axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                               transform=axes[1, 1].transAxes, fontsize=12,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('data/results/visualizations/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_prediction_analysis(self) -> None:
        """Create prediction accuracy and residual analysis"""
        print("🎯 Creating prediction analysis...")
        
        # Extract predictions and actuals
        prediction_data = {}
        
        for model_name, data in self.models_data.items():
            predictions_data = data.get('predictions', {})
            if 'test_predictions' in predictions_data and 'test_actuals' in predictions_data:
                pred = predictions_data['test_predictions']
                actual = predictions_data['test_actuals']
                
                # Flatten if nested lists
                if pred and isinstance(pred[0], list):
                    pred = [p[0] if isinstance(p, list) else p for p in pred]
                if actual and isinstance(actual[0], list):
                    actual = [a[0] if isinstance(a, list) else a for a in actual]
                
                prediction_data[model_name] = {
                    'predictions': np.array(pred),
                    'actuals': np.array(actual)
                }
        
        if not prediction_data:
            print("⚠️ No prediction data found in models")
            return
        
        n_models = len(prediction_data)
        fig, axes = plt.subplots(2, min(3, n_models), figsize=(18, 12))
        if n_models == 1:
            axes = axes.reshape(2, 1)
        elif n_models == 2:
            axes = axes.reshape(2, 2)
        
        fig.suptitle('Prediction Analysis Across Models', fontsize=16, fontweight='bold')
        
        model_names = list(prediction_data.keys())
        
        for i, model_name in enumerate(model_names[:3]):  # Show first 3 models
            data = prediction_data[model_name]
            predictions = data['predictions']
            actuals = data['actuals']
            
            col_idx = i if n_models > 1 else 0
            
            # Predictions vs Actuals scatter plot
            axes[0, col_idx].scatter(actuals, predictions, alpha=0.6, s=30)
            
            # Perfect prediction line
            min_val = min(min(actuals), min(predictions))
            max_val = max(max(actuals), max(predictions))
            axes[0, col_idx].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
            
            axes[0, col_idx].set_xlabel('Actual Returns')
            axes[0, col_idx].set_ylabel('Predicted Returns')
            axes[0, col_idx].set_title(f'{model_name.replace(" Predictor", "")} - Predictions vs Actuals')
            axes[0, col_idx].grid(True, alpha=0.3)
            
            # Add R² score
            r2 = np.corrcoef(actuals, predictions)[0, 1] ** 2
            axes[0, col_idx].text(0.05, 0.95, f'R² = {r2:.3f}', 
                                 transform=axes[0, col_idx].transAxes, fontsize=10,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residuals plot
            residuals = predictions - actuals
            axes[1, col_idx].scatter(predictions, residuals, alpha=0.6, s=30, color='green')
            axes[1, col_idx].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[1, col_idx].set_xlabel('Predicted Returns')
            axes[1, col_idx].set_ylabel('Residuals')
            axes[1, col_idx].set_title(f'{model_name.replace(" Predictor", "")} - Residuals')
            axes[1, col_idx].grid(True, alpha=0.3)
            
            # Add residual statistics
            residual_std = np.std(residuals)
            axes[1, col_idx].text(0.05, 0.95, f'Residual Std: {residual_std:.4f}', 
                                 transform=axes[1, col_idx].transAxes, fontsize=10,
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        if n_models < 3:
            for i in range(n_models, min(3, axes.shape[1])):
                axes[0, i].set_visible(False)
                axes[1, i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('data/results/visualizations/prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_summary_report(self) -> str:
        """Create a comprehensive summary report"""
        print("📋 Generating comprehensive model summary report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"data/results/reports/model_performance_summary_{timestamp}.md"
        
        with open(report_filename, 'w') as f:
            f.write("# Comprehensive Model Performance Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Best performers
            best_mae = self.performance_df.loc[self.performance_df['MAE'].idxmin()]
            best_dir_acc = self.performance_df.loc[self.performance_df['Directional_Accuracy'].idxmax()]
            lowest_bias = self.performance_df.loc[abs(self.performance_df['Bias']).idxmin()]
            
            f.write(f"### 🏆 Best Performers\n")
            f.write(f"- **Lowest MAE**: {best_mae['Model']} ({best_mae['MAE']:.6f})\n")
            f.write(f"- **Best Directional Accuracy**: {best_dir_acc['Model']} ({best_dir_acc['Directional_Accuracy']:.1f}%)\n")
            f.write(f"- **Lowest Bias**: {lowest_bias['Model']} ({lowest_bias['Bias']:.6f})\n\n")
            
            f.write("## Detailed Performance Metrics\n\n")
            f.write("| Model | MAE | RMSE | Bias | Dir Acc (%) | Model Type |\n")
            f.write("|-------|-----|------|------|-------------|------------|\n")
            
            for _, row in self.performance_df.iterrows():
                f.write(f"| {row['Model']} | {row['MAE']:.6f} | {row['RMSE']:.6f} | "
                       f"{row['Bias']:+.6f} | {row['Directional_Accuracy']:.1f}% | {row['Model_Type']} |\n")
            
            f.write("\n## Key Insights\n\n")
            
            # Performance insights
            avg_mae = self.performance_df['MAE'].mean()
            avg_dir_acc = self.performance_df['Directional_Accuracy'].mean()
            
            f.write(f"### Performance Analysis\n")
            f.write(f"- **Average MAE across all models**: {avg_mae:.6f}\n")
            f.write(f"- **Average Directional Accuracy**: {avg_dir_acc:.1f}%\n")
            f.write(f"- **Models above 50% directional accuracy**: {len(self.performance_df[self.performance_df['Directional_Accuracy'] > 50])}/{len(self.performance_df)}\n")
            
            # Bias analysis
            positive_bias = len(self.performance_df[self.performance_df['Bias'] > 0])
            negative_bias = len(self.performance_df[self.performance_df['Bias'] < 0])
            
            f.write(f"\n### Bias Analysis\n")
            f.write(f"- **Models with positive bias (overestimate)**: {positive_bias}\n")
            f.write(f"- **Models with negative bias (underestimate)**: {negative_bias}\n")
            
            # Model type analysis
            f.write(f"\n### Model Type Distribution\n")
            type_counts = self.performance_df['Model_Type'].value_counts()
            for model_type, count in type_counts.items():
                f.write(f"- **{model_type}**: {count} model(s)\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("### For Production Use\n")
            f.write(f"1. **Primary Model**: {best_mae['Model']} - Best overall accuracy (lowest MAE)\n")
            f.write(f"2. **Direction Prediction**: {best_dir_acc['Model']} - Best for trend following strategies\n")
            f.write(f"3. **Unbiased Predictions**: {lowest_bias['Model']} - Most calibrated predictions\n")
            
            f.write("\n### Model Ensemble Strategy\n")
            f.write("Consider combining multiple models for improved robustness:\n")
            top_3_mae = self.performance_df.nsmallest(3, 'MAE')['Model'].tolist()
            for i, model in enumerate(top_3_mae, 1):
                f.write(f"{i}. {model}\n")
            
            f.write("\n### Areas for Improvement\n")
            worst_mae = self.performance_df.loc[self.performance_df['MAE'].idxmax()]
            worst_dir_acc = self.performance_df.loc[self.performance_df['Directional_Accuracy'].idxmin()]
            
            f.write(f"- **{worst_mae['Model']}**: Highest MAE - consider hyperparameter tuning\n")
            f.write(f"- **{worst_dir_acc['Model']}**: Lowest directional accuracy - review feature engineering\n")
            
            f.write("\n## Technical Configuration\n\n")
            sample_config = list(self.models_data.values())[0].get('configuration', {})
            f.write(f"- **Symbol**: {sample_config.get('symbol', 'NVDA')}\n")
            f.write(f"- **Target Horizon**: {sample_config.get('target_horizon', 1)} day(s)\n")
            f.write(f"- **Test Size**: {sample_config.get('test_size', 0.2)}\n")
            f.write(f"- **Total Models Analyzed**: {len(self.models_data)}\n")
            
            f.write("\n---\n")
            f.write("*Report generated by Model Performance Visualizer*\n")
            f.write("*Comprehensive analysis of all stock prediction models*")
        
        print(f"✅ Report saved to: {report_filename}")
        return report_filename
    
    def run_complete_analysis(self) -> None:
        """Run the complete model performance analysis"""
        print("🚀 Starting comprehensive model performance analysis...")
        print("="*80)
        
        # Load all model results
        self.load_model_results()
        
        # Prepare performance data
        self.prepare_performance_dataframe()
        
        # Create all visualizations
        self.create_performance_overview()
        self.create_feature_importance_analysis()
        self.create_prediction_analysis()
        
        # Generate summary report
        report_file = self.create_model_summary_report()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE ANALYSIS COMPLETED!")
        print("="*80)
        print(f"✅ Performance overview: data/results/visualizations/comprehensive_model_performance.png")
        print(f"✅ Feature analysis: data/results/visualizations/feature_importance_analysis.png")
        print(f"✅ Prediction analysis: data/results/visualizations/prediction_analysis.png")
        print(f"✅ Summary report: {report_file}")
        print("\n📊 Check the generated visualizations and report in the data/results/ directory!")

def main():
    """Main function to run the visualization analysis"""
    visualizer = ModelPerformanceVisualizer()
    visualizer.run_complete_analysis()

if __name__ == "__main__":
    main()
