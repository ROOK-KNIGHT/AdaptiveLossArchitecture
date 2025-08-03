# Adaptive Loss Function Architecture

A comprehensive, production-ready implementation of adaptive loss functions for financial time series prediction, featuring 7 different models, unified preprocessing pipeline, and advanced technical indicator analysis.

## GOAL - What This Project Does (In Simple Terms)

**The Problem**: Traditional stock prediction models treat all market indicators equally, but some indicators are more useful than others at different times.

**Our Solution**: The Enhanced Adaptive Predictor is a "smart" model that automatically figures out which market indicators (like volume, price trends, momentum) are most important for predicting stock prices. It continuously adjusts how much attention it pays to each indicator based on how well they're performing.

**Real-World Analogy**: Imagine you're a weather forecaster with 60 different instruments (thermometer, barometer, wind gauge, etc.). Instead of treating all readings equally, our adaptive model is like having an intelligent system that automatically knows "today the wind patterns are most important" or "right now the pressure readings are the best predictor" and adjusts accordingly.

**The Result**: More accurate stock price predictions because the model focuses on the most relevant market signals at any given time, rather than using a one-size-fits-all approach.

## Architecture Overview

This project implements a sophisticated multi-model comparison framework with adaptive loss functions that dynamically adjust feature weights based on their predictive performance. The architecture is designed for modularity, scalability, and research reproducibility with live data integration.

```
AdaptiveLossArchitecture/
├── src/
│   ├── models/                               # 7 Production-Ready Models
│   │   ├── enhanced_adaptive_predictor.py    # Adaptive loss with neural networks
│   │   ├── lstm_predictor.py                 # LSTM for sequential modeling
│   │   ├── xgboost_predictor.py             # Gradient boosting with SHAP
│   │   ├── random_forest_predictor.py        # Ensemble learning
│   │   ├── lightgbm_predictor.py            # Efficient gradient boosting
│   │   ├── svr_predictor.py                 # Support Vector Regression
│   │   └── linear_regression_predictor.py    # Regularized linear models
│   ├── data_processing/                      # Unified Data Pipeline
│   │   ├── unified_preprocessing.py          # 60+ technical indicators
│   │   ├── unified_targets.py               # Dual-target system
│   │   ├── technical_indicators_calculator.py # Advanced indicators
│   │   └── visualize_model_performance.py    # Performance visualization
│   ├── evaluation/                          # Model Evaluation
│   │   └── comprehensive_comparison.py       # Parallel model comparison
│   └── utils/                               # Utilities
│       └── target_integration.py            # Target validation
├── data/                                    # Organized Data Structure
│   └── results/
│       ├── json/                           # Model results (JSON format)
│       ├── visualizations/                 # Performance plots
│       └── reports/                        # Analysis reports
├── config/                                 # Configuration Management
│   ├── model_config.yaml                  # Centralized configuration
│   └── config_loader.py                   # Config utilities
├── handlers/                               # Data Handlers
│   ├── fetch_data.py                      # Live data fetching
│   ├── enhanced_data_processor.py         # Data processing
│   └── connection_manager.py              # API connections
├── venv/                                  # Virtual environment
├── requirements.txt                       # Python dependencies
└── README.md                             # This file
```

## Key Features

### Advanced Model Suite
- **Enhanced Adaptive Predictor**: Proprietary adaptive loss with dynamic feature weighting
- **LSTM Networks**: Sequential modeling with memory mechanisms
- **XGBoost**: Gradient boosting with SHAP interpretability
- **Random Forest**: Ensemble learning with feature importance
- **LightGBM**: Efficient gradient boosting for large datasets
- **SVR**: Non-linear support vector regression with kernel optimization
- **Linear Regression**: Regularized models (Ridge, Lasso, ElasticNet)

### Unified Data Pipeline
- **Live Data Integration**: Real-time data from Schwab API
- **60+ Technical Indicators**: Comprehensive indicator suite
- **Dual-Target System**: Returns and price prediction
- **Advanced Feature Engineering**: Lagged features for true forecasting
- **Robust Preprocessing**: Missing value handling, scaling, validation

### Adaptive Loss Innovation
- **Dynamic Weight Adjustment**: Weights adapt based on feature-target covariance
- **Category-Aware Learning**: Technical indicator categorization
- **Penalty Terms**: Temporal, volatility, and smoothness penalties
- **Real-Time Adaptation**: Continuous weight updates during training

### Production Features
- **Parallel Execution**: Multi-threaded model training
- **Comprehensive Evaluation**: Statistical significance testing
- **Interactive Visualizations**: Performance comparison plots
- **Automated Reporting**: Detailed analysis reports
- **Configuration Management**: YAML-based centralized config

## Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Schwab API credentials (for live data)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/yourusername/AdaptiveLossArchitecture.git
cd AdaptiveLossArchitecture

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API credentials (optional - for live data)
# Edit config/model_config.yaml with your API keys
```

## Usage

### Run Individual Models
```bash
# Enhanced Adaptive Predictor
python3 src/models/enhanced_adaptive_predictor.py

# LSTM Predictor
python3 src/models/lstm_predictor.py

# XGBoost with SHAP analysis
python3 src/models/xgboost_predictor.py

# Any other model...
python3 src/models/[model_name].py
```

### Comprehensive Model Comparison
```bash
# Run all 7 models in parallel
python3 src/evaluation/comprehensive_comparison.py
```

### Performance Visualization
```bash
# Generate comprehensive performance analysis
python3 src/data_processing/visualize_model_performance.py
```

## Model Performance

Recent performance on NVDA (1-day horizon):

| Model | MAE | RMSE | Directional Accuracy | Speed |
|-------|-----|------|---------------------|-------|
| **LSTM** | 0.013498 | 0.020249 | 47.50% | 63.47s |
| **SVR** | 0.013932 | 0.019268 | 48.00% | 127.51s |
| **Random Forest** | 0.014197 | 0.020961 | 38.00% | 167.08s |
| **LightGBM** | 0.014801 | 0.022250 | 34.00% | 557.81s |
| **Linear Regression** | 0.015998 | 0.020866 | **66.00%** | 53.11s |
| **XGBoost** | 0.017638 | 0.023474 | 48.00% | 630.90s |
| **Enhanced Adaptive** | 0.020787 | 0.024640 | 50.00% | **29.63s** |

*Best MAE: LSTM | Best Directional Accuracy: Linear Regression | Fastest: Enhanced Adaptive*

## Technical Innovation

### Adaptive Loss Function
The core innovation is a covariance-based adaptive loss function that dynamically weights technical indicators:

```python
def forward(self, predictions, target, features):
    # Base MSE loss
    mse_loss = self.mse_loss(predictions, target)
    
    # Adaptive covariance-based weighting
    total_correlation_loss = 0.0
    for i, feature_name in enumerate(self.feature_names):
        cov_loss = self.compute_covariance_loss(features, target, i)
        total_correlation_loss += self.weights[feature_name] * cov_loss
    
    # Enhanced penalty terms
    temporal_loss = self.compute_temporal_consistency_loss(predictions)
    volatility_loss = self.compute_volatility_penalty(predictions, target)
    
    return mse_loss + total_correlation_loss + penalties
```

### Technical Indicators Suite
60+ comprehensive technical indicators including:
- **Trend**: SMA, EMA, VWAP, Bollinger Bands
- **Momentum**: RSI, MACD, Stochastic, CCI, MFI
- **Volume**: OBV, ADL, Volume ratios
- **Volatility**: ATR, Bollinger Band width, custom volatility metrics
- **Advanced**: Parabolic SAR, Williams %R, Rate of Change

### True Forecasting Methodology
- **Lagged Features**: All indicators use historical lags (1, 2, 3 periods)
- **No Data Leakage**: Strict temporal separation
- **Forward-Looking**: Predicts future returns using past indicators
- **Bias Correction**: Statistical bias adjustment in evaluation

## Output Structure

All results are organized in a clean directory structure:

```
data/results/
├── json/                    # Model results in JSON format
│   ├── enhanced_adaptive_predictor_results.json
│   ├── lstm_predictor_results.json
│   ├── xgboost_predictor_results.json
│   └── [other_model_results].json
├── visualizations/          # Performance plots and charts
│   ├── comprehensive_model_performance.png
│   ├── feature_importance_analysis.png
│   └── prediction_analysis.png
└── reports/                # Detailed analysis reports
    ├── comprehensive_model_comparison_report.md
    └── model_performance_summary.md
```

## Configuration

Centralized configuration in `config/model_config.yaml`:

```yaml
# Model parameters
model:
  hidden_size: 128
  dropout_rates:
    layer1: 0.3
    layer2: 0.4
    layer3: 0.5

# Training parameters
training:
  epochs: 150
  batch_size: 32
  learning_rate: 0.001

# Data processing
data_processing:
  days_lookback: 365
  target_horizon: 1
  test_size: 0.2
  lag_periods: [1, 2, 3]

# Adaptive loss configuration
adaptive_loss:
  min_weight: 0.01
  max_weight: 0.5
  learning_rate_factor: 0.1
```

## Key Insights

### Model Strengths
- **LSTM**: Best overall accuracy (lowest MAE)
- **Linear Regression**: Exceptional directional accuracy (66%)
- **Enhanced Adaptive**: Fastest training with adaptive learning
- **XGBoost**: Strong feature importance with SHAP analysis
- **SVR**: Robust non-linear modeling

### Technical Findings
- **Volume indicators** (OBV, ADL) show strong predictive power
- **Momentum indicators** (MACD, RSI) provide directional signals
- **Trend indicators** (VWAP, SMA) offer stability
- **Adaptive weighting** improves feature utilization efficiency

## Advanced Features

### Parallel Processing
- Multi-threaded model execution
- Thread-safe result storage
- Progress monitoring
- Error isolation

### Live Data Integration
- Real-time Schwab API integration
- Automatic data refresh
- Historical data caching
- API error handling

### Comprehensive Evaluation
- Bias-corrected metrics
- Directional accuracy analysis
- Statistical significance testing
- Cross-model comparison

## Development

### Adding New Models
1. Create new model file in `src/models/`
2. Implement unified preprocessing integration
3. Add to comprehensive comparison framework
4. Update configuration as needed

### Extending Technical Indicators
1. Add indicators to `technical_indicators_calculator.py`
2. Update feature categorization
3. Test with existing models
4. Document new indicators

## Research Applications

This architecture supports various research directions:
- **Adaptive Learning**: Dynamic weight adjustment mechanisms
- **Multi-Model Ensembles**: Combining different model strengths
- **Feature Engineering**: Advanced technical indicator development
- **Market Regime Analysis**: Performance across different market conditions
- **Risk Management**: Volatility and bias analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Technical indicator implementations based on TA-Lib
- Neural network architectures inspired by modern deep learning research
- Financial modeling best practices from quantitative finance literature

---

**Built for researchers, traders, and developers who demand production-ready financial ML systems.**
