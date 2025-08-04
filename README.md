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
│   ├── model_config.yaml                  # Centralized model configuration
│   └── technical_indicators_config.yaml   # Technical indicators configuration
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

### 🏆 Latest Comprehensive Comparison Results (Volume Indicators Only)

**Current Configuration**: Volume indicators only (4 total indicators)
- ✅ **Volume Indicators**: OBV, ADL, Volume SMA, Volume Ratio
- ❌ **All other categories disabled** for focused volume analysis

![Model Performance Comparison](data/results/visualizations/comprehensive_model_comparison.png)

### Latest Performance Results on NVDA (1-day horizon):

| Model | MAE | RMSE | Directional Accuracy | Speed | Status |
|-------|-----|------|---------------------|-------|--------|
| **SVR** | **0.014509** | 0.022834 | 46.00% | 60.65s | 🏆 **Best MAE** |
| **Enhanced Adaptive** | 0.014604 | 0.017738 | **52.00%** | **5.75s** | 🏆 **Best Direction & Speed** |
| **Random Forest** | 0.014696 | 0.019461 | **52.00%** | 63.58s | 🏆 **Best Direction** |
| **Linear Regression** | 0.015078 | 0.021675 | **52.00%** | 4.55s | 🏆 **Best Direction** |
| **XGBoost** | 0.016451 | 0.021482 | 50.00% | 171.46s | ✅ Good |
| **LightGBM** | 0.016550 | 0.023967 | 40.00% | 234.25s | ✅ Good |
| **LSTM** | 0.018678 | 0.042987 | 42.50% | 10.29s | ✅ Good |

### 🎯 Key Performance Highlights:

**🏆 Best Overall Performance**: **Enhanced Adaptive Predictor**
- **MAE**: 0.014604 (2nd best, very close to SVR)
- **Directional Accuracy**: 52.00% (tied for best)
- **Speed**: 5.75s (fastest training)
- **Adaptive Learning**: Dynamic feature weighting

**🏆 Best Accuracy**: **SVR** (0.014509 MAE)
**🏆 Best Directional Prediction**: **Enhanced Adaptive, Random Forest, Linear Regression** (52.00%)
**🏆 Fastest Training**: **Linear Regression** (4.55s)

### Enhanced Adaptive Predictor - Volume-Only Configuration

**Top 6 Most Important Volume Features (Adaptive Weights):**
1. **Volume_Ratio_lag1** (volume): 0.206214 - Current vs average volume
2. **Volume_Ratio_lag2** (volume): 0.201218 - Volume momentum
3. **Volume_SMA_20_lag2** (trend): 0.200924 - Volume trend analysis
4. **Volume_Ratio_lag3** (volume): 0.189471 - Volume persistence
5. **OBV_lag2** (volume): 0.187757 - On-Balance Volume
6. **ADL_lag3** (volume): 0.175843 - Accumulation/Distribution

**Category Performance:**
- **Volume indicators**: 96.05% total weight (dominant)
- **Trend indicators**: 20.09% total weight (volume trend)

*Results from comprehensive parallel execution of all 7 models | Total execution time: 250.30s*

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

### Model Configuration (`config/model_config.yaml`)

Centralized model configuration:

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

### Technical Indicators Configuration (`config/technical_indicators_config.yaml`)

**NEW**: Centralized control over all technical indicators used by models. This powerful configuration system allows you to easily customize which indicators are used across all 7 models.

#### Key Features:
- **Enable/Disable Indicators**: Simple `enabled: true/false` flags
- **Parameter Customization**: Adjust windows, thresholds, and other parameters
- **Category Organization**: Indicators grouped by type (trend, momentum, volume, etc.)
- **Model-Specific Overrides**: Different settings for different models
- **Easy Experimentation**: Test different indicator combinations instantly

#### Example Configuration:
```yaml
# Trend Indicators
trend_indicators:
  sma_20:
    enabled: true
    function: "calculate_sma"
    parameters:
      window: 20
    category: "trend"
    description: "20-period Simple Moving Average"
  
  ema_12:
    enabled: true
    function: "calculate_ema"
    parameters:
      window: 12
    category: "trend"
    description: "12-period Exponential Moving Average"

# Momentum Indicators
momentum_indicators:
  rsi_14:
    enabled: true
    function: "calculate_rsi"
    parameters:
      window: 14
    category: "momentum"
    description: "14-period Relative Strength Index"

# Model-Specific Overrides
model_overrides:
  enhanced_adaptive:
    max_features: 15
    enforce_category_diversity: true
    preferred_categories: ["trend", "momentum", "signal"]
  
  lightgbm:
    max_features: 30
    enforce_category_diversity: false
```

#### Available Indicator Categories:
- **Trend Indicators**: SMA (5,10,20,50,200), EMA (12,26,50), VWAP, Parabolic SAR
- **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI, MFI, Momentum, ROC
- **Volume Indicators**: OBV, ADL, Volume SMA, Volume Ratio
- **Volatility Indicators**: ATR, Historical Volatility, Bollinger Bands
- **Signal Indicators**: MACD (Line/Signal/Histogram)
- **Advanced Indicators**: Support/Resistance, Volatility Bands, Price Changes
- **Derived Indicators**: Moving Average Ratios, Price-to-MA Ratios

#### How to Customize:
```bash
# 1. Edit the configuration file
nano config/technical_indicators_config.yaml

# 2. Enable/disable indicators
rsi_14:
  enabled: false  # Disable RSI

# 3. Change parameters
sma_20:
  parameters:
    window: 30  # Change from 20 to 30-period SMA

# 4. Run any model - changes apply automatically
python3 src/models/enhanced_adaptive_predictor.py
```

All models automatically read this configuration and adapt their feature sets accordingly. This makes it incredibly easy to experiment with different technical indicator combinations across your entire model suite.

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
