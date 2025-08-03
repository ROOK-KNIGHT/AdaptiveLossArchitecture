"""
LSTM Stock Price Predictor with Unified Targets
Uses Long Short-Term Memory neural networks for time series prediction

Usage:
    python3 src/models/lstm_predictor.py
    
Features:
    - LSTM neural network optimized for time series
    - Uses unified preprocessing and target system
    - Sequence-based prediction with lookback windows
    - Comprehensive evaluation metrics
    - Configuration-driven approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

class LSTMModel(nn.Module):
    """LSTM neural network for stock price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and final linear layer
        output = self.dropout(last_output)
        output = self.fc(output)
        
        return output

class LSTMStockPredictor:
    """
    LSTM predictor using unified preprocessing and targets
    """
    
    def __init__(self, symbol: str = None, days_lookback: int = None, target_horizon: int = None):
        """Initialize the LSTM predictor"""
        
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
        self.target_manager.validate_target_consistency(model_target_config, "LSTM Predictor")
        
        # LSTM-specific parameters
        self.sequence_length = 10  # Look back 10 time steps
        self.hidden_size = 64
        self.num_layers = 2
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.epochs = 100
        self.batch_size = 32
        
        # Initialize model
        self.model = None
        self.data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X)):
            sequences.append(X[i-self.sequence_length:i])
            targets.append(y[i])
        
        return np.array(sequences), np.array(targets)
    
    def prepare_data_for_lstm(self) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for LSTM training"""
        print(f"\n🔄 Preparing sequence data for LSTM (sequence length: {self.sequence_length})...")
        
        # Get scaled data
        X_train_scaled = self.data['X_train_scaled']
        X_test_scaled = self.data['X_test_scaled']
        y_train_scaled = self.data['y_returns_train_scaled']
        y_test_scaled = self.data['y_returns_test_scaled']
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        print(f"   Training sequences: {len(X_train_seq)}")
        print(f"   Test sequences: {len(X_test_seq)}")
        print(f"   Sequence shape: {X_train_seq.shape}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).reshape(-1, 1).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)  # Don't shuffle time series
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train_model(self, train_loader: DataLoader) -> Dict[str, List[float]]:
        """Train the LSTM model"""
        print(f"\n🧠 Training LSTM model...")
        print(f"   Device: {self.device}")
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Num layers: {self.num_layers}")
        print(f"   Epochs: {self.epochs}")
        
        # Get input size from first batch
        sample_batch = next(iter(train_loader))
        input_size = sample_batch[0].shape[2]
        
        # Initialize model
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        history = {'loss': []}
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            history['loss'].append(avg_loss)
            
            # Print progress
            if epoch % 20 == 0 or epoch == self.epochs - 1:
                print(f"   Epoch {epoch:3d}/{self.epochs}: Loss = {avg_loss:.6f}")
        
        print("✅ LSTM training completed!")
        return history
    
    def evaluate_model(self, test_loader: DataLoader, scaler_y: Any, train_test_diff: float = 0) -> Dict:
        """Evaluate the LSTM model"""
        print("\n📈 Evaluating LSTM model...")
        
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Convert back to original scale
        predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(actuals_orig, predictions_orig)
        mse = mean_squared_error(actuals_orig, predictions_orig)
        rmse = np.sqrt(mse)
        
        # Bias metrics
        bias = np.mean(predictions_orig - actuals_orig)
        bias_corrected_predictions = predictions_orig - bias
        bias_corrected_mae = mean_absolute_error(actuals_orig, bias_corrected_predictions)
        
        # Directional accuracy
        actual_direction = np.sign(actuals_orig)
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
            'actuals': actuals_orig,
            'bias_metrics': {
                'bias': bias,
                'corrected_mae': bias_corrected_mae
            }
        }
        
        print(f"✅ LSTM Evaluation Results:")
        print(f"   MAE: {mae:.6f}")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   Bias: {bias:.6f}")
        print(f"   Bias-Corrected MAE: {bias_corrected_mae:.6f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2%}")
        
        return results
    
    def create_visualizations(self, results: Dict, history: Dict) -> None:
        """Create visualizations for LSTM analysis"""
        print("\n📊 Creating LSTM visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('LSTM Stock Predictor Analysis', fontsize=16, fontweight='bold')
        
        # Training loss
        axes[0, 0].plot(history['loss'], color='blue', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions vs Actuals
        axes[0, 1].scatter(results['actuals'], results['predictions'], alpha=0.6, color='green')
        axes[0, 1].plot([results['actuals'].min(), results['actuals'].max()], 
                       [results['actuals'].min(), results['actuals'].max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Returns')
        axes[0, 1].set_ylabel('Predicted Returns')
        axes[0, 1].set_title('Predictions vs Actuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series plot (last 50 predictions)
        n_plot = min(50, len(results['predictions']))
        x_range = range(n_plot)
        axes[1, 0].plot(x_range, results['actuals'][-n_plot:], label='Actual', color='blue', linewidth=2)
        axes[1, 0].plot(x_range, results['predictions'][-n_plot:], label='Predicted', color='red', linewidth=2)
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Returns')
        axes[1, 0].set_title(f'Time Series Prediction (Last {n_plot} points)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
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
        plt.savefig('lstm_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    

def main(symbol: str = None, days_lookback: int = None, target_horizon: int = None):
    """Main function for LSTM predictor"""
    
    print("=" * 80)
    print("LSTM STOCK PRICE PREDICTOR WITH UNIFIED TARGETS")
    print("=" * 80)
    print("Features: Sequential modeling + unified preprocessing + memory networks")
    print()
    
    # Initialize predictor
    predictor = LSTMStockPredictor(symbol, days_lookback, target_horizon)
    
    # Get data
    data = predictor.get_data()
    
    print(f"\n📊 Data Summary:")
    print(f"   Features: {data['total_features']} (lagged technical indicators)")
    print(f"   Training samples: {len(data['X_train'])}")
    print(f"   Test samples: {len(data['X_test'])}")
    print(f"   Target: Returns (log returns)")
    
    # Prepare sequence data
    train_loader, test_loader = predictor.prepare_data_for_lstm()
    
    # Train model
    history = predictor.train_model(train_loader)
    
    # Evaluate model
    results = predictor.evaluate_model(
        test_loader, 
        data['scaler_y_returns'], 
        data['train_test_returns_diff']
    )
    
    # Skip visualizations - save data to JSON instead
    # predictor.create_visualizations(results, history)
    
    print("\n" + "="*80)
    print("LSTM PREDICTOR SUMMARY")
    print("="*80)
    print(f"✅ Model trained and evaluated successfully")
    print(f"📊 MAE: {results['mae']:.6f}")
    print(f"📊 Directional Accuracy: {results['directional_accuracy']:.2%}")
    print(f"🎯 Using unified targets and preprocessing")
    print(f"🧠 Sequential modeling with {predictor.sequence_length}-step lookback")
    
    # Save comprehensive results to JSON
    results_data = {
        'model_name': 'LSTM Predictor',
        'model_type': 'recurrent_neural_network_lstm',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'symbol': predictor.symbol,
            'days_lookback': predictor.days_lookback,
            'target_horizon': predictor.target_horizon,
            'test_size': predictor.test_size,
            'sequence_length': predictor.sequence_length,
            'hidden_size': predictor.hidden_size,
            'num_layers': predictor.num_layers,
            'dropout': predictor.dropout,
            'learning_rate': predictor.learning_rate,
            'epochs': predictor.epochs,
            'batch_size': predictor.batch_size,
            'device': str(predictor.device)
        },
        'performance_metrics': {
            'mae': results.get('mae', 0),
            'rmse': results.get('rmse', 0),
            'bias': results.get('bias', 0),
            'bias_corrected_mae': results.get('bias_corrected_mae', 0),
            'directional_accuracy': results.get('directional_accuracy', 0),
            'mse': results.get('mse', 0)
        },
        'training_history': {
            'loss_history': history['loss'],
            'final_loss': history['loss'][-1] if history['loss'] else 0,
            'epochs_trained': len(history['loss'])
        },
        'feature_analysis': {
            'total_features': data['total_features'],
            'sequence_based': True,
            'temporal_modeling': True
        },
        'data_summary': {
            'training_samples': len(data['X_train']),
            'test_samples': len(data['X_test']),
            'train_test_distribution_shift': data['train_test_returns_diff']
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
    
    json_filename = os.path.join(results_dir, "lstm_predictor_results.json")
    
    with open(json_filename, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"📊 Results saved to JSON: {json_filename}")
    
    return results_data

if __name__ == "__main__":
    main()
