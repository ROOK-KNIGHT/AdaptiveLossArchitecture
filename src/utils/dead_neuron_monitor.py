"""
Universal Dead Neuron Monitoring System for Neural Networks
Detects and tracks inactive neurons during training across all neural network models

Features:
- Real-time activation pattern monitoring
- Gradient flow analysis
- Layer-wise dead neuron statistics
- Training diagnostics and early warnings
- Universal compatibility with PyTorch models
- Integration with existing training loops

Compatible Models:
- Enhanced Adaptive Predictor (ReLU networks)
- LSTM Predictor (LSTM cells)
- Any PyTorch nn.Module with activation functions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import warnings
import json
import os
from datetime import datetime

class DeadNeuronMonitor:
    """
    Universal dead neuron monitoring system for PyTorch neural networks
    
    Tracks:
    - Neurons that consistently output zero (ReLU dead neurons)
    - LSTM cell states and gates that become inactive
    - Gradient flow patterns across all layers
    - Activation statistics per layer
    - Training health metrics
    - Learning rate acceleration patterns
    """
    
    def __init__(self, model: nn.Module, threshold: float = 0.01, 
                 monitoring_frequency: int = 10, verbose: bool = True,
                 save_results: bool = True, results_dir: str = "data/results/monitoring"):
        """
        Initialize universal dead neuron monitor
        
        Args:
            model: PyTorch model to monitor
            threshold: Minimum activation threshold (below this = potentially dead)
            monitoring_frequency: Check every N batches
            verbose: Print monitoring reports
            save_results: Save monitoring results to JSON
            results_dir: Directory to save monitoring results
        """
        self.model = model
        self.threshold = threshold
        self.monitoring_frequency = monitoring_frequency
        self.verbose = verbose
        self.save_results = save_results
        self.results_dir = results_dir
        
        # Create results directory
        if self.save_results:
            os.makedirs(self.results_dir, exist_ok=True)
        
        # Tracking dictionaries
        self.activation_stats = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        self.dead_neuron_counts = defaultdict(list)
        self.layer_health = defaultdict(dict)
        self.lstm_gate_stats = defaultdict(dict)
        
        # Batch and epoch counters
        self.batch_count = 0
        self.epoch_count = 0
        
        # Hook storage
        self.hooks = []
        self.activation_data = {}
        self.gradient_data = {}
        
        # Learning rate tracking
        self.learning_rates = []
        self.lr_acceleration = []
        
        # Model architecture analysis
        self.model_info = self._analyze_model_architecture()
        
        # Register hooks
        self._register_hooks()
        
        if self.verbose:
            print(f"🔍 Universal Dead Neuron Monitor initialized")
            print(f"   Model: {self.model_info['model_type']}")
            print(f"   Layers monitored: {len(self.model_info['monitorable_layers'])}")
            print(f"   Threshold: {threshold}")
            print(f"   Monitoring frequency: every {monitoring_frequency} batches")
            print(f"   Results directory: {results_dir}")
    
    def _analyze_model_architecture(self) -> Dict[str, Any]:
        """Analyze model architecture to determine monitoring strategy"""
        model_info = {
            'model_type': 'unknown',
            'total_layers': 0,
            'monitorable_layers': [],
            'activation_functions': [],
            'has_lstm': False,
            'has_relu': False,
            'has_dropout': False
        }
        
        # Detect model type and layers
        for name, module in self.model.named_modules():
            model_info['total_layers'] += 1
            
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
                model_info['monitorable_layers'].append(name)
                model_info['activation_functions'].append(type(module).__name__)
                model_info['has_relu'] = True
                
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                model_info['monitorable_layers'].append(name)
                model_info['activation_functions'].append(type(module).__name__)
                model_info['has_lstm'] = True
                
            elif isinstance(module, nn.Dropout):
                model_info['has_dropout'] = True
                
            elif isinstance(module, nn.Linear):
                model_info['monitorable_layers'].append(name)
        
        # Determine model type
        if 'StockPricePredictor' in str(type(self.model)):
            model_info['model_type'] = 'enhanced_adaptive_predictor'
        elif 'LSTMModel' in str(type(self.model)):
            model_info['model_type'] = 'lstm_predictor'
        elif model_info['has_lstm']:
            model_info['model_type'] = 'lstm_based'
        elif model_info['has_relu']:
            model_info['model_type'] = 'feedforward_relu'
        else:
            model_info['model_type'] = 'other_neural_network'
        
        return model_info
    
    def _register_hooks(self):
        """Register forward and backward hooks for monitorable layers"""
        
        for name, module in self.model.named_modules():
            if name in self.model_info['monitorable_layers']:
                
                # Forward hook for activation monitoring
                def forward_hook(layer_name):
                    def hook(module, input, output):
                        if self.batch_count % self.monitoring_frequency == 0:
                            self._analyze_activations(layer_name, output, module)
                    return hook
                
                # Backward hook for gradient monitoring
                def backward_hook(layer_name):
                    def hook(module, grad_input, grad_output):
                        if self.batch_count % self.monitoring_frequency == 0:
                            self._analyze_gradients(layer_name, grad_output, module)
                    return hook
                
                # Register hooks
                forward_handle = module.register_forward_hook(forward_hook(name))
                backward_handle = module.register_backward_hook(backward_hook(name))
                
                self.hooks.extend([forward_handle, backward_handle])
    
    def _analyze_activations(self, layer_name: str, output: torch.Tensor, module: nn.Module):
        """Analyze activation patterns for different layer types"""
        
        if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.ELU, nn.SELU)):
            self._analyze_relu_activations(layer_name, output)
            
        elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
            self._analyze_rnn_activations(layer_name, output, module)
            
        elif isinstance(module, nn.Linear):
            self._analyze_linear_activations(layer_name, output)
    
    def _analyze_relu_activations(self, layer_name: str, output: torch.Tensor):
        """Analyze ReLU activation patterns for dead neurons with error handling"""
        try:
            activations = output.detach().cpu().numpy()
            
            # Safety checks
            if activations.size == 0:
                if self.verbose:
                    print(f"⚠️  Warning: Empty activations for layer {layer_name}")
                return
            
            # Calculate dead neuron statistics with safe division
            zero_activations = (activations <= self.threshold).sum()
            total_activations = activations.size
            dead_ratio = zero_activations / total_activations if total_activations > 0 else 0.0
            
            # Calculate effective inactivity (combining dead neurons + dropout effect)
            mean_activation = float(np.mean(activations))
            std_activation = float(np.std(activations))
            
            # Per-neuron analysis (for fully connected layers)
            if len(activations.shape) == 2 and activations.shape[1] > 0:  # Batch x Features
                neuron_dead_ratios = []
                for neuron_idx in range(activations.shape[1]):
                    neuron_activations = activations[:, neuron_idx]
                    if len(neuron_activations) > 0:
                        neuron_dead_ratio = (neuron_activations <= self.threshold).mean()
                        neuron_dead_ratios.append(neuron_dead_ratio)
                    else:
                        neuron_dead_ratios.append(1.0)  # Consider empty as dead
                
                # Calculate dropout-compounded inactivity
                completely_dead = sum(1 for ratio in neuron_dead_ratios if ratio > 0.95)
                partially_dead = sum(1 for ratio in neuron_dead_ratios if 0.5 < ratio <= 0.95)
                
                # Estimate effective capacity loss (dead neurons + dropout)
                # This is a rough estimate of how much capacity is lost
                dropout_rate = self._estimate_dropout_rate(layer_name)
                effective_dead_ratio = dead_ratio + (1 - dead_ratio) * dropout_rate
                
                # Store statistics
                self.activation_stats[layer_name].append({
                    'batch': self.batch_count,
                    'epoch': self.epoch_count,
                    'dead_ratio': dead_ratio,
                    'effective_dead_ratio': effective_dead_ratio,
                    'dropout_rate': dropout_rate,
                    'mean_activation': mean_activation,
                    'std_activation': std_activation,
                    'neuron_dead_ratios': neuron_dead_ratios,
                    'completely_dead_neurons': completely_dead,
                    'partially_dead_neurons': partially_dead,
                    'total_neurons': len(neuron_dead_ratios),
                    'capacity_utilization': 1.0 - effective_dead_ratio
                })
            else:
                # For other shapes, store general statistics
                dropout_rate = self._estimate_dropout_rate(layer_name)
                effective_dead_ratio = dead_ratio + (1 - dead_ratio) * dropout_rate
                
                self.activation_stats[layer_name].append({
                    'batch': self.batch_count,
                    'epoch': self.epoch_count,
                    'dead_ratio': dead_ratio,
                    'effective_dead_ratio': effective_dead_ratio,
                    'dropout_rate': dropout_rate,
                    'mean_activation': mean_activation,
                    'std_activation': std_activation,
                    'shape': list(activations.shape),
                    'capacity_utilization': 1.0 - effective_dead_ratio
                })
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error analyzing ReLU activations for {layer_name}: {str(e)}")
    
    def _estimate_dropout_rate(self, layer_name: str) -> float:
        """Estimate dropout rate for a layer based on naming convention"""
        try:
            # Try to extract dropout rate from model architecture
            # This is a heuristic based on common patterns
            if 'network.2' in layer_name or 'dropout' in layer_name.lower():
                return 0.3  # First dropout layer
            elif 'network.5' in layer_name:
                return 0.2  # Second dropout layer  
            elif 'network.8' in layer_name:
                return 0.1  # Third dropout layer
            else:
                return 0.0  # No dropout
        except:
            return 0.0
    
    def _analyze_rnn_activations(self, layer_name: str, output: torch.Tensor, module: nn.Module):
        """Analyze RNN/LSTM activation patterns"""
        if isinstance(output, tuple):
            # LSTM returns (output, (hidden, cell))
            activations = output[0].detach().cpu().numpy()
            if len(output) > 1 and isinstance(output[1], tuple):
                hidden_state = output[1][0].detach().cpu().numpy()
                cell_state = output[1][1].detach().cpu().numpy() if len(output[1]) > 1 else None
            else:
                hidden_state = None
                cell_state = None
        else:
            activations = output.detach().cpu().numpy()
            hidden_state = None
            cell_state = None
        
        # Analyze main output
        zero_activations = (np.abs(activations) <= self.threshold).sum()
        total_activations = activations.size
        dead_ratio = zero_activations / total_activations if total_activations > 0 else 0
        
        stats = {
            'batch': self.batch_count,
            'epoch': self.epoch_count,
            'output_dead_ratio': dead_ratio,
            'output_mean': float(np.mean(activations)),
            'output_std': float(np.std(activations)),
            'output_shape': list(activations.shape)
        }
        
        # Analyze hidden states if available
        if hidden_state is not None:
            hidden_dead_ratio = (np.abs(hidden_state) <= self.threshold).sum() / hidden_state.size
            stats.update({
                'hidden_dead_ratio': hidden_dead_ratio,
                'hidden_mean': float(np.mean(hidden_state)),
                'hidden_std': float(np.std(hidden_state))
            })
        
        # Analyze cell states if available (LSTM)
        if cell_state is not None:
            cell_dead_ratio = (np.abs(cell_state) <= self.threshold).sum() / cell_state.size
            stats.update({
                'cell_dead_ratio': cell_dead_ratio,
                'cell_mean': float(np.mean(cell_state)),
                'cell_std': float(np.std(cell_state))
            })
        
        self.activation_stats[layer_name].append(stats)
    
    def _analyze_linear_activations(self, layer_name: str, output: torch.Tensor):
        """Analyze linear layer outputs (before activation)"""
        activations = output.detach().cpu().numpy()
        
        stats = {
            'batch': self.batch_count,
            'epoch': self.epoch_count,
            'mean_output': float(np.mean(activations)),
            'std_output': float(np.std(activations)),
            'min_output': float(np.min(activations)),
            'max_output': float(np.max(activations)),
            'shape': list(activations.shape)
        }
        
        self.activation_stats[layer_name].append(stats)
    
    def _analyze_gradients(self, layer_name: str, grad_output: Tuple[torch.Tensor], module: nn.Module):
        """Analyze gradient flow patterns with error handling"""
        try:
            if grad_output is None or grad_output[0] is None:
                if self.verbose:
                    print(f"⚠️  Warning: No gradients for layer {layer_name}")
                return
            
            gradients = grad_output[0].detach().cpu().numpy()
            
            # Safety checks
            if gradients.size == 0:
                if self.verbose:
                    print(f"⚠️  Warning: Empty gradients for layer {layer_name}")
                return
            
            # Calculate gradient statistics with safe operations
            abs_gradients = np.abs(gradients)
            zero_gradients = (abs_gradients <= 1e-8).sum()
            total_gradients = gradients.size
            zero_grad_ratio = zero_gradients / total_gradients if total_gradients > 0 else 0.0
            
            # Safe statistical calculations
            mean_grad = float(np.mean(abs_gradients)) if abs_gradients.size > 0 else 0.0
            std_grad = float(np.std(gradients)) if gradients.size > 0 else 0.0
            max_grad = float(np.max(abs_gradients)) if abs_gradients.size > 0 else 0.0
            
            stats = {
                'batch': self.batch_count,
                'epoch': self.epoch_count,
                'zero_grad_ratio': zero_grad_ratio,
                'mean_grad': mean_grad,
                'std_grad': std_grad,
                'max_grad': max_grad,
                'shape': list(gradients.shape),
                'gradient_health': 'healthy' if zero_grad_ratio < 0.2 else 'warning' if zero_grad_ratio < 0.5 else 'critical'
            }
            
            self.gradient_stats[layer_name].append(stats)
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error analyzing gradients for {layer_name}: {str(e)}")
    
    def update_batch_count(self):
        """Update batch counter - call this after each batch"""
        self.batch_count += 1
    
    def update_epoch_count(self):
        """Update epoch counter - call this after each epoch"""
        self.epoch_count += 1
        
        # Generate epoch report
        if self.verbose and self.epoch_count % 10 == 0:
            self._generate_epoch_report()
    
    def track_learning_rate(self, optimizer: torch.optim.Optimizer):
        """Track learning rate changes to detect acceleration"""
        current_lr = optimizer.param_groups[0]['lr']
        self.learning_rates.append(current_lr)
        
        # Calculate acceleration
        if len(self.learning_rates) >= 3:
            recent_lrs = self.learning_rates[-3:]
            acceleration = (recent_lrs[-1] - recent_lrs[-2]) - (recent_lrs[-2] - recent_lrs[-3])
            self.lr_acceleration.append(acceleration)
            
            # Warn about learning rate acceleration (potential dead neuron indicator)
            if acceleration > 0.001 and self.verbose:
                print(f"⚠️  Learning rate acceleration detected: {acceleration:.6f}")
                print(f"   This may indicate dead neurons in the network")
    
    def _generate_epoch_report(self):
        """Generate comprehensive epoch monitoring report"""
        print(f"\n🔍 Dead Neuron Monitor Report - Epoch {self.epoch_count}")
        print("=" * 60)
        
        for layer_name in self.model_info['monitorable_layers']:
            if layer_name in self.activation_stats and self.activation_stats[layer_name]:
                recent_stats = self.activation_stats[layer_name][-1]
                
                print(f"\n📊 Layer: {layer_name}")
                
                if 'dead_ratio' in recent_stats:
                    # ReLU layer analysis
                    dead_ratio = recent_stats['dead_ratio']
                    status = "🔴 CRITICAL" if dead_ratio > 0.5 else "🟡 WARNING" if dead_ratio > 0.2 else "🟢 HEALTHY"
                    
                    print(f"   Dead Neuron Ratio: {dead_ratio:.3f} {status}")
                    print(f"   Mean Activation: {recent_stats['mean_activation']:.6f}")
                    
                    if 'completely_dead_neurons' in recent_stats:
                        print(f"   Completely Dead Neurons: {recent_stats['completely_dead_neurons']}")
                        print(f"   Partially Dead Neurons: {recent_stats['partially_dead_neurons']}")
                
                elif 'output_dead_ratio' in recent_stats:
                    # LSTM layer analysis
                    output_dead = recent_stats['output_dead_ratio']
                    status = "🔴 CRITICAL" if output_dead > 0.3 else "🟡 WARNING" if output_dead > 0.1 else "🟢 HEALTHY"
                    
                    print(f"   Output Dead Ratio: {output_dead:.3f} {status}")
                    if 'hidden_dead_ratio' in recent_stats:
                        print(f"   Hidden State Dead Ratio: {recent_stats['hidden_dead_ratio']:.3f}")
                    if 'cell_dead_ratio' in recent_stats:
                        print(f"   Cell State Dead Ratio: {recent_stats['cell_dead_ratio']:.3f}")
                
                # Gradient analysis
                if layer_name in self.gradient_stats and self.gradient_stats[layer_name]:
                    grad_stats = self.gradient_stats[layer_name][-1]
                    zero_grad_ratio = grad_stats['zero_grad_ratio']
                    grad_status = "🔴 CRITICAL" if zero_grad_ratio > 0.5 else "🟡 WARNING" if zero_grad_ratio > 0.2 else "🟢 HEALTHY"
                    
                    print(f"   Zero Gradient Ratio: {zero_grad_ratio:.3f} {grad_status}")
                    print(f"   Mean Gradient Magnitude: {grad_stats['mean_grad']:.8f}")
        
        # Learning rate analysis
        if len(self.lr_acceleration) > 0:
            recent_acceleration = self.lr_acceleration[-1]
            if abs(recent_acceleration) > 0.0001:
                print(f"\n📈 Learning Rate Acceleration: {recent_acceleration:.6f}")
                if recent_acceleration > 0.001:
                    print("   ⚠️  High acceleration may indicate dead neurons")
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report with error handling"""
        report = {
            'model_info': self.model_info,
            'monitoring_summary': {
                'total_batches_monitored': self.batch_count,
                'total_epochs_monitored': self.epoch_count,
                'monitoring_frequency': self.monitoring_frequency,
                'threshold': self.threshold
            },
            'layer_analysis': {},
            'overall_health': 'unknown',
            'recommendations': [],
            'dropout_impact_analysis': {}
        }
        
        critical_layers = 0
        warning_layers = 0
        healthy_layers = 0
        
        try:
            for layer_name in self.model_info['monitorable_layers']:
                if layer_name in self.activation_stats and len(self.activation_stats[layer_name]) > 0:
                    layer_stats = self.activation_stats[layer_name]
                    
                    # Safety check for empty stats
                    if not layer_stats:
                        continue
                    
                    # Calculate average statistics with error handling
                    if 'dead_ratio' in layer_stats[-1]:
                        # ReLU layer analysis
                        try:
                            dead_ratios = [s.get('dead_ratio', 0.0) for s in layer_stats if 'dead_ratio' in s]
                            if not dead_ratios:
                                continue
                                
                            avg_dead_ratio = np.mean(dead_ratios) if dead_ratios else 0.0
                            max_dead_ratio = np.max(dead_ratios) if dead_ratios else 0.0
                            
                            # Get effective dead ratio (including dropout impact)
                            effective_ratios = [s.get('effective_dead_ratio', s.get('dead_ratio', 0.0)) for s in layer_stats if 'dead_ratio' in s or 'effective_dead_ratio' in s]
                            avg_effective_dead = np.mean(effective_ratios) if effective_ratios else avg_dead_ratio
                            
                            # Determine status based on effective dead ratio
                            if max_dead_ratio > 0.5:
                                status = 'critical'
                                critical_layers += 1
                            elif avg_dead_ratio > 0.2:
                                status = 'warning'
                                warning_layers += 1
                            else:
                                status = 'healthy'
                                healthy_layers += 1
                            
                            # Calculate trend safely
                            trend = 'stable'
                            if len(layer_stats) > 5:
                                try:
                                    recent_ratio = layer_stats[-1].get('dead_ratio', 0.0)
                                    old_ratio = layer_stats[-5].get('dead_ratio', 0.0)
                                    if recent_ratio < old_ratio - 0.05:  # Significant improvement
                                        trend = 'improving'
                                    elif recent_ratio > old_ratio + 0.05:  # Significant degradation
                                        trend = 'degrading'
                                except (IndexError, KeyError):
                                    trend = 'stable'
                            
                            # Get dropout impact
                            dropout_rate = layer_stats[-1].get('dropout_rate', 0.0)
                            capacity_util = layer_stats[-1].get('capacity_utilization', 1.0 - avg_dead_ratio)
                            
                            report['layer_analysis'][layer_name] = {
                                'type': 'relu',
                                'status': status,
                                'avg_dead_ratio': float(avg_dead_ratio),
                                'max_dead_ratio': float(max_dead_ratio),
                                'avg_effective_dead_ratio': float(avg_effective_dead),
                                'dropout_rate': float(dropout_rate),
                                'capacity_utilization': float(capacity_util),
                                'trend': trend,
                                'samples_analyzed': len(dead_ratios)
                            }
                            
                            # Store dropout impact analysis
                            if dropout_rate > 0:
                                report['dropout_impact_analysis'][layer_name] = {
                                    'dropout_rate': float(dropout_rate),
                                    'dead_neuron_ratio': float(avg_dead_ratio),
                                    'combined_inactivity': float(avg_effective_dead),
                                    'capacity_loss_pct': float((avg_effective_dead) * 100),
                                    'recommendation': 'Consider reducing dropout rate' if avg_effective_dead > 0.7 else 'Dropout rate acceptable'
                                }
                                
                        except Exception as e:
                            if self.verbose:
                                print(f"⚠️  Error analyzing ReLU layer {layer_name}: {str(e)}")
                            continue
                    
                    elif 'output_dead_ratio' in layer_stats[-1]:
                        # LSTM layer analysis
                        try:
                            output_dead_ratios = [s.get('output_dead_ratio', 0.0) for s in layer_stats if 'output_dead_ratio' in s]
                            if not output_dead_ratios:
                                continue
                                
                            avg_output_dead = np.mean(output_dead_ratios) if output_dead_ratios else 0.0
                            max_output_dead = np.max(output_dead_ratios) if output_dead_ratios else 0.0
                            
                            if max_output_dead > 0.3:
                                status = 'critical'
                                critical_layers += 1
                            elif avg_output_dead > 0.1:
                                status = 'warning'
                                warning_layers += 1
                            else:
                                status = 'healthy'
                                healthy_layers += 1
                            
                            report['layer_analysis'][layer_name] = {
                                'type': 'lstm',
                                'status': status,
                                'avg_output_dead_ratio': float(avg_output_dead),
                                'max_output_dead_ratio': float(max_output_dead),
                                'samples_analyzed': len(output_dead_ratios)
                            }
                            
                        except Exception as e:
                            if self.verbose:
                                print(f"⚠️  Error analyzing LSTM layer {layer_name}: {str(e)}")
                            continue
            
            # Overall health assessment with safe division
            total_layers = critical_layers + warning_layers + healthy_layers
            if total_layers == 0:
                report['overall_health'] = 'unknown'
                report['recommendations'].append("No layer statistics available for analysis")
            elif critical_layers > 0:
                report['overall_health'] = 'critical'
                report['recommendations'].append("Immediate action required: Consider using Leaky ReLU or reducing learning rate")
                if any(layer.get('dropout_rate', 0) > 0.2 for layer in report['layer_analysis'].values()):
                    report['recommendations'].append("High dropout rates may compound dead neuron problem - consider reducing dropout")
            elif warning_layers > healthy_layers:
                report['overall_health'] = 'warning'
                report['recommendations'].append("Monitor closely: Consider adjusting hyperparameters")
            else:
                report['overall_health'] = 'healthy'
                report['recommendations'].append("Model appears healthy: Continue current training")
            
            # Learning rate recommendations with safe operations
            if len(self.lr_acceleration) > 0:
                try:
                    recent_accelerations = self.lr_acceleration[-5:] if len(self.lr_acceleration) >= 5 else self.lr_acceleration
                    if recent_accelerations and np.mean(recent_accelerations) > 0.001:
                        report['recommendations'].append("Learning rate is accelerating: Potential dead neuron issue")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Error analyzing learning rate acceleration: {str(e)}")
            
            # Add summary statistics
            report['summary_statistics'] = {
                'total_layers_analyzed': total_layers,
                'critical_layers': critical_layers,
                'warning_layers': warning_layers,
                'healthy_layers': healthy_layers,
                'layers_with_dropout': len([l for l in report['layer_analysis'].values() if l.get('dropout_rate', 0) > 0]),
                'avg_capacity_utilization': np.mean([l.get('capacity_utilization', 1.0) for l in report['layer_analysis'].values()]) if report['layer_analysis'] else 1.0
            }
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error generating summary report: {str(e)}")
            report['overall_health'] = 'error'
            report['recommendations'].append(f"Error during analysis: {str(e)}")
        
        return report
    
    def save_monitoring_results(self, filename: str = None):
        """Save monitoring results to JSON file"""
        if not self.save_results:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dead_neuron_monitoring_{self.model_info['model_type']}_{timestamp}.json"
        
        filepath = os.path.join(self.results_dir, filename)
        
        # Prepare data for JSON serialization
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.model_info,
            'monitoring_config': {
                'threshold': self.threshold,
                'monitoring_frequency': self.monitoring_frequency,
                'total_batches': self.batch_count,
                'total_epochs': self.epoch_count
            },
            'activation_stats': dict(self.activation_stats),
            'gradient_stats': dict(self.gradient_stats),
            'learning_rates': self.learning_rates,
            'lr_acceleration': self.lr_acceleration,
            'summary_report': self.get_summary_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        if self.verbose:
            print(f"📊 Monitoring results saved to: {filepath}")
    
    def cleanup(self):
        """Remove hooks and cleanup resources"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        # Save final results
        if self.save_results:
            self.save_monitoring_results()
        
        if self.verbose:
            print("🧹 Dead Neuron Monitor cleanup completed")

# Utility functions for easy integration

def create_monitor_for_model(model: nn.Module, **kwargs) -> DeadNeuronMonitor:
    """
    Factory function to create appropriate monitor for any model
    
    Args:
        model: PyTorch model to monitor
        **kwargs: Additional arguments for DeadNeuronMonitor
    
    Returns:
        DeadNeuronMonitor instance configured for the model
    """
    return DeadNeuronMonitor(model, **kwargs)

def integrate_with_training_loop(model: nn.Module, optimizer: torch.optim.Optimizer, 
                                monitor_config: Dict[str, Any] = None) -> DeadNeuronMonitor:
    """
    Integrate dead neuron monitoring with existing training loop
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        monitor_config: Configuration for monitoring
    
    Returns:
        DeadNeuronMonitor instance ready for training integration
    
    Usage:
        monitor = integrate_with_training_loop(model, optimizer)
        
        for epoch in range(epochs):
            for batch in dataloader:
                # ... training code ...
                monitor.update_batch_count()
                monitor.track_learning_rate(optimizer)
            monitor.update_epoch_count()
        
        monitor.cleanup()
    """
    config = monitor_config or {}
    monitor = DeadNeuronMonitor(model, **config)
    
    print(f"✅ Dead neuron monitoring integrated with {monitor.model_info['model_type']}")
    print(f"   Call monitor.update_batch_count() after each batch")
    print(f"   Call monitor.update_epoch_count() after each epoch")
    print(f"   Call monitor.track_learning_rate(optimizer) to track LR changes")
    print(f"   Call monitor.cleanup() when training is complete")
    
    return monitor

# Example usage functions for different models

def monitor_enhanced_adaptive_predictor(model, optimizer, **kwargs):
    """Specific monitoring setup for Enhanced Adaptive Predictor"""
    config = {
        'threshold': 0.01,
        'monitoring_frequency': 5,
        'verbose': True,
        **kwargs
    }
    return integrate_with_training_loop(model, optimizer, config)

def monitor_lstm_predictor(model, optimizer, **kwargs):
    """Specific monitoring setup for LSTM Predictor"""
    config = {
        'threshold': 0.001,  # Lower threshold for LSTM
        'monitoring_frequency': 10,
        'verbose': True,
        **kwargs
    }
    return integrate_with_training_loop(model, optimizer, config)
