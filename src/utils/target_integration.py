"""
Target Integration Utility
Integrates unified target system into existing models

This utility helps integrate the UnifiedTargetManager into existing models
to ensure consistent target definitions across the entire project.

Usage:
    from src.utils.target_integration import integrate_unified_targets
    
    # For enhanced_adaptive_predictor.py
    integrate_unified_targets('enhanced_adaptive_predictor')
    
    # For comprehensive_comparison.py
    integrate_unified_targets('comprehensive_comparison')

Features:
    - Automatic target validation
    - Model configuration updates
    - Target consistency checks
    - Integration reporting
"""

import sys
import os
