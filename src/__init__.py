"""
Forex RL Trading System
======================

A comprehensive reinforcement learning system for forex trading using PPO.
"""

__version__ = "2.0.0"
__author__ = "Forex RL Team"

from .environments import EnhancedForexTradingEnv
from .indicators import load_and_preprocess_data_enhanced
from .training import train_enhanced_agent
from .rewards import HighWinrateRewardSystem
from .risk import AdaptiveRiskManager

__all__ = [
    'EnhancedForexTradingEnv',
    'load_and_preprocess_data_enhanced', 
    'train_enhanced_agent',
    'HighWinrateRewardSystem',
    'AdaptiveRiskManager'
]