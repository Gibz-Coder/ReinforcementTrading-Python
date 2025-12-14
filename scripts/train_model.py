#!/usr/bin/env python3
"""
Main Training Script
===================

Unified entry point for all training systems with configuration support.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
from datetime import datetime

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Forex RL Training System')
    parser.add_argument('--system', type=str, choices=['ultra-aggressive', 'enhanced', 'custom'],
                       default='ultra-aggressive', help='Training system to use')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--data', type=str, 
                       default='data/raw/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
                       help='Training data path')
    parser.add_argument('--timesteps', type=int, help='Override timesteps from config')
    parser.add_argument('--model-name', type=str, help='Override model name')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.timesteps:
        config['training']['total_timesteps'] = args.timesteps
    
    model_name = args.model_name or f"{args.system}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("🚀 Forex RL Training System")
    print("=" * 50)
    print(f"System: {args.system}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data}")
    print(f"Timesteps: {config['training']['total_timesteps']:,}")
    print(f"Model: {model_name}")
    print()
    
    # Import and run appropriate training system
    if args.system == 'ultra-aggressive':
        from src.training.ultra_aggressive_winrate_system import train_ultra_aggressive_winrate_system
        
        model, results = train_ultra_aggressive_winrate_system(
            data_path=args.data,
            total_timesteps=config['training']['total_timesteps'],
            target_winrate=config['ultra_aggressive']['reward_system'].get('target_winrate', 0.8),
            model_name=model_name
        )
        
    elif args.system == 'enhanced':
        from src.training.enhanced_train_agent import train_enhanced_agent
        
        results = train_enhanced_agent(
            data_path=args.data,
            total_timesteps=config['training']['total_timesteps'],
            window_size=config['environment']['window_size'],
            use_curriculum=config['training']['use_curriculum'],
            use_walk_forward=config['training']['use_walk_forward'],
            learning_rate=config['model']['learning_rate'],
            n_steps=config['model']['n_steps'],
            batch_size=config['model']['batch_size'],
            n_epochs=config['model']['n_epochs'],
            gamma=config['model']['gamma'],
            model_name=model_name
        )
        
    else:  # custom
        print("Custom training not implemented yet. Use 'ultra-aggressive' or 'enhanced'.")
        return
    
    print("\n🎉 Training completed successfully!")
    return results

if __name__ == "__main__":
    main()