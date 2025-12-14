#!/usr/bin/env python3
"""
Ultra-Aggressive Win Rate Training Script
========================================

Main entry point for training the ultra-high win rate system.
Targets 80%+ win rate through ultra-conservative entry criteria.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.ultra_aggressive_winrate_system import train_ultra_aggressive_winrate_system
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train Ultra-Aggressive Win Rate System')
    parser.add_argument('--data', type=str, 
                       default='data/raw/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--target-winrate', type=float, default=0.8,
                       help='Target win rate (0.8 = 80%)')
    parser.add_argument('--model-name', type=str, 
                       default=f'ultra_aggressive_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Model name for saving')
    
    args = parser.parse_args()
    
    print("🚀 Starting Ultra-Aggressive Win Rate Training")
    print(f"📊 Data: {args.data}")
    print(f"⏱️  Timesteps: {args.timesteps:,}")
    print(f"🎯 Target Win Rate: {args.target_winrate*100:.1f}%")
    print(f"💾 Model Name: {args.model_name}")
    print()
    
    # Train the model
    model, results = train_ultra_aggressive_winrate_system(
        data_path=args.data,
        total_timesteps=args.timesteps,
        target_winrate=args.target_winrate,
        model_name=args.model_name
    )
    
    # Save to production if good results
    winrate = results.get('win_rate', 0)
    if winrate >= args.target_winrate:
        production_path = f"models/production/{args.model_name}_winrate_{winrate*100:.1f}pct.zip"
        model.save(production_path)
        print(f"🎉 Model saved to production: {production_path}")
    else:
        experimental_path = f"models/experimental/{args.model_name}.zip"
        model.save(experimental_path)
        print(f"📝 Model saved to experimental: {experimental_path}")
    
    return model, results

if __name__ == "__main__":
    main()