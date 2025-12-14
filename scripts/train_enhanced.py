#!/usr/bin/env python3
"""
Enhanced Standard Training Script
================================

Main entry point for training the enhanced standard system.
Targets 65-75% win rate with balanced risk/reward approach.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.training.enhanced_train_agent import train_enhanced_agent
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Train Enhanced Standard System')
    parser.add_argument('--data', type=str, 
                       default='data/raw/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
                       help='Path to training data CSV file')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Observation window size')
    parser.add_argument('--use-curriculum', action='store_true', default=True,
                       help='Use curriculum learning')
    parser.add_argument('--use-walk-forward', action='store_true', default=False,
                       help='Use walk-forward analysis')
    parser.add_argument('--model-name', type=str, 
                       default=f'enhanced_standard_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                       help='Model name for saving')
    
    args = parser.parse_args()
    
    print("🚀 Starting Enhanced Standard Training")
    print(f"📊 Data: {args.data}")
    print(f"⏱️  Timesteps: {args.timesteps:,}")
    print(f"🪟 Window Size: {args.window_size}")
    print(f"📚 Curriculum Learning: {args.use_curriculum}")
    print(f"🔄 Walk-Forward: {args.use_walk_forward}")
    print(f"💾 Model Name: {args.model_name}")
    print()
    
    # Train the model
    results = train_enhanced_agent(
        data_path=args.data,
        total_timesteps=args.timesteps,
        window_size=args.window_size,
        use_curriculum=args.use_curriculum,
        use_walk_forward=args.use_walk_forward,
        model_name=args.model_name
    )
    
    # Determine where to save based on performance
    if results:
        result = results[0] if isinstance(results, list) else results
        winrate = result.get('win_rate', 0)
        return_pct = result.get('return_pct', 0)
        
        if winrate >= 0.65 and return_pct > 10:
            print(f"🎉 Good performance! Win Rate: {winrate*100:.1f}%, Return: {return_pct:.1f}%")
            print(f"💾 Model saved to production")
        else:
            print(f"📝 Model saved to experimental for further development")
    
    return results

if __name__ == "__main__":
    main()