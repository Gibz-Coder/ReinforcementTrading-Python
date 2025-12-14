#!/usr/bin/env python3
"""
Model Testing Script
===================

Test trained models on out-of-sample data and generate performance reports.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import pandas as pd
from stable_baselines3 import PPO
from src.environments.enhanced_trading_env import EnhancedForexTradingEnv
from src.indicators.enhanced_indicators import load_and_preprocess_data_enhanced
import json
from datetime import datetime

def test_model(model_path, data_path, output_dir="results/reports"):
    """Test a trained model and generate performance report."""
    
    print(f"🧪 Testing Model: {model_path}")
    print(f"📊 Data: {data_path}")
    
    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Load and preprocess data
    try:
        df = load_and_preprocess_data_enhanced(data_path, normalize=True)
        print(f"✅ Data loaded: {len(df)} bars")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Create test environment
    test_env = EnhancedForexTradingEnv(
        df=df,
        window_size=30,
        initial_balance=10000.0,
        max_trades_per_day=5,
        spread_pips=1.5,
        slippage_pips=0.5
    )
    
    print("🔄 Running backtest...")
    
    # Run backtest
    obs, _ = test_env.reset()
    step_count = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        step_count += 1
        
        if step_count % 1000 == 0:
            print(f"  Step {step_count:,} - Equity: ${info.get('equity', 0):,.2f}")
        
        if terminated or truncated:
            break
    
    # Get results
    results = test_env.get_enhanced_statistics()
    
    # Print summary
    print("\n" + "="*60)
    print("📈 BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    print(f"Total Return: {results.get('return_pct', 0):.1f}%")
    print(f"Final Equity: ${results.get('final_equity', 0):,.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.1f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    
    # Save detailed report
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"backtest_report_{timestamp}.json")
    
    report_data = {
        'model_path': model_path,
        'data_path': data_path,
        'test_date': timestamp,
        'results': results,
        'trade_history': test_env.trade_history,
        'equity_curve': test_env.equity_curve
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved: {report_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Test Trained Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--data', type=str, 
                       default='data/raw/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
                       help='Path to test data CSV file')
    parser.add_argument('--output', type=str, default='results/reports',
                       help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        return
    
    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        return
    
    # Run test
    results = test_model(args.model, args.data, args.output)
    
    if results:
        # Performance evaluation
        winrate = results.get('win_rate', 0)
        return_pct = results.get('return_pct', 0)
        max_dd = results.get('max_drawdown', 0)
        
        print("\n" + "="*60)
        print("🎯 PERFORMANCE EVALUATION")
        print("="*60)
        
        if winrate >= 0.75:
            print("🟢 EXCELLENT win rate (75%+)")
        elif winrate >= 0.65:
            print("🟡 GOOD win rate (65-75%)")
        elif winrate >= 0.55:
            print("🟠 ACCEPTABLE win rate (55-65%)")
        else:
            print("🔴 POOR win rate (<55%)")
        
        if return_pct >= 20:
            print("🟢 EXCELLENT returns (20%+)")
        elif return_pct >= 10:
            print("🟡 GOOD returns (10-20%)")
        elif return_pct >= 0:
            print("🟠 POSITIVE returns (0-10%)")
        else:
            print("🔴 NEGATIVE returns")
        
        if max_dd <= 0.05:
            print("🟢 EXCELLENT drawdown control (<5%)")
        elif max_dd <= 0.10:
            print("🟡 GOOD drawdown control (5-10%)")
        elif max_dd <= 0.20:
            print("🟠 ACCEPTABLE drawdown (10-20%)")
        else:
            print("🔴 HIGH drawdown (>20%)")
        
        # Overall rating
        score = 0
        if winrate >= 0.65: score += 1
        if return_pct >= 10: score += 1
        if max_dd <= 0.15: score += 1
        
        print(f"\n🏆 Overall Rating: {score}/3")
        if score == 3:
            print("🌟 PRODUCTION READY - Excellent performance!")
        elif score == 2:
            print("✅ GOOD - Consider for production with monitoring")
        elif score == 1:
            print("⚠️  NEEDS IMPROVEMENT - More training recommended")
        else:
            print("❌ NOT READY - Significant improvements needed")

if __name__ == "__main__":
    main()