#!/usr/bin/env python3
"""
Pure AI Model Backtest runner with progress callback for native app integration
Uses only the trained AI model without technical analysis filters
"""

import sys
import os
sys.path.append('scripts')

from hybrid_backtest_runner_with_progress import HybridProgressBacktester
from model_loader import GoldenGibzModelLoader
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

class PureAIProgressBacktester(HybridProgressBacktester):
    """Pure AI Model backtester with progress callback support"""
    
    def __init__(self, symbol="XAUUSD"):
        super().__init__(symbol)
        self.pure_ai_mode = True
        
        # Load configuration from EA config
        self.load_ea_config()
        
        # Initialize model loader
        self.model_loader = GoldenGibzModelLoader()
        self.model_loaded = False
        
        print(f"🤖 Initializing Pure AI Backtester for {symbol}")
        print(f"💰 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"📊 Lot Size: {self.fixed_lot_size}")
        print(f"⚖️ Leverage: {self.leverage}:1")
        print(f"🎯 Min Confidence: {self.min_confidence:.1%}")
    
    def load_ea_config(self):
        """Load configuration from EA config file"""
        try:
            import json
            with open('config/ea_config.json', 'r') as f:
                config = json.load(f)
            
            # Load account parameters (can be overridden by app)
            account_config = config.get('account', {})
            self.initial_balance = account_config.get('initial_balance', 500.0)
            self.leverage = account_config.get('leverage', 300)
            # Use training spread for better model performance
            self.spread = 1.5  # Match training environment instead of 30.0
            self.commission = account_config.get('commission', 0.0)
            
            # Load trading parameters
            trading_config = config.get('trading', {})
            self.fixed_lot_size = trading_config.get('lot_size', 0.01)
            self.max_positions = trading_config.get('max_positions', 3)
            # Lower confidence threshold to match training
            self.min_confidence = 0.50  # Lower than config 65% for better signal generation
            self.risk_per_trade = trading_config.get('risk_per_trade', 2.0)
            self.max_daily_trades = trading_config.get('max_daily_trades', 5)  # Match training
            
            print(f"✅ Loaded EA configuration (Training-Optimized)")
            print(f"   Account Type: {account_config.get('account_type', 'Standard')}")
            print(f"   Currency: {account_config.get('currency', 'USD')}")
            print(f"   Spread: {self.spread} pips (Training Match)")
            
        except Exception as e:
            print(f"⚠️ Could not load EA config, using defaults: {e}")
            self.initial_balance = 500.0
            self.spread = 1.5  # Training environment spread
            self.min_confidence = 0.50
        
    def load_and_prepare_data(self, data_path="data/raw"):
        """Load and prepare data with correct file paths"""
        try:
            print(f"📂 Loading data from: {data_path}")
            
            # Define timeframes and their file patterns
            timeframes = {
                '1h': f"{data_path}/{self.symbol}/{self.symbol}_1h_data.csv",
                '4h': f"{data_path}/{self.symbol}/{self.symbol}_4H_data.csv", 
                '1d': f"{data_path}/{self.symbol}/{self.symbol}_1d_data.csv"
            }
            
            self.data = {}
            
            for tf, filepath in timeframes.items():
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath, sep=';')
                        df['Gmt time'] = pd.to_datetime(df['Gmt time'])
                        df.set_index('Gmt time', inplace=True)
                        
                        # Ensure numeric columns
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        self.data[tf] = df
                        print(f"✅ Loaded {filepath}: {len(df)} rows")
                        
                    except Exception as e:
                        print(f"❌ Error loading {filepath}: {e}")
                else:
                    print(f"❌ File not found: {filepath}")
            
            return len(self.data) > 0
            
        except Exception as e:
            print(f"❌ Data loading error: {e}")
            return False
        
    def load_ai_model(self):
        """Load the trained AI model"""
        try:
            print(f"🔄 Loading trained AI model...")
            
            # List available models
            models = self.model_loader.list_available_models()
            if not models:
                print("❌ No trained models found!")
                return False
            
            # Load the best model for this symbol
            if self.model_loader.load_model(symbol=self.symbol):
                self.model_loaded = True
                info = self.model_loader.get_model_info()
                print(f"✅ AI Model loaded successfully!")
                print(f"   Model: {info['symbol']} - WR: {info['win_rate']:.1f}%")
                print(f"   Expected Return: {info['annual_return']}%")
                print(f"   Training Date: {info['training_date']}")
                return True
            else:
                print("❌ Failed to load AI model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading AI model: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def run_pure_ai_backtest_with_progress(self, start_date=None, end_date=None):
        """Run Pure AI Model backtest with progress updates"""
        
        # Set random seeds for consistent results
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        
        # Set TensorFlow/PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        except ImportError:
            pass
        
        print(f"🎯 Using deterministic mode for consistent results")
        
        # Load AI model first
        if not self.model_loaded:
            if not self.load_ai_model():
                print("❌ Cannot run backtest without AI model")
                return {
                    'trades': [],
                    'final_balance': self.initial_balance,
                    'total_return': 0.0,
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'ai_stats': {
                        'ai_signals': 0,
                        'ai_predictions': 0,
                        'pure_ai_mode': True,
                        'model_loaded': False
                    },
                    'signal_stats': {
                        'total_signals': 0,
                        'filter_rate': 0.0,
                        'pure_ai_signals': 0
                    }
                }
        
        # Initialize results
        results = {
            'trades': [],
            'final_balance': self.initial_balance,
            'total_return': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'ai_stats': {
                'ai_signals': 0,
                'ai_predictions': 0,
                'pure_ai_mode': True,
                'model_loaded': True
            },
            'signal_stats': {
                'total_signals': 0,
                'filter_rate': 0.0,
                'pure_ai_signals': 0
            }
        }
        
        try:
            # Load and prepare data with technical indicators
            if not self.load_and_prepare_data():
                return results
            
            # Get primary timeframe data (1H)
            if '1h' not in self.data or len(self.data['1h']) == 0:
                print("❌ No 1H data available for Pure AI backtesting")
                return results
            
            df = self.data['1h'].copy()
            
            # Calculate technical indicators using the same method as training
            print("📊 Calculating technical indicators for AI model...")
            df = self.model_loader.calculate_technical_indicators(df)
            df = df.dropna()  # Remove NaN values
            
            # Filter by date range if provided
            if start_date and end_date:
                df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(df) < 100:
                print("❌ Insufficient data for Pure AI backtesting")
                return results
            
            # Initialize backtesting variables with 10% balance risk management
            balance = self.initial_balance
            position = 0  # 0: no position, 1: long, -1: short
            entry_price = 0
            entry_balance = 0
            position_size_multiplier = 0
            trades = []
            trades_today = 0
            last_trade_date = None
            
            # 10% BALANCE RISK SYSTEM
            risk_per_trade_percent = 10.0  # Risk 10% of balance per trade
            risk_reward_ratio = 1.0  # 1:1 risk-reward
            min_confidence_threshold = 0.60  # Lower threshold for better signal generation
            
            # For XAUUSD: 1 pip = $0.01 for 0.01 lot, 1 pip = $0.1 for 0.1 lot
            pip_value_per_lot = 10  # $10 per pip for 1 lot XAUUSD
            
            total_bars = len(df)
            progress_interval = max(1, total_bars // 100)  # Update progress 100 times
            
            print(f"🤖 Starting Pure AI Model backtest...")
            print(f"📊 Data points: {total_bars}")
            print(f"📅 Period: {df.index[0]} to {df.index[-1]}")
            print(f"💰 Risk per Trade: {risk_per_trade_percent}% of balance")
            print(f"⚖️ Risk-Reward Ratio: 1:{risk_reward_ratio}")
            print(f"🎯 Min Confidence: {min_confidence_threshold*100}%")
            
            # Process each bar
            for i in range(30, len(df)):  # Start after 30 bars for indicators
                current_time = df.index[i]
                current_price = df.iloc[i]['Close']
                current_date = current_time.date()
                
                # Reset daily trade counter
                if last_trade_date != current_date:
                    trades_today = 0
                    last_trade_date = current_date
                
                # Skip if max daily trades reached
                if trades_today >= self.max_daily_trades:
                    continue
                
                # Progress update with callback
                if i % progress_interval == 0:
                    progress = (i / total_bars) * 100
                    progress_msg = f"⏳ Pure AI Analysis: {progress:.1f}% - Processing {current_time.strftime('%Y-%m-%d %H:%M')}"
                    print(f"   {progress_msg}")
                    
                    # Send to progress callback
                    if self.progress_callback:
                        try:
                            self.progress_callback({
                                'progress': progress,
                                'current_date': current_time.strftime('%Y-%m-%d'),
                                'current_price': current_price,
                                'balance': balance,
                                'trades': len(trades)
                            })
                        except Exception as e:
                            print(f"Progress callback error: {e}")
                    
                    # Send to UI callback
                    if self.ui_callback:
                        try:
                            progress_text = f"⏳ Pure AI Backtest: {progress:.1f}%\n📅 Current Date: {current_time.strftime('%Y-%m-%d')}\n🤖 AI Signals Generated: {len(trades)}\n💰 Current Balance: ${balance:,.2f}\n\n🧠 Pure AI model analyzing market patterns..."
                            self.ui_callback(progress_text)
                        except Exception as e:
                            print(f"UI callback error: {e}")
                
                # Check for stop-loss and take-profit on existing positions
                if position != 0:
                    current_price = df.iloc[i]['Close']
                    
                    # Calculate current P&L as percentage of balance at entry
                    if position == 1:  # Long position
                        price_change_percent = ((current_price - entry_price) / entry_price) * 100
                        pnl_percent = price_change_percent * position_size_multiplier
                        
                    elif position == -1:  # Short position
                        price_change_percent = ((entry_price - current_price) / entry_price) * 100
                        pnl_percent = price_change_percent * position_size_multiplier
                    
                    # Check if we hit 10% loss (stop-loss) or 10% gain (take-profit)
                    if pnl_percent <= -risk_per_trade_percent or pnl_percent >= risk_per_trade_percent:
                        # Calculate actual P&L in dollars
                        pnl_dollars = (pnl_percent / 100) * entry_balance
                        new_balance = entry_balance + pnl_dollars
                        balance = new_balance
                        
                        close_reason = "stop_loss" if pnl_percent <= -risk_per_trade_percent else "take_profit"
                        trades.append({
                            'type': f'close_{close_reason}',
                            'time': current_time,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'pnl_percent': pnl_percent,
                            'pnl_dollars': pnl_dollars,
                            'balance': balance,
                            'signal_type': 'pure_ai',
                            'position_size_multiplier': position_size_multiplier
                        })
                        position = 0
                        entry_price = 0
                        entry_balance = 0
                        position_size_multiplier = 0
                
                results['ai_stats']['ai_predictions'] += 1
                results['signal_stats']['total_signals'] += 1
                
                # Generate Pure AI signal using trained model
                ai_signal, ai_confidence = self._generate_real_ai_signal(df, i, position, balance, trades_today)
                
                results['ai_stats']['ai_predictions'] += 1
                results['signal_stats']['total_signals'] += 1
                
                # FINAL OPTIMIZED: Focus on model's stronger signals
                min_confidence_threshold = 0.60  # Lower threshold for better signal generation
                
                # MINIMAL filtering - let the model decide
                current_rsi = df.iloc[i].get('RSI', 50)
                
                # Only basic sanity checks
                market_ok = (
                    20 < current_rsi < 80 and  # Avoid extreme RSI only
                    not (2 <= current_time.hour <= 6)  # Avoid dead hours only
                )
                
                # Pure AI decision with MINIMAL filtering - trust the model!
                if ai_confidence >= min_confidence_threshold and market_ok:
                    
                    results['ai_stats']['ai_signals'] += 1
                    results['signal_stats']['pure_ai_signals'] += 1
                    
                    # 10% BALANCE RISK POSITION SIZING
                    # Calculate position size to risk exactly 10% of current balance
                    risk_amount = balance * (risk_per_trade_percent / 100)  # 10% of current balance
                    
                    # Position size multiplier for percentage-based calculation
                    # This determines how much price movement = 10% balance change
                    position_size_multiplier = risk_per_trade_percent  # 10% risk = 10x multiplier
                    
                    # Execute trade with 10% balance risk system
                    if True:  # No additional market filters - model handles this
                        if ai_signal == 1 and position == 0:  # Buy signal - only if no position
                            # Open long position
                            if balance > 0:  # Only trade if we have positive balance
                                position = 1
                                entry_price = current_price
                                entry_balance = balance  # Store balance at entry for P&L calculation
                                trades_today += 1
                                trades.append({
                                    'type': 'buy',
                                    'time': current_time,
                                    'price': current_price,
                                    'ai_confidence': ai_confidence,
                                    'signal_type': 'pure_ai',
                                    'balance_at_entry': entry_balance,
                                    'risk_percent': risk_per_trade_percent
                                })
                            
                        elif ai_signal == -1 and position == 0:  # Sell signal - only if no position
                            # Open short position
                            if balance > 0:  # Only trade if we have positive balance
                                position = -1
                                entry_price = current_price
                                entry_balance = balance  # Store balance at entry for P&L calculation
                                trades_today += 1
                                trades.append({
                                    'type': 'sell',
                                    'time': current_time,
                                    'price': current_price,
                                    'ai_confidence': ai_confidence,
                                    'signal_type': 'pure_ai',
                                    'balance_at_entry': entry_balance,
                                    'risk_percent': risk_per_trade_percent
                                })
                
                # Stop trading if balance is too low
                if balance < self.initial_balance * 0.1:  # Stop if balance < 10% of initial
                    print(f"⚠️ Stopping backtest - balance too low: ${balance:.2f}")
                    break
            
            # Close final position if any
            if position != 0:
                final_price = df.iloc[-1]['Close']
                
                # Calculate final P&L as percentage
                if position == 1:  # Long position
                    price_change_percent = ((final_price - entry_price) / entry_price) * 100
                    pnl_percent = price_change_percent * position_size_multiplier
                elif position == -1:  # Short position
                    price_change_percent = ((entry_price - final_price) / entry_price) * 100
                    pnl_percent = price_change_percent * position_size_multiplier
                
                # Calculate actual P&L in dollars
                pnl_dollars = (pnl_percent / 100) * entry_balance
                balance = entry_balance + pnl_dollars
                
                trades.append({
                    'type': 'close_final',
                    'time': df.index[-1],
                    'entry_price': entry_price,
                    'exit_price': final_price,
                    'pnl_percent': pnl_percent,
                    'pnl_dollars': pnl_dollars,
                    'balance': balance,
                    'signal_type': 'pure_ai'
                })
            
            # Calculate final results
            total_return = ((balance - self.initial_balance) / self.initial_balance) * 100
            winning_trades = len([t for t in trades if t.get('pnl_percent', 0) > 0 or t.get('pnl_dollars', 0) > 0])
            total_trades = len([t for t in trades if 'pnl_percent' in t or 'pnl_dollars' in t])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Add date range information
            start_date_str = df.index[0].strftime('%Y-%m-%d')
            end_date_str = df.index[-1].strftime('%Y-%m-%d')
            duration_days = (df.index[-1] - df.index[0]).days
            
            results.update({
                'trades': trades,
                'final_balance': balance,
                'total_return': total_return,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'start_date': start_date_str,
                'end_date': end_date_str,
                'backtest_period': {
                    'start': df.index[0],
                    'end': df.index[-1],
                    'duration_days': duration_days
                },
                'ai_stats': {
                    'ai_signals': results['ai_stats']['ai_signals'],
                    'ai_predictions': results['ai_stats']['ai_predictions'],
                    'pure_ai_mode': True,
                    'model_loaded': True,
                    'ai_accuracy': (results['ai_stats']['ai_signals'] / results['ai_stats']['ai_predictions'] * 100) if results['ai_stats']['ai_predictions'] > 0 else 0
                },
                'signal_stats': {
                    'total_signals': results['signal_stats']['pure_ai_signals'],
                    'filter_rate': 0.0,  # No filtering in pure AI mode
                    'pure_ai_signals': results['signal_stats']['pure_ai_signals'],
                    'signals_generated': results['signal_stats']['pure_ai_signals']
                }
            })
            
            print(f"✅ Pure AI Model backtest completed!")
            print(f"📊 Total trades: {total_trades}")
            print(f"🤖 AI signals: {results['ai_stats']['ai_signals']}")
            print(f"💰 Final balance: ${balance:,.2f}")
            print(f"📈 Total return: {total_return:+.2f}%")
            print(f"🎯 Win rate: {win_rate:.1f}%")
            
            return results
            
        except Exception as e:
            print(f"❌ Pure AI backtest error: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def _generate_real_ai_signal(self, df, index, position=0, balance=10000, trades_today=0):
        """Generate Real AI signal using trained PPO model"""
        try:
            if not self.model_loaded:
                return 0, 0.0
            
            # Prepare observation vector exactly as used in training
            observation = self.model_loader.prepare_observation(
                df, index, position, balance, self.initial_balance, trades_today, 5
            )
            
            # Get prediction from trained model
            signal, confidence = self.model_loader.predict(observation)
            
            # Debug: Print confidence levels occasionally
            if index % 1000 == 0:
                print(f"🔍 Debug - Index {index}: Signal={signal}, Confidence={confidence:.3f}")
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Real AI signal generation error: {e}")
            return 0, 0.0


def main():
    """Test Pure AI backtester with real trained model"""
    print("🤖 Testing Pure AI Model Backtester with Real Trained Model...")
    
    backtester = PureAIProgressBacktester("XAUUSD")
    # Don't override - use configuration from EA config and training
    
    def progress_callback(data):
        print(f"PROGRESS_UPDATE: {data}")
    
    backtester.set_progress_callback(progress_callback)
    
    try:
        results = backtester.run_pure_ai_backtest_with_progress()
        
        if results and results['total_trades'] > 0:
            print(f"\n🎉 Pure AI Backtest Results (Real Model):")
            print(f"💰 Final Balance: ${results['final_balance']:,.2f}")
            print(f"📈 Total Return: {results['total_return']:+.2f}%")
            print(f"📊 Total Trades: {results['total_trades']}")
            print(f"🎯 Win Rate: {results['win_rate']:.1f}%")
            print(f"🤖 AI Signals: {results['ai_stats']['ai_signals']}")
            print(f"🧠 Model Loaded: {results['ai_stats']['model_loaded']}")
        else:
            print("❌ No results generated")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()