#!/usr/bin/env python3
"""
Pure AI Golden-Gibz Signal Generator
===================================
Uses only the trained AI model for signal generation without technical analysis filters.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import warnings
import os
import json
from colorama import init, Fore, Back, Style
import threading
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import GoldenGibzModelLoader

warnings.filterwarnings('ignore')
init(autoreset=True)  # Initialize colorama

class PureAIGoldenGibzEA:
    """Pure AI Model Expert Advisor - Uses only trained AI model predictions"""
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.lot_size = 0.01
        self.max_positions = 1
        self.min_confidence = 0.50  # Lower threshold for pure AI
        self.signal_frequency = 60  # Signal check frequency in seconds
        self.current_position = 0
        self.last_signal_time = None
        self.signal_cooldown = 300  # 5 minutes between signals
        
        # Initialize AI model loader
        self.model_loader = GoldenGibzModelLoader()
        self.model_loaded = False
        
        print(f"{Fore.CYAN}🤖 PURE AI GOLDEN-GIBZ EXPERT ADVISOR")
        print(f"{'='*60}")
        print(f"{Fore.GREEN}System: Pure AI Model (Real Trained PPO)")
        print(f"{Fore.GREEN}Symbol: {self.symbol}")
        print(f"{Fore.GREEN}Lot Size: {self.lot_size}")
        print(f"{Fore.GREEN}Min Confidence: {self.min_confidence}")
        print(f"{Fore.GREEN}Max Positions: {self.max_positions}")
        print(f"{'='*60}")
    
    def load_ai_model(self):
        """Load the trained AI model"""
        try:
            print(f"🔄 Loading trained AI model...")
            
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
            return False
    
    def initialize(self):
        """Initialize MT5 connection and load AI model."""
        self.print_header()
        
        # Load AI model first
        if not self.load_ai_model():
            print("❌ Cannot initialize without AI model")
            return False
        
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account = mt5.account_info()
        print(f"✅ Connected to MT5 - Account: {account.login}")
        print(f"✅ Balance: ${account.balance:.2f}")
        print(f"✅ Server: {account.server}")
        
        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            print(f"❌ Failed to select symbol {self.symbol}: {mt5.last_error()}")
            return False
        
        print(f"✅ Symbol {self.symbol} selected")
        print(f"🤖 Real AI model ready for signal generation")
        return True
    
    def print_header(self):
        """Print EA header"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}🤖 PURE AI GOLDEN-GIBZ EXPERT ADVISOR")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.GREEN}⚡ Real-time Pure AI signal generation")
        print(f"{Fore.GREEN}🧠 Trained PPO model predictions")
        print(f"{Fore.GREEN}🎯 No technical analysis filters")
        print(f"{Fore.CYAN}{'='*60}")
    
    def get_market_data(self):
        """Get market data from MT5 for AI analysis."""
        print("📊 Getting market data for AI analysis...")
        
        try:
            # Get 1H data for AI model
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_H1, 0, 100)
            
            if rates is None or len(rates) == 0:
                print(f"❌ Failed to get market data: {mt5.last_error()}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Rename columns to match expected format
            df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'tick_volume': 'Volume'
            }, inplace=True)
            
            # Calculate technical indicators using the same method as training
            if self.model_loaded:
                df = self.model_loader.calculate_technical_indicators(df)
                df = df.dropna()  # Remove NaN values
            
            print(f"✅ Retrieved {len(df)} bars of {self.symbol} 1H data with indicators")
            return df
            
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            return None
    
    def generate_ai_signal(self, df):
        """Generate Pure AI signal using real trained model"""
        try:
            if not self.model_loaded:
                return 0, 0.0, "AI model not loaded"
            
            if len(df) < 30:
                return 0, 0.0, "Insufficient data for AI analysis"
            
            # Prepare observation vector for the trained model
            observation = self.model_loader.prepare_observation(
                df, len(df) - 1, self.current_position, 10000, 10000, 0, 5
            )
            
            # Get prediction from trained model
            signal, confidence = self.model_loader.predict(observation)
            
            # Generate reason based on signal
            if signal == 1:
                reason = f"AI Bullish: Trained model predicts upward movement (confidence: {confidence:.2f})"
            elif signal == -1:
                reason = f"AI Bearish: Trained model predicts downward movement (confidence: {confidence:.2f})"
            else:
                reason = f"AI Neutral: No clear directional signal (confidence: {confidence:.2f})"
            
            return signal, confidence, reason
            
        except Exception as e:
            print(f"❌ AI signal generation error: {e}")
            return 0, 0.0, f"Error: {str(e)}"
    
    def check_signal_cooldown(self):
        """Check if enough time has passed since last signal"""
        if self.last_signal_time is None:
            return True
        
        time_diff = time.time() - self.last_signal_time
        return time_diff >= self.signal_cooldown
    
    def execute_trade(self, signal, confidence, reason):
        """Execute trade based on AI signal"""
        try:
            if not self.check_signal_cooldown():
                return False
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                print("❌ Failed to get current price")
                return False
            
            current_price = tick.bid if signal == -1 else tick.ask
            
            # Check position limits
            positions = mt5.positions_get(symbol=self.symbol)
            if positions and len(positions) >= self.max_positions:
                print(f"⚠️ Maximum positions ({self.max_positions}) already open")
                return False
            
            # Prepare trade request
            if signal == 1:  # Buy
                trade_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = price - (price * 0.01)  # 1% stop loss
                tp = price + (price * 0.02)  # 2% take profit
                action_text = "BUY"
            else:  # Sell
                trade_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = price + (price * 0.01)  # 1% stop loss
                tp = price - (price * 0.02)  # 2% take profit
                action_text = "SELL"
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": trade_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"Pure AI {confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                print(f"❌ Trade failed: {result.comment}")
                return False
            
            self.last_signal_time = time.time()
            self.current_position = signal
            
            print(f"{Fore.GREEN}✅ {action_text} ORDER EXECUTED")
            print(f"{Fore.GREEN}💰 Price: {price:.5f}")
            print(f"{Fore.GREEN}🛡️ Stop Loss: {sl:.5f}")
            print(f"{Fore.GREEN}🎯 Take Profit: {tp:.5f}")
            print(f"{Fore.GREEN}🤖 AI Confidence: {confidence:.2f}")
            print(f"{Fore.GREEN}📝 Reason: {reason}")
            
            return True
            
        except Exception as e:
            print(f"❌ Trade execution error: {e}")
            return False
    
    def run(self):
        """Main EA loop"""
        print(f"\n🚀 Starting Pure AI Golden-Gibz EA...")
        print(f"⏰ Signal check interval: 60 seconds")
        print(f"🤖 Pure AI model active")
        print(f"{'='*60}")
        
        while True:
            try:
                current_time = datetime.now()
                print(f"\n⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')} - Pure AI Analysis")
                
                # Get market data
                df = self.get_market_data()
                if df is None:
                    print("⚠️ No market data - waiting...")
                    time.sleep(60)
                    continue
                
                # Generate AI signal
                signal, confidence, reason = self.generate_ai_signal(df)
                
                print(f"🤖 AI Signal: {signal} (Confidence: {confidence:.2f})")
                print(f"📝 Reason: {reason}")
                
                # Execute trade if confidence is high enough
                if abs(signal) > 0 and confidence >= self.min_confidence:
                    print(f"🎯 High confidence signal detected!")
                    success = self.execute_trade(signal, confidence, reason)
                    if success:
                        print(f"✅ Trade executed successfully")
                    else:
                        print(f"❌ Trade execution failed")
                else:
                    if confidence < self.min_confidence:
                        print(f"⚠️ Low confidence ({confidence:.2f} < {self.min_confidence}) - No trade")
                    else:
                        print(f"➡️ Neutral signal - No action")
                
                # Wait before next analysis
                print(f"⏳ Waiting 60 seconds for next analysis...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}⚠️ EA stopped by user")
                break
            except Exception as e:
                print(f"❌ EA error: {e}")
                print("⏳ Waiting 60 seconds before retry...")
                time.sleep(60)
        
        # Cleanup
        mt5.shutdown()
        print(f"{Fore.CYAN}👋 Pure AI Golden-Gibz EA stopped")


def main():
    """Main entry point"""
    ea = PureAIGoldenGibzEA("XAUUSD")
    
    if ea.initialize():
        ea.run()
    else:
        print("❌ Failed to initialize EA")


if __name__ == "__main__":
    main()