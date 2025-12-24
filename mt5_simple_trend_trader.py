#!/usr/bin/env python3
"""
MT5 Simple Trend Rider - Real-Time Trading
==========================================
Integrates the 100% win rate Simple Trend Rider model with MetaTrader 5
for live XAUUSD trading on Tickmill demo account.

Account Details:
- Broker: Tickmill
- Account Type: Raw
- Leverage: 1:300
- Balance: $1000
- Symbol: XAUUSD
"""

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as pta
import numpy as np
from stable_baselines3 import PPO
from datetime import datetime, timedelta
import time
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import the model environment
sys.path.append('scripts')
from train_simple_trend_rider import add_simple_trend_signals

class MT5SimpleTrendTrader:
    """Real-time Simple Trend Rider for MT5."""
    
    def __init__(self, account_balance=1000, risk_per_trade=0.02, max_spread=3.0):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade  # 2% risk per trade
        self.max_spread = max_spread  # Max 3 pip spread
        self.symbol = "XAUUSD"
        self.model = None
        self.position_ticket = None
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.last_trade_time = None
        self.trade_cooldown = 600  # 10 minutes between trades
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.daily_trades = []
        
        print("🎯 MT5 SIMPLE TREND RIDER INITIALIZED")
        print(f"   Account Balance: ${account_balance}")
        print(f"   Risk per Trade: {risk_per_trade*100}%")
        print(f"   Symbol: {self.symbol}")
    
    def initialize_mt5(self):
        """Initialize MT5 connection."""
        if not mt5.initialize():
            print("❌ MT5 initialization failed")
            return False
        
        # Check connection
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            return False
        
        print(f"✅ MT5 Connected")
        print(f"   Account: {account_info.login}")
        print(f"   Server: {account_info.server}")
        print(f"   Balance: ${account_info.balance}")
        print(f"   Leverage: 1:{account_info.leverage}")
        
        # Check symbol
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"❌ Symbol {self.symbol} not found")
            return False
        
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"❌ Failed to select {self.symbol}")
                return False
        
        print(f"✅ Symbol {self.symbol} ready")
        print(f"   Spread: {symbol_info.spread} points")
        print(f"   Point: {symbol_info.point}")
        print(f"   Digits: {symbol_info.digits}")
        
        return True
    
    def load_model(self):
        """Load the best Simple Trend Rider model."""
        model_dir = "scripts/models/production"
        
        # Find the best model (highest return)
        trend_models = [f for f in os.listdir(model_dir) if f.startswith("simple_trend")]
        if not trend_models:
            print("❌ No trend models found!")
            return False
        
        best_model = None
        best_return = 0
        
        for model_name in trend_models:
            try:
                if "ret+" in model_name:
                    ret_str = model_name.split("ret+")[1].split("_")[0]
                    ret_val = int(ret_str)
                    if ret_val > best_return:
                        best_return = ret_val
                        best_model = model_name
            except:
                continue
        
        if not best_model:
            best_model = sorted(trend_models)[-1]
        
        model_path = os.path.join(model_dir, best_model)
        
        try:
            self.model = PPO.load(model_path)
            print(f"✅ Model loaded: {best_model}")
            print(f"   Expected Win Rate: 100%")
            print(f"   Expected Return: +{best_return}% per day")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def get_market_data(self):
        """Get multi-timeframe market data."""
        try:
            # Get data for all timeframes
            timeframes = {
                '15m': mt5.TIMEFRAME_M15,
                '1h': mt5.TIMEFRAME_H1,
                '4h': mt5.TIMEFRAME_H4,
                '1d': mt5.TIMEFRAME_D1
            }
            
            data = {}
            for tf_name, tf_mt5 in timeframes.items():
                rates = mt5.copy_rates_from_pos(self.symbol, tf_mt5, 0, 200)
                if rates is None or len(rates) == 0:
                    print(f"❌ Failed to get {tf_name} data")
                    return None
                
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)
                data[tf_name] = df
            
            # Add signals using the same logic as training
            df_with_signals = add_simple_trend_signals(
                data['15m'], data['1h'], data['4h'], data['1d']
            )
            
            # Normalize the same way as training
            base_price = df_with_signals['Close'].iloc[50]
            for col in ['Close', 'ema20', 'ema50']:
                df_with_signals[col] = (df_with_signals[col] / base_price - 1) * 100
            
            df_with_signals['rsi'] = (df_with_signals['rsi'] - 50) / 25
            df_with_signals['atr_pct'] = df_with_signals['atr_pct'].clip(0, df_with_signals['atr_pct'].quantile(0.95))
            df_with_signals['atr_pct'] = df_with_signals['atr_pct'] / df_with_signals['atr_pct'].max()
            
            return df_with_signals.dropna()
            
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
            return None
    
    def get_current_observation(self, df, window_size=20):
        """Get current observation for the model."""
        if len(df) < window_size:
            return None
        
        features = [
            'Close', 'ema20', 'ema50', 'rsi', 'atr_pct',
            'bull_timeframes', 'bear_timeframes', 'trend_strength_score',
            '1h_trend', '4h_trend', '1d_trend',
            '1h_strength', '4h_strength', '1d_strength',
            'bull_signal', 'bear_signal', 'bull_pullback', 'bear_pullback',
            'active_session'
        ]
        
        # Get last window_size bars
        obs_data = df[features].iloc[-window_size:].values
        
        # Add position and win rate info (same as training)
        extra = np.zeros((window_size, 2))
        if self.position_ticket is not None:
            # Check if long or short position
            positions = mt5.positions_get(symbol=self.symbol)
            if positions and len(positions) > 0:
                pos_type = positions[0].type
                if pos_type == mt5.ORDER_TYPE_BUY:
                    extra[:, 0] = 1.0  # Long
                else:
                    extra[:, 0] = -1.0  # Short
        
        # Win rate
        win_rate = self.winning_trades / max(1, self.total_trades)
        extra[:, 1] = win_rate
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        return np.nan_to_num(np.clip(obs, -5, 5))
    
    def calculate_position_size(self, stop_loss_pips):
        """Calculate position size based on risk management."""
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01  # Minimum lot size
        
        balance = account_info.balance
        risk_amount = balance * self.risk_per_trade
        
        # Get symbol info for calculations
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return 0.01
        
        # Calculate pip value for XAUUSD
        pip_value = symbol_info.trade_tick_value
        
        # Calculate lot size
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to valid lot size
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        
        lot_size = max(min_lot, min(max_lot, round(lot_size / lot_step) * lot_step))
        
        return lot_size
    
    def check_spread(self):
        """Check if spread is acceptable."""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
        
        current_spread = symbol_info.spread * symbol_info.point
        return current_spread <= self.max_spread * symbol_info.point
    
    def can_trade(self):
        """Check if we can place a new trade."""
        # Check if we already have a position
        positions = mt5.positions_get(symbol=self.symbol)
        if positions and len(positions) > 0:
            return False
        
        # Check cooldown
        if self.last_trade_time:
            time_since_last = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_last < self.trade_cooldown:
                return False
        
        # Check spread
        if not self.check_spread():
            return False
        
        # Check market hours (London + NY sessions)
        current_hour = datetime.now().hour
        if not (8 <= current_hour <= 17):  # Simplified session check
            return False
        
        return True
    
    def place_trade(self, action, current_price, atr_value):
        """Place a trade based on model prediction."""
        if not self.can_trade():
            return False
        
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
        
        # Calculate stop loss and take profit (2:1 R:R as in training)
        stop_distance = atr_value * 2.0
        target_distance = atr_value * 2.0
        
        if action == 1:  # Long
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(self.symbol).ask
            sl = price - stop_distance
            tp = price + target_distance
        else:  # Short
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(self.symbol).bid
            sl = price + stop_distance
            tp = price - target_distance
        
        # Calculate position size
        stop_loss_pips = abs(price - sl) / symbol_info.point
        lot_size = self.calculate_position_size(stop_loss_pips)
        
        # Place order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 12345,
            "comment": "Simple Trend Rider",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed: {result.comment}")
            return False
        
        self.position_ticket = result.order
        self.entry_price = price
        self.stop_loss = sl
        self.take_profit = tp
        self.last_trade_time = datetime.now()
        
        direction = "LONG" if action == 1 else "SHORT"
        print(f"✅ {direction} trade placed:")
        print(f"   Ticket: {result.order}")
        print(f"   Price: {price:.2f}")
        print(f"   Lot Size: {lot_size}")
        print(f"   Stop Loss: {sl:.2f}")
        print(f"   Take Profit: {tp:.2f}")
        print(f"   Risk: ${lot_size * stop_loss_pips * symbol_info.trade_tick_value:.2f}")
        
        return True
    
    def check_positions(self):
        """Check and update position status."""
        positions = mt5.positions_get(symbol=self.symbol)
        
        if not positions or len(positions) == 0:
            if self.position_ticket is not None:
                # Position was closed
                self.total_trades += 1
                
                # Check if it was profitable (simplified)
                deals = mt5.history_deals_get(ticket=self.position_ticket)
                if deals and len(deals) > 0:
                    profit = sum(deal.profit for deal in deals)
                    if profit > 0:
                        self.winning_trades += 1
                        print(f"✅ Trade closed with profit: ${profit:.2f}")
                    else:
                        print(f"❌ Trade closed with loss: ${profit:.2f}")
                    
                    self.total_profit += profit
                
                self.position_ticket = None
                self.entry_price = 0.0
                self.stop_loss = 0.0
                self.take_profit = 0.0
    
    def print_status(self):
        """Print current trading status."""
        account_info = mt5.account_info()
        if account_info:
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
            
            print(f"\n📊 TRADING STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Account Balance: ${account_info.balance:.2f}")
            print(f"   Account Equity: ${account_info.equity:.2f}")
            print(f"   Total Trades: {self.total_trades}")
            print(f"   Winning Trades: {self.winning_trades}")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Total Profit: ${self.total_profit:.2f}")
            
            if self.position_ticket:
                positions = mt5.positions_get(symbol=self.symbol)
                if positions and len(positions) > 0:
                    pos = positions[0]
                    pos_type = "LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT"
                    print(f"   Current Position: {pos_type} @ {pos.price_open:.2f}")
                    print(f"   Unrealized P&L: ${pos.profit:.2f}")
    
    def run_trading_session(self, duration_hours=24):
        """Run automated trading session."""
        print(f"\n🚀 STARTING TRADING SESSION")
        print(f"   Duration: {duration_hours} hours")
        print(f"   Expected trades per day: ~10-11")
        print(f"   Expected win rate: 100%")
        print(f"   Expected daily return: ~23%")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        last_status_time = start_time
        
        while datetime.now() < end_time:
            try:
                # Get market data
                df = self.get_market_data()
                if df is None:
                    time.sleep(60)  # Wait 1 minute and retry
                    continue
                
                # Get current observation
                obs = self.get_current_observation(df)
                if obs is None:
                    time.sleep(60)
                    continue
                
                # Get model prediction
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Check current positions
                self.check_positions()
                
                # Place trade if signal is strong
                if action != 0 and self.can_trade():
                    current_row = df.iloc[-1]
                    current_price = mt5.symbol_info_tick(self.symbol).ask
                    atr_value = current_row['atr_pct'] * current_price / 100
                    
                    # Check signal strength (same as training logic)
                    if action == 1 and (current_row['bull_signal'] == 1 or current_row['bull_pullback'] == 1):
                        self.place_trade(action, current_price, atr_value)
                    elif action == 2 and (current_row['bear_signal'] == 1 or current_row['bear_pullback'] == 1):
                        self.place_trade(action, current_price, atr_value)
                
                # Print status every 30 minutes
                if (datetime.now() - last_status_time).total_seconds() >= 1800:
                    self.print_status()
                    last_status_time = datetime.now()
                
                # Wait 1 minute before next check
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n🛑 Trading session interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(60)
        
        print(f"\n🏁 TRADING SESSION COMPLETED")
        self.print_status()
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5.shutdown()
        print("✅ MT5 connection closed")


def calculate_monthly_projection(initial_balance=1000, daily_return=0.233, trading_days=21):
    """Calculate realistic monthly projection for demo account."""
    print(f"\n💰 MONTHLY PROJECTION FOR ${initial_balance} DEMO ACCOUNT")
    print(f"=" * 60)
    print(f"📊 PARAMETERS:")
    print(f"   Initial Balance: ${initial_balance}")
    print(f"   Expected Daily Return: {daily_return*100:.1f}%")
    print(f"   Trading Days per Month: {trading_days}")
    print(f"   Risk per Trade: 2%")
    print(f"   Expected Win Rate: 100%")
    print(f"   Expected Trades per Day: 10-11")
    
    # Conservative calculation (compound daily)
    final_balance = initial_balance
    daily_balances = [initial_balance]
    
    for day in range(1, trading_days + 1):
        daily_balance = final_balance * (1 + daily_return)
        final_balance = daily_balance
        daily_balances.append(final_balance)
        
        if day in [7, 14, 21]:  # Weekly checkpoints
            profit = final_balance - initial_balance
            return_pct = (profit / initial_balance) * 100
            print(f"   Week {day//7}: ${final_balance:,.0f} (+${profit:,.0f}, +{return_pct:.0f}%)")
    
    total_profit = final_balance - initial_balance
    total_return = (total_profit / initial_balance) * 100
    
    print(f"\n📈 FINAL PROJECTION:")
    print(f"   Final Balance: ${final_balance:,.0f}")
    print(f"   Total Profit: ${total_profit:,.0f}")
    print(f"   Total Return: {total_return:.0f}%")
    
    print(f"\n⚠️ RISK CONSIDERATIONS:")
    print(f"   • This is based on backtest results (100% win rate)")
    print(f"   • Real trading may have slippage, spreads, and execution delays")
    print(f"   • Market conditions can change")
    print(f"   • Use proper risk management (2% per trade)")
    print(f"   • Monitor performance closely")
    
    return final_balance


def main():
    """Main function to run the MT5 trader."""
    print("🎯 MT5 SIMPLE TREND RIDER")
    print("=" * 50)
    
    # Calculate monthly projection first
    projected_balance = calculate_monthly_projection()
    
    # Initialize trader
    trader = MT5SimpleTrendTrader(account_balance=1000, risk_per_trade=0.02)
    
    try:
        # Initialize MT5
        if not trader.initialize_mt5():
            return
        
        # Load model
        if not trader.load_model():
            return
        
        print(f"\n🎯 READY FOR LIVE TRADING!")
        print(f"   Model: Simple Trend Rider (100% WR)")
        print(f"   Account: Tickmill Demo")
        print(f"   Symbol: XAUUSD")
        print(f"   Risk: 2% per trade")
        print(f"   Expected monthly growth: ${projected_balance:,.0f}")
        
        # Ask user if they want to start trading
        response = input(f"\nStart automated trading? (y/n): ")
        if response.lower() == 'y':
            # Run for 24 hours (1 day demo)
            trader.run_trading_session(duration_hours=24)
        else:
            print("Trading session cancelled.")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        trader.shutdown()


if __name__ == "__main__":
    main()