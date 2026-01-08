#!/usr/bin/env python3
"""
Backtest runner with progress callback for native app integration
"""

import sys
import os
sys.path.append('scripts')

from technical_goldengibz_backtest import TechnicalGoldenGibzBacktester
import time

class ProgressBacktester(TechnicalGoldenGibzBacktester):
    """Extended backtester with progress callback support"""
    
    def __init__(self, symbol="XAUUSD"):
        super().__init__(symbol)
        self.progress_callback = None
        self.ui_callback = None
    
    def set_progress_callback(self, callback):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def set_ui_callback(self, callback):
        """Set UI update callback function"""
        self.ui_callback = callback
    
    def calculate_stop_loss_distance(self, conditions, current_price):
        """Calculate stop loss distance based on symbol-specific logic"""
        if self.symbol.upper() == "EURUSD":
            # EURUSD: Use fixed pip-based stop loss (15 pips for tighter control)
            pip_value = 0.0001  # 1 pip for EURUSD
            stop_loss_pips = 15  # 15 pips stop loss (15:15 RR)
            return stop_loss_pips * pip_value
        elif self.symbol.upper() == "XAUUSD":
            # XAUUSD: Use ATR-based (original logic)
            return (conditions['atr_pct'] / 100) * current_price * 2.0
        else:
            # Default: Conservative ATR-based
            return (conditions['atr_pct'] / 100) * current_price * 1.5
    
    def run_backtest_with_progress(self, start_date=None, end_date=None):
        """Run backtest with progress updates"""
        print(f"\n🚀 STARTING TECHNICAL GOLDEN-GIBZ BACKTEST...")
        print(f"{'='*70}")
        
        if self.execution_tf not in self.data:
            print("❌ Data not available")
            return None
        
        # Prepare execution data
        df = self.data[self.execution_tf].copy()
        
        # Filter date range
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        if len(df) < 100:
            print("❌ Insufficient data")
            return None
        
        print(f"📊 Period: {df.index[0]} to {df.index[-1]}")
        print(f"📊 Total bars: {len(df):,}")
        print(f"📊 Fixed lot size: {self.fixed_lot_size} lots")
        print(f"📊 Risk-Reward: 1:{self.rr_ratio}")
        print(f"📊 Min confidence: {self.min_confidence}")
        
        # Initialize tracking
        balance = self.initial_balance
        equity = self.initial_balance
        positions = []
        trades = []
        equity_curve = []
        peak_equity = self.initial_balance
        
        # Fixed risk amount
        risk_amount = self.initial_balance * (self.risk_per_trade / 100)
        
        # Progress tracking
        total_bars = len(df)
        progress_interval = max(1, total_bars // 50)
        
        print(f"\n⏳ Processing {total_bars:,} bars with technical analysis...")
        print(f"💰 Fixed risk per trade: ${risk_amount:.2f}")
        
        signals_generated = 0
        signals_filtered = 0
        
        # Main loop with progress updates
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['Close']
            
            # Progress update with callback
            if i % progress_interval == 0:
                progress = (i / total_bars) * 100
                progress_msg = f"Progress: {progress:.1f}% - {current_time.strftime('%Y-%m-%d')} - Signals: {signals_generated}"
                print(f"   {progress_msg}")
                
                # Send to progress callback
                if self.progress_callback:
                    try:
                        self.progress_callback({
                            'progress': progress,
                            'current_date': current_time.strftime('%Y-%m-%d'),
                            'signals': signals_generated,
                            'total_bars': total_bars,
                            'current_bar': i,
                            'message': progress_msg
                        })
                    except Exception as e:
                        print(f"Progress callback error: {e}")
                
                # Send to UI callback
                if self.ui_callback:
                    try:
                        progress_text = f"⏳ Backtest Progress: {progress:.1f}%\n📅 Current Date: {current_time.strftime('%Y-%m-%d')}\n📊 Signals Generated: {signals_generated}\n💰 Current Balance: ${balance:,.2f}\n\nProcessing market data..."
                        self.ui_callback(progress_text)
                    except Exception as e:
                        print(f"UI callback error: {e}")
            
            # Update equity
            current_equity = balance
            for pos in positions:
                if pos['type'] == 'LONG':
                    # Symbol-specific P&L calculation for equity
                    if self.symbol.upper() == "EURUSD":
                        pnl = (current_price - pos['entry_price']) * pos['size'] * 100000
                    else:
                        pnl = (current_price - pos['entry_price']) * pos['size'] * 100
                else:
                    # Symbol-specific P&L calculation for equity
                    if self.symbol.upper() == "EURUSD":
                        pnl = (pos['entry_price'] - current_price) * pos['size'] * 100000
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['size'] * 100
                current_equity += pnl
            
            equity = current_equity
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            
            # Store equity curve (every 100 bars)
            if i % 100 == 0:
                equity_curve.append({
                    'timestamp': current_time,
                    'equity': equity,
                    'balance': balance,
                    'drawdown': drawdown,
                    'positions': len(positions)
                })
            
            # Generate signal (every 4 bars = 1 hour)
            if i % 4 == 0 and len(positions) < self.max_positions:
                signal = self.generate_technical_signal(current_time)
                signals_generated += 1
                
                if signal and signal['confidence'] >= self.min_confidence:
                    action = signal['action']
                    confidence = signal['confidence']
                    conditions = signal['conditions']
                    
                    # Calculate stop loss distance based on ATR - SYMBOL SPECIFIC
                    atr_distance = self.calculate_stop_loss_distance(conditions, current_price)
                    
                    # Use FIXED lot size for realistic results
                    position_size = self.fixed_lot_size
                    
                    # Execute trade with 1:1 RR
                    if action == 1:  # LONG
                        # Fix spread calculation for EURUSD
                        if self.symbol.upper() == "EURUSD":
                            spread_value = self.spread * 0.0001  # 1 pip = 0.0001 for EURUSD
                        else:
                            spread_value = self.spread * 0.01  # Original for other symbols
                        
                        entry_price = current_price + spread_value
                        stop_loss = entry_price - atr_distance
                        take_profit = entry_price + atr_distance  # 1:1 RR
                        
                        positions.append({
                            'type': 'LONG',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': confidence,
                            'signal_type': signal.get('signal_type', 'TECHNICAL_ONLY')
                        })
                        
                    elif action == 2:  # SHORT
                        # Fix spread calculation for EURUSD
                        if self.symbol.upper() == "EURUSD":
                            spread_value = self.spread * 0.0001  # 1 pip = 0.0001 for EURUSD
                        else:
                            spread_value = self.spread * 0.01  # Original for other symbols
                        
                        entry_price = current_price - spread_value
                        stop_loss = entry_price + atr_distance
                        take_profit = entry_price - atr_distance  # 1:1 RR
                        
                        positions.append({
                            'type': 'SHORT',
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'confidence': confidence,
                            'signal_type': signal.get('signal_type', 'TECHNICAL_ONLY')
                        })
                else:
                    signals_filtered += 1
            
            # Check exits (simplified for brevity - same logic as original)
            positions_to_remove = []
            for j, pos in enumerate(positions):
                exit_trade = False
                exit_reason = ""
                exit_price = current_price
                
                if pos['type'] == 'LONG':
                    if current_price <= pos['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                        exit_price = pos['stop_loss']
                    elif current_price >= pos['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                        exit_price = pos['take_profit']
                
                else:  # SHORT
                    if current_price >= pos['stop_loss']:
                        exit_trade = True
                        exit_reason = "Stop Loss"
                        exit_price = pos['stop_loss']
                    elif current_price <= pos['take_profit']:
                        exit_trade = True
                        exit_reason = "Take Profit"
                        exit_price = pos['take_profit']
                
                if exit_trade:
                    # Calculate P&L with symbol-specific multiplier
                    price_diff = exit_price - pos['entry_price'] if pos['type'] == 'LONG' else pos['entry_price'] - exit_price
                    
                    # Symbol-specific P&L calculation
                    if self.symbol.upper() == "EURUSD":
                        # EURUSD: 1 pip = $1 per 0.01 lot, so multiply by 100000
                        pnl = price_diff * pos['size'] * 100000
                    else:
                        # Other symbols: use original calculation
                        pnl = price_diff * pos['size'] * 100
                    
                    balance += pnl
                    
                    # Record trade
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': current_time,
                        'type': pos['type'],
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'size': pos['size'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'confidence': pos['confidence'],
                        'duration': (current_time - pos['entry_time']).total_seconds() / 3600,
                        'signal_type': pos.get('signal_type', 'TECHNICAL_ONLY')
                    })
                    
                    positions_to_remove.append(j)
            
            # Remove closed positions
            for j in reversed(positions_to_remove):
                positions.pop(j)
        
        # Close remaining positions (symbol-specific P&L calculation)
        final_price = df.iloc[-1]['Close']
        for pos in positions:
            # Symbol-specific P&L calculation
            if self.symbol.upper() == "EURUSD":
                if pos['type'] == 'LONG':
                    pnl = (final_price - pos['entry_price']) * pos['size'] * 100000
                else:
                    pnl = (pos['entry_price'] - final_price) * pos['size'] * 100000
            else:
                if pos['type'] == 'LONG':
                    pnl = (final_price - pos['entry_price']) * pos['size'] * 100
                else:
                    pnl = (pos['entry_price'] - final_price) * pos['size'] * 100
            
            balance += pnl
            
            trades.append({
                'entry_time': pos['entry_time'],
                'exit_time': df.index[-1],
                'type': pos['type'],
                'entry_price': pos['entry_price'],
                'exit_price': final_price,
                'size': pos['size'],
                'pnl': pnl,
                'exit_reason': "End of Test",
                'confidence': pos['confidence'],
                'duration': (df.index[-1] - pos['entry_time']).total_seconds() / 3600,
                'signal_type': pos.get('signal_type', 'TECHNICAL_ONLY')
            })
        
        # Store results with additional metrics
        self.results = {
            'trades': trades,
            'equity_curve': equity_curve,
            'final_balance': balance,
            'initial_balance': self.initial_balance,
            'total_return': ((balance - self.initial_balance) / self.initial_balance) * 100,
            'backtest_period': {
                'start': df.index[0],
                'end': df.index[-1],
                'duration_days': (df.index[-1] - df.index[0]).days
            },
            'signal_stats': {
                'signals_generated': signals_generated,
                'signals_filtered': signals_filtered,
                'filter_rate': (signals_filtered / signals_generated * 100) if signals_generated > 0 else 0
            }
        }
        
        # Calculate additional metrics
        if len(trades) > 0:
            win_rate = (sum(1 for t in trades if t['pnl'] > 0) / len(trades)) * 100
            
            # Calculate profit factor
            winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
            losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
            
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate max drawdown from equity curve
            if equity_curve:
                max_drawdown = max([eq.get('drawdown', 0) for eq in equity_curve])
            else:
                # Simple drawdown calculation
                peak = self.initial_balance
                max_dd = 0
                running_balance = self.initial_balance
                for trade in trades:
                    running_balance += trade['pnl']
                    if running_balance > peak:
                        peak = running_balance
                    drawdown = (peak - running_balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, drawdown)
                max_drawdown = max_dd
            
            # Simple Sharpe Ratio calculation
            if len(trades) > 1:
                returns = [t['pnl'] / self.initial_balance * 100 for t in trades]
                avg_return = sum(returns) / len(returns)
                variance = sum([(r - avg_return) ** 2 for r in returns]) / (len(returns) - 1)
                std_dev = variance ** 0.5 if variance > 0 else 0
                sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Recovery Factor
            recovery_factor = self.results['total_return'] / max_drawdown if max_drawdown > 0 else 0
            
            # Add calculated metrics to results
            self.results.update({
                'max_drawdown': max_drawdown,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'recovery_factor': recovery_factor
            })
        else:
            win_rate = 0
            self.results.update({
                'max_drawdown': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'recovery_factor': 0
            })
        
        print(f"\n✅ Technical Golden-Gibz backtest completed!")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {self.results['total_return']:+.2f}%")
        print(f"   Signal quality: {self.results['signal_stats']['filter_rate']:.1f}% filtered")
        
        return self.results

if __name__ == "__main__":
    # Test the progress backtester
    backtester = ProgressBacktester()
    backtester.initial_balance = 500.0
    backtester.min_confidence = 0.60
    
    def progress_callback(data):
        print(f"PROGRESS_UPDATE: {data}")
    
    backtester.set_progress_callback(progress_callback)
    
    try:
        data_loaded = backtester.load_and_prepare_data()
        if data_loaded:
            results = backtester.run_backtest_with_progress()
        else:
            print("No data loaded")
    except Exception as e:
        print(f"Error: {e}")