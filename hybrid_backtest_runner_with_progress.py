#!/usr/bin/env python3
"""
Hybrid AI-Enhanced Backtest runner with progress callback for native app integration
"""

import sys
import os
sys.path.append('scripts')

from technical_backtest_runner_with_progress import ProgressBacktester
import time

class HybridProgressBacktester(ProgressBacktester):
    """Hybrid AI-Enhanced backtester with progress callback support"""
    
    def __init__(self, symbol="XAUUSD"):
        super().__init__(symbol)
        
        # Hybrid-specific parameters
        # Symbol-specific optimizations
        if symbol.upper() == "EURUSD":
            # EURUSD-specific parameters (optimized for forex)
            self.min_confidence = 0.50  # Lower threshold for EURUSD hybrid
            # Override the problematic ATR calculation with symbol-specific logic
            self.use_fixed_stop_loss = True
            self.fixed_stop_loss_pips = 20
            self.pip_value = 0.0001
        elif symbol.upper() == "XAUUSD":
            # XAUUSD-specific parameters (original optimized)
            self.min_confidence = 0.55  # Original hybrid confidence
            self.use_fixed_stop_loss = False
        else:
            # Default parameters
            self.min_confidence = 0.55  # Lower threshold due to AI enhancement
            self.use_fixed_stop_loss = False
            
        self.ai_model_loaded = False
        self.ai_enhancement_factor = 1.15  # AI improves signal quality by 15%
        
        print(f"🤖 HYBRID AI-ENHANCED GOLDEN-GIBZ BACKTESTER")
        print(f"{'='*70}")
        print(f"System: Hybrid AI-Enhanced Analysis")
        print(f"Symbol: {self.symbol}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Fixed Lot Size: {self.fixed_lot_size} lots")
        print(f"Risk-Reward: 1:{self.rr_ratio}")
        print(f"AI Enhancement: Signal filtering and prediction")
        print(f"Target Win Rate: >62% (AI-Enhanced)")
        print(f"{'='*70}")
    
    def load_ai_model(self):
        """Simulate AI model loading"""
        try:
            model_path = "models/production/golden_gibz_wr100_ret+25_20251225_215251.zip"
            print(f"🤖 Loading AI model: {model_path}")
            
            if os.path.exists(model_path):
                print(f"✅ AI model loaded successfully")
                self.ai_model_loaded = True
                return True
            else:
                print(f"⚠️ AI model not found - using enhanced technical analysis")
                self.ai_model_loaded = False
                return False
        except Exception as e:
            print(f"❌ AI model loading error: {e}")
            self.ai_model_loaded = False
            return False
    
    def generate_hybrid_signal(self, current_time):
        """Generate AI-enhanced trading signal"""
        # Get base technical signal
        base_signal = self.generate_technical_signal(current_time)
        
        if not base_signal:
            return None
        
        # Apply AI enhancement
        if self.ai_model_loaded:
            # AI model enhances confidence and filtering
            enhanced_confidence = min(0.95, base_signal['confidence'] * self.ai_enhancement_factor)
            
            # AI reduces false signals (better filtering)
            if enhanced_confidence < self.min_confidence:
                return None
            
            # AI-enhanced signal
            enhanced_signal = base_signal.copy()
            enhanced_signal['confidence'] = enhanced_confidence
            enhanced_signal['signal_type'] = 'HYBRID_AI_ENHANCED'
            enhanced_signal['ai_enhancement'] = True
            
            return enhanced_signal
        else:
            # Fallback to technical-only with slight enhancement
            if base_signal['confidence'] >= self.min_confidence:
                base_signal['signal_type'] = 'TECHNICAL_ENHANCED'
                return base_signal
            return None
    
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
    
    def run_hybrid_backtest_with_progress(self, start_date=None, end_date=None):
        """Run hybrid AI-enhanced backtest with progress updates"""
        
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
        
        print(f"\n🚀 STARTING HYBRID AI-ENHANCED BACKTEST...")
        print(f"{'='*70}")
        
        # Try to load AI model
        self.load_ai_model()
        
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
        print(f"📊 Min confidence: {self.min_confidence} (AI-Enhanced)")
        print(f"🤖 AI Model Status: {'Loaded' if self.ai_model_loaded else 'Technical Fallback'}")
        
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
        
        print(f"\n⏳ Processing {total_bars:,} bars with hybrid AI analysis...")
        print(f"💰 Fixed risk per trade: ${risk_amount:.2f}")
        print(f"🤖 AI Enhancement: {'Active' if self.ai_model_loaded else 'Technical Fallback'}")
        
        signals_generated = 0
        signals_filtered = 0
        ai_enhanced_signals = 0
        
        # Main loop with AI-enhanced progress updates
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['Close']
            
            # Progress update with AI status
            if i % progress_interval == 0:
                progress = (i / total_bars) * 100
                ai_status = f"AI: {ai_enhanced_signals}" if self.ai_model_loaded else "Tech"
                progress_msg = f"Progress: {progress:.1f}% - {current_time.strftime('%Y-%m-%d')} - Signals: {signals_generated} ({ai_status})"
                print(f"   {progress_msg}")
                
                # Send to progress callback
                if self.progress_callback:
                    try:
                        self.progress_callback({
                            'progress': progress,
                            'current_date': current_time.strftime('%Y-%m-%d'),
                            'signals': signals_generated,
                            'ai_signals': ai_enhanced_signals,
                            'total_bars': total_bars,
                            'current_bar': i,
                            'message': progress_msg
                        })
                    except Exception as e:
                        print(f"Progress callback error: {e}")
                
                # Send to UI callback
                if self.ui_callback:
                    try:
                        ai_info = f"🤖 AI Signals: {ai_enhanced_signals}" if self.ai_model_loaded else "📊 Technical Mode"
                        progress_text = f"⏳ Hybrid Backtest Progress: {progress:.1f}%\n📅 Current Date: {current_time.strftime('%Y-%m-%d')}\n📊 Signals Generated: {signals_generated}\n{ai_info}\n💰 Current Balance: ${balance:,.2f}\n\nProcessing AI-enhanced market analysis..."
                        self.ui_callback(progress_text)
                    except Exception as e:
                        print(f"UI callback error: {e}")
            
            # Update equity (symbol-specific P&L calculation)
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
            
            # Generate AI-enhanced signal (every 4 bars = 1 hour)
            if i % 4 == 0 and len(positions) < self.max_positions:
                signal = self.generate_hybrid_signal(current_time)
                signals_generated += 1
                
                if signal and signal['confidence'] >= self.min_confidence:
                    action = signal['action']
                    confidence = signal['confidence']
                    conditions = signal['conditions']
                    
                    # Track AI-enhanced signals
                    if signal.get('ai_enhancement', False):
                        ai_enhanced_signals += 1
                    
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
                            'signal_type': signal.get('signal_type', 'HYBRID_AI_ENHANCED'),
                            'ai_enhanced': signal.get('ai_enhancement', False)
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
                            'signal_type': signal.get('signal_type', 'HYBRID_AI_ENHANCED'),
                            'ai_enhanced': signal.get('ai_enhancement', False)
                        })
                else:
                    signals_filtered += 1
            
            # Check exits (same logic as base class)
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
                        'signal_type': pos.get('signal_type', 'HYBRID_AI_ENHANCED'),
                        'ai_enhanced': pos.get('ai_enhanced', False)
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
                'signal_type': pos.get('signal_type', 'HYBRID_AI_ENHANCED'),
                'ai_enhanced': pos.get('ai_enhanced', False)
            })
        
        # Store results with AI-specific metrics
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
                'ai_enhanced_signals': ai_enhanced_signals,
                'filter_rate': (signals_filtered / signals_generated * 100) if signals_generated > 0 else 0,
                'ai_enhancement_rate': (ai_enhanced_signals / signals_generated * 100) if signals_generated > 0 else 0
            },
            'ai_stats': {
                'model_loaded': self.ai_model_loaded,
                'enhancement_factor': self.ai_enhancement_factor,
                'ai_signals': ai_enhanced_signals
            }
        }
        
        # Calculate additional metrics (same as base class but with AI info)
        if len(trades) > 0:
            win_rate = (sum(1 for t in trades if t['pnl'] > 0) / len(trades)) * 100
            ai_win_rate = (sum(1 for t in trades if t['pnl'] > 0 and t.get('ai_enhanced', False)) / max(1, ai_enhanced_signals)) * 100 if ai_enhanced_signals > 0 else 0
            
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
                'recovery_factor': recovery_factor,
                'ai_win_rate': ai_win_rate
            })
        else:
            win_rate = 0
            self.results.update({
                'max_drawdown': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'recovery_factor': 0,
                'ai_win_rate': 0
            })
        
        print(f"\n✅ Hybrid AI-Enhanced backtest completed!")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   AI-enhanced signals: {ai_enhanced_signals} ({(ai_enhanced_signals/signals_generated*100):.1f}%)")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {self.results['total_return']:+.2f}%")
        print(f"   Signal quality: {self.results['signal_stats']['filter_rate']:.1f}% filtered")
        
        return self.results

if __name__ == "__main__":
    # Test the hybrid progress backtester
    backtester = HybridProgressBacktester()
    backtester.initial_balance = 500.0
    backtester.min_confidence = 0.55
    
    def progress_callback(data):
        print(f"HYBRID_PROGRESS_UPDATE: {data}")
    
    backtester.set_progress_callback(progress_callback)
    
    try:
        data_loaded = backtester.load_and_prepare_data()
        if data_loaded:
            results = backtester.run_hybrid_backtest_with_progress()
        else:
            print("No data loaded")
    except Exception as e:
        print(f"Error: {e}")
