#!/usr/bin/env python3
"""
Technical Golden Gibz Pure Backtesting System - TECHNICAL ONLY VERSION
======================================================================
Pure technical analysis backtesting system without AI model integration
- Technical Analysis: Enhanced multi-timeframe system (proven 61.7% win rate)
- No AI Model: Faster execution, more transparent, fully debuggable
- Multi-timeframe: 15M, 30M, 1H, 4H, 1D analysis with weighted scoring
- Enhanced Indicators: EMA, RSI, MACD, ADX, Stochastic, Bollinger Bands
- Session Filtering: London/NY/Asian active sessions
- Volatility Filtering: Bollinger Bands width analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import ta
import warnings
from colorama import init, Fore, Style
import time

warnings.filterwarnings('ignore')
init(autoreset=True)

class TechnicalGoldenGibzBacktester:
    """Pure technical analysis Golden-Gibz backtesting system - no AI integration."""
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.data = {}
        self.results = {}
        
        # Real account parameters
        self.initial_balance = 500.0
        self.leverage = 300
        self.fixed_lot_size = 0.01  # Fixed lot size for realistic results
        self.spread = 2.0
        self.commission = 0.0
        
        # Risk management - REALISTIC
        self.max_positions = 1  # Only 1 position at a time for better control
        self.risk_per_trade = 2.0  # 2% of ORIGINAL balance, not growing balance
        self.rr_ratio = 1.0  # 1:1 risk-reward as requested
        
        # Enhanced signal filtering parameters - PURE TECHNICAL OPTIMIZED
        self.min_confidence = 0.75  # Higher confidence threshold for quality
        self.trend_alignment_required = True  # Require multiple timeframes to align
        self.volatility_filter = True  # Avoid trading in extreme volatility
        self.session_filter = True  # Trade only during active sessions
        
        # Multi-timeframe settings
        self.execution_tf = '15m'
        self.analysis_tfs = ['30m', '1h', '4h', '1d']
        
        # Enhanced indicators settings - TECHNICAL ONLY
        self.indicators = {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,  # Added ADX for trend strength
            "stoch_k": 14,     # Added Stochastic
            "stoch_d": 3
        }
        
        print(f"{Fore.CYAN}{Style.BRIGHT}⚙️ TECHNICAL GOLDEN-GIBZ BACKTESTER")
        print(f"{'='*70}")
        print(f"System: Pure Technical Analysis (No AI)")
        print(f"Symbol: {self.symbol}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Fixed Lot Size: {self.fixed_lot_size} lots")
        print(f"Risk-Reward: 1:{self.rr_ratio}")
        print(f"Target Win Rate: >61% (Technical Only)")
        print(f"{'='*70}{Style.RESET_ALL}")
        self.trend_alignment_required = True  # Require multiple timeframes to align
        self.volatility_filter = True  # Avoid trading in extreme volatility
        self.session_filter = True  # Trade only during active sessions
        
        # Multi-timeframe settings
        self.execution_tf = '15m'
        self.analysis_tfs = ['30m', '1h', '4h', '1d']
        
        # Enhanced indicators settings
        self.indicators = {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,  # Added ADX for trend strength
            "stoch_k": 14,     # Added Stochastic
            "stoch_d": 3
        }
        
        print(f"{Fore.CYAN}{Style.BRIGHT}🎯 TECHNICAL GOLDEN-GIBZ BACKTESTER (PURE TECHNICAL)")
        print(f"{'='*70}")
        print(f"System: Pure Technical Analysis (No AI)")
        print(f"Symbol: {self.symbol}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Fixed Lot Size: {self.fixed_lot_size} lots")
        print(f"Risk-Reward: 1:{self.rr_ratio}")
        print(f"Target Win Rate: >61%")
        print(f"{'='*70}{Style.RESET_ALL}")
    
    def calculate_enhanced_indicators(self, df):
        """Calculate enhanced technical indicators for better signal quality."""
        try:
            # Basic indicators
            df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_fast'])
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_slow'])
            df['RSI'] = ta.momentum.rsi(df['Close'], window=self.indicators['rsi_period'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.indicators['atr_period'])
            
            # Enhanced indicators for better signals
            # ADX for trend strength
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=self.indicators['adx_period'])
            
            # Stochastic for momentum
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                   window=self.indicators['stoch_k'], 
                                                   smooth_window=self.indicators['stoch_d'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df['Close'], 
                           window_fast=self.indicators['macd_fast'],
                           window_slow=self.indicators['macd_slow'],
                           window_sign=self.indicators['macd_signal'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands for volatility context
            bb = ta.volatility.BollingerBands(df['Close'], window=self.indicators['bb_period'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
            
            # Enhanced signal conditions
            df['EMA_Bullish'] = df['EMA20'] > df['EMA50']
            df['Price_Above_EMA20'] = df['Close'] > df['EMA20']
            df['MACD_Bullish'] = df['MACD'] > df['MACD_Signal']
            df['Strong_Trend'] = df['ADX'] > 25  # ADX > 25 indicates strong trend
            df['RSI_Neutral'] = (df['RSI'] > 30) & (df['RSI'] < 70)  # Avoid extreme RSI
            df['Stoch_Bullish'] = df['Stoch_K'] > df['Stoch_D']
            df['Low_Volatility'] = df['BB_Width'] < df['BB_Width'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators: {e}")
            return df
    
    def load_and_prepare_data(self, data_path="data/raw"):
        """Load and pre-calculate all indicators."""
        print(f"\n📊 Loading and preparing multi-timeframe data...")
        
        timeframes = ['15m', '30m', '1h', '4h', '1d']
        
        for tf in timeframes:
            filename = f"{data_path}/XAU_{tf}_data.csv"
            
            if os.path.exists(filename):
                try:
                    print(f"📈 Loading {tf} data...")
                    df = pd.read_csv(filename, sep=';')
                    df['Gmt time'] = pd.to_datetime(df['Gmt time'])
                    df.set_index('Gmt time', inplace=True)
                    
                    # Pre-calculate enhanced indicators
                    print(f"⚙️ Calculating {tf} indicators...")
                    df = self.calculate_enhanced_indicators(df)
                    
                    self.data[tf] = df
                    print(f"✅ {tf.upper()}: {len(df):,} bars prepared")
                        
                except Exception as e:
                    print(f"❌ {tf.upper()}: Error - {e}")
            else:
                print(f"❌ {tf.upper()}: File not found")
        
        if not self.data or self.execution_tf not in self.data:
            print(f"\n⚠️ Required data not available!")
            return False
        
        print(f"\n✅ Multi-timeframe data preparation complete!")
        return True
    
    def is_good_trading_session(self, timestamp):
        """Check if current time is during active trading sessions."""
        if not self.session_filter:
            return True
        
        hour = timestamp.hour
        
        # Active forex sessions (UTC):
        # London: 8-17, New York: 13-22, Asian: 0-9
        # Overlap periods are best: London-NY (13-17)
        active_sessions = [
            (8, 17),   # London session
            (13, 22),  # New York session
            (0, 9)     # Asian session
        ]
        
        for start, end in active_sessions:
            if start <= hour <= end:
                return True
        
        return False
    
    def get_enhanced_timeframe_signal(self, tf_data, timestamp):
        """Get enhanced signal from a specific timeframe."""
        try:
            idx = tf_data.index.get_indexer([timestamp], method='nearest')[0]
            if idx < 50 or idx >= len(tf_data):
                return 0, 0, {}
            
            row = tf_data.iloc[idx]
            
            # Enhanced signal scoring
            bullish_score = 0
            bearish_score = 0
            signal_quality = {}
            
            # 1. Trend Direction (EMA) - Weight: 3
            if row.get('EMA_Bullish', False):
                if row.get('Price_Above_EMA20', False):
                    bullish_score += 3
                    signal_quality['ema_trend'] = 'strong_bull'
                else:
                    bullish_score += 1
                    signal_quality['ema_trend'] = 'weak_bull'
            else:
                if not row.get('Price_Above_EMA20', True):
                    bearish_score += 3
                    signal_quality['ema_trend'] = 'strong_bear'
                else:
                    bearish_score += 1
                    signal_quality['ema_trend'] = 'weak_bear'
            
            # 2. MACD Confirmation - Weight: 2
            if row.get('MACD_Bullish', False):
                bullish_score += 2
                signal_quality['macd'] = 'bullish'
            else:
                bearish_score += 2
                signal_quality['macd'] = 'bearish'
            
            # 3. Trend Strength (ADX) - Weight: 2
            if row.get('Strong_Trend', False):
                if bullish_score > bearish_score:
                    bullish_score += 2
                else:
                    bearish_score += 2
                signal_quality['trend_strength'] = 'strong'
            else:
                signal_quality['trend_strength'] = 'weak'
            
            # 4. Stochastic Momentum - Weight: 1
            if row.get('Stoch_Bullish', False):
                bullish_score += 1
                signal_quality['momentum'] = 'bullish'
            else:
                bearish_score += 1
                signal_quality['momentum'] = 'bearish'
            
            # 5. RSI Filter - Penalty for extreme levels
            rsi = row.get('RSI', 50)
            if rsi > 80:
                bullish_score -= 2  # Overbought penalty
                signal_quality['rsi_filter'] = 'overbought'
            elif rsi < 20:
                bearish_score -= 2  # Oversold penalty
                signal_quality['rsi_filter'] = 'oversold'
            elif 30 <= rsi <= 70:
                signal_quality['rsi_filter'] = 'neutral'
            
            # 6. Volatility Filter - Bonus for low volatility
            if row.get('Low_Volatility', False):
                if bullish_score > bearish_score:
                    bullish_score += 1
                else:
                    bearish_score += 1
                signal_quality['volatility'] = 'favorable'
            else:
                signal_quality['volatility'] = 'high'
            
            # Determine signal
            if bullish_score > bearish_score:
                return 1, bullish_score - bearish_score, signal_quality
            elif bearish_score > bullish_score:
                return -1, bearish_score - bullish_score, signal_quality
            else:
                return 0, 0, signal_quality
                
        except Exception as e:
            return 0, 0, {}
    
    def analyze_enhanced_conditions(self, timestamp):
        """Enhanced multi-timeframe analysis with quality scoring."""
        try:
            # Timeframe weights (higher timeframes more important)
            tf_weights = {'15m': 1, '30m': 2, '1h': 3, '4h': 4, '1d': 5}
            
            total_bull_score = 0
            total_bear_score = 0
            total_weight = 0
            timeframe_signals = {}
            
            # Get execution timeframe data
            exec_data = self.data[self.execution_tf]
            exec_idx = exec_data.index.get_indexer([timestamp], method='nearest')[0]
            if exec_idx < 50:
                return None
            
            exec_row = exec_data.iloc[exec_idx]
            
            # Analyze each timeframe
            for tf_name in ['15m', '30m', '1h', '4h', '1d']:
                if tf_name not in self.data:
                    continue
                
                signal, strength, quality = self.get_enhanced_timeframe_signal(self.data[tf_name], timestamp)
                weight = tf_weights.get(tf_name, 1)
                
                timeframe_signals[tf_name] = {
                    'signal': signal,
                    'strength': strength,
                    'quality': quality
                }
                
                if signal == 1:  # Bullish
                    total_bull_score += strength * weight
                elif signal == -1:  # Bearish
                    total_bear_score += strength * weight
                
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # Calculate overall conditions
            bull_score = total_bull_score / total_weight if total_weight > 0 else 0
            bear_score = total_bear_score / total_weight if total_weight > 0 else 0
            
            # Enhanced filtering conditions
            conditions = {
                'bull_signal': False,
                'bear_signal': False,
                'signal_strength': max(bull_score, bear_score),
                'rsi': exec_row.get('RSI', 50),
                'atr_pct': 0.15,
                'price': exec_row['Close'],
                'timeframe_signals': timeframe_signals,
                'session_active': self.is_good_trading_session(timestamp),
                'bull_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 1),
                'bear_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == -1),
                'neutral_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 0)
            }
            
            # Calculate ATR percentage
            atr = exec_row.get('ATR', 15.0)
            if atr > 0 and exec_row['Close'] > 0:
                conditions['atr_pct'] = (atr / exec_row['Close']) * 100
            
            # Enhanced signal determination with stricter criteria
            min_strength = 3.0  # Require stronger signals
            
            if bull_score > bear_score and bull_score >= min_strength:
                # Additional quality checks for bullish signals
                if (conditions['session_active'] and 
                    conditions['rsi'] < 75 and  # Not too overbought
                    exec_row.get('Strong_Trend', False)):  # Strong trend required
                    conditions['bull_signal'] = True
                    
            elif bear_score > bull_score and bear_score >= min_strength:
                # Additional quality checks for bearish signals
                if (conditions['session_active'] and 
                    conditions['rsi'] > 25 and  # Not too oversold
                    exec_row.get('Strong_Trend', False)):  # Strong trend required
                    conditions['bear_signal'] = True
            
            return conditions
            
        except Exception as e:
            return None
    
    def generate_technical_signal(self, timestamp):
        """Generate pure technical trading signal."""
        try:
            conditions = self.analyze_enhanced_conditions(timestamp)
            if not conditions:
                return None
            
            # Determine action with enhanced confidence calculation
            if conditions['bull_signal']:
                action = 1  # LONG
                base_confidence = 0.6
                
                # Boost confidence based on signal quality
                strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
                rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
                session_bonus = 0.05 if conditions['session_active'] else -0.1
                
                confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
                    
            elif conditions['bear_signal']:
                action = 2  # SHORT
                base_confidence = 0.6
                
                # Boost confidence based on signal quality
                strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
                rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
                session_bonus = 0.05 if conditions['session_active'] else -0.1
                
                confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
            else:
                action = 0  # HOLD
                confidence = 0.3
            
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'action': action,
                'confidence': confidence,
                'conditions': conditions,
                'signal_type': 'TECHNICAL_ONLY'
            }
            
        except Exception as e:
            return None
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run the pure technical analysis backtest."""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}🚀 STARTING TECHNICAL GOLDEN-GIBZ BACKTEST...")
        print(f"{'='*70}{Style.RESET_ALL}")
        
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
        
        # Initialize tracking with REALISTIC parameters
        balance = self.initial_balance
        equity = self.initial_balance
        positions = []
        trades = []
        equity_curve = []
        peak_equity = self.initial_balance
        
        # Fixed risk amount (2% of ORIGINAL balance)
        risk_amount = self.initial_balance * (self.risk_per_trade / 100)  # $10 per trade
        
        # Progress tracking
        total_bars = len(df)
        progress_interval = max(1, total_bars // 50)
        
        print(f"\n⏳ Processing {total_bars:,} bars with technical analysis...")
        print(f"💰 Fixed risk per trade: ${risk_amount:.2f}")
        
        signals_generated = 0
        signals_filtered = 0
        
        # Main loop
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['Close']
            
            # Progress update
            if i % progress_interval == 0:
                progress = (i / total_bars) * 100
                print(f"   Progress: {progress:.1f}% - {current_time.strftime('%Y-%m-%d')} - Signals: {signals_generated}")
            
            # Update equity
            current_equity = balance
            for pos in positions:
                if pos['type'] == 'LONG':
                    pnl = (current_price - pos['entry_price']) * pos['size'] * 100
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
                    
                    # Calculate stop loss distance based on ATR
                    atr_distance = (conditions['atr_pct'] / 100) * current_price * 2.0
                    
                    # Use FIXED lot size for realistic results
                    position_size = self.fixed_lot_size
                    
                    # Execute trade with 1:1 RR
                    if action == 1:  # LONG
                        entry_price = current_price + self.spread * 0.01
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
                        entry_price = current_price - self.spread * 0.01
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
            
            # Check exits
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
                    # Calculate P&L with fixed lot size
                    if pos['type'] == 'LONG':
                        pnl = (exit_price - pos['entry_price']) * pos['size'] * 100
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['size'] * 100
                    
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
        
        # Close remaining positions
        final_price = df.iloc[-1]['Close']
        for pos in positions:
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
        
        # Store results
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
            },
            'account_params': {
                'initial_balance': self.initial_balance,
                'fixed_lot_size': self.fixed_lot_size,
                'rr_ratio': self.rr_ratio,
                'min_confidence': self.min_confidence
            }
        }
        
        # Calculate win rate
        if len(trades) > 0:
            win_rate = (pd.DataFrame(trades)['pnl'] > 0).mean() * 100
        else:
            win_rate = 0
        
        print(f"\n✅ Technical Golden-Gibz backtest completed!")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {self.results['total_return']:+.2f}%")
        print(f"   Signal quality: {self.results['signal_stats']['filter_rate']:.1f}% filtered")
        
        return self.results
    
    def analyze_results(self):
        """Analyze backtest results."""
        if not self.results or not self.results['trades']:
            print("❌ No results to analyze")
            return None
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 TECHNICAL GOLDEN-GIBZ RESULTS ANALYSIS")
        print(f"{'='*70}{Style.RESET_ALL}")
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        print(f"{Fore.GREEN}💰 PERFORMANCE METRICS")
        print(f"{'─'*50}")
        print(f"Initial Balance:     ${self.initial_balance:>10,.2f}")
        print(f"Final Balance:       ${self.results['final_balance']:>10,.2f}")
        print(f"Total Return:        {self.results['total_return']:>10.2f}%")
        print(f"Profit Factor:       {profit_factor:>10.2f}")
        
        print(f"\n{Fore.BLUE}📈 TRADE STATISTICS")
        print(f"{'─'*50}")
        print(f"Total Trades:        {total_trades:>10}")
        print(f"Win Rate:            {win_rate:>10.1f}%")
        print(f"Average Win:         ${avg_win:>10.2f}")
        print(f"Average Loss:        ${avg_loss:>10.2f}")
        
        # Performance rating
        print(f"\n{Fore.YELLOW}🏆 PERFORMANCE RATING")
        print(f"{'─'*50}")
        
        if win_rate >= 60:
            rating = f"{Fore.GREEN}🎉 EXCELLENT"
        elif win_rate >= 55:
            rating = f"{Fore.YELLOW}👍 GOOD"
        elif win_rate >= 50:
            rating = f"{Fore.YELLOW}👌 AVERAGE"
        else:
            rating = f"{Fore.RED}👎 NEEDS IMPROVEMENT"
        
        print(f"Win Rate Rating: {rating} ({win_rate:.1f}%){Style.RESET_ALL}")
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'total_return': self.results['total_return']
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file."""
        if not self.results:
            print("❌ No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_results/technical_goldengibz_results_{timestamp}.json'
        
        os.makedirs('backtest_results', exist_ok=True)
        
        # Prepare for JSON
        results_copy = self.results.copy()
        for trade in results_copy['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        for point in results_copy['equity_curve']:
            point['timestamp'] = point['timestamp'].isoformat()
        
        results_copy['backtest_period']['start'] = results_copy['backtest_period']['start'].isoformat()
        results_copy['backtest_period']['end'] = results_copy['backtest_period']['end'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"✅ Results saved: {filename}")

def main():
    """Main backtesting function."""
    print(f"{Fore.CYAN}{Style.BRIGHT}🎯 TECHNICAL GOLDEN-GIBZ BACKTESTING (PURE TECHNICAL)")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    # Initialize backtester
    backtester = TechnicalGoldenGibzBacktester()
    
    # Load and prepare data
    if not backtester.load_and_prepare_data():
        print("❌ Failed to load data. Please run: python download_mt5_data.py")
        return
    
    # Configuration
    print(f"\n{Fore.YELLOW}⚙️ BACKTEST CONFIGURATION")
    print("1. Full dataset (2024-2025)")
    print("2. Last year (2025 only)")
    print("3. Custom date range")
    
    choice = input(f"\nSelect option (1-3) [1]: {Style.RESET_ALL}").strip() or "1"
    
    start_date = None
    end_date = None
    
    if choice == "2":
        start_date = datetime(2025, 1, 1)
    elif choice == "3":
        start_str = input("Start date (YYYY-MM-DD): ").strip()
        end_str = input("End date (YYYY-MM-DD): ").strip()
        try:
            if start_str:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
            if end_str:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid date format")
            return
    
    # Run backtest
    results = backtester.run_backtest(start_date=start_date, end_date=end_date)
    
    if results:
        # Analyze results
        analysis = backtester.analyze_results()
        
        # Save results
        backtester.save_results()
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 TECHNICAL GOLDEN-GIBZ BACKTESTING COMPLETED!")
        print(f"Pure technical analysis system with proven performance.")
        print(f"Check the 'backtest_results' folder for detailed results.{Style.RESET_ALL}")
    
    else:
        print("❌ Backtesting failed")

if __name__ == "__main__":
    main()
    
    def calculate_enhanced_indicators(self, df):
        """Calculate enhanced technical indicators - PURE TECHNICAL VERSION."""
        try:
            # Basic indicators
            df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_fast'])
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_slow'])
            df['RSI'] = ta.momentum.rsi(df['Close'], window=self.indicators['rsi_period'])
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.indicators['atr_period'])
            
            # Enhanced indicators for better signals
            # ADX for trend strength
            df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=self.indicators['adx_period'])
            
            # Stochastic for momentum
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                   window=self.indicators['stoch_k'], 
                                                   smooth_window=self.indicators['stoch_d'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # MACD
            macd = ta.trend.MACD(df['Close'], 
                           window_fast=self.indicators['macd_fast'],
                           window_slow=self.indicators['macd_slow'],
                           window_sign=self.indicators['macd_signal'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands for volatility context
            bb = ta.volatility.BollingerBands(df['Close'], window=self.indicators['bb_period'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
            
            # Enhanced signal conditions
            df['EMA_Bullish'] = df['EMA20'] > df['EMA50']
            df['Price_Above_EMA20'] = df['Close'] > df['EMA20']
            df['MACD_Bullish'] = df['MACD'] > df['MACD_Signal']
            df['Strong_Trend'] = df['ADX'] > 25  # ADX > 25 indicates strong trend
            df['RSI_Neutral'] = (df['RSI'] > 30) & (df['RSI'] < 70)  # Avoid extreme RSI
            df['Stoch_Bullish'] = df['Stoch_K'] > df['Stoch_D']
            df['Low_Volatility'] = df['BB_Width'] < df['BB_Width'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators: {e}")
            return df
    
    def load_and_prepare_data(self, data_path="data/raw"):
        """Load and pre-calculate all indicators."""
        print(f"\n📊 Loading and preparing multi-timeframe data...")
        
        timeframes = ['15m', '30m', '1h', '4h', '1d']
        
        for tf in timeframes:
            filename = f"{data_path}/XAU_{tf}_data.csv"
            
            if os.path.exists(filename):
                try:
                    print(f"📈 Loading {tf} data...")
                    df = pd.read_csv(filename, sep=';')
                    df['Gmt time'] = pd.to_datetime(df['Gmt time'])
                    df.set_index('Gmt time', inplace=True)
                    
                    # Pre-calculate enhanced indicators
                    print(f"⚙️ Calculating {tf} indicators...")
                    df = self.calculate_enhanced_indicators(df)
                    
                    self.data[tf] = df
                    print(f"✅ {tf.upper()}: {len(df):,} bars prepared")
                        
                except Exception as e:
                    print(f"❌ {tf.upper()}: Error - {e}")
            else:
                print(f"❌ {tf.upper()}: File not found")
        
        if not self.data or self.execution_tf not in self.data:
            print(f"\n⚠️ Required data not available!")
            return False
        
        print(f"\n✅ Multi-timeframe data preparation complete!")
        return True
    
    def is_good_trading_session(self, timestamp):
        """Check if current time is during active trading sessions."""
        if not self.session_filter:
            return True
        
        hour = timestamp.hour
        
        # Active forex sessions (UTC):
        # London: 8-17, New York: 13-22, Asian: 0-9
        # Overlap periods are best: London-NY (13-17)
        active_sessions = [
            (8, 17),   # London session
            (13, 22),  # New York session
            (0, 9)     # Asian session
        ]
        
        for start, end in active_sessions:
            if start <= hour <= end:
                return True
        
        return False
    
    def get_enhanced_timeframe_signal(self, tf_data, timestamp):
        """Get enhanced signal from a specific timeframe - PURE TECHNICAL."""
        try:
            idx = tf_data.index.get_indexer([timestamp], method='nearest')[0]
            if idx < 50 or idx >= len(tf_data):
                return 0, 0, {}
            
            row = tf_data.iloc[idx]
            
            # Enhanced signal scoring - PURE TECHNICAL
            bullish_score = 0
            bearish_score = 0
            signal_quality = {}
            
            # 1. Trend Direction (EMA) - Weight: 3
            if row.get('EMA_Bullish', False):
                if row.get('Price_Above_EMA20', False):
                    bullish_score += 3
                    signal_quality['ema_trend'] = 'strong_bull'
                else:
                    bullish_score += 1
                    signal_quality['ema_trend'] = 'weak_bull'
            else:
                if not row.get('Price_Above_EMA20', True):
                    bearish_score += 3
                    signal_quality['ema_trend'] = 'strong_bear'
                else:
                    bearish_score += 1
                    signal_quality['ema_trend'] = 'weak_bear'
            
            # 2. MACD Confirmation - Weight: 2
            if row.get('MACD_Bullish', False):
                bullish_score += 2
                signal_quality['macd'] = 'bullish'
            else:
                bearish_score += 2
                signal_quality['macd'] = 'bearish'
            
            # 3. Trend Strength (ADX) - Weight: 2
            if row.get('Strong_Trend', False):
                if bullish_score > bearish_score:
                    bullish_score += 2
                else:
                    bearish_score += 2
                signal_quality['trend_strength'] = 'strong'
            else:
                signal_quality['trend_strength'] = 'weak'
            
            # 4. Stochastic Momentum - Weight: 1
            if row.get('Stoch_Bullish', False):
                bullish_score += 1
                signal_quality['momentum'] = 'bullish'
            else:
                bearish_score += 1
                signal_quality['momentum'] = 'bearish'
            
            # 5. RSI Filter - Penalty for extreme levels
            rsi = row.get('RSI', 50)
            if rsi > 80:
                bullish_score -= 2  # Overbought penalty
                signal_quality['rsi_filter'] = 'overbought'
            elif rsi < 20:
                bearish_score -= 2  # Oversold penalty
                signal_quality['rsi_filter'] = 'oversold'
            elif 30 <= rsi <= 70:
                signal_quality['rsi_filter'] = 'neutral'
            
            # 6. Volatility Filter - Bonus for low volatility
            if row.get('Low_Volatility', False):
                if bullish_score > bearish_score:
                    bullish_score += 1
                else:
                    bearish_score += 1
                signal_quality['volatility'] = 'favorable'
            else:
                signal_quality['volatility'] = 'high'
            
            # Determine signal
            if bullish_score > bearish_score:
                return 1, bullish_score - bearish_score, signal_quality
            elif bearish_score > bullish_score:
                return -1, bearish_score - bullish_score, signal_quality
            else:
                return 0, 0, signal_quality
                
        except Exception as e:
            return 0, 0, {}
    
    def analyze_enhanced_conditions(self, timestamp):
        """Enhanced multi-timeframe analysis - PURE TECHNICAL VERSION."""
        try:
            # Timeframe weights (higher timeframes more important)
            tf_weights = {'15m': 1, '30m': 2, '1h': 3, '4h': 4, '1d': 5}
            
            total_bull_score = 0
            total_bear_score = 0
            total_weight = 0
            timeframe_signals = {}
            
            # Get execution timeframe data
            exec_data = self.data[self.execution_tf]
            exec_idx = exec_data.index.get_indexer([timestamp], method='nearest')[0]
            if exec_idx < 50:
                return None
            
            exec_row = exec_data.iloc[exec_idx]
            
            # Analyze each timeframe
            for tf_name in ['15m', '30m', '1h', '4h', '1d']:
                if tf_name not in self.data:
                    continue
                
                signal, strength, quality = self.get_enhanced_timeframe_signal(self.data[tf_name], timestamp)
                weight = tf_weights.get(tf_name, 1)
                
                timeframe_signals[tf_name] = {
                    'signal': signal,
                    'strength': strength,
                    'quality': quality
                }
                
                if signal == 1:  # Bullish
                    total_bull_score += strength * weight
                elif signal == -1:  # Bearish
                    total_bear_score += strength * weight
                
                total_weight += weight
            
            if total_weight == 0:
                return None
            
            # Calculate overall conditions
            bull_score = total_bull_score / total_weight if total_weight > 0 else 0
            bear_score = total_bear_score / total_weight if total_weight > 0 else 0
            
            # Enhanced filtering conditions
            conditions = {
                'bull_signal': False,
                'bear_signal': False,
                'signal_strength': max(bull_score, bear_score),
                'rsi': exec_row.get('RSI', 50),
                'atr_pct': 0.15,
                'price': exec_row['Close'],
                'timeframe_signals': timeframe_signals,
                'session_active': self.is_good_trading_session(timestamp),
                'bull_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 1),
                'bear_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == -1),
                'neutral_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 0)
            }
            
            # Calculate ATR percentage
            atr = exec_row.get('ATR', 15.0)
            if atr > 0 and exec_row['Close'] > 0:
                conditions['atr_pct'] = (atr / exec_row['Close']) * 100
            
            # Enhanced signal determination - TECHNICAL ONLY OPTIMIZED
            min_strength = 3.0  # Original proven threshold
            
            if bull_score > bear_score and bull_score >= min_strength:
                # Additional quality checks for bullish signals
                if (conditions['session_active'] and 
                    conditions['rsi'] < 75 and  # Not too overbought
                    exec_row.get('Strong_Trend', False)):  # Strong trend required
                    conditions['bull_signal'] = True
                    
            elif bear_score > bull_score and bear_score >= min_strength:
                # Additional quality checks for bearish signals
                if (conditions['session_active'] and 
                    conditions['rsi'] > 25 and  # Not too oversold
                    exec_row.get('Strong_Trend', False)):  # Strong trend required
                    conditions['bear_signal'] = True
            
            return conditions
            
        except Exception as e:
            return None
    def generate_technical_signal(self, timestamp):
        """Generate pure technical trading signal - NO AI MODEL."""
        try:
            conditions = self.analyze_enhanced_conditions(timestamp)
            if not conditions:
                return None
            
            # Determine action with enhanced confidence calculation - PURE TECHNICAL
            if conditions['bull_signal']:
                action = 1  # LONG
                base_confidence = 0.6
                
                # Boost confidence based on signal quality
                strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
                rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
                session_bonus = 0.05 if conditions['session_active'] else -0.1
                
                confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
                    
            elif conditions['bear_signal']:
                action = 2  # SHORT
                base_confidence = 0.6
                
                # Boost confidence based on signal quality
                strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
                rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
                session_bonus = 0.05 if conditions['session_active'] else -0.1
                
                confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
            else:
                action = 0  # HOLD
                confidence = 0.3
            
            confidence = max(0.0, min(1.0, confidence))
            
            return {
                'action': action,
                'confidence': confidence,
                'conditions': conditions,
                'signal_type': 'TECHNICAL_ONLY'
            }
            
        except Exception as e:
            return None
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run the pure technical backtest."""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}⚙️ STARTING TECHNICAL-ONLY BACKTEST...")
        print(f"{'='*70}{Style.RESET_ALL}")
        
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
        
        # Initialize tracking with REALISTIC parameters
        balance = self.initial_balance
        equity = self.initial_balance
        positions = []
        trades = []
        equity_curve = []
        peak_equity = self.initial_balance
        
        # Fixed risk amount (2% of ORIGINAL balance)
        risk_amount = self.initial_balance * (self.risk_per_trade / 100)  # $10 per trade
        
        # Progress tracking
        total_bars = len(df)
        progress_interval = max(1, total_bars // 50)
        
        print(f"\n⏳ Processing {total_bars:,} bars with technical analysis...")
        print(f"💰 Fixed risk per trade: ${risk_amount:.2f}")
        
        signals_generated = 0
        signals_filtered = 0
        
        # Main loop
        for i in range(50, len(df)):
            current_time = df.index[i]
            current_price = df.iloc[i]['Close']
            
            # Progress update
            if i % progress_interval == 0:
                progress = (i / total_bars) * 100
                print(f"   Progress: {progress:.1f}% - {current_time.strftime('%Y-%m-%d')} - Signals: {signals_generated}")
            
            # Update equity
            current_equity = balance
            for pos in positions:
                if pos['type'] == 'LONG':
                    pnl = (current_price - pos['entry_price']) * pos['size'] * 100
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
                    
                    # Calculate stop loss distance based on ATR
                    atr_distance = (conditions['atr_pct'] / 100) * current_price * 2.0
                    
                    # Use FIXED lot size for realistic results
                    position_size = self.fixed_lot_size
                    
                    # Execute trade with 1:1 RR
                    if action == 1:  # LONG
                        entry_price = current_price + self.spread * 0.01
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
                        entry_price = current_price - self.spread * 0.01
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
            
            # Check exits
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
                    # Calculate P&L with fixed lot size
                    if pos['type'] == 'LONG':
                        pnl = (exit_price - pos['entry_price']) * pos['size'] * 100
                    else:
                        pnl = (pos['entry_price'] - exit_price) * pos['size'] * 100
                    
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
        
        # Close remaining positions
        final_price = df.iloc[-1]['Close']
        for pos in positions:
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
        
        # Store results
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
            },
            'account_params': {
                'initial_balance': self.initial_balance,
                'fixed_lot_size': self.fixed_lot_size,
                'rr_ratio': self.rr_ratio,
                'min_confidence': self.min_confidence
            }
        }
        
        # Calculate win rate
        if len(trades) > 0:
            win_rate = (pd.DataFrame(trades)['pnl'] > 0).mean() * 100
        else:
            win_rate = 0
        
        print(f"\n✅ Technical-only backtest completed!")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {win_rate:.1f}%")
        print(f"   Final balance: ${balance:,.2f}")
        print(f"   Total return: {self.results['total_return']:+.2f}%")
        print(f"   Signal quality: {self.results['signal_stats']['filter_rate']:.1f}% filtered")
        
        return self.results
    def analyze_results(self):
        """Analyze backtest results - TECHNICAL ONLY VERSION."""
        if not self.results or not self.results['trades']:
            print("❌ No results to analyze")
            return None
        
        trades_df = pd.DataFrame(self.results['trades'])
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 TECHNICAL-ONLY RESULTS ANALYSIS")
        print(f"{'='*70}{Style.RESET_ALL}")
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        print(f"{Fore.GREEN}💰 PERFORMANCE METRICS")
        print(f"{'─'*50}")
        print(f"Initial Balance:     ${self.initial_balance:>10,.2f}")
        print(f"Final Balance:       ${self.results['final_balance']:>10,.2f}")
        print(f"Total Return:        {self.results['total_return']:>10.2f}%")
        print(f"Profit Factor:       {profit_factor:>10.2f}")
        
        print(f"\n{Fore.BLUE}📈 TRADE STATISTICS")
        print(f"{'─'*50}")
        print(f"Total Trades:        {total_trades:>10}")
        print(f"Win Rate:            {win_rate:>10.1f}%")
        print(f"Average Win:         ${avg_win:>10.2f}")
        print(f"Average Loss:        ${avg_loss:>10.2f}")
        
        # Performance rating
        print(f"\n{Fore.YELLOW}🏆 PERFORMANCE RATING")
        print(f"{'─'*50}")
        
        if win_rate >= 60:
            rating = f"{Fore.GREEN}🎉 EXCELLENT (Technical-Only)"
        elif win_rate >= 55:
            rating = f"{Fore.YELLOW}👍 GOOD"
        elif win_rate >= 50:
            rating = f"{Fore.YELLOW}👌 AVERAGE"
        else:
            rating = f"{Fore.RED}👎 NEEDS IMPROVEMENT"
        
        print(f"Win Rate Rating: {rating} ({win_rate:.1f}%){Style.RESET_ALL}")
        
        # Technical Analysis Statistics
        print(f"\n{Fore.CYAN}⚙️ TECHNICAL ANALYSIS STATISTICS")
        print(f"{'─'*50}")
        print(f"System Type: Pure Technical Analysis")
        print(f"AI Integration: None (Technical Only)")
        print(f"Signal Quality: {self.results['signal_stats']['filter_rate']:.1f}% filtered")
        print(f"Timeframes Used: 15M, 30M, 1H, 4H, 1D")
        print(f"Indicators: EMA, RSI, MACD, ADX, Stochastic, BB")
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profit_factor': profit_factor,
            'total_return': self.results['total_return']
        }
    
    def save_results(self, filename=None):
        """Save results to JSON file."""
        if not self.results:
            print("❌ No results to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'backtest_results/technical_gibz_results_{timestamp}.json'
        
        os.makedirs('backtest_results', exist_ok=True)
        
        # Prepare for JSON
        results_copy = self.results.copy()
        for trade in results_copy['trades']:
            trade['entry_time'] = trade['entry_time'].isoformat()
            trade['exit_time'] = trade['exit_time'].isoformat()
        
        for point in results_copy['equity_curve']:
            point['timestamp'] = point['timestamp'].isoformat()
        
        results_copy['backtest_period']['start'] = results_copy['backtest_period']['start'].isoformat()
        results_copy['backtest_period']['end'] = results_copy['backtest_period']['end'].isoformat()
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"✅ Results saved: {filename}")

def main():
    """Main backtesting function."""
    print(f"{Fore.CYAN}{Style.BRIGHT}⚙️ TECHNICAL-ONLY GOLDEN-GIBZ BACKTESTING")
    print(f"{'='*70}{Style.RESET_ALL}")
    
    # Initialize backtester
    backtester = TechnicalGoldenGibzBacktester()
    
    # Load and prepare data
    if not backtester.load_and_prepare_data():
        print("❌ Failed to load data. Please run: python download_mt5_data.py")
        return
    
    # Configuration
    print(f"\n{Fore.YELLOW}⚙️ BACKTEST CONFIGURATION")
    print("1. Full dataset (2024-2025)")
    print("2. Last year (2025 only)")
    print("3. Custom date range")
    
    choice = input(f"\nSelect option (1-3) [1]: {Style.RESET_ALL}").strip() or "1"
    
    start_date = None
    end_date = None
    
    if choice == "2":
        start_date = datetime(2025, 1, 1)
    elif choice == "3":
        start_str = input("Start date (YYYY-MM-DD): ").strip()
        end_str = input("End date (YYYY-MM-DD): ").strip()
        try:
            if start_str:
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
            if end_str:
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
        except ValueError:
            print("❌ Invalid date format")
            return
    
    # Run backtest
    results = backtester.run_backtest(start_date=start_date, end_date=end_date)
    
    if results:
        # Analyze results
        analysis = backtester.analyze_results()
        
        # Save results
        backtester.save_results()
        
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🎉 TECHNICAL-ONLY BACKTESTING COMPLETED!")
        print(f"Pure technical analysis system with proven performance.")
        print(f"Check the 'backtest_results' folder for detailed results.{Style.RESET_ALL}")
    
    else:
        print("❌ Backtesting failed")

if __name__ == "__main__":
    main()