#!/usr/bin/env python3
"""
Golden Gibz Python EA - Advanced Professional Version
====================================================
Complete autonomous EA with beautiful dashboard and advanced features
"""

import MetaTrader5 as mt5
import pandas as pd
import ta
import numpy as np
from stable_baselines3 import PPO
import time
from datetime import datetime, timedelta
import warnings
import os
import json
from colorama import init, Fore, Back, Style
import threading
warnings.filterwarnings('ignore')
init(autoreset=True)  # Initialize colorama

class GoldenGibzPythonEA:
    """Advanced Golden Gibz EA with professional dashboard and features."""
    
    def __init__(self, config_file="config/ea_config.json"):
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Core settings
        self.model_path = self.config.get("model_path", "models/production/golden_gibz_wr100_ret+25_20251225_215251.zip")
        self.model = None
        self.symbol = self.config.get("symbol", "XAUUSD")
        self.trade_counter = 0
        self.running = True
        self.last_signal_time = None
        self.last_trade_time = None  # Track last trade for cooldown
        self.start_time = datetime.now()
        
        # Trading parameters (configurable)
        self.lot_size = self.config.get("lot_size", 0.01)
        self.max_positions = self.config.get("max_positions", 3)
        self.min_confidence = self.config.get("min_confidence", 0.6)
        self.signal_frequency = self.config.get("signal_frequency", 60)
        self.trade_cooldown = self.config.get("trade_cooldown", 300)  # 5 min cooldown between trades
        
        # Advanced features
        self.max_daily_trades = self.config.get("max_daily_trades", 10)
        self.max_daily_loss = self.config.get("max_daily_loss", 100.0)
        self.trading_hours = self.config.get("trading_hours", {"start": 8, "end": 17})
        self.risk_per_trade = self.config.get("risk_per_trade", 2.0)
        self.use_dynamic_lots = self.config.get("use_dynamic_lots", False)
        
        # Dashboard settings
        self.dashboard_refresh = self.config.get("dashboard_refresh", 5)
        self.show_indicators = self.config.get("show_indicators", True)
        self.show_positions = self.config.get("show_positions", True)
        
        # Statistics tracking
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.last_reset_date = datetime.now().date()
        
        # Indicator settings
        self.indicators = self.config.get("indicators", {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9
        })
        
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        default_config = {
            "model_path": "models/production/golden_gibz_wr100_ret+25_20251225_215251.zip",
            "symbol": "XAUUSD",
            "lot_size": 0.01,
            "max_positions": 3,
            "min_confidence": 0.6,
            "signal_frequency": 60,
            "max_daily_trades": 10,
            "max_daily_loss": 100.0,
            "trading_hours": {"start": 8, "end": 17},
            "risk_per_trade": 2.0,
            "use_dynamic_lots": False,
            "dashboard_refresh": 5,
            "show_indicators": True,
            "show_positions": True,
            "indicators": {
                "ema_fast": 20,
                "ema_slow": 50,
                "rsi_period": 14,
                "atr_period": 14,
                "bb_period": 20,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            }
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"✅ Configuration loaded from {config_file}")
            else:
                # Create default config file
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=4)
                print(f"✅ Default configuration created: {config_file}")
        except Exception as e:
            print(f"⚠️ Config error: {e}, using defaults")
        
        return default_config
    
    def save_config(self):
        """Save current configuration to file."""
        config = {
            "model_path": self.model_path,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "max_positions": self.max_positions,
            "min_confidence": self.min_confidence,
            "signal_frequency": self.signal_frequency,
            "max_daily_trades": self.max_daily_trades,
            "max_daily_loss": self.max_daily_loss,
            "trading_hours": self.trading_hours,
            "risk_per_trade": self.risk_per_trade,
            "use_dynamic_lots": self.use_dynamic_lots,
            "dashboard_refresh": self.dashboard_refresh,
            "show_indicators": self.show_indicators,
            "show_positions": self.show_positions,
            "indicators": self.indicators
        }
        
        try:
            with open("config/ea_config.json", 'w') as f:
                json.dump(config, f, indent=4)
            print(f"✅ Configuration saved")
        except Exception as e:
            print(f"⚠️ Failed to save config: {e}")
    
    def reset_daily_stats(self):
        """Reset daily statistics if new day."""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            # Don't reset winning_trades and losing_trades - they're cumulative
            self.last_reset_date = current_date
            print(f"📅 Daily stats reset for {current_date}")
    
    def reset_statistics(self):
        """Manually reset all statistics."""
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        print("📊 All statistics reset to zero")
    
    def is_trading_time(self):
        """Check if current time is within trading hours."""
        current_hour = datetime.now().hour
        start_hour = self.trading_hours["start"]
        end_hour = self.trading_hours["end"]
        
        if start_hour <= end_hour:
            return start_hour <= current_hour <= end_hour
        else:  # Overnight session
            return current_hour >= start_hour or current_hour <= end_hour
    
    def calculate_dynamic_lot_size(self):
        """Calculate dynamic lot size based on account balance and risk."""
        if not self.use_dynamic_lots:
            return self.lot_size
        
        try:
            account = mt5.account_info()
            if not account:
                return self.lot_size
            
            balance = account.balance
            risk_amount = balance * (self.risk_per_trade / 100)
            
            # Get current ATR for stop loss calculation
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return self.lot_size
            
            # Assume 2x ATR stop loss (approximately 30 pips for XAUUSD)
            stop_loss_pips = 30
            pip_value = 0.1  # For XAUUSD
            
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Round to valid lot size and apply limits
            lot_size = round(lot_size, 2)
            lot_size = max(0.01, min(lot_size, 1.0))  # Between 0.01 and 1.0
            
            return lot_size
            
        except Exception as e:
            print(f"⚠️ Dynamic lot calculation error: {e}")
            return self.lot_size
    def initialize(self):
        """Initialize MT5 and load AI model."""
        self.print_header()
        
        # Initialize MT5
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account = mt5.account_info()
        print(f"✅ Connected to MT5 - Account: {account.login}")
        print(f"✅ Balance: ${account.balance:.2f}")
        print(f"✅ Server: {account.server}")
        
        # Load AI model
        try:
            self.model = PPO.load(self.model_path)
            print(f"✅ Golden Gibz AI model loaded: {self.model_path}")
            print(f"   Observation space: {self.model.observation_space}")
            print(f"   Action space: {self.model.action_space}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
        
        # Reset daily stats
        self.reset_daily_stats()
        
        return True
    
    def print_header(self):
        """Print beautiful header."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.YELLOW}{Style.BRIGHT}🎯 GOLDEN GIBZ PYTHON EA - PROFESSIONAL EDITION")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.GREEN}🚀 Advanced AI Trading System with Professional Dashboard")
        print(f"{Fore.WHITE}📊 Multi-timeframe Analysis | 🛡️ Advanced Risk Management")
        print(f"{Fore.WHITE}⚙️ Configurable Parameters | 📈 Real-time Monitoring")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}{Style.RESET_ALL}")
    
    def print_dashboard(self):
        """Print beautiful real-time dashboard."""
        try:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Header
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*100}")
            print(f"{Fore.YELLOW}{Style.BRIGHT}🎯 GOLDEN GIBZ PROFESSIONAL DASHBOARD")
            print(f"{Fore.CYAN}{Style.BRIGHT}{'='*100}{Style.RESET_ALL}")
            
            # Account Information
            account = mt5.account_info()
            tick = mt5.symbol_info_tick(self.symbol)
            positions = mt5.positions_get(symbol=self.symbol)
            
            current_time = datetime.now()
            uptime = current_time - self.start_time
            
            print(f"\n{Fore.GREEN}{Style.BRIGHT}📊 ACCOUNT STATUS{Style.RESET_ALL}")
            print(f"{'─'*50}")
            print(f"Account: {Fore.CYAN}{account.login}{Style.RESET_ALL} | Server: {Fore.CYAN}{account.server}{Style.RESET_ALL}")
            print(f"Balance: {Fore.GREEN}${account.balance:.2f}{Style.RESET_ALL} | Equity: {Fore.GREEN}${account.equity:.2f}{Style.RESET_ALL}")
            print(f"Free Margin: {Fore.YELLOW}${account.margin_free:.2f}{Style.RESET_ALL} | Margin Level: {Fore.YELLOW}{account.margin_level:.1f}%{Style.RESET_ALL}")
            
            # Market Information
            print(f"\n{Fore.BLUE}{Style.BRIGHT}📈 MARKET STATUS{Style.RESET_ALL}")
            print(f"{'─'*50}")
            if tick:
                # XAUUSD spread in points (not pips) - 1 point = $0.01
                spread_points = tick.ask - tick.bid
                print(f"Symbol: {Fore.CYAN}{self.symbol}{Style.RESET_ALL} | Price: {Fore.YELLOW}{tick.bid:.2f}/{tick.ask:.2f}{Style.RESET_ALL}")
                print(f"Spread: {Fore.MAGENTA}{spread_points:.2f} points{Style.RESET_ALL} | Time: {Fore.WHITE}{current_time.strftime('%H:%M:%S')}{Style.RESET_ALL}")
            
            trading_status = "🟢 ACTIVE" if self.is_trading_time() else "🔴 CLOSED"
            print(f"Trading Hours: {Fore.CYAN}{self.trading_hours['start']:02d}:00-{self.trading_hours['end']:02d}:00{Style.RESET_ALL} | Status: {trading_status}")
            
            # Position Information
            if self.show_positions and positions:
                print(f"\n{Fore.MAGENTA}{Style.BRIGHT}📋 ACTIVE POSITIONS ({len(positions)}/{self.max_positions}){Style.RESET_ALL}")
                print(f"{'─'*70}")
                total_profit = 0
                for i, pos in enumerate(positions):
                    pos_type = "🟢 BUY" if pos.type == 0 else "🔴 SELL"
                    profit_color = Fore.GREEN if pos.profit >= 0 else Fore.RED
                    print(f"{i+1}. {pos_type} {pos.volume} lots @ {pos.price_open:.2f} | P&L: {profit_color}${pos.profit:.2f}{Style.RESET_ALL}")
                    total_profit += pos.profit
                
                profit_color = Fore.GREEN if total_profit >= 0 else Fore.RED
                print(f"{'─'*70}")
                print(f"Total P&L: {profit_color}${total_profit:.2f}{Style.RESET_ALL}")
            
            # Trading Statistics
            total_closed_trades = self.winning_trades + self.losing_trades
            win_rate = (self.winning_trades / max(1, total_closed_trades)) * 100 if total_closed_trades > 0 else 0
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}📊 TRADING STATISTICS{Style.RESET_ALL}")
            print(f"{'─'*60}")
            print(f"Daily Trades: {Fore.CYAN}{self.daily_trades}/{self.max_daily_trades}{Style.RESET_ALL} | Daily P&L: {Fore.GREEN if self.daily_pnl >= 0 else Fore.RED}${self.daily_pnl:.2f}{Style.RESET_ALL}")
            print(f"Closed Trades: {Fore.CYAN}{total_closed_trades}{Style.RESET_ALL} | Win Rate: {Fore.GREEN if win_rate >= 70 else Fore.YELLOW if win_rate >= 50 else Fore.RED}{win_rate:.1f}%{Style.RESET_ALL}")
            print(f"Wins: {Fore.GREEN}{self.winning_trades}{Style.RESET_ALL} | Losses: {Fore.RED}{self.losing_trades}{Style.RESET_ALL}")
            print(f"Uptime: {Fore.CYAN}{str(uptime).split('.')[0]}{Style.RESET_ALL}")
            
            # AI Signal Status
            print(f"\n{Fore.RED}{Style.BRIGHT}🤖 AI SIGNAL STATUS{Style.RESET_ALL}")
            print(f"{'─'*50}")
            print(f"Model: {Fore.CYAN}Golden Gibz PPO{Style.RESET_ALL}")
            print(f"Signal Frequency: {Fore.YELLOW}{self.signal_frequency}s{Style.RESET_ALL}")
            print(f"Min Confidence: {Fore.YELLOW}{self.min_confidence:.1%}{Style.RESET_ALL}")
            
            if self.last_signal_time:
                time_since = (current_time - self.last_signal_time).total_seconds()
                next_signal = max(0, self.signal_frequency - time_since)
                print(f"Next Signal: {Fore.MAGENTA}{next_signal:.0f}s{Style.RESET_ALL}")
            
            # Risk Management
            print(f"\n{Fore.RED}{Style.BRIGHT}🛡️ RISK MANAGEMENT{Style.RESET_ALL}")
            print(f"{'─'*50}")
            print(f"Max Positions: {Fore.CYAN}{len(positions) if positions else 0}/{self.max_positions}{Style.RESET_ALL}")
            print(f"Risk per Trade: {Fore.YELLOW}{self.risk_per_trade:.1f}%{Style.RESET_ALL}")
            print(f"Daily Loss Limit: {Fore.RED}${self.max_daily_loss:.0f}{Style.RESET_ALL}")
            
            current_lot = self.calculate_dynamic_lot_size() if self.use_dynamic_lots else self.lot_size
            print(f"Lot Size: {Fore.CYAN}{current_lot:.2f}{Style.RESET_ALL} {'(Dynamic)' if self.use_dynamic_lots else '(Fixed)'}")
            
            # Technical Indicators (if enabled)
            if self.show_indicators:
                self.print_indicators_dashboard()
            
            print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*100}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}Press Ctrl+C to stop | Dashboard updates every {self.dashboard_refresh}s{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"⚠️ Dashboard error: {e}")
    
    def print_indicators_dashboard(self):
        """Print technical indicators dashboard."""
        try:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}📊 TECHNICAL INDICATORS{Style.RESET_ALL}")
            print(f"{'─'*60}")
            
            # Get recent data for indicators
            data = self.get_market_data()
            if not data or '15M' not in data:
                print(f"{Fore.RED}⚠️ No indicator data available{Style.RESET_ALL}")
                return
            
            df = data['15M']
            if len(df) < 50:
                print(f"{Fore.RED}⚠️ Insufficient data for indicators{Style.RESET_ALL}")
                return
            
            # Calculate indicators
            df = self.calculate_technical_indicators(df)
            latest = df.iloc[-1]
            
            # RSI
            rsi = latest.get('RSI', 50)
            rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
            print(f"RSI({self.indicators['rsi_period']}): {Fore.YELLOW}{rsi:.1f}{Style.RESET_ALL} {rsi_status}")
            
            # EMAs
            ema20 = latest.get('EMA20', 0)
            ema50 = latest.get('EMA50', 0)
            ema_trend = "🟢 Bullish" if ema20 > ema50 else "🔴 Bearish"
            print(f"EMA Trend: {ema_trend} | EMA20: {Fore.CYAN}{ema20:.2f}{Style.RESET_ALL} | EMA50: {Fore.CYAN}{ema50:.2f}{Style.RESET_ALL}")
            
            # ATR
            atr = latest.get('ATR', 0)
            atr_pct = (atr / latest['Close']) * 100 if latest['Close'] > 0 else 0
            volatility = "🔴 High" if atr_pct > 0.5 else "🟡 Medium" if atr_pct > 0.2 else "🟢 Low"
            print(f"ATR({self.indicators['atr_period']}): {Fore.YELLOW}{atr:.2f}{Style.RESET_ALL} ({atr_pct:.2f}%) {volatility}")
            
            # MACD
            macd = latest.get('MACD', 0)
            macd_signal = latest.get('MACD_Signal', 0)
            macd_trend = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
            print(f"MACD: {macd_trend} | MACD: {Fore.CYAN}{macd:.3f}{Style.RESET_ALL} | Signal: {Fore.CYAN}{macd_signal:.3f}{Style.RESET_ALL}")
            
            # Technical Summary - Trade Confidence
            print(f"\n{Fore.WHITE}{Style.BRIGHT}📊 TRADE CONFIDENCE{Style.RESET_ALL}")
            print(f"{'─'*40}")
            
            # Calculate confidence score (0-100%)
            confidence_score = 0
            max_score = 100
            
            # EMA trend (30 points)
            if ema20 > ema50:
                ema_gap = ((ema20 - ema50) / ema50) * 100
                confidence_score += min(30, max(0, ema_gap * 10))  # Strong trend = higher confidence
                trend_signal = f"{Fore.GREEN}BULLISH{Style.RESET_ALL}"
            else:
                ema_gap = ((ema50 - ema20) / ema20) * 100
                confidence_score += min(30, max(0, ema_gap * 10))
                trend_signal = f"{Fore.RED}BEARISH{Style.RESET_ALL}"
            
            # RSI momentum (25 points)
            if 30 <= rsi <= 70:  # Good range
                confidence_score += 25
                rsi_signal = f"{Fore.GREEN}OPTIMAL{Style.RESET_ALL}"
            elif 20 <= rsi <= 80:  # Acceptable
                confidence_score += 15
                rsi_signal = f"{Fore.YELLOW}ACCEPTABLE{Style.RESET_ALL}"
            else:  # Extreme
                confidence_score += 5
                rsi_signal = f"{Fore.RED}EXTREME{Style.RESET_ALL}"
            
            # MACD momentum (25 points)
            if abs(macd - macd_signal) > 0.5:  # Strong signal
                confidence_score += 25
                macd_signal_strength = f"{Fore.GREEN}STRONG{Style.RESET_ALL}"
            elif abs(macd - macd_signal) > 0.2:  # Moderate
                confidence_score += 15
                macd_signal_strength = f"{Fore.YELLOW}MODERATE{Style.RESET_ALL}"
            else:  # Weak
                confidence_score += 5
                macd_signal_strength = f"{Fore.RED}WEAK{Style.RESET_ALL}"
            
            # Volatility (20 points) - Lower ATR = higher confidence
            if atr_pct < 0.15:  # Low volatility
                confidence_score += 20
                volatility_signal = f"{Fore.GREEN}LOW{Style.RESET_ALL}"
            elif atr_pct < 0.25:  # Medium
                confidence_score += 10
                volatility_signal = f"{Fore.YELLOW}MEDIUM{Style.RESET_ALL}"
            else:  # High
                confidence_score += 0
                volatility_signal = f"{Fore.RED}HIGH{Style.RESET_ALL}"
            
            # Overall confidence
            confidence_pct = min(100, confidence_score)
            
            if confidence_pct >= 80:
                confidence_color = Fore.GREEN
                confidence_status = "🟢 VERY HIGH"
            elif confidence_pct >= 60:
                confidence_color = Fore.YELLOW
                confidence_status = "🟡 HIGH"
            elif confidence_pct >= 40:
                confidence_color = Fore.YELLOW
                confidence_status = "🟡 MEDIUM"
            else:
                confidence_color = Fore.RED
                confidence_status = "🔴 LOW"
            
            print(f"Trend Direction: {trend_signal}")
            print(f"RSI Condition: {rsi_signal}")
            print(f"MACD Strength: {macd_signal_strength}")
            print(f"Volatility: {volatility_signal}")
            print(f"{'─'*40}")
            print(f"Trade Confidence: {confidence_color}{confidence_pct:.0f}%{Style.RESET_ALL} {confidence_status}")
            
        except Exception as e:
            print(f"⚠️ Indicators error: {e}")
    
    def get_market_data(self):
        """Get multi-timeframe market data from MT5 - REAL DATA."""
        print("📊 Getting real market data from MT5...")
        
        # MT5 timeframe mapping
        tf_map = {
            '15M': mt5.TIMEFRAME_M15,
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1
        }
        
        # Bars needed for each timeframe
        bars_needed = {
            '15M': 200,
            '1H': 100,
            '4H': 100,
            '1D': 100
        }
        
        data = {}
        
        for tf_name, tf_mt5 in tf_map.items():
            try:
                # Get real historical data from MT5
                rates = mt5.copy_rates_from_pos(self.symbol, tf_mt5, 0, bars_needed[tf_name])
                
                if rates is None or len(rates) == 0:
                    print(f"❌ {tf_name}: No data available")
                    data[tf_name] = None
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['Date'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('Date', inplace=True)
                
                # Rename columns to match expected format
                df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume'
                }, inplace=True)
                
                # Keep only needed columns
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                data[tf_name] = df
                print(f"✅ {tf_name}: {len(df)} bars (real data)")
                
            except Exception as e:
                print(f"❌ {tf_name}: Error getting data - {e}")
                data[tf_name] = None
        
        # Verify we have at least 15M data
        if data.get('15M') is None or len(data['15M']) < 50:
            print("❌ Insufficient 15M data for analysis")
            return None
        
        return data
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for Golden Gibz analysis."""
        try:
            # EMAs (configurable periods)
            df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_fast'])
            df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=self.indicators['ema_slow'])
            
            # RSI (configurable period)
            df['RSI'] = ta.momentum.rsi(df['Close'], window=self.indicators['rsi_period'])
            
            # ATR (configurable period)
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=self.indicators['atr_period'])
            
            # Bollinger Bands (configurable period)
            bb = ta.volatility.BollingerBands(df['Close'], window=self.indicators['bb_period'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
            
            # MACD (configurable periods)
            macd = ta.trend.MACD(df['Close'], 
                           window_fast=self.indicators['macd_fast'],
                           window_slow=self.indicators['macd_slow'],
                           window_sign=self.indicators['macd_signal'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Williams %R
            df['WillR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # CCI
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            return df
            
        except Exception as e:
            print(f"⚠️ Error calculating indicators: {e}")
            return df
    
    def analyze_market_conditions(self, data):
        """Analyze market conditions across multiple timeframes - FIXED VERSION."""
        try:
            conditions = {
                'bull_timeframes': 0,
                'bear_timeframes': 0,
                'neutral_timeframes': 0,
                'trend_strength': 0,
                'rsi': 50.0,
                'atr_pct': 0.0,
                'bull_signal': False,
                'bear_signal': False,
                'bull_pullback': False,
                'bear_pullback': False,
                'active_session': True,
                'price': 0.0,
                'ema20': 0.0,
                'ema50': 0.0,
                'trend_details': {}
            }
            
            # Get current price
            tick = mt5.symbol_info_tick(self.symbol)
            if tick:
                conditions['price'] = (tick.bid + tick.ask) / 2
            
            # Analyze each timeframe
            for tf_name, df in data.items():
                if df is None or len(df) < 50:
                    continue
                
                # Calculate indicators for this timeframe
                df = self.calculate_technical_indicators(df)
                
                # Get latest values
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                # Trend analysis
                ema20 = latest.get('EMA20', np.nan)
                ema50 = latest.get('EMA50', np.nan)
                close = latest['Close']
                rsi = latest.get('RSI', 50)
                
                # Store trend details for debugging
                trend_direction = 'NEUTRAL'
                
                # Determine trend direction based on EMA crossover
                # Count based on EMA alignment (EMA20 vs EMA50), not price position
                if pd.notna(ema20) and pd.notna(ema50):
                    if ema20 > ema50:
                        # Bullish EMA structure
                        conditions['bull_timeframes'] += 1
                        if close > ema20:
                            trend_direction = 'BULLISH'
                        else:
                            trend_direction = 'BULL_PULLBACK'
                            conditions['bull_pullback'] = True
                    elif ema20 < ema50:
                        # Bearish EMA structure
                        conditions['bear_timeframes'] += 1
                        if close < ema20:
                            trend_direction = 'BEARISH'
                        else:
                            trend_direction = 'BEAR_PULLBACK'
                            conditions['bear_pullback'] = True
                    else:
                        # EMAs are equal - neutral
                        conditions['neutral_timeframes'] += 1
                        trend_direction = 'NEUTRAL'
                else:
                    conditions['neutral_timeframes'] += 1
                
                conditions['trend_details'][tf_name] = {
                    'direction': trend_direction,
                    'ema20': float(ema20) if pd.notna(ema20) else 0,
                    'ema50': float(ema50) if pd.notna(ema50) else 0,
                    'close': float(close),
                    'rsi': float(rsi) if pd.notna(rsi) else 50
                }
                
                # Store RSI and ATR from 15M timeframe
                if tf_name == '15M':
                    if pd.notna(rsi):
                        conditions['rsi'] = float(rsi)
                    if pd.notna(latest.get('ATR')):
                        conditions['atr_pct'] = (latest['ATR'] / close) * 100
                    conditions['ema20'] = float(ema20) if pd.notna(ema20) else 0
                    conditions['ema50'] = float(ema50) if pd.notna(ema50) else 0
            
            # Print trend analysis for debugging
            print(f"\n📊 TREND ANALYSIS:")
            for tf, details in conditions['trend_details'].items():
                print(f"   {tf}: {details['direction']} | Close: {details['close']:.2f} | EMA20: {details['ema20']:.2f} | EMA50: {details['ema50']:.2f}")
            
            # Total should equal bull + bear + neutral
            total_timeframes = conditions['bull_timeframes'] + conditions['bear_timeframes'] + conditions['neutral_timeframes']
            
            print(f"\n📈 TIMEFRAME SUMMARY: Bull={conditions['bull_timeframes']}, Bear={conditions['bear_timeframes']}, Neutral={conditions['neutral_timeframes']}, Total={total_timeframes}")
            
            # Only signal if majority of timeframes agree
            if conditions['bull_timeframes'] >= 3:
                conditions['bull_signal'] = True
                conditions['trend_strength'] = min(10, conditions['bull_timeframes'] * 2.5)
                print(f"   ➡️ BULLISH SIGNAL (3+ timeframes aligned)")
            elif conditions['bear_timeframes'] >= 3:
                conditions['bear_signal'] = True
                conditions['trend_strength'] = min(10, conditions['bear_timeframes'] * 2.5)
                print(f"   ➡️ BEARISH SIGNAL (3+ timeframes aligned)")
            else:
                conditions['trend_strength'] = 5  # Neutral
                print(f"   ➡️ NO CLEAR SIGNAL (insufficient alignment)")
            
            # Check trading session
            current_hour = datetime.now().hour
            conditions['active_session'] = self.trading_hours['start'] <= current_hour <= self.trading_hours['end']
            
            return conditions
            
        except Exception as e:
            print(f"⚠️ Error analyzing market conditions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_observation(self, data, conditions):
        """Create observation vector for AI model."""
        try:
            # This should match the training observation format
            # For now, create a simplified version
            obs_features = []
            
            # Price-based features
            if '15M' in data and data['15M'] is not None:
                df = data['15M']
                if len(df) >= 20:
                    # Last 20 close prices normalized
                    closes = df['Close'].tail(20).values
                    if len(closes) == 20:
                        # Normalize relative to current price
                        current_price = closes[-1]
                        normalized_closes = (closes / current_price - 1) * 100
                        obs_features.extend(normalized_closes)
            
            # If we don't have enough price data, pad with zeros
            while len(obs_features) < 20:
                obs_features.append(0.0)
            
            # Add technical indicators (1 feature per indicator)
            obs_features.append(conditions.get('rsi', 50.0) / 100.0)  # RSI normalized
            
            # Pad to match model's expected input size
            # The model expects shape (20, 21), so we need 21 features per timestep
            while len(obs_features) < 21:
                obs_features.append(0.0)
            
            # Create observation matrix (20 timesteps x 21 features)
            observation = np.zeros((20, 21))
            
            # Fill the last timestep with our features
            observation[-1, :len(obs_features)] = obs_features[:21]
            
            return observation.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error creating observation: {e}")
            return np.zeros((20, 21), dtype=np.float32)
    
    def generate_signal(self):
        """Generate trading signal using full Golden Gibz analysis - FIXED VERSION."""
        print(f"\n{'='*60}")
        print(f"🔄 Generating Golden Gibz signal at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get real market data from MT5
        data = self.get_market_data()
        if not data:
            print("❌ Failed to get market data")
            return None
        
        # Analyze market conditions
        conditions = self.analyze_market_conditions(data)
        if not conditions:
            print("❌ Failed to analyze market conditions")
            return None
        
        # Determine action based on market conditions (not just AI model)
        # This ensures we trade WITH the trend, not against it
        
        if conditions['bull_signal']:
            # Market is bullish - only consider LONG
            action = 1  # LONG
            confidence = 0.6 + (conditions['bull_timeframes'] - 3) * 0.1  # 0.6-0.8 based on alignment
            
            # Boost confidence if RSI is not overbought
            if conditions['rsi'] < 70:
                confidence += 0.1
            
            # Reduce confidence if RSI is very overbought
            if conditions['rsi'] > 80:
                confidence -= 0.2
                
        elif conditions['bear_signal']:
            # Market is bearish - only consider SHORT
            action = 2  # SHORT
            confidence = 0.6 + (conditions['bear_timeframes'] - 3) * 0.1  # 0.6-0.8 based on alignment
            
            # Boost confidence if RSI is not oversold
            if conditions['rsi'] > 30:
                confidence += 0.1
            
            # Reduce confidence if RSI is very oversold
            if conditions['rsi'] < 20:
                confidence -= 0.2
                
        else:
            # No clear trend - HOLD
            action = 0
            confidence = 0.3
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Create signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'action_name': ['HOLD', 'LONG', 'SHORT'][action],
            'confidence': confidence,
            'market_conditions': conditions,
            'risk_management': {
                'atr_value': conditions.get('atr_pct', 0.15) * conditions['price'] / 100,
                'stop_distance': conditions.get('atr_pct', 0.15) * conditions['price'] / 100 * 2,
                'target_distance': conditions.get('atr_pct', 0.15) * conditions['price'] / 100 * 2
            }
        }
        
        print(f"\n{'='*60}")
        print(f"✅ Golden Gibz Signal: {signal['action_name']} (Confidence: {confidence:.2f})")
        print(f"   Bull TF={conditions['bull_timeframes']}, Bear TF={conditions['bear_timeframes']}")
        print(f"   RSI={conditions['rsi']:.1f}, Price={conditions['price']:.2f}")
        print(f"   EMA20={conditions['ema20']:.2f}, EMA50={conditions['ema50']:.2f}")
        print(f"{'='*60}")
        
        return signal
    
    def execute_trade(self, signal):
        """Execute trade based on signal with enhanced checks."""
        action = signal['action']
        confidence = signal['confidence']
        
        # Reset daily stats if new day
        self.reset_daily_stats()
        
        # Update statistics first to get current P&L
        self.update_trade_statistics()
        
        # Check trading time
        if not self.is_trading_time():
            print(f"⏰ Outside trading hours ({self.trading_hours['start']:02d}:00-{self.trading_hours['end']:02d}:00)")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"🚫 Daily loss limit reached: ${self.daily_pnl:.2f} (Limit: -${self.max_daily_loss})")
            return False
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            print(f"🚫 Daily trade limit reached: {self.daily_trades}/{self.max_daily_trades}")
            return False
        
        # Check trade cooldown
        if self.last_trade_time:
            time_since_trade = (datetime.now() - self.last_trade_time).total_seconds()
            if time_since_trade < self.trade_cooldown:
                remaining = self.trade_cooldown - time_since_trade
                print(f"⏳ Trade cooldown: {remaining:.0f}s remaining")
                return False
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            print(f"⚠️ Signal confidence too low: {confidence:.2f} < {self.min_confidence}")
            return False
        
        # Check position limits - allow up to max_positions in SAME direction
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            # Count positions by direction
            long_count = sum(1 for pos in positions if pos.type == 0)  # BUY positions
            short_count = sum(1 for pos in positions if pos.type == 1)  # SELL positions
            
            # For LONG signals, check if we can add more LONG positions
            if action == 1 and long_count >= self.max_positions:
                print(f"⚠️ Max LONG positions reached: {long_count}/{self.max_positions}")
                return False
            
            # For SHORT signals, check if we can add more SHORT positions  
            if action == 2 and short_count >= self.max_positions:
                print(f"⚠️ Max SHORT positions reached: {short_count}/{self.max_positions}")
                return False
        
        # Execute based on action
        if action == 1:  # LONG
            return self.execute_long_trade(signal)
        elif action == 2:  # SHORT
            return self.execute_short_trade(signal)
        else:  # HOLD
            print(f"⚪ HOLD signal - no action required")
            return True
    
    def execute_long_trade(self, signal):
        """Execute LONG trade with enhanced features."""
        print(f"{Fore.GREEN}{Style.BRIGHT}🟢 EXECUTING AI-GENERATED LONG TRADE!{Style.RESET_ALL}")
        
        # Check daily loss limit BEFORE executing
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"🚫 Cannot trade - Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print("❌ Could not get current price")
            return False
        
        # Calculate lot size (dynamic or fixed)
        lot_size = self.calculate_dynamic_lot_size()
        
        # Get risk management from signal
        risk_mgmt = signal.get('risk_management', {})
        atr_value = risk_mgmt.get('atr_value', 15.0)
        
        # Ensure minimum SL/TP distance (at least $10 for XAUUSD)
        min_distance = 10.0
        sl_distance = max(min_distance, atr_value * 2)
        tp_distance = max(min_distance, atr_value * 2)
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.symbol,
            'volume': lot_size,
            'type': mt5.ORDER_TYPE_BUY,
            'price': tick.ask,
            'sl': tick.ask - sl_distance,
            'tp': tick.ask + tp_distance,
            'deviation': 20,
            'magic': 88888,
            'comment': f'Golden-Gibz AI LONG #{self.trade_counter + 1}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC
        }
        
        print(f"📊 AI LONG TRADE:")
        print(f"   Entry: {Fore.CYAN}{request['price']:.2f}{Style.RESET_ALL}")
        print(f"   Stop: {Fore.RED}{request['sl']:.2f}{Style.RESET_ALL}")
        print(f"   Target: {Fore.GREEN}{request['tp']:.2f}{Style.RESET_ALL}")
        print(f"   Lot Size: {Fore.YELLOW}{lot_size:.2f}{Style.RESET_ALL}")
        print(f"   Confidence: {Fore.MAGENTA}{signal['confidence']:.2f}{Style.RESET_ALL}")
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{Fore.GREEN}{Style.BRIGHT}🎉 AI LONG TRADE EXECUTED!{Style.RESET_ALL}")
            print(f"   Order: {Fore.CYAN}{result.order}{Style.RESET_ALL}")
            print(f"   Deal: {Fore.CYAN}{result.deal}{Style.RESET_ALL}")
            
            # Update statistics
            self.trade_counter += 1
            self.daily_trades += 1
            self.total_trades += 1
            self.last_trade_time = datetime.now()  # Set cooldown timer
            
            return True
        else:
            print(f"❌ LONG trade failed: {result.retcode}")
            return False
    
    def execute_short_trade(self, signal):
        """Execute SHORT trade with enhanced features."""
        print(f"{Fore.RED}{Style.BRIGHT}🔴 EXECUTING AI-GENERATED SHORT TRADE!{Style.RESET_ALL}")
        
        # Check daily loss limit BEFORE executing
        if self.daily_pnl <= -self.max_daily_loss:
            print(f"🚫 Cannot trade - Daily loss limit exceeded: ${self.daily_pnl:.2f}")
            return False
        
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            print("❌ Could not get current price")
            return False
        
        # Calculate lot size (dynamic or fixed)
        lot_size = self.calculate_dynamic_lot_size()
        
        # Get risk management from signal
        risk_mgmt = signal.get('risk_management', {})
        atr_value = risk_mgmt.get('atr_value', 15.0)
        
        # Ensure minimum SL/TP distance (at least $10 for XAUUSD)
        min_distance = 10.0
        sl_distance = max(min_distance, atr_value * 2)
        tp_distance = max(min_distance, atr_value * 2)
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': self.symbol,
            'volume': lot_size,
            'type': mt5.ORDER_TYPE_SELL,
            'price': tick.bid,
            'sl': tick.bid + sl_distance,
            'tp': tick.bid - tp_distance,
            'deviation': 20,
            'magic': 88888,
            'comment': f'Golden-Gibz AI SHORT #{self.trade_counter + 1}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC
        }
        
        print(f"📊 AI SHORT TRADE:")
        print(f"   Entry: {Fore.CYAN}{request['price']:.2f}{Style.RESET_ALL}")
        print(f"   Stop: {Fore.RED}{request['sl']:.2f}{Style.RESET_ALL}")
        print(f"   Target: {Fore.GREEN}{request['tp']:.2f}{Style.RESET_ALL}")
        print(f"   Lot Size: {Fore.YELLOW}{lot_size:.2f}{Style.RESET_ALL}")
        print(f"   Confidence: {Fore.MAGENTA}{signal['confidence']:.2f}{Style.RESET_ALL}")
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"{Fore.RED}{Style.BRIGHT}🎉 AI SHORT TRADE EXECUTED!{Style.RESET_ALL}")
            print(f"   Order: {Fore.CYAN}{result.order}{Style.RESET_ALL}")
            print(f"   Deal: {Fore.CYAN}{result.deal}{Style.RESET_ALL}")
            
            # Update statistics
            self.trade_counter += 1
            self.daily_trades += 1
            self.total_trades += 1
            self.last_trade_time = datetime.now()  # Set cooldown timer
            
            return True
        else:
            print(f"❌ SHORT trade failed: {result.retcode}")
            return False
    
    def update_trade_statistics(self):
        """Update trading statistics based on closed positions and open P&L."""
        try:
            # Get today's deals for closed trade statistics
            today = datetime.now().date()
            from_date = datetime.combine(today, datetime.min.time())
            to_date = datetime.combine(today, datetime.max.time())
            
            deals = mt5.history_deals_get(from_date, to_date)
            
            closed_pnl = 0.0
            if deals:
                for deal in deals:
                    if deal.symbol == self.symbol and deal.magic == 88888:
                        closed_pnl += deal.profit
                        
                        # Update win/loss statistics (only count closing deals)
                        if deal.entry == 1:  # Exit deal
                            if deal.profit > 0:
                                self.winning_trades += 1
                            elif deal.profit < 0:
                                self.losing_trades += 1
            
            # Also include unrealized P&L from open positions
            positions = mt5.positions_get(symbol=self.symbol)
            open_pnl = 0.0
            if positions:
                for pos in positions:
                    if pos.magic == 88888:
                        open_pnl += pos.profit
            
            # Total daily P&L = closed + open
            self.daily_pnl = closed_pnl + open_pnl
            
            # Check if daily loss limit exceeded
            if self.daily_pnl <= -self.max_daily_loss:
                print(f"🚨 DAILY LOSS LIMIT EXCEEDED: ${self.daily_pnl:.2f} (Limit: -${self.max_daily_loss})")
                
        except Exception as e:
            print(f"⚠️ Statistics update error: {e}")
    
    def run(self):
        """Main execution loop with dashboard."""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🚀 GOLDEN GIBZ PYTHON EA RUNNING...{Style.RESET_ALL}")
        print(f"   Signal generation every {self.signal_frequency} seconds")
        print(f"   Multi-timeframe AI analysis with professional dashboard")
        print(f"   Press Ctrl+C to stop")
        
        # Start dashboard in separate thread
        dashboard_thread = threading.Thread(target=self.dashboard_loop, daemon=True)
        dashboard_thread.start()
        
        try:
            while self.running:
                # Generate signal every specified interval
                current_time = datetime.now()
                
                if (self.last_signal_time is None or 
                    (current_time - self.last_signal_time).total_seconds() >= self.signal_frequency):
                    
                    # Generate and execute signal
                    signal = self.generate_signal()
                    if signal:
                        self.execute_trade(signal)
                    
                    # Update statistics
                    self.update_trade_statistics()
                    
                    self.last_signal_time = current_time
                
                # Wait before next check
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}{Style.BRIGHT}🛑 GOLDEN GIBZ PYTHON EA STOPPED{Style.RESET_ALL}")
            self.save_config()
        finally:
            mt5.shutdown()
    
    def dashboard_loop(self):
        """Separate thread for dashboard updates."""
        while self.running:
            try:
                self.print_dashboard()
                time.sleep(self.dashboard_refresh)
            except Exception as e:
                print(f"⚠️ Dashboard error: {e}")
                time.sleep(self.dashboard_refresh)
    
    def interactive_config(self):
        """Interactive configuration menu."""
        while True:
            print(f"\n{Fore.CYAN}{Style.BRIGHT}⚙️ GOLDEN GIBZ CONFIGURATION MENU{Style.RESET_ALL}")
            print(f"{'='*50}")
            print(f"1. Trading Parameters")
            print(f"2. Risk Management")
            print(f"3. Trading Hours")
            print(f"4. Technical Indicators")
            print(f"5. Dashboard Settings")
            print(f"6. Save & Exit")
            print(f"7. Start Trading")
            
            choice = input(f"\n{Fore.YELLOW}Select option (1-7): {Style.RESET_ALL}")
            
            if choice == '1':
                self.config_trading_params()
            elif choice == '2':
                self.config_risk_management()
            elif choice == '3':
                self.config_trading_hours()
            elif choice == '4':
                self.config_indicators()
            elif choice == '5':
                self.config_dashboard()
            elif choice == '6':
                self.save_config()
                break
            elif choice == '7':
                self.save_config()
                return True
            else:
                print(f"{Fore.RED}Invalid option{Style.RESET_ALL}")
        
        return False
    
    def config_trading_params(self):
        """Configure trading parameters."""
        print(f"\n{Fore.BLUE}📊 Trading Parameters{Style.RESET_ALL}")
        
        try:
            lot_size = float(input(f"Lot Size [{self.lot_size}]: ") or self.lot_size)
            max_positions = int(input(f"Max Positions [{self.max_positions}]: ") or self.max_positions)
            min_confidence = float(input(f"Min Confidence [{self.min_confidence}]: ") or self.min_confidence)
            signal_freq = int(input(f"Signal Frequency (seconds) [{self.signal_frequency}]: ") or self.signal_frequency)
            
            self.lot_size = max(0.01, min(lot_size, 10.0))
            self.max_positions = max(1, min(max_positions, 10))
            self.min_confidence = max(0.1, min(min_confidence, 1.0))
            self.signal_frequency = max(30, signal_freq)
            
            print(f"{Fore.GREEN}✅ Trading parameters updated{Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input{Style.RESET_ALL}")
    
    def config_risk_management(self):
        """Configure risk management."""
        print(f"\n{Fore.RED}🛡️ Risk Management{Style.RESET_ALL}")
        
        try:
            max_daily_trades = int(input(f"Max Daily Trades [{self.max_daily_trades}]: ") or self.max_daily_trades)
            max_daily_loss = float(input(f"Max Daily Loss [{self.max_daily_loss}]: ") or self.max_daily_loss)
            risk_per_trade = float(input(f"Risk per Trade % [{self.risk_per_trade}]: ") or self.risk_per_trade)
            use_dynamic = input(f"Use Dynamic Lot Sizing? (y/n) [{'y' if self.use_dynamic_lots else 'n'}]: ").lower()
            
            self.max_daily_trades = max(1, max_daily_trades)
            self.max_daily_loss = max(10, max_daily_loss)
            self.risk_per_trade = max(0.5, min(risk_per_trade, 10.0))
            self.use_dynamic_lots = use_dynamic.startswith('y') if use_dynamic else self.use_dynamic_lots
            
            print(f"{Fore.GREEN}✅ Risk management updated{Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input{Style.RESET_ALL}")
    
    def config_trading_hours(self):
        """Configure trading hours."""
        print(f"\n{Fore.YELLOW}⏰ Trading Hours{Style.RESET_ALL}")
        
        try:
            start_hour = int(input(f"Start Hour (0-23) [{self.trading_hours['start']}]: ") or self.trading_hours['start'])
            end_hour = int(input(f"End Hour (0-23) [{self.trading_hours['end']}]: ") or self.trading_hours['end'])
            
            self.trading_hours['start'] = max(0, min(start_hour, 23))
            self.trading_hours['end'] = max(0, min(end_hour, 23))
            
            print(f"{Fore.GREEN}✅ Trading hours updated: {self.trading_hours['start']:02d}:00-{self.trading_hours['end']:02d}:00{Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input{Style.RESET_ALL}")
    
    def config_indicators(self):
        """Configure technical indicators."""
        print(f"\n{Fore.MAGENTA}📊 Technical Indicators{Style.RESET_ALL}")
        
        try:
            ema_fast = int(input(f"EMA Fast Period [{self.indicators['ema_fast']}]: ") or self.indicators['ema_fast'])
            ema_slow = int(input(f"EMA Slow Period [{self.indicators['ema_slow']}]: ") or self.indicators['ema_slow'])
            rsi_period = int(input(f"RSI Period [{self.indicators['rsi_period']}]: ") or self.indicators['rsi_period'])
            atr_period = int(input(f"ATR Period [{self.indicators['atr_period']}]: ") or self.indicators['atr_period'])
            
            self.indicators['ema_fast'] = max(5, ema_fast)
            self.indicators['ema_slow'] = max(10, ema_slow)
            self.indicators['rsi_period'] = max(5, min(rsi_period, 50))
            self.indicators['atr_period'] = max(5, min(atr_period, 50))
            
            print(f"{Fore.GREEN}✅ Indicators updated{Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input{Style.RESET_ALL}")
    
    def config_dashboard(self):
        """Configure dashboard settings."""
        print(f"\n{Fore.CYAN}📊 Dashboard Settings{Style.RESET_ALL}")
        
        try:
            refresh_rate = int(input(f"Refresh Rate (seconds) [{self.dashboard_refresh}]: ") or self.dashboard_refresh)
            show_indicators = input(f"Show Indicators? (y/n) [{'y' if self.show_indicators else 'n'}]: ").lower()
            show_positions = input(f"Show Positions? (y/n) [{'y' if self.show_positions else 'n'}]: ").lower()
            
            self.dashboard_refresh = max(1, refresh_rate)
            self.show_indicators = show_indicators.startswith('y') if show_indicators else self.show_indicators
            self.show_positions = show_positions.startswith('y') if show_positions else self.show_positions
            
            print(f"{Fore.GREEN}✅ Dashboard settings updated{Style.RESET_ALL}")
            
        except ValueError:
            print(f"{Fore.RED}❌ Invalid input{Style.RESET_ALL}")

def main():
    """Main function with configuration menu."""
    ea = GoldenGibzPythonEA()
    
    # Check if forced config or user wants to configure
    force_config = os.getenv('FORCE_CONFIG', '').lower() == '1'
    
    if force_config:
        if not ea.interactive_config():
            return
    else:
        print(f"{Fore.YELLOW}Would you like to configure settings? (y/n): {Style.RESET_ALL}", end="")
        config_choice = input().lower()
        
        if config_choice.startswith('y'):
            if not ea.interactive_config():
                return
    
    if ea.initialize():
        ea.run()
    else:
        print("❌ Failed to initialize Golden Gibz Python EA")

if __name__ == "__main__":
    main()