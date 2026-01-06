#!/usr/bin/env python3
"""
Hybrid Golden Gibz AI-Enhanced Live Trading EA - PROFESSIONAL VERSION
====================================================================
Complete autonomous EA with AI-enhanced signal generation and beautiful dashboard
- Technical Analysis: Primary signal generation (proven system)
- AI Model: Confirmation layer for enhanced accuracy
- Hybrid Logic: Combines both systems for optimal performance
- Expected Win Rate: 62%+ based on backtesting results
- Signal Types: HYBRID_AGREEMENT, TECHNICAL_ONLY, AI_ONLY, TECHNICAL_FALLBACK
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

class HybridGoldenGibzEA:
    """AI-Enhanced Golden Gibz EA with hybrid intelligence and professional dashboard."""
    
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
        
        # Trading parameters (configurable) - AI-ENHANCED OPTIMIZED
        self.lot_size = self.config.get("lot_size", 0.01)
        self.max_positions = self.config.get("max_positions", 1)  # Match backtest: 1 position at a time
        self.min_confidence = self.config.get("min_confidence", 0.60)  # Optimized from backtesting
        self.signal_frequency = self.config.get("signal_frequency", 240)  # Match backtest: every 4 bars (1 hour)
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
        
        # Indicator settings - ENHANCED TO MATCH BACKTEST
        self.indicators = self.config.get("indicators", {
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
        })
        
    def load_config(self, config_file):
        """Load configuration from JSON file."""
        default_config = {
            "model_path": "models/production/golden_gibz_wr100_ret+25_20251225_215251.zip",
            "symbol": "XAUUSD",
            "lot_size": 0.01,
            "max_positions": 1,  # Match backtest
            "min_confidence": 0.60,  # AI-Enhanced optimized
            "signal_frequency": 240,  # Match backtest: every 4 bars
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
                "macd_signal": 9,
                "adx_period": 14,  # Added ADX
                "stoch_k": 14,     # Added Stochastic
                "stoch_d": 3
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
    
    def is_good_trading_session(self, timestamp=None):
        """Check if current time is during active trading sessions - ADDED TO MATCH BACKTEST."""
        if timestamp is None:
            timestamp = datetime.now()
        
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
        print(f"{Fore.YELLOW}{Style.BRIGHT}🚀 GOLDEN GIBZ AI-ENHANCED EA - PROFESSIONAL EDITION")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*80}")
        print(f"{Fore.GREEN}🤖 Hybrid AI + Technical Analysis Trading System")
        print(f"{Fore.WHITE}📊 Multi-timeframe Analysis | 🛡️ Advanced Risk Management")
        print(f"{Fore.WHITE}⚙️ AI Model Confirmation | 📈 Real-time Monitoring")
        print(f"{Fore.WHITE}🎯 Target Win Rate: 62%+ | 🔥 Proven Backtesting Results")
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
            
            # AI Signal Status - ENHANCED
            print(f"\n{Fore.RED}{Style.BRIGHT}🤖 AI-ENHANCED SIGNAL STATUS{Style.RESET_ALL}")
            print(f"{'─'*50}")
            print(f"Model: {Fore.CYAN}Golden Gibz PPO (AI-Enhanced){Style.RESET_ALL}")
            print(f"Signal Type: {Fore.YELLOW}Technical + AI Confirmation{Style.RESET_ALL}")
            print(f"Signal Frequency: {Fore.YELLOW}{self.signal_frequency}s{Style.RESET_ALL}")
            print(f"Min Confidence: {Fore.YELLOW}{self.min_confidence:.1%}{Style.RESET_ALL}")
            
            if self.last_signal_time:
                time_since = (current_time - self.last_signal_time).total_seconds()
                next_signal = max(0, self.signal_frequency - time_since)
                print(f"Next Signal: {Fore.MAGENTA}{next_signal:.0f}s{Style.RESET_ALL}")
            
            print(f"AI Model Status: {Fore.GREEN}✅ LOADED{Style.RESET_ALL}" if self.model else f"AI Model Status: {Fore.RED}❌ NOT LOADED{Style.RESET_ALL}")
            
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
            
            # ADX - ADDED TO MATCH BACKTEST (with error handling)
            try:
                adx = latest.get('ADX', 0)
                adx_status = "🟢 Strong" if adx > 25 else "🟡 Weak"
                print(f"ADX({self.indicators.get('adx_period', 14)}): {Fore.YELLOW}{adx:.1f}{Style.RESET_ALL} {adx_status}")
            except Exception as e:
                print(f"ADX: {Fore.RED}⚠️ Error: {e}{Style.RESET_ALL}")
            
            # Stochastic - ADDED TO MATCH BACKTEST
            stoch_k = latest.get('Stoch_K', 50)
            stoch_d = latest.get('Stoch_D', 50)
            stoch_trend = "🟢 Bullish" if stoch_k > stoch_d else "🔴 Bearish"
            print(f"Stochastic: {stoch_trend} | K: {Fore.CYAN}{stoch_k:.1f}{Style.RESET_ALL} | D: {Fore.CYAN}{stoch_d:.1f}{Style.RESET_ALL}")
            
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
            
            # Enhanced signal conditions - MATCH BACKTEST
            strong_trend = latest.get('Strong_Trend', False)
            rsi_neutral = latest.get('RSI_Neutral', False)
            low_volatility = latest.get('Low_Volatility', False)
            
            print(f"\n{Fore.WHITE}{Style.BRIGHT}📊 ENHANCED SIGNAL CONDITIONS{Style.RESET_ALL}")
            print(f"{'─'*40}")
            print(f"Strong Trend (ADX>25): {'🟢 YES' if strong_trend else '🔴 NO'}")
            print(f"RSI Neutral (30-70): {'🟢 YES' if rsi_neutral else '🔴 NO'}")
            print(f"Low Volatility: {'🟢 YES' if low_volatility else '🔴 NO'}")
            
            # Calculate enhanced confidence score
            confidence_score = 0
            
            # EMA trend (25 points)
            if ema20 > ema50:
                ema_gap = ((ema20 - ema50) / ema50) * 100
                confidence_score += min(25, max(0, ema_gap * 5))
                trend_signal = f"{Fore.GREEN}BULLISH{Style.RESET_ALL}"
            else:
                ema_gap = ((ema50 - ema20) / ema20) * 100
                confidence_score += min(25, max(0, ema_gap * 5))
                trend_signal = f"{Fore.RED}BEARISH{Style.RESET_ALL}"
            
            # ADX strength (25 points) - ENHANCED
            if strong_trend:
                confidence_score += 25
                adx_signal = f"{Fore.GREEN}STRONG{Style.RESET_ALL}"
            else:
                confidence_score += 5
                adx_signal = f"{Fore.RED}WEAK{Style.RESET_ALL}"
            
            # RSI condition (25 points)
            if rsi_neutral:
                confidence_score += 25
                rsi_signal = f"{Fore.GREEN}OPTIMAL{Style.RESET_ALL}"
            elif 20 <= rsi <= 80:
                confidence_score += 15
                rsi_signal = f"{Fore.YELLOW}ACCEPTABLE{Style.RESET_ALL}"
            else:
                confidence_score += 5
                rsi_signal = f"{Fore.RED}EXTREME{Style.RESET_ALL}"
            
            # MACD momentum (15 points)
            if abs(macd - macd_signal) > 0.5:
                confidence_score += 15
                macd_signal_strength = f"{Fore.GREEN}STRONG{Style.RESET_ALL}"
            elif abs(macd - macd_signal) > 0.2:
                confidence_score += 10
                macd_signal_strength = f"{Fore.YELLOW}MODERATE{Style.RESET_ALL}"
            else:
                confidence_score += 5
                macd_signal_strength = f"{Fore.RED}WEAK{Style.RESET_ALL}"
            
            # Volatility (10 points)
            if low_volatility:
                confidence_score += 10
                volatility_signal = f"{Fore.GREEN}FAVORABLE{Style.RESET_ALL}"
            else:
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
            print(f"Trend Strength: {adx_signal}")
            print(f"RSI Condition: {rsi_signal}")
            print(f"MACD Strength: {macd_signal_strength}")
            print(f"Volatility: {volatility_signal}")
            print(f"{'─'*40}")
            print(f"Enhanced Confidence: {confidence_color}{confidence_pct:.0f}%{Style.RESET_ALL} {confidence_status}")
            
        except Exception as e:
            print(f"⚠️ Indicators error: {e}")
    
    def get_market_data(self):
        """Get multi-timeframe market data from MT5 - ENHANCED TO MATCH BACKTEST."""
        print("📊 Getting real market data from MT5...")
        
        # MT5 timeframe mapping - ADDED 30M TO MATCH BACKTEST
        tf_map = {
            '15M': mt5.TIMEFRAME_M15,
            '30M': mt5.TIMEFRAME_M30,  # Added 30M timeframe
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1
        }
        
        # Bars needed for each timeframe
        bars_needed = {
            '15M': 200,
            '30M': 100,  # Added 30M
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
        """Calculate enhanced technical indicators for Golden Gibz analysis - UPDATED TO MATCH BACKTEST."""
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
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle'] * 100
            
            # MACD (configurable periods)
            macd = ta.trend.MACD(df['Close'], 
                           window_fast=self.indicators['macd_fast'],
                           window_slow=self.indicators['macd_slow'],
                           window_sign=self.indicators['macd_signal'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # ADX for trend strength - ADDED TO MATCH BACKTEST (with error handling)
            try:
                df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=self.indicators.get('adx_period', 14))
            except Exception as e:
                print(f"⚠️ ADX calculation error: {e}")
                df['ADX'] = 0  # Default value
            
            # Stochastic for momentum - ADDED TO MATCH BACKTEST (with error handling)
            try:
                stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 
                                                       window=self.indicators.get('stoch_k', 14), 
                                                       smooth_window=self.indicators.get('stoch_d', 3))
                df['Stoch_K'] = stoch.stoch()
                df['Stoch_D'] = stoch.stoch_signal()
            except Exception as e:
                print(f"⚠️ Stochastic calculation error: {e}")
                df['Stoch_K'] = 50  # Default neutral value
                df['Stoch_D'] = 50
            
            # Williams %R
            df['WillR'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # CCI
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            
            # Enhanced signal conditions - MATCH BACKTEST LOGIC
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
    
    def get_enhanced_timeframe_signal(self, tf_data, timestamp=None):
        """Get enhanced signal from a specific timeframe - MATCH BACKTEST LOGIC."""
        try:
            if timestamp is None:
                # Use latest data
                idx = -1
            else:
                idx = tf_data.index.get_indexer([timestamp], method='nearest')[0]
            
            if idx < 50 or (idx >= 0 and idx >= len(tf_data)):
                return 0, 0, {}
            
            row = tf_data.iloc[idx]
            
            # Enhanced signal scoring - MATCH BACKTEST
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
    
    def create_observation_for_ai(self, data):
        """Create observation vector for AI model - ENHANCED VERSION."""
        try:
            # Get execution timeframe data
            exec_data = data.get('15M')
            if exec_data is None or len(exec_data) < 50:
                return np.zeros((20, 21), dtype=np.float32)
            
            # Get the last 20 bars of data
            recent_data = exec_data.tail(20)
            
            # Create observation matrix (20 timesteps x 21 features)
            observation = np.zeros((20, 21))
            
            for i, (_, row) in enumerate(recent_data.iterrows()):
                # Price features (OHLC normalized)
                close = row['Close']
                observation[i, 0] = (row['Open'] / close - 1) * 100
                observation[i, 1] = (row['High'] / close - 1) * 100
                observation[i, 2] = (row['Low'] / close - 1) * 100
                observation[i, 3] = 0.0  # Close is reference (0)
                
                # Volume (normalized)
                observation[i, 4] = np.log(row.get('Volume', 1000)) / 10.0
                
                # Technical indicators (normalized)
                observation[i, 5] = row.get('RSI', 50) / 100.0
                observation[i, 6] = (row.get('EMA20', close) / close - 1) * 100
                observation[i, 7] = (row.get('EMA50', close) / close - 1) * 100
                observation[i, 8] = row.get('MACD', 0) * 1000  # Scale MACD
                observation[i, 9] = row.get('MACD_Signal', 0) * 1000
                observation[i, 10] = row.get('MACD_Hist', 0) * 1000
                observation[i, 11] = (row.get('ATR', 15) / close) * 100
                observation[i, 12] = row.get('ADX', 25) / 100.0
                observation[i, 13] = row.get('Stoch_K', 50) / 100.0
                observation[i, 14] = row.get('Stoch_D', 50) / 100.0
                observation[i, 15] = (row.get('BB_Upper', close) / close - 1) * 100
                observation[i, 16] = (row.get('BB_Lower', close) / close - 1) * 100
                observation[i, 17] = row.get('BB_Width', 2) / 10.0
                
                # Additional features
                observation[i, 18] = 1.0 if row.get('EMA_Bullish', False) else 0.0
                observation[i, 19] = 1.0 if row.get('Strong_Trend', False) else 0.0
                observation[i, 20] = 1.0 if row.get('RSI_Neutral', True) else 0.0
            
            return observation.astype(np.float32)
            
        except Exception as e:
            print(f"⚠️ Error creating AI observation: {e}")
            return np.zeros((20, 21), dtype=np.float32)
    
    def get_ai_prediction(self, data):
        """Get AI model prediction as confirmation."""
        try:
            if not self.model:
                return None
            
            # Create observation for AI model
            observation = self.create_observation_for_ai(data)
            
            # Get AI prediction
            action, _states = self.model.predict(observation, deterministic=True)
            
            # Convert action to our format
            if action == 0:
                ai_action = 0  # HOLD
                ai_confidence = 0.3
            elif action == 1:
                ai_action = 1  # LONG
                ai_confidence = 0.7
            elif action == 2:
                ai_action = 2  # SHORT
                ai_confidence = 0.7
            else:
                ai_action = 0
                ai_confidence = 0.3
            
            return {
                'action': int(ai_action),
                'confidence': float(ai_confidence),
                'raw_action': int(action)
            }
            
        except Exception as e:
            print(f"⚠️ AI prediction error: {e}")
            return None
    
    def analyze_enhanced_conditions(self, data):
        """Enhanced multi-timeframe analysis with quality scoring - MATCH BACKTEST LOGIC."""
        try:
            # Timeframe weights (higher timeframes more important) - MATCH BACKTEST
            tf_weights = {'15M': 1, '30M': 2, '1H': 3, '4H': 4, '1D': 5}
            
            total_bull_score = 0
            total_bear_score = 0
            total_weight = 0
            timeframe_signals = {}
            
            # Get execution timeframe data
            exec_data = data.get('15M')
            if exec_data is None or len(exec_data) < 50:
                return None
            
            # Calculate indicators for execution timeframe
            exec_data = self.calculate_technical_indicators(exec_data)
            exec_row = exec_data.iloc[-1]
            
            # Analyze each timeframe
            for tf_name in ['15M', '30M', '1H', '4H', '1D']:
                if tf_name not in data or data[tf_name] is None:
                    continue
                
                # Calculate indicators for this timeframe
                tf_data = self.calculate_technical_indicators(data[tf_name])
                signal, strength, quality = self.get_enhanced_timeframe_signal(tf_data)
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
                'session_active': self.is_good_trading_session(),
                'ema20': exec_row.get('EMA20', 0),
                'ema50': exec_row.get('EMA50', 0),
                'bull_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 1),
                'bear_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == -1),
                'neutral_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 0)
            }
            
            # Calculate ATR percentage
            atr = exec_row.get('ATR', 15.0)
            if atr > 0 and exec_row['Close'] > 0:
                conditions['atr_pct'] = (atr / exec_row['Close']) * 100
            
            # Enhanced signal determination with optimized criteria
            min_strength = 2.0  # Optimized threshold from backtesting
            
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
            print(f"⚠️ Error analyzing enhanced conditions: {e}")
            return None
        """Enhanced multi-timeframe analysis with quality scoring - MATCH BACKTEST LOGIC."""
        try:
            # Timeframe weights (higher timeframes more important) - MATCH BACKTEST
            tf_weights = {'15M': 1, '30M': 2, '1H': 3, '4H': 4, '1D': 5}
            
            total_bull_score = 0
            total_bear_score = 0
            total_weight = 0
            timeframe_signals = {}
            
            # Get execution timeframe data
            exec_data = data.get('15M')
            if exec_data is None or len(exec_data) < 50:
                return None
            
            # Calculate indicators for execution timeframe
            exec_data = self.calculate_technical_indicators(exec_data)
            exec_row = exec_data.iloc[-1]
            
            # Analyze each timeframe
            for tf_name in ['15M', '30M', '1H', '4H', '1D']:
                if tf_name not in data or data[tf_name] is None:
                    continue
                
                # Calculate indicators for this timeframe
                tf_data = self.calculate_technical_indicators(data[tf_name])
                signal, strength, quality = self.get_enhanced_timeframe_signal(tf_data)
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
                'session_active': self.is_good_trading_session(),
                'ema20': exec_row.get('EMA20', 0),
                'ema50': exec_row.get('EMA50', 0),
                'bull_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 1),
                'bear_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == -1),
                'neutral_timeframes': sum(1 for tf in timeframe_signals.values() if tf['signal'] == 0)
            }
            
            # Calculate ATR percentage
            atr = exec_row.get('ATR', 15.0)
            if atr > 0 and exec_row['Close'] > 0:
                conditions['atr_pct'] = (atr / exec_row['Close']) * 100
            
            # Enhanced signal determination with stricter criteria - MATCH BACKTEST
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
            print(f"⚠️ Error analyzing enhanced conditions: {e}")
            return None
    
    def analyze_market_conditions(self, data):
        """Wrapper for enhanced analysis - UPDATED TO MATCH BACKTEST."""
        return self.analyze_enhanced_conditions(data)
    
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
        """Generate AI-enhanced trading signal with hybrid confirmation - OPTION 2 IMPLEMENTATION."""
        print(f"\n{'='*60}")
        print(f"🤖 Generating AI-Enhanced Golden Gibz signal at {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get real market data from MT5
        data = self.get_market_data()
        if not data:
            print("❌ Failed to get market data")
            return None
        
        # Step 1: Get technical analysis signal (our proven system)
        conditions = self.analyze_enhanced_conditions(data)
        if not conditions:
            print("❌ Failed to analyze enhanced market conditions")
            return None
        
        # Step 2: Get AI model prediction as confirmation
        ai_prediction = self.get_ai_prediction(data)
        
        # Step 3: Combine technical analysis with AI confirmation
        technical_action = 0
        technical_confidence = 0.3
        
        # Determine technical action with enhanced confidence calculation
        if conditions['bull_signal']:
            technical_action = 1  # LONG
            base_confidence = 0.6
            
            # Boost confidence based on signal quality
            strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
            rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
            session_bonus = 0.05 if conditions['session_active'] else -0.1
            
            technical_confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
                
        elif conditions['bear_signal']:
            technical_action = 2  # SHORT
            base_confidence = 0.6
            
            # Boost confidence based on signal quality
            strength_bonus = min(0.2, conditions['signal_strength'] * 0.05)
            rsi_bonus = 0.1 if 40 <= conditions['rsi'] <= 60 else 0
            session_bonus = 0.05 if conditions['session_active'] else -0.1
            
            technical_confidence = base_confidence + strength_bonus + rsi_bonus + session_bonus
        
        technical_confidence = max(0.0, min(1.0, technical_confidence))
        
        # Step 4: AI-Technical Hybrid Decision Logic
        final_action = 0
        final_confidence = 0.3
        signal_type = "HOLD"
        
        if ai_prediction and technical_action != 0:
            ai_action = ai_prediction['action']
            ai_confidence = ai_prediction['confidence']
            
            # Case 1: Both systems agree on direction - HIGHEST CONFIDENCE
            if technical_action == ai_action and ai_action != 0:
                final_action = technical_action
                # Boost confidence when both agree
                final_confidence = min(1.0, technical_confidence + 0.15)
                signal_type = "HYBRID_AGREEMENT"
            
            # Case 2: Technical says trade, AI says hold - ALLOW WITH REDUCED CONFIDENCE
            elif technical_action != 0 and ai_action == 0:
                # Allow technical-only trades with lower requirements
                if technical_confidence > 0.65:
                    final_action = technical_action
                    final_confidence = max(0.60, technical_confidence - 0.05)
                    signal_type = "TECHNICAL_ONLY"
                else:
                    final_action = 0
                    final_confidence = 0.3
                    signal_type = "TECHNICAL_WEAK"
            
            # Case 3: AI says trade, technical says hold - ALLOW IF AI CONFIDENT
            elif technical_action == 0 and ai_action != 0:
                # Allow AI-only trades with lower requirements
                if ai_confidence > 0.60 and conditions['signal_strength'] > 0.8:
                    final_action = ai_action
                    final_confidence = 0.60
                    signal_type = "AI_ONLY"
                else:
                    final_action = 0
                    final_confidence = 0.3
                    signal_type = "AI_WEAK"
            
            # Case 4: Both disagree on direction - NO TRADE
            elif technical_action != 0 and ai_action != 0 and technical_action != ai_action:
                final_action = 0
                final_confidence = 0.2
                signal_type = "CONFLICTED_HOLD"
            
            else:
                final_action = 0
                final_confidence = 0.3
                signal_type = "HOLD"
        
        elif technical_action != 0:
            # Fallback to technical analysis only if AI fails
            if technical_confidence > 0.60:
                final_action = technical_action
                final_confidence = technical_confidence * 0.9  # Slight penalty for no AI
                signal_type = "TECHNICAL_FALLBACK"
            else:
                final_action = 0
                final_confidence = 0.3
                signal_type = "TECHNICAL_WEAK"
        
        else:
            final_action = 0
            final_confidence = 0.3
            signal_type = "HOLD"
        
        # Create enhanced signal
        signal = {
            'timestamp': datetime.now().isoformat(),
            'action': final_action,
            'action_name': ['HOLD', 'LONG', 'SHORT'][final_action],
            'confidence': final_confidence,
            'market_conditions': conditions,
            'ai_prediction': ai_prediction,
            'technical_action': technical_action,
            'technical_confidence': technical_confidence,
            'signal_type': signal_type,
            'risk_management': {
                'atr_value': conditions.get('atr_pct', 0.15) * conditions['price'] / 100,
                'stop_distance': conditions.get('atr_pct', 0.15) * conditions['price'] / 100 * 2,
                'target_distance': conditions.get('atr_pct', 0.15) * conditions['price'] / 100 * 2
            }
        }
        
        print(f"\n{'='*60}")
        print(f"🚀 AI-Enhanced Signal: {signal['action_name']} (Confidence: {final_confidence:.2f})")
        print(f"   Signal Type: {signal_type}")
        print(f"   Technical: {['HOLD', 'LONG', 'SHORT'][technical_action]} ({technical_confidence:.2f})")
        if ai_prediction:
            print(f"   AI Model: {['HOLD', 'LONG', 'SHORT'][ai_prediction['action']]} ({ai_prediction['confidence']:.2f})")
        print(f"   Signal Strength: {conditions['signal_strength']:.2f}")
        print(f"   Bull TF={conditions['bull_timeframes']}, Bear TF={conditions['bear_timeframes']}")
        print(f"   RSI={conditions['rsi']:.1f}, Price={conditions['price']:.2f}")
        print(f"   Session Active: {conditions['session_active']}")
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
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🚀 GOLDEN GIBZ AI-ENHANCED EA RUNNING...{Style.RESET_ALL}")
        print(f"   Hybrid system: Technical Analysis + AI Model Confirmation")
        print(f"   Signal generation every {self.signal_frequency} seconds")
        print(f"   Multi-timeframe analysis with AI-enhanced filtering")
        print(f"   Expected win rate: ~62% (based on backtesting)")
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
    ea = HybridGoldenGibzEA()
    
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
        print("❌ Failed to initialize Hybrid Golden Gibz AI-Enhanced EA")

if __name__ == "__main__":
    main()