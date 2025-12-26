#!/usr/bin/env python3
"""
Golden-Gibz Signal Generator - Hybrid System Phase 1
===================================================
Generates trading signals from trained Golden-Gibz model for MT5 EA consumption
"""

import numpy as np
import pandas as pd
import pandas_ta as pta
import MetaTrader5 as mt5
import json
import os
import time
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import warnings
warnings.filterwarnings('ignore')


class MT5DataProvider:
    """Fetches live data from MT5."""
    
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.connected = False
        
    def connect(self):
        """Connect to MT5."""
        if not mt5.initialize():
            print(f"❌ MT5 initialization failed: {mt5.last_error()}")
            return False
        
        account_info = mt5.account_info()
        if account_info is None:
            print("❌ Failed to get account info")
            return False
            
        print(f"✅ Connected to MT5 - Account: {account_info.login}")
        
        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            print(f"❌ Failed to select symbol {self.symbol}: {mt5.last_error()}")
            return False
        
        print(f"✅ Symbol {self.symbol} selected")
        
        self.connected = True
        return True
    
    def get_timeframe_data(self, timeframe_str, bars=100):
        """Get data for specific timeframe."""
        if not self.connected:
            return None
            
        # MT5 timeframe mapping
        tf_map = {
            '15m': mt5.TIMEFRAME_M15,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1
        }
        
        if timeframe_str not in tf_map:
            return None
            
        rates = mt5.copy_rates_from_pos(self.symbol, tf_map[timeframe_str], 0, bars)
        
        if rates is None or len(rates) == 0:
            print(f"❌ No data for {timeframe_str}")
            return None
            
        df = pd.DataFrame(rates)
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('Date', inplace=True)
        
        # Rename columns to match training data
        df.rename(columns={
            'open': 'Open',
            'high': 'High', 
            'low': 'Low',
            'close': 'Close',
            'tick_volume': 'Volume'
        }, inplace=True)
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def get_all_timeframes(self):
        """Get data for all required timeframes."""
        timeframes = ['15m', '1h', '4h', '1d']
        data = {}
        
        for tf in timeframes:
            bars_needed = 200 if tf == '15m' else 100  # More bars for main timeframe
            df = self.get_timeframe_data(tf, bars_needed)
            if df is not None:
                data[tf] = df
                print(f"✅ {tf.upper()}: {len(df)} bars")
            else:
                print(f"❌ {tf.upper()}: Failed to get data")
                data[tf] = None
                
        return data
    
    def disconnect(self):
        """Disconnect from MT5."""
        mt5.shutdown()
        self.connected = False


def add_live_trend_signals(df_15m, df_1h, df_4h, df_1d):
    """Add trend signals to live data - same logic as training."""
    
    # === 15M INDICATORS ===
    df_15m['ema20'] = pta.ema(df_15m['Close'], 20)
    df_15m['ema50'] = pta.ema(df_15m['Close'], 50)
    df_15m['rsi'] = pta.rsi(df_15m['Close'], 14)
    df_15m['atr'] = pta.atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14)
    df_15m['atr_pct'] = (df_15m['atr'] / df_15m['Close']) * 100
    
    # === HIGHER TIMEFRAME TRENDS ===
    htf_trends = {}
    
    for name, htf_df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if htf_df is not None and len(htf_df) > 50:
            htf_df['ema20'] = pta.ema(htf_df['Close'], 20)
            htf_df['ema50'] = pta.ema(htf_df['Close'], 50)
            
            # Current trend direction
            current_trend = 1 if htf_df['ema20'].iloc[-1] > htf_df['ema50'].iloc[-1] else -1
            
            # Trend strength
            trend_bars = 0
            for i in range(min(20, len(htf_df))):
                if htf_df['ema20'].iloc[-(i+1)] > htf_df['ema50'].iloc[-(i+1)]:
                    if current_trend == 1:
                        trend_bars += 1
                    else:
                        break
                else:
                    if current_trend == -1:
                        trend_bars += 1
                    else:
                        break
            
            # Strength score
            if trend_bars >= 15:
                strength = 3
            elif trend_bars >= 10:
                strength = 2
            elif trend_bars >= 5:
                strength = 1
            else:
                strength = 0
            
            htf_trends[name] = {
                'direction': current_trend,
                'strength': strength,
                'bars': trend_bars
            }
        else:
            htf_trends[name] = {'direction': 0, 'strength': 0, 'bars': 0}
    
    # Apply HTF trends to 15M data
    for tf in ['1h', '4h', '1d']:
        df_15m[f'{tf}_trend'] = htf_trends[tf]['direction']
        df_15m[f'{tf}_strength'] = htf_trends[tf]['strength']
        df_15m[f'{tf}_bars'] = htf_trends[tf]['bars']
    
    # === TREND ALIGNMENT ===
    df_15m['bull_timeframes'] = (
        (df_15m['ema20'] > df_15m['ema50']).astype(int) +
        (df_15m['1h_trend'] == 1).astype(int) +
        (df_15m['4h_trend'] == 1).astype(int) +
        (df_15m['1d_trend'] == 1).astype(int)
    )
    
    df_15m['bear_timeframes'] = (
        (df_15m['ema20'] < df_15m['ema50']).astype(int) +
        (df_15m['1h_trend'] == -1).astype(int) +
        (df_15m['4h_trend'] == -1).astype(int) +
        (df_15m['1d_trend'] == -1).astype(int)
    )
    
    # === TREND STRENGTH SCORE ===
    df_15m['trend_strength_score'] = (
        df_15m['1h_strength'] + 
        df_15m['4h_strength'] + 
        df_15m['1d_strength']
    )
    
    # === ENTRY SIGNALS ===
    df_15m['bull_signal'] = (
        (df_15m['bull_timeframes'] >= 3) &
        (df_15m['trend_strength_score'] >= 3) &
        (df_15m['rsi'] < 70) &
        (df_15m['Close'] > df_15m['ema20'])
    ).astype(int)
    
    df_15m['bear_signal'] = (
        (df_15m['bear_timeframes'] >= 3) &
        (df_15m['trend_strength_score'] >= 3) &
        (df_15m['rsi'] > 30) &
        (df_15m['Close'] < df_15m['ema20'])
    ).astype(int)
    
    # === PULLBACK SIGNALS ===
    df_15m['bull_pullback'] = (
        (df_15m['bull_timeframes'] >= 2) &
        (df_15m['trend_strength_score'] >= 2) &
        (df_15m['Close'] <= df_15m['ema20'] * 1.001) &
        (df_15m['Close'] >= df_15m['ema20'] * 0.999) &
        (df_15m['rsi'] < 60)
    ).astype(int)
    
    df_15m['bear_pullback'] = (
        (df_15m['bear_timeframes'] >= 2) &
        (df_15m['trend_strength_score'] >= 2) &
        (df_15m['Close'] >= df_15m['ema20'] * 0.999) &
        (df_15m['Close'] <= df_15m['ema20'] * 1.001) &
        (df_15m['rsi'] > 40)
    ).astype(int)
    
    # === SESSION FILTER ===
    df_15m['hour'] = df_15m.index.hour
    df_15m['active_session'] = ((df_15m['hour'] >= 8) & (df_15m['hour'] <= 17)).astype(int)
    
    return df_15m


class GoldenGibzSignalGenerator:
    """Generates signals using trained Golden-Gibz model."""
    
    def __init__(self, model_path, normalization_path=None):
        self.model_path = model_path
        self.normalization_path = normalization_path
        self.model = None
        self.vec_env = None
        self.window_size = 20
        
        # Features must match training exactly
        self.features = [
            'Close', 'ema20', 'ema50', 'rsi', 'atr_pct',
            'bull_timeframes', 'bear_timeframes', 'trend_strength_score',
            '1h_trend', '4h_trend', '1d_trend',
            '1h_strength', '4h_strength', '1d_strength',
            'bull_signal', 'bear_signal', 'bull_pullback', 'bear_pullback',
            'active_session'
        ]
        
    def load_model(self):
        """Load trained Golden-Gibz model."""
        try:
            self.model = PPO.load(self.model_path)
            
            # Load normalization if available
            if self.normalization_path and os.path.exists(self.normalization_path):
                # Create dummy env for normalization
                dummy_env = DummyVecEnv([lambda: None])
                self.vec_env = VecNormalize.load(self.normalization_path, dummy_env)
                print("✅ Golden-Gibz model and normalization loaded")
            else:
                print("✅ Golden-Gibz model loaded (no normalization)")
                
            return True
        except Exception as e:
            print(f"❌ Failed to load Golden-Gibz model: {e}")
            return False
    
    def normalize_data(self, df, base_price=None):
        """Normalize data same as training."""
        df = df.copy()
        
        if base_price is None:
            base_price = df['Close'].iloc[-50:].mean()  # Use recent average
        
        # Normalize prices
        for col in ['Close', 'ema20', 'ema50']:
            if col in df.columns:
                df[col] = (df[col] / base_price - 1) * 100
        
        # Normalize RSI
        if 'rsi' in df.columns:
            df['rsi'] = (df['rsi'] - 50) / 25
        
        # Normalize ATR percentage
        if 'atr_pct' in df.columns:
            df['atr_pct'] = df['atr_pct'].clip(0, df['atr_pct'].quantile(0.95))
            max_atr = df['atr_pct'].max()
            if max_atr > 0:
                df['atr_pct'] = df['atr_pct'] / max_atr
        
        return df
    
    def prepare_observation(self, df):
        """Prepare observation for Golden-Gibz model."""
        # Get last window_size rows
        obs_data = df[self.features].tail(self.window_size).values
        
        # Pad if needed
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), len(self.features)))
            obs_data = np.vstack([pad, obs_data])
        
        # Add position and win rate (set to neutral for live trading)
        extra = np.zeros((self.window_size, 2))
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        obs = np.nan_to_num(np.clip(obs, -5, 5))
        
        return obs
    
    def generate_signal(self, df):
        """Generate Golden-Gibz trading signal from processed data."""
        if self.model is None:
            return None
            
        try:
            # Normalize data
            df_norm = self.normalize_data(df)
            
            # Prepare observation
            obs = self.prepare_observation(df_norm)
            
            # Get prediction
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Get current market conditions
            current_row = df.iloc[-1]
            
            signal_data = {
                'timestamp': datetime.now().isoformat(),
                'action': int(action),
                'action_name': ['HOLD', 'LONG', 'SHORT'][action],
                'confidence': self._calculate_confidence(current_row),
                'market_conditions': {
                    'price': float(current_row['Close']),
                    'bull_timeframes': int(current_row['bull_timeframes']),
                    'bear_timeframes': int(current_row['bear_timeframes']),
                    'trend_strength': int(current_row['trend_strength_score']),
                    'rsi': float(current_row['rsi']),
                    'atr_pct': float(current_row['atr_pct']),
                    'bull_signal': bool(current_row['bull_signal']),
                    'bear_signal': bool(current_row['bear_signal']),
                    'bull_pullback': bool(current_row['bull_pullback']),
                    'bear_pullback': bool(current_row['bear_pullback']),
                    'active_session': bool(current_row['active_session'])
                },
                'risk_management': {
                    'atr_value': float(current_row['atr']),
                    'stop_distance': float(current_row['atr'] * 2.0),
                    'target_distance': float(current_row['atr'] * 2.0)
                }
            }
            
            return signal_data
            
        except Exception as e:
            print(f"❌ Golden-Gibz signal generation error: {e}")
            return None
    
    def _calculate_confidence(self, row):
        """Calculate signal confidence based on market conditions."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for strong trend alignment
        if row['bull_timeframes'] >= 3 or row['bear_timeframes'] >= 3:
            confidence += 0.2
        
        # Higher confidence for strong trend strength
        if row['trend_strength_score'] >= 4:
            confidence += 0.2
        
        # Higher confidence during active sessions
        if row['active_session']:
            confidence += 0.1
        
        # Lower confidence for extreme RSI
        if row['rsi'] > 80 or row['rsi'] < 20:
            confidence -= 0.2
        
        return min(1.0, max(0.0, confidence))


def main():
    """Main Golden-Gibz signal generation loop."""
    
    # Configuration
    SYMBOL = "XAUUSD"
    MODEL_PATH = "../models/production/golden_gibz_wr100_ret+25_20251225_215251"  # Best model: 100% WR, +25% return
    
    # MT5 Files directory paths (write to all possible locations for compatibility)
    import os
    mt5_appdata_path = os.path.join(os.getenv('APPDATA'), 'MetaQuotes', 'Terminal', '29E91DA909EB4475AB204481D1C2CE7D', 'MQL5', 'Files')
    mt5_documents_path = os.path.join(os.path.expanduser('~'), 'Documents', 'MT5', 'MQL5', 'Files')
    mt5_program_path = r"C:\Program Files\Tickmill MT5 Terminal\MQL5\Files"
    
    # Ensure all directories exist
    os.makedirs(mt5_appdata_path, exist_ok=True)
    os.makedirs(mt5_documents_path, exist_ok=True)
    os.makedirs(mt5_program_path, exist_ok=True)
    
    SIGNAL_FILES = [
        os.path.join(mt5_appdata_path, "signals.json"),
        os.path.join(mt5_documents_path, "signals.json"),
        os.path.join(mt5_program_path, "signals.json")
    ]
    LOG_FILE = "../logs/golden_gibz_signals.log"
    
    # Ensure directories exist (already done above)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    print("🎯 Golden-Gibz Signal Generator Starting...")
    print(f"Symbol: {SYMBOL}")
    print(f"Model: {MODEL_PATH}")
    print(f"Signal Files: {len(SIGNAL_FILES)} locations")
    for i, path in enumerate(SIGNAL_FILES, 1):
        print(f"  {i}. {path}")
    
    # Initialize components
    mt5_data = MT5DataProvider(SYMBOL)
    signal_gen = GoldenGibzSignalGenerator(MODEL_PATH)
    
    # Connect to MT5
    if not mt5_data.connect():
        print("❌ Cannot connect to MT5")
        return
    
    # Load model
    if not signal_gen.load_model():
        print("❌ Cannot load Golden-Gibz model")
        return
    
    print("✅ All Golden-Gibz components initialized")
    
    # Main loop
    last_signal_time = None
    
    try:
        while True:
            current_time = datetime.now()
            
            # Generate signal every 15 minutes (at :00, :15, :30, :45)
            if (current_time.minute % 15 == 0 and 
                current_time.second < 30 and  # Only in first 30 seconds
                (last_signal_time is None or 
                 (current_time - last_signal_time).total_seconds() > 600)):  # At least 10 min gap
                
                print(f"\n🔄 Generating Golden-Gibz signal at {current_time.strftime('%H:%M:%S')}")
                
                # Get fresh data
                data = mt5_data.get_all_timeframes()
                
                if data['15m'] is not None:
                    # Add indicators
                    df_with_signals = add_live_trend_signals(
                        data['15m'], data['1h'], data['4h'], data['1d']
                    )
                    
                    # Generate signal
                    signal = signal_gen.generate_signal(df_with_signals)
                    
                    if signal:
                        # Save signal to multiple file locations for compatibility
                        for signal_file_path in SIGNAL_FILES:
                            with open(signal_file_path, 'w') as f:
                                json.dump(signal, f, indent=2)
                        
                        print(f"✅ Golden-Gibz Signal: {signal['action_name']} "
                              f"(Confidence: {signal['confidence']:.2f})")
                        print(f"   Market: Bull TF={signal['market_conditions']['bull_timeframes']}, "
                              f"Bear TF={signal['market_conditions']['bear_timeframes']}, "
                              f"Strength={signal['market_conditions']['trend_strength']}")
                        
                        # Log to file
                        with open(LOG_FILE, 'a') as f:
                            f.write(f"{current_time.isoformat()}: {signal['action_name']} "
                                   f"(Conf: {signal['confidence']:.2f})\n")
                        
                        last_signal_time = current_time
                    else:
                        print("❌ Failed to generate Golden-Gibz signal")
                else:
                    print("❌ No 15M data available")
            
            # Sleep for 30 seconds
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n🛑 Golden-Gibz signal generator stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        mt5_data.disconnect()
        print("✅ Disconnected from MT5")


if __name__ == "__main__":
    main()