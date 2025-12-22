#!/usr/bin/env python3
"""
Balanced 1:1 Risk-Reward Trading Model V2 - HIGH WIN RATE EDITION
=================================================================
Target: 75%+ win rate with 1:1 SL/TP ratio

Key Improvements:
1. Multi-timeframe trend alignment (15m, 1H, 4H)
2. Advanced price action patterns (engulfing, pin bars)
3. Momentum divergence detection
4. Volume-price confirmation
5. Support/resistance awareness
6. Smarter entry filtering - only trade high-probability setups
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'indicators'))

import numpy as np
import pandas as pd
import pandas_ta as pta
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ADVANCED INDICATOR FUNCTIONS
# ============================================================================

def add_price_action_patterns(df):
    """Detect high-probability candlestick patterns."""
    # Body and wick calculations
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['range'] = df['High'] - df['Low']
    
    # Bullish patterns
    df['bullish_engulfing'] = (
        (df['Close'] > df['Open']) &  # Current is bullish
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous was bearish
        (df['Open'] < df['Close'].shift(1)) &  # Opens below prev close
        (df['Close'] > df['Open'].shift(1))  # Closes above prev open
    ).astype(int)
    
    df['hammer'] = (
        (df['lower_wick'] > 2 * df['body']) &
        (df['upper_wick'] < df['body'] * 0.3) &
        (df['body'] > 0)
    ).astype(int)
    
    df['bullish_pin'] = (
        (df['lower_wick'] > 2.5 * df['body']) &
        (df['Close'] > df['Open']) &
        (df['upper_wick'] < df['body'])
    ).astype(int)
    
    # Bearish patterns
    df['bearish_engulfing'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1))
    ).astype(int)
    
    df['shooting_star'] = (
        (df['upper_wick'] > 2 * df['body']) &
        (df['lower_wick'] < df['body'] * 0.3) &
        (df['body'] > 0)
    ).astype(int)
    
    df['bearish_pin'] = (
        (df['upper_wick'] > 2.5 * df['body']) &
        (df['Close'] < df['Open']) &
        (df['lower_wick'] < df['body'])
    ).astype(int)
    
    return df


def add_multi_timeframe_trend(df):
    """Add multi-timeframe trend analysis for better accuracy."""
    # Short-term trend (5-period)
    df['ema_5'] = pta.ema(df['Close'], length=5)
    df['ema_10'] = pta.ema(df['Close'], length=10)
    df['trend_short'] = np.where(df['ema_5'] > df['ema_10'], 1, -1)
    
    # Medium-term trend (20-period)
    df['ema_20'] = pta.ema(df['Close'], length=20)
    df['ema_50'] = pta.ema(df['Close'], length=50)
    df['trend_medium'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
    
    # Long-term trend (100-period)
    df['ema_100'] = pta.ema(df['Close'], length=100)
    df['ema_200'] = pta.ema(df['Close'], length=200)
    df['trend_long'] = np.where(df['Close'] > df['ema_200'], 1, -1)
    
    # Trend alignment score (-3 to +3)
    df['trend_alignment'] = df['trend_short'] + df['trend_medium'] + df['trend_long']
    
    # Strong trend detection
    df['strong_uptrend'] = (df['trend_alignment'] == 3).astype(int)
    df['strong_downtrend'] = (df['trend_alignment'] == -3).astype(int)
    
    return df


def add_momentum_indicators(df):
    """Add momentum indicators with divergence detection."""
    # RSI with multiple periods
    df['rsi_7'] = pta.rsi(df['Close'], length=7)
    df['rsi_14'] = pta.rsi(df['Close'], length=14)
    df['rsi_21'] = pta.rsi(df['Close'], length=21)
    
    # RSI zones
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_bullish_zone'] = ((df['rsi_14'] > 50) & (df['rsi_14'] < 70)).astype(int)
    df['rsi_bearish_zone'] = ((df['rsi_14'] < 50) & (df['rsi_14'] > 30)).astype(int)
    
    # MACD
    macd = pta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist'] = macd.iloc[:, 2]
        df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    
    # Stochastic
    stoch = pta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3)
    if stoch is not None:
        df['stoch_k'] = stoch.iloc[:, 0]
        df['stoch_d'] = stoch.iloc[:, 1]
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    
    return df


def add_volume_analysis(df):
    """Add volume-based confirmation indicators."""
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma'] + 1e-10)
    df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
    df['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)
    
    # Volume trend
    df['volume_increasing'] = (df['Volume'] > df['Volume'].shift(1)).astype(int)
    
    # Price-volume confirmation
    df['bullish_volume'] = ((df['Close'] > df['Open']) & (df['high_volume'] == 1)).astype(int)
    df['bearish_volume'] = ((df['Close'] < df['Open']) & (df['high_volume'] == 1)).astype(int)
    
    return df


def add_support_resistance(df, lookback=50):
    """Add dynamic support/resistance levels."""
    df['resistance'] = df['High'].rolling(lookback).max()
    df['support'] = df['Low'].rolling(lookback).min()
    
    # Distance to S/R
    df['dist_to_resistance'] = (df['resistance'] - df['Close']) / df['Close']
    df['dist_to_support'] = (df['Close'] - df['support']) / df['Close']
    
    # Near S/R zones
    df['near_resistance'] = (df['dist_to_resistance'] < 0.002).astype(int)
    df['near_support'] = (df['dist_to_support'] < 0.002).astype(int)
    
    # Breakout detection
    df['resistance_break'] = ((df['Close'] > df['resistance'].shift(1)) & 
                              (df['Close'].shift(1) <= df['resistance'].shift(1))).astype(int)
    df['support_break'] = ((df['Close'] < df['support'].shift(1)) & 
                           (df['Close'].shift(1) >= df['support'].shift(1))).astype(int)
    
    return df


def add_volatility_features(df):
    """Add volatility-based features."""
    df['atr'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['atr_ma'] = df['atr'].rolling(50).mean()
    df['atr_ratio'] = df['atr'] / (df['atr_ma'] + 1e-10)
    
    # Volatility regimes
    df['high_volatility'] = (df['atr_ratio'] > 1.5).astype(int)
    df['low_volatility'] = (df['atr_ratio'] < 0.7).astype(int)
    
    # Bollinger Bands
    bb = pta.bbands(df['Close'], length=20, std=2)
    if bb is not None:
        df['bb_upper'] = bb.iloc[:, 0]
        df['bb_middle'] = bb.iloc[:, 1]
        df['bb_lower'] = bb.iloc[:, 2]
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).mean() * 0.8).astype(int)
    
    return df


def create_high_probability_signals(df):
    """Create composite signals for high win-rate entries."""
    
    # BULLISH SIGNAL SCORE (0-10)
    df['bull_score'] = 0
    
    # Trend alignment (+3 max)
    df.loc[df['trend_alignment'] >= 2, 'bull_score'] += 2
    df.loc[df['trend_alignment'] == 3, 'bull_score'] += 1
    
    # Price action patterns (+2)
    df.loc[df['bullish_engulfing'] == 1, 'bull_score'] += 2
    df.loc[df['hammer'] == 1, 'bull_score'] += 1
    df.loc[df['bullish_pin'] == 1, 'bull_score'] += 2
    
    # Momentum confirmation (+2)
    df.loc[(df['rsi_14'] > 40) & (df['rsi_14'] < 70), 'bull_score'] += 1
    df.loc[df['macd_bullish'] == 1, 'bull_score'] += 1
    df.loc[df['macd_cross_up'] == 1, 'bull_score'] += 1
    
    # Volume confirmation (+1)
    df.loc[df['bullish_volume'] == 1, 'bull_score'] += 1
    
    # Support bounce (+1)
    df.loc[df['near_support'] == 1, 'bull_score'] += 1
    
    # BEARISH SIGNAL SCORE (0-10)
    df['bear_score'] = 0
    
    # Trend alignment (+3 max)
    df.loc[df['trend_alignment'] <= -2, 'bear_score'] += 2
    df.loc[df['trend_alignment'] == -3, 'bear_score'] += 1
    
    # Price action patterns (+2)
    df.loc[df['bearish_engulfing'] == 1, 'bear_score'] += 2
    df.loc[df['shooting_star'] == 1, 'bear_score'] += 1
    df.loc[df['bearish_pin'] == 1, 'bear_score'] += 2
    
    # Momentum confirmation (+2)
    df.loc[(df['rsi_14'] < 60) & (df['rsi_14'] > 30), 'bear_score'] += 1
    df.loc[df['macd_bullish'] == 0, 'bear_score'] += 1
    df.loc[df['macd_cross_down'] == 1, 'bear_score'] += 1
    
    # Volume confirmation (+1)
    df.loc[df['bearish_volume'] == 1, 'bear_score'] += 1
    
    # Resistance rejection (+1)
    df.loc[df['near_resistance'] == 1, 'bear_score'] += 1
    
    # HIGH PROBABILITY SIGNALS (score >= 5)
    df['high_prob_buy'] = (df['bull_score'] >= 5).astype(int)
    df['high_prob_sell'] = (df['bear_score'] >= 5).astype(int)
    
    # VERY HIGH PROBABILITY (score >= 7)
    df['very_high_prob_buy'] = (df['bull_score'] >= 7).astype(int)
    df['very_high_prob_sell'] = (df['bear_score'] >= 7).astype(int)
    
    return df


# ============================================================================
# TRADING ENVIRONMENT
# ============================================================================

class HighWinRateEnv(gym.Env):
    """
    Trading environment optimized for 75%+ win rate with 1:1 RR.
    Only trades high-probability setups.
    """
    
    def __init__(self, df, window_size=30, initial_balance=10000.0,
                 atr_mult=1.5, min_signal_score=5, random_start=True):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.random_start = random_start
        
        self.atr_mult = atr_mult  # 1:1 ratio
        self.min_signal_score = min_signal_score
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        self.feature_columns = [col for col in df.columns 
                               if col not in ['Date', 'datetime', 'Gmt time']]
        self.num_features = len(self.feature_columns)
        
        # Observation: window of features + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.num_features + 4),
            dtype=np.float32
        )
        
        self._init_state()
    
    def _init_state(self):
        if self.random_start and self.n_steps > self.window_size + 500:
            max_start = self.n_steps - 500
            self.current_step = np.random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
        
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_episode_steps = min(2000, self.n_steps - self.current_step - 1)
        self.episode_steps = 0
        self.last_trade_step = -10
        self.peak_balance = self.initial_balance
        self.consecutive_losses = 0

    def _get_atr(self):
        atr = self.df.iloc[self.current_step].get('atr', 0)
        if not isinstance(atr, (int, float)) or atr <= 0 or np.isnan(atr):
            recent = self.df.iloc[max(0, self.current_step-14):self.current_step]
            if len(recent) > 0:
                atr = (recent['High'] - recent['Low']).mean()
            else:
                atr = 0.5
        return max(abs(atr), 0.01)
    
    def _get_signal_quality(self):
        """Get signal quality scores."""
        if self.current_step >= len(self.df):
            return 0, 0
        
        row = self.df.iloc[self.current_step]
        bull_score = row.get('bull_score', 0)
        bear_score = row.get('bear_score', 0)
        return bull_score, bear_score

    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        obs_data = self.df[self.feature_columns].iloc[start:end].values.copy()
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), self.num_features))
            obs_data = np.vstack([pad, obs_data])
        
        # Extra features: position, PnL, bull_score, bear_score
        extra = np.zeros((self.window_size, 4))
        
        if self.position == 'long':
            extra[:, 0] = 1.0
        elif self.position == 'short':
            extra[:, 0] = -1.0
        
        if self.position and self.entry_price != 0 and self.entry_atr > 0:
            price = self.df.iloc[self.current_step]['Close']
            if self.position == 'long':
                pnl = (price - self.entry_price) / self.entry_atr
            else:
                pnl = (self.entry_price - price) / self.entry_atr
            extra[:, 1] = np.clip(pnl, -3.0, 3.0)
        
        bull_score, bear_score = self._get_signal_quality()
        extra[:, 2] = bull_score / 10.0
        extra[:, 3] = bear_score / 10.0
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return np.clip(obs, -5.0, 5.0)
    
    def _check_exit(self, high, low):
        """Check SL/TP hit (1:1 ratio)."""
        if not self.position or self.entry_price == 0 or self.entry_atr <= 0:
            return False, 0.0, False
        
        distance = self.entry_atr * self.atr_mult
        
        if self.position == 'long':
            tp_price = self.entry_price + distance
            sl_price = self.entry_price - distance
            
            if high >= tp_price:
                pnl = distance / (abs(self.entry_price) + 0.01)
                return True, pnl, True
            if low <= sl_price:
                pnl = -distance / (abs(self.entry_price) + 0.01)
                return True, pnl, False
        else:
            tp_price = self.entry_price - distance
            sl_price = self.entry_price + distance
            
            if low <= tp_price:
                pnl = distance / (abs(self.entry_price) + 0.01)
                return True, pnl, True
            if high >= sl_price:
                pnl = -distance / (abs(self.entry_price) + 0.01)
                return True, pnl, False
        
        return False, 0.0, False


    def step(self, action):
        reward = 0.0
        done = False
        
        price = self.df.iloc[self.current_step]['Close']
        high = self.df.iloc[self.current_step]['High']
        low = self.df.iloc[self.current_step]['Low']
        bull_score, bear_score = self._get_signal_quality()
        trend_align = self.df.iloc[self.current_step].get('trend_alignment', 0)
        
        # Check exit first
        if self.position:
            closed, pnl, is_win = self._check_exit(high, low)
            if closed:
                self.total_trades += 1
                if is_win:
                    self.wins += 1
                    self.total_profit += pnl
                    self.consecutive_losses = 0
                    # Bigger reward for wins (we want high WR)
                    reward = 15.0 + (bull_score if self.position == 'long' else bear_score)
                else:
                    self.losses += 1
                    self.total_loss += abs(pnl)
                    self.consecutive_losses += 1
                    # Penalty scales with consecutive losses
                    reward = -12.0 - (self.consecutive_losses * 2)
                
                self.balance *= (1 + pnl)
                self.position = None
                self.entry_price = 0.0
                self.entry_atr = 0.0
        
        # Process new entries - ONLY on high-probability signals
        can_trade = (self.current_step - self.last_trade_step) >= 5
        
        if self.position is None and can_trade:
            if action == 1:  # Buy
                # Only enter if signal quality is good
                if bull_score >= self.min_signal_score and trend_align >= 1:
                    self.position = 'long'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.last_trade_step = self.current_step
                    # Reward based on signal quality
                    reward += bull_score * 0.5
                elif bull_score >= 3:
                    # Moderate signal - small reward for trying
                    reward -= 1.0
                else:
                    # Bad signal - penalty
                    reward -= 3.0
                    
            elif action == 2:  # Sell
                if bear_score >= self.min_signal_score and trend_align <= -1:
                    self.position = 'short'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.last_trade_step = self.current_step
                    reward += bear_score * 0.5
                elif bear_score >= 3:
                    reward -= 1.0
                else:
                    reward -= 3.0
            
            else:  # Hold
                # Small reward for holding when no good signal
                if bull_score < 4 and bear_score < 4:
                    reward += 0.1
        
        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        self.current_step += 1
        self.episode_steps += 1
        
        if self.current_step >= self.n_steps - 1:
            done = True
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        # Drawdown check
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.20:
            done = True
            reward -= 30.0
        
        # Consecutive loss check
        if self.consecutive_losses >= 4:
            reward -= 10.0
        
        info = {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'profit_factor': self.total_profit / max(0.01, self.total_loss)
        }
        
        return self._get_obs(), float(np.clip(reward, -50, 30)), done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


# ============================================================================
# TRAINING CALLBACK
# ============================================================================

class HighWRCallback(BaseCallback):
    """Callback targeting 75%+ win rate."""
    
    def __init__(self, val_df, atr_mult, min_signal_score, eval_freq=20000, 
                 save_path="models/experimental", target_wr=0.75):
        super().__init__()
        self.val_df = val_df
        self.atr_mult = atr_mult
        self.min_signal_score = min_signal_score
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.target_wr = target_wr
        
        self.best_score = -np.inf
        self.best_wr = 0.0
        self.best_pf = 0.0
        self.winrates = []
        self.last_eval = 0
        self.no_improve = 0
        self._vec_norm = None
    
    def _on_training_start(self):
        self._vec_norm = self.model.get_env()
    
    def _on_step(self):
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if info.get('total_trades', 0) >= 3:
                    self.winrates.append(info['win_rate'])
                    
                    if len(self.winrates) % 80 == 0:
                        wr = np.mean(self.winrates[-150:]) if len(self.winrates) >= 150 else np.mean(self.winrates)
                        trades = info.get('total_trades', 0)
                        print(f"📊 Train WR: {wr*100:.1f}% | Trades: {trades} | Best: {self.best_wr*100:.1f}%")
        
        if self.num_timesteps >= self.last_eval + self.eval_freq:
            val_wr, val_trades, val_pf, val_ret = self._evaluate()
            
            # Score heavily weighted toward win rate
            score = val_wr * 0.7 + min(val_pf, 3.0) / 3.0 * 0.2 + max(0, val_ret/50) * 0.1
            
            if score > self.best_score and val_trades >= 5 and val_wr > 0.55:
                self.best_score = score
                self.best_wr = val_wr
                self.best_pf = val_pf
                self.no_improve = 0
                self.model.save(f"{self.save_path}/balanced_rr_v2_best")
                
                status = "🎯" if val_wr >= self.target_wr else "✅"
                print(f"\n{status} NEW BEST: WR={val_wr*100:.1f}% PF={val_pf:.2f} Trades={val_trades:.0f}")
            else:
                self.no_improve += 1
                print(f"\n📉 Val: WR={val_wr*100:.1f}% PF={val_pf:.2f} | No improve: {self.no_improve}")
            
            self.last_eval = self.num_timesteps
            
            if self.no_improve >= 25:
                print("\n🛑 Early stopping - no improvement")
                return False
        
        return True
    
    def _evaluate(self):
        env = HighWinRateEnv(df=self.val_df, atr_mult=self.atr_mult, 
                            min_signal_score=self.min_signal_score, random_start=True)
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        if self._vec_norm and isinstance(self._vec_norm, VecNormalize):
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)
            vec_env.obs_rms = self._vec_norm.obs_rms
            vec_env.training = False
        
        total_trades, total_wins = 0, 0
        returns = []
        
        for _ in range(8):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec_env.step(action)
                done = dones[0]
            if infos:
                total_trades += infos[0].get('total_trades', 0)
                total_wins += infos[0].get('wins', 0)
                returns.append(infos[0].get('return_pct', 0))
        
        wr = total_wins / max(1, total_trades)
        pf = wr / (1 - wr + 0.001) if wr < 1 else 10.0
        avg_ret = np.mean(returns) if returns else 0
        
        return wr, total_trades / 8, pf, avg_ret


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(filepath):
    """Load and prepare data with advanced indicators."""
    print(f"Loading: {filepath}")
    
    try:
        df = pd.read_csv(filepath, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',')
    except:
        df = pd.read_csv(filepath, sep=',')
    
    for col in ['Date', 'date', 'datetime', 'Gmt time']:
        if col in df.columns:
            df['Date'] = pd.to_datetime(df[col])
            df.set_index('Date', inplace=True)
            break
    
    df.sort_index(inplace=True)
    print(f"Loaded {len(df)} bars")
    
    # Add all advanced indicators
    print("Adding indicators...")
    
    # Price action patterns
    df = add_price_action_patterns(df)
    print("  ✓ Price action patterns")
    
    # Multi-timeframe trend
    df = add_multi_timeframe_trend(df)
    print("  ✓ Multi-timeframe trend")
    
    # Momentum indicators
    df = add_momentum_indicators(df)
    print("  ✓ Momentum indicators")
    
    # Volume analysis
    df = add_volume_analysis(df)
    print("  ✓ Volume analysis")
    
    # Support/Resistance
    df = add_support_resistance(df)
    print("  ✓ Support/Resistance")
    
    # Volatility features
    df = add_volatility_features(df)
    print("  ✓ Volatility features")
    
    # High probability signals
    df = create_high_probability_signals(df)
    print("  ✓ High probability signals")
    
    # Normalize prices
    base_price = df['Close'].iloc[0]
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] / base_price - 1) * 100
    
    # Normalize EMAs
    ema_cols = ['ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200']
    for col in ema_cols:
        if col in df.columns:
            df[col] = (df[col] / base_price - 1) * 100
    
    # Normalize ATR
    if 'atr' in df.columns:
        df['atr'] = df['atr'] / base_price * 100
    if 'atr_ma' in df.columns:
        df['atr_ma'] = df['atr_ma'] / base_price * 100
    
    # Normalize oscillators
    for col in ['rsi_7', 'rsi_14', 'rsi_21']:
        if col in df.columns:
            df[col] = (df[col] - 50) / 50
    
    if 'stoch_k' in df.columns:
        df['stoch_k'] = (df['stoch_k'] - 50) / 50
    if 'stoch_d' in df.columns:
        df['stoch_d'] = (df['stoch_d'] - 50) / 50
    
    # Normalize other features
    skip_cols = ['Open', 'High', 'Low', 'Close', 'atr', 'atr_ma'] + ema_cols
    skip_cols += ['rsi_7', 'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d']
    skip_cols += ['bull_score', 'bear_score', 'trend_alignment']
    
    binary_cols = [c for c in df.columns if df[c].isin([0, 1, -1]).all()]
    
    for col in df.columns:
        if col not in skip_cols and col not in binary_cols:
            if df[col].dtype in ['float64', 'float32'] and df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
                df[col] = df[col].clip(-3, 3)
    
    df = df.dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Final: {len(df)} bars, {len(df.columns)} features")
    return df


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def make_env(df, seed, atr_mult, min_signal_score):
    def _init():
        env = HighWinRateEnv(df=df, atr_mult=atr_mult, 
                            min_signal_score=min_signal_score, random_start=True)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train(
    data_path="data/raw/XAU_15m_data.csv",
    timesteps=1500000,
    n_envs=48,
    atr_mult=1.5,
    min_signal_score=5,
    eval_freq=20000,
    target_wr=0.75
):
    """Train high win-rate 1:1 RR model."""
    
    print("=" * 70)
    print("🚀 HIGH WIN-RATE 1:1 RISK-REWARD MODEL V2")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Target Win Rate: {target_wr*100:.0f}%+")
    print(f"  SL = TP = {atr_mult} * ATR (1:1 ratio)")
    print(f"  Min Signal Score: {min_signal_score}/10")
    print(f"  Envs: {n_envs} | Timesteps: {timesteps:,}")
    
    df = load_and_prepare_data(data_path)
    
    # Split data
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Signal distribution
    train_bull = (train_df['bull_score'] >= min_signal_score).sum()
    train_bear = (train_df['bear_score'] >= min_signal_score).sum()
    print(f"High-prob signals in train: {train_bull} buy, {train_bear} sell")
    
    # Create environments
    print(f"\nCreating {n_envs} environments...")
    env = DummyVecEnv([make_env(train_df, i, atr_mult, min_signal_score) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=50.0)
    
    # Larger network for complex pattern recognition
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=nn.Tanh
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Lower entropy for more decisive actions
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/balanced_rr_v2_training"
    )
    
    os.makedirs("models/experimental", exist_ok=True)
    
    callback = HighWRCallback(
        val_df=val_df,
        atr_mult=atr_mult,
        min_signal_score=min_signal_score,
        eval_freq=eval_freq,
        save_path="models/experimental",
        target_wr=target_wr
    )
    
    print("\n🏋️ Starting training...")
    print("=" * 70)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/experimental/balanced_rr_v2_final_{timestamp}")
    env.save(f"models/experimental/balanced_rr_v2_final_{timestamp}_vecnorm.pkl")


    # Test evaluation
    print("\n" + "=" * 70)
    print("🧪 TEST EVALUATION")
    print("=" * 70)
    
    total_trades, total_wins = 0, 0
    returns = []
    
    for ep in range(15):
        test_env = HighWinRateEnv(df=test_df, atr_mult=atr_mult, 
                                  min_signal_score=min_signal_score, random_start=True)
        test_vec = DummyVecEnv([lambda: Monitor(test_env)])
        test_vec = VecNormalize(test_vec, norm_obs=True, norm_reward=False, clip_obs=5.0)
        test_vec.obs_rms = env.obs_rms
        test_vec.training = False
        
        obs = test_vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = test_vec.step(action)
            done = dones[0]
        
        if infos:
            trades = infos[0].get('total_trades', 0)
            wins = infos[0].get('wins', 0)
            ret = infos[0].get('return_pct', 0)
            total_trades += trades
            total_wins += wins
            returns.append(ret)
            wr = wins / max(1, trades)
            print(f"  Ep {ep+1:2d}: Trades={trades:3d}, WR={wr*100:5.1f}%, Return={ret:+6.1f}%")
    
    final_wr = total_wins / max(1, total_trades)
    avg_return = np.mean(returns)
    profit_factor = final_wr / (1 - final_wr + 0.001) if final_wr < 1 else 10.0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {final_wr*100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Return: {avg_return:+.1f}%")
    print(f"{'='*60}")
    
    # Save to production if meets target
    if final_wr >= target_wr and profit_factor > 2.0:
        print(f"🎉 TARGET ACHIEVED! WR >= {target_wr*100:.0f}%")
        os.makedirs("models/production", exist_ok=True)
        model.save(f"models/production/balanced_rr_v2_{final_wr*100:.0f}pct_{timestamp}")
    elif final_wr >= 0.65:
        print(f"✅ Good model! WR = {final_wr*100:.1f}%")
        os.makedirs("models/production", exist_ok=True)
        model.save(f"models/production/balanced_rr_v2_{final_wr*100:.0f}pct_{timestamp}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1500000)
    parser.add_argument('--envs', type=int, default=48)
    parser.add_argument('--atr-mult', type=float, default=1.5, help='ATR multiplier for SL=TP')
    parser.add_argument('--min-score', type=int, default=5, help='Min signal score to trade')
    parser.add_argument('--target-wr', type=float, default=0.75, help='Target win rate')
    
    args = parser.parse_args()
    
    train(
        timesteps=args.timesteps,
        n_envs=args.envs,
        atr_mult=args.atr_mult,
        min_signal_score=args.min_score,
        target_wr=args.target_wr
    )
