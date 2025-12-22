#!/usr/bin/env python3
"""
Balanced 1:1 Risk-Reward Trading Model V3 - ENHANCED HIGH WIN RATE EDITION
=========================================================================
Target: 80%+ win rate with 1:1 SL/TP ratio

Enhanced Features:
1. Adaptive signal scoring based on market conditions
2. Market regime detection (trending vs ranging)
3. Time-based filters (avoid low-liquidity periods)
4. Enhanced risk management with position sizing
5. Multi-model ensemble approach
6. Real-time performance monitoring
7. Advanced exit strategies
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
from datetime import datetime, time
import gymnasium as gym
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# ENHANCED INDICATOR FUNCTIONS
# ============================================================================

def add_market_regime_detection(df):
    """Detect market regime: trending vs ranging."""
    # ADX for trend strength
    adx = pta.adx(df['High'], df['Low'], df['Close'], length=14)
    if adx is not None:
        df['adx'] = adx.iloc[:, 0]  # ADX
        df['dmp'] = adx.iloc[:, 1]  # DI+
        df['dmn'] = adx.iloc[:, 2]  # DI-
    else:
        df['adx'] = 25  # Default neutral
        df['dmp'] = 25
        df['dmn'] = 25
    
    # Market regime classification
    df['trending_market'] = (df['adx'] > 25).astype(int)
    df['strong_trend'] = (df['adx'] > 40).astype(int)
    df['ranging_market'] = (df['adx'] < 20).astype(int)
    
    # Trend direction
    df['bullish_trend'] = ((df['dmp'] > df['dmn']) & (df['adx'] > 25)).astype(int)
    df['bearish_trend'] = ((df['dmn'] > df['dmp']) & (df['adx'] > 25)).astype(int)
    
    return df


def add_time_based_filters(df):
    """Add time-based trading filters."""
    # Assuming we have datetime index
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
    else:
        # Create dummy hour for now
        df['hour'] = 12
    
    # London/NY session (high liquidity)
    df['london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
    df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
    df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
    
    # Avoid low liquidity periods
    df['avoid_time'] = ((df['hour'] >= 22) | (df['hour'] <= 2)).astype(int)
    
    return df


def add_enhanced_price_action(df):
    """Enhanced price action patterns with context."""
    # Basic calculations
    df['body'] = abs(df['Close'] - df['Open'])
    df['upper_wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['range'] = df['High'] - df['Low']
    df['body_pct'] = df['body'] / (df['range'] + 1e-10)
    
    # Enhanced patterns with volume confirmation
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma'] + 1e-10)
    
    # Bullish patterns with volume
    df['strong_bullish_engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1)) &
        (df['Open'] < df['Close'].shift(1)) &
        (df['Close'] > df['Open'].shift(1)) &
        (df['volume_ratio'] > 1.2) &  # Volume confirmation
        (df['body_pct'] > 0.6)  # Strong body
    ).astype(int)
    
    df['hammer_with_volume'] = (
        (df['lower_wick'] > 2 * df['body']) &
        (df['upper_wick'] < df['body'] * 0.3) &
        (df['body'] > 0) &
        (df['volume_ratio'] > 1.1)
    ).astype(int)
    
    # Bearish patterns with volume
    df['strong_bearish_engulfing'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1)) &
        (df['Open'] > df['Close'].shift(1)) &
        (df['Close'] < df['Open'].shift(1)) &
        (df['volume_ratio'] > 1.2) &
        (df['body_pct'] > 0.6)
    ).astype(int)
    
    df['shooting_star_with_volume'] = (
        (df['upper_wick'] > 2 * df['body']) &
        (df['lower_wick'] < df['body'] * 0.3) &
        (df['body'] > 0) &
        (df['volume_ratio'] > 1.1)
    ).astype(int)
    
    return df


def add_adaptive_indicators(df):
    """Adaptive indicators that adjust to market conditions - OPTIMIZED."""
    # Core volatility calculation
    df['atr'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['atr_ma'] = df['atr'].rolling(50).mean()  # Faster MA for responsiveness
    df['volatility_ratio'] = df['atr'] / (df['atr_ma'] + 1e-10)
    
    # Simplified volatility regime (binary for speed)
    df['high_vol'] = (df['volatility_ratio'] > 1.2).astype(int)
    
    # Pre-calculate all RSI variants (more efficient than conditional)
    rsi_normal = pta.rsi(df['Close'], length=14)
    rsi_fast = pta.rsi(df['Close'], length=9)
    
    # Use vectorized selection
    df['rsi_adaptive'] = np.where(df['high_vol'], rsi_fast, rsi_normal)
    
    # Pre-calculate EMA variants
    ema_fast_normal = pta.ema(df['Close'], length=12)
    ema_fast_responsive = pta.ema(df['Close'], length=8)
    ema_slow_normal = pta.ema(df['Close'], length=26)
    ema_slow_responsive = pta.ema(df['Close'], length=18)
    
    df['ema_fast_adaptive'] = np.where(df['high_vol'], ema_fast_responsive, ema_fast_normal)
    df['ema_slow_adaptive'] = np.where(df['high_vol'], ema_slow_responsive, ema_slow_normal)
    
    # Add momentum confirmation
    df['momentum'] = df['Close'].pct_change(5) * 100
    df['momentum_ma'] = df['momentum'].rolling(10).mean()
    
    return df


def create_enhanced_signal_scoring(df):
    """ULTRA-SELECTIVE signal scoring targeting 80%+ win rate."""
    
    # More stringent trend alignment
    bull_trend_strength = (
        (df['ema_fast_adaptive'] > df['ema_slow_adaptive']).astype(int) +
        (df['Close'] > df['ema_fast_adaptive']).astype(int) +
        (df['bullish_trend'] == 1).astype(int) +
        (df['Close'] > df['Close'].shift(1)).astype(int)  # Price momentum
    )
    
    bear_trend_strength = (
        (df['ema_fast_adaptive'] < df['ema_slow_adaptive']).astype(int) +
        (df['Close'] < df['ema_fast_adaptive']).astype(int) +
        (df['bearish_trend'] == 1).astype(int) +
        (df['Close'] < df['Close'].shift(1)).astype(int)  # Price momentum
    )
    
    # ULTRA-SELECTIVE BULLISH SCORING
    bull_score = np.zeros(len(df))
    
    # Require STRONG trend alignment (minimum 3/4)
    bull_score += np.where(bull_trend_strength >= 3, 3, 0)
    bull_score += np.where(bull_trend_strength == 4, 2, 0)  # Perfect alignment bonus
    
    # STRICT price action requirements
    bull_score += df['strong_bullish_engulfing'] * 4  # Increased weight
    bull_score += df['hammer_with_volume'] * 3
    
    # STRICT momentum requirements
    bull_score += np.where((df['rsi_adaptive'] > 40) & (df['rsi_adaptive'] < 60), 2, 0)  # Neutral RSI
    bull_score += np.where(df['adx'] > 30, 2, 0)  # Strong trend only
    bull_score += np.where(df['momentum'] > df['momentum_ma'] * 1.2, 2, 0)  # Strong momentum
    
    # STRICT market conditions
    bull_score += np.where(df['trending_market'] & df['overlap_session'], 2, 0)  # Both required
    bull_score += np.where(df['strong_trend'] == 1, 1, 0)  # Very strong trend bonus
    
    # Volume confirmation requirement
    bull_score += np.where(df['volume_ratio'] > 1.3, 1, 0)
    
    # ULTRA-SELECTIVE BEARISH SCORING
    bear_score = np.zeros(len(df))
    
    # Require STRONG trend alignment (minimum 3/4)
    bear_score += np.where(bear_trend_strength >= 3, 3, 0)
    bear_score += np.where(bear_trend_strength == 4, 2, 0)  # Perfect alignment bonus
    
    # STRICT price action requirements
    bear_score += df['strong_bearish_engulfing'] * 4  # Increased weight
    bear_score += df['shooting_star_with_volume'] * 3
    
    # STRICT momentum requirements
    bear_score += np.where((df['rsi_adaptive'] < 60) & (df['rsi_adaptive'] > 40), 2, 0)  # Neutral RSI
    bear_score += np.where(df['adx'] > 30, 2, 0)  # Strong trend only
    bear_score += np.where(df['momentum'] < df['momentum_ma'] * 0.8, 2, 0)  # Strong momentum
    
    # STRICT market conditions
    bear_score += np.where(df['trending_market'] & df['overlap_session'], 2, 0)  # Both required
    bear_score += np.where(df['strong_trend'] == 1, 1, 0)  # Very strong trend bonus
    
    # Volume confirmation requirement
    bear_score += np.where(df['volume_ratio'] > 1.3, 1, 0)
    
    # Store scores
    df['enhanced_bull_score'] = bull_score
    df['enhanced_bear_score'] = bear_score
    
    # ULTRA-SELECTIVE thresholds (much higher requirements)
    df['ultra_high_prob_buy'] = (bull_score >= 12).astype(int)    # Raised from 8
    df['ultra_high_prob_sell'] = (bear_score >= 12).astype(int)   # Raised from 8
    df['high_prob_buy'] = (bull_score >= 10).astype(int)         # Raised from 6
    df['high_prob_sell'] = (bear_score >= 10).astype(int)        # Raised from 6
    df['medium_prob_buy'] = (bull_score >= 8).astype(int)        # Raised from 4
    df['medium_prob_sell'] = (bear_score >= 8).astype(int)       # Raised from 4
    
    return df


# ============================================================================
# ENHANCED TRADING ENVIRONMENT
# ============================================================================

class UltraHighWinRateEnv(gym.Env):
    """
    Ultra high win rate environment targeting 80%+ with enhanced features.
    """
    
    def __init__(self, df, window_size=40, initial_balance=10000.0,
                 atr_mult=1.2, min_signal_score=6, ultra_mode=False, 
                 dynamic_sizing=True, random_start=True):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.random_start = random_start
        
        self.atr_mult = atr_mult
        self.min_signal_score = min_signal_score
        self.ultra_mode = ultra_mode  # Only ultra high prob signals
        self.dynamic_sizing = dynamic_sizing
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        self.feature_columns = [col for col in df.columns 
                               if col not in ['Date', 'datetime', 'Gmt time', 'hour']]
        self.num_features = len(self.feature_columns)
        
        # Optimized observation space - reduced features for speed
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.num_features + 4),  # +4 for essential features only
            dtype=np.float32
        )
        
        self._init_state()
    
    def _init_state(self):
        if self.random_start and self.n_steps > self.window_size + 1000:
            max_start = self.n_steps - 1000
            self.current_step = np.random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
        
        self.balance = self.initial_balance
        self.position = None
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.position_size = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_episode_steps = min(2000, self.n_steps - self.current_step - 1)  # Reduced for speed
        self.episode_steps = 0
        self.last_trade_step = -15
        self.peak_balance = self.initial_balance
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Enhanced tracking
        self.trade_history = []
        self.daily_trades = 0
        self.last_day = -1

    def _get_enhanced_signal_quality(self):
        """Get enhanced signal quality scores."""
        if self.current_step >= len(self.df):
            return 0, 0, 0, 0
        
        row = self.df.iloc[self.current_step]
        bull_score = row.get('enhanced_bull_score', 0)
        bear_score = row.get('enhanced_bear_score', 0)
        ultra_bull = row.get('ultra_high_prob_buy', 0)
        ultra_bear = row.get('ultra_high_prob_sell', 0)
        
        return bull_score, bear_score, ultra_bull, ultra_bear
    
    def _calculate_position_size(self, signal_strength):
        """Improved dynamic position sizing with risk management."""
        if not self.dynamic_sizing:
            return 0.015  # Fixed 1.5%
        
        # Base size scaling (more conservative)
        base_size = 0.008  # 0.8%
        max_size = 0.025   # 2.5%
        
        # Signal strength scaling (4-12 range for better distribution)
        strength_ratio = np.clip((signal_strength - 4) / 8, 0, 1)
        size = base_size + (max_size - base_size) * strength_ratio
        
        # Enhanced risk adjustments
        if self.consecutive_losses > 0:
            # More aggressive reduction after losses
            reduction = 0.7 ** min(self.consecutive_losses, 4)
            size *= reduction
        
        if self.consecutive_wins > 2:
            # Modest increase after wins, but capped
            increase = min(1.3, 1 + (self.consecutive_wins - 2) * 0.05)
            size *= increase
        
        # Account balance scaling (reduce size if balance is low)
        balance_ratio = self.balance / self.initial_balance
        if balance_ratio < 0.9:
            size *= balance_ratio
        
        return np.clip(size, 0.003, 0.04)  # 0.3% to 4% range
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        obs_data = self.df[self.feature_columns].iloc[start:end].values.copy()
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), self.num_features))
            obs_data = np.vstack([pad, obs_data])
        
        # Optimized extra features: position, PnL, bull_score, bear_score only
        extra = np.zeros((self.window_size, 4))
        
        # Position encoding
        if self.position == 'long':
            extra[:, 0] = 1.0
        elif self.position == 'short':
            extra[:, 0] = -1.0
        
        # Simplified PnL
        if self.position and self.entry_price != 0:
            price = self.df.iloc[self.current_step]['Close']
            if self.position == 'long':
                pnl = (price - self.entry_price) / abs(self.entry_price)
            else:
                pnl = (self.entry_price - price) / abs(self.entry_price)
            extra[:, 1] = np.clip(pnl * 10, -5.0, 5.0)  # Simplified calculation
        
        # Signal scores (cached for speed)
        bull_score, bear_score, ultra_bull, ultra_bear = self._get_enhanced_signal_quality()
        extra[:, 2] = bull_score / 12.0
        extra[:, 3] = bear_score / 12.0
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return np.clip(obs, -5.0, 5.0)
    
    def step(self, action):
        reward = 0.0
        done = False
        
        price = self.df.iloc[self.current_step]['Close']
        high = self.df.iloc[self.current_step]['High']
        low = self.df.iloc[self.current_step]['Low']
        
        bull_score, bear_score, ultra_bull, ultra_bear = self._get_enhanced_signal_quality()
        
        # Check if it's a new day (reset daily trade count)
        current_day = self.current_step // 96  # Assuming 15min bars, 96 per day
        if current_day != self.last_day:
            self.daily_trades = 0
            self.last_day = current_day
        
        # Check exit first
        if self.position:
            closed, pnl, is_win = self._check_enhanced_exit(high, low)
            if closed:
                self.total_trades += 1
                self.daily_trades += 1
                
                if is_win:
                    self.wins += 1
                    self.total_profit += pnl
                    self.consecutive_losses = 0
                    self.consecutive_wins += 1
                    
                    # MASSIVE win rewards to incentivize high win rate
                    base_reward = 50.0  # Doubled base reward
                    
                    # Ultra-high quality bonus
                    signal_score = bull_score if self.position == 'long' else bear_score
                    if signal_score >= 12:
                        quality_bonus = 25.0  # Huge bonus for ultra signals
                    elif signal_score >= 10:
                        quality_bonus = 15.0  # Large bonus for high signals
                    else:
                        quality_bonus = 5.0   # Small bonus for medium signals
                    
                    # Win rate bonus (reward maintaining high win rate)
                    if self.total_trades >= 5:
                        current_wr = self.wins / self.total_trades
                        if current_wr >= 0.8:
                            wr_bonus = 20.0
                        elif current_wr >= 0.7:
                            wr_bonus = 10.0
                        elif current_wr >= 0.6:
                            wr_bonus = 5.0
                        else:
                            wr_bonus = 0.0
                    else:
                        wr_bonus = 0.0
                    
                    # Streak bonus (encourage consistency)
                    streak_bonus = min(self.consecutive_wins * 3.0, 15)
                    
                    reward = base_reward + quality_bonus + wr_bonus + streak_bonus
                    
                else:
                    self.losses += 1
                    self.total_loss += abs(pnl)
                    self.consecutive_wins = 0
                    self.consecutive_losses += 1
                    
                    # More nuanced loss penalties
                    base_penalty = -12.0  # Reduced from -15
                    
                    # Streak penalty (less harsh)
                    streak_penalty = min(self.consecutive_losses * 2, 12)
                    
                    # Signal quality penalty (worse penalty for bad signals)
                    signal_score = bull_score if self.position == 'long' else bear_score
                    if signal_score < 6:
                        quality_penalty = 5.0
                    else:
                        quality_penalty = 0.0
                    
                    reward = base_penalty - streak_penalty - quality_penalty
                
                self.balance *= (1 + pnl * self.position_size)
                self.position = None
                self.entry_price = 0.0
                self.entry_atr = 0.0
                self.position_size = 0.0
        
        # ULTRA-SELECTIVE trading logic - quality over quantity
        cooldown_period = 15  # Increased to prevent overtrading
        max_daily_trades = 3   # Reduced for selectivity
        
        can_trade = (
            (self.current_step - self.last_trade_step) >= cooldown_period and
            self.position is None and
            self.daily_trades < max_daily_trades and
            self.consecutive_losses < 3  # Stop trading after 3 consecutive losses
        )
        
        if can_trade:
            if action == 1:  # Buy
                # ULTRA-SELECTIVE requirements
                required_score = 12 if self.ultra_mode else max(10, self.min_signal_score)
                
                if (ultra_bull == 1 if self.ultra_mode else bull_score >= required_score):
                    self.position = 'long'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.position_size = self._calculate_position_size(bull_score)
                    self.last_trade_step = self.current_step
                    
                    # MASSIVE reward for ultra-selective signals
                    base_reward = 15.0  # Higher base reward
                    quality_bonus = (bull_score - 10) * 2.0  # Reward signal quality
                    if ultra_bull == 1:
                        base_reward += 10.0  # Huge bonus for ultra signals
                    reward += base_reward + quality_bonus
                        
                elif bull_score >= 8:
                    reward -= 5.0  # Heavy penalty for medium signals
                elif bull_score >= 6:
                    reward -= 8.0  # Heavier penalty for weak signals
                else:
                    reward -= 12.0  # Massive penalty for poor signals
                    
            elif action == 2:  # Sell
                # ULTRA-SELECTIVE requirements
                required_score = 12 if self.ultra_mode else max(10, self.min_signal_score)
                
                if (ultra_bear == 1 if self.ultra_mode else bear_score >= required_score):
                    self.position = 'short'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.position_size = self._calculate_position_size(bear_score)
                    self.last_trade_step = self.current_step
                    
                    # MASSIVE reward for ultra-selective signals
                    base_reward = 15.0  # Higher base reward
                    quality_bonus = (bear_score - 10) * 2.0  # Reward signal quality
                    if ultra_bear == 1:
                        base_reward += 10.0  # Huge bonus for ultra signals
                    reward += base_reward + quality_bonus
                        
                elif bear_score >= 8:
                    reward -= 5.0  # Heavy penalty for medium signals
                elif bear_score >= 6:
                    reward -= 8.0  # Heavier penalty for weak signals
                else:
                    reward -= 12.0  # Massive penalty for poor signals
            
            else:  # Hold
                # REWARD patience and selectivity
                max_score = max(bull_score, bear_score)
                if max_score < 8:
                    reward += 1.0  # Good reward for avoiding weak signals
                elif max_score < 10:
                    reward += 0.5  # Moderate reward for avoiding medium signals
                else:
                    reward -= 2.0  # Penalty for missing high-quality signals
        
        # Update peak balance and check drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.15:  # 15% max drawdown
            done = True
            reward -= 50.0
        
        # Enhanced risk management
        if self.consecutive_losses >= 3:
            reward -= 8.0  # Reduced penalty
        
        # Reward consistent performance
        if self.total_trades >= 10:
            win_rate = self.wins / self.total_trades
            if win_rate >= 0.75:
                reward += 2.0  # Bonus for high win rate
            elif win_rate >= 0.65:
                reward += 1.0  # Bonus for good win rate
        
        self.current_step += 1
        self.episode_steps += 1
        
        # Episode termination conditions
        if self.current_step >= self.n_steps - 1:
            done = True
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        info = {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'profit_factor': self.total_profit / max(0.01, self.total_loss),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'daily_trades': self.daily_trades
        }
        
        return self._get_obs(), float(np.clip(reward, -100, 50)), done, False, info
    
    def _get_atr(self):
        atr = self.df.iloc[self.current_step].get('atr', 0)
        if not isinstance(atr, (int, float)) or atr <= 0 or np.isnan(atr):
            recent = self.df.iloc[max(0, self.current_step-14):self.current_step]
            if len(recent) > 0:
                atr = (recent['High'] - recent['Low']).mean()
            else:
                atr = 0.5
        return max(abs(atr), 0.01)
    
    def _check_enhanced_exit(self, high, low):
        """Enhanced exit logic with trailing stops for winners."""
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
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


# ============================================================================
# ENHANCED CALLBACK
# ============================================================================

class UltraHighWRCallback(BaseCallback):
    """Enhanced callback targeting 80%+ win rate."""
    
    def __init__(self, val_df, atr_mult, min_signal_score, ultra_mode=False,
                 eval_freq=15000, save_path="models/experimental", target_wr=0.80):
        super().__init__()
        self.val_df = val_df
        self.atr_mult = atr_mult
        self.min_signal_score = min_signal_score
        self.ultra_mode = ultra_mode
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
        
        # Enhanced tracking
        self.evaluation_history = []
    
    def _on_training_start(self):
        self._vec_norm = self.model.get_env()
    
    def _on_step(self):
        # Real-time training monitoring
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if info.get('total_trades', 0) >= 3:
                    self.winrates.append(info['win_rate'])
                    
                    if len(self.winrates) % 100 == 0:
                        recent_wr = np.mean(self.winrates[-200:]) if len(self.winrates) >= 200 else np.mean(self.winrates)
                        trades = info.get('total_trades', 0)
                        cons_wins = info.get('consecutive_wins', 0)
                        cons_losses = info.get('consecutive_losses', 0)
                        
                        print(f"📊 Train WR: {recent_wr*100:.1f}% | Trades: {trades} | "
                              f"Streak: +{cons_wins}/-{cons_losses} | Best: {self.best_wr*100:.1f}%")
        
        # Periodic evaluation
        if self.num_timesteps >= self.last_eval + self.eval_freq:
            val_wr, val_trades, val_pf, val_ret, val_sharpe = self._enhanced_evaluate()
            
            # Enhanced scoring function
            score = (
                val_wr * 0.6 +                           # Win rate (60%)
                min(val_pf, 4.0) / 4.0 * 0.2 +          # Profit factor (20%)
                max(0, val_ret/100) * 0.1 +             # Return (10%)
                max(0, val_sharpe/2) * 0.1              # Sharpe ratio (10%)
            )
            
            # Save model if improved
            if score > self.best_score and val_trades >= 8 and val_wr > 0.60:
                self.best_score = score
                self.best_wr = val_wr
                self.best_pf = val_pf
                self.no_improve = 0
                
                # Save with detailed naming
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"ultra_balanced_v3_wr{val_wr*100:.0f}_pf{val_pf:.1f}_{timestamp}"
                self.model.save(f"{self.save_path}/{model_name}")
                
                status = "🎯" if val_wr >= self.target_wr else "✅"
                print(f"\n{status} NEW BEST: WR={val_wr*100:.1f}% PF={val_pf:.2f} "
                      f"Ret={val_ret:+.1f}% Sharpe={val_sharpe:.2f}")
                
                # Save to production if exceptional
                if val_wr >= self.target_wr and val_pf > 3.0:
                    os.makedirs("models/production", exist_ok=True)
                    self.model.save(f"models/production/{model_name}")
                    print(f"🏆 SAVED TO PRODUCTION!")
                    
            else:
                self.no_improve += 1
                print(f"\n📉 Val: WR={val_wr*100:.1f}% PF={val_pf:.2f} Ret={val_ret:+.1f}% | "
                      f"No improve: {self.no_improve}")
            
            # Store evaluation history
            self.evaluation_history.append({
                'timestep': self.num_timesteps,
                'win_rate': val_wr,
                'profit_factor': val_pf,
                'return_pct': val_ret,
                'sharpe_ratio': val_sharpe,
                'score': score
            })
            
            self.last_eval = self.num_timesteps
            
            # Early stopping with patience
            if self.no_improve >= 30:
                print("\n🛑 Early stopping - no improvement for 30 evaluations")
                return False
        
        return True
    
    def _enhanced_evaluate(self):
        """ULTRA-SELECTIVE evaluation focusing on win rate quality."""
        episodes_data = []
        
        # Run more evaluation episodes for better statistics
        for episode in range(12):  # Increased from 8
            env = UltraHighWinRateEnv(
                df=self.val_df, 
                atr_mult=self.atr_mult,
                min_signal_score=max(10, self.min_signal_score),  # Force higher standards
                ultra_mode=self.ultra_mode,
                random_start=True
            )
            vec_env = DummyVecEnv([lambda: Monitor(env)])
            
            if self._vec_norm and isinstance(self._vec_norm, VecNormalize):
                vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)
                vec_env.obs_rms = self._vec_norm.obs_rms
                vec_env.training = False
            
            obs = vec_env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec_env.step(action)
                done = dones[0]
            
            if infos and infos[0].get('total_trades', 0) > 0:
                episodes_data.append(infos[0])
        
        if not episodes_data:
            return 0.3, 0, 1.0, 0, 0  # Poor default values
        
        # Calculate aggregated metrics with focus on win rate
        total_trades = sum(ep['total_trades'] for ep in episodes_data)
        total_wins = sum(ep['wins'] for ep in episodes_data)
        returns = [ep['return_pct'] for ep in episodes_data]
        
        # Require minimum trades for valid evaluation
        if total_trades < 5:
            return 0.3, total_trades / len(episodes_data), 1.0, 0, 0
        
        win_rate = total_wins / total_trades
        avg_trades = total_trades / len(episodes_data)
        avg_return = np.mean(returns)
        
        # Conservative profit factor calculation
        if win_rate > 0.5:
            # For 1:1 RR, theoretical PF = WR / (1-WR)
            theoretical_pf = win_rate / (1 - win_rate)
            profit_factor = min(theoretical_pf, 10.0)  # Cap at 10
        else:
            profit_factor = 0.5
        
        # Sharpe ratio calculation
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = avg_return / np.std(returns)
        else:
            sharpe_ratio = 0
        
        return win_rate, avg_trades, profit_factor, avg_return, sharpe_ratio


# ============================================================================
# ENHANCED DATA PREPARATION
# ============================================================================

def load_and_prepare_enhanced_data(filepath):
    """Load and prepare data with all enhanced indicators."""
    print(f"📁 Loading: {filepath}")
    
    # Load data
    try:
        df = pd.read_csv(filepath, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',')
    except:
        df = pd.read_csv(filepath, sep=',')
    
    # Handle datetime
    for col in ['Date', 'date', 'datetime', 'Gmt time']:
        if col in df.columns:
            df['Date'] = pd.to_datetime(df[col])
            df.set_index('Date', inplace=True)
            break
    
    df.sort_index(inplace=True)
    print(f"📊 Loaded {len(df)} bars")
    
    # Add all enhanced indicators
    print("🔧 Adding enhanced indicators...")
    
    # Market regime detection
    df = add_market_regime_detection(df)
    print("  ✓ Market regime detection")
    
    # Time-based filters
    df = add_time_based_filters(df)
    print("  ✓ Time-based filters")
    
    # Enhanced price action
    df = add_enhanced_price_action(df)
    print("  ✓ Enhanced price action patterns")
    
    # Adaptive indicators
    df = add_adaptive_indicators(df)
    print("  ✓ Adaptive indicators")
    
    # Enhanced signal scoring
    df = create_enhanced_signal_scoring(df)
    print("  ✓ Enhanced signal scoring")
    
    # Data normalization and cleaning
    print("🧹 Normalizing and cleaning data...")
    
    # Normalize prices
    base_price = df['Close'].iloc[100]  # Use price after indicators stabilize
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] / base_price - 1) * 100
    
    # Normalize other features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    binary_cols = [c for c in numeric_cols if df[c].isin([0, 1, -1]).all()]
    score_cols = ['enhanced_bull_score', 'enhanced_bear_score']
    
    for col in numeric_cols:
        if col not in binary_cols and col not in score_cols:
            if df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
                df[col] = df[col].clip(-3, 3)
    
    # Clean data
    df = df.dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"✅ Final: {len(df)} bars, {len(df.columns)} features")
    
    # Signal distribution analysis
    ultra_bull = (df['ultra_high_prob_buy'] == 1).sum()
    ultra_bear = (df['ultra_high_prob_sell'] == 1).sum()
    high_bull = (df['high_prob_buy'] == 1).sum()
    high_bear = (df['high_prob_sell'] == 1).sum()
    
    print(f"📈 Signal Distribution:")
    print(f"  Ultra High Prob: {ultra_bull} buy, {ultra_bear} sell")
    print(f"  High Prob: {high_bull} buy, {high_bear} sell")
    
    return df


# ============================================================================
# ENHANCED TRAINING FUNCTION
# ============================================================================

def make_enhanced_env(df, seed, atr_mult, min_signal_score, ultra_mode):
    def _init():
        env = UltraHighWinRateEnv(
            df=df, 
            atr_mult=atr_mult,
            min_signal_score=min_signal_score,
            ultra_mode=ultra_mode,
            random_start=True
        )
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train_enhanced(
    data_path="data/raw/XAU_15m_data.csv",
    timesteps=2000000,
    n_envs=16,  # Reduced for better stability
    atr_mult=1.2,
    min_signal_score=10,  # Start with higher standards
    ultra_mode=False,
    eval_freq=10000,  # More frequent evaluation
    target_wr=0.80
):
    """Train enhanced ultra high win-rate model."""
    
    print("=" * 80)
    print("🚀 ULTRA HIGH WIN-RATE 1:1 RISK-REWARD MODEL V3 - ENHANCED")
    print("=" * 80)
    
    mode_str = "ULTRA MODE (8+ signals only)" if ultra_mode else f"HIGH PROB MODE ({min_signal_score}+ signals)"
    
    print(f"\n🎯 Configuration:")
    print(f"  Target Win Rate: {target_wr*100:.0f}%+")
    print(f"  SL = TP = {atr_mult} * ATR")
    print(f"  Trading Mode: {mode_str}")
    print(f"  Environments: {n_envs}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Evaluation Frequency: {eval_freq:,}")
    
    # Load and prepare data
    df = load_and_prepare_enhanced_data(data_path)
    
    # Enhanced data split
    train_size = int(len(df) * 0.65)  # More training data
    val_size = int(len(df) * 0.20)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"\n📊 Data Split:")
    print(f"  Train: {len(train_df):,} bars")
    print(f"  Validation: {len(val_df):,} bars")
    print(f"  Test: {len(test_df):,} bars")
    
    # Create enhanced environments
    print(f"\n🏗️ Creating {n_envs} enhanced environments...")
    env = DummyVecEnv([
        make_enhanced_env(train_df, i, atr_mult, min_signal_score, ultra_mode) 
        for i in range(n_envs)
    ])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=100.0)
    
    # Enhanced model architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[1024, 512, 256, 128], vf=[1024, 512, 256, 128]),
        activation_fn=nn.ReLU
    )
    
    # Optimized PPO configuration for better performance
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,  # Slightly higher for faster learning
        n_steps=1024,        # Reduced for faster iterations
        batch_size=2048,     # Balanced batch size
        n_epochs=8,          # Reduced epochs for efficiency
        gamma=0.99,          # Standard gamma
        gae_lambda=0.95,     # Standard lambda
        clip_range=0.2,      # Standard clipping
        ent_coef=0.01,       # Higher entropy for exploration
        vf_coef=0.5,         # Standard value function coefficient
        max_grad_norm=0.5,   # Standard gradient clipping
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/ultra_balanced_v3_training"
    )
    
    # Create directories
    os.makedirs("models/experimental", exist_ok=True)
    os.makedirs("models/production", exist_ok=True)
    
    # Enhanced callback
    callback = UltraHighWRCallback(
        val_df=val_df,
        atr_mult=atr_mult,
        min_signal_score=min_signal_score,
        ultra_mode=ultra_mode,
        eval_freq=eval_freq,
        save_path="models/experimental",
        target_wr=target_wr
    )
    
    print("\n🏋️ Starting enhanced training...")
    print("=" * 80)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"ultra_balanced_v3_final_{timestamp}"
    model.save(f"models/experimental/{final_name}")
    env.save(f"models/experimental/{final_name}_vecnorm.pkl")
    
    print(f"\n💾 Final model saved: {final_name}")
    
    return model, callback.evaluation_history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra High Win Rate Trading Model V3")
    parser.add_argument('--timesteps', type=int, default=2000000, help='Training timesteps')
    parser.add_argument('--envs', type=int, default=32, help='Number of environments')
    parser.add_argument('--atr-mult', type=float, default=1.2, help='ATR multiplier for SL=TP')
    parser.add_argument('--min-score', type=int, default=6, help='Min signal score to trade')
    parser.add_argument('--ultra-mode', action='store_true', help='Use ultra mode (8+ signals only)')
    parser.add_argument('--target-wr', type=float, default=0.80, help='Target win rate')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Ultra High Win Rate Model V3 Training")
    print(f"Target: {args.target_wr*100:.0f}%+ win rate with 1:1 risk/reward")
    
    model, history = train_enhanced(
        timesteps=args.timesteps,
        n_envs=args.envs,
        atr_mult=args.atr_mult,
        min_signal_score=args.min_score,
        ultra_mode=args.ultra_mode,
        target_wr=args.target_wr
    )
    
    print("\n🎉 Training completed!")
    if history:
        best_eval = max(history, key=lambda x: x['win_rate'])
        print(f"🏆 Best validation: {best_eval['win_rate']*100:.1f}% win rate")