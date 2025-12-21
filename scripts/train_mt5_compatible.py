#!/usr/bin/env python3
"""
MT5-Compatible High Win-Rate Training
=====================================
Uses ONLY features that MT5 can easily calculate:
- OHLC (normalized)
- ATR, RSI, MACD, MA20, Stochastic
- No SST or complex derived features

Target: 80% win rate with 1:4 TP/SL ratio
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pandas_ta as pta
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import warnings
import argparse
warnings.filterwarnings('ignore')


# Feature names - MUST match EA exactly
FEATURE_NAMES = [
    'open', 'high', 'low', 'close',  # 4 - price normalized
    'atr',                            # 1 - ATR normalized
    'ma20',                           # 1 - MA20 normalized
    'rsi',                            # 1 - RSI normalized [-3,3]
    'macd', 'macd_signal', 'macd_hist',  # 3 - MACD normalized
    'trend',                          # 1 - trend direction
    'stoch_k',                        # 1 - Stochastic K normalized
    'stoch_d',                        # 1 - Stochastic D normalized
    'stoch_cross',                    # 1 - K vs D
    'stoch_ob',                       # 1 - overbought
    'stoch_os',                       # 1 - oversold
]
NUM_MARKET_FEATURES = 16
NUM_STATE_FEATURES = 2  # position, unrealized_pnl
NUM_FEATURES = NUM_MARKET_FEATURES + NUM_STATE_FEATURES  # 18 total


class MT5CompatibleEnv(gym.Env):
    """Environment with MT5-compatible features only."""
    
    def __init__(self, df, window_size=20, initial_balance=10000.0,
                 tp_atr_mult=0.5, sl_atr_mult=2.0, random_start=True):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.random_start = random_start
        
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, NUM_FEATURES),
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
        self.max_steps = min(1500, self.n_steps - self.current_step - 1)
        self.episode_steps = 0
        self.last_trade_step = -5
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        # Get market features
        market_cols = ['open', 'high', 'low', 'close', 'atr', 'ma20', 'rsi',
                       'macd', 'macd_signal', 'macd_hist', 'trend',
                       'stoch_k', 'stoch_d', 'stoch_cross', 'stoch_ob', 'stoch_os']
        
        obs_data = self.df[market_cols].iloc[start:end].values.copy()
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), NUM_MARKET_FEATURES))
            obs_data = np.vstack([pad, obs_data])
        
        # Add state features
        state = np.zeros((self.window_size, NUM_STATE_FEATURES))
        
        # Position: 1=long, -1=short, 0=none
        if self.position == 'long':
            state[:, 0] = 1.0
        elif self.position == 'short':
            state[:, 0] = -1.0
        
        # Unrealized PnL
        if self.position and self.entry_price > 0 and self.entry_atr > 0:
            price = self.df.iloc[self.current_step]['close_raw']
            if self.position == 'long':
                pnl = (price - self.entry_price) / self.entry_atr
            else:
                pnl = (self.entry_price - price) / self.entry_atr
            state[:, 1] = np.clip(pnl, -3, 3)
        
        obs = np.hstack([obs_data, state])
        return obs.astype(np.float32)
    
    def _get_atr(self):
        return max(self.df.iloc[self.current_step]['atr_raw'], 0.01)
    
    def step(self, action):
        reward = 0.0
        done = False
        
        current_price = self.df.iloc[self.current_step]['close_raw']
        current_atr = self._get_atr()
        
        # Process action
        if action == 1 and self.position is None:  # BUY
            if self.current_step - self.last_trade_step >= 3:
                self.position = 'long'
                self.entry_price = current_price
                self.entry_atr = current_atr
                self.last_trade_step = self.current_step
                reward = -0.1  # Small cost
        
        elif action == 2 and self.position is None:  # SELL
            if self.current_step - self.last_trade_step >= 3:
                self.position = 'short'
                self.entry_price = current_price
                self.entry_atr = current_atr
                self.last_trade_step = self.current_step
                reward = -0.1
        
        # Check TP/SL
        if self.position:
            tp_dist = self.entry_atr * self.tp_atr_mult
            sl_dist = self.entry_atr * self.sl_atr_mult
            
            high = self.df.iloc[self.current_step]['high_raw']
            low = self.df.iloc[self.current_step]['low_raw']
            
            if self.position == 'long':
                if high >= self.entry_price + tp_dist:
                    reward = 5.0
                    self.wins += 1
                    self.total_trades += 1
                    self.position = None
                elif low <= self.entry_price - sl_dist:
                    reward = -2.0
                    self.losses += 1
                    self.total_trades += 1
                    self.position = None
            
            elif self.position == 'short':
                if low <= self.entry_price - tp_dist:
                    reward = 5.0
                    self.wins += 1
                    self.total_trades += 1
                    self.position = None
                elif high >= self.entry_price + sl_dist:
                    reward = -2.0
                    self.losses += 1
                    self.total_trades += 1
                    self.position = None
        
        # Move forward
        self.current_step += 1
        self.episode_steps += 1
        
        if self.episode_steps >= self.max_steps:
            done = True
        if self.current_step >= self.n_steps - 1:
            done = True
        
        info = {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses
        }
        
        return self._get_obs(), float(reward), done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


def prepare_data(data_path):
    """Prepare data with MT5-compatible features."""
    
    print(f"Loading: {data_path}")
    
    # Try different separators
    try:
        df = pd.read_csv(data_path)
        if len(df.columns) == 1:
            df = pd.read_csv(data_path, sep=';')
    except:
        df = pd.read_csv(data_path, sep=';')
    
    print(f"Columns: {df.columns.tolist()}")
    
    # Find date column
    for col in ['Date', 'datetime', 'time', 'Datetime']:
        if col in df.columns:
            df['Date'] = pd.to_datetime(df[col])
            df.set_index('Date', inplace=True)
            break
    
    df.sort_index(inplace=True)
    print(f"Loaded {len(df)} bars")
    
    # Keep raw prices for trading logic
    df['open_raw'] = df['Open']
    df['high_raw'] = df['High']
    df['low_raw'] = df['Low']
    df['close_raw'] = df['Close']
    
    # Calculate indicators on RAW prices
    df['atr_raw'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ma20_raw'] = pta.sma(df['Close'], length=20)
    df['rsi_raw'] = pta.rsi(df['Close'], length=14)
    
    macd = pta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['macd_raw'] = macd.iloc[:, 0]
        df['macd_signal_raw'] = macd.iloc[:, 1]
        df['macd_hist_raw'] = macd.iloc[:, 2]
    
    stoch = pta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df['stoch_k_raw'] = stoch.iloc[:, 0]
        df['stoch_d_raw'] = stoch.iloc[:, 1]
    
    # === NORMALIZE FEATURES (same as EA will do) ===
    
    # Use rolling base price for normalization (like EA does per window)
    base_price = df['Close'].rolling(20).apply(lambda x: x[0], raw=True)
    base_price = base_price.fillna(df['Close'].iloc[0])
    
    # Price features: (price / base - 1) * 100
    df['open'] = (df['Open'] / base_price - 1) * 100
    df['high'] = (df['High'] / base_price - 1) * 100
    df['low'] = (df['Low'] / base_price - 1) * 100
    df['close'] = (df['Close'] / base_price - 1) * 100
    
    # ATR: normalized by base price
    df['atr'] = df['atr_raw'] / base_price * 100
    
    # MA20: normalized like price
    df['ma20'] = (df['ma20_raw'] / base_price - 1) * 100
    
    # RSI: center at 0, scale to [-3, 3]
    df['rsi'] = ((df['rsi_raw'] - 50) / 25).clip(-3, 3)
    
    # MACD: normalize by base price, clip
    df['macd'] = (df['macd_raw'] / base_price * 1000).clip(-3, 3)
    df['macd_signal'] = (df['macd_signal_raw'] / base_price * 1000).clip(-3, 3)
    df['macd_hist'] = (df['macd_hist_raw'] / base_price * 1000).clip(-3, 3)
    
    # Trend: 1 if above MA20, -1 if below
    df['trend'] = np.where(df['Close'] > df['ma20_raw'], 1.0, -1.0)
    
    # Stochastic: center at 0, scale
    df['stoch_k'] = ((df['stoch_k_raw'] - 50) / 25).clip(-3, 3)
    df['stoch_d'] = ((df['stoch_d_raw'] - 50) / 25).clip(-3, 3)
    
    # Stochastic cross: 1 if K > D, -1 if K < D
    df['stoch_cross'] = np.where(df['stoch_k_raw'] > df['stoch_d_raw'], 1.0, -1.0)
    
    # Overbought/Oversold
    df['stoch_ob'] = np.where(df['stoch_k_raw'] > 80, 1.0, 0.0)
    df['stoch_os'] = np.where(df['stoch_k_raw'] < 20, 1.0, 0.0)
    
    # Clean up
    df = df.dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Final: {len(df)} bars, {NUM_FEATURES} features")
    
    return df


def make_env(df, seed, tp_atr_mult, sl_atr_mult):
    def _init():
        env = MT5CompatibleEnv(df, tp_atr_mult=tp_atr_mult, sl_atr_mult=sl_atr_mult)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


class TrainingCallback(BaseCallback):
    def __init__(self, val_df, tp_atr_mult, sl_atr_mult, eval_freq=25000, save_path="models/production"):
        super().__init__()
        self.val_df = val_df
        self.tp_atr_mult = tp_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_wr = 0.0
        self.last_eval = 0
        self.no_improve = 0
        self.train_wrs = []
    
    def _on_step(self):
        # Track training win rate
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if info.get('total_trades', 0) >= 5:
                    self.train_wrs.append(info['win_rate'])
                    
                    if len(self.train_wrs) % 50 == 0:
                        recent_wr = np.mean(self.train_wrs[-100:])
                        print(f"📊 Train WR: {recent_wr*100:.1f}% | Best Val: {self.best_wr*100:.1f}%")
        
        # Evaluate periodically
        if self.num_timesteps >= self.last_eval + self.eval_freq:
            self.last_eval = self.num_timesteps
            val_wr, val_trades = self._evaluate()
            
            print(f"📉 Val WR: {val_wr*100:.1f}% | Trades: {val_trades}")
            
            if val_wr > self.best_wr and val_trades >= 50:
                self.best_wr = val_wr
                self.no_improve = 0
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"{self.save_path}/mt5_compat_{val_wr*100:.0f}pct_{timestamp}"
                self.model.save(save_name)
                print(f"💾 Saved: {save_name}")
            else:
                self.no_improve += 1
            
            if self.no_improve >= 15:
                print("🛑 Early stopping")
                return False
        
        return True
    
    def _evaluate(self, n_episodes=10):
        total_trades = 0
        total_wins = 0
        
        for _ in range(n_episodes):
            env = MT5CompatibleEnv(self.val_df, tp_atr_mult=self.tp_atr_mult, 
                                   sl_atr_mult=self.sl_atr_mult, random_start=True)
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            
            total_trades += info['total_trades']
            total_wins += info['wins']
        
        wr = total_wins / max(1, total_trades)
        return wr, total_trades


def train(data_path, timesteps, n_envs, tp_atr_mult, sl_atr_mult, eval_freq):
    print("=" * 60)
    print("🚀 MT5-COMPATIBLE HIGH WIN-RATE TRAINING")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  TP: {tp_atr_mult} × ATR | SL: {sl_atr_mult} × ATR")
    print(f"  Ratio: 1:{sl_atr_mult/tp_atr_mult:.1f}")
    print(f"  Expected WR: ~{sl_atr_mult/(tp_atr_mult+sl_atr_mult)*100:.0f}%")
    print(f"  Features: {NUM_FEATURES} (MT5 compatible)")
    
    df = prepare_data(data_path)
    
    # Split data
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create environments
    print(f"\nCreating {n_envs} environments...")
    env = DummyVecEnv([make_env(train_df, i, tp_atr_mult, sl_atr_mult) for i in range(n_envs)])
    
    # Model
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=nn.Tanh
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=2048,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/mt5_compat_training"
    )
    
    os.makedirs("models/production", exist_ok=True)
    
    callback = TrainingCallback(
        val_df=val_df,
        tp_atr_mult=tp_atr_mult,
        sl_atr_mult=sl_atr_mult,
        eval_freq=eval_freq,
        save_path="models/production"
    )
    
    print("\n🏋️ Starting training...")
    print("=" * 60)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    # Final save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/production/mt5_compat_final_{timestamp}")
    
    # Test evaluation
    print("\n" + "=" * 60)
    print("🧪 TEST EVALUATION")
    print("=" * 60)
    
    total_trades = 0
    total_wins = 0
    
    for ep in range(10):
        env = MT5CompatibleEnv(test_df, tp_atr_mult=tp_atr_mult, 
                               sl_atr_mult=sl_atr_mult, random_start=True)
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
        
        print(f"Ep {ep+1}: Trades={info['total_trades']}, WR={info['win_rate']*100:.1f}%")
        total_trades += info['total_trades']
        total_wins += info['wins']
    
    final_wr = total_wins / max(1, total_trades)
    print(f"\n{'='*50}")
    print(f"FINAL: {total_trades} trades, {final_wr*100:.1f}% WR")
    print(f"{'='*50}")
    
    if final_wr >= 0.75:
        print("🎉 TARGET ACHIEVED!")
    else:
        print(f"⚠️ Below target: {final_wr*100:.1f}% < 75%")
    
    return model, final_wr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/raw/XAU_15m_data.csv')
    parser.add_argument('--timesteps', type=int, default=500000)
    parser.add_argument('--envs', type=int, default=24)
    parser.add_argument('--tp-atr', type=float, default=0.5)
    parser.add_argument('--sl-atr', type=float, default=2.0)
    parser.add_argument('--eval-freq', type=int, default=25000)
    
    args = parser.parse_args()
    
    train(
        data_path=args.data,
        timesteps=args.timesteps,
        n_envs=args.envs,
        tp_atr_mult=args.tp_atr,
        sl_atr_mult=args.sl_atr,
        eval_freq=args.eval_freq
    )
