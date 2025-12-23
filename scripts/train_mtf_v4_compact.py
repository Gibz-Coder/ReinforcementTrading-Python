#!/usr/bin/env python3
"""
Compact Multi-Timeframe V4 - 75%+ Win Rate Target
=================================================
Uses real 1D/4H/1H data to filter 15M trades for higher win rates.
"""

import numpy as np
import pandas as pd
import pandas_ta as pta
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import os
import warnings
warnings.filterwarnings('ignore')


def load_timeframe_data(base_path="data/raw"):
    """Load all timeframes quickly."""
    timeframes = ['15m', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        try:
            df = pd.read_csv(f"{base_path}/XAU_{tf}_data.csv", sep=';')
            if len(df.columns) == 1:
                df = pd.read_csv(f"{base_path}/XAU_{tf}_data.csv", sep=',')
            
            # Simple datetime handling
            if 'Gmt time' in df.columns:
                df['Date'] = pd.to_datetime(df['Gmt time'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            if 'Date' in df.columns:
                df.set_index('Date', inplace=True)
            
            df.sort_index(inplace=True)
            data[tf] = df
            print(f"✅ {tf.upper()}: {len(df):,} bars")
        except Exception as e:
            print(f"❌ {tf.upper()}: {e}")
            data[tf] = None
    
    return data


def add_mtf_signals(df_15m, df_1h, df_4h, df_1d):
    """Add multi-timeframe trend signals."""
    
    # Add basic indicators to 15M
    df_15m['rsi'] = pta.rsi(df_15m['Close'], 14)
    df_15m['adx'] = pta.adx(df_15m['High'], df_15m['Low'], df_15m['Close'], 14).iloc[:, 0]
    df_15m['atr'] = pta.atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14)
    df_15m['ema12'] = pta.ema(df_15m['Close'], 12)
    df_15m['ema26'] = pta.ema(df_15m['Close'], 26)
    
    # Add higher timeframe trends
    for htf_name, htf_df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if htf_df is not None:
            htf_df['ema_fast'] = pta.ema(htf_df['Close'], 8)
            htf_df['ema_slow'] = pta.ema(htf_df['Close'], 21)
            htf_df['trend'] = np.where(htf_df['ema_fast'] > htf_df['ema_slow'], 1, -1)
            
            # Align with 15M (simple forward fill)
            df_15m[f'{htf_name}_trend'] = 0
            if hasattr(df_15m.index, 'to_pydatetime') and hasattr(htf_df.index, 'to_pydatetime'):
                for i in range(len(df_15m)):
                    try:
                        current_time = df_15m.index[i]
                        recent_htf = htf_df[htf_df.index <= current_time]
                        if len(recent_htf) > 0:
                            df_15m.iloc[i, df_15m.columns.get_loc(f'{htf_name}_trend')] = recent_htf['trend'].iloc[-1]
                    except:
                        continue
    
    # Multi-timeframe alignment
    df_15m['mtf_bull'] = (
        (df_15m['ema12'] > df_15m['ema26']).astype(int) +
        (df_15m.get('1h_trend', 0) == 1).astype(int) +
        (df_15m.get('4h_trend', 0) == 1).astype(int) +
        (df_15m.get('1d_trend', 0) == 1).astype(int)
    )
    
    df_15m['mtf_bear'] = (
        (df_15m['ema12'] < df_15m['ema26']).astype(int) +
        (df_15m.get('1h_trend', 0) == -1).astype(int) +
        (df_15m.get('4h_trend', 0) == -1).astype(int) +
        (df_15m.get('1d_trend', 0) == -1).astype(int)
    )
    
    # Trading sessions
    if hasattr(df_15m.index, 'hour'):
        df_15m['hour'] = df_15m.index.hour
    else:
        df_15m['hour'] = 12
    
    df_15m['high_liquidity'] = ((df_15m['hour'] >= 8) & (df_15m['hour'] <= 21)).astype(int)
    
    # Final signals (require 3+ timeframe alignment)
    df_15m['bull_score'] = (
        (df_15m['mtf_bull'] >= 3).astype(int) * 4 +
        ((df_15m['rsi'] > 45) & (df_15m['rsi'] < 65)).astype(int) * 2 +
        (df_15m['adx'] > 25).astype(int) * 2 +
        df_15m['high_liquidity'] * 2
    )
    
    df_15m['bear_score'] = (
        (df_15m['mtf_bear'] >= 3).astype(int) * 4 +
        ((df_15m['rsi'] < 55) & (df_15m['rsi'] > 35)).astype(int) * 2 +
        (df_15m['adx'] > 25).astype(int) * 2 +
        df_15m['high_liquidity'] * 2
    )
    
    # High probability signals
    df_15m['high_prob_bull'] = (df_15m['bull_score'] >= 8).astype(int)
    df_15m['high_prob_bear'] = (df_15m['bear_score'] >= 8).astype(int)
    
    return df_15m


class MTFTradingEnv(gym.Env):
    """Compact multi-timeframe trading environment."""
    
    def __init__(self, df, window_size=20, curriculum_stage=1):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.curriculum_stage = curriculum_stage
        
        self.action_space = spaces.Discrete(3)
        
        # Compact feature set
        self.features = ['Close', 'rsi', 'adx', 'atr', 'bull_score', 'bear_score', 
                        'mtf_bull', 'mtf_bear', 'high_liquidity']
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, len(self.features) + 2),
            dtype=np.float32
        )
        
        self._init_state()
    
    def _init_state(self):
        self.current_step = np.random.randint(self.window_size, self.n_steps - 100)
        self.balance = 10000.0
        self.position = None
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.total_trades = 0
        self.wins = 0
        self.episode_steps = 0
        self.max_episode_steps = 300
        self.last_trade_step = -20
        self.peak_balance = 10000.0
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        obs_data = self.df[self.features].iloc[start:end].values
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), len(self.features)))
            obs_data = np.vstack([pad, obs_data])
        
        # Add position and win rate
        extra = np.zeros((self.window_size, 2))
        if self.position == 'long':
            extra[:, 0] = 1.0
        elif self.position == 'short':
            extra[:, 0] = -1.0
        
        extra[:, 1] = self.wins / max(1, self.total_trades)
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        return np.nan_to_num(np.clip(obs, -5, 5))
    
    def step(self, action):
        reward = 0.0
        done = False
        
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        row = self.df.iloc[self.current_step]
        price = row['Close']
        high = row['High']
        low = row['Low']
        
        # Check exit
        if self.position:
            distance = self.entry_atr * 1.0  # 1:1 RR
            
            if self.position == 'long':
                if high >= self.entry_price + distance:
                    reward = 100.0
                    if self.total_trades >= 3 and self.wins / self.total_trades >= 0.75:
                        reward += 50.0
                    self.wins += 1
                    self.total_trades += 1
                    self.balance *= 1.02
                    self.position = None
                elif low <= self.entry_price - distance:
                    reward = -50.0
                    self.total_trades += 1
                    self.balance *= 0.98
                    self.position = None
            else:
                if low <= self.entry_price - distance:
                    reward = 100.0
                    if self.total_trades >= 3 and self.wins / self.total_trades >= 0.75:
                        reward += 50.0
                    self.wins += 1
                    self.total_trades += 1
                    self.balance *= 1.02
                    self.position = None
                elif high >= self.entry_price + distance:
                    reward = -50.0
                    self.total_trades += 1
                    self.balance *= 0.98
                    self.position = None
        
        # Trading logic
        can_trade = (
            (self.current_step - self.last_trade_step) >= 20 and
            self.position is None and
            row['high_liquidity'] == 1
        )
        
        if can_trade:
            min_score = 8 if self.curriculum_stage == 1 else 6
            
            if action == 1 and row['bull_score'] >= min_score:
                self.position = 'long'
                self.entry_price = price
                self.entry_atr = max(row['atr'], 0.01)
                self.last_trade_step = self.current_step
                reward += 10.0
            elif action == 2 and row['bear_score'] >= min_score:
                self.position = 'short'
                self.entry_price = price
                self.entry_atr = max(row['atr'], 0.01)
                self.last_trade_step = self.current_step
                reward += 10.0
            elif action != 0:
                reward -= 5.0
        
        # Risk management
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        if (self.peak_balance - self.balance) / self.peak_balance > 0.15:
            done = True
            reward -= 100.0
        
        self.current_step += 1
        self.episode_steps += 1
        
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        return self._get_obs(), reward, done, False, self._get_info()
    
    def _get_info(self):
        return {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'balance': self.balance,
            'return_pct': (self.balance - 10000) / 100,
            'curriculum_stage': self.curriculum_stage
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


class MTFCallback(BaseCallback):
    """Compact callback for MTF training."""
    
    def __init__(self, val_df, eval_freq=3000):
        super().__init__()
        self.val_df = val_df
        self.eval_freq = eval_freq
        self.best_wr = 0.0
        self.curriculum_stage = 1
        self.no_improve = 0
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0:
            wr, trades, ret = self._evaluate()
            
            print(f"\n🏛️ MTF Eval at {self.num_timesteps:,}:")
            print(f"   Win Rate: {wr*100:.1f}%")
            print(f"   Trades: {trades:.1f}")
            print(f"   Return: {ret:+.1f}%")
            print(f"   Stage: {self.curriculum_stage}")
            
            if wr > self.best_wr and trades >= 3:
                self.best_wr = wr
                self.no_improve = 0
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"mtf_v4_wr{wr*100:.0f}_{timestamp}"
                self.model.save(f"models/experimental/{name}")
                print(f"✅ NEW BEST: {name}")
                
                if wr >= 0.75:
                    os.makedirs("models/production", exist_ok=True)
                    self.model.save(f"models/production/{name}")
                    print(f"🏆 PRODUCTION MODEL!")
            else:
                self.no_improve += 1
            
            # Curriculum progression
            if wr >= 0.65 and self.curriculum_stage == 1:
                self.curriculum_stage = 2
                print("📈 CURRICULUM STAGE 2")
                self._update_envs()
            
            if self.no_improve >= 12:
                print("🛑 Early stopping")
                return False
        
        return True
    
    def _update_envs(self):
        if hasattr(self.model.env, 'envs'):
            for env in self.model.env.envs:
                if hasattr(env, 'curriculum_stage'):
                    env.curriculum_stage = self.curriculum_stage
    
    def _evaluate(self):
        episodes = []
        for _ in range(6):
            env = MTFTradingEnv(self.val_df, curriculum_stage=self.curriculum_stage)
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            
            if info['total_trades'] > 0:
                episodes.append(info)
        
        if not episodes:
            return 0.4, 0, -10
        
        total_trades = sum(ep['total_trades'] for ep in episodes)
        total_wins = sum(ep['wins'] for ep in episodes)
        returns = [ep['return_pct'] for ep in episodes]
        
        return total_wins / max(1, total_trades), total_trades / len(episodes), np.mean(returns)


def train_mtf_compact(timesteps=300000, n_envs=6):
    """Train compact multi-timeframe model."""
    
    print("🏛️ COMPACT MULTI-TIMEFRAME V4")
    print("Target: 75%+ win rate using real MTF data")
    
    # Load data
    data = load_timeframe_data()
    df = add_mtf_signals(data['15m'], data['1h'], data['4h'], data['1d'])
    
    # Normalize
    base_price = df['Close'].iloc[100]
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] / base_price - 1) * 100
    
    for col in ['rsi', 'adx']:
        if df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            df[col] = df[col].clip(-3, 3)
    
    df = df.dropna().reset_index(drop=True)
    
    # Split
    train_size = int(len(df) * 0.7)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+int(len(df)*0.2)].reset_index(drop=True)
    
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")
    
    # Signal stats
    high_bull = (train_df['high_prob_bull'] == 1).sum()
    high_bear = (train_df['high_prob_bear'] == 1).sum()
    print(f"High Prob Signals: {high_bull} bull, {high_bear} bear")
    
    # Create env
    env = DummyVecEnv([lambda: Monitor(MTFTradingEnv(train_df)) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=3.0)
    
    # Model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=256,
        batch_size=512,
        n_epochs=6,
        gamma=0.99,
        policy_kwargs=dict(net_arch=[512, 256, 128]),
        verbose=1
    )
    
    # Train
    os.makedirs("models/experimental", exist_ok=True)
    callback = MTFCallback(val_df)
    
    print("\n🏋️ Training MTF model...")
    model.learn(timesteps, callback=callback, progress_bar=True)
    
    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/experimental/mtf_v4_final_{timestamp}")
    
    print(f"\n🎉 Best MTF win rate: {callback.best_wr*100:.1f}%")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=300000)
    parser.add_argument('--envs', type=int, default=6)
    args = parser.parse_args()
    
    train_mtf_compact(args.timesteps, args.envs)