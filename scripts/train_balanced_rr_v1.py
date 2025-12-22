#!/usr/bin/env python3
"""
Balanced 1:1 Risk-Reward Trading Model V1
==========================================
Based on high win-rate v7, modified to use 1:1 SL/TP ratio:
- Equal SL and TP distances (balanced risk-reward)
- Focus on profitable entries with 50%+ win rate needed
- Signal quality becomes critical for profitability
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'indicators'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'environments'))

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

from stochastic_supertrend import add_stochastic_supertrend_features, get_stochastic_supertrend_signals


class BalancedRREnv(gym.Env):
    """
    Trading environment with balanced 1:1 risk-reward ratio.
    SL distance = TP distance (both based on ATR multiplier)
    """
    
    def __init__(self, df, window_size=20, initial_balance=10000.0,
                 atr_mult=1.0,  # Both SL and TP use same ATR multiplier
                 spread_pct=0.0001, random_start=True):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.random_start = random_start
        
        # 1:1 Risk-Reward - same multiplier for both
        self.atr_mult = atr_mult
        self.spread_pct = spread_pct
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        self.feature_columns = [col for col in df.columns if col not in ['Date', 'datetime']]
        self.num_features = len(self.feature_columns)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.num_features + 3),
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
        self.max_episode_steps = min(1500, self.n_steps - self.current_step - 1)
        self.episode_steps = 0
        self.last_trade_step = -5
        self.peak_balance = self.initial_balance

    def _get_atr(self):
        """Get current ATR value."""
        atr = self.df.iloc[self.current_step].get('atr', 0)
        if not isinstance(atr, (int, float)) or atr <= 0 or np.isnan(atr):
            recent = self.df.iloc[max(0, self.current_step-14):self.current_step]
            if len(recent) > 0:
                atr = (recent['High'] - recent['Low']).mean()
            else:
                atr = 0.5
        return max(abs(atr), 0.01)
    
    def _get_signal(self):
        """Get trading signal: 1=buy, -1=sell, 0=neutral."""
        if self.current_step >= len(self.df):
            return 0
        
        row = self.df.iloc[self.current_step]
        
        buy_sig = row.get('sst_buy_signal', 0) == 1
        sell_sig = row.get('sst_sell_signal', 0) == 1
        sst_dir = row.get('sst_direction', 0)
        trend = row.get('trend', 0)
        
        # Stronger signal requirements for 1:1 RR
        buy_score = int(buy_sig) * 2 + (1 if sst_dir == -1 else 0) + (1 if trend == 1 else 0)
        sell_score = int(sell_sig) * 2 + (1 if sst_dir == 1 else 0) + (1 if trend == -1 else 0)
        
        # Require stronger signals for 1:1 (need >50% accuracy)
        if buy_score >= 3 and buy_score > sell_score:
            return 1
        elif sell_score >= 3 and sell_score > buy_score:
            return -1
        return 0

    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        obs_data = self.df[self.feature_columns].iloc[start:end].values.copy()
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), self.num_features))
            obs_data = np.vstack([pad, obs_data])
        
        extra = np.zeros((self.window_size, 3))
        
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
        
        extra[:, 2] = self._get_signal()
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=3.0, neginf=-3.0)
        return np.clip(obs, -5.0, 5.0)
    
    def _check_exit(self, high, low):
        """Check if position hits SL or TP (both at same distance)."""
        if not self.position or self.entry_price == 0 or self.entry_atr <= 0:
            return False, 0.0, False
        
        # 1:1 ratio - same distance for both
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
        signal = self._get_signal()
        
        # Check exit first
        if self.position:
            closed, pnl, is_win = self._check_exit(high, low)
            if closed:
                self.total_trades += 1
                if is_win:
                    self.wins += 1
                    self.total_profit += pnl
                    reward = 10.0
                else:
                    self.losses += 1
                    self.total_loss += abs(pnl)
                    reward = -10.0
                
                self.balance *= (1 + pnl)
                self.position = None
                self.entry_price = 0.0
                self.entry_atr = 0.0
        
        # Process new entries - stronger signal requirement for 1:1
        can_trade = (self.current_step - self.last_trade_step) >= 3
        
        if self.position is None and can_trade:
            if action == 1:  # Buy
                self.position = 'long'
                self.entry_price = price
                self.entry_atr = self._get_atr()
                self.last_trade_step = self.current_step
                # Higher rewards for signal alignment (critical for 1:1)
                if signal == 1:
                    reward += 3.0
                elif signal == -1:
                    reward -= 5.0  # Stronger penalty for counter-signal
                else:
                    reward -= 1.0  # Small penalty for no signal
                    
            elif action == 2:  # Sell
                self.position = 'short'
                self.entry_price = price
                self.entry_atr = self._get_atr()
                self.last_trade_step = self.current_step
                if signal == -1:
                    reward += 3.0
                elif signal == 1:
                    reward -= 5.0
                else:
                    reward -= 1.0
        
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
        if drawdown > 0.25:
            done = True
            reward -= 25.0
        
        info = {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'profit_factor': self.total_profit / max(0.01, self.total_loss)
        }
        
        return self._get_obs(), float(np.clip(reward, -35, 20)), done, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


class TrainingCallback(BaseCallback):
    """Callback with validation for 1:1 RR model."""
    
    def __init__(self, val_df, atr_mult, eval_freq=25000, save_path="models/experimental"):
        super().__init__()
        self.val_df = val_df
        self.atr_mult = atr_mult
        self.eval_freq = eval_freq
        self.save_path = save_path
        
        self.best_score = -np.inf  # Track best combined score
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
                if info.get('total_trades', 0) >= 5:
                    self.winrates.append(info['win_rate'])
                    
                    if len(self.winrates) % 100 == 0:
                        wr = np.mean(self.winrates[-200:]) if len(self.winrates) >= 200 else np.mean(self.winrates)
                        print(f"📊 Train WR: {wr*100:.1f}% | Best: WR={self.best_wr*100:.1f}% PF={self.best_pf:.2f}")
        
        if self.num_timesteps >= self.last_eval + self.eval_freq:
            val_wr, val_trades, val_pf, val_ret = self._evaluate()
            
            # Combined score: win rate + profit factor (both matter for 1:1)
            score = val_wr * 0.5 + min(val_pf, 2.0) * 0.25 + max(0, val_ret/100) * 0.25
            
            if score > self.best_score and val_trades >= 10 and val_wr > 0.50:
                self.best_score = score
                self.best_wr = val_wr
                self.best_pf = val_pf
                self.no_improve = 0
                self.model.save(f"{self.save_path}/balanced_rr_best")
                print(f"\n🎯 NEW BEST: WR={val_wr*100:.1f}% PF={val_pf:.2f} Ret={val_ret:.1f}%")
            else:
                self.no_improve += 1
                print(f"\n📉 Val: WR={val_wr*100:.1f}% PF={val_pf:.2f} | No improve: {self.no_improve}")
            
            self.last_eval = self.num_timesteps
            
            if self.no_improve >= 20:
                print("\n🛑 Early stopping")
                return False
        
        return True
    
    def _evaluate(self):
        env = BalancedRREnv(df=self.val_df, atr_mult=self.atr_mult, random_start=True)
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        if self._vec_norm and isinstance(self._vec_norm, VecNormalize):
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=5.0)
            vec_env.obs_rms = self._vec_norm.obs_rms
            vec_env.training = False
        
        total_trades, total_wins, total_profit, total_loss = 0, 0, 0.0, 0.0
        returns = []
        
        for _ in range(5):
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
        # Estimate profit factor from win rate for 1:1 RR
        pf = wr / (1 - wr + 0.001) if wr < 1 else 10.0
        avg_ret = np.mean(returns) if returns else 0
        
        return wr, total_trades / 5, pf, avg_ret


def load_and_prepare_data(filepath):
    """Load and prepare data."""
    print(f"Loading: {filepath}")
    
    try:
        df = pd.read_csv(filepath, sep=';')
        if len(df.columns) == 1:
            df = pd.read_csv(filepath, sep=',')
    except:
        df = pd.read_csv(filepath, sep=',')
    
    for col in ['Date', 'date', 'datetime']:
        if col in df.columns:
            df['Date'] = pd.to_datetime(df[col])
            df.set_index('Date', inplace=True)
            break
    
    df.sort_index(inplace=True)
    print(f"Loaded {len(df)} bars")
    
    df['atr'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['ma_20'] = pta.sma(df['Close'], length=20)
    df['ma_50'] = pta.sma(df['Close'], length=50)
    df['rsi'] = pta.rsi(df['Close'], length=14)
    
    macd = pta.macd(df['Close'])
    if macd is not None:
        df['macd'] = macd.iloc[:, 0]
        df['macd_signal'] = macd.iloc[:, 1]
        df['macd_hist'] = macd.iloc[:, 2]
    
    # Trend indicators
    df['trend'] = 0
    df.loc[df['Close'] > df['ma_20'], 'trend'] = 1
    df.loc[df['Close'] < df['ma_20'], 'trend'] = -1
    
    # Strong trend (for 1:1 we want strong signals)
    df['strong_trend'] = 0
    df.loc[(df['Close'] > df['ma_20']) & (df['ma_20'] > df['ma_50']), 'strong_trend'] = 1
    df.loc[(df['Close'] < df['ma_20']) & (df['ma_20'] < df['ma_50']), 'strong_trend'] = -1
    
    # SST
    print("Adding SST features...")
    df = add_stochastic_supertrend_features(df)
    df = get_stochastic_supertrend_signals(df)
    
    # Normalize prices
    base_price = df['Close'].iloc[0]
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = (df[col] / base_price - 1) * 100
    df['atr'] = df['atr'] / base_price * 100
    
    # Normalize other features
    binary_cols = ['sst_buy_signal', 'sst_sell_signal', 'sst_direction', 'trend', 'strong_trend']
    for col in df.columns:
        if col not in binary_cols + ['Open', 'High', 'Low', 'Close', 'atr']:
            if df[col].dtype in ['float64', 'float32'] and df[col].std() > 0:
                df[col] = (df[col] - df[col].mean()) / df[col].std()
                df[col] = df[col].clip(-3, 3)
    
    df = df.dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Final: {len(df)} bars")
    return df


def make_env(df, seed, atr_mult):
    def _init():
        env = BalancedRREnv(df=df, atr_mult=atr_mult, random_start=True)
        env.reset(seed=seed)
        return Monitor(env)
    return _init


def train(
    data_path="data/raw/XAU_15m_data.csv",
    timesteps=1000000,
    n_envs=40,
    atr_mult=1.0,  # 1:1 ratio - same distance for SL and TP
    eval_freq=25000
):
    """Train balanced 1:1 RR model."""
    
    print("=" * 60)
    print("🚀 BALANCED 1:1 RISK-REWARD MODEL")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    print(f"  SL = TP = {atr_mult} * ATR (1:1 ratio)")
    print(f"  Need >50% win rate for profitability")
    print(f"  Envs: {n_envs} | Timesteps: {timesteps:,}")
    
    df = load_and_prepare_data(data_path)
    
    # Split data
    train_size = int(len(df) * 0.6)
    val_size = int(len(df) * 0.2)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    test_df = df.iloc[train_size+val_size:].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Create environments
    print(f"\nCreating {n_envs} environments...")
    env = DummyVecEnv([make_env(train_df, i, atr_mult) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, clip_reward=30.0)
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
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
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/balanced_rr_training"
    )
    
    os.makedirs("models/experimental", exist_ok=True)
    
    callback = TrainingCallback(
        val_df=val_df,
        atr_mult=atr_mult,
        eval_freq=eval_freq,
        save_path="models/experimental"
    )
    
    print("\n🏋️ Starting training...")
    print("=" * 60)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"models/experimental/balanced_rr_final_{timestamp}")
    env.save(f"models/experimental/balanced_rr_final_{timestamp}_vecnorm.pkl")

    # Test evaluation
    print("\n" + "=" * 60)
    print("🧪 TEST EVALUATION")
    print("=" * 60)
    
    total_trades, total_wins = 0, 0
    total_profit, total_loss = 0.0, 0.0
    returns = []
    
    for ep in range(10):
        test_env = BalancedRREnv(df=test_df, atr_mult=atr_mult, random_start=True)
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
            print(f"  Ep {ep+1}: Trades={trades}, WR={wr*100:.1f}%, Return={ret:.1f}%")
    
    final_wr = total_wins / max(1, total_trades)
    avg_return = np.mean(returns)
    
    # For 1:1 RR, profit factor = WR / (1 - WR)
    profit_factor = final_wr / (1 - final_wr + 0.001) if final_wr < 1 else 10.0
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS:")
    print(f"  Trades: {total_trades}")
    print(f"  Win Rate: {final_wr*100:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Return: {avg_return:.1f}%")
    print(f"{'='*50}")
    
    # Save to production if profitable
    if final_wr > 0.52 and profit_factor > 1.1:
        print(f"🎉 PROFITABLE MODEL!")
        os.makedirs("models/production", exist_ok=True)
        model.save(f"models/production/balanced_rr_{final_wr*100:.0f}pct_{timestamp}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000000)
    parser.add_argument('--envs', type=int, default=40)
    parser.add_argument('--atr-mult', type=float, default=1.0, help='ATR multiplier for SL=TP')
    
    args = parser.parse_args()
    
    train(
        timesteps=args.timesteps,
        n_envs=args.envs,
        atr_mult=args.atr_mult
    )
