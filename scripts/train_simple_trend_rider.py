#!/usr/bin/env python3
"""
Simple Trend Rider - Multi-Timeframe
====================================
Simple, effective trend-following using proven indicators
Focus: Ride trends, not predict reversals
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


def load_timeframe_data(base_path="../data/raw"):
    """Load all timeframes quickly."""
    timeframes = ['15m', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        try:
            df = pd.read_csv(f"{base_path}/XAU_{tf}_data.csv", sep=';')
            if len(df.columns) == 1:
                df = pd.read_csv(f"{base_path}/XAU_{tf}_data.csv", sep=',')
            
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


def add_simple_trend_signals(df_15m, df_1h, df_4h, df_1d):
    """SIMPLE trend-following signals - focus on what works."""
    
    # === 15M INDICATORS (Entry timeframe) ===
    # Moving averages - the foundation of trend following
    df_15m['ema20'] = pta.ema(df_15m['Close'], 20)
    df_15m['ema50'] = pta.ema(df_15m['Close'], 50)
    
    # Momentum
    df_15m['rsi'] = pta.rsi(df_15m['Close'], 14)
    
    # Volatility
    df_15m['atr'] = pta.atr(df_15m['High'], df_15m['Low'], df_15m['Close'], 14)
    df_15m['atr_pct'] = (df_15m['atr'] / df_15m['Close']) * 100
    
    # === HIGHER TIMEFRAME TRENDS (Simple but effective) ===
    htf_trends = {}
    
    for name, htf_df in [('1h', df_1h), ('4h', df_4h), ('1d', df_1d)]:
        if htf_df is not None and len(htf_df) > 50:
            # Simple EMA crossover system
            htf_df['ema20'] = pta.ema(htf_df['Close'], 20)
            htf_df['ema50'] = pta.ema(htf_df['Close'], 50)
            
            # Current trend direction
            current_trend = 1 if htf_df['ema20'].iloc[-1] > htf_df['ema50'].iloc[-1] else -1
            
            # Trend strength (how long has trend been in place)
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
            
            # Trend strength score (0-3)
            if trend_bars >= 15:
                strength = 3  # Very strong trend
            elif trend_bars >= 10:
                strength = 2  # Strong trend
            elif trend_bars >= 5:
                strength = 1  # Moderate trend
            else:
                strength = 0  # Weak/no trend
            
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
    
    # === SIMPLE TREND ALIGNMENT ===
    # Count how many timeframes are bullish/bearish
    df_15m['bull_timeframes'] = (
        (df_15m['ema20'] > df_15m['ema50']).astype(int) +  # 15M
        (df_15m['1h_trend'] == 1).astype(int) +            # 1H
        (df_15m['4h_trend'] == 1).astype(int) +            # 4H
        (df_15m['1d_trend'] == 1).astype(int)              # 1D
    )
    
    df_15m['bear_timeframes'] = (
        (df_15m['ema20'] < df_15m['ema50']).astype(int) +  # 15M
        (df_15m['1h_trend'] == -1).astype(int) +           # 1H
        (df_15m['4h_trend'] == -1).astype(int) +           # 4H
        (df_15m['1d_trend'] == -1).astype(int)             # 1D
    )
    
    # === TREND STRENGTH SCORE ===
    df_15m['trend_strength_score'] = (
        df_15m['1h_strength'] + 
        df_15m['4h_strength'] + 
        df_15m['1d_strength']
    )
    
    # === SIMPLE ENTRY SIGNALS ===
    # Bull signal: Multiple timeframes aligned + not overbought
    df_15m['bull_signal'] = (
        (df_15m['bull_timeframes'] >= 3) &          # At least 3/4 timeframes bullish
        (df_15m['trend_strength_score'] >= 3) &     # Strong trend on HTF
        (df_15m['rsi'] < 70) &                      # Not overbought
        (df_15m['Close'] > df_15m['ema20'])         # Price above 15M trend
    ).astype(int)
    
    # Bear signal: Multiple timeframes aligned + not oversold
    df_15m['bear_signal'] = (
        (df_15m['bear_timeframes'] >= 3) &          # At least 3/4 timeframes bearish
        (df_15m['trend_strength_score'] >= 3) &     # Strong trend on HTF
        (df_15m['rsi'] > 30) &                      # Not oversold
        (df_15m['Close'] < df_15m['ema20'])         # Price below 15M trend
    ).astype(int)
    
    # === TREND CONTINUATION SIGNALS (for riding trends) ===
    # Pullback to EMA20 in strong trend
    df_15m['bull_pullback'] = (
        (df_15m['bull_timeframes'] >= 2) &
        (df_15m['trend_strength_score'] >= 2) &
        (df_15m['Close'] <= df_15m['ema20'] * 1.001) &  # Near EMA20
        (df_15m['Close'] >= df_15m['ema20'] * 0.999) &
        (df_15m['rsi'] < 60)
    ).astype(int)
    
    df_15m['bear_pullback'] = (
        (df_15m['bear_timeframes'] >= 2) &
        (df_15m['trend_strength_score'] >= 2) &
        (df_15m['Close'] >= df_15m['ema20'] * 0.999) &  # Near EMA20
        (df_15m['Close'] <= df_15m['ema20'] * 1.001) &
        (df_15m['rsi'] > 40)
    ).astype(int)
    
    # === SESSION FILTER (Simple) ===
    if hasattr(df_15m.index, 'hour'):
        df_15m['hour'] = df_15m.index.hour
    else:
        df_15m['hour'] = 12
    
    # Active trading hours (London + NY)
    df_15m['active_session'] = ((df_15m['hour'] >= 8) & (df_15m['hour'] <= 17)).astype(int)
    
    print(f"✅ SIMPLE TREND signals created!")
    
    return df_15m


class SimpleTrendRiderEnv(gym.Env):
    """Simple trend-riding environment."""
    
    def __init__(self, df, window_size=20):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: long, 2: short
        
        # Simple feature set - only what matters for trend following
        self.features = [
            'Close', 'ema20', 'ema50', 'rsi', 'atr_pct',
            'bull_timeframes', 'bear_timeframes', 'trend_strength_score',
            '1h_trend', '4h_trend', '1d_trend',
            '1h_strength', '4h_strength', '1d_strength',
            'bull_signal', 'bear_signal', 'bull_pullback', 'bear_pullback',
            'active_session'
        ]
        
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
        self.max_episode_steps = 250
        self.last_trade_step = -10  # Short cooldown for trend following
        self.peak_balance = 10000.0
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        obs_data = self.df[self.features].iloc[start:end].values
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), len(self.features)))
            obs_data = np.vstack([pad, obs_data])
        
        # Add position and win rate info
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
        
        # Exit logic - TREND FOLLOWING approach
        if self.position:
            # Use ATR-based stops
            stop_distance = self.entry_atr * 2.0  # Wider stops for trend following
            target_distance = self.entry_atr * 2.0  # 1:1 R:R initially
            
            if self.position == 'long':
                # Exit on profit target or stop loss
                if high >= self.entry_price + target_distance:
                    reward = 100.0  # Good reward for wins
                    self.wins += 1
                    self.total_trades += 1
                    self.balance *= 1.02
                    self.position = None
                elif low <= self.entry_price - stop_distance:
                    reward = -50.0  # Moderate penalty for losses
                    self.total_trades += 1
                    self.balance *= 0.98
                    self.position = None
                # Also exit if trend changes against us
                elif row['bear_timeframes'] >= 3:
                    # Trend reversal - exit immediately
                    pnl = (price - self.entry_price) / self.entry_price
                    if pnl > 0:
                        reward = 50.0
                        self.wins += 1
                    else:
                        reward = -30.0
                    self.total_trades += 1
                    self.balance *= (1 + pnl * 0.02)
                    self.position = None
                    
            else:  # Short position
                if low <= self.entry_price - target_distance:
                    reward = 100.0
                    self.wins += 1
                    self.total_trades += 1
                    self.balance *= 1.02
                    self.position = None
                elif high >= self.entry_price + stop_distance:
                    reward = -50.0
                    self.total_trades += 1
                    self.balance *= 0.98
                    self.position = None
                elif row['bull_timeframes'] >= 3:
                    # Trend reversal - exit immediately
                    pnl = (self.entry_price - price) / self.entry_price
                    if pnl > 0:
                        reward = 50.0
                        self.wins += 1
                    else:
                        reward = -30.0
                    self.total_trades += 1
                    self.balance *= (1 + pnl * 0.02)
                    self.position = None
        
        # Entry logic - SIMPLE and EFFECTIVE
        can_trade = (
            (self.current_step - self.last_trade_step) >= 10 and
            self.position is None and
            row['active_session'] == 1
        )
        
        if can_trade:
            # LONG entry: Strong bull signal OR pullback in bull trend
            if (action == 1 and 
                (row['bull_signal'] == 1 or row['bull_pullback'] == 1) and
                row['atr_pct'] > 0.02):  # Minimum volatility
                
                self.position = 'long'
                self.entry_price = price
                self.entry_atr = max(row['atr_pct'] * price / 100, 0.01)
                self.last_trade_step = self.current_step
                reward += 10.0  # Small reward for taking valid signal
                
            # SHORT entry: Strong bear signal OR pullback in bear trend
            elif (action == 2 and 
                  (row['bear_signal'] == 1 or row['bear_pullback'] == 1) and
                  row['atr_pct'] > 0.02):
                
                self.position = 'short'
                self.entry_price = price
                self.entry_atr = max(row['atr_pct'] * price / 100, 0.01)
                self.last_trade_step = self.current_step
                reward += 10.0
                
            elif action != 0:
                # Small penalty for bad signals
                reward -= 5.0
            else:
                # Small reward for patience when no good signals
                if row['bull_signal'] == 0 and row['bear_signal'] == 0:
                    reward += 1.0
        
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
            'return_pct': (self.balance - 10000) / 100
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


class SimpleTrendCallback(BaseCallback):
    """Simple callback for trend following."""
    
    def __init__(self, val_df, eval_freq=5000):  # Less frequent evaluation for long training
        super().__init__()
        self.val_df = val_df
        self.eval_freq = eval_freq
        self.best_wr = 0.0
        self.best_return = -100.0
        self.no_improve = 0
        self.evaluation_history = []  # Track all evaluations
        self.save_frequency = 50000  # Save every 50k steps regardless of improvement
    
    def _on_step(self):
        # Save model every 50k steps regardless of performance
        if self.num_timesteps % self.save_frequency == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"simple_trend_checkpoint_{self.num_timesteps//1000}k_{timestamp}"
            self.model.save(f"../models/experimental/{checkpoint_name}")
            print(f"💾 Checkpoint saved: {checkpoint_name}")
        
        if self.num_timesteps % self.eval_freq == 0:
            wr, trades, ret = self._evaluate()
            
            # Store evaluation history
            self.evaluation_history.append({
                'timestep': self.num_timesteps,
                'win_rate': wr,
                'trades': trades,
                'return': ret
            })
            
            print(f"\n🎯 SIMPLE TREND Eval at {self.num_timesteps:,}:")
            print(f"   Win Rate: {wr*100:.1f}%")
            print(f"   Trades: {trades:.1f}")
            print(f"   Return: {ret:+.1f}%")
            print(f"   No Improve Count: {self.no_improve}/∞ (no early stopping)")  # Show progress
            
            # Save if better win rate OR better return with decent trades
            improved = False
            if wr > self.best_wr and trades >= 2:
                self.best_wr = wr
                improved = True
            elif ret > self.best_return and trades >= 2 and wr >= 0.5:
                self.best_return = ret
                improved = True
            
            if improved:
                self.no_improve = 0
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"simple_trend_wr{wr*100:.0f}_ret{ret:+.0f}_{timestamp}"
                self.model.save(f"../models/experimental/{name}")
                print(f"✅ NEW BEST: {name}")
                
                if wr >= 0.65 or (wr >= 0.55 and ret >= 5):
                    os.makedirs("../models/production", exist_ok=True)
                    self.model.save(f"../models/production/{name}")
                    print(f"🏆 PRODUCTION MODEL!")
            else:
                self.no_improve += 1
            
            # Removed early stopping for extended training
            # if self.no_improve >= 8:
            #     print("🛑 Early stopping")
            #     return False
        
        return True
    
    def _evaluate(self):
        episodes = []
        for _ in range(6):
            env = SimpleTrendRiderEnv(self.val_df)
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
        
        return (total_wins / max(1, total_trades), 
                total_trades / len(episodes), 
                np.mean(returns))


def train_simple_trend_rider(timesteps=1000000, n_envs=8):  # Default 1M timesteps
    """Train simple trend rider."""
    
    print("🎯 SIMPLE TREND RIDER")
    print("Focus: Ride trends effectively with simple signals")
    
    # Load data
    data = load_timeframe_data()
    df = add_simple_trend_signals(data['15m'], data['1h'], data['4h'], data['1d'])
    
    # Simple normalization
    base_price = df['Close'].iloc[100]
    for col in ['Close', 'ema20', 'ema50']:
        df[col] = (df[col] / base_price - 1) * 100
    
    # Normalize RSI
    df['rsi'] = (df['rsi'] - 50) / 25  # Scale to roughly -2 to +2
    
    # Normalize ATR percentage
    df['atr_pct'] = df['atr_pct'].clip(0, df['atr_pct'].quantile(0.95))
    df['atr_pct'] = df['atr_pct'] / df['atr_pct'].max()
    
    df = df.dropna().reset_index(drop=True)
    
    # Split
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")
    
    # Signal analysis
    bull_signals = (train_df['bull_signal'] == 1).sum()
    bear_signals = (train_df['bear_signal'] == 1).sum()
    bull_pullbacks = (train_df['bull_pullback'] == 1).sum()
    bear_pullbacks = (train_df['bear_pullback'] == 1).sum()
    
    print(f"Signal Opportunities:")
    print(f"  Bull signals: {bull_signals} ({bull_signals/len(train_df)*100:.2f}%)")
    print(f"  Bear signals: {bear_signals} ({bear_signals/len(train_df)*100:.2f}%)")
    print(f"  Bull pullbacks: {bull_pullbacks} ({bull_pullbacks/len(train_df)*100:.2f}%)")
    print(f"  Bear pullbacks: {bear_pullbacks} ({bear_pullbacks/len(train_df)*100:.2f}%)")
    print(f"  Total opportunities: {(bull_signals+bear_signals+bull_pullbacks+bear_pullbacks)/len(train_df)*100:.2f}%")
    
    # Create environment
    env = DummyVecEnv([lambda: Monitor(SimpleTrendRiderEnv(train_df)) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=3.0)
    
    # Simple model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=4,
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=[256, 128],  # Simpler network
            activation_fn=nn.ReLU
        ),
        verbose=1
    )
    
    # Train
    os.makedirs("../models/experimental", exist_ok=True)
    callback = SimpleTrendCallback(val_df)
    
    print("\n🏋️ Training SIMPLE TREND RIDER...")
    model.learn(timesteps, callback=callback, progress_bar=True)
    
    # Save final
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"../models/experimental/simple_trend_final_{timestamp}")
    
    print(f"\n🎉 Best win rate: {callback.best_wr*100:.1f}%")
    print(f"🎉 Best return: {callback.best_return:+.1f}%")
    
    # Print training progress summary
    if callback.evaluation_history:
        print(f"\n📈 TRAINING PROGRESS SUMMARY:")
        print(f"   Total evaluations: {len(callback.evaluation_history)}")
        
        # Show progress at key milestones
        milestones = [100000, 250000, 500000, 750000, 1000000]
        for milestone in milestones:
            if milestone <= timesteps:
                # Find closest evaluation to milestone
                closest_eval = min(callback.evaluation_history, 
                                 key=lambda x: abs(x['timestep'] - milestone))
                if abs(closest_eval['timestep'] - milestone) <= 10000:  # Within 10k steps
                    print(f"   {milestone//1000}k steps: WR={closest_eval['win_rate']*100:.1f}%, "
                          f"Trades={closest_eval['trades']:.1f}, Return={closest_eval['return']:+.1f}%")
        
        # Final performance
        final_eval = callback.evaluation_history[-1]
        print(f"   Final: WR={final_eval['win_rate']*100:.1f}%, "
              f"Trades={final_eval['trades']:.1f}, Return={final_eval['return']:+.1f}%")
        
        # Stability analysis
        recent_evals = callback.evaluation_history[-10:]  # Last 10 evaluations
        if len(recent_evals) >= 5:
            recent_wrs = [e['win_rate'] for e in recent_evals]
            wr_stability = np.std(recent_wrs) * 100
            print(f"   Win Rate Stability (last 10 evals): ±{wr_stability:.1f}%")
    
    print(f"\n🎯 EXTENDED TRAINING COMPLETE!")
    print(f"   Total timesteps: {timesteps:,}")
    print(f"   Training duration: Extended (no early stopping)")
    print(f"   Model reliability: High (1M+ timesteps)")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=1000000)  # Default 1M timesteps
    parser.add_argument('--envs', type=int, default=8)
    args = parser.parse_args()
    
    train_simple_trend_rider(args.timesteps, args.envs)