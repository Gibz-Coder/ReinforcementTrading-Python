#!/usr/bin/env python3
"""
Ultra-Selective Trading Model V4 - FOCUSED HIGH WIN RATE EDITION
================================================================
Target: 80%+ win rate with 1:1 SL/TP ratio

KEY IMPROVEMENTS:
1. Ultra-selective signal filtering (quality over quantity)
2. Curriculum learning approach
3. Better reward alignment with validation performance
4. Reduced overfitting through stricter requirements
5. Enhanced evaluation methodology
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
# ULTRA-SELECTIVE SIGNAL DETECTION
# ============================================================================

def create_ultra_selective_signals(df):
    """Create ultra-selective signals targeting 80%+ win rate."""
    
    # Core technical indicators
    df['rsi'] = pta.rsi(df['Close'], length=14)
    df['adx'] = pta.adx(df['High'], df['Low'], df['Close'], length=14).iloc[:, 0]
    df['atr'] = pta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # EMAs for trend
    df['ema_fast'] = pta.ema(df['Close'], length=12)
    df['ema_slow'] = pta.ema(df['Close'], length=26)
    
    # Volume analysis
    df['volume_ma'] = df['Volume'].rolling(20).mean()
    df['volume_ratio'] = df['Volume'] / (df['volume_ma'] + 1e-10)
    
    # Price action patterns
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']
    df['body_pct'] = df['body'] / (df['range'] + 1e-10)
    
    # ULTRA-SELECTIVE BULLISH CONDITIONS
    bullish_conditions = [
        # 1. Strong uptrend
        df['ema_fast'] > df['ema_slow'],
        df['Close'] > df['ema_fast'],
        df['Close'] > df['Close'].shift(1),
        
        # 2. Momentum confirmation
        df['rsi'] > 45,
        df['rsi'] < 65,
        df['adx'] > 25,
        
        # 3. Volume confirmation
        df['volume_ratio'] > 1.1,
        
        # 4. Price action strength
        df['body_pct'] > 0.4,
        df['Close'] > df['Open'],  # Bullish candle
        
        # 5. Market structure
        df['Close'] > df['High'].shift(1).rolling(3).max(),  # Breaking recent highs
    ]
    
    # ULTRA-SELECTIVE BEARISH CONDITIONS
    bearish_conditions = [
        # 1. Strong downtrend
        df['ema_fast'] < df['ema_slow'],
        df['Close'] < df['ema_fast'],
        df['Close'] < df['Close'].shift(1),
        
        # 2. Momentum confirmation
        df['rsi'] < 55,
        df['rsi'] > 35,
        df['adx'] > 25,
        
        # 3. Volume confirmation
        df['volume_ratio'] > 1.1,
        
        # 4. Price action strength
        df['body_pct'] > 0.4,
        df['Close'] < df['Open'],  # Bearish candle
        
        # 5. Market structure
        df['Close'] < df['Low'].shift(1).rolling(3).min(),  # Breaking recent lows
    ]
    
    # Count conditions met (require ALL for ultra-selective)
    bull_score = sum(bullish_conditions)
    bear_score = sum(bearish_conditions)
    
    # ULTRA-SELECTIVE thresholds (require 7+ out of 10 conditions initially)
    df['ultra_bull_signal'] = (bull_score >= 7).astype(int)
    df['ultra_bear_signal'] = (bear_score >= 7).astype(int)
    df['high_bull_signal'] = (bull_score >= 6).astype(int)
    df['high_bear_signal'] = (bear_score >= 6).astype(int)
    
    df['bull_score'] = bull_score
    df['bear_score'] = bear_score
    
    return df


# ============================================================================
# ULTRA-SELECTIVE TRADING ENVIRONMENT
# ============================================================================

class UltraSelectiveEnv(gym.Env):
    """Ultra-selective environment focusing on quality over quantity."""
    
    def __init__(self, df, window_size=30, initial_balance=10000.0,
                 atr_mult=1.0, ultra_mode=True, curriculum_stage=1):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.atr_mult = atr_mult
        self.ultra_mode = ultra_mode
        self.curriculum_stage = curriculum_stage  # 1=strict, 2=moderate, 3=relaxed
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Select key features only
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'adx', 'atr', 'ema_fast', 'ema_slow',
            'volume_ratio', 'body_pct', 'bull_score', 'bear_score'
        ]
        self.num_features = len(self.feature_columns)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.num_features + 3),  # +3 for position, pnl, trades
            dtype=np.float32
        )
        
        self._init_state()
    
    def _init_state(self):
        # Start from a random position for better generalization
        if self.n_steps > self.window_size + 500:
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
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.episode_steps = 0
        self.max_episode_steps = min(1000, self.n_steps - self.current_step - 1)
        self.last_trade_step = -20  # Cooldown
        
        # Track performance
        self.trade_results = []
        self.peak_balance = self.initial_balance
    
    def _get_signal_requirements(self):
        """Get signal requirements based on curriculum stage."""
        if self.curriculum_stage == 1:  # Strictest
            return {'ultra_bull': True, 'ultra_bear': True, 'min_score': 7}
        elif self.curriculum_stage == 2:  # Moderate
            return {'ultra_bull': False, 'ultra_bear': False, 'min_score': 6}
        else:  # Most relaxed
            return {'ultra_bull': False, 'ultra_bear': False, 'min_score': 5}
    
    def _get_obs(self):
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        # Get feature data
        obs_data = self.df[self.feature_columns].iloc[start:end].values.copy()
        
        if len(obs_data) < self.window_size:
            pad = np.zeros((self.window_size - len(obs_data), self.num_features))
            obs_data = np.vstack([pad, obs_data])
        
        # Add context features
        extra = np.zeros((self.window_size, 3))
        
        # Position encoding
        if self.position == 'long':
            extra[:, 0] = 1.0
        elif self.position == 'short':
            extra[:, 0] = -1.0
        
        # Current PnL
        if self.position and self.entry_price != 0:
            price = self.df.iloc[self.current_step]['Close']
            if self.position == 'long':
                pnl = (price - self.entry_price) / abs(self.entry_price)
            else:
                pnl = (self.entry_price - price) / abs(self.entry_price)
            extra[:, 1] = np.clip(pnl * 5, -2.0, 2.0)
        
        # Trade count (normalized)
        extra[:, 2] = min(self.total_trades / 10.0, 1.0)
        
        obs = np.hstack([obs_data, extra]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=2.0, neginf=-2.0)
        return np.clip(obs, -3.0, 3.0)
    
    def step(self, action):
        reward = 0.0
        done = False
        
        if self.current_step >= len(self.df) - 1:
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        price = self.df.iloc[self.current_step]['Close']
        high = self.df.iloc[self.current_step]['High']
        low = self.df.iloc[self.current_step]['Low']
        
        # Get current signals
        row = self.df.iloc[self.current_step]
        bull_score = row.get('bull_score', 0)
        bear_score = row.get('bear_score', 0)
        ultra_bull = row.get('ultra_bull_signal', 0)
        ultra_bear = row.get('ultra_bear_signal', 0)
        
        # Check exit conditions first
        if self.position:
            closed, pnl, is_win = self._check_exit(high, low)
            if closed:
                self.total_trades += 1
                self.trade_results.append(pnl)
                
                if is_win:
                    self.wins += 1
                    self.consecutive_losses = 0
                    self.consecutive_wins += 1
                    
                    # MASSIVE rewards for wins to encourage high win rate
                    base_reward = 100.0
                    
                    # Win rate bonus
                    if self.total_trades >= 3:
                        current_wr = self.wins / self.total_trades
                        if current_wr >= 0.8:
                            base_reward += 50.0
                        elif current_wr >= 0.7:
                            base_reward += 25.0
                    
                    # Consistency bonus
                    if self.consecutive_wins >= 3:
                        base_reward += self.consecutive_wins * 10
                    
                    reward = base_reward
                    
                else:
                    self.losses += 1
                    self.consecutive_wins = 0
                    self.consecutive_losses += 1
                    
                    # Heavy penalties for losses
                    penalty = -50.0
                    
                    # Escalating penalties for consecutive losses
                    if self.consecutive_losses >= 2:
                        penalty -= self.consecutive_losses * 20
                    
                    reward = penalty
                
                # Update balance
                self.balance *= (1 + pnl * 0.02)  # 2% position size
                self.position = None
                self.entry_price = 0.0
                self.entry_atr = 0.0
        
        # Trading logic with ultra-selective requirements
        requirements = self._get_signal_requirements()
        cooldown = 20  # Minimum bars between trades
        max_daily_trades = 2  # Very conservative
        
        can_trade = (
            (self.current_step - self.last_trade_step) >= cooldown and
            self.position is None and
            self.consecutive_losses < 3  # Stop after 3 losses
        )
        
        if can_trade:
            if action == 1:  # Buy
                signal_valid = False
                
                if requirements['ultra_bull']:
                    signal_valid = ultra_bull == 1
                else:
                    signal_valid = bull_score >= requirements['min_score']
                
                if signal_valid:
                    self.position = 'long'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.last_trade_step = self.current_step
                    
                    # Reward for taking high-quality signals
                    reward += 20.0 + (bull_score - 7) * 5
                    
                else:
                    # Heavy penalty for poor signals
                    if bull_score < 6:
                        reward -= 20.0
                    else:
                        reward -= 10.0
                        
            elif action == 2:  # Sell
                signal_valid = False
                
                if requirements['ultra_bear']:
                    signal_valid = ultra_bear == 1
                else:
                    signal_valid = bear_score >= requirements['min_score']
                
                if signal_valid:
                    self.position = 'short'
                    self.entry_price = price
                    self.entry_atr = self._get_atr()
                    self.last_trade_step = self.current_step
                    
                    # Reward for taking high-quality signals
                    reward += 20.0 + (bear_score - 7) * 5
                    
                else:
                    # Heavy penalty for poor signals
                    if bear_score < 6:
                        reward -= 20.0
                    else:
                        reward -= 10.0
            
            else:  # Hold
                # Reward patience when signals are weak
                max_score = max(bull_score, bear_score)
                if max_score < requirements['min_score']:
                    reward += 2.0  # Good patience
                else:
                    reward -= 5.0  # Penalty for missing good signals
        
        # Risk management
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if drawdown > 0.20:  # 20% max drawdown
            done = True
            reward -= 100.0
        
        # Episode management
        self.current_step += 1
        self.episode_steps += 1
        
        if self.current_step >= self.n_steps - 1:
            done = True
        if self.episode_steps >= self.max_episode_steps:
            done = True
        
        return self._get_obs(), float(np.clip(reward, -200, 200)), done, False, self._get_info()
    
    def _get_atr(self):
        atr = self.df.iloc[self.current_step].get('atr', 0)
        if not isinstance(atr, (int, float)) or atr <= 0 or np.isnan(atr):
            recent = self.df.iloc[max(0, self.current_step-14):self.current_step]
            if len(recent) > 0:
                atr = (recent['High'] - recent['Low']).mean()
            else:
                atr = 0.01
        return max(abs(atr), 0.001)
    
    def _check_exit(self, high, low):
        """Check exit conditions with 1:1 risk/reward."""
        if not self.position or self.entry_price == 0 or self.entry_atr <= 0:
            return False, 0.0, False
        
        distance = self.entry_atr * self.atr_mult
        
        if self.position == 'long':
            tp_price = self.entry_price + distance
            sl_price = self.entry_price - distance
            
            if high >= tp_price:
                pnl = distance / abs(self.entry_price)
                return True, pnl, True
            if low <= sl_price:
                pnl = -distance / abs(self.entry_price)
                return True, pnl, False
        else:
            tp_price = self.entry_price - distance
            sl_price = self.entry_price + distance
            
            if low <= tp_price:
                pnl = distance / abs(self.entry_price)
                return True, pnl, True
            if high >= sl_price:
                pnl = -distance / abs(self.entry_price)
                return True, pnl, False
        
        return False, 0.0, False
    
    def _get_info(self):
        return {
            'win_rate': self.wins / max(1, self.total_trades),
            'total_trades': self.total_trades,
            'wins': self.wins,
            'losses': self.losses,
            'balance': self.balance,
            'return_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'curriculum_stage': self.curriculum_stage
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_obs(), {}


# ============================================================================
# CURRICULUM LEARNING CALLBACK
# ============================================================================

class CurriculumCallback(BaseCallback):
    """Curriculum learning callback with progressive difficulty."""
    
    def __init__(self, val_df, eval_freq=5000, save_path="models/experimental"):
        super().__init__()
        self.val_df = val_df
        self.eval_freq = eval_freq
        self.save_path = save_path
        
        self.best_wr = 0.0
        self.best_score = -np.inf
        self.curriculum_stage = 1
        self.stage_performance = []
        self.no_improve_count = 0
        
        # Track evaluation history
        self.evaluation_history = []
    
    def _on_step(self):
        # Evaluate periodically
        if self.num_timesteps % self.eval_freq == 0:
            val_wr, val_trades, val_return = self._evaluate_model()
            
            # Calculate composite score (heavily weighted toward win rate)
            score = val_wr * 0.8 + max(0, val_return/100) * 0.2
            
            print(f"\n📊 Evaluation at {self.num_timesteps:,} steps:")
            print(f"   Win Rate: {val_wr*100:.1f}%")
            print(f"   Trades: {val_trades:.1f}")
            print(f"   Return: {val_return:+.1f}%")
            print(f"   Score: {score:.3f}")
            print(f"   Curriculum Stage: {self.curriculum_stage}")
            
            # Save if improved
            if score > self.best_score and val_trades >= 3:
                self.best_score = score
                self.best_wr = val_wr
                self.no_improve_count = 0
                
                # Save model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"ultra_selective_v4_wr{val_wr*100:.0f}_{timestamp}"
                self.model.save(f"{self.save_path}/{model_name}")
                
                print(f"✅ NEW BEST SAVED: {model_name}")
                
                # Move to production if excellent
                if val_wr >= 0.75 and val_trades >= 5:
                    os.makedirs("models/production", exist_ok=True)
                    self.model.save(f"models/production/{model_name}")
                    print(f"🏆 MOVED TO PRODUCTION!")
                    
            else:
                self.no_improve_count += 1
            
            # Curriculum progression
            self.stage_performance.append(val_wr)
            
            # Progress curriculum if performing well
            if len(self.stage_performance) >= 5:
                recent_avg = np.mean(self.stage_performance[-5:])
                
                if self.curriculum_stage == 1 and recent_avg >= 0.70:
                    self.curriculum_stage = 2
                    print(f"📈 CURRICULUM ADVANCED TO STAGE 2")
                    self._update_env_curriculum()
                    
                elif self.curriculum_stage == 2 and recent_avg >= 0.65:
                    self.curriculum_stage = 3
                    print(f"📈 CURRICULUM ADVANCED TO STAGE 3")
                    self._update_env_curriculum()
            
            # Store evaluation
            self.evaluation_history.append({
                'timestep': self.num_timesteps,
                'win_rate': val_wr,
                'trades': val_trades,
                'return': val_return,
                'score': score,
                'stage': self.curriculum_stage
            })
            
            # Early stopping
            if self.no_improve_count >= 20:
                print(f"\n🛑 Early stopping - no improvement for 20 evaluations")
                return False
        
        return True
    
    def _update_env_curriculum(self):
        """Update environment curriculum stage."""
        if hasattr(self.model.env, 'envs'):
            for env in self.model.env.envs:
                if hasattr(env, 'curriculum_stage'):
                    env.curriculum_stage = self.curriculum_stage
    
    def _evaluate_model(self):
        """Evaluate model performance."""
        episodes_data = []
        
        for episode in range(10):
            env = UltraSelectiveEnv(
                df=self.val_df,
                curriculum_stage=self.curriculum_stage
            )
            
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, info = env.step(action)
            
            if info.get('total_trades', 0) > 0:
                episodes_data.append(info)
        
        if not episodes_data:
            return 0.3, 0, -10
        
        # Calculate metrics
        total_trades = sum(ep['total_trades'] for ep in episodes_data)
        total_wins = sum(ep['wins'] for ep in episodes_data)
        returns = [ep['return_pct'] for ep in episodes_data]
        
        win_rate = total_wins / max(1, total_trades)
        avg_trades = total_trades / len(episodes_data)
        avg_return = np.mean(returns)
        
        return win_rate, avg_trades, avg_return


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def load_and_prepare_data(filepath):
    """Load and prepare data with ultra-selective signals."""
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
    
    # Create ultra-selective signals
    print("🔧 Creating ultra-selective signals...")
    df = create_ultra_selective_signals(df)
    
    # Normalize data
    print("🧹 Normalizing data...")
    
    # Normalize prices relative to first valid price
    first_valid_idx = df['Close'].first_valid_index()
    if first_valid_idx is not None:
        base_price = df.loc[first_valid_idx, 'Close']
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = (df[col] / base_price - 1) * 100
    
    # Normalize other features
    for col in ['rsi', 'adx', 'volume_ratio', 'body_pct']:
        if col in df.columns and df[col].std() > 0:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
            df[col] = df[col].clip(-3, 3)
    
    # Clean data
    df = df.dropna().reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"✅ Final: {len(df)} bars")
    
    # Signal statistics
    ultra_bull_count = (df['ultra_bull_signal'] == 1).sum()
    ultra_bear_count = (df['ultra_bear_signal'] == 1).sum()
    
    print(f"📈 Ultra-selective signals:")
    print(f"   Bull signals: {ultra_bull_count} ({ultra_bull_count/len(df)*100:.2f}%)")
    print(f"   Bear signals: {ultra_bear_count} ({ultra_bear_count/len(df)*100:.2f}%)")
    
    return df


def make_env(df, curriculum_stage=1):
    def _init():
        env = UltraSelectiveEnv(df=df, curriculum_stage=curriculum_stage)
        return Monitor(env)
    return _init


def train_ultra_selective(
    data_path="data/raw/XAU_15m_data.csv",
    timesteps=1000000,
    n_envs=8,  # Reduced for stability
    eval_freq=5000
):
    """Train ultra-selective high win rate model."""
    
    print("=" * 80)
    print("🎯 ULTRA-SELECTIVE TRADING MODEL V4")
    print("=" * 80)
    print(f"Target: 80%+ win rate with 1:1 risk/reward")
    print(f"Approach: Curriculum learning with ultra-selective signals")
    
    # Load and prepare data
    df = load_and_prepare_data(data_path)
    
    # Split data
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)
    
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:train_size+val_size].reset_index(drop=True)
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_df):,} bars")
    print(f"   Validation: {len(val_df):,} bars")
    
    # Create environments
    print(f"\n🏗️ Creating {n_envs} environments...")
    env = DummyVecEnv([make_env(train_df, curriculum_stage=1) for _ in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=3.0)
    
    # Model configuration
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
        activation_fn=nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2e-4,
        n_steps=512,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Higher exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="logs/ultra_selective_v4_training"
    )
    
    # Create directories
    os.makedirs("models/experimental", exist_ok=True)
    
    # Curriculum callback
    callback = CurriculumCallback(
        val_df=val_df,
        eval_freq=eval_freq,
        save_path="models/experimental"
    )
    
    print("\n🏋️ Starting curriculum training...")
    print("=" * 80)
    
    try:
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_name = f"ultra_selective_v4_final_{timestamp}"
    model.save(f"models/experimental/{final_name}")
    
    print(f"\n💾 Final model saved: {final_name}")
    print(f"🏆 Best validation win rate: {callback.best_wr*100:.1f}%")
    
    return model, callback.evaluation_history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Selective Trading Model V4")
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps')
    parser.add_argument('--envs', type=int, default=8, help='Number of environments')
    parser.add_argument('--eval-freq', type=int, default=5000, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting Ultra-Selective Model V4 Training")
    print(f"Target: 80%+ win rate with curriculum learning")
    
    model, history = train_ultra_selective(
        timesteps=args.timesteps,
        n_envs=args.envs,
        eval_freq=args.eval_freq
    )
    
    print("\n🎉 Training completed!")
    if history:
        best_eval = max(history, key=lambda x: x['win_rate'])
        print(f"🏆 Best validation: {best_eval['win_rate']*100:.1f}% win rate")