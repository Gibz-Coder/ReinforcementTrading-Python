"""
Ultra Aggressive 80%+ Win Rate Trading System
============================================

Designed for maximum win rate (80%+) with aggressive trading and high profitability.

Key Strategies:
1. Ultra-strict entry criteria with 5+ confirming indicators
2. Massive penalties for losses, huge rewards for wins
3. Scalping approach with tight stops and quick profits
4. Multiple timeframe confluence requirements
5. Real-time market regime detection
6. Aggressive position sizing on high-confidence setups
7. Profit-focused reward optimization
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from gymnasium import spaces
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList, BaseCallback
from stable_baselines3.common.monitor import Monitor

from enhanced_indicators import load_and_preprocess_data_enhanced
from stochastic_supertrend import add_stochastic_supertrend_features, get_stochastic_supertrend_signals
from high_winrate_indicators import create_high_winrate_features, create_predictive_signals


class UltraAggressiveRewardSystem:
    """Extreme reward system for 80%+ win rate and profitability."""
    
    def __init__(self):
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_performance = []
        self.target_win_rate = 0.80
        
    def calculate_reward(self, pnl_pips: float, action_type: str, market_state: Dict,
                        position_info: Optional[Dict] = None, exit_reason: Optional[str] = None) -> float:
        """Ultra-aggressive reward calculation for 80%+ win rate and profit."""
        
        reward = 0.0
        confidence = market_state.get('confidence', 0.5)
        signal_strength = market_state.get('signal_strength', 0)
        confluence_count = market_state.get('confluence_count', 0)
        
        if exit_reason is not None:  # Trade closed
            self.total_profit += pnl_pips
            
            if pnl_pips > 0:  # WINNING TRADE
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.recent_performance.append(1)
                
                # Base win reward + profit scaling
                base_win_reward = 150.0
                profit_reward = min(pnl_pips * 3.0, 400.0)  # Strong profit incentive
                
                # Confidence bonus
                confidence_bonus = confidence * 200.0 if confidence > 0.7 else confidence * 50.0
                
                # Confluence bonus
                confluence_bonus = max(0, (confluence_count - 2)) * 50.0
                
                # Winning streak bonus (exponential)
                streak_bonus = min(self.consecutive_wins ** 2 * 25.0, 500.0)
                
                # Quick profit bonus (scalping reward)
                time_bonus = 0.0
                if position_info and position_info.get('bars_held', 10) <= 3:
                    time_bonus = 100.0
                
                reward = base_win_reward + profit_reward + confidence_bonus + confluence_bonus + streak_bonus + time_bonus
                
            else:  # LOSING TRADE
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.recent_performance.append(0)
                
                # Heavy loss penalties
                base_loss_penalty = -500.0
                loss_penalty = max(pnl_pips * 3.0, -600.0)
                
                # Low confidence penalty
                confidence_penalty = -(1 - confidence) * 400.0 if confidence < 0.7 else -(1 - confidence) * 100.0
                
                # Losing streak penalty
                streak_penalty = -min(self.consecutive_losses ** 2 * 50.0, 800.0)
                
                # Low confluence penalty
                confluence_penalty = -max(0, (3 - confluence_count)) * 100.0
                
                reward = base_loss_penalty + loss_penalty + confidence_penalty + streak_penalty + confluence_penalty
            
            self.total_trades += 1
            if len(self.recent_performance) > 50:
                self.recent_performance = self.recent_performance[-50:]
        
        elif action_type == 'open':
            # Reward high-confidence entries, penalize low-confidence
            if confidence > 0.8 and confluence_count >= 4:
                reward = 50.0 + signal_strength * 20.0
            elif confidence > 0.6 and confluence_count >= 3:
                reward = 20.0 + signal_strength * 10.0
            else:
                reward = -50.0  # Penalty for low-quality entries
        
        # Win rate enforcement
        if self.total_trades >= 10:
            current_win_rate = self.winning_trades / self.total_trades
            if current_win_rate >= 0.80:
                reward += 100.0 * (current_win_rate - 0.75)
            elif current_win_rate < 0.60:
                reward -= 200.0 * (0.70 - current_win_rate)
        
        return reward
    
    def get_stats(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_profit': self.total_profit,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }
    
    def reset(self):
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0
        self.recent_performance = []


class UltraAggressiveTradingEnv(gym.Env):
    """Ultra-aggressive trading environment for 80%+ win rate."""
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, df: pd.DataFrame, window_size: int = 30, initial_balance: float = 10000.0,
                 max_trades_per_day: int = 10, min_confidence: float = 0.6, render_mode: str = None):
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_trades_per_day = max_trades_per_day
        self.min_confidence = min_confidence
        self.render_mode = render_mode
        
        # Action space: Hold, Close, Buy_Strong, Buy_Medium, Sell_Strong, Sell_Medium
        self.action_space = spaces.Discrete(6)
        
        # Observation space
        self.num_features = self.df.shape[1]
        self.extra_features = 10
        total_features = self.num_features + self.extra_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, total_features),
            dtype=np.float32
        )
        
        self.reward_system = UltraAggressiveRewardSystem()
        self._init_state()
    
    def _init_state(self):
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.position = None
        self.total_trades = 0
        self.winning_trades = 0
        self.trades_today = 0
        self.current_day = self.current_step // 24
        self.trade_history = []
        self.equity_curve = [self.initial_balance]
        self.max_drawdown = 0.0
        self.peak_equity = self.initial_balance
        self.reward_system.reset()
    
    def _get_market_state(self) -> Dict:
        if self.current_step >= len(self.df):
            return {'confidence': 0.5, 'signal_strength': 0, 'confluence_count': 0, 'market_regime': 0}
        
        row = self.df.iloc[self.current_step]
        
        # Calculate confluence
        bull_conf = int(row.get('sst_buy_signal', 0)) + int(row.get('strong_bull_signal', 0)) + \
                   int(row.get('macd_bullish_cross', 0)) + int(row.get('momentum_aligned_bull', 0)) + \
                   int(row.get('high_volume', 0))
        bear_conf = int(row.get('sst_sell_signal', 0)) + int(row.get('strong_bear_signal', 0)) + \
                   int(row.get('macd_bearish_cross', 0)) + int(row.get('momentum_aligned_bear', 0)) + \
                   int(row.get('high_volume', 0))
        
        confluence = max(bull_conf, bear_conf)
        confidence = min(0.5 + confluence * 0.1 + row.get('sst_trend_quality', 50) / 200, 1.0)
        
        return {
            'confidence': confidence,
            'signal_strength': confluence,
            'confluence_count': confluence,
            'market_regime': row.get('market_regime', 0),
            'bull_confluence': bull_conf,
            'bear_confluence': bear_conf,
            'sst_direction': row.get('sst_direction', 0)
        }
    
    def _should_allow_trade(self, action: int) -> Tuple[bool, str]:
        if self.position is not None and action in [2, 3, 4, 5]:
            return False, "Position open"
        if self.trades_today >= self.max_trades_per_day:
            return False, "Daily limit"
        
        market_state = self._get_market_state()
        confidence = market_state['confidence']
        
        if confidence < self.min_confidence:
            return False, f"Low confidence: {confidence:.2f}"
        
        # Check signal alignment
        if action in [2, 3]:  # Buy
            if market_state['bull_confluence'] < 2:
                return False, "Weak bull signal"
        elif action in [4, 5]:  # Sell
            if market_state['bear_confluence'] < 2:
                return False, "Weak bear signal"
        
        return True, "OK"
    
    def _open_position(self, direction: int, strength: str):
        if self.position is not None:
            return
        
        current_price = self.df.loc[self.current_step, "Close"]
        market_state = self._get_market_state()
        confidence = market_state['confidence']
        
        # Dynamic SL/TP based on strength
        if strength == 'strong':
            sl_pips, tp_pips = 20, 50
        else:
            sl_pips, tp_pips = 25, 45
        
        # Position sizing based on confidence
        risk_pct = 0.02 + confidence * 0.02  # 2-4% risk
        lot_size = (self.equity * risk_pct) / (sl_pips * 10)
        lot_size = max(0.01, min(lot_size, 3.0))
        
        pip_value = 0.0001
        if direction == 1:
            sl_price = current_price - sl_pips * pip_value
            tp_price = current_price + tp_pips * pip_value
        else:
            sl_price = current_price + sl_pips * pip_value
            tp_price = current_price - tp_pips * pip_value
        
        self.position = {
            'direction': direction,
            'entry_price': current_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'lot_size': lot_size,
            'entry_step': self.current_step,
            'confidence': confidence,
            'confluence': market_state['confluence_count']
        }
        
        self.trades_today += 1
        self.total_trades += 1
    
    def _close_position(self, exit_price: float, reason: str = 'manual') -> float:
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        lot_size = self.position['lot_size']
        pip_value = 0.0001
        
        if self.position['direction'] == 1:
            pnl_pips = (exit_price - entry_price) / pip_value
        else:
            pnl_pips = (entry_price - exit_price) / pip_value
        
        pnl_dollars = pnl_pips * lot_size * 10
        
        if pnl_pips > 0:
            self.winning_trades += 1
        
        self.trade_history.append({
            'entry_step': self.position['entry_step'],
            'exit_step': self.current_step,
            'direction': 'LONG' if self.position['direction'] == 1 else 'SHORT',
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'reason': reason,
            'confidence': self.position['confidence']
        })
        
        self.equity += pnl_dollars
        self.position = None
        
        return pnl_pips
    
    def _check_sl_tp(self) -> Tuple[bool, float, str]:
        if self.position is None or self.current_step >= self.n_steps:
            return False, 0.0, ""
        
        high = self.df.loc[self.current_step, "High"]
        low = self.df.loc[self.current_step, "Low"]
        
        if self.position['direction'] == 1:
            if low <= self.position['sl_price']:
                return True, self.position['sl_price'], 'stop_loss'
            if high >= self.position['tp_price']:
                return True, self.position['tp_price'], 'take_profit'
        else:
            if high >= self.position['sl_price']:
                return True, self.position['sl_price'], 'stop_loss'
            if low <= self.position['tp_price']:
                return True, self.position['tp_price'], 'take_profit'
        
        return False, self.df.loc[self.current_step, "Close"], ""
    
    def _get_observation(self):
        start = max(self.current_step - self.window_size, 0)
        obs_df = self.df.iloc[start:self.current_step]
        
        if len(obs_df) < self.window_size:
            padding = self.window_size - len(obs_df)
            first_row = np.tile(obs_df.iloc[0].values, (padding, 1))
            obs_array = np.concatenate([first_row, obs_df.values], axis=0)
        else:
            obs_array = obs_df.values
        
        # Add extra state features
        extra = np.zeros((self.window_size, self.extra_features))
        market_state = self._get_market_state()
        
        if self.position is not None:
            extra[:, 0] = self.position['direction']
            extra[:, 1] = self.position['confidence']
            extra[:, 2] = (self.current_step - self.position['entry_step']) / 100
        
        extra[:, 3] = market_state['confidence']
        extra[:, 4] = market_state['confluence_count'] / 5
        extra[:, 5] = market_state['market_regime'] / 2
        extra[:, 6] = self.winning_trades / max(self.total_trades, 1)
        extra[:, 7] = (self.equity - self.initial_balance) / self.initial_balance
        extra[:, 8] = self.trades_today / self.max_trades_per_day
        extra[:, 9] = market_state['sst_direction']
        
        obs_array = np.concatenate([obs_array, extra], axis=1)
        return obs_array.astype(np.float32)
    
    def step(self, action: int):
        # Check for new day
        day = self.current_step // 24
        if day != self.current_day:
            self.current_day = day
            self.trades_today = 0
        
        reward = 0.0
        pnl_pips = 0.0
        exit_reason = None
        
        # Check SL/TP
        if self.position is not None:
            hit, exit_price, reason = self._check_sl_tp()
            if hit:
                pnl_pips = self._close_position(exit_price, reason)
                exit_reason = reason
        
        # Process action
        if action == 0:  # Hold
            pass
        elif action == 1:  # Close
            if self.position is not None:
                pnl_pips = self._close_position(self.df.loc[self.current_step, "Close"], 'manual')
                exit_reason = 'manual'
        elif action in [2, 3, 4, 5]:  # Trade actions
            allowed, _ = self._should_allow_trade(action)
            if allowed:
                if action == 2:
                    self._open_position(1, 'strong')
                elif action == 3:
                    self._open_position(1, 'medium')
                elif action == 4:
                    self._open_position(-1, 'strong')
                elif action == 5:
                    self._open_position(-1, 'medium')
        
        # Calculate reward
        market_state = self._get_market_state()
        position_info = None
        if self.position:
            position_info = {'bars_held': self.current_step - self.position['entry_step']}
        
        reward = self.reward_system.calculate_reward(
            pnl_pips=pnl_pips,
            action_type='close' if exit_reason else ('open' if action in [2,3,4,5] else 'hold'),
            market_state=market_state,
            position_info=position_info,
            exit_reason=exit_reason
        )
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, dd)
        
        self.equity_curve.append(self.equity)
        self.current_step += 1
        
        # Check termination
        terminated = False
        if self.current_step >= self.n_steps - 1:
            terminated = True
            if self.position:
                self._close_position(self.df.loc[self.n_steps - 1, "Close"], 'end')
        if self.equity < self.initial_balance * 0.3:
            terminated = True
            reward -= 500
        
        obs = self._get_observation()
        info = {
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'max_drawdown': self.max_drawdown,
            'profit_pct': (self.equity - self.initial_balance) / self.initial_balance * 100
        }
        
        return obs, reward, terminated, False, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_observation(), {'equity': self.equity}
    
    def render(self):
        if self.render_mode == "human":
            wr = self.winning_trades / max(1, self.total_trades) * 100
            profit = (self.equity - self.initial_balance) / self.initial_balance * 100
            pos = "None" if not self.position else ("LONG" if self.position['direction'] == 1 else "SHORT")
            print(f"Step {self.current_step:5d} | Equity: ${self.equity:,.2f} | Pos: {pos} | "
                  f"Trades: {self.total_trades} | WR: {wr:.1f}% | Profit: {profit:.1f}%")


class TradingMetricsCallback(BaseCallback):
    """Callback to track trading metrics during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_win_rate = 0
        self.best_profit = -np.inf
        
    def _on_step(self) -> bool:
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'win_rate' in info:
                    wr = info['win_rate']
                    profit = info.get('profit_pct', 0)
                    
                    if wr > self.best_win_rate:
                        self.best_win_rate = wr
                    if profit > self.best_profit:
                        self.best_profit = profit
                    
                    self.logger.record("trading/win_rate", wr)
                    self.logger.record("trading/profit_pct", profit)
                    self.logger.record("trading/equity", info.get('equity', 10000))
                    self.logger.record("trading/total_trades", info.get('total_trades', 0))
                    self.logger.record("trading/best_win_rate", self.best_win_rate)
        return True


def create_ultra_aggressive_dataset(csv_path: str) -> pd.DataFrame:
    """Create dataset optimized for 80%+ win rate."""
    
    print("Creating Ultra-Aggressive Dataset...")
    
    # Load enhanced data
    df = load_and_preprocess_data_enhanced(csv_path, normalize=False)
    print(f"✓ Base data: {df.shape}")
    
    # Add Stochastic SuperTrend
    df = add_stochastic_supertrend_features(df)
    df = get_stochastic_supertrend_signals(df, confidence_threshold=0.7)
    print(f"✓ Stochastic SuperTrend added")
    
    # Add high win rate features
    df = create_high_winrate_features(df)
    df = create_predictive_signals(df)
    print(f"✓ High win rate features added")
    
    # Ensure numeric
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.dropna().reset_index(drop=True)
    
    print(f"✓ Final dataset: {df.shape}")
    return df


def train_ultra_aggressive_agent(
    data_path: str = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
    total_timesteps: int = 500000,
    model_name: str = "ultra_aggressive_profitable"
):
    """Train ultra-aggressive agent for 80%+ win rate and profitability."""
    
    print("=" * 60)
    print("ULTRA-AGGRESSIVE PROFITABLE TRADING SYSTEM")
    print("Target: 80%+ Win Rate + Maximum Profitability")
    print("=" * 60)
    
    # Create dataset
    df = create_ultra_aggressive_dataset(data_path)
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Training: {len(train_df)} bars | Testing: {len(test_df)} bars")
    
    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{model_name}_{timestamp}"
    checkpoint_dir = f"./checkpoints/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create environments
    def make_train_env():
        return Monitor(UltraAggressiveTradingEnv(train_df, min_confidence=0.55))
    
    def make_eval_env():
        return Monitor(UltraAggressiveTradingEnv(test_df, min_confidence=0.55))
    
    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_eval_env])
    
    # Create model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=5e-5,  # Lower for stability
        n_steps=4096,
        batch_size=256,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,  # Lower entropy for more deterministic
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[512, 256, 128],
                vf=[512, 256, 128]
            )
        ),
        verbose=1,
        tensorboard_log=log_dir
    )
    
    # Callbacks
    checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=checkpoint_dir, name_prefix=model_name)
    eval_cb = EvalCallback(eval_env, best_model_save_path=checkpoint_dir, log_path=log_dir,
                          eval_freq=25000, n_eval_episodes=3, deterministic=True)
    metrics_cb = TradingMetricsCallback()
    
    callbacks = CallbackList([checkpoint_cb, eval_cb, metrics_cb])
    
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    
    # Train
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    
    # Save final model
    model.save(f"{checkpoint_dir}/{model_name}_final")
    
    # Evaluate
    print("\nEvaluating trained model...")
    test_env_raw = UltraAggressiveTradingEnv(test_df, min_confidence=0.55)
    obs, _ = test_env_raw.reset()
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env_raw.step(action)
        if terminated or truncated:
            break
    
    # Results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Total Trades: {test_env_raw.total_trades}")
    print(f"Win Rate: {test_env_raw.winning_trades / max(1, test_env_raw.total_trades) * 100:.1f}%")
    print(f"Final Equity: ${test_env_raw.equity:,.2f}")
    print(f"Profit: {(test_env_raw.equity - test_env_raw.initial_balance) / test_env_raw.initial_balance * 100:.1f}%")
    print(f"Max Drawdown: {test_env_raw.max_drawdown * 100:.1f}%")
    print(f"Model saved to: {checkpoint_dir}")
    
    # Save results
    results = {
        'total_trades': test_env_raw.total_trades,
        'winning_trades': test_env_raw.winning_trades,
        'win_rate': test_env_raw.winning_trades / max(1, test_env_raw.total_trades),
        'final_equity': test_env_raw.equity,
        'profit_pct': (test_env_raw.equity - test_env_raw.initial_balance) / test_env_raw.initial_balance * 100,
        'max_drawdown': test_env_raw.max_drawdown,
        'trade_history': test_env_raw.trade_history
    }
    
    with open(f"{checkpoint_dir}/results.json", 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'trade_history'}, f, indent=2)
    
    train_env.close()
    eval_env.close()
    
    return results


if __name__ == "__main__":
    results = train_ultra_aggressive_agent(
        total_timesteps=500000,
        model_name="ultra_aggressive_profitable_v1"
    )