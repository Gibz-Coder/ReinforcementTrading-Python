"""
Enhanced Training Script with Advanced Features
==============================================

Key improvements:
1. Uses enhanced indicators and multi-timeframe analysis
2. Integrates advanced risk management
3. Implements curriculum learning
4. Walk-forward analysis for robust validation
5. Advanced model architectures (LSTM support)
6. Comprehensive monitoring and logging
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn

from enhanced_indicators import load_and_preprocess_data_enhanced
from enhanced_trading_env import EnhancedForexTradingEnv


class LSTMPolicy(ActorCriticPolicy):
    """
    Custom policy with LSTM layers for better sequence modeling.
    """
    
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.lstm_hidden_size = kwargs.pop('lstm_hidden_size', 128)
        self.lstm_num_layers = kwargs.pop('lstm_num_layers', 2)
        
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        """Build the LSTM-based feature extractor."""
        
        # Get input dimensions
        input_dim = self.observation_space.shape[-1]  # Last dimension (features)
        sequence_length = self.observation_space.shape[0]  # First dimension (time steps)
        
        # LSTM feature extractor
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            dropout=0.1 if self.lstm_num_layers > 1 else 0
        )
        
        # Policy and value networks
        self.policy_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space.n)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, obs, deterministic: bool = False):
        """Forward pass through LSTM and policy/value networks."""
        
        # Reshape observation for LSTM: (batch_size, seq_len, features)
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        features = obs.shape[2]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(obs)
        
        # Use the last output of the sequence
        features = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Policy and value outputs
        action_logits = self.policy_net(features)
        value = self.value_net(features)
        
        return action_logits, value


class EnhancedTradingCallback(BaseCallback):
    """
    Enhanced callback for comprehensive trading metrics logging.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.trading_metrics = []
        self.best_performance = -np.inf
        
    def _on_step(self) -> bool:
        # Collect info from environments
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'equity' in info:
                    # Log trading-specific metrics
                    self.logger.record("trading/equity", info['equity'])
                    self.logger.record("trading/win_rate", info.get('win_rate', 0))
                    self.logger.record("trading/max_drawdown", info.get('max_drawdown', 0))
                    self.logger.record("trading/total_trades", info.get('total_trades', 0))
                    self.logger.record("trading/market_regime", info.get('market_regime', 0))
                    self.logger.record("trading/confidence", info.get('confidence', 0))
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Log aggregated metrics
        if self.trading_metrics:
            avg_equity = np.mean([m.get('equity', 10000) for m in self.trading_metrics[-100:]])
            avg_win_rate = np.mean([m.get('win_rate', 0) for m in self.trading_metrics[-100:]])
            
            self.logger.record("trading/avg_equity_100", avg_equity)
            self.logger.record("trading/avg_win_rate_100", avg_win_rate)
            
            # Track best performance
            if avg_equity > self.best_performance:
                self.best_performance = avg_equity
                self.logger.record("trading/best_equity", self.best_performance)


class CurriculumLearning:
    """
    Curriculum learning for gradual difficulty increase.
    
    Starts with easier market conditions and gradually introduces
    more challenging scenarios.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.current_difficulty = 0
        self.max_difficulty = 3
        
    def get_training_data(self, difficulty_level: int = None) -> pd.DataFrame:
        """
        Get training data based on difficulty level.
        
        Levels:
        0: Strong trending periods only
        1: Trending + some ranging periods
        2: Mixed conditions
        3: All market conditions including high volatility
        """
        if difficulty_level is None:
            difficulty_level = self.current_difficulty
        
        if difficulty_level == 0:
            # Only strong trending periods
            mask = abs(self.df.get('market_regime', 0)) >= 2
        elif difficulty_level == 1:
            # Trending periods (moderate and strong)
            mask = abs(self.df.get('market_regime', 0)) >= 1
        elif difficulty_level == 2:
            # All except extreme volatility
            mask = self.df.get('vol_regime', 0) != 1  # Exclude high vol
        else:
            # All market conditions
            mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        return self.df[mask].reset_index(drop=True)
    
    def should_increase_difficulty(self, performance_metrics: Dict) -> bool:
        """
        Decide whether to increase curriculum difficulty.
        
        Criteria:
        - Win rate > 50%
        - Positive returns
        - Reasonable number of trades
        """
        win_rate = performance_metrics.get('win_rate', 0)
        return_pct = performance_metrics.get('return_pct', 0)
        total_trades = performance_metrics.get('total_trades', 0)
        
        return (win_rate > 0.5 and 
                return_pct > 5.0 and 
                total_trades > 20 and
                self.current_difficulty < self.max_difficulty)
    
    def increase_difficulty(self):
        """Increase curriculum difficulty."""
        if self.current_difficulty < self.max_difficulty:
            self.current_difficulty += 1
            print(f"📈 Curriculum difficulty increased to level {self.current_difficulty}")


class WalkForwardAnalysis:
    """
    Walk-forward analysis for robust model validation.
    
    Trains on a rolling window and tests on out-of-sample data
    to ensure the model generalizes well.
    """
    
    def __init__(self, df: pd.DataFrame, train_size: int = 8760, test_size: int = 2190):
        """
        Initialize walk-forward analysis.
        
        Args:
            df: Full dataset
            train_size: Training window size (hours) - default 1 year
            test_size: Test window size (hours) - default 3 months
        """
        self.df = df
        self.train_size = train_size
        self.test_size = test_size
        self.results = []
    
    def get_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for walk-forward analysis.
        
        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        
        # Calculate number of possible splits
        total_size = len(self.df)
        step_size = self.test_size  # Move forward by test_size each time
        
        start_idx = 0
        while start_idx + self.train_size + self.test_size <= total_size:
            train_end = start_idx + self.train_size
            test_end = train_end + self.test_size
            
            train_df = self.df.iloc[start_idx:train_end].reset_index(drop=True)
            test_df = self.df.iloc[train_end:test_end].reset_index(drop=True)
            
            splits.append((train_df, test_df))
            start_idx += step_size
        
        return splits
    
    def analyze_split_performance(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """
        Analyze the characteristics of a train/test split.
        
        Returns:
            Dictionary with split characteristics
        """
        train_regimes = train_df.get('market_regime', pd.Series([0] * len(train_df)))
        test_regimes = test_df.get('market_regime', pd.Series([0] * len(test_df)))
        
        return {
            'train_period': f"{train_df.index[0]} to {train_df.index[-1]}",
            'test_period': f"{test_df.index[0]} to {test_df.index[-1]}",
            'train_regime_dist': train_regimes.value_counts().to_dict(),
            'test_regime_dist': test_regimes.value_counts().to_dict(),
            'train_vol_mean': train_df.get('realized_vol_24h', pd.Series([0] * len(train_df))).mean(),
            'test_vol_mean': test_df.get('realized_vol_24h', pd.Series([0] * len(test_df))).mean()
        }


def create_enhanced_env(df: pd.DataFrame, window_size: int = 30, rank: int = 0):
    """Create enhanced trading environment."""
    def _init():
        env = EnhancedForexTradingEnv(
            df=df,
            window_size=window_size,
            initial_balance=10000.0,
            max_trades_per_day=5,
            spread_pips=1.5,
            slippage_pips=0.5
        )
        env = Monitor(env)
        return env
    return _init


def train_enhanced_agent(
    data_path: str = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
    total_timesteps: int = 500000,
    window_size: int = 30,
    use_lstm: bool = True,
    use_curriculum: bool = True,
    use_walk_forward: bool = False,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.1,
    save_freq: int = 25000,
    eval_freq: int = 10000,
    n_envs: int = 1,
    model_name: str = "enhanced_ppo_forex"
):
    """
    Train enhanced PPO agent with advanced features.
    """
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{model_name}_{timestamp}"
    checkpoint_dir = f"./checkpoints/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("=" * 80)
    print("ENHANCED FOREX TRADING RL AGENT - TRAINING")
    print("=" * 80)
    print(f"Loading enhanced data from: {data_path}")
    
    # Load enhanced data
    df = load_and_preprocess_data_enhanced(data_path, normalize=True)
    print(f"Enhanced data loaded: {len(df)} bars, {df.shape[1]} features")
    
    # Initialize curriculum learning
    curriculum = None
    if use_curriculum:
        curriculum = CurriculumLearning(df)
        print("✓ Curriculum learning enabled")
    
    # Initialize walk-forward analysis
    walk_forward = None
    if use_walk_forward:
        walk_forward = WalkForwardAnalysis(df)
        splits = walk_forward.get_splits()
        print(f"✓ Walk-forward analysis enabled: {len(splits)} splits")
    else:
        # Standard train/test split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        splits = [(train_df, test_df)]
    
    # Training configuration
    config = {
        'total_timesteps': total_timesteps,
        'window_size': window_size,
        'use_lstm': use_lstm,
        'use_curriculum': use_curriculum,
        'use_walk_forward': use_walk_forward,
        'learning_rate': learning_rate,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'n_envs': n_envs
    }
    
    # Save configuration
    with open(f"{checkpoint_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train on each split (or just one split for standard training)
    all_results = []
    
    for split_idx, (train_df, test_df) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"TRAINING SPLIT {split_idx + 1}/{len(splits)}")
        print(f"{'='*60}")
        
        if use_walk_forward:
            split_analysis = walk_forward.analyze_split_performance(train_df, test_df)
            print(f"Train period: {split_analysis['train_period']}")
            print(f"Test period: {split_analysis['test_period']}")
        
        # Get training data (with curriculum if enabled)
        if curriculum:
            current_train_df = curriculum.get_training_data()
            print(f"Curriculum level {curriculum.current_difficulty}: {len(current_train_df)} bars")
        else:
            current_train_df = train_df
        
        # Create environments
        if n_envs > 1:
            train_env = SubprocVecEnv([
                create_enhanced_env(current_train_df, window_size, i) 
                for i in range(n_envs)
            ])
        else:
            train_env = DummyVecEnv([
                create_enhanced_env(current_train_df, window_size)
            ])
        
        eval_env = DummyVecEnv([
            create_enhanced_env(test_df, window_size)
        ])
        
        # Model configuration
        if use_lstm:
            policy_kwargs = dict(
                lstm_hidden_size=128,
                lstm_num_layers=2
            )
            policy = LSTMPolicy
        else:
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[256, 256, 128],
                    vf=[256, 256, 128]
                )
            )
            policy = "MlpPolicy"
        
        # Create model
        model = PPO(
            policy=policy,
            env=train_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=f"{log_dir}/split_{split_idx}"
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{checkpoint_dir}/split_{split_idx}",
            name_prefix=f"{model_name}_split_{split_idx}"
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{checkpoint_dir}/split_{split_idx}",
            log_path=f"{log_dir}/split_{split_idx}",
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            render=False
        )
        
        trading_callback = EnhancedTradingCallback(verbose=1)
        
        callback_list = CallbackList([
            checkpoint_callback,
            eval_callback,
            trading_callback
        ])
        
        # Training loop with curriculum learning
        if curriculum:
            # Train in curriculum stages
            timesteps_per_stage = total_timesteps // (curriculum.max_difficulty + 1)
            
            for stage in range(curriculum.max_difficulty + 1):
                print(f"\n--- Curriculum Stage {stage} ---")
                
                # Update training data for current difficulty
                stage_train_df = curriculum.get_training_data(stage)
                
                # Recreate environments with new data
                train_env.close()
                if n_envs > 1:
                    train_env = SubprocVecEnv([
                        create_enhanced_env(stage_train_df, window_size, i) 
                        for i in range(n_envs)
                    ])
                else:
                    train_env = DummyVecEnv([
                        create_enhanced_env(stage_train_df, window_size)
                    ])
                
                model.set_env(train_env)
                
                # Train for this stage
                model.learn(
                    total_timesteps=timesteps_per_stage,
                    callback=callback_list,
                    progress_bar=True,
                    reset_num_timesteps=False
                )
        else:
            # Standard training
            print(f"\nStarting training for {total_timesteps:,} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                progress_bar=True
            )
        
        # Evaluate on test set
        print(f"\nEvaluating split {split_idx + 1}...")
        test_env_raw = EnhancedForexTradingEnv(test_df, window_size=window_size)
        obs, _ = test_env_raw.reset()
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env_raw.step(action)
            
            if terminated or truncated:
                break
        
        # Get results
        split_results = test_env_raw.get_enhanced_statistics()
        split_results['split_idx'] = split_idx
        all_results.append(split_results)
        
        print(f"\nSplit {split_idx + 1} Results:")
        print(f"  Total Trades: {split_results.get('total_trades', 0)}")
        print(f"  Win Rate: {split_results.get('win_rate', 0)*100:.1f}%")
        print(f"  Return: {split_results.get('return_pct', 0):.1f}%")
        print(f"  Max Drawdown: {split_results.get('max_drawdown', 0)*100:.1f}%")
        
        # Save split model
        model.save(f"{checkpoint_dir}/split_{split_idx}_final")
        
        # Clean up
        train_env.close()
        eval_env.close()
        
        # For walk-forward, continue to next split
        # For standard training, break after first split
        if not use_walk_forward:
            break
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    if len(all_results) > 1:
        # Walk-forward results
        avg_return = np.mean([r.get('return_pct', 0) for r in all_results])
        avg_win_rate = np.mean([r.get('win_rate', 0) for r in all_results])
        avg_drawdown = np.mean([r.get('max_drawdown', 0) for r in all_results])
        
        print(f"Walk-Forward Analysis Results ({len(all_results)} splits):")
        print(f"  Average Return: {avg_return:.1f}%")
        print(f"  Average Win Rate: {avg_win_rate*100:.1f}%")
        print(f"  Average Max Drawdown: {avg_drawdown*100:.1f}%")
        
        # Save walk-forward results
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{checkpoint_dir}/walk_forward_results.csv", index=False)
    else:
        # Single split results
        result = all_results[0]
        print(f"Training Results:")
        print(f"  Total Trades: {result.get('total_trades', 0)}")
        print(f"  Win Rate: {result.get('win_rate', 0)*100:.1f}%")
        print(f"  Return: {result.get('return_pct', 0):.1f}%")
        print(f"  Max Drawdown: {result.get('max_drawdown', 0)*100:.1f}%")
        print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
    
    # Save final results
    with open(f"{checkpoint_dir}/final_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nTraining completed!")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    
    return all_results


def main():
    """Main training function with enhanced configuration."""
    
    results = train_enhanced_agent(
        data_path="data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
        total_timesteps=500000,
        window_size=30,
        use_lstm=True,
        use_curriculum=True,
        use_walk_forward=False,  # Set to True for robust validation
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.1,
        save_freq=25000,
        eval_freq=10000,
        n_envs=1,
        model_name="enhanced_ppo_forex_v2"
    )


if __name__ == "__main__":
    main()