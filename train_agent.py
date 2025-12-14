import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


class TradingMetricsCallback(BaseCallback):
    """
    Custom callback to log trading-specific metrics during training.
    """
    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.max_drawdowns = []

    def _on_step(self) -> bool:
        # Log info from the environment
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'win_rate' in info:
                    self.win_rates.append(info['win_rate'])
                if 'max_drawdown' in info:
                    self.max_drawdowns.append(info['max_drawdown'])
        return True

    def _on_rollout_end(self) -> None:
        # Log average metrics at the end of each rollout
        if self.win_rates:
            avg_win_rate = np.mean(self.win_rates[-100:])
            self.logger.record("trading/avg_win_rate", avg_win_rate)
        if self.max_drawdowns:
            avg_drawdown = np.mean(self.max_drawdowns[-100:])
            self.logger.record("trading/avg_max_drawdown", avg_drawdown)


def make_env(df, window_size, sl_options, tp_options, rank=0, seed=0):
    """Create a wrapped environment."""
    def _init():
        env = ForexTradingEnv(
            df=df,
            window_size=window_size,
            sl_options=sl_options,
            tp_options=tp_options,
            initial_balance=10000.0,
            risk_per_trade=0.02,
            max_trades_per_day=5
        )
        env = Monitor(env)
        return env
    return _init


def train_agent(
    data_path: str = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
    total_timesteps: int = 100000,
    window_size: int = 30,
    sl_options: list = None,
    tp_options: list = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.05,  # Increased for more exploration
    save_freq: int = 10000,
    eval_freq: int = 5000,
    n_envs: int = 1,
    model_name: str = "ppo_forex_trading"
):
    """
    Train a PPO agent on Forex trading environment with enhanced configuration.

    Args:
        data_path: Path to training data CSV
        total_timesteps: Total training steps
        window_size: Observation window size
        sl_options: Stop loss options in pips
        tp_options: Take profit options in pips
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient (exploration)
        save_freq: Checkpoint save frequency
        eval_freq: Evaluation frequency
        n_envs: Number of parallel environments
        model_name: Name for saved model
    """
    # Default options
    sl_options = sl_options or [30, 50, 70]
    tp_options = tp_options or [30, 50, 70]

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/{model_name}_{timestamp}"
    checkpoint_dir = f"./checkpoints/{model_name}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 60)
    print("FOREX TRADING RL AGENT - TRAINING")
    print("=" * 60)
    print(f"Loading data from: {data_path}")

    # Load and preprocess data
    df = load_and_preprocess_data(data_path, normalize=True)
    print(f"Data loaded: {len(df)} bars, {df.shape[1]} features")

    # Split data for training and evaluation (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    eval_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Training data: {len(train_df)} bars")
    print(f"Evaluation data: {len(eval_df)} bars")

    # Create training environments
    if n_envs > 1:
        train_env = SubprocVecEnv([
            make_env(train_df, window_size, sl_options, tp_options, i)
            for i in range(n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_env(train_df, window_size, sl_options, tp_options)
        ])

    # Create evaluation environment
    eval_env = DummyVecEnv([
        make_env(eval_df, window_size, sl_options, tp_options)
    ])

    # Define policy network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128]   # Value network
        )
    )

    print("\nModel Configuration:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  N Steps: {n_steps}")
    print(f"  N Epochs: {n_epochs}")
    print(f"  Gamma: {gamma}")
    print(f"  Entropy Coef: {ent_coef}")
    print(f"  Total Timesteps: {total_timesteps:,}")

    # Create PPO model with optimized hyperparameters
    model = PPO(
        policy="MlpPolicy",
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
        tensorboard_log=log_dir
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_name
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    trading_callback = TradingMetricsCallback(verbose=1)

    callback_list = CallbackList([
        checkpoint_callback,
        eval_callback,
        trading_callback
    ])

    print("\nStarting training...")
    print("=" * 60)

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True
    )

    # Save final model
    final_model_path = f"./models/{model_name}_final"
    os.makedirs("./models", exist_ok=True)
    model.save(final_model_path)
    print(f"\nModel saved to: {final_model_path}")

    # Quick evaluation on training data using raw environment (not VecEnv)
    print("\n" + "=" * 60)
    print("QUICK EVALUATION ON TRAINING DATA")
    print("=" * 60)

    # Create a fresh environment for evaluation (not wrapped in VecEnv)
    eval_env_raw = ForexTradingEnv(train_df, window_size=window_size)
    obs, info = eval_env_raw.reset()
    equity_curve = []

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env_raw.step(action)

        equity_curve.append(eval_env_raw.equity)

        if terminated or truncated:
            break

    # Get final statistics
    stats = eval_env_raw.get_trade_statistics()

    print(f"\nTraining Results:")
    print(f"  Total Trades: {stats.get('total_trades', 0)}")
    print(f"  Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
    print(f"  Total PnL: ${stats.get('total_pnl_dollars', 0):,.2f}")
    print(f"  Max Drawdown: {stats.get('max_drawdown', 0)*100:.1f}%")
    print(f"  Return: {stats.get('return_pct', 0):.1f}%")
    print(f"  Sharpe Ratio: {stats.get('sharpe_ratio', 0):.2f}")

    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(equity_curve, label='Equity', color='blue')
    plt.axhline(y=10000, color='gray', linestyle='--', label='Initial Balance')
    plt.title("Equity Curve (Training Data)")
    plt.xlabel("Time Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(np.array(equity_curve) / 10000 - 1, label='Returns', color='green')
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.title("Cumulative Returns (%)")
    plt.xlabel("Time Steps")
    plt.ylabel("Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"./models/{model_name}_training_results.png", dpi=150)
    plt.show()

    print(f"\nTraining complete! Model saved to: {final_model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir {log_dir}")

    return model, stats


def main():
    """Main entry point with default configuration."""
    model, stats = train_agent(
        data_path="data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
        total_timesteps=100000,  # Increased from 10100 for better training
        window_size=30,
        sl_options=[30, 50, 70],
        tp_options=[30, 50, 70],
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.1,  # Higher entropy for more exploration
        save_freq=10000,
        eval_freq=5000,
        n_envs=1,
        model_name="ppo_forex_eurusd"
    )


if __name__ == "__main__":
    main()
