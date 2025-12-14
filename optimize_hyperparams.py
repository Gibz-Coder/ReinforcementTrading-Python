"""
Hyperparameter optimization for Forex Trading RL Agent using Optuna.

GENERALIZATION-FOCUSED VERSION:
- Higher entropy coefficients (0.15 - 0.5) to prevent overfitting
- Simpler network architectures (tiny, small)
- Validation on OUT-OF-SAMPLE test data
- Lower learning rates for stability
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import torch
from typing import Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv


def create_env(df, window_size=30):
    """Create a trading environment."""
    env = ForexTradingEnv(
        df=df,
        window_size=window_size,
        sl_options=[30, 50, 70],
        tp_options=[30, 50, 70],
        initial_balance=10000.0,
        risk_per_trade=0.02,
        max_trades_per_day=5
    )
    return env


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sample PPO hyperparameters with GENERALIZATION focus.

    Key changes for preventing overfitting:
    - Higher entropy coefficient (0.15 - 0.5) forces more exploration
    - Simpler networks (tiny, small) prevent memorization
    - Lower learning rates (1e-5 - 5e-4) for stable learning
    - Fewer epochs (3-10) to prevent overfitting each batch
    """

    # Learning rate (log scale) - lower range for stability
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)

    # Discount factor - slightly lower to focus on near-term rewards
    gamma = trial.suggest_float("gamma", 0.85, 0.99)

    # GAE lambda
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)

    # Entropy coefficient (exploration) - HIGH values for generalization
    ent_coef = trial.suggest_float("ent_coef", 0.15, 0.5, log=True)

    # Value function coefficient
    vf_coef = trial.suggest_float("vf_coef", 0.3, 0.7)

    # Max gradient norm - lower for stability
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 0.7)

    # Number of steps per update - larger for more stable gradients
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])

    # Batch size
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Number of epochs - fewer to prevent overfitting
    n_epochs = trial.suggest_int("n_epochs", 3, 10)

    # Clip range
    clip_range = trial.suggest_float("clip_range", 0.15, 0.35)

    # Network architecture - SIMPLER networks for better generalization
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small"])

    net_arch_map = {
        "tiny": dict(pi=[32, 32], vf=[32, 32]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
    }

    return {
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "clip_range": clip_range,
        "policy_kwargs": dict(net_arch=net_arch_map[net_arch_type]),
    }


class TrialEvalCallback(EvalCallback):
    """Callback for pruning unpromising trials."""

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=5000, **kwargs):
        super().__init__(eval_env, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            result = super()._on_step()
            self.eval_idx += 1
            # Report to Optuna
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune if needed
            if self.trial.should_prune():
                self.is_pruned = True
                return False
            return result
        return True


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function with OUT-OF-SAMPLE validation.

    Key for generalization: Train on training data, but evaluate
    on separate test data to find hyperparameters that generalize.
    """

    # Load TRAINING data
    train_df = load_and_preprocess_data(
        'data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
        normalize=True
    )

    # Load OUT-OF-SAMPLE test data for validation
    test_df = load_and_preprocess_data(
        'data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv',
        normalize=True
    )

    # Create environments - TRAIN and TEST are different!
    train_env = DummyVecEnv([lambda: create_env(train_df)])
    eval_env = DummyVecEnv([lambda: create_env(test_df)])  # Out-of-sample validation!

    # Sample hyperparameters
    params = sample_ppo_params(trial)

    # Create model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=f"./logs/optuna_trial_{trial.number}",
        **params
    )

    # Evaluation callback with pruning on TEST data
    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=3,
        eval_freq=10000,
        deterministic=True,
        verbose=0
    )

    try:
        # Train for limited steps during optimization
        model.learn(total_timesteps=75000, callback=eval_callback)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

    if eval_callback.is_pruned:
        raise optuna.TrialPruned()

    # Final evaluation on TEST data (out-of-sample)
    mean_reward = eval_callback.last_mean_reward

    # Clean up
    train_env.close()
    eval_env.close()

    return mean_reward


def run_optimization(n_trials: int = 50, n_jobs: int = 1, study_name: str = "ppo_forex_optimization"):
    """
    Run hyperparameter optimization.

    Args:
        n_trials: Number of trials to run
        n_jobs: Number of parallel jobs (-1 for all cores)
        study_name: Name of the study for saving/resuming
    """
    # Create study with TPE sampler and median pruner
    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=f"sqlite:///optuna_{study_name}.db",
        load_if_exists=True
    )

    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 60)
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Parallel jobs: {n_jobs}")
    print("=" * 60)

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        print(f"\nBest trial:")
        print(f"  Value (Mean Reward): {study.best_trial.value:.2f}")
        print(f"  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        # Save best params to file
        import json
        with open("best_hyperparams.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        print(f"\nBest hyperparameters saved to: best_hyperparams.json")

    return study


def train_with_best_params(
    params_file: str = "best_hyperparams.json",
    total_timesteps: int = 200000
):
    """Train a model using the best hyperparameters found."""
    import json

    print("=" * 60)
    print("TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 60)

    # Load best params
    with open(params_file, "r") as f:
        best_params = json.load(f)

    print(f"Loaded hyperparameters from: {params_file}")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Map net_arch string to actual architecture (includes tiny for generalization)
    net_arch_map = {
        "tiny": dict(pi=[32, 32], vf=[32, 32]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[128, 128], vf=[128, 128]),
        "large": dict(pi=[256, 256], vf=[256, 256]),
    }

    if "net_arch" in best_params:
        net_arch_type = best_params.pop("net_arch")
        best_params["policy_kwargs"] = dict(net_arch=net_arch_map[net_arch_type])

    # Load data
    train_df = load_and_preprocess_data(
        'data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv',
        normalize=True
    )

    # Create environment
    train_env = DummyVecEnv([lambda: create_env(train_df)])
    eval_env = DummyVecEnv([lambda: create_env(train_df)])

    # Create model with best params
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./logs/optimized_training",
        **best_params
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/optimized/",
        log_path="./logs/optimized/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1
    )

    # Train
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save("./models/ppo_forex_optimized_final")
    print(f"\nModel saved to: ./models/ppo_forex_optimized_final")

    # Final evaluation
    raw_env = create_env(train_df)
    obs, _ = raw_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = raw_env.step(action)
        if term or trunc:
            break

    stats = raw_env.get_trade_statistics()
    print("\nFinal Results:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"  Total PnL: ${stats['total_pnl_dollars']:,.2f}")
    print(f"  Return: {stats['return_pct']:.1f}%")

    return model, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter optimization for Forex RL")
    parser.add_argument("--mode", choices=["optimize", "train"], default="optimize",
                        help="Mode: 'optimize' for hyperparam search, 'train' to use best params")
    parser.add_argument("--trials", type=int, default=30, help="Number of optimization trials")
    parser.add_argument("--timesteps", type=int, default=200000, help="Training timesteps")

    args = parser.parse_args()

    if args.mode == "optimize":
        run_optimization(n_trials=args.trials)
    else:
        train_with_best_params(total_timesteps=args.timesteps)
