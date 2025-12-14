"""
Improved Training Configuration for Higher Winrate
================================================

Key improvements:
1. Ensemble learning with multiple models
2. Advanced hyperparameter optimization
3. Better reward shaping for winrate focus
4. Enhanced curriculum learning
"""

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np

class ImprovedTrainingConfig:
    """Enhanced training configuration for better winrate."""
    
    @staticmethod
    def get_optimized_hyperparams():
        """Hyperparameters optimized for forex trading winrate."""
        return {
            # Learning rate schedule (start high, decay)
            'learning_rate': lambda progress: 0.0005 * (1 - progress * 0.8),
            
            # Larger batch sizes for more stable learning
            'batch_size': 128,  # Increased from 64
            'n_steps': 4096,    # Increased from 2048
            
            # More conservative policy updates
            'clip_range': 0.15,  # Reduced from 0.2
            'clip_range_vf': 0.15,
            
            # Enhanced exploration
            'ent_coef': 0.05,    # Reduced from 0.1 for more focused actions
            'vf_coef': 0.5,
            
            # Better value function learning
            'n_epochs': 15,      # Increased from 10
            'gamma': 0.995,      # Increased from 0.99 for longer-term thinking
            'gae_lambda': 0.98,  # Increased from 0.95
            
            # Network architecture
            'policy_kwargs': {
                'net_arch': {
                    'pi': [512, 512, 256, 128],  # Deeper policy network
                    'vf': [512, 512, 256, 128]   # Deeper value network
                },
                'activation_fn': 'relu',
                'ortho_init': False,
                'log_std_init': -2.0  # More conservative initial exploration
            }
        }
    
    @staticmethod
    def get_winrate_focused_reward_config():
        """Reward configuration focused on high winrate."""
        return {
            'win_bonus': 15.0,           # Increased from 10.0
            'loss_penalty': -8.0,        # Increased penalty
            'confidence_multiplier': 3.0, # Reward high-confidence trades more
            'regime_alignment_bonus': 5.0,
            'overtrading_penalty': -10.0,
            'drawdown_penalty_threshold': 0.10,  # 10% instead of 15%
            'consecutive_loss_penalty': -5.0,    # Penalty for losing streaks
        }

def optimize_hyperparameters_with_optuna(n_trials=50):
    """Use Optuna to find optimal hyperparameters for winrate."""
    
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        n_steps = trial.suggest_categorical('n_steps', [2048, 4096, 8192])
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 0.01, 0.2)
        
        # Train model with suggested parameters
        # (Implementation would go here)
        
        # Return winrate as objective to maximize
        return winrate
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

class EnsembleTraining:
    """Train multiple models and use ensemble for better performance."""
    
    def __init__(self, n_models=3):
        self.n_models = n_models
        self.models = []
    
    def train_ensemble(self, env, total_timesteps):
        """Train multiple models with different configurations."""
        
        configs = [
            {'learning_rate': 0.0003, 'ent_coef': 0.05},  # Conservative
            {'learning_rate': 0.0005, 'ent_coef': 0.1},   # Balanced  
            {'learning_rate': 0.0002, 'ent_coef': 0.02}   # Very conservative
        ]
        
        for i, config in enumerate(configs[:self.n_models]):
            print(f"Training model {i+1}/{self.n_models}")
            
            model = PPO(
                'MlpPolicy',
                env,
                **config,
                verbose=1
            )
            
            model.learn(total_timesteps=total_timesteps // self.n_models)
            self.models.append(model)
    
    def predict_ensemble(self, obs):
        """Get ensemble prediction from all models."""
        predictions = []
        
        for model in self.models:
            action, _ = model.predict(obs, deterministic=True)
            predictions.append(action)
        
        # Use majority voting or confidence-weighted average
        return self._ensemble_decision(predictions)
    
    def _ensemble_decision(self, predictions):
        """Combine predictions from multiple models."""
        # Simple majority voting for now
        return max(set(predictions), key=predictions.count)