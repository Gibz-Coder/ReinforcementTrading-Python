"""
Ultra-Aggressive Win Rate Training System
========================================

Designed to achieve 80%+ win rates through:
1. Extremely strict entry criteria
2. Multiple confirmation requirements
3. Conservative position sizing
4. Aggressive loss penalties
5. Ensemble decision making
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from enhanced_indicators import load_and_preprocess_data_enhanced
from enhanced_trading_env import EnhancedForexTradingEnv
from high_winrate_rewards import HighWinrateRewardSystem


class UltraConservativePolicy:
    """Ultra-conservative trading policy with multiple confirmations."""
    
    def __init__(self):
        self.min_confidence = 0.85  # Very high confidence required
        self.min_regime_strength = 1.5  # Strong trend required
        self.max_volatility = 0.5  # Avoid high volatility
        self.required_confirmations = 4  # Need 4+ confirmations
        
    def should_enter_long(self, market_state: Dict, indicators: Dict) -> Tuple[bool, float]:
        """Check if should enter long position with ultra-strict criteria."""
        
        confirmations = 0
        confidence_score = 0.0
        
        # 1. Multi-timeframe trend alignment
        if market_state.get('mtf_trend_alignment', 0) >= 1:
            confirmations += 1
            confidence_score += 0.2
        
        # 2. Strong market regime
        regime = market_state.get('market_regime', 0)
        if regime >= 1.5:
            confirmations += 1
            confidence_score += 0.25
        
        # 3. RSI not overbought but bullish
        rsi = indicators.get('rsi_14', 50)
        if 45 < rsi < 70:
            confirmations += 1
            confidence_score += 0.15
        
        # 4. MACD bullish
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal and macd > 0:
            confirmations += 1
            confidence_score += 0.15
        
        # 5. Price above key moving averages
        close = indicators.get('Close', 0)
        ma_20 = indicators.get('ma_20', 0)
        ma_50 = indicators.get('ma_50', 0)
        if close > ma_20 > ma_50:
            confirmations += 1
            confidence_score += 0.15
        
        # 6. ADX shows strong trend
        adx = indicators.get('adx', 0)
        if adx > 0.5:  # Normalized ADX > 25
            confirmations += 1
            confidence_score += 0.1
        
        # 7. Low volatility environment
        vol_regime = market_state.get('vol_regime', 0)
        if vol_regime <= 0:
            confirmations += 1
            confidence_score += 0.1
        
        # 8. Active trading session
        session_activity = market_state.get('session_activity', 0)
        if session_activity >= 2:
            confirmations += 1
            confidence_score += 0.05
        
        # Ultra-strict requirements
        meets_criteria = (
            confirmations >= self.required_confirmations and
            confidence_score >= self.min_confidence and
            regime >= self.min_regime_strength and
            vol_regime <= self.max_volatility
        )
        
        return meets_criteria, confidence_score
    
    def should_enter_short(self, market_state: Dict, indicators: Dict) -> Tuple[bool, float]:
        """Check if should enter short position with ultra-strict criteria."""
        
        confirmations = 0
        confidence_score = 0.0
        
        # 1. Multi-timeframe trend alignment (bearish)
        if market_state.get('mtf_trend_alignment', 0) <= -1:
            confirmations += 1
            confidence_score += 0.2
        
        # 2. Strong bearish market regime
        regime = market_state.get('market_regime', 0)
        if regime <= -1.5:
            confirmations += 1
            confidence_score += 0.25
        
        # 3. RSI not oversold but bearish
        rsi = indicators.get('rsi_14', 50)
        if 30 < rsi < 55:
            confirmations += 1
            confidence_score += 0.15
        
        # 4. MACD bearish
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd < macd_signal and macd < 0:
            confirmations += 1
            confidence_score += 0.15
        
        # 5. Price below key moving averages
        close = indicators.get('Close', 0)
        ma_20 = indicators.get('ma_20', 0)
        ma_50 = indicators.get('ma_50', 0)
        if close < ma_20 < ma_50:
            confirmations += 1
            confidence_score += 0.15
        
        # 6. ADX shows strong trend
        adx = indicators.get('adx', 0)
        if adx > 0.5:
            confirmations += 1
            confidence_score += 0.1
        
        # 7. Low volatility environment
        vol_regime = market_state.get('vol_regime', 0)
        if vol_regime <= 0:
            confirmations += 1
            confidence_score += 0.1
        
        # 8. Active trading session
        session_activity = market_state.get('session_activity', 0)
        if session_activity >= 2:
            confirmations += 1
            confidence_score += 0.05
        
        # Ultra-strict requirements
        meets_criteria = (
            confirmations >= self.required_confirmations and
            confidence_score >= self.min_confidence and
            regime <= -self.min_regime_strength and
            vol_regime <= self.max_volatility
        )
        
        return meets_criteria, confidence_score


class UltraAggressiveEnv(EnhancedForexTradingEnv):
    """Enhanced environment with ultra-aggressive win rate focus."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_system = HighWinrateRewardSystem()
        self.conservative_policy = UltraConservativePolicy()
        
        # Ultra-conservative settings
        self.max_trades_per_day = 2  # Very limited trading
        self.min_trade_spacing = 4   # Minimum 4 hours between trades
        
    def _calculate_enhanced_reward(self, pnl_pips: float, action: int, exit_reason: str = None) -> float:
        """Ultra-aggressive reward calculation focused on win rate."""
        
        market_state = self._get_market_state()
        confidence = self._get_model_confidence()
        
        # Account information
        account_info = {
            'current_drawdown': self.current_drawdown,
            'trades_today': self.trades_today,
            'max_trades_per_day': self.max_trades_per_day,
            'equity_growth_pct': (self.equity - self.initial_balance) / self.initial_balance * 100
        }
        
        # Position information
        position_info = {}
        if self.position:
            current_price = self.df.loc[self.current_step - 1, "Close"] if self.current_step > 0 else 0
            position_info = {
                'unrealized_pnl': self._calculate_unrealized_pnl(current_price),
                'bars_held': self.bars_in_position
            }
        
        # Use high win-rate reward system
        reward = self.reward_system.calculate_reward(
            pnl_pips=pnl_pips,
            action=action,
            exit_reason=exit_reason,
            confidence=confidence,
            market_regime=market_state.get('market_regime', 0),
            position_info=position_info,
            account_info=account_info
        )
        
        # Additional ultra-aggressive penalties
        
        # Massive penalty for any loss
        if exit_reason in ['stop_loss'] and pnl_pips < 0:
            reward -= 50.0  # Extreme loss penalty
        
        # Penalty for low-confidence actions
        if action in [2, 3, 4, 5] and confidence < 0.8:
            reward -= 20.0  # Don't trade without high confidence
        
        # Penalty for trading in unfavorable conditions
        if action in [2, 3, 4, 5]:
            vol_regime = market_state.get('vol_regime', 0)
            if vol_regime > 0:  # High volatility
                reward -= 15.0
            
            regime = abs(market_state.get('market_regime', 0))
            if regime < 1:  # Weak or no trend
                reward -= 10.0
        
        # Bonus for perfect trades (high confidence + good outcome)
        if exit_reason == 'take_profit' and confidence > 0.85:
            reward += 30.0  # Perfect trade bonus
        
        return reward
    
    def _open_position(self, direction: int, aggressive: bool = False):
        """Override to add ultra-conservative entry checks."""
        
        if self.position is not None:
            return  # Already have position
        
        # Check trade spacing
        if self.current_step - self.last_trade_step < self.min_trade_spacing:
            return  # Too soon since last trade
        
        # Get current market data
        current_data = self.df.iloc[self.current_step]
        market_state = self._get_market_state()
        
        # Convert current data to indicators dict
        indicators = current_data.to_dict()
        
        # Ultra-conservative entry checks
        if direction == 1:  # Long
            should_enter, confidence = self.conservative_policy.should_enter_long(market_state, indicators)
        else:  # Short
            should_enter, confidence = self.conservative_policy.should_enter_short(market_state, indicators)
        
        # Only enter if ultra-strict criteria are met
        if should_enter:
            super()._open_position(direction, aggressive)
        # If criteria not met, no position is opened (implicit rejection)


class WinRateCallback(BaseCallback):
    """Callback to monitor and optimize for win rate."""
    
    def __init__(self, target_winrate=0.8, verbose=0):
        super().__init__(verbose)
        self.target_winrate = target_winrate
        self.episode_winrates = []
        self.best_winrate = 0.0
        
    def _on_step(self) -> bool:
        # Monitor win rate from environment info
        if self.locals.get('infos'):
            for info in self.locals['infos']:
                if 'win_rate' in info:
                    current_winrate = info['win_rate']
                    
                    # Log win rate
                    self.logger.record("winrate/current", current_winrate)
                    self.logger.record("winrate/target", self.target_winrate)
                    self.logger.record("winrate/best", self.best_winrate)
                    
                    # Update best win rate
                    if current_winrate > self.best_winrate:
                        self.best_winrate = current_winrate
                        self.logger.record("winrate/new_best", self.best_winrate)
                    
                    # Check if target achieved
                    if current_winrate >= self.target_winrate:
                        self.logger.record("winrate/target_achieved", 1)
                    
                    # Store for analysis
                    self.episode_winrates.append(current_winrate)
        
        return True


def train_ultra_aggressive_winrate_system(
    data_path: str = "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
    total_timesteps: int = 1000000,  # More training for ultra-conservative approach
    target_winrate: float = 0.8,
    model_name: str = "ultra_aggressive_winrate"
):
    """
    Train ultra-aggressive win rate focused system.
    """
    
    print("=" * 80)
    print("ULTRA-AGGRESSIVE WIN RATE TRAINING SYSTEM")
    print("=" * 80)
    print(f"Target Win Rate: {target_winrate*100:.1f}%")
    
    # Load enhanced data
    df = load_and_preprocess_data_enhanced(data_path, normalize=True)
    print(f"Data loaded: {len(df)} bars")
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Training data: {len(train_df)} bars")
    print(f"Test data: {len(test_df)} bars")
    
    # Create ultra-aggressive environment
    def make_env():
        env = UltraAggressiveEnv(
            df=train_df,
            window_size=30,
            initial_balance=10000.0,
            max_trades_per_day=2,  # Very conservative
            spread_pips=1.5,
            slippage_pips=0.5
        )
        return env
    
    env = DummyVecEnv([make_env])
    
    # Ultra-conservative model configuration
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256, 128],  # Deep networks
            vf=[512, 512, 256, 128]
        ),
        activation_fn=torch.nn.ReLU
    )
    
    # Create model with conservative hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0001,  # Very slow learning
        n_steps=4096,          # Large batch
        batch_size=128,        # Large batch size
        n_epochs=20,           # More epochs
        gamma=0.999,           # Very long-term thinking
        gae_lambda=0.98,
        clip_range=0.1,        # Very conservative updates
        ent_coef=0.01,         # Low exploration
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Setup callbacks
    winrate_callback = WinRateCallback(target_winrate=target_winrate, verbose=1)
    
    print(f"\nStarting ultra-aggressive training for {total_timesteps:,} timesteps...")
    print("Focus: Achieving 80%+ win rate through ultra-conservative entries")
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=winrate_callback,
        progress_bar=False
    )
    
    # Save model
    model.save(f"{model_name}_final")
    print(f"Model saved: {model_name}_final.zip")
    
    # Test on out-of-sample data
    print("\n" + "="*60)
    print("TESTING ON OUT-OF-SAMPLE DATA")
    print("="*60)
    
    test_env = UltraAggressiveEnv(
        df=test_df,
        window_size=30,
        initial_balance=10000.0,
        max_trades_per_day=2,
        spread_pips=1.5,
        slippage_pips=0.5
    )
    
    obs, _ = test_env.reset()
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        if terminated or truncated:
            break
    
    # Get final results
    results = test_env.get_enhanced_statistics()
    
    print(f"\nULTRA-AGGRESSIVE SYSTEM RESULTS:")
    print(f"{'='*50}")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    print(f"Total Return: {results.get('return_pct', 0):.1f}%")
    print(f"Max Drawdown: {results.get('max_drawdown', 0)*100:.1f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    
    # Check if target achieved
    achieved_winrate = results.get('win_rate', 0)
    if achieved_winrate >= target_winrate:
        print(f"\n🎯 TARGET ACHIEVED! Win rate: {achieved_winrate*100:.1f}%")
    else:
        print(f"\n⚠️  Target not achieved. Win rate: {achieved_winrate*100:.1f}% (Target: {target_winrate*100:.1f}%)")
        print("Consider:")
        print("- Increasing training timesteps")
        print("- Raising confidence thresholds")
        print("- Adding more confirmation requirements")
    
    return model, results


if __name__ == "__main__":
    # Run ultra-aggressive win rate training
    model, results = train_ultra_aggressive_winrate_system(
        total_timesteps=1000000,
        target_winrate=0.8
    )