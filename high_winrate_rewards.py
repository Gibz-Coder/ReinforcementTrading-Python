"""
High Win-Rate Focused Reward System
===================================

Designed to maximize win percentage while maintaining profitability.
Key principles:
1. Heavy penalties for losses
2. Bonuses for consecutive wins
3. Confidence-based reward scaling
4. Risk-adjusted returns
"""

import numpy as np
from typing import Dict, List, Optional

class HighWinrateRewardSystem:
    """Reward system optimized for high win rates."""
    
    def __init__(self):
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.recent_trades = []  # Store last 10 trades
        self.confidence_threshold = 0.75
        
    def calculate_reward(self, 
                        pnl_pips: float,
                        action: int,
                        exit_reason: str = None,
                        confidence: float = 0.5,
                        market_regime: int = 0,
                        position_info: Dict = None,
                        account_info: Dict = None) -> float:
        """
        Calculate reward with heavy focus on win rate.
        
        Args:
            pnl_pips: Profit/loss in pips
            action: Action taken (0=hold, 1=close, 2-5=open positions)
            exit_reason: How trade was closed
            confidence: Model confidence (0-1)
            market_regime: Market condition (-2 to 2)
            position_info: Current position details
            account_info: Account metrics (equity, drawdown, etc.)
        """
        
        reward = 0.0
        
        # === TRADE COMPLETION REWARDS ===
        if exit_reason is not None:  # Trade was closed
            if pnl_pips > 0:  # Winning trade
                reward += self._calculate_win_reward(pnl_pips, confidence, market_regime)
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                
            else:  # Losing trade
                reward += self._calculate_loss_penalty(pnl_pips, confidence, market_regime)
                self.consecutive_losses += 1
                self.consecutive_wins = 0
            
            # Track recent trades
            self.recent_trades.append({
                'pnl': pnl_pips,
                'confidence': confidence,
                'regime': market_regime
            })
            if len(self.recent_trades) > 10:
                self.recent_trades.pop(0)
        
        # === ACTION-SPECIFIC REWARDS ===
        reward += self._calculate_action_reward(action, confidence, market_regime, position_info)
        
        # === STREAK BONUSES/PENALTIES ===
        reward += self._calculate_streak_reward()
        
        # === RISK MANAGEMENT REWARDS ===
        if account_info:
            reward += self._calculate_risk_reward(account_info)
        
        # === CONFIDENCE-BASED SCALING ===
        reward = self._apply_confidence_scaling(reward, confidence)
        
        return reward
    
    def _calculate_win_reward(self, pnl_pips: float, confidence: float, market_regime: int) -> float:
        """Calculate reward for winning trades."""
        
        # Base win reward (higher than standard)
        base_reward = 20.0  # Increased from typical 10.0
        
        # PnL-based component (capped to prevent overemphasis on large wins)
        pnl_reward = min(pnl_pips * 0.3, 15.0)  # Cap at 15 points
        
        # Confidence bonus (reward high-confidence wins more)
        confidence_bonus = 0.0
        if confidence > self.confidence_threshold:
            confidence_bonus = (confidence - self.confidence_threshold) * 20.0
        
        # Market regime alignment bonus
        regime_bonus = 0.0
        if abs(market_regime) >= 1:  # Clear trend
            regime_bonus = abs(market_regime) * 3.0
        
        # Quick win bonus (encourage efficient trades)
        if pnl_pips > 10:  # Decent win
            quick_win_bonus = max(0, 10 - (pnl_pips / 5))  # Bonus for faster wins
        else:
            quick_win_bonus = 0
        
        total_reward = base_reward + pnl_reward + confidence_bonus + regime_bonus + quick_win_bonus
        
        return total_reward
    
    def _calculate_loss_penalty(self, pnl_pips: float, confidence: float, market_regime: int) -> float:
        """Calculate penalty for losing trades (heavy penalties)."""
        
        # Base loss penalty (much higher than standard)
        base_penalty = -25.0  # Increased from typical -5.0
        
        # PnL-based penalty (escalating with loss size)
        pnl_penalty = pnl_pips * 0.5  # More severe than wins
        
        # Confidence-based adjustment (less penalty for reasonable high-confidence bets)
        confidence_adjustment = 0.0
        if confidence > self.confidence_threshold:
            confidence_adjustment = (confidence - 0.5) * 8.0  # Reduce penalty slightly
        
        # Large loss penalty (catastrophic losses)
        large_loss_penalty = 0.0
        if pnl_pips < -30:  # Large loss
            large_loss_penalty = -20.0
        
        total_penalty = base_penalty + pnl_penalty + confidence_adjustment + large_loss_penalty
        
        return total_penalty
    
    def _calculate_action_reward(self, action: int, confidence: float, 
                               market_regime: int, position_info: Dict) -> float:
        """Calculate rewards/penalties for specific actions."""
        
        reward = 0.0
        
        if action == 0:  # Hold
            # Small penalty for inaction in high-confidence situations
            if confidence > 0.8 and abs(market_regime) >= 1:
                reward -= 2.0  # Missed opportunity
            
        elif action == 1:  # Close position
            if position_info and position_info.get('unrealized_pnl', 0) > 0:
                reward += 1.0  # Good profit-taking
            
        elif action in [2, 3, 4, 5]:  # Open position
            # Only reward high-confidence entries
            if confidence > self.confidence_threshold:
                reward += confidence * 5.0
                
                # Regime alignment bonus
                if action in [2, 3] and market_regime > 0:  # Long in uptrend
                    reward += abs(market_regime) * 2.0
                elif action in [4, 5] and market_regime < 0:  # Short in downtrend
                    reward += abs(market_regime) * 2.0
                else:
                    reward -= 5.0  # Counter-trend penalty
            else:
                reward -= 3.0  # Low-confidence entry penalty
        
        return reward
    
    def _calculate_streak_reward(self) -> float:
        """Calculate bonuses/penalties for winning/losing streaks."""
        
        reward = 0.0
        
        # Winning streak bonus (exponential growth)
        if self.consecutive_wins >= 3:
            streak_bonus = min(self.consecutive_wins ** 1.5, 25.0)  # Cap at 25
            reward += streak_bonus
        
        # Losing streak penalty (escalating)
        if self.consecutive_losses >= 2:
            streak_penalty = -(self.consecutive_losses ** 2) * 2.0
            reward += streak_penalty
        
        return reward
    
    def _calculate_risk_reward(self, account_info: Dict) -> float:
        """Calculate risk management rewards/penalties."""
        
        reward = 0.0
        
        # Drawdown penalties (progressive)
        drawdown = account_info.get('current_drawdown', 0)
        if drawdown > 0.05:  # 5% drawdown
            reward -= (drawdown - 0.05) * 100  # Escalating penalty
        
        # Overtrading penalty
        trades_today = account_info.get('trades_today', 0)
        max_trades = account_info.get('max_trades_per_day', 5)
        if trades_today > max_trades:
            reward -= (trades_today - max_trades) * 10.0
        
        # Equity growth bonus
        equity_growth = account_info.get('equity_growth_pct', 0)
        if equity_growth > 0:
            reward += min(equity_growth * 0.5, 5.0)  # Cap at 5 points
        
        return reward
    
    def _apply_confidence_scaling(self, reward: float, confidence: float) -> float:
        """Scale rewards based on model confidence."""
        
        # Scale positive rewards up for high confidence
        if reward > 0 and confidence > self.confidence_threshold:
            scaling_factor = 1.0 + (confidence - self.confidence_threshold) * 2.0
            reward *= scaling_factor
        
        # Scale negative rewards down for high confidence (reasonable bets)
        elif reward < 0 and confidence > self.confidence_threshold:
            scaling_factor = 1.0 - (confidence - self.confidence_threshold) * 0.5
            reward *= max(scaling_factor, 0.5)  # Don't reduce penalty too much
        
        return reward
    
    def get_recent_performance_metrics(self) -> Dict:
        """Get performance metrics from recent trades."""
        
        if not self.recent_trades:
            return {}
        
        wins = [t for t in self.recent_trades if t['pnl'] > 0]
        losses = [t for t in self.recent_trades if t['pnl'] <= 0]
        
        return {
            'recent_win_rate': len(wins) / len(self.recent_trades),
            'recent_avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'recent_avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'recent_avg_confidence': np.mean([t['confidence'] for t in self.recent_trades]),
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }
    
    def reset_streaks(self):
        """Reset streak counters (call at start of new episode)."""
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.recent_trades = []


class AdaptiveRewardSystem(HighWinrateRewardSystem):
    """Adaptive reward system that adjusts based on performance."""
    
    def __init__(self):
        super().__init__()
        self.performance_history = []
        self.adaptation_frequency = 100  # Adapt every 100 trades
        
    def adapt_rewards(self):
        """Adapt reward parameters based on recent performance."""
        
        if len(self.performance_history) < self.adaptation_frequency:
            return
        
        recent_performance = self.performance_history[-self.adaptation_frequency:]
        win_rate = sum(1 for p in recent_performance if p > 0) / len(recent_performance)
        
        # If win rate is too low, increase loss penalties
        if win_rate < 0.6:
            self.base_loss_penalty *= 1.1
            self.confidence_threshold = max(0.6, self.confidence_threshold - 0.05)
        
        # If win rate is good, can be slightly more aggressive
        elif win_rate > 0.75:
            self.base_loss_penalty *= 0.95
            self.confidence_threshold = min(0.85, self.confidence_threshold + 0.02)


# Example usage in trading environment
def integrate_high_winrate_rewards():
    """Example of how to integrate this reward system."""
    
    reward_system = HighWinrateRewardSystem()
    
    # In your trading environment's step function:
    def calculate_step_reward(self, pnl_pips, action, exit_reason, **kwargs):
        
        # Get current market state
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
            position_info = {
                'unrealized_pnl': self._calculate_unrealized_pnl(self.current_price),
                'bars_held': self.bars_in_position
            }
        
        # Calculate reward using high win-rate system
        reward = reward_system.calculate_reward(
            pnl_pips=pnl_pips,
            action=action,
            exit_reason=exit_reason,
            confidence=confidence,
            market_regime=market_state.get('market_regime', 0),
            position_info=position_info,
            account_info=account_info
        )
        
        return reward


if __name__ == "__main__":
    # Test the reward system
    reward_system = HighWinrateRewardSystem()
    
    # Test winning trade
    win_reward = reward_system.calculate_reward(
        pnl_pips=25.0,
        action=1,  # Close position
        exit_reason='take_profit',
        confidence=0.85,
        market_regime=1
    )
    print(f"Win reward: {win_reward:.2f}")
    
    # Test losing trade
    loss_reward = reward_system.calculate_reward(
        pnl_pips=-15.0,
        action=1,
        exit_reason='stop_loss',
        confidence=0.65,
        market_regime=1
    )
    print(f"Loss penalty: {loss_reward:.2f}")
    
    print(f"Recent performance: {reward_system.get_recent_performance_metrics()}")