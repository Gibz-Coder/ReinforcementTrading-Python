"""
High Win Rate Reward System
===========================

Advanced reward shaping specifically designed to maximize win rate while maintaining profitability.

Key principles:
1. Heavy penalties for losses to discourage bad trades
2. Progressive rewards for winning streaks
3. Bonuses for high-confidence signal alignment
4. Risk-adjusted rewards based on market conditions
5. Time-based rewards for holding winning positions
"""

import numpy as np
from typing import Dict, Optional


class HighWinRateRewardSystem:
    """
    Reward system optimized for achieving high win rates.
    """
    
    def __init__(self):
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.recent_performance = []  # Track last 20 trades
        self.confidence_threshold = 0.7
        
    def calculate_reward(self, 
                        pnl_pips: float,
                        action_type: str,
                        market_state: Dict,
                        position_info: Optional[Dict] = None,
                        exit_reason: Optional[str] = None) -> float:
        """
        Calculate reward with heavy emphasis on win rate optimization.
        
        Args:
            pnl_pips: Profit/loss in pips
            action_type: Type of action taken
            market_state: Current market conditions
            position_info: Information about current position
            exit_reason: Reason for position exit
            
        Returns:
            Calculated reward value
        """
        
        reward = 0.0
        
        # Get market state information
        confidence = market_state.get('confidence', 0.5)
        regime = market_state.get('market_regime', 0)
        signal_strength = market_state.get('signal_strength', 0)
        volatility = market_state.get('volatility', 0.5)
        
        # ============ TRADE OUTCOME REWARDS (Primary Focus) ============
        if exit_reason is not None:  # Trade was closed
            
            if pnl_pips > 0:  # WINNING TRADE
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.recent_performance.append(1)
                
                # Base reward for any win (encourage winning)
                base_win_reward = 50.0
                
                # Scale reward by profit size (but cap to prevent over-optimization for large wins)
                profit_reward = min(pnl_pips * 0.5, 100.0)
                
                # MASSIVE bonus for high-confidence wins
                if confidence > self.confidence_threshold:
                    confidence_bonus = 100.0 * confidence
                else:
                    confidence_bonus = 20.0 * confidence
                
                # Signal strength bonus
                signal_bonus = signal_strength * 25.0
                
                # Winning streak bonus (exponential growth)
                if self.consecutive_wins >= 2:
                    streak_bonus = min(self.consecutive_wins ** 2 * 10.0, 200.0)
                else:
                    streak_bonus = 0.0
                
                # Market regime alignment bonus
                regime_bonus = 0.0
                if position_info:
                    trade_direction = position_info.get('direction', 0)
                    if (trade_direction == 1 and regime > 0) or (trade_direction == -1 and regime < 0):
                        regime_bonus = 50.0  # Big bonus for trend-aligned wins
                
                # Time-based bonus for holding winning positions
                time_bonus = 0.0
                if position_info:
                    bars_held = position_info.get('bars_held', 0)
                    if bars_held >= 4:  # Held for at least 4 hours
                        time_bonus = min(bars_held * 2.0, 30.0)
                
                # Recent performance bonus (if win rate is high)
                recent_win_rate = np.mean(self.recent_performance[-10:]) if self.recent_performance else 0
                if recent_win_rate > 0.6:
                    performance_bonus = 50.0 * recent_win_rate
                else:
                    performance_bonus = 0.0
                
                reward = (base_win_reward + profit_reward + confidence_bonus + 
                         signal_bonus + streak_bonus + regime_bonus + 
                         time_bonus + performance_bonus)
                
            else:  # LOSING TRADE
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.recent_performance.append(0)
                
                # HEAVY penalty for losses (discourage losing trades)
                base_loss_penalty = -100.0
                
                # Scale penalty by loss size
                loss_penalty = max(pnl_pips * 1.0, -200.0)  # Cap maximum penalty
                
                # EXTRA penalty for low-confidence losses
                if confidence < self.confidence_threshold:
                    confidence_penalty = -100.0 * (1 - confidence)
                else:
                    confidence_penalty = -50.0 * (1 - confidence)
                
                # Losing streak penalty (exponential growth)
                if self.consecutive_losses >= 2:
                    streak_penalty = -min(self.consecutive_losses ** 2 * 15.0, 300.0)
                else:
                    streak_penalty = 0.0
                
                # Counter-trend penalty
                regime_penalty = 0.0
                if position_info:
                    trade_direction = position_info.get('direction', 0)
                    if (trade_direction == 1 and regime < 0) or (trade_direction == -1 and regime > 0):
                        regime_penalty = -75.0  # Heavy penalty for counter-trend losses
                
                # Recent performance penalty (if win rate is dropping)
                recent_win_rate = np.mean(self.recent_performance[-10:]) if self.recent_performance else 0
                if recent_win_rate < 0.4:
                    performance_penalty = -100.0 * (0.5 - recent_win_rate)
                else:
                    performance_penalty = 0.0
                
                reward = (base_loss_penalty + loss_penalty + confidence_penalty + 
                         streak_penalty + regime_penalty + performance_penalty)
            
            self.total_trades += 1
            
            # Keep recent performance history manageable
            if len(self.recent_performance) > 20:
                self.recent_performance = self.recent_performance[-20:]
        
        # ============ ACTION-BASED REWARDS ============
        elif action_type == 'open':  # Opening a new position
            
            # Reward high-confidence entries
            if confidence > self.confidence_threshold:
                entry_reward = 30.0 * confidence
            else:
                entry_reward = 5.0 * confidence
            
            # Signal strength reward
            signal_reward = signal_strength * 10.0
            
            # Market regime alignment reward
            regime_reward = 0.0
            if position_info:
                trade_direction = position_info.get('direction', 0)
                if (trade_direction == 1 and regime > 0) or (trade_direction == -1 and regime < 0):
                    regime_reward = 25.0
                elif (trade_direction == 1 and regime < 0) or (trade_direction == -1 and regime > 0):
                    regime_reward = -25.0  # Penalty for counter-trend entries
            
            # Volatility-based reward (prefer entries during optimal volatility)
            if 0.3 < volatility < 0.7:  # Optimal volatility range
                vol_reward = 15.0
            else:
                vol_reward = -10.0
            
            reward = entry_reward + signal_reward + regime_reward + vol_reward
            
        elif action_type == 'hold':  # Holding position or staying out
            
            # Small penalty for inaction when high-confidence signals are present
            if confidence > 0.8 and signal_strength >= 3:
                reward = -10.0  # Missed opportunity penalty
            else:
                reward = 0.0  # Neutral for normal holding
                
        elif action_type == 'close':  # Manually closing position
            
            # Small reward for active management
            reward = 5.0
            
            # Bonus if closing at profit
            if position_info and position_info.get('unrealized_pnl', 0) > 0:
                reward += 15.0
        
        # ============ RISK MANAGEMENT PENALTIES ============
        
        # Penalty for overtrading
        if self.total_trades > 0:
            recent_trade_frequency = len([x for x in self.recent_performance[-5:] if x is not None])
            if recent_trade_frequency > 3:  # More than 3 trades in last 5 periods
                reward -= 20.0
        
        # Penalty for poor recent performance
        if len(self.recent_performance) >= 5:
            recent_win_rate = np.mean(self.recent_performance[-5:])
            if recent_win_rate < 0.3:  # Less than 30% win rate recently
                reward -= 50.0 * (0.3 - recent_win_rate)
        
        # ============ WIN RATE OPTIMIZATION BONUS ============
        
        # Overall win rate bonus/penalty
        if self.total_trades >= 10:
            current_win_rate = self.winning_trades / self.total_trades
            
            if current_win_rate >= 0.6:  # Excellent win rate
                reward += 50.0 * (current_win_rate - 0.5)
            elif current_win_rate >= 0.5:  # Good win rate
                reward += 25.0 * (current_win_rate - 0.4)
            elif current_win_rate < 0.4:  # Poor win rate
                reward -= 100.0 * (0.4 - current_win_rate)
        
        return reward
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        
        win_rate = self.winning_trades / max(self.total_trades, 1)
        recent_win_rate = np.mean(self.recent_performance[-10:]) if self.recent_performance else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'recent_win_rate': recent_win_rate,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'performance_trend': 'improving' if recent_win_rate > win_rate else 'declining'
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.recent_performance = []


class AdaptiveRewardScaling:
    """
    Adaptive reward scaling based on market conditions and performance.
    """
    
    def __init__(self):
        self.performance_history = []
        self.market_difficulty = 0.5  # 0 = easy, 1 = very difficult
        
    def scale_reward(self, base_reward: float, market_state: Dict, performance_stats: Dict) -> float:
        """
        Scale reward based on market difficulty and recent performance.
        """
        
        # Market difficulty adjustment
        volatility = market_state.get('volatility', 0.5)
        regime_clarity = abs(market_state.get('market_regime', 0)) / 2.0
        
        # High volatility + unclear regime = difficult market
        self.market_difficulty = volatility * (1 - regime_clarity)
        
        # Scale rewards based on difficulty
        if self.market_difficulty > 0.7:  # Very difficult market
            difficulty_multiplier = 1.5  # Higher rewards for success in difficult conditions
        elif self.market_difficulty < 0.3:  # Easy market
            difficulty_multiplier = 0.8  # Lower rewards in easy conditions
        else:
            difficulty_multiplier = 1.0
        
        # Performance-based scaling
        recent_win_rate = performance_stats.get('recent_win_rate', 0.5)
        
        if recent_win_rate > 0.6:  # Good performance
            performance_multiplier = 1.2
        elif recent_win_rate < 0.4:  # Poor performance
            performance_multiplier = 0.8
        else:
            performance_multiplier = 1.0
        
        # Apply scaling
        scaled_reward = base_reward * difficulty_multiplier * performance_multiplier
        
        return scaled_reward


# Example usage and testing
if __name__ == "__main__":
    
    # Test the high win rate reward system
    reward_system = HighWinRateRewardSystem()
    adaptive_scaling = AdaptiveRewardScaling()
    
    print("Testing High Win Rate Reward System")
    print("=" * 50)
    
    # Simulate some trades
    test_scenarios = [
        # (pnl_pips, confidence, signal_strength, regime, direction, expected_outcome)
        (50, 0.8, 4, 1, 1, "High-confidence trend-aligned win"),
        (-30, 0.3, 1, -1, 1, "Low-confidence counter-trend loss"),
        (25, 0.9, 5, 2, 1, "Very high-confidence strong trend win"),
        (-15, 0.7, 3, 0, -1, "Medium-confidence ranging market loss"),
        (40, 0.6, 2, 1, 1, "Medium-confidence trend win"),
    ]
    
    for i, (pnl, conf, signal, regime, direction, description) in enumerate(test_scenarios):
        
        market_state = {
            'confidence': conf,
            'signal_strength': signal,
            'market_regime': regime,
            'volatility': 0.5
        }
        
        position_info = {
            'direction': direction,
            'bars_held': 6,
            'unrealized_pnl': pnl
        }
        
        # Calculate reward
        reward = reward_system.calculate_reward(
            pnl_pips=pnl,
            action_type='close',
            market_state=market_state,
            position_info=position_info,
            exit_reason='take_profit' if pnl > 0 else 'stop_loss'
        )
        
        # Apply adaptive scaling
        stats = reward_system.get_performance_stats()
        scaled_reward = adaptive_scaling.scale_reward(reward, market_state, stats)
        
        print(f"\nScenario {i+1}: {description}")
        print(f"  PnL: {pnl} pips, Confidence: {conf}, Signal: {signal}")
        print(f"  Base Reward: {reward:.1f}")
        print(f"  Scaled Reward: {scaled_reward:.1f}")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
    
    # Final performance summary
    final_stats = reward_system.get_performance_stats()
    print(f"\nFinal Performance:")
    print(f"  Total Trades: {final_stats['total_trades']}")
    print(f"  Win Rate: {final_stats['win_rate']*100:.1f}%")
    print(f"  Consecutive Wins: {final_stats['consecutive_wins']}")
    print(f"  Performance Trend: {final_stats['performance_trend']}")