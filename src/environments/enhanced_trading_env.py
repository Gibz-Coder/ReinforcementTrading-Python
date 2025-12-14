"""
Enhanced Forex Trading Environment with Advanced Features
========================================================

Key improvements over the original trading_env.py:
1. Integration with enhanced indicators and multi-timeframe analysis
2. Advanced risk management with Kelly Criterion and volatility-based sizing
3. Market regime-aware reward system
4. Spread and slippage simulation
5. Enhanced observation space with confidence scores
6. Better action space with dynamic SL/TP based on ATR
7. Comprehensive performance tracking
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from enhanced_indicators import load_and_preprocess_data_enhanced
from risk_management import AdaptiveRiskManager, Position


class EnhancedForexTradingEnv(gym.Env):
    """
    Enhanced Forex Trading Environment with advanced risk management and multi-timeframe analysis.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 1}
    
    def __init__(self, 
                 df: pd.DataFrame,
                 window_size: int = 30,
                 initial_balance: float = 10000.0,
                 max_trades_per_day: int = 5,
                 spread_pips: float = 1.5,
                 slippage_pips: float = 0.5,
                 render_mode: str = None):
        """
        Initialize enhanced trading environment.
        
        Args:
            df: Enhanced dataframe with multi-timeframe indicators
            window_size: Observation window size
            initial_balance: Starting account balance
            max_trades_per_day: Maximum trades per day
            spread_pips: Bid-ask spread in pips
            slippage_pips: Slippage in pips
            render_mode: Rendering mode
        """
        super().__init__()
        
        # Store enhanced dataframe
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        
        # Environment parameters
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_trades_per_day = max_trades_per_day
        self.spread_pips = spread_pips
        self.slippage_pips = slippage_pips
        self.pip_value = 0.0001
        self.render_mode = render_mode
        
        # Initialize adaptive risk manager
        self.risk_manager = AdaptiveRiskManager(initial_balance)
        
        # Dynamic action space based on ATR
        # Actions: [Hold, Close, Open_Long_Conservative, Open_Long_Aggressive, 
        #          Open_Short_Conservative, Open_Short_Aggressive]
        self.action_space = spaces.Discrete(6)
        
        # Enhanced observation space
        # Original features + position info + market regime + confidence scores
        self.num_features = self.df.shape[1]
        self.extra_features = 8  # position_type, unrealized_pnl, bars_held, confidence, regime, session, vol_regime, trend_strength
        total_features = self.num_features + self.extra_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, total_features),
            dtype=np.float32
        )
        
        # Initialize state
        self._init_state()
    
    def _init_state(self):
        """Initialize environment state."""
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.done = False
        self.truncated = False
        
        # Position tracking
        self.position = None
        self.bars_in_position = 0
        
        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trades_today = 0
        self.current_day = self.current_step // 24
        self.last_trade_step = 0
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        self.reward_history = []
        self.action_history = []
        
        # Risk metrics
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Market state tracking
        self.market_regime_history = []
        self.volatility_history = []
    
    def _get_market_state(self) -> Dict:
        """Get current market state information."""
        if self.current_step >= len(self.df):
            return {}
        
        current_data = self.df.iloc[self.current_step]
        
        return {
            'market_regime': current_data.get('market_regime', 0),
            'mtf_trend_alignment': current_data.get('mtf_trend_alignment', 0),
            'vol_regime': current_data.get('vol_regime', 0),
            'session_activity': current_data.get('session_activity', 0),
            'atr': current_data.get('atr', 0.001),
            'adx': current_data.get('adx', 0.25),
            'london_ny_overlap': current_data.get('london_ny_overlap', 0),
            'near_support': current_data.get('near_support', 0),
            'near_resistance': current_data.get('near_resistance', 0)
        }
    
    def _calculate_dynamic_sl_tp(self, direction: int, aggressive: bool = False) -> Tuple[float, float]:
        """
        Calculate dynamic SL/TP based on current ATR and market conditions.
        
        Args:
            direction: 1 for long, -1 for short
            aggressive: True for tighter SL/TP, False for conservative
            
        Returns:
            (sl_pips, tp_pips)
        """
        market_state = self._get_market_state()
        atr_normalized = market_state.get('atr', 0.001)
        
        # Convert normalized ATR back to pips (approximate)
        atr_pips = abs(atr_normalized) * 1000  # Rough conversion
        atr_pips = max(10, min(atr_pips, 100))  # Clamp between 10-100 pips
        
        if aggressive:
            # Tighter stops for scalping
            sl_pips = max(15, atr_pips * 1.5)
            tp_pips = max(20, atr_pips * 2.0)
        else:
            # Conservative stops for swing trading
            sl_pips = max(25, atr_pips * 2.5)
            tp_pips = max(35, atr_pips * 3.5)
        
        # Adjust based on market regime
        regime = market_state.get('market_regime', 0)
        if abs(regime) >= 2:  # Strong trend
            tp_pips *= 1.5  # Larger targets in strong trends
        elif regime == 0:  # Ranging market
            sl_pips *= 0.8  # Tighter stops in ranging markets
            tp_pips *= 0.8
        
        return sl_pips, tp_pips
    
    def _get_model_confidence(self) -> float:
        """
        Calculate model confidence based on market conditions.
        
        Higher confidence when:
        - Clear market regime
        - High session activity
        - Not near support/resistance
        - Stable volatility
        """
        market_state = self._get_market_state()
        
        confidence = 0.5  # Base confidence
        
        # Market regime clarity
        regime = abs(market_state.get('market_regime', 0))
        confidence += regime * 0.15  # +0.15 per regime strength level
        
        # Multi-timeframe alignment
        mtf_alignment = abs(market_state.get('mtf_trend_alignment', 0))
        confidence += mtf_alignment * 0.1
        
        # Session activity (higher activity = more confidence)
        session_activity = market_state.get('session_activity', 0)
        confidence += session_activity * 0.05
        
        # Penalty for being near support/resistance (less predictable)
        if market_state.get('near_support', 0) or market_state.get('near_resistance', 0):
            confidence -= 0.15
        
        # Volatility regime (stable vol = higher confidence)
        vol_regime = market_state.get('vol_regime', 0)
        if vol_regime == 0:  # Normal volatility
            confidence += 0.1
        elif abs(vol_regime) == 1:  # Extreme volatility
            confidence -= 0.1
        
        return max(0.2, min(confidence, 1.0))  # Clamp between 0.2 and 1.0
    
    def _get_observation(self):
        """Get enhanced observation with market state and confidence."""
        start = max(self.current_step - self.window_size, 0)
        obs_df = self.df.iloc[start:self.current_step]
        
        # Pad if needed
        if len(obs_df) < self.window_size:
            padding_rows = self.window_size - len(obs_df)
            first_part = np.tile(obs_df.iloc[0].values, (padding_rows, 1))
            obs_array = np.concatenate([first_part, obs_df.values], axis=0)
        else:
            obs_array = obs_df.values
        
        # Add enhanced position and market state
        extra_state = np.zeros((self.window_size, self.extra_features))
        
        # Get current market state
        market_state = self._get_market_state()
        confidence = self._get_model_confidence()
        
        # Position information
        if self.position is not None:
            current_price = self.df.loc[self.current_step - 1, "Close"] if self.current_step > 0 else 0
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            
            extra_state[:, 0] = 1 if self.position['direction'] == 1 else -1  # Position type
            extra_state[:, 1] = unrealized_pnl / 1000  # Normalized unrealized PnL
            extra_state[:, 2] = min(self.bars_in_position / 100, 1.0)  # Normalized bars held
        
        # Market state information (same for all rows in window)
        extra_state[:, 3] = confidence  # Model confidence
        extra_state[:, 4] = market_state.get('market_regime', 0) / 2.0  # Normalized regime
        extra_state[:, 5] = market_state.get('session_activity', 0) / 3.0  # Normalized session activity
        extra_state[:, 6] = market_state.get('vol_regime', 0)  # Volatility regime
        extra_state[:, 7] = market_state.get('adx', 0.25)  # Trend strength
        
        # Combine original features with extra state
        obs_array = np.concatenate([obs_array, extra_state], axis=1)
        
        return obs_array.astype(np.float32)
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL including spread and slippage."""
        if self.position is None:
            return 0.0
        
        entry_price = self.position['entry_price']
        lot_size = self.position['lot_size']
        
        # Apply spread (always against the trader)
        if self.position['direction'] == 1:  # Long position
            # Use bid price for exit
            exit_price = current_price - self.spread_pips * self.pip_value
            pnl_pips = (exit_price - entry_price) / self.pip_value
        else:  # Short position
            # Use ask price for exit
            exit_price = current_price + self.spread_pips * self.pip_value
            pnl_pips = (entry_price - exit_price) / self.pip_value
        
        return pnl_pips * lot_size
    
    def _open_position(self, direction: int, aggressive: bool = False):
        """Open position with enhanced risk management."""
        if self.position is not None:
            return  # Already have a position
        
        current_price = self.df.loc[self.current_step, "Close"]
        market_state = self._get_market_state()
        confidence = self._get_model_confidence()
        
        # Calculate dynamic SL/TP
        sl_pips, tp_pips = self._calculate_dynamic_sl_tp(direction, aggressive)
        
        # Calculate position size using adaptive risk management
        lot_size = self.risk_manager.calculate_position_size(
            account_balance=self.equity,
            current_atr=abs(market_state.get('atr', 0.001)),
            sl_distance_pips=sl_pips,
            market_regime=market_state.get('market_regime', 0),
            confidence_score=confidence
        )
        
        # Apply spread and slippage to entry price
        if direction == 1:  # Long
            entry_price = current_price + (self.spread_pips + self.slippage_pips) * self.pip_value
            sl_price = entry_price - sl_pips * self.pip_value
            tp_price = entry_price + tp_pips * self.pip_value
        else:  # Short
            entry_price = current_price - (self.spread_pips + self.slippage_pips) * self.pip_value
            sl_price = entry_price + sl_pips * self.pip_value
            tp_price = entry_price - tp_pips * self.pip_value
        
        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'lot_size': lot_size,
            'entry_step': self.current_step,
            'confidence': confidence,
            'market_regime': market_state.get('market_regime', 0)
        }
        
        self.bars_in_position = 0
        self.trades_today += 1
        self.total_trades += 1
        self.last_trade_step = self.current_step
    
    def _close_position(self, exit_price: float, exit_reason: str = 'manual') -> float:
        """Close position with enhanced tracking."""
        if self.position is None:
            return 0.0
        
        # Apply spread and slippage to exit price
        if self.position['direction'] == 1:  # Long position
            # Use bid price for exit
            actual_exit_price = exit_price - (self.spread_pips + self.slippage_pips) * self.pip_value
        else:  # Short position
            # Use ask price for exit
            actual_exit_price = exit_price + (self.spread_pips + self.slippage_pips) * self.pip_value
        
        # Calculate PnL
        entry_price = self.position['entry_price']
        lot_size = self.position['lot_size']
        
        if self.position['direction'] == 1:  # Long
            pnl_pips = (actual_exit_price - entry_price) / self.pip_value
        else:  # Short
            pnl_pips = (entry_price - actual_exit_price) / self.pip_value
        
        pnl_dollars = pnl_pips * lot_size * 10  # $10 per pip per standard lot
        
        # Update statistics
        if pnl_pips > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Record trade
        trade_record = {
            'entry_step': self.position['entry_step'],
            'exit_step': self.current_step,
            'direction': 'LONG' if self.position['direction'] == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': actual_exit_price,
            'sl_pips': self.position['sl_pips'],
            'tp_pips': self.position['tp_pips'],
            'lot_size': lot_size,
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'exit_reason': exit_reason,
            'bars_held': self.bars_in_position,
            'confidence': self.position['confidence'],
            'market_regime': self.position['market_regime']
        }
        
        self.trade_history.append(trade_record)
        
        # Update equity
        self.equity += pnl_dollars
        
        # Update risk manager
        risk_amount = self.position['sl_pips'] * lot_size * 10
        self.risk_manager.update_trade_result(pnl_dollars, risk_amount, 
                                            abs(self._get_market_state().get('atr', 0.001)))
        self.risk_manager.update_equity(self.equity)
        
        # Clear position
        self.position = None
        self.bars_in_position = 0
        
        return pnl_pips
    
    def _check_sl_tp(self) -> Tuple[bool, float, str]:
        """Check if SL or TP is hit."""
        if self.position is None or self.current_step >= self.n_steps:
            return False, 0.0, ""
        
        current_high = self.df.loc[self.current_step, "High"]
        current_low = self.df.loc[self.current_step, "Low"]
        current_close = self.df.loc[self.current_step, "Close"]
        
        sl_price = self.position['sl_price']
        tp_price = self.position['tp_price']
        
        if self.position['direction'] == 1:  # Long
            if current_low <= sl_price:
                return True, sl_price, 'stop_loss'
            elif current_high >= tp_price:
                return True, tp_price, 'take_profit'
        else:  # Short
            if current_high >= sl_price:
                return True, sl_price, 'stop_loss'
            elif current_low <= tp_price:
                return True, tp_price, 'take_profit'
        
        return False, current_close, ""
    
    def _calculate_enhanced_reward(self, pnl_pips: float, action: int, exit_reason: str = None) -> float:
        """
        Enhanced reward function with market regime awareness and risk-adjusted returns.
        """
        reward = 0.0
        market_state = self._get_market_state()
        confidence = self._get_model_confidence()
        
        # Base reward from PnL (risk-adjusted)
        if exit_reason is not None:  # Trade closed
            if exit_reason == 'take_profit':
                # Reward successful trades more in difficult conditions
                base_reward = 10.0 + pnl_pips * 0.2
                
                # Bonus for high-confidence trades
                confidence_bonus = confidence * 5.0
                
                # Bonus for regime-aligned trades
                if self.position and 'market_regime' in self.position:
                    trade_regime = self.position['market_regime']
                    trade_direction = self.position['direction']
                    
                    if (trade_direction == 1 and trade_regime > 0) or (trade_direction == -1 and trade_regime < 0):
                        base_reward += 3.0  # Regime alignment bonus
                
                reward += base_reward + confidence_bonus
                
            elif exit_reason == 'stop_loss':
                # Penalize losses, but less for high-confidence trades in difficult conditions
                base_penalty = -5.0 - abs(pnl_pips) * 0.1
                
                # Reduce penalty for high-confidence trades (they were reasonable bets)
                confidence_adjustment = confidence * 2.0
                
                reward += base_penalty + confidence_adjustment
        
        # Action-specific rewards
        if action == 0:  # Hold
            # Small penalty for inaction when opportunities exist
            if confidence > 0.8 and abs(market_state.get('market_regime', 0)) >= 1:
                reward -= 0.5  # Missed opportunity penalty
            
            # Penalty for holding too long without trading
            bars_since_trade = self.current_step - self.last_trade_step
            if bars_since_trade > 48:  # More than 2 days
                reward -= min(bars_since_trade / 100, 2.0)
        
        elif action == 1:  # Close position
            if self.position is not None:
                # Small reward for active management
                reward += 0.5
        
        elif action in [2, 3, 4, 5]:  # Open position
            if self.position is None and self.trades_today < self.max_trades_per_day:
                # Reward for taking action in high-confidence situations
                if confidence > 0.7:
                    reward += confidence * 2.0
                
                # Reward for regime-aligned entries
                regime = market_state.get('market_regime', 0)
                if action in [2, 3] and regime > 0:  # Long in uptrend
                    reward += abs(regime) * 1.0
                elif action in [4, 5] and regime < 0:  # Short in downtrend
                    reward += abs(regime) * 1.0
                elif regime == 0:  # Ranging market
                    reward += 0.5  # Small reward for any action in ranging market
                else:
                    reward -= 1.0  # Penalty for counter-trend trades
        
        # Risk management penalties
        if self.trades_today > self.max_trades_per_day:
            reward -= 5.0  # Overtrading penalty
        
        if self.current_drawdown > 0.15:  # More than 15% drawdown
            reward -= (self.current_drawdown - 0.15) * 20  # Escalating penalty
        
        # Session-based adjustments
        session_activity = market_state.get('session_activity', 0)
        if session_activity >= 2:  # High activity sessions
            reward *= 1.1  # 10% bonus during active sessions
        
        return reward
    
    def step(self, action: int):
        """Execute one step with enhanced logic."""
        
        # Check for new day
        current_bar_day = self.current_step // 24
        if current_bar_day != self.current_day:
            self.current_day = current_bar_day
            self.trades_today = 0
        
        reward = 0.0
        pnl_pips = 0.0
        exit_reason = None
        
        # Check SL/TP first
        if self.position is not None:
            hit, exit_price, reason = self._check_sl_tp()
            if hit:
                pnl_pips = self._close_position(exit_price, reason)
                exit_reason = reason
        
        # Process action
        if action == 0:  # Hold
            if self.position is not None:
                self.bars_in_position += 1
        
        elif action == 1:  # Close position
            if self.position is not None:
                current_close = self.df.loc[self.current_step, "Close"]
                pnl_pips = self._close_position(current_close, 'manual')
                exit_reason = 'manual'
        
        elif action == 2:  # Open Long Conservative
            self._open_position(direction=1, aggressive=False)
        
        elif action == 3:  # Open Long Aggressive
            self._open_position(direction=1, aggressive=True)
        
        elif action == 4:  # Open Short Conservative
            self._open_position(direction=-1, aggressive=False)
        
        elif action == 5:  # Open Short Aggressive
            self._open_position(direction=-1, aggressive=True)
        
        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(pnl_pips, action, exit_reason)
        
        # Update tracking
        self.reward_history.append(reward)
        self.action_history.append(action)
        
        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Log equity
        self.equity_curve.append(self.equity)
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        if self.current_step >= self.n_steps - 1:
            terminated = True
            if self.position is not None:
                final_close = self.df.loc[self.n_steps - 1, "Close"]
                self._close_position(final_close, 'end_of_data')
        
        # Stop if account blown (less than 20% of initial)
        if self.equity < self.initial_balance * 0.2:
            terminated = True
            reward -= 100  # Large penalty for account blow-up
        
        # Get next observation
        obs = self._get_observation()
        
        # Enhanced info dictionary
        info = {
            'equity': self.equity,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'position_open': self.position is not None,
            'market_regime': self._get_market_state().get('market_regime', 0),
            'confidence': self._get_model_confidence(),
            'session_activity': self._get_market_state().get('session_activity', 0),
            'risk_summary': self.risk_manager.get_risk_summary() if self.total_trades > 5 else {}
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_observation()
        info = {'equity': self.equity}
        return obs, info
    
    def render(self):
        """Render current state with enhanced information."""
        if self.render_mode == "human":
            market_state = self._get_market_state()
            confidence = self._get_model_confidence()
            
            pos_str = "None"
            if self.position is not None:
                direction = "LONG" if self.position['direction'] == 1 else "SHORT"
                pos_str = f"{direction}@{self.position['entry_price']:.5f} (SL:{self.position['sl_pips']:.0f}p TP:{self.position['tp_pips']:.0f}p)"
            
            print(f"Step: {self.current_step:5d} | "
                  f"Equity: ${self.equity:,.2f} | "
                  f"Position: {pos_str} | "
                  f"Regime: {market_state.get('market_regime', 0):+d} | "
                  f"Confidence: {confidence:.2f} | "
                  f"Trades: {self.total_trades} | "
                  f"WR: {self.winning_trades/max(1,self.total_trades)*100:.1f}% | "
                  f"DD: {self.max_drawdown*100:.1f}%")
    
    def get_enhanced_statistics(self) -> Dict:
        """Get comprehensive trading statistics."""
        base_stats = self._get_base_statistics()
        risk_summary = self.risk_manager.get_risk_summary()
        
        # Enhanced metrics
        enhanced_stats = {
            **base_stats,
            'risk_management': risk_summary,
            'market_regime_distribution': self._get_regime_distribution(),
            'session_performance': self._get_session_performance(),
            'confidence_analysis': self._get_confidence_analysis(),
            'action_distribution': self._get_action_distribution()
        }
        
        return enhanced_stats
    
    def _get_base_statistics(self) -> Dict:
        """Get base trading statistics."""
        if not self.trade_history:
            return {}
        
        pnls = [t['pnl_pips'] for t in self.trade_history]
        dollars = [t['pnl_dollars'] for t in self.trade_history]
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_pnl_pips': sum(pnls),
            'total_pnl_dollars': sum(dollars),
            'avg_pnl_pips': np.mean(pnls),
            'avg_win_pips': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss_pips': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            'max_drawdown': self.max_drawdown,
            'final_equity': self.equity,
            'return_pct': (self.equity - self.initial_balance) / self.initial_balance * 100,
            'profit_factor': abs(sum(p for p in pnls if p > 0) / min(-1, sum(p for p in pnls if p < 0))) if any(p < 0 for p in pnls) else float('inf'),
            'sharpe_ratio': np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252) if pnls else 0
        }
    
    def _get_regime_distribution(self) -> Dict:
        """Analyze performance by market regime."""
        if not self.trade_history:
            return {}
        
        regime_performance = {}
        for trade in self.trade_history:
            regime = trade.get('market_regime', 0)
            if regime not in regime_performance:
                regime_performance[regime] = {'trades': 0, 'wins': 0, 'total_pnl': 0}
            
            regime_performance[regime]['trades'] += 1
            if trade['pnl_pips'] > 0:
                regime_performance[regime]['wins'] += 1
            regime_performance[regime]['total_pnl'] += trade['pnl_pips']
        
        # Calculate win rates
        for regime in regime_performance:
            perf = regime_performance[regime]
            perf['win_rate'] = perf['wins'] / perf['trades'] if perf['trades'] > 0 else 0
            perf['avg_pnl'] = perf['total_pnl'] / perf['trades'] if perf['trades'] > 0 else 0
        
        return regime_performance
    
    def _get_session_performance(self) -> Dict:
        """Analyze performance by trading session."""
        # This would require session information in trade records
        # Placeholder for now
        return {}
    
    def _get_confidence_analysis(self) -> Dict:
        """Analyze performance by confidence levels."""
        if not self.trade_history:
            return {}
        
        high_conf_trades = [t for t in self.trade_history if t.get('confidence', 0.5) > 0.7]
        low_conf_trades = [t for t in self.trade_history if t.get('confidence', 0.5) <= 0.7]
        
        def analyze_trades(trades):
            if not trades:
                return {'count': 0, 'win_rate': 0, 'avg_pnl': 0}
            
            wins = sum(1 for t in trades if t['pnl_pips'] > 0)
            avg_pnl = np.mean([t['pnl_pips'] for t in trades])
            
            return {
                'count': len(trades),
                'win_rate': wins / len(trades),
                'avg_pnl': avg_pnl
            }
        
        return {
            'high_confidence': analyze_trades(high_conf_trades),
            'low_confidence': analyze_trades(low_conf_trades)
        }
    
    def _get_action_distribution(self) -> Dict:
        """Get distribution of actions taken."""
        if not self.action_history:
            return {}
        
        action_names = ['Hold', 'Close', 'Long_Conservative', 'Long_Aggressive', 'Short_Conservative', 'Short_Aggressive']
        action_counts = {}
        
        for i, name in enumerate(action_names):
            action_counts[name] = self.action_history.count(i)
        
        total_actions = len(self.action_history)
        action_percentages = {name: count/total_actions*100 for name, count in action_counts.items()}
        
        return {
            'counts': action_counts,
            'percentages': action_percentages
        }


# Test the enhanced environment
if __name__ == "__main__":
    from enhanced_indicators import load_and_preprocess_data_enhanced
    
    # Test with sample data
    try:
        print("Testing Enhanced Trading Environment...")
        
        # Load enhanced data
        df = load_and_preprocess_data_enhanced(
            "data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv",
            normalize=True
        )
        
        # Create environment
        env = EnhancedForexTradingEnv(df, window_size=30)
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful. Observation shape: {obs.shape}")
        
        # Test a few steps
        for i in range(10):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i % 3 == 0:
                env.render()
            
            if terminated or truncated:
                break
        
        # Get statistics
        stats = env.get_enhanced_statistics()
        print(f"\n✓ Enhanced environment test completed!")
        print(f"Final equity: ${env.equity:.2f}")
        print(f"Total trades: {env.total_trades}")
        
    except FileNotFoundError:
        print("Test data file not found. Please ensure the data file exists.")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()