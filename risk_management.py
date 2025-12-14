"""
Advanced Risk Management System for Forex Trading
=================================================

Features:
1. Kelly Criterion for optimal position sizing
2. Dynamic position sizing based on volatility (ATR)
3. Portfolio risk management across multiple positions
4. VaR (Value at Risk) calculations
5. Drawdown-based position adjustment
6. Correlation-based risk assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    lot_size: float
    sl_price: float
    tp_price: float
    entry_time: pd.Timestamp
    unrealized_pnl: float = 0.0
    bars_held: int = 0


@dataclass
class RiskMetrics:
    """Risk metrics for portfolio assessment."""
    portfolio_var_95: float
    portfolio_var_99: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_risk_exposure: float


class KellyCriterion:
    """
    Kelly Criterion implementation for optimal position sizing.
    
    The Kelly formula: f* = (bp - q) / b
    Where:
    - f* = fraction of capital to wager
    - b = odds received on the wager (reward/risk ratio)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """
    
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period
        self.trade_history = []
    
    def add_trade(self, pnl: float, risk_amount: float):
        """Add a completed trade to history."""
        self.trade_history.append({
            'pnl': pnl,
            'risk': risk_amount,
            'return': pnl / risk_amount if risk_amount > 0 else 0
        })
        
        # Keep only recent trades
        if len(self.trade_history) > self.lookback_period:
            self.trade_history = self.trade_history[-self.lookback_period:]
    
    def calculate_kelly_fraction(self) -> float:
        """
        Calculate optimal Kelly fraction based on trade history.
        
        Returns:
            Kelly fraction (0.0 to 1.0), capped at 0.25 for safety
        """
        if len(self.trade_history) < 10:  # Need minimum sample size
            return 0.02  # Conservative default
        
        returns = [trade['return'] for trade in self.trade_history]
        
        # Calculate win rate
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        if not wins or not losses:
            return 0.02  # Conservative if no wins or no losses
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        # Kelly formula: f* = (bp - q) / b
        # Where b = avg_win/avg_loss (reward/risk ratio)
        if avg_loss == 0:
            return 0.02
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly fraction for safety (never risk more than 25%)
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))
        
        # Additional safety: reduce if recent performance is poor
        recent_returns = returns[-20:] if len(returns) >= 20 else returns
        if np.mean(recent_returns) < 0:
            kelly_fraction *= 0.5  # Halve position size if recent performance is negative
        
        return kelly_fraction
    
    def get_statistics(self) -> Dict:
        """Get Kelly criterion statistics."""
        if len(self.trade_history) < 5:
            return {}
        
        returns = [trade['return'] for trade in self.trade_history]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': len(wins) / len(returns) if returns else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'kelly_fraction': self.calculate_kelly_fraction(),
            'expected_return': np.mean(returns) if returns else 0
        }


class VolatilityBasedSizing:
    """
    Dynamic position sizing based on market volatility (ATR).
    
    Adjusts position size inversely to volatility:
    - High volatility = smaller positions
    - Low volatility = larger positions
    """
    
    def __init__(self, base_risk_pct: float = 0.02, vol_lookback: int = 20):
        self.base_risk_pct = base_risk_pct
        self.vol_lookback = vol_lookback
        self.vol_history = []
    
    def update_volatility(self, atr_value: float):
        """Update volatility history with new ATR value."""
        self.vol_history.append(atr_value)
        if len(self.vol_history) > self.vol_lookback * 5:  # Keep 5x lookback for percentile calc
            self.vol_history = self.vol_history[-self.vol_lookback * 5:]
    
    def calculate_vol_adjusted_risk(self, current_atr: float) -> float:
        """
        Calculate volatility-adjusted risk percentage.
        
        Args:
            current_atr: Current ATR value
            
        Returns:
            Adjusted risk percentage
        """
        if len(self.vol_history) < self.vol_lookback:
            return self.base_risk_pct
        
        # Calculate volatility percentile
        vol_percentile = stats.percentileofscore(self.vol_history, current_atr) / 100
        
        # Adjust risk inversely to volatility
        # High volatility (90th percentile) = 50% of base risk
        # Low volatility (10th percentile) = 150% of base risk
        vol_multiplier = 2.0 - vol_percentile  # Range: 1.0 to 2.0
        vol_multiplier = max(0.5, min(vol_multiplier, 1.5))  # Cap between 0.5x and 1.5x
        
        adjusted_risk = self.base_risk_pct * vol_multiplier
        
        return adjusted_risk
    
    def get_position_size(self, account_balance: float, current_atr: float, 
                         sl_distance_pips: float, pip_value: float = 10.0) -> float:
        """
        Calculate position size based on volatility-adjusted risk.
        
        Args:
            account_balance: Current account balance
            current_atr: Current ATR value
            sl_distance_pips: Stop loss distance in pips
            pip_value: Value per pip for standard lot
            
        Returns:
            Position size in lots
        """
        adjusted_risk_pct = self.calculate_vol_adjusted_risk(current_atr)
        risk_amount = account_balance * adjusted_risk_pct
        
        # Calculate lot size
        lot_size = risk_amount / (sl_distance_pips * pip_value)
        
        # Apply minimum and maximum limits
        lot_size = max(0.01, min(lot_size, 10.0))
        
        return lot_size


class PortfolioRiskManager:
    """
    Portfolio-level risk management across multiple positions.
    
    Features:
    - Maximum portfolio exposure limits
    - Correlation-based position sizing
    - VaR calculations
    - Drawdown monitoring
    """
    
    def __init__(self, max_portfolio_risk: float = 0.10, max_positions: int = 5):
        self.max_portfolio_risk = max_portfolio_risk  # Max 10% portfolio risk
        self.max_positions = max_positions
        self.positions: List[Position] = []
        self.equity_history = []
        self.returns_history = []
        
    def add_position(self, position: Position) -> bool:
        """
        Add a new position if it passes risk checks.
        
        Returns:
            True if position was added, False if rejected
        """
        # Check maximum positions limit
        if len(self.positions) >= self.max_positions:
            return False
        
        # Check if adding this position would exceed portfolio risk
        projected_risk = self._calculate_portfolio_risk_with_new_position(position)
        if projected_risk > self.max_portfolio_risk:
            return False
        
        self.positions.append(position)
        return True
    
    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove and return a position by symbol."""
        for i, pos in enumerate(self.positions):
            if pos.symbol == symbol:
                return self.positions.pop(i)
        return None
    
    def update_equity(self, new_equity: float):
        """Update equity history for risk calculations."""
        self.equity_history.append(new_equity)
        
        if len(self.equity_history) > 1:
            returns = (new_equity - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(returns)
        
        # Keep only recent history
        if len(self.equity_history) > 1000:
            self.equity_history = self.equity_history[-1000:]
            self.returns_history = self.returns_history[-999:]
    
    def _calculate_portfolio_risk_with_new_position(self, new_position: Position) -> float:
        """Calculate total portfolio risk including a potential new position."""
        total_risk = 0.0
        
        # Risk from existing positions
        for pos in self.positions:
            position_risk = abs(pos.entry_price - pos.sl_price) * pos.lot_size * 10  # Assuming $10/pip
            total_risk += position_risk
        
        # Risk from new position
        new_position_risk = abs(new_position.entry_price - new_position.sl_price) * new_position.lot_size * 10
        total_risk += new_position_risk
        
        # Calculate as percentage of current equity
        if self.equity_history:
            current_equity = self.equity_history[-1]
            return total_risk / current_equity
        
        return 0.0
    
    def calculate_var(self, confidence_level: float = 0.95, lookback_days: int = 252) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            confidence_level: Confidence level (0.95 for 95% VaR)
            lookback_days: Number of days to look back
            
        Returns:
            VaR as a percentage of portfolio value
        """
        if len(self.returns_history) < 30:  # Need minimum sample
            return 0.0
        
        recent_returns = self.returns_history[-lookback_days:] if len(self.returns_history) > lookback_days else self.returns_history
        
        # Calculate VaR using historical simulation
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(recent_returns, var_percentile)
        
        return abs(var_value)  # Return as positive value
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics for the portfolio."""
        
        # VaR calculations
        var_95 = self.calculate_var(0.95)
        var_99 = self.calculate_var(0.99)
        
        # Drawdown calculations
        max_dd, current_dd = self._calculate_drawdowns()
        
        # Performance ratios
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = self._calculate_calmar_ratio(max_dd)
        
        # Total risk exposure
        total_exposure = sum(pos.lot_size * pos.entry_price for pos in self.positions)
        
        return RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_risk_exposure=total_exposure
        )
    
    def _calculate_drawdowns(self) -> Tuple[float, float]:
        """Calculate maximum and current drawdown."""
        if len(self.equity_history) < 2:
            return 0.0, 0.0
        
        equity_series = np.array(self.equity_history)
        peak = equity_series[0]
        max_dd = 0.0
        current_dd = 0.0
        
        for equity in equity_series:
            if equity > peak:
                peak = equity
            
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        # Current drawdown
        current_peak = np.max(equity_series)
        current_dd = (current_peak - equity_series[-1]) / current_peak
        
        return max_dd, current_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (uses downside deviation instead of total volatility)."""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - risk_free_rate / 252
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, max_drawdown: float) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(self.returns_history) < 30 or max_drawdown == 0:
            return 0.0
        
        annual_return = np.mean(self.returns_history) * 252
        return annual_return / max_drawdown
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary."""
        risk_metrics = self.calculate_risk_metrics()
        
        return {
            'total_positions': len(self.positions),
            'total_exposure': risk_metrics.total_risk_exposure,
            'var_95': risk_metrics.portfolio_var_95,
            'var_99': risk_metrics.portfolio_var_99,
            'max_drawdown': risk_metrics.max_drawdown,
            'current_drawdown': risk_metrics.current_drawdown,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'sortino_ratio': risk_metrics.sortino_ratio,
            'calmar_ratio': risk_metrics.calmar_ratio,
            'positions': [
                {
                    'symbol': pos.symbol,
                    'direction': 'LONG' if pos.direction == 1 else 'SHORT',
                    'lot_size': pos.lot_size,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'bars_held': pos.bars_held
                }
                for pos in self.positions
            ]
        }


class AdaptiveRiskManager:
    """
    Adaptive risk management that combines multiple approaches.
    
    Integrates:
    - Kelly Criterion for optimal sizing
    - Volatility-based adjustments
    - Portfolio-level risk management
    - Market regime awareness
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.kelly = KellyCriterion()
        self.vol_sizing = VolatilityBasedSizing()
        self.portfolio = PortfolioRiskManager()
        
    def calculate_position_size(self, 
                              account_balance: float,
                              current_atr: float,
                              sl_distance_pips: float,
                              market_regime: int = 0,
                              confidence_score: float = 1.0) -> float:
        """
        Calculate optimal position size using multiple risk management approaches.
        
        Args:
            account_balance: Current account balance
            current_atr: Current ATR value
            sl_distance_pips: Stop loss distance in pips
            market_regime: Market regime (-2 to 2, where 2 = strong uptrend)
            confidence_score: Model confidence (0.0 to 1.0)
            
        Returns:
            Optimal position size in lots
        """
        
        # Base position size using Kelly Criterion
        kelly_fraction = self.kelly.calculate_kelly_fraction()
        kelly_risk_amount = account_balance * kelly_fraction
        
        # Volatility-adjusted position size
        vol_adjusted_size = self.vol_sizing.get_position_size(
            account_balance, current_atr, sl_distance_pips
        )
        
        # Market regime adjustment
        regime_multiplier = self._get_regime_multiplier(market_regime)
        
        # Confidence adjustment
        confidence_multiplier = 0.5 + 0.5 * confidence_score  # Range: 0.5 to 1.0
        
        # Combine all factors
        base_size = min(
            kelly_risk_amount / (sl_distance_pips * 10),  # Kelly-based size
            vol_adjusted_size  # Volatility-adjusted size
        )
        
        final_size = base_size * regime_multiplier * confidence_multiplier
        
        # Apply absolute limits
        final_size = max(0.01, min(final_size, 2.0))  # Between 0.01 and 2.0 lots
        
        return final_size
    
    def _get_regime_multiplier(self, market_regime: int) -> float:
        """
        Get position size multiplier based on market regime.
        
        Strong trends = larger positions
        Ranging markets = smaller positions
        """
        regime_multipliers = {
            -2: 1.2,  # Strong downtrend - larger short positions
            -1: 1.0,  # Weak downtrend - normal size
             0: 0.7,  # Ranging market - smaller positions
             1: 1.0,  # Weak uptrend - normal size
             2: 1.2   # Strong uptrend - larger long positions
        }
        
        return regime_multipliers.get(market_regime, 1.0)
    
    def update_trade_result(self, pnl_dollars: float, risk_amount: float, atr_value: float):
        """Update risk management systems with trade result."""
        self.kelly.add_trade(pnl_dollars, risk_amount)
        self.vol_sizing.update_volatility(atr_value)
    
    def update_equity(self, new_equity: float):
        """Update portfolio equity for risk calculations."""
        self.portfolio.update_equity(new_equity)
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk management summary."""
        kelly_stats = self.kelly.get_statistics()
        portfolio_summary = self.portfolio.get_portfolio_summary()
        
        return {
            'kelly_criterion': kelly_stats,
            'portfolio_metrics': portfolio_summary,
            'volatility_history_length': len(self.vol_sizing.vol_history),
            'current_vol_percentile': stats.percentileofscore(
                self.vol_sizing.vol_history, 
                self.vol_sizing.vol_history[-1]
            ) / 100 if self.vol_sizing.vol_history else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the adaptive risk manager
    risk_manager = AdaptiveRiskManager(initial_balance=10000.0)
    
    # Simulate some trades
    np.random.seed(42)
    
    for i in range(50):
        # Simulate market conditions
        atr = np.random.uniform(0.0008, 0.0020)  # ATR between 8-20 pips
        sl_distance = np.random.uniform(30, 70)   # SL between 30-70 pips
        market_regime = np.random.choice([-2, -1, 0, 1, 2])
        confidence = np.random.uniform(0.6, 1.0)
        
        # Calculate position size
        position_size = risk_manager.calculate_position_size(
            account_balance=10000 + i * 100,  # Growing account
            current_atr=atr,
            sl_distance_pips=sl_distance,
            market_regime=market_regime,
            confidence_score=confidence
        )
        
        # Simulate trade result
        win_prob = 0.55  # 55% win rate
        if np.random.random() < win_prob:
            pnl = np.random.uniform(50, 150)  # Win
        else:
            pnl = -np.random.uniform(30, 70)  # Loss
        
        risk_amount = sl_distance * position_size * 10
        
        # Update systems
        risk_manager.update_trade_result(pnl, risk_amount, atr)
        risk_manager.update_equity(10000 + i * 100 + pnl)
        
        if i % 10 == 0:
            print(f"Trade {i}: Size={position_size:.3f}, PnL=${pnl:.2f}, Regime={market_regime}")
    
    # Print final summary
    summary = risk_manager.get_risk_summary()
    print("\nRisk Management Summary:")
    print(f"Kelly Fraction: {summary['kelly_criterion'].get('kelly_fraction', 0):.3f}")
    print(f"Win Rate: {summary['kelly_criterion'].get('win_rate', 0)*100:.1f}%")
    print(f"Profit Factor: {summary['kelly_criterion'].get('profit_factor', 0):.2f}")
    print(f"Portfolio Sharpe: {summary['portfolio_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {summary['portfolio_metrics']['max_drawdown']*100:.1f}%")