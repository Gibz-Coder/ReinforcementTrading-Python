import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ForexTradingEnv(gym.Env):
    """
    Enhanced Forex Trading Environment for EUR/USD using Gymnasium API.

    Improvements over basic version:
    - Multi-step trade simulation (holds position until SL/TP hit)
    - Position sizing based on risk percentage
    - Improved reward shaping with trading frequency penalties
    - Proper account management with drawdown tracking
    - Support for both training and evaluation modes
    """

    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, df, window_size=30, sl_options=None, tp_options=None,
                 initial_balance=10000.0, risk_per_trade=0.02, max_trades_per_day=5,
                 render_mode=None):
        super(ForexTradingEnv, self).__init__()

        # Store the dataframe containing prices and indicators
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)

        # Observation parameters
        self.window_size = window_size

        # Risk management parameters
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade  # Risk 2% per trade
        self.max_trades_per_day = max_trades_per_day

        # Discretize SL and TP distances in pips
        self.sl_options = sl_options if sl_options else [30, 50, 70]
        self.tp_options = tp_options if tp_options else [30, 50, 70]

        # --- Construct action space ---
        # Action 0 => No Trade / Hold
        # Action 1 => Close Position (if open)
        # Then for direction in [0=short, 1=long] and each sl, tp => trading actions
        self.action_map = [
            ('hold', None, None, None),   # 0: Do nothing / hold position
            ('close', None, None, None),  # 1: Close current position
        ]
        for direction in [0, 1]:  # 0=short, 1=long
            for sl in self.sl_options:
                for tp in self.tp_options:
                    self.action_map.append(('open', direction, sl, tp))

        self.action_space = spaces.Discrete(len(self.action_map))

        # Number of features in the observation
        self.num_features = self.df.shape[1]

        # Add position state to observation (position_type, unrealized_pnl, bars_held)
        self.extra_features = 3
        total_features = self.num_features + self.extra_features

        # Observation space: window of features + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, total_features),
            dtype=np.float32
        )

        # Render mode
        self.render_mode = render_mode

        # Initialize state variables
        self._init_state()

    def _init_state(self):
        """Initialize or reset all state variables."""
        self.current_step = self.window_size
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance
        self.done = False
        self.truncated = False

        # Position tracking
        self.position = None  # None = no position, dict with position info
        self.bars_in_position = 0

        # Trade statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.trades_today = 0
        self.current_day = self.current_step // 24  # Initialize to current day
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.last_trade_step = 0  # Track when we last traded

        # Logging
        self.equity_curve = []
        self.trade_history = []
        self.last_trade_info = None

        # Drawdown tracking
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0

    def _get_observation(self):
        """
        Returns the observation with market data + position state.
        Shape: (window_size, num_features + extra_features)
        """
        start = max(self.current_step - self.window_size, 0)
        obs_df = self.df.iloc[start:self.current_step]

        # Pad if needed
        if len(obs_df) < self.window_size:
            padding_rows = self.window_size - len(obs_df)
            first_part = np.tile(obs_df.iloc[0].values, (padding_rows, 1))
            obs_array = np.concatenate([first_part, obs_df.values], axis=0)
        else:
            obs_array = obs_df.values

        # Add position state to each row
        # [position_type: -1=short, 0=none, 1=long, unrealized_pnl, bars_held]
        position_state = np.zeros((self.window_size, self.extra_features))

        if self.position is not None:
            current_price = self.df.loc[self.current_step - 1, "Close"] if self.current_step > 0 else 0
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            position_state[:, 0] = 1 if self.position['direction'] == 1 else -1
            position_state[:, 1] = unrealized_pnl / 100  # Normalize
            position_state[:, 2] = min(self.bars_in_position / 50, 1.0)  # Normalize bars held

        obs_array = np.concatenate([obs_array, position_state], axis=1)
        return obs_array.astype(np.float32)

    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized PnL for open position in pips."""
        if self.position is None:
            return 0.0

        entry_price = self.position['entry_price']
        pip_value = 0.0001

        if self.position['direction'] == 1:  # Long
            pnl_pips = (current_price - entry_price) / pip_value
        else:  # Short
            pnl_pips = (entry_price - current_price) / pip_value

        return pnl_pips * self.position['lot_size']

    def _calculate_lot_size(self, sl_pips):
        """Calculate lot size based on risk percentage and stop loss."""
        risk_amount = self.equity * self.risk_per_trade
        pip_value = 0.0001
        # Standard lot = 100,000 units, pip value = $10 per pip
        lot_size = risk_amount / (sl_pips * 10)  # 10 = pip value for standard lot
        return max(0.01, min(lot_size, 10.0))  # Clamp between 0.01 and 10 lots

    def _open_position(self, direction, sl_pips, tp_pips):
        """Open a new position."""
        entry_price = self.df.loc[self.current_step, "Close"]
        lot_size = self._calculate_lot_size(sl_pips)
        pip_value = 0.0001

        if direction == 1:  # Long
            sl_price = entry_price - sl_pips * pip_value
            tp_price = entry_price + tp_pips * pip_value
        else:  # Short
            sl_price = entry_price + sl_pips * pip_value
            tp_price = entry_price - tp_pips * pip_value

        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'lot_size': lot_size,
            'entry_step': self.current_step
        }
        self.bars_in_position = 0
        self.trades_today += 1
        self.total_trades += 1
        self.last_trade_step = self.current_step  # Track when we last traded

    def _close_position(self, exit_price, exit_reason='manual'):
        """Close current position and calculate PnL."""
        if self.position is None:
            return 0.0

        pip_value = 0.0001
        entry_price = self.position['entry_price']
        lot_size = self.position['lot_size']

        if self.position['direction'] == 1:  # Long
            pnl_pips = (exit_price - entry_price) / pip_value
        else:  # Short
            pnl_pips = (entry_price - exit_price) / pip_value

        # PnL in dollars (10 = pip value for standard lot)
        pnl_dollars = pnl_pips * lot_size * 10

        # Update statistics
        if pnl_pips > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Record trade
        trade_record = {
            'entry_step': self.position['entry_step'],
            'exit_step': self.current_step,
            'direction': 'long' if self.position['direction'] == 1 else 'short',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'sl_pips': self.position['sl_pips'],
            'tp_pips': self.position['tp_pips'],
            'lot_size': lot_size,
            'pnl_pips': pnl_pips,
            'pnl_dollars': pnl_dollars,
            'exit_reason': exit_reason,
            'bars_held': self.bars_in_position
        }
        self.trade_history.append(trade_record)
        self.last_trade_info = trade_record

        # Update equity
        self.equity += pnl_dollars

        # Clear position
        self.position = None
        self.bars_in_position = 0

        return pnl_pips

    def _check_sl_tp(self):
        """Check if SL or TP is hit for current position. Returns (hit, exit_price, reason)."""
        if self.position is None:
            return False, None, None

        if self.current_step >= self.n_steps:
            return False, None, None

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

        return False, current_close, None

    def _calculate_reward(self, pnl_pips, action_type, exit_reason=None, trade_direction=None):
        """
        Improved reward function with MARKET REGIME AWARENESS.

        Key principles:
        1. Reward actual profits heavily, penalize losses proportionally
        2. REWARD TRADES ALIGNED WITH MARKET REGIME (key for fixing directional bias)
        3. Penalize trades against the trend
        4. Activity incentive to prevent inaction
        5. Penalize excessive risk-taking (drawdown)
        """
        reward = 0.0

        # Get current market regime from dataframe
        current_idx = max(0, self.current_step - 1)
        market_regime = 0
        trend_direction = 0
        adx_value = 0.25  # Default normalized value

        if 'market_regime' in self.df.columns:
            market_regime = self.df.iloc[current_idx]['market_regime']
        if 'trend_direction' in self.df.columns:
            trend_direction = self.df.iloc[current_idx]['trend_direction']
        if 'adx' in self.df.columns:
            adx_value = self.df.iloc[current_idx]['adx']

        # ============ TRADE OUTCOME REWARDS (Primary) ============
        if exit_reason is not None:
            # This is a closed trade - the most important signal

            if exit_reason == 'take_profit':
                # Strong reward for profitable trades
                reward += 5.0 + pnl_pips * 0.15

                # BONUS: Reward if trade was aligned with trend
                if trade_direction is not None:
                    if (trade_direction == 1 and market_regime >= 1) or \
                       (trade_direction == -1 and market_regime <= -1):
                        reward += 2.0  # Big bonus for trend-aligned profitable trade
                        if adx_value > 0.5:  # Strong trend
                            reward += 1.0  # Extra bonus in strong trends

                # Bonus for winning streaks
                if self.consecutive_wins >= 2:
                    reward += min(self.consecutive_wins * 0.5, 2.0)

            elif exit_reason == 'stop_loss':
                # Base penalty for losses
                reward -= 2.0 + abs(pnl_pips) * 0.05

                # EXTRA PENALTY: For trades against the trend
                if trade_direction is not None:
                    if (trade_direction == 1 and market_regime <= -1) or \
                       (trade_direction == -1 and market_regime >= 1):
                        reward -= 2.0  # Strong penalty for counter-trend loss

                # Extra penalty for losing streaks
                if self.consecutive_losses >= 3:
                    reward -= min(self.consecutive_losses * 0.3, 1.5)

            elif exit_reason == 'max_bars':
                if pnl_pips > 0:
                    reward += pnl_pips * 0.1
                else:
                    reward += pnl_pips * 0.05

        # ============ REGIME-AWARE TRADE OPENING REWARDS ============
        if action_type == 'open' and trade_direction is not None:
            # Strong reward for opening trades aligned with the trend
            if (trade_direction == 1 and market_regime >= 1):
                reward += 1.0  # Reward LONG in uptrend
                if trend_direction > 0:
                    reward += 0.5  # Extra for strong bullish signal
            elif (trade_direction == -1 and market_regime <= -1):
                reward += 1.0  # Reward SHORT in downtrend
                if trend_direction < 0:
                    reward += 0.5  # Extra for strong bearish signal
            elif (trade_direction == 1 and market_regime <= -1):
                reward -= 1.0  # Penalize LONG in downtrend
            elif (trade_direction == -1 and market_regime >= 1):
                reward -= 1.0  # Penalize SHORT in uptrend
            else:
                # Ranging market - small reward for any trade
                reward += 0.3

        # ============ ACTIVITY INCENTIVES (Secondary) ============
        if action_type == 'hold' and self.position is None:
            bars_since_last_trade = self.current_step - self.last_trade_step
            if bars_since_last_trade > 24:
                reward -= 0.3
            if bars_since_last_trade > 48:
                reward -= 0.5
            if bars_since_last_trade > 72:
                reward -= 1.0

        # ============ RISK MANAGEMENT PENALTIES ============
        if self.trades_today > self.max_trades_per_day:
            reward -= 1.0

        if self.position is not None and self.bars_in_position > 48:
            reward -= 0.1 * (self.bars_in_position - 48) / 24

        if self.current_drawdown > 0.15:
            reward -= (self.current_drawdown - 0.15) * 5

        # ============ EQUITY-BASED REWARDS ============
        equity_return = (self.equity - self.initial_balance) / self.initial_balance
        if equity_return > 0:
            reward += equity_return * 0.5

        return reward

    def step(self, action):
        """
        Execute one step in the environment.

        Actions:
        - 0: Hold (do nothing)
        - 1: Close position
        - 2+: Open position with specific direction/SL/TP
        """
        action_type, direction, sl, tp = self.action_map[action]
        reward = 0.0
        pnl_pips = 0.0
        exit_reason = None
        trade_direction = None  # Track direction for reward calculation

        # Check for new day and reset trades_today counter (every 24 bars for hourly data)
        current_bar_day = self.current_step // 24
        if current_bar_day != self.current_day:
            self.current_day = current_bar_day
            self.trades_today = 0

        # Check if SL/TP hit for existing position
        if self.position is not None:
            trade_direction = self.position['direction']  # Get direction before closing
            hit, exit_price, reason = self._check_sl_tp()
            if hit:
                pnl_pips = self._close_position(exit_price, reason)
                exit_reason = reason

        # Process action
        if action_type == 'hold':
            # Just hold, increment bars if in position
            if self.position is not None:
                self.bars_in_position += 1

        elif action_type == 'close':
            # Close existing position at current close
            if self.position is not None:
                trade_direction = self.position['direction']  # Get direction before closing
                current_close = self.df.loc[self.current_step, "Close"]
                pnl_pips = self._close_position(current_close, 'manual')
                exit_reason = 'manual'

        elif action_type == 'open':
            # Only open if no position and not overtrading
            if self.position is None and self.trades_today < self.max_trades_per_day:
                self._open_position(direction, sl, tp)
                trade_direction = direction  # New trade direction

        # Calculate reward with trade direction for regime-aware rewards
        reward = self._calculate_reward(pnl_pips, action_type, exit_reason, trade_direction)

        # Update drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)

        # Log equity
        self.equity_curve.append(self.equity)

        # Move forward
        self.current_step += 1

        # Check if done
        terminated = False
        truncated = False

        if self.current_step >= self.n_steps - 1:
            terminated = True
            # Close any open position at end
            if self.position is not None:
                final_close = self.df.loc[self.n_steps - 1, "Close"]
                self._close_position(final_close, 'end_of_data')

        # Stop if equity drops below 20% of initial (blown account)
        if self.equity < self.initial_balance * 0.2:
            terminated = True
            reward -= 50  # Large penalty for blowing account

        # Get next observation
        obs = self._get_observation()

        # Info dict for debugging
        info = {
            'equity': self.equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'position': self.position is not None,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_observation()
        info = {'equity': self.equity}
        return obs, info

    def render(self):
        """Render the environment state."""
        if self.render_mode == "human":
            pos_str = "None"
            if self.position is not None:
                direction = "LONG" if self.position['direction'] == 1 else "SHORT"
                pos_str = f"{direction} @ {self.position['entry_price']:.5f}"

            print(f"Step: {self.current_step:5d} | "
                  f"Equity: ${self.equity:,.2f} | "
                  f"Position: {pos_str} | "
                  f"Trades: {self.total_trades} | "
                  f"Win Rate: {self.winning_trades/max(1,self.total_trades)*100:.1f}% | "
                  f"Max DD: {self.max_drawdown*100:.1f}%")

    def get_trade_statistics(self):
        """Return comprehensive trade statistics."""
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
            'avg_pnl_pips': np.mean(pnls) if pnls else 0,
            'avg_win_pips': np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0,
            'avg_loss_pips': np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0,
            'max_drawdown': self.max_drawdown,
            'final_equity': self.equity,
            'return_pct': (self.equity - self.initial_balance) / self.initial_balance * 100,
            'profit_factor': abs(sum(p for p in pnls if p > 0) / min(-1, sum(p for p in pnls if p < 0))) if any(p < 0 for p in pnls) else float('inf'),
            'sharpe_ratio': np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252) if pnls else 0
        }
