# API Documentation

## trading_env.py - ForexTradingEnv

The core environment for Forex trading simulation using the Gymnasium API.

### Class: ForexTradingEnv

```python
class ForexTradingEnv(gym.Env)
```

Custom environment for EUR/USD Forex trading with technical indicator observations.

#### Constructor

```python
__init__(
    df: pd.DataFrame,
    window_size: int = 30,
    sl_options: List[int] = None,
    tp_options: List[int] = None,
    initial_balance: float = 10000.0,
    risk_per_trade: float = 0.02,
    max_trades_per_day: int = 5,
    render_mode: str = None
)
```

**Parameters:**

- `df`: EURUSD data with technical indicators (must include Open, High, Low, Close)
- `window_size`: Number of historical candles to include in observation (default: 30)
- `sl_options`: List of stop-loss distances in pips (default: [30, 50, 70])
- `tp_options`: List of take-profit distances in pips (default: [30, 50, 70])
- `initial_balance`: Starting account balance in USD (default: 10000)
- `risk_per_trade`: Risk percentage per trade (default: 0.02 = 2%)
- `max_trades_per_day`: Maximum trades allowed per day
- `render_mode`: Visualization mode ("human" or None)

#### Methods

##### reset()

```python
def reset(self, *, seed=None, options=None):
```

Reset environment to initial state.

**Returns:**

- `observation`: Initial state observation (window_size × num_features)
- `info`: Dictionary with environment metadata

##### step(action)

```python
def step(self, action: int):
```

Execute one trading action.

**Parameters:**

- `action`: Action index from the action space

**Returns:**

- `observation`: Next state observation
- `reward`: Reward signal (profit/loss in pips)
- `terminated`: Episode ended (max steps reached)
- `truncated`: Episode truncated (drawdown limit)
- `info`: Dictionary with trade metadata

**Info Keys:**

- `trade_open`: Boolean, whether a position is currently open
- `entry_price`: Entry price if position open
- `pnl_pips`: Unrealized profit/loss in pips
- `trades_today`: Number of trades executed today
- `balance`: Current account balance
- `equity`: Current total equity
- `max_drawdown`: Maximum drawdown percentage

##### render()

```python
def render(self):
```

Render the current state (text output).

#### Action Space

The action space is discrete with size = 2 + (2 × len(sl_options) × len(tp_options))

- **Action 0**: Hold/Do nothing (keep current position)
- **Action 1**: Close current position
- **Actions 2+**: Open new trade with specific SL/TP pair

Example with default options (3×3=9 SL/TP combinations):

- Actions 2-10: Open trades with different stop-loss and take-profit levels

#### Observation Space

Box space with shape (window_size, num_features) where features include:

- OHLC prices
- Technical indicators (MACD, Bollinger Bands, RSI, Stochastic, ADX, etc.)
- Market regime indicators
- Position information (if open)

---

## indicators.py - Technical Analysis

Data loading and preprocessing with technical indicators.

### Function: load_and_preprocess_data()

```python
def load_and_preprocess_data(
    csv_path: str,
    normalize: bool = True
) -> pd.DataFrame
```

Load EURUSD CSV data and add technical indicators.

**Parameters:**

- `csv_path`: Path to EURUSD candlestick CSV file
- `normalize`: Whether to z-score normalize indicators (default: True)

**CSV Format:**

```
Gmt time,Open,High,Low,Close,Volume
2020-07-01 00:00:00,1.1234,1.1245,1.1230,1.1240,1000000
...
```

**Returns:**

- `pd.DataFrame`: DataFrame with columns:

**Trend Indicators:**

- `ma_20`, `ma_50`, `ma_200`: Simple moving averages
- `ema_12`, `ema_26`: Exponential moving averages
- `ma_20_slope`, `ma_50_slope`: MA slopes
- `ma_cross`: MA crossover signal (1 if ma_20 > ma_50)

**Momentum Indicators:**

- `macd`: MACD line
- `macd_signal`: MACD signal line
- `macd_histogram`: MACD histogram
- `rsi_14`: Relative Strength Index
- `roc`: Rate of change

**Volatility Indicators:**

- `bb_upper`, `bb_middle`, `bb_lower`: Bollinger Bands
- `bb_width`: Band width
- `bb_pct`: Price position within bands
- `atr`: Average True Range

**Oscillators:**

- `stoch_k`: Stochastic %K
- `stoch_d`: Stochastic %D
- `adx`: Average Directional Index (trend strength)

**Normalized Features:**

- All indicators normalized using z-score if `normalize=True`
- Prefix `_norm` added to normalized columns

---

## train_agent.py - Training Pipeline

PPO agent training with callbacks and monitoring.

### Class: TradingMetricsCallback

```python
class TradingMetricsCallback(BaseCallback)
```

Custom callback for logging trading-specific metrics during training.

**Tracked Metrics:**

- Episode rewards
- Episode lengths
- Win rates
- Maximum drawdowns

### Function: train_ppo_agent()

```python
def train_ppo_agent(
    env,
    total_timesteps: int = 1000000,
    learning_rate: float = 3e-5,
    batch_size: int = 128,
    n_epochs: int = 10,
    model_name: str = "ppo_forex_trading",
    save_interval: int = 50000
)
```

Train PPO agent on Forex trading environment.

**Parameters:**

- `env`: Training environment (vectorized)
- `total_timesteps`: Total timesteps to train
- `learning_rate`: PPO learning rate
- `batch_size`: Training batch size
- `n_epochs`: Number of training epochs per update
- `model_name`: Name for checkpoint directory
- `save_interval`: Save checkpoint every N timesteps

**Returns:**

- `model`: Trained PPO model

---

## test_agent.py - Backtesting & Evaluation

Performance evaluation and visualization.

### Function: calculate_sharpe_ratio()

```python
def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252*24
) -> float
```

Calculate annualized Sharpe ratio.

**Parameters:**

- `returns`: List of returns
- `risk_free_rate`: Annual risk-free rate (default: 2%)
- `periods_per_year`: Number of periods per year (default: 5,880 for hourly)

**Returns:**

- Sharpe ratio (float)

### Function: calculate_max_drawdown()

```python
def calculate_max_drawdown(
    equity_curve: List[float]
) -> Tuple[float, List[float]]
```

Calculate maximum drawdown and drawdown series.

**Parameters:**

- `equity_curve`: Equity values over time

**Returns:**

- `max_dd`: Maximum drawdown (percentage)
- `drawdowns`: Drawdown values at each step

### Function: evaluate_agent()

```python
def evaluate_agent(
    model,
    env,
    num_episodes: int = 10
) -> Tuple[float, float, List[dict]]
```

Evaluate trained model on test environment.

**Returns:**

- `mean_reward`: Average reward per episode
- `std_reward`: Standard deviation of rewards
- `trades`: List of trade dictionaries with entry/exit details

---

## optimize_hyperparams.py - Hyperparameter Tuning

Optuna-based hyperparameter optimization.

### Function: objective()

```python
def objective(trial: optuna.Trial) -> float
```

Objective function for Optuna optimization.

**Optimized Parameters:**

- `learning_rate`: 1e-6 to 1e-3
- `gamma`: 0.95 to 0.999
- `gae_lambda`: 0.95 to 0.999
- `ent_coef`: 0.0 to 0.5
- `vf_coef`: 0.0 to 1.0
- `max_grad_norm`: 0.1 to 1.0
- `n_steps`: 2048 to 8192
- `batch_size`: 64 to 256
- `n_epochs`: 5 to 20
- `clip_range`: 0.1 to 0.4
- `net_arch`: Small, Medium, or Large

### Function: optimize_hyperparameters()

```python
def optimize_hyperparameters(
    n_trials: int = 100,
    timeout: int = None
) -> dict
```

Run hyperparameter optimization.

**Parameters:**

- `n_trials`: Number of trials to run
- `timeout`: Timeout in seconds (None for no timeout)

**Returns:**

- Best hyperparameters dictionary

---

## Data Format

### Input CSV Format

```
Gmt time,Open,High,Low,Close,Volume
2020-07-01 00:00:00,1.1234,1.1245,1.1230,1.1240,1000000
2020-07-01 01:00:00,1.1240,1.1250,1.1238,1.1248,950000
```

### Trade History Output

```
timestamp,direction,entry_price,exit_price,sl_pips,tp_pips,
exit_reason,duration_hours,pnl_pips,pnl_percent,balance
2023-02-01 10:00:00,LONG,1.1234,1.1240,50,70,TP,2.5,6,0.05,10060
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameters or data format
- `RuntimeError`: Environment step error or model training failure
- `FileNotFoundError`: Data file not found

### Best Practices

1. Always validate CSV data before loading
2. Check for missing values in indicators
3. Verify action index is within action space
4. Monitor reward signals during training
5. Use try-except for file operations

---

## Performance Tips

1. **Data Preprocessing**: Normalize and handle missing values
2. **Environment**: Use SubprocVecEnv for parallel training
3. **Model**: Start with small networks, increase if needed
4. **Training**: Use appropriate learning rates (1e-4 to 1e-5)
5. **Evaluation**: Always backtest on unseen data
6. **Monitoring**: Check TensorBoard logs regularly

---

**Last Updated**: December 2025  
**API Version**: 1.0
