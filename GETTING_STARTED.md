# Getting Started Guide

## Quick Start (5 minutes)

### 1. Installation

```bash
# Clone and setup
git clone https://github.com/Gibz-Coder/ReinforcementTrading-Python.git
cd ReinforcementTrading-Python

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your EURUSD hourly candlestick data CSV in the `data/` folder:

```
data/
├── EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv
└── test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv
```

Expected CSV columns: `Gmt time, Open, High, Low, Close, Volume`

### 3. Train Your First Agent

```bash
python train_agent.py
```

The script will:

- Load and preprocess data with technical indicators
- Train a PPO agent (default: 1,000,000 timesteps)
- Save checkpoints to `models/`
- Log to TensorBoard

### 4. Backtest the Agent

```bash
python test_agent.py
```

This will:

- Load your trained model
- Run backtest on test data
- Generate trade history CSV
- Display statistics and plots

---

## Detailed Workflow

### Step 1: Data Preparation

Your data should be in hourly candlestick format:

```python
# Example: Load and check your data
import pandas as pd
df = pd.read_csv('data/EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv')
print(df.head())
print(df.info())
```

The data should have columns: `Gmt time`, `Open`, `High`, `Low`, `Close`, `Volume`

### Step 2: Understand the Environment

The trading environment simulates:

1. **Observation**: Window of OHLC + 25+ technical indicators
2. **Actions**:
   - Do nothing (Hold)
   - Close position
   - Open trades (various SL/TP combinations)
3. **Rewards**: Profit/Loss in pips + penalties for overtrading
4. **Risk Management**:
   - Fixed 2% risk per trade
   - Position sizing based on risk
   - Drawdown tracking

### Step 3: Training Process

The training script:

1. Splits data into train/test
2. Creates vectorized environments (parallel)
3. Uses callbacks for monitoring:
   - TradingMetricsCallback: Custom trading metrics
   - EvalCallback: Periodic evaluation
   - CheckpointCallback: Model saving
4. Trains PPO agent
5. Saves best model and logs

**Training takes 2-8 hours depending on hardware.**

### Step 4: Hyperparameter Tuning (Optional)

For optimal results, tune hyperparameters:

```bash
python optimize_hyperparams.py
```

This will:

- Run 100 Optuna trials
- Test different learning rates, architectures, etc.
- Save best hyperparameters to `best_hyperparams.json`
- Takes 24-48 hours

To use optimized hyperparameters:

```python
import json
with open('best_hyperparams.json') as f:
    params = json.load(f)
# Pass params to PPO model
```

### Step 5: Evaluation and Analysis

After backtesting, analyze results:

```python
import pandas as pd

# Load trade history
trades = pd.read_csv('results/trade_history_YYYYMMDD_HHMMSS.csv')

# Metrics
print(f"Total trades: {len(trades)}")
print(f"Win rate: {len(trades[trades['pnl_pips']>0])/len(trades)*100:.2f}%")
print(f"Total PnL: {trades['pnl_pips'].sum()} pips")
print(f"Profit factor: {trades[trades['pnl_pips']>0]['pnl_pips'].sum() / abs(trades[trades['pnl_pips']<0]['pnl_pips'].sum()):.2f}")
```

---

## Customization Guide

### 1. Modify Environment Parameters

Edit `trading_env.py`:

```python
# Risk management
initial_balance = 10000.0       # Starting capital
risk_per_trade = 0.02           # 2% risk per trade
max_trades_per_day = 5          # Max trades daily

# Stop-loss and Take-profit options (pips)
sl_options = [30, 50, 70, 100]  # Add more options
tp_options = [30, 50, 70, 100]

# Observation window
window_size = 50  # More history = more context
```

### 2. Change Training Parameters

Edit `train_agent.py`:

```python
# Training hyperparameters
total_timesteps = 2_000_000     # Longer training
learning_rate = 5e-5
batch_size = 256
n_epochs = 10
```

### 3. Adjust Technical Indicators

Edit `indicators.py` to add/remove indicators:

```python
# Add custom indicator
df['custom_indicator'] = your_calculation

# Remove unwanted indicator
# df = df.drop('unwanted_column', axis=1)
```

### 4. Custom Reward Function

Modify reward calculation in `trading_env.py`:

```python
# Current: reward = pnl_in_pips
# Custom example: Sharpe-based reward
reward = pnl_pips * sharpe_ratio - overtrading_penalty
```

---

## Troubleshooting

### Issue: "No data files found"

**Solution**: Ensure CSV files are in `data/` folder with correct naming.

### Issue: "CUDA out of memory"

**Solution**:

```python
# In train_agent.py, use CPU or reduce batch size
model = PPO(policy, env, learning_rate=5e-5, batch_size=64, device='cpu')
```

### Issue: "Model not improving"

**Solutions**:

1. Check reward signal is positive
2. Verify technical indicators are calculated correctly
3. Increase training timesteps
4. Adjust reward function
5. Try different learning rates

### Issue: "Backtest shows negative returns"

**Solutions**:

1. Check market regime (trending vs range-bound)
2. Verify SL/TP distances are realistic
3. Consider increasing training time
4. Review individual trades for errors

### Issue: "Training is very slow"

**Solutions**:

```python
# Use more workers
from stable_baselines3.common.vec_env import SubprocVecEnv
n_envs = 16  # Increase from default
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```

---

## Best Practices

### Data Management

- Use at least 2 years of historical data
- Keep train/test data separate (no look-ahead)
- Verify data quality and handle gaps
- Use consistent time zones

### Training

- Start with small models to debug quickly
- Monitor training metrics in TensorBoard
- Save checkpoints regularly
- Validate on out-of-sample data

### Backtesting

- Always test on data model hasn't seen
- Use realistic slippage and spread assumptions
- Check individual trades for logic errors
- Compare with random trading baseline

### Production

- Never use real money without extensive validation
- Start with micro lots
- Monitor live performance continuously
- Be ready to stop trading if performance degrades

---

## Next Steps

1. **Experiment with different data ranges**

   - Try different market conditions
   - Test multiple currency pairs

2. **Optimize for your broker's conditions**

   - Adjust spread assumptions
   - Tune position sizing

3. **Combine with other strategies**

   - Use multiple agents
   - Ensemble methods

4. **Implement real-time trading**
   - Use trading APIs (MetaTrader, Interactive Brokers)
   - Monitor performance continuously

---

## Resources

- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **Optuna**: https://optuna.readthedocs.io/
- **Pandas-TA**: https://github.com/twopirllc/pandas-ta

---

## Support

For issues or questions:

1. Check existing GitHub issues
2. Review documentation and examples
3. Check TensorBoard logs for training insights
4. Open a new issue with detailed description

**Happy Trading!** 🚀

---

_Last Updated: December 2025_
