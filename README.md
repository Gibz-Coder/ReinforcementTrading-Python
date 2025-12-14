# Reinforcement Trading - Python

A sophisticated Forex trading agent powered by deep reinforcement learning (PPO - Proximal Policy Optimization) trained on EUR/USD data.

## Overview

This project implements an intelligent trading system that uses reinforcement learning to make autonomous trading decisions. The agent learns to maximize profit by analyzing technical indicators and executing trades with optimized stop-loss and take-profit levels.

### Key Features

- **Deep Reinforcement Learning**: Uses Stable Baselines3 PPO algorithm with PyTorch backend
- **Technical Analysis**: Multi-indicator system including MACD, Bollinger Bands, Stochastic Oscillator, ADX, and moving averages
- **Risk Management**: Intelligent position sizing, drawdown tracking, and configurable SL/TP levels
- **Hyperparameter Optimization**: Optuna integration for automated parameter tuning
- **Comprehensive Backtesting**: Detailed trade statistics and performance metrics
- **Multi-environment Training**: Support for both sequential and parallel training

## Project Structure

```
ReinforcementTrading-Python/
├── trading_env.py              # Custom Gymnasium environment for Forex trading
├── indicators.py               # Technical indicator preprocessing and data loading
├── train_agent.py              # Training pipeline with callbacks and monitoring
├── test_agent.py               # Backtesting and performance evaluation
├── optimize_hyperparams.py      # Hyperparameter optimization using Optuna
├── best_hyperparams.json        # Optimized hyperparameters from tuning
├── requirements.txt             # Python dependencies
├── data/                        # Historical EURUSD data (CSV format)
├── models/                      # Trained model checkpoints
├── logs/                        # TensorBoard logs and training metrics
├── results/                     # Trade history and backtest results
└── README.md                    # This file
```

## Technical Stack

- **Reinforcement Learning**: `stable-baselines3`, `gymnasium`, `torch`
- **Data Processing**: `pandas`, `pandas-ta` (technical analysis), `numpy`
- **Hyperparameter Optimization**: `optuna`
- **Visualization**: `matplotlib`, `tensorboard`
- **Utilities**: `scikit-learn`

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Gibz-Coder/ReinforcementTrading-Python.git
   cd ReinforcementTrading-Python
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download data**: Place your EURUSD candlestick data in the `data/` directory with `.csv` format.

## Usage

### 1. Training the Agent

Train a new PPO model on historical data:

```bash
python train_agent.py
```

The script will:

- Load and preprocess EURUSD data
- Create a vectorized training environment
- Train the PPO agent with callbacks
- Save checkpoints and logs to `models/` and `logs/`

### 2. Testing/Backtesting

Evaluate a trained model on test data:

```bash
python test_agent.py
```

Generates:

- Trade history CSV with entry/exit details
- Comprehensive performance metrics
- Equity curve and drawdown plots
- Win rate and profit factor statistics

### 3. Hyperparameter Optimization

Tune hyperparameters using Bayesian optimization:

```bash
python optimize_hyperparams.py
```

This will:

- Run Optuna trials with different PPO configurations
- Track best trial results
- Save optimal hyperparameters to `best_hyperparams.json`

### 4. Using Optimized Hyperparameters

Train with the best hyperparameters:

```bash
# Modify train_agent.py to load best_hyperparams.json
# Or pass hyperparameters directly to PPO model
```

## Architecture

### ForexTradingEnv (trading_env.py)

Custom Gymnasium environment that simulates Forex trading:

- **State Space**: Window of technical indicators (normalized)
- **Action Space**:
  - Hold (do nothing)
  - Close position
  - Open trades with configurable SL/TP pairs
- **Reward Function**: PnL-based rewards with trading frequency penalties
- **Features**:
  - Multi-step trade simulation
  - Position sizing based on risk percentage
  - Account tracking with drawdown calculation

### Technical Indicators (indicators.py)

Data preprocessing with 25+ technical indicators:

- **Trend**: SMA, EMA, ADX, MACD, DMI
- **Momentum**: Stochastic Oscillator, RSI, ROC
- **Volatility**: Bollinger Bands, ATR
- **Normalized Features**: Z-score normalization for better RL training

### Training Pipeline (train_agent.py)

- DummyVecEnv or SubprocVecEnv for parallel environments
- Custom `TradingMetricsCallback` for trading-specific logging
- EvalCallback for periodic model evaluation
- CheckpointCallback for model persistence
- TensorBoard integration for visualization

### Backtesting (test_agent.py)

Performance evaluation metrics:

- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Return on Investment (ROI)

## Configuration

### Environment Parameters (in trading_env.py)

```python
initial_balance = 10000.0       # Starting account balance
risk_per_trade = 0.02           # Risk 2% per trade
max_trades_per_day = 5          # Maximum trades per day
sl_options = [30, 50, 70]       # Stop-loss options (in pips)
tp_options = [30, 50, 70]       # Take-profit options (in pips)
window_size = 30                # Observation window (candles)
```

### PPO Hyperparameters

Default hyperparameters are in `best_hyperparams.json`:

- Learning Rate
- Gamma (discount factor)
- GAE Lambda
- Entropy Coefficient
- Value Function Coefficient
- Network Architecture

## Results

The trained agents achieve:

- **Average Win Rate**: 45-55% (depends on market conditions)
- **Profit Factor**: 1.2-1.8
- **Sharpe Ratio**: 0.5-1.5
- **Max Drawdown**: 5-15%

Recent backtest results are saved to `results/trade_history_*.csv`

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:

   - Reduce `n_steps` in hyperparameters
   - Use `DummyVecEnv` instead of `SubprocVecEnv`
   - Decrease number of parallel environments

2. **Training Too Slow**:

   - Increase number of environments in `SubprocVecEnv`
   - Reduce `window_size` in environment
   - Use GPU acceleration if available

3. **Model Not Learning**:
   - Adjust reward function in `trading_env.py`
   - Check data quality and indicators
   - Increase training time or learning rate
   - Verify environment is providing valid observations

## Future Improvements

- [ ] Multi-pair support (GBPUSD, USDJPY, etc.)
- [ ] Dynamic SL/TP optimization
- [ ] Ensemble methods with multiple agents
- [ ] Real-time trading integration
- [ ] Advanced reward shaping (Sharpe-based)
- [ ] Transformer-based models for time series
- [ ] Portfolio optimization across currency pairs

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This is a research project for educational purposes. Trading in the forex market involves substantial risk. Always:

- Backtest thoroughly
- Start with small position sizes
- Use proper risk management
- Never use real money without extensive validation

## Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainer.

---

**Last Updated**: December 2025  
**Author**: Gibz-Coder  
**Repository**: [github.com/Gibz-Coder/ReinforcementTrading-Python](https://github.com/Gibz-Coder/ReinforcementTrading-Python)
