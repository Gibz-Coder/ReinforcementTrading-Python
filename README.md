# Forex RL Trading Agent

Reinforcement Learning trading agent for EUR/USD forex using PPO (Proximal Policy Optimization).

## Project Structure

```
├── enhanced_indicators.py      # Technical indicators (RSI, MACD, Bollinger, etc.)
├── enhanced_trading_env.py     # Gymnasium trading environment
├── enhanced_train_agent.py     # Training script
├── enhanced_test_agent.py      # Testing/evaluation script
├── high_winrate_indicators.py  # Advanced confluence indicators
├── high_winrate_rewards.py     # Win-rate focused reward system
├── stochastic_supertrend.py    # Stochastic SuperTrend indicator
├── risk_management.py          # Position sizing and risk controls
├── ultra_aggressive_winrate_system.py  # High win-rate training system
├── data/                       # Price data (CSV)
├── checkpoints/                # Saved models
└── logs/                       # TensorBoard logs
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train agent
python enhanced_train_agent.py

# Or use ultra-aggressive system for higher win rate
python ultra_aggressive_winrate_system.py

# Test trained model
python enhanced_test_agent.py
```

## Training Systems

### Standard Training (`enhanced_train_agent.py`)
- Balanced risk/reward approach
- Good for general market conditions

### Ultra-Aggressive (`ultra_aggressive_winrate_system.py`)
- Targets 80%+ win rate
- Strict entry criteria with multiple confirmations
- Higher penalties for losses

## Key Features

- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Stochastic SuperTrend**: Adaptive trend following with volatility adjustment
- **Confluence Trading**: Multiple indicator confirmation required
- **Risk Management**: Dynamic position sizing, stop-loss, take-profit
- **Market Regime Detection**: Trending vs ranging market identification

## Saved Models

Models are saved in `checkpoints/` with timestamps. Best performing models:
- `best_model.zip` - Highest evaluation score
- `*_final.zip` - Final model after training

## Monitoring

View training progress with TensorBoard:
```bash
tensorboard --logdir=logs/
```

## Data Format

CSV with columns: `Local time, Open, High, Low, Close, Volume`
