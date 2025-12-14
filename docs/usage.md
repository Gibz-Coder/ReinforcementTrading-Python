# Usage Guide

## Quick Start

### 1. Prepare Your Data

Place your forex data CSV files in the `data/raw/` directory:

```
data/raw/
├── EURUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv
├── GBPUSD_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv
└── USDJPY_Candlestick_1_Hour_BID_01.07.2020-15.07.2023.csv
```

**Required CSV Format:**
```csv
Gmt time,Open,High,Low,Close,Volume
01.07.2020 00:00:00.000,1.12336,1.12336,1.12275,1.12306,4148.0298
01.07.2020 01:00:00.000,1.12306,1.12395,1.12288,1.12385,5375.5801
```

### 2. Train Your First Model

**Ultra-High Win Rate System (Recommended for beginners):**
```bash
python scripts/train_ultra_aggressive.py --timesteps 500000
```

**Enhanced Standard System:**
```bash
python scripts/train_enhanced.py --timesteps 500000
```

### 3. Test Your Model

```bash
python scripts/test_model.py --model models/production/your_model.zip
```

## Training Systems

### Ultra-Aggressive Win Rate System

**Purpose**: Achieve 80%+ win rate through ultra-conservative trading
**Best for**: Consistent profits with minimal drawdowns

```bash
# Basic training
python scripts/train_ultra_aggressive.py

# Advanced options
python scripts/train_ultra_aggressive.py \
    --data data/raw/EURUSD_H1.csv \
    --timesteps 1000000 \
    --target-winrate 0.85 \
    --model-name my_ultra_model
```

**Key Features:**
- Requires 4+ confirmations before entry
- 85%+ confidence threshold
- Maximum 2 trades per day
- Strong trend requirement
- Low volatility only

### Enhanced Standard System

**Purpose**: Balanced approach targeting 65-75% win rate
**Best for**: Higher returns with acceptable risk

```bash
# Basic training
python scripts/train_enhanced.py

# With curriculum learning
python scripts/train_enhanced.py \
    --use-curriculum \
    --timesteps 1000000

# With walk-forward validation
python scripts/train_enhanced.py \
    --use-walk-forward \
    --timesteps 2000000
```

**Key Features:**
- Multi-timeframe analysis
- Adaptive risk management
- Market regime detection
- Curriculum learning support

## Configuration

### Training Configuration

Edit `config/training_config.yaml` to customize training:

```yaml
# Model parameters
model:
  learning_rate: 0.0002
  batch_size: 128
  n_steps: 4096

# Environment settings
environment:
  window_size: 30
  max_trades_per_day: 5
  spread_pips: 1.5

# Ultra-aggressive settings
ultra_aggressive:
  min_confidence: 0.85
  required_confirmations: 4
```

### Custom Training Script

```python
import sys
sys.path.append('.')

from src.training.enhanced_train_agent import train_enhanced_agent

# Custom training
results = train_enhanced_agent(
    data_path="data/raw/my_data.csv",
    total_timesteps=1000000,
    learning_rate=0.0001,
    batch_size=256,
    model_name="my_custom_model"
)
```

## Testing and Evaluation

### Basic Testing

```bash
# Test on same data used for training
python scripts/test_model.py --model models/production/best_model.zip

# Test on different data
python scripts/test_model.py \
    --model models/production/best_model.zip \
    --data data/raw/test_data.csv \
    --output results/reports/
```

### Advanced Testing

```python
from src.testing.enhanced_test_agent import test_model_comprehensive

# Comprehensive testing
results = test_model_comprehensive(
    model_path="models/production/best_model.zip",
    test_data_paths=[
        "data/raw/EURUSD_2023.csv",
        "data/raw/GBPUSD_2023.csv"
    ],
    generate_report=True
)
```

### Performance Metrics

The system tracks comprehensive metrics:

- **Win Rate**: Percentage of profitable trades
- **Return**: Total percentage return
- **Max Drawdown**: Maximum equity decline
- **Sharpe Ratio**: Risk-adjusted returns
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Average profit/loss per trade

## Monitoring Training

### TensorBoard

Monitor training progress in real-time:

```bash
tensorboard --logdir logs/
```

Open http://localhost:6006 in your browser to view:
- Training loss and rewards
- Win rate progression
- Equity curves
- Action distributions

### Log Files

Training logs are saved in:
```
logs/
├── enhanced_ppo_forex_20231214_120000/
│   ├── events.out.tfevents.xxx
│   └── progress.csv
└── ultra_aggressive_20231214_130000/
    ├── events.out.tfevents.xxx
    └── progress.csv
```

## Model Management

### Saving Models

Models are automatically saved during training:

```
models/
├── production/          # Production-ready models
│   ├── best_model_winrate_82.3pct.zip
│   └── ultra_aggressive_final.zip
└── experimental/        # Development models
    ├── test_model_v1.zip
    └── experimental_lstm.zip
```

### Loading Models

```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("models/production/best_model.zip")

# Use for prediction
action, _states = model.predict(observation, deterministic=True)
```

## Data Management

### Data Preprocessing

The system automatically:
- Normalizes price data
- Calculates 80+ technical indicators
- Detects market regimes
- Handles missing data

### Adding New Data

1. Place CSV files in `data/raw/`
2. Ensure proper format (see above)
3. Update data paths in training scripts

### Data Quality Checks

```python
from src.indicators.enhanced_indicators import load_and_preprocess_data_enhanced

# Load and validate data
df = load_and_preprocess_data_enhanced("data/raw/new_data.csv")
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Missing values: {df.isnull().sum().sum()}")
```

## Advanced Features

### Ensemble Learning

Train multiple models and combine predictions:

```python
from src.training.improved_training_config import EnsembleTraining

ensemble = EnsembleTraining(n_models=3)
ensemble.train_ensemble(env, total_timesteps=1000000)

# Get ensemble prediction
action = ensemble.predict_ensemble(observation)
```

### Hyperparameter Optimization

Use Optuna for automated hyperparameter search:

```python
from src.training.improved_training_config import optimize_hyperparameters_with_optuna

best_params = optimize_hyperparameters_with_optuna(n_trials=50)
print(f"Best parameters: {best_params}")
```

### Walk-Forward Analysis

For robust validation:

```bash
python scripts/train_enhanced.py --use-walk-forward --timesteps 2000000
```

## Production Deployment

### Model Selection

Choose models based on:
1. **Win Rate**: Target 70%+ for production
2. **Drawdown**: Keep under 15%
3. **Consistency**: Stable performance across different periods
4. **Trade Frequency**: Sufficient opportunities

### Risk Management

Before live trading:
1. **Paper Trading**: Test with virtual money
2. **Position Sizing**: Start with small positions
3. **Stop Losses**: Always use protective stops
4. **Monitoring**: Continuously monitor performance

### Live Trading Integration

```python
# Example integration (pseudo-code)
class LiveTrader:
    def __init__(self, model_path):
        self.model = PPO.load(model_path)
        self.env = create_live_environment()
    
    def get_signal(self, current_data):
        obs = self.env.preprocess(current_data)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.env.interpret_action(action)
```

## Troubleshooting

### Common Issues

**1. Poor Performance**
- Increase training timesteps
- Adjust hyperparameters
- Use more/better quality data
- Try different reward systems

**2. Overfitting**
- Use walk-forward validation
- Reduce model complexity
- Add regularization
- Use more diverse training data

**3. Memory Issues**
- Reduce batch size
- Use gradient checkpointing
- Process data in chunks
- Use CPU instead of GPU

### Getting Help

1. Check [Troubleshooting Guide](troubleshooting.md)
2. Review training logs
3. Validate data quality
4. Test with smaller datasets first
5. Open an issue with detailed information

## Best Practices

1. **Start Simple**: Begin with ultra-aggressive system
2. **Quality Data**: Use clean, gap-free data
3. **Sufficient Training**: Use at least 500k timesteps
4. **Validate Thoroughly**: Test on out-of-sample data
5. **Monitor Continuously**: Track performance metrics
6. **Risk Management**: Always use proper position sizing
7. **Paper Trade First**: Never risk real money without testing