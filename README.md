# High Win Rate Forex Trading with Reinforcement Learning

AI-powered trading system using PPO (Proximal Policy Optimization) for XAUUSD M15 trading with **Ultra-Selective V4** model achieving 58.5%+ win rate.

## 🚀 Latest Achievement - Ultra-Selective V4 Model (Dec 23, 2025)

**BREAKTHROUGH**: New V4 model with curriculum learning approach!

| Metric | V4 Results | Previous V3 | Improvement |
|--------|------------|-------------|-------------|
| **Validation Win Rate** | **58.5%** | 28-42% | **+30%** |
| **Training Stability** | ✅ Improving | ❌ Declining | **Stable Learning** |
| **Trade Quality** | **10+ trades/eval** | Poor/None | **Selective & Active** |
| **Risk/Reward** | **1:1 (Balanced)** | 1:4 | **Better Risk Profile** |
| **Overfitting** | ✅ Minimal | ❌ High | **Better Generalization** |

### 🎯 V4 Key Features
- **Ultra-Selective Signals**: Only trades 7/10 perfect conditions
- **Curriculum Learning**: Progressive difficulty stages
- **Balanced 1:1 Risk/Reward**: Equal TP and SL distances
- **Quality Over Quantity**: 2-3 high-probability trades per day
- **Target**: 80%+ win rate (currently progressing toward this goal)

## Previous Best Results (High WR v7)

| Metric | Value |
|--------|-------|
| Win Rate | 80.5% |
| Total Trades | 2,754 |
| Timeframe | M15 |
| Symbol | XAUUSD |
| TP/SL Ratio | 1:4 |

## Strategy Evolution

### V4 Ultra-Selective Model (Current - RECOMMENDED)
- **Risk/Reward**: 1:1 (Balanced TP/SL using ATR)
- **Approach**: Ultra-selective signal filtering + curriculum learning
- **Target Win Rate**: 80%+ (currently achieving 58.5%+)
- **Trade Frequency**: 2-3 high-quality trades per day
- **Key Innovation**: Only trades when 7/10 perfect conditions are met

### V7 High Win Rate Model (Legacy)
- **Risk/Reward**: 1:4 (Small TP / Large SL)
- **Achieved**: 80.5% win rate
- **Approach**: Quick profits with room for recovery
- **Trade Frequency**: Higher volume, smaller individual profits

## Project Structure

```
├── models/
│   ├── production/             # Production-ready models
│   │   ├── highwr_v7_81pct_*   # Legacy V7 model (80.5% WR)
│   │   └── ultra_selective_v4_* # New V4 models (58.5%+ WR)
│   └── experimental/           # Training checkpoints
├── scripts/
│   ├── train_ultra_selective_v4.py  # 🚀 NEW V4 training (RECOMMENDED)
│   ├── train_balanced_rr_v3.py      # V3 training (improved)
│   └── train_highwr_v7.py           # Legacy V7 training
├── docs/
│   ├── ultra_selective_v4_improvements.md  # V4 detailed analysis
│   ├── installation.md
│   ├── usage.md
│   └── troubleshooting.md
├── mt5_ea/                     # MetaTrader 5 Expert Advisor
│   └── HighWinRateEA_v2.mq5   # EA source code
├── mt5_export/                 # ONNX export for MT5
│   └── trading_model.onnx     # Exported model
├── src/                        # Core modules
│   ├── environments/          # Trading environment
│   ├── indicators/            # Technical indicators
│   └── ...
└── deploy_to_mt5.bat          # MT5 deployment script
```

## Quick Start

### 🚀 V4 Ultra-Selective Model (RECOMMENDED)

#### 1. Setup Environment
```bash
python -m venv forex_env
forex_env\Scripts\activate
pip install -r requirements.txt
```

#### 2. Train V4 Model
```bash
# Quick test (25K timesteps)
python scripts/train_ultra_selective_v4.py --timesteps 25000 --envs 4

# Full training for production (500K timesteps)
python scripts/train_ultra_selective_v4.py --timesteps 500000 --envs 8
```

#### 3. Monitor Training Progress
- Models achieving 75%+ win rate automatically move to `models/production/`
- Watch for curriculum progression through stages 1→2→3
- Best models are saved with descriptive names (e.g., `ultra_selective_v4_wr75_20251223_120000`)

### Legacy V7 Model

#### 2. Train V7 Model
```bash
python scripts/train_highwr_v7.py --timesteps 1000000
```

### 3. Export to ONNX
```bash
python scripts/export_to_onnx_mt5.py --model models/production/highwr_v7_81pct_20251221_104308.zip
```

### 4. Deploy to MT5
1. Edit `deploy_to_mt5.bat` with your MT5 data path
2. Run `deploy_to_mt5.bat`
3. Open MetaEditor (F4) and compile (F7)
4. Attach EA to XAUUSD M15 chart

## MT5 Installation

See [mt5_ea/README.md](mt5_ea/README.md) for detailed MT5 installation instructions.

## Features Used

- OHLC prices (normalized)
- ATR (14)
- RSI (14)
- MACD (12, 26, 9)
- MA20
- Stochastic (14, 3, 3)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Stable-Baselines3
- MetaTrader 5 (for live trading)

## Risk Warning

⚠️ **Trading involves substantial risk of loss.**

- Past performance does not guarantee future results
- Always test on demo account first
- The 1:4 TP/SL ratio means individual losses are larger than wins
- Use proper position sizing and risk management

## License

For personal use only. Not financial advice.
