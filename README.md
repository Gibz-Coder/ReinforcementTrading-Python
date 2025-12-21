# High Win Rate Forex Trading with Reinforcement Learning

AI-powered trading system using PPO (Proximal Policy Optimization) achieving **80.5% win rate** on XAUUSD M15.

## Results

| Metric | Value |
|--------|-------|
| Win Rate | 80.5% |
| Total Trades | 2,754 |
| Timeframe | M15 |
| Symbol | XAUUSD |
| TP/SL Ratio | 1:4 |

## Strategy

The model uses a **small TP / large SL** approach:
- Take Profit: 0.5 × ATR
- Stop Loss: 2.0 × ATR

This asymmetric risk/reward achieves high win rate by:
- Taking quick profits on small moves
- Allowing trades room to recover before stopping out

## Project Structure

```
├── models/production/          # Trained models
│   └── highwr_v7_81pct_*.zip  # Best model (80.5% WR)
├── mt5_ea/                     # MetaTrader 5 Expert Advisor
│   └── HighWinRateEA_v2.mq5   # EA source code
├── mt5_export/                 # ONNX export for MT5
│   └── trading_model.onnx     # Exported model
├── scripts/
│   ├── train_highwr_v7.py     # Training script
│   └── export_to_onnx_mt5.py  # ONNX export script
├── src/                        # Core modules
│   ├── environments/          # Trading environment
│   ├── indicators/            # Technical indicators
│   └── ...
└── deploy_to_mt5.bat          # MT5 deployment script
```

## Quick Start

### 1. Setup Environment
```bash
python -m venv forex_env
forex_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Train Model
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
