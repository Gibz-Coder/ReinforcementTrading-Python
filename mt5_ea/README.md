# High Win Rate AI Trading EA for MT5

## Overview
AI-powered Expert Advisor trained with Reinforcement Learning (PPO) achieving **80.5% win rate** on XAUUSD M15.

## Files
- `HighWinRateEA_v2.mq5` - Expert Advisor source code
- `trading_model.onnx` - Trained AI model (in mt5_export folder)

## Installation

### Method 1: Run Batch File (Windows)
1. Edit `deploy_to_mt5.bat` and update `MT5_DATA` path to match your MT5 installation
2. Run `deploy_to_mt5.bat`
3. Open MT5 → Press F4 (MetaEditor) → Compile (F7)

### Method 2: Manual Installation
1. Copy `HighWinRateEA_v2.mq5` to:
   ```
   [MT5 Data Folder]\MQL5\Experts\
   ```

2. Copy `trading_model.onnx` to:
   ```
   [MT5 Data Folder]\MQL5\Files\
   ```

3. Open MetaEditor (F4) and compile the EA (F7)

### Finding MT5 Data Folder
In MT5: File → Open Data Folder

Common locations:
- `C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\`

## Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| LotSize | 0.01 | Trade lot size |
| TP_ATR_Mult | 0.5 | Take Profit = 0.5 × ATR |
| SL_ATR_Mult | 2.0 | Stop Loss = 2.0 × ATR |
| MagicNumber | 888888 | EA identifier |
| MaxDailyTrades | 20 | Max trades per day |
| MinBarsBetweenTrades | 3 | Cooldown between trades |

## Strategy
- **Timeframe**: M15 (15-minute)
- **Symbol**: XAUUSD (Gold)
- **TP/SL Ratio**: 1:4 (small TP, large SL = high win rate)
- **Model**: PPO (Proximal Policy Optimization)
- **Features**: Price, ATR, RSI, MACD, MA20, Stochastic

## Risk Warning
- Past performance does not guarantee future results
- Always test on demo account first
- Use proper risk management
- The 1:4 TP/SL ratio means losses are larger than wins

## Backtest Results
- Win Rate: ~80%
- Trades: ~2,700 over test period
- Timeframe: 2025 data
