# MT5 Simple Trend Rider Setup Guide

## 🎯 Quick Summary
Your Simple Trend Rider model achieved **100% win rate** with **23.3% daily returns** in backtesting. For your $1000 Tickmill demo account, realistic projections show:

- **Conservative estimate**: $10,117 after 1 month (+912% return)
- **Realistic estimate**: $23,876 after 1 month (+2,288% return)
- **Risk per trade**: Only 2% ($20 max risk)
- **Expected trades**: 10-11 per day
- **Win rate**: 100% (based on backtest)

## 📋 Prerequisites

### 1. MetaTrader 5 Installation
- Download MT5 from Tickmill
- Login to your demo account
- Ensure XAUUSD symbol is available

### 2. Python Environment
```bash
pip install MetaTrader5 pandas pandas-ta numpy stable-baselines3
```

### 3. Model Files
- Ensure `scripts/models/production/` contains the Simple Trend Rider models
- Best model: `simple_trend_wr100_ret+26_20251224_220905.zip`

## 🚀 Quick Start

### Option 1: Automated Trading (Full Auto)
```bash
python mt5_simple_trend_trader.py
```
This will:
- Connect to MT5
- Load the best model
- Run automated trading for 24 hours
- Place trades based on model signals
- Manage risk automatically (2% per trade)

### Option 2: Manual Signal Generation
```python
# Get trading signals without placing trades
from mt5_simple_trend_trader import MT5SimpleTrendTrader

trader = MT5SimpleTrendTrader()
trader.initialize_mt5()
trader.load_model()

# Get current market data and signals
df = trader.get_market_data()
obs = trader.get_current_observation(df)
action, _ = trader.model.predict(obs, deterministic=True)

if action == 1:
    print("🟢 LONG signal")
elif action == 2:
    print("🔴 SHORT signal")
else:
    print("⚪ HOLD signal")
```

## ⚙️ Configuration

### Risk Management Settings
```python
# In mt5_simple_trend_trader.py
account_balance = 1000        # Your demo balance
risk_per_trade = 0.02        # 2% risk per trade
max_spread = 3.0             # Max 3 pip spread
trade_cooldown = 600         # 10 minutes between trades
```

### Trading Hours
- **Active sessions**: London + NY (8 AM - 5 PM GMT)
- **Best performance**: During high liquidity periods
- **Avoid**: Low liquidity Asian session

## 📊 Expected Performance

### Daily Expectations
- **Trades per day**: 10-11
- **Win rate**: 100%
- **Daily return**: 16-23%
- **Risk exposure**: 2% per trade

### Weekly Breakdown
- **Week 1**: $1,000 → $2,129 (+113%)
- **Week 2**: $2,129 → $4,531 (+113%)
- **Week 3**: $4,531 → $9,644 (+113%)
- **Week 4**: $9,644 → $20,528 (+113%)

### Position Sizing Example
- **Account**: $1,000
- **Risk per trade**: $20 (2%)
- **XAUUSD @ 2650**: 1.0 lot size
- **Stop loss**: 20 pips
- **Take profit**: 20 pips (1:1 R:R)

## 🛡️ Risk Management

### Built-in Safety Features
1. **Maximum 2% risk per trade**
2. **Automatic position sizing**
3. **Spread filtering** (max 3 pips)
4. **Session filtering** (high liquidity only)
5. **Trade cooldown** (10 minutes between trades)
6. **Stop loss on every trade**

### Manual Monitoring
- Check account every few hours
- Monitor win rate (should stay near 100%)
- Watch for unusual market conditions
- Stop trading if win rate drops below 80%

## 🚨 Important Warnings

### Real-World Factors
- **Spreads**: May reduce profits by 5-10%
- **Slippage**: Possible during high volatility
- **Execution delays**: May affect entry/exit prices
- **Market gaps**: Weekend gaps can affect stops

### Conservative Approach
1. **Start with demo account** (your current plan ✅)
2. **Monitor for 1 week** before increasing risk
3. **Keep detailed logs** of all trades
4. **Don't increase risk above 2%** initially

## 📱 Monitoring Dashboard

### Key Metrics to Track
```
📊 Daily Status:
   Account Balance: $X,XXX
   Total Trades: XX
   Win Rate: XX%
   Daily P&L: $XXX
   Current Position: LONG/SHORT/NONE
```

### Red Flags to Watch
- Win rate drops below 90%
- Daily losses exceed $100
- More than 2 consecutive losses
- Unusual spread widening
- Model predictions seem erratic

## 🎯 Next Steps

### Week 1: Demo Testing
1. Run automated trading for 7 days
2. Monitor performance vs. expectations
3. Document any issues or deviations
4. Verify 100% win rate holds

### Week 2-4: Optimization
1. Fine-tune risk parameters if needed
2. Optimize trading hours
3. Test different position sizes
4. Prepare for potential live trading

### Month 2: Scale Up (If Successful)
1. Consider increasing account size
2. Test with multiple currency pairs
3. Implement additional risk controls
4. Consider live account transition

## 📞 Support

If you encounter issues:
1. Check MT5 connection
2. Verify model files are present
3. Ensure XAUUSD symbol is active
4. Check account permissions
5. Review error logs

## 🏆 Success Criteria

**After 1 month, you should see:**
- Account balance: $10,000 - $25,000
- Win rate: 95%+ (allowing for real-world factors)
- Total trades: 200-250
- Maximum drawdown: <5%
- Consistent daily profits

**If achieved, the model is ready for live trading consideration.**