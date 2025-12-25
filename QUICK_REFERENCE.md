# 🚀 Golden-Gibz Quick Reference

## ⚡ **Quick Commands**

### **Training**
```bash
# Train Golden-Gibz model (500k timesteps)
python scripts/train_golden_gibz.py --timesteps 500000 --envs 6

# Quick training (100k timesteps)
python scripts/train_golden_gibz.py --timesteps 100000 --envs 4
```

### **Deployment**
```bash
# Start signal generator
python scripts/golden_gibz_signal_generator.py

# Monitor signals
tail -f logs/golden_gibz_signals.log
```

### **Model Management**
```bash
# List available models
ls -la models/production/golden_gibz_*

# Check model performance
grep "🏆 PRODUCTION MODEL" logs/training.log
```

## 📊 **Key Performance Metrics**

| Metric | Target | Actual |
|--------|--------|--------|
| **Win Rate** | 85%+ | 100% |
| **Returns** | 15-25% | +25.2% |
| **Confidence** | 60%+ | 60-90% |
| **Trades/Day** | 2-4 | 10-11 |

## 🎯 **Signal Types**

| Signal | Condition | Action |
|--------|-----------|--------|
| **LONG** | 3+ bull timeframes + strong trend + RSI<70 | Buy |
| **SHORT** | 3+ bear timeframes + strong trend + RSI>30 | Sell |
| **HOLD** | Mixed signals or low confidence | Wait |

## 🛡️ **Risk Management**

### **EA Settings**
```mql5
LotSize = 0.01                    // Fixed lot size
MaxRiskPercent = 2.0              // Max risk per trade
StopLossATRMultiplier = 2.0       // Stop loss distance
TakeProfitATRMultiplier = 2.0     // Take profit distance
MinConfidence = 0.6               // Minimum signal confidence
MaxDailyLoss = 100.0              // Daily loss limit ($)
MaxDailyTrades = 10               // Max trades per day
```

### **Safety Limits**
- **Daily Loss**: $100 (configurable)
- **Trade Cooldown**: 5 minutes
- **Session Hours**: 8:00-17:00 GMT
- **Confidence Filter**: 60% minimum

## 📁 **File Locations**

### **Key Files**
```
📊 Models:     models/production/golden_gibz_wr100_ret+25_*
📡 Signals:    mt5_ea/signals.json
📋 Logs:       logs/golden_gibz_signals.log
🤖 EA:         mt5_ea/GoldenGibzEA.mq5
🧠 Training:   scripts/train_golden_gibz.py
```

### **Configuration**
```
Signal Generator: Line 381 - MODEL_PATH
EA Parameters:    MT5 EA inputs
Training Config:  Command line arguments
```

## 🔧 **Troubleshooting**

### **Common Issues**
| Problem | Solution |
|---------|----------|
| No signals | Check market hours (8-17 GMT) |
| Model not found | Verify MODEL_PATH in signal generator |
| EA not trading | Check confidence threshold and balance |
| Signal timeout | Restart signal generator |

### **Quick Fixes**
```bash
# Restart signal generator
Ctrl+C
python scripts/golden_gibz_signal_generator.py

# Check EA status
# Look at MT5 terminal logs and chart comment

# Verify model exists
ls -la models/production/golden_gibz_wr100_ret+25_*
```

## 📈 **Monitoring Checklist**

### **Daily Checks**
- [ ] Signal generator running
- [ ] EA attached to XAUUSD chart
- [ ] Win rate above 80%
- [ ] Daily P&L within limits
- [ ] No error messages in logs

### **Weekly Reviews**
- [ ] Overall performance vs targets
- [ ] Signal quality and confidence
- [ ] Risk management effectiveness
- [ ] System stability and uptime

## 🎯 **Best Practices**

### **Setup**
1. **Always test on demo first**
2. **Start with minimum lot sizes**
3. **Monitor closely for first week**
4. **Keep signal generator running 24/7**
5. **Regular model retraining (monthly)**

### **Risk Management**
1. **Never exceed 2% risk per trade**
2. **Set daily loss limits**
3. **Monitor drawdown periods**
4. **Use proper position sizing**
5. **Keep emergency stop procedures ready**

### **Performance**
1. **Track win rate daily**
2. **Monitor signal confidence**
3. **Analyze losing trades**
4. **Compare with training results**
5. **Document any issues**

## 🚨 **Emergency Procedures**

### **Stop Trading Immediately**
```mql5
// In EA inputs
EnableTrading = false
```

### **System Restart**
```bash
# 1. Stop signal generator (Ctrl+C)
# 2. Remove EA from chart
# 3. Restart signal generator
python scripts/golden_gibz_signal_generator.py
# 4. Reattach EA to chart
```

### **Rollback to v1.x**
```bash
# Use backup system if needed
cp backup_simple_trend_rider/scripts/* scripts/
python mt5_simple_trend_trader.py
```

## 📞 **Support Resources**

### **Documentation**
- `GOLDEN_GIBZ_SETUP.md` - Complete setup guide
- `GOLDEN_GIBZ_RESULTS.md` - Performance analysis
- `MIGRATION_GUIDE.md` - Upgrade instructions
- `CHANGELOG.md` - Version history

### **Log Files**
- `logs/golden_gibz_signals.log` - Signal generation
- MT5 Terminal logs - EA execution
- Training output - Model performance

## 🏆 **Success Metrics**

### **Excellent Performance**
- Win Rate: 90%+
- Daily Return: 3%+
- Confidence: 70%+
- Drawdown: <5%

### **Good Performance**
- Win Rate: 80-90%
- Daily Return: 2-3%
- Confidence: 60-70%
- Drawdown: 5-10%

### **Review Required**
- Win Rate: <80%
- Daily Return: <2%
- Confidence: <60%
- Drawdown: >10%

---

## 🎯 **Quick Start (30 seconds)**

```bash
# 1. Train model (if needed)
python scripts/train_golden_gibz.py --timesteps 100000

# 2. Start signals
python scripts/golden_gibz_signal_generator.py

# 3. In MT5: Attach GoldenGibzEA.mq5 to XAUUSD chart

# 4. Monitor
tail -f logs/golden_gibz_signals.log
```

**You're ready to trade with Golden-Gibz! 🚀**

---

*Quick Reference v2.0.0*  
*Last updated: December 25, 2024*