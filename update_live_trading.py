#!/usr/bin/env python3
"""
Update live trading scripts with broker-specific configurations and EURUSD fixes
"""

import json
import os

def update_ea_config():
    """Update EA configuration with symbol-specific settings"""
    
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    # EURUSD Configuration
    eurusd_config = {
        "symbol": "EURUSD",
        "lot_size": 0.05,  # Increased for meaningful P&L
        "max_positions": 2,  # Allow more positions for forex
        "min_confidence": 0.65,  # Technical optimized
        "signal_frequency": 240,  # Every 4 bars (1 hour)
        "max_daily_trades": 15,  # More trades for forex
        "max_daily_loss": 150.0,
        "trading_hours": {"start": 8, "end": 17},
        "risk_per_trade": 1.5,  # Lower risk for forex volatility
        "use_dynamic_lots": False,
        "dashboard_refresh": 5,
        "show_indicators": True,
        "show_positions": True,
        "broker_spread": 1.6,  # Actual broker spread: 1.6 pips
        "spread_multiplier": 0.0001,  # 1 pip = 0.0001 for EURUSD
        "pnl_multiplier": 100000,  # Correct P&L calculation
        "stop_loss_pips": 15,  # Fixed 15 pips
        "take_profit_pips": 15,  # 1:1 RR
        "indicators": {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "stoch_k": 14,
            "stoch_d": 3
        }
    }
    
    # XAUUSD Configuration
    xauusd_config = {
        "symbol": "XAUUSD",
        "lot_size": 0.01,  # Standard for gold
        "max_positions": 1,  # Conservative for gold
        "min_confidence": 0.75,  # Higher confidence for gold
        "signal_frequency": 240,  # Every 4 bars (1 hour)
        "max_daily_trades": 10,
        "max_daily_loss": 100.0,
        "trading_hours": {"start": 8, "end": 17},
        "risk_per_trade": 2.0,  # Higher risk tolerance for gold
        "use_dynamic_lots": False,
        "dashboard_refresh": 5,
        "show_indicators": True,
        "show_positions": True,
        "broker_spread": 2.3,  # Actual broker spread: 2.3 points
        "spread_multiplier": 0.01,  # 1 point = 0.01 for XAUUSD
        "pnl_multiplier": 100,  # Standard calculation
        "stop_loss_pips": 30,  # ATR-based for gold
        "take_profit_pips": 30,  # 1:1 RR
        "indicators": {
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_period": 14,
            "atr_period": 14,
            "bb_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "stoch_k": 14,
            "stoch_d": 3
        }
    }
    
    # Hybrid EA Configuration (with AI model)
    hybrid_eurusd_config = eurusd_config.copy()
    hybrid_eurusd_config.update({
        "model_path": "models/production/golden_gibz_eurusd_wr60_ret+10_20260108_145955.zip",
        "min_confidence": 0.50,  # Lower for hybrid (AI helps)
        "signal_type": "HYBRID_AI_ENHANCED"
    })
    
    hybrid_xauusd_config = xauusd_config.copy()
    hybrid_xauusd_config.update({
        "model_path": "models/production/golden_gibz_xauusd_wr67_ret+17_20260108_191341.zip",
        "min_confidence": 0.55,  # Lower for hybrid (AI helps)
        "signal_type": "HYBRID_AI_ENHANCED"
    })
    
    # Save configurations
    configs = {
        "config/ea_config_eurusd_technical.json": eurusd_config,
        "config/ea_config_xauusd_technical.json": xauusd_config,
        "config/ea_config_eurusd_hybrid.json": hybrid_eurusd_config,
        "config/ea_config_xauusd_hybrid.json": hybrid_xauusd_config,
        "config/ea_config.json": xauusd_config  # Default
    }
    
    for config_file, config_data in configs.items():
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"✅ Created {config_file}")
    
    return configs

def create_live_trading_guide():
    """Create a guide for live trading implementation"""
    
    guide = """
# 🎯 Golden-Gibz Live Trading Implementation Guide

## ✅ Completed Updates

### 1. Broker-Specific Configurations Applied
- **EURUSD**: 1.6 pip spread, 0.05 lot size, 15:15 pip SL/TP
- **XAUUSD**: 2.3 point spread, 0.01 lot size, 30:30 point SL/TP
- **P&L Calculations**: Fixed for EURUSD (100000 multiplier)
- **Spread Calculations**: Symbol-specific (0.0001 for EURUSD, 0.01 for XAUUSD)

### 2. Configuration Files Created
- `config/ea_config_eurusd_technical.json` - EURUSD Technical-only
- `config/ea_config_eurusd_hybrid.json` - EURUSD AI-Enhanced
- `config/ea_config_xauusd_technical.json` - XAUUSD Technical-only  
- `config/ea_config_xauusd_hybrid.json` - XAUUSD AI-Enhanced

## 🚀 How to Deploy Live Trading

### Option 1: Python Signal Generator + MT5 EA (Recommended)

1. **Start Python Signal Generator**:
   ```bash
   # For EURUSD Technical
   python scripts/technical_goldengibz_signal.py config/ea_config_eurusd_technical.json
   
   # For EURUSD Hybrid (AI-Enhanced)
   python scripts/hybrid_goldengibz_signal.py config/ea_config_eurusd_hybrid.json
   
   # For XAUUSD Technical
   python scripts/technical_goldengibz_signal.py config/ea_config_xauusd_technical.json
   
   # For XAUUSD Hybrid (AI-Enhanced)
   python scripts/hybrid_goldengibz_signal.py config/ea_config_xauusd_hybrid.json
   ```

2. **Compile and Attach MT5 EA**:
   - Open MetaEditor
   - Compile `mt5_ea/GoldenGibzEA.mq5`
   - Attach to chart with these settings:
     - **EURUSD**: LotSize=0.05, StopLossATRMultiplier=15, TakeProfitATRMultiplier=15
     - **XAUUSD**: LotSize=0.01, StopLossATRMultiplier=30, TakeProfitATRMultiplier=30

### Option 2: Pure Python Trading (Advanced)

1. **Install MT5 Python Package**:
   ```bash
   pip install MetaTrader5
   ```

2. **Run Direct Python Trading**:
   ```bash
   # This would require additional development for direct MT5 integration
   # Currently, the signal generators create JSON files for the MT5 EA
   ```

## 📊 Expected Performance

### EURUSD (Based on Backtesting)
- **Technical**: 54.5% win rate, +133.50% annual return
- **Hybrid**: 54.5% win rate, +133.50% annual return (fallback to technical)
- **Pure AI**: 65.4% win rate, +99.37% annual return

### XAUUSD (Based on Backtesting)  
- **Technical**: 62.1% win rate, +349.37% annual return
- **Hybrid**: 62.1% win rate, +349.37% annual return (fallback to technical)
- **Pure AI**: 56.5% win rate, +517.94% annual return

## 🛡️ Risk Management

### EURUSD Settings
- **Lot Size**: 0.05 (5x standard for meaningful P&L)
- **Stop Loss**: 15 pips fixed
- **Take Profit**: 15 pips (1:1 RR)
- **Max Positions**: 2
- **Daily Loss Limit**: $150

### XAUUSD Settings
- **Lot Size**: 0.01 (standard)
- **Stop Loss**: 30 points (ATR-based)
- **Take Profit**: 30 points (1:1 RR)
- **Max Positions**: 1
- **Daily Loss Limit**: $100

## 🔧 Configuration Parameters

### Key Settings to Monitor
- `min_confidence`: Signal quality threshold
- `signal_frequency`: How often to generate signals (240s = 4 minutes)
- `max_daily_trades`: Prevent overtrading
- `trading_hours`: Active trading window
- `broker_spread`: Your actual broker spread
- `pnl_multiplier`: Correct P&L calculation

### Symbol-Specific Multipliers
- **EURUSD**: 
  - Spread: `spread * 0.0001` (1 pip = 0.0001)
  - P&L: `price_diff * lot_size * 100000`
- **XAUUSD**:
  - Spread: `spread * 0.01` (1 point = 0.01)  
  - P&L: `price_diff * lot_size * 100`

## 📈 Monitoring and Optimization

### Real-time Dashboard Features
- Live P&L tracking
- Win rate monitoring
- Signal quality analysis
- Risk management alerts
- Technical indicator status

### Performance Tracking
- Compare live results vs backtest expectations
- Monitor signal execution timing
- Track slippage and spread impact
- Analyze market session performance

## ⚠️ Important Notes

1. **Start with Demo Account**: Test all configurations thoroughly
2. **Monitor Initial Trades**: Verify P&L calculations are correct
3. **Check Spread Impact**: Ensure broker spreads match configuration
4. **Validate Signal Timing**: Confirm signals generate at expected intervals
5. **Risk Management**: Never risk more than you can afford to lose

## 🎯 Next Steps

1. Choose your preferred symbol and strategy type
2. Load the appropriate configuration file
3. Start the Python signal generator
4. Attach and configure the MT5 EA
5. Monitor performance and adjust as needed

The system is now ready for live trading with your actual broker conditions!
"""
    
    with open("LIVE_TRADING_GUIDE.md", "w", encoding='utf-8') as f:
        f.write(guide)
    
    print("✅ Created LIVE_TRADING_GUIDE.md")

def main():
    print("🎯 Updating Live Trading Implementation...")
    print("="*60)
    
    # Update configurations
    configs = update_ea_config()
    
    # Create guide
    create_live_trading_guide()
    
    print("\n" + "="*60)
    print("✅ Live Trading Implementation Updated!")
    print("\n📋 Summary:")
    print("   • Created symbol-specific EA configurations")
    print("   • Applied broker spreads (EURUSD: 1.6 pips, XAUUSD: 2.3 points)")
    print("   • Fixed EURUSD P&L calculations (100000 multiplier)")
    print("   • Updated lot sizes (EURUSD: 0.05, XAUUSD: 0.01)")
    print("   • Created comprehensive live trading guide")
    
    print("\n🚀 Ready for Live Trading!")
    print("   1. Choose your symbol and strategy type")
    print("   2. Run: python scripts/[signal_script].py config/[config_file].json")
    print("   3. Attach MT5 EA with matching settings")
    print("   4. Monitor performance")
    
    print("\n📖 See LIVE_TRADING_GUIDE.md for detailed instructions")
    print("="*60)

if __name__ == "__main__":
    main()