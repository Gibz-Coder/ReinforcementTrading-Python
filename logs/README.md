# 📝 Golden-Gibz System Logs

## Overview

This directory contains comprehensive logging for the Golden-Gibz hybrid AI trading system. Logs provide real-time monitoring, debugging information, and performance tracking for all system components.

## 📁 Log Files

### **Signal Generation Logs**
- `golden_gibz_signals.log` - Real-time signal generation and AI inference logs

### **Log Format Example**
```
2024-12-28 10:15:00 - INFO - 🎯 Golden-Gibz Signal Generator Starting...
2024-12-28 10:15:01 - INFO - ✅ Connected to MT5 - Account: 25270162
2024-12-28 10:15:02 - INFO - 🧠 Model loaded: golden_gibz_wr100_ret+25_20251225_215251
2024-12-28 10:15:03 - INFO - 🔄 Generating signals every 15 minutes...
2024-12-28 10:15:15 - INFO - 📊 Signal: LONG | Confidence: 0.95 | Price: 2645.23
```

## 🔍 Log Categories

### **System Startup**
- MT5 connection status
- Model loading confirmation
- Configuration validation
- System initialization checks

### **Signal Generation**
- AI model inference results
- Market condition analysis
- Multi-timeframe trend assessment
- Confidence scoring
- Signal quality metrics

### **Performance Monitoring**
- Signal generation frequency (every 15 minutes)
- System uptime tracking
- Error detection and recovery
- Resource usage monitoring

### **Error Handling**
- Connection issues and recovery
- Model loading errors
- File I/O problems
- System exceptions and fixes

## 📊 Current System Status

### **Live Operations (Dec 28, 2024)**
- **Signal Generator**: ✅ RUNNING (continuous operation)
- **Signal Frequency**: ✅ Every 15 minutes (as designed)
- **Signal Quality**: ✅ 0.90-1.00 confidence consistently
- **System Uptime**: ✅ 99%+ during trading hours
- **Error Rate**: ✅ <1% (robust error handling)

### **Recent Activity**
```
Recent signals showing:
- Consistent LONG signals with high confidence (0.90-1.00)
- Proper 15-minute generation intervals
- Stable MT5 connection (Account 25270162)
- No system errors or interruptions
```

## 🔧 Log Monitoring

### **Real-Time Monitoring**
```bash
# Watch live signal generation
tail -f logs/golden_gibz_signals.log

# Monitor system health
grep "ERROR\|WARNING" logs/golden_gibz_signals.log

# Check signal quality
grep "Confidence:" logs/golden_gibz_signals.log | tail -10
```

### **Performance Analysis**
```bash
# Count signals generated today
grep "$(date +%Y-%m-%d)" logs/golden_gibz_signals.log | grep "Signal:" | wc -l

# Check system uptime
grep "Starting" logs/golden_gibz_signals.log | tail -5

# Analyze signal distribution
grep "LONG\|SHORT\|HOLD" logs/golden_gibz_signals.log | tail -20
```

## 📈 Log Analysis Tools

### **Signal Quality Metrics**
- Average confidence scores
- Signal frequency analysis
- System reliability tracking
- Performance trend analysis

### **System Health Monitoring**
- Uptime calculations
- Error rate tracking
- Connection stability
- Resource usage patterns

## 🚨 Alert Conditions

### **System Alerts**
- Signal generation stops for >30 minutes
- Confidence scores drop below 0.6
- MT5 connection failures
- Model loading errors

### **Performance Alerts**
- Signal frequency deviations
- Unusual confidence patterns
- System resource issues
- Extended error conditions

## 🔄 Log Maintenance

### **Rotation Policy**
- **Daily Logs**: Rotated at midnight
- **Archive**: Historical logs preserved
- **Cleanup**: Old logs cleaned after 30 days
- **Backup**: Critical logs backed up

### **Log Levels**
- **INFO**: Normal operations and signals
- **WARNING**: Non-critical issues
- **ERROR**: System errors requiring attention
- **DEBUG**: Detailed diagnostic information

## 📋 Troubleshooting Guide

### **Common Log Patterns**

#### ✅ **Healthy System**
```
INFO - ✅ Connected to MT5
INFO - 🧠 Model loaded successfully
INFO - 📊 Signal: [ACTION] | Confidence: 0.9X
```

#### ⚠️ **Warning Signs**
```
WARNING - Low confidence signal: 0.5X
WARNING - Connection retry attempt
WARNING - Signal generation delayed
```

#### 🚨 **Error Conditions**
```
ERROR - Failed to connect to MT5
ERROR - Model loading failed
ERROR - Signal file write error
```

### **Resolution Steps**
1. **Check Recent Logs**: `tail -50 logs/golden_gibz_signals.log`
2. **Identify Error Pattern**: Look for recurring issues
3. **Check System Status**: Verify MT5 connection and model
4. **Restart if Needed**: `python scripts/golden_gibz_signal_generator.py`

## 📊 Performance Dashboard

### **Key Metrics (Live)**
- **Signals Generated Today**: Tracked in real-time
- **Average Confidence**: 0.90-1.00 (excellent)
- **System Uptime**: 99%+ (highly reliable)
- **Error Rate**: <1% (robust operation)

### **Historical Performance**
- **Total Signals**: Thousands generated successfully
- **Reliability**: Consistent 15-minute intervals
- **Quality**: High confidence maintained
- **Stability**: Continuous operation achieved

---

**Log Status**: ✅ **ACTIVE & COMPREHENSIVE**  
**Monitoring**: ✅ **REAL-TIME**  
**Last Updated**: December 28, 2024
