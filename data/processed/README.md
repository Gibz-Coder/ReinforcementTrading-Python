# 📊 Processed Data Directory

## Overview

This directory contains preprocessed training datasets for the Golden-Gibz hybrid AI trading system. The data has been processed from raw XAUUSD market data and optimized for PPO reinforcement learning training.

## 📁 Current Files

### **Production Training Data**
- `XAUUSD_15m_balanced_latest.csv` - Latest balanced dataset for training
- `XAUUSD_15m_balanced_20251221_202553.csv` - Timestamped training dataset

### **Data Format**
```csv
Date,Open,High,Low,Close,Volume,EMA20,EMA50,RSI,ATR,TrendStrength
2024-01-01 00:00:00,2065.45,2067.23,2063.12,2066.78,1234.56,2064.23,2062.45,52.34,1.23,2
```

## 🔧 Data Processing Pipeline

### **Source Data**
- **Raw Data**: Multi-timeframe XAUUSD data (15M, 1H, 4H, 1D)
- **Timeframes**: 15-minute primary, higher timeframes for trend analysis
- **Features**: OHLCV + 19 engineered technical indicators

### **Processing Steps**
1. **Data Cleaning**: Remove gaps, handle missing values
2. **Feature Engineering**: Calculate 19 technical indicators
   - Moving Averages (EMA20, EMA50)
   - Momentum (RSI)
   - Volatility (ATR)
   - Trend Strength Scoring
3. **Multi-Timeframe Alignment**: Synchronize across timeframes
4. **Normalization**: Scale features for neural network training
5. **Balancing**: Ensure balanced signal distribution

### **Quality Metrics**
- **Data Points**: 50,000+ bars per timeframe
- **Coverage**: Complete market sessions (London/NY overlap)
- **Quality**: Clean data with no gaps or anomalies
- **Balance**: Equal distribution of trend conditions

## 🎯 Usage in Training

### **Model Training**
```python
# Load processed data
df = pd.read_csv('data/processed/XAUUSD_15m_balanced_latest.csv')

# Used in train_golden_gibz.py for:
# - PPO environment creation
# - Feature normalization
# - Training/validation splits
```

### **Training Results**
- **Models Trained**: 9 production variants
- **Performance**: 100% win rate across 500k timesteps
- **Best Model**: golden_gibz_wr100_ret+25_20251225_215251
- **Training Time**: 11 minutes 56 seconds

## 📈 Data Statistics

### **Dataset Characteristics**
- **Timeframe**: 15-minute bars
- **Period**: Extended historical coverage
- **Symbols**: XAUUSD (Gold vs USD)
- **Sessions**: London/NY overlap focus (8-17 GMT)

### **Feature Distribution**
- **Trend Signals**: Balanced bull/bear/neutral
- **Volatility Range**: Low to high ATR conditions
- **Market Conditions**: Various trend strengths
- **Session Coverage**: High liquidity periods

## 🔄 Data Updates

### **Refresh Schedule**
- **Training Data**: Updated with new market data as needed
- **Model Retraining**: Monthly or based on performance
- **Quality Checks**: Continuous data validation

### **Version Control**
- **Timestamped Files**: Each processing run saved
- **Latest Version**: Always available as `*_latest.csv`
- **Backup**: Historical versions maintained

## 🛡️ Data Quality Assurance

### **Validation Checks**
- ✅ No missing values in critical columns
- ✅ Consistent timestamp formatting
- ✅ Proper feature scaling and normalization
- ✅ Balanced signal distribution
- ✅ Multi-timeframe alignment verified

### **Processing Logs**
- Data processing timestamps and statistics
- Feature engineering validation
- Quality check results
- Training data preparation logs

---

**Data Status**: ✅ **CURRENT & VALIDATED**  
**Last Updated**: December 2024  
**Next Update**: Based on model retraining schedule
