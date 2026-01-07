# 🤖 Golden Gibz Trading System - Desktop Application

A comprehensive desktop application for automated forex trading with AI-enhanced signals, backtesting, and model training capabilities.

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Application Tabs](#-application-tabs)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Support](#-support)

## ✨ Features

### 🎯 **Multi-Tab Interface**
- **Dashboard**: Real-time performance metrics and system status
- **Live Trading**: Start/stop trading with Technical or Hybrid AI modes
- **Backtesting**: Comprehensive backtesting with detailed analytics
- **Model Training**: Train and optimize AI models
- **Configuration**: Complete system configuration management
- **Logs**: Real-time system logs and monitoring
- **About**: System information and documentation links

### 🚀 **Trading Capabilities**
- **Technical-Only Mode**: Pure technical analysis (61.7% win rate, +331.87% return)
- **Hybrid AI-Enhanced Mode**: AI + Technical analysis (62.1% win rate, +285.15% return)
- **Real-time monitoring**: Live trading dashboard with performance metrics
- **Risk management**: Advanced position sizing and risk controls

### 📊 **Analytics & Reporting**
- **Comprehensive backtesting**: Full year validation with detailed metrics
- **Performance tracking**: Win rates, returns, signal quality analysis
- **Export capabilities**: Save results in JSON/CSV formats
- **Visual dashboards**: Real-time performance monitoring

### 🧠 **AI Model Management**
- **Model training**: Train custom AI models on historical data
- **Model optimization**: Hyperparameter tuning and validation
- **Performance comparison**: Compare different model versions
- **Production deployment**: Seamless model updates

## 🛠️ Installation

### Prerequisites
- **Python 3.7+** (Required)
- **MetaTrader 5** (For live trading)
- **Windows 10/11** (Recommended, also works on Linux/Mac)

### Quick Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/Gibz-Coder/ReinforcementTrading-Python.git
   cd ReinforcementTrading-Python
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   
   **Windows:**
   ```bash
   # Double-click Golden_Gibz_App.bat
   # OR run in command prompt:
   Golden_Gibz_App.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x golden_gibz_app.sh
   ./golden_gibz_app.sh
   ```
   
   **Cross-platform:**
   ```bash
   python launch_golden_gibz_app.py
   ```

## 🚀 Quick Start

### 1. **First Launch**
- Run the application using one of the launch methods above
- The application will open with the Dashboard tab active
- Check the system status and performance metrics

### 2. **Configuration Setup**
- Go to the **⚙️ Configuration** tab
- Review and adjust trading parameters:
  - Symbol: XAUUSD (default)
  - Lot size: 0.01 (default)
  - Risk per trade: 2% (default)
  - Confidence threshold: 0.75 (Technical) / 0.55 (Hybrid)
- Click **💾 Save Config** to save your settings

### 3. **Run a Backtest**
- Go to the **🧪 Backtest** tab
- Select system type: Technical or Hybrid AI-Enhanced
- Choose backtest period: 1 Year (recommended)
- Set initial balance: $500 (default)
- Click **🚀 Run Backtest**
- View results in the results panel

### 4. **Start Live Trading**
- Go to the **📈 Live Trading** tab
- Select trading mode: Technical or Hybrid AI-Enhanced
- Click **▶️ Start Trading**
- Monitor progress in the trading log
- Use **⏹️ Stop Trading** to stop when needed

## 📑 Application Tabs

### 📊 **Dashboard Tab**
- **System Status**: Current trading status and mode
- **Performance Metrics**: Latest backtest results and statistics
- **Quick Actions**: Fast access to start trading or run backtests
- **Real-time Updates**: Live performance monitoring

### 📈 **Live Trading Tab**
- **Trading Controls**: Start/stop/pause trading operations
- **Mode Selection**: Choose between Technical and Hybrid AI modes
- **Trading Log**: Real-time trading activity and signals
- **Status Monitoring**: Current positions and account status

### 🧪 **Backtest Tab**
- **Parameter Configuration**: Set backtest period, initial balance, risk settings
- **System Selection**: Choose Technical or Hybrid AI system
- **Results Display**: Detailed performance metrics and statistics
- **Export Options**: Save results to JSON/CSV files

### 🧠 **Train Model Tab**
- **Training Parameters**: Configure epochs, batch size, data source
- **Progress Monitoring**: Real-time training progress and logs
- **Model Management**: Save and load trained models
- **Performance Validation**: Validate model performance

### ⚙️ **Configuration Tab**
- **Trading Settings**: Symbol, lot size, position limits
- **Indicator Parameters**: EMA, RSI, MACD, ADX, Stochastic settings
- **Risk Management**: Stop loss, take profit, drawdown limits
- **Save/Load**: Configuration file management

### 📋 **Logs Tab**
- **System Logs**: All application activities and events
- **Log Filtering**: Filter by log level (INFO, WARNING, ERROR)
- **Export Logs**: Save logs to text files
- **Real-time Updates**: Live log streaming

### ℹ️ **About Tab**
- **System Information**: Version, features, performance stats
- **Documentation Links**: GitHub repository, issues, support
- **Support Options**: Ways to contribute and get help

## ⚙️ Configuration

### Trading Parameters
```json
{
    "symbol": "XAUUSD",
    "lot_size": 0.01,
    "max_positions": 1,
    "min_confidence": 0.75,
    "signal_frequency": 240,
    "max_daily_trades": 10,
    "max_daily_loss": 100.0,
    "risk_per_trade": 2.0
}
```

### Indicator Settings
```json
{
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
```

### Risk Management
- **Risk per Trade**: 2% of account balance (default)
- **Risk-Reward Ratio**: 1:1 (default)
- **Max Drawdown**: 20% (recommended)
- **Stop Loss**: 2% (default)
- **Take Profit**: 2% (default)

## 🔧 Troubleshooting

### Common Issues

#### 1. **Application Won't Start**
- **Check Python Installation**: Ensure Python 3.7+ is installed
- **Verify Dependencies**: Run `pip install -r requirements.txt`
- **Check File Permissions**: Ensure scripts are executable
- **Run from Correct Directory**: Must be in project root

#### 2. **Trading Not Starting**
- **MT5 Connection**: Verify MetaTrader 5 is installed and running
- **Account Settings**: Check login credentials and account balance
- **Configuration**: Verify EA configuration in config/ea_config.json
- **Trading Hours**: Ensure within configured trading hours

#### 3. **Backtest Errors**
- **Data Files**: Ensure historical data exists in data/ directory
- **Date Range**: Check backtest period settings
- **Memory**: Ensure sufficient RAM for large datasets
- **Permissions**: Check file read/write permissions

#### 4. **Model Training Issues**
- **Data Preprocessing**: Verify data is properly formatted
- **Dependencies**: Check ML libraries (tensorflow, sklearn) are installed
- **Disk Space**: Ensure sufficient space for model files
- **GPU Support**: Check CUDA installation for GPU training

### Error Messages

#### "Script not found"
- Ensure all Python scripts are in the correct directories
- Check that scripts/ directory contains the trading scripts

#### "Configuration file not found"
- Run the application once to generate default config
- Check config/ea_config.json exists and is valid JSON

#### "MT5 connection failed"
- Verify MetaTrader 5 is installed and running
- Check login credentials and server settings
- Ensure MT5 allows automated trading

### Getting Help

1. **Check Logs**: Review the Logs tab for detailed error messages
2. **Troubleshooting Guide**: Use Help → Troubleshooting in the menu
3. **Documentation**: Visit the GitHub repository for detailed docs
4. **Report Issues**: Create an issue on GitHub with error details

## 📞 Support

### 🔗 **Links**
- **GitHub Repository**: [ReinforcementTrading-Python](https://github.com/Gibz-Coder/ReinforcementTrading-Python)
- **Documentation**: [Wiki Pages](https://github.com/Gibz-Coder/ReinforcementTrading-Python/wiki)
- **Issues**: [Report Bugs](https://github.com/Gibz-Coder/ReinforcementTrading-Python/issues)

### 💝 **Support Development**
- ⭐ **Star the project** on GitHub
- 🐛 **Report bugs** and issues
- 💡 **Suggest features** and improvements
- 📚 **Contribute** to documentation
- 💰 **Consider a donation** to support development

### 📧 **Contact**
For questions and support, please use the GitHub Issues page or discussions.

---

## 🎯 **Performance Summary**

### **Validated Results (January 7, 2026)**

| System | Win Rate | Total Return | Trades/Year | Signal Quality |
|--------|----------|--------------|-------------|----------------|
| **Technical-Only** | **61.7%** | **+331.87%** | **590** | **82.3%** |
| **Hybrid AI-Enhanced** | **62.1%** | **+285.15%** | **509** | **86.1%** |

### **Recommendation**
- **Use Technical-Only** for maximum profit potential (+331.87% return)
- **Use Hybrid AI-Enhanced** for superior risk management (86.1% signal quality)
- **Both systems** are production-ready with excellent win rates (>61%)

---

**🤖 Golden Gibz Trading System - Professional Automated Trading Made Simple**