#!/usr/bin/env python3
"""
Quick test to check MT5 data retrieval
"""

import MetaTrader5 as mt5
from datetime import datetime

def test_mt5_connection():
    """Test MT5 connection and data retrieval"""
    
    print("🔍 Testing MT5 Data Connection...")
    
    # Initialize MT5
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return False
    
    # Check connection
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ Failed to get account info")
        return False
    
    print(f"✅ Connected to account: {account_info.login}")
    
    # Test symbol info
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        # Try to find available symbols
        symbols = mt5.symbols_get()
        gold_symbols = [s.name for s in symbols if 'XAU' in s.name or 'GOLD' in s.name]
        print(f"Available gold symbols: {gold_symbols}")
        return False
    
    print(f"✅ Symbol {symbol} found: {symbol_info.description}")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select symbol {symbol}")
        return False
    
    print(f"✅ Symbol {symbol} selected")
    
    # Test data retrieval for different timeframes
    timeframes = {
        '15m': mt5.TIMEFRAME_M15,
        '1h': mt5.TIMEFRAME_H1,
        '4h': mt5.TIMEFRAME_H4,
        '1d': mt5.TIMEFRAME_D1
    }
    
    for tf_name, tf_const in timeframes.items():
        print(f"\n📊 Testing {tf_name} data...")
        
        # Try to get recent data
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, 100)
        
        if rates is None or len(rates) == 0:
            print(f"❌ {tf_name}: No data available")
            
            # Try alternative method
            rates = mt5.copy_rates_from(symbol, tf_const, datetime.now(), 100)
            if rates is None or len(rates) == 0:
                print(f"❌ {tf_name}: Alternative method also failed")
            else:
                print(f"✅ {tf_name}: {len(rates)} bars (alternative method)")
        else:
            print(f"✅ {tf_name}: {len(rates)} bars")
            print(f"   Latest: {rates[-1]['time']} - Close: {rates[-1]['close']}")
    
    mt5.shutdown()
    return True

if __name__ == "__main__":
    test_mt5_connection()