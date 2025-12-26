#!/usr/bin/env python3
"""
EA Debug Helper - Check all possible reasons why EA isn't trading
"""

import MetaTrader5 as mt5
import json
import os
from datetime import datetime

def debug_ea_issues():
    """Debug all possible EA trading issues"""
    
    print("🔍 EA Trading Debug Analysis")
    print("=" * 50)
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # 1. Check account info
    account = mt5.account_info()
    print(f"\n📊 Account Information:")
    print(f"   Login: {account.login}")
    print(f"   Balance: ${account.balance:.2f}")
    print(f"   Equity: ${account.equity:.2f}")
    print(f"   Margin Free: ${account.margin_free:.2f}")
    print(f"   Trade Allowed: {account.trade_allowed}")
    print(f"   Trade Expert: {account.trade_expert}")
    
    # 2. Check symbol info
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    print(f"\n📈 Symbol Information ({symbol}):")
    print(f"   Trade Mode: {symbol_info.trade_mode}")
    print(f"   Min Volume: {symbol_info.volume_min}")
    print(f"   Max Volume: {symbol_info.volume_max}")
    print(f"   Volume Step: {symbol_info.volume_step}")
    print(f"   Spread: {symbol_info.spread}")
    print(f"   Bid: {symbol_info.bid}")
    print(f"   Ask: {symbol_info.ask}")
    
    # 3. Check current positions
    positions = mt5.positions_get(symbol=symbol)
    print(f"\n📍 Current Positions:")
    if positions:
        for pos in positions:
            print(f"   Position: {pos.type} {pos.volume} lots @ {pos.price_open}")
    else:
        print("   No open positions")
    
    # 4. Check signal file
    signal_path = r"C:\Program Files\Tickmill MT5 Terminal\MQL5\Files\signals.json"
    print(f"\n📄 Signal File Analysis:")
    print(f"   Path: {signal_path}")
    
    if os.path.exists(signal_path):
        print("   ✅ File exists")
        try:
            with open(signal_path, 'r') as f:
                signal = json.loads(f.read())
            
            timestamp = signal.get('timestamp')
            action = signal.get('action')
            confidence = signal.get('confidence')
            
            print(f"   Timestamp: {timestamp}")
            print(f"   Action: {action} ({signal.get('action_name')})")
            print(f"   Confidence: {confidence}")
            
            # Check signal age
            if timestamp:
                signal_time = datetime.fromisoformat(timestamp)
                age = (datetime.now() - signal_time).total_seconds()
                print(f"   Age: {age:.1f} seconds")
                
                if age > 1800:  # 30 minutes
                    print("   ⚠️ Signal is too old (>30 min)")
                else:
                    print("   ✅ Signal is fresh")
            
            # Check trading conditions
            if confidence >= 0.6:
                print("   ✅ Confidence above threshold")
            else:
                print("   ❌ Confidence below threshold")
                
        except Exception as e:
            print(f"   ❌ Error reading signal: {e}")
    else:
        print("   ❌ Signal file not found")
    
    # 5. Check trading hours (current time)
    now = datetime.now()
    print(f"\n⏰ Trading Time Analysis:")
    print(f"   Current Time: {now.strftime('%H:%M:%S')}")
    print(f"   Day of Week: {now.weekday()} (0=Monday, 6=Sunday)")
    
    # EA default settings: StartHour=0, EndHour=24
    if 0 <= now.hour < 24:
        print("   ✅ Within trading hours (0-24)")
    else:
        print("   ❌ Outside trading hours")
    
    # 6. Check lot size calculation
    lot_size = 0.01  # EA default
    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    
    print(f"\n💰 Lot Size Analysis:")
    print(f"   EA Lot Size: {lot_size}")
    print(f"   Min Allowed: {min_lot}")
    print(f"   Max Allowed: {max_lot}")
    
    if min_lot <= lot_size <= max_lot:
        print("   ✅ Lot size is valid")
    else:
        print("   ❌ Lot size is invalid")
    
    # 7. Check margin requirements
    margin_required = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lot_size, symbol_info.ask)
    print(f"\n💳 Margin Analysis:")
    print(f"   Required Margin: ${margin_required:.2f}")
    print(f"   Free Margin: ${account.margin_free:.2f}")
    
    if margin_required and account.margin_free >= margin_required:
        print("   ✅ Sufficient margin")
    else:
        print("   ❌ Insufficient margin")
    
    mt5.shutdown()
    
    print(f"\n🎯 Summary:")
    print("If all checks show ✅, the EA should be trading.")
    print("Look for any ❌ items above - those are blocking trade execution.")

if __name__ == "__main__":
    debug_ea_issues()