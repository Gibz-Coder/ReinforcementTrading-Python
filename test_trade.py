#!/usr/bin/env python3
"""
Test actual trade execution
"""

import MetaTrader5 as mt5

def test_trade():
    """Test actual trade with different fill types."""
    
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed")
        return
    
    symbol = "XAUUSD"
    tick = mt5.symbol_info_tick(symbol)
    
    if not tick:
        print(f"❌ No tick data")
        return
    
    print(f"Current price: {tick.bid}/{tick.ask}")
    
    # Try different fill types
    fill_types = [
        (mt5.ORDER_FILLING_FOK, "Fill or Kill"),
        (mt5.ORDER_FILLING_IOC, "Immediate or Cancel"), 
        (mt5.ORDER_FILLING_RETURN, "Return")
    ]
    
    for fill_type, name in fill_types:
        print(f"\n🧪 Testing {name} ({fill_type}):")
        
        request = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': 0.01,
            'type': mt5.ORDER_TYPE_BUY,
            'price': tick.ask,
            'sl': tick.ask - 10.0,
            'tp': tick.ask + 10.0,
            'deviation': 20,
            'magic': 88888,
            'comment': f'Test {name}',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': fill_type
        }
        
        result = mt5.order_send(request)
        print(f"   Result: {result.retcode} - {result.comment}")
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"   ✅ SUCCESS! Order: {result.order}, Deal: {result.deal}")
            break
        else:
            print(f"   ❌ Failed: {result.retcode}")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_trade()