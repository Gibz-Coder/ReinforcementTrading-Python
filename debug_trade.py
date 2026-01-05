#!/usr/bin/env python3
"""
Debug MT5 Trading Issues
"""

import MetaTrader5 as mt5

def check_account_and_symbol():
    """Check account permissions and symbol info."""
    
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return
    
    # Account info
    account = mt5.account_info()
    print(f"📊 ACCOUNT INFO:")
    print(f"   Login: {account.login}")
    print(f"   Server: {account.server}")
    print(f"   Balance: ${account.balance}")
    print(f"   Equity: ${account.equity}")
    print(f"   Margin Free: ${account.margin_free}")
    print(f"   Trade Allowed: {account.trade_allowed}")
    print(f"   Trade Expert: {account.trade_expert}")
    
    # Symbol info
    symbol = "XAUUSD"
    symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info is None:
        print(f"❌ Symbol {symbol} not found")
        return
    
    print(f"\n📈 SYMBOL INFO ({symbol}):")
    print(f"   Visible: {symbol_info.visible}")
    print(f"   Trade Mode: {symbol_info.trade_mode}")
    print(f"   Min Volume: {symbol_info.volume_min}")
    print(f"   Max Volume: {symbol_info.volume_max}")
    print(f"   Volume Step: {symbol_info.volume_step}")
    print(f"   Margin Required: {symbol_info.margin_initial}")
    print(f"   Contract Size: {symbol_info.trade_contract_size}")
    
    # Current tick
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"   Current Bid: {tick.bid}")
        print(f"   Current Ask: {tick.ask}")
        print(f"   Spread: {tick.ask - tick.bid}")
    
    # Test order request
    print(f"\n🧪 TEST ORDER REQUEST:")
    lot_size = 0.01
    
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot_size,
        'type': mt5.ORDER_TYPE_BUY,
        'price': tick.ask,
        'sl': tick.ask - 10.0,
        'tp': tick.ask + 10.0,
        'deviation': 20,
        'magic': 88888,
        'comment': 'Debug test',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC
    }
    
    print(f"   Request: {request}")
    
    # Check order
    result = mt5.order_check(request)
    if result:
        print(f"   Order Check: {result.retcode} - {result.comment}")
        print(f"   Margin: {result.margin}")
        print(f"   Profit: {result.profit}")
    else:
        print(f"   Order Check Failed: {mt5.last_error()}")
    
    # Try to send (comment out if you don't want actual trade)
    # result = mt5.order_send(request)
    # print(f"   Order Send: {result.retcode} - {result.comment}")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_account_and_symbol()