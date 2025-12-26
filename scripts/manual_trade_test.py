#!/usr/bin/env python3
"""
Manual trade test - Execute what the EA should be doing
"""

import MetaTrader5 as mt5
import json

def manual_trade_test():
    """Test manual trade execution like EA should do"""
    
    print("🔍 Manual Trade Test (Simulating EA Behavior)")
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Read signal like EA does
    signal_path = r"C:\Program Files\Tickmill MT5 Terminal\MQL5\Files\signals.json"
    
    try:
        with open(signal_path, 'r') as f:
            signal = json.loads(f.read())
        
        print(f"✅ Signal loaded:")
        print(f"   Action: {signal['action']} ({signal['action_name']})")
        print(f"   Confidence: {signal['confidence']}")
        
        # Check if we should trade
        if signal['action'] == 1 and signal['confidence'] >= 0.6:
            print(f"✅ Conditions met for LONG trade")
            
            # Get current prices
            symbol = "XAUUSD"
            symbol_info = mt5.symbol_info(symbol)
            ask = symbol_info.ask
            
            # Calculate stops like EA does
            atr_value = signal['risk_management']['atr_value']
            stop_distance = atr_value * 2.0  # EA uses 2x ATR
            target_distance = atr_value * 2.0
            
            sl = ask - stop_distance
            tp = ask + target_distance
            lot_size = 0.01
            
            print(f"📊 Trade Parameters:")
            print(f"   Symbol: {symbol}")
            print(f"   Price: {ask}")
            print(f"   Lot Size: {lot_size}")
            print(f"   Stop Loss: {sl:.2f} (Distance: {stop_distance:.2f})")
            print(f"   Take Profit: {tp:.2f} (Distance: {target_distance:.2f})")
            
            # Create trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": ask,
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": 12345,
                "comment": "Golden-Gibz Manual Test",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            print(f"\n🚀 Executing LONG trade...")
            
            # Send trade request
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Trade executed successfully!")
                print(f"   Order: {result.order}")
                print(f"   Deal: {result.deal}")
                print(f"   Price: {result.price}")
            else:
                print(f"❌ Trade failed!")
                print(f"   Return Code: {result.retcode}")
                print(f"   Comment: {result.comment}")
                
                # Decode error
                if result.retcode == 10004:
                    print("   Error: Requote")
                elif result.retcode == 10006:
                    print("   Error: Request rejected")
                elif result.retcode == 10007:
                    print("   Error: Request canceled by trader")
                elif result.retcode == 10008:
                    print("   Error: Order placed")
                elif result.retcode == 10009:
                    print("   Error: Request completed")
                elif result.retcode == 10010:
                    print("   Error: Only part of the request was completed")
                elif result.retcode == 10011:
                    print("   Error: Request processing error")
                elif result.retcode == 10012:
                    print("   Error: Request canceled by timeout")
                elif result.retcode == 10013:
                    print("   Error: Invalid request")
                elif result.retcode == 10014:
                    print("   Error: Invalid volume in the request")
                elif result.retcode == 10015:
                    print("   Error: Invalid price in the request")
                elif result.retcode == 10016:
                    print("   Error: Invalid stops in the request")
                elif result.retcode == 10017:
                    print("   Error: Trade is disabled")
                elif result.retcode == 10018:
                    print("   Error: Market is closed")
                elif result.retcode == 10019:
                    print("   Error: There is not enough money to complete the request")
                elif result.retcode == 10020:
                    print("   Error: Prices changed")
                elif result.retcode == 10021:
                    print("   Error: There are no quotes to process the request")
                elif result.retcode == 10022:
                    print("   Error: Invalid order expiration date in the request")
                elif result.retcode == 10023:
                    print("   Error: Order state changed")
                elif result.retcode == 10024:
                    print("   Error: Too frequent requests")
                elif result.retcode == 10025:
                    print("   Error: No changes in request")
                elif result.retcode == 10026:
                    print("   Error: Autotrading disabled by server")
                elif result.retcode == 10027:
                    print("   Error: Autotrading disabled by client terminal")
                elif result.retcode == 10028:
                    print("   Error: Request locked for processing")
                elif result.retcode == 10029:
                    print("   Error: Order or position frozen")
                elif result.retcode == 10030:
                    print("   Error: Invalid order filling type")
                else:
                    print(f"   Error: Unknown error code {result.retcode}")
        
        else:
            print(f"⚠️ No trade conditions met")
            if signal['action'] != 1:
                print(f"   Action is {signal['action']} (not LONG)")
            if signal['confidence'] < 0.6:
                print(f"   Confidence {signal['confidence']} < 0.6")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    mt5.shutdown()

if __name__ == "__main__":
    manual_trade_test()