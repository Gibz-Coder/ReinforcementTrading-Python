#!/usr/bin/env python3
"""
Check what path the EA is actually using
"""

import MetaTrader5 as mt5
import os

def check_ea_paths():
    """Check EA terminal data path"""
    
    if not mt5.initialize():
        print("❌ Failed to initialize MT5")
        return
    
    # Get terminal info like EA does
    terminal_data_path = mt5.terminal_info().data_path
    ea_signal_path = os.path.join(terminal_data_path, "MQL5", "Files", "signals.json")
    
    print(f"🔍 EA Terminal Data Path Analysis:")
    print(f"Terminal Data Path: {terminal_data_path}")
    print(f"EA Signal File Path: {ea_signal_path}")
    print(f"File Exists: {os.path.exists(ea_signal_path)}")
    
    if os.path.exists(ea_signal_path):
        print(f"✅ EA can find signal file")
        # Check file content
        try:
            with open(ea_signal_path, 'r') as f:
                content = f.read()
            print(f"File size: {len(content)} characters")
            if len(content) > 10:
                print("✅ File has content")
            else:
                print("❌ File is empty or too small")
        except Exception as e:
            print(f"❌ Error reading file: {e}")
    else:
        print(f"❌ EA cannot find signal file at expected path")
        print(f"Need to copy signal to: {ea_signal_path}")
    
    mt5.shutdown()

if __name__ == "__main__":
    check_ea_paths()