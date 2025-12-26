#!/usr/bin/env python3
"""
Test if EA can read the signal file by simulating EA behavior
"""

import json
import os
from datetime import datetime

def test_ea_signal_reading():
    """Test signal file reading like the EA does"""
    
    print("🔍 Testing EA Signal File Reading...")
    
    # Test both possible EA paths
    ea_paths = [
        r"C:\Users\Admin\Documents\MT5\MQL5\Files\signals.json",
        r"C:\Users\Admin\AppData\Roaming\MetaQuotes\Terminal\29E91DA909EB4475AB204481D1C2CE7D\MQL5\Files\signals.json"
    ]
    
    for i, path in enumerate(ea_paths, 1):
        print(f"\n📁 Testing path {i}: {path}")
        
        if not os.path.exists(path):
            print(f"❌ File not found")
            continue
        
        try:
            with open(path, 'r') as f:
                content = f.read()
            
            if len(content) < 10:
                print(f"❌ File too small: {len(content)} characters")
                continue
            
            # Parse JSON like EA does
            signal_data = json.loads(content)
            
            print(f"✅ File found and readable")
            print(f"   Timestamp: {signal_data.get('timestamp')}")
            print(f"   Action: {signal_data.get('action')} ({signal_data.get('action_name')})")
            print(f"   Confidence: {signal_data.get('confidence')}")
            
            # Check if signal is fresh (like EA does)
            timestamp_str = signal_data.get('timestamp', '')
            if timestamp_str:
                signal_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                current_time = datetime.now()
                age_seconds = (current_time - signal_time).total_seconds()
                
                print(f"   Age: {age_seconds:.1f} seconds")
                
                if age_seconds > 1800:  # 30 minutes timeout
                    print(f"⚠️ Signal too old (>{1800}s)")
                else:
                    print(f"✅ Signal fresh (<{1800}s)")
            
            # Check trading conditions
            confidence = signal_data.get('confidence', 0)
            if confidence >= 0.6:
                print(f"✅ Confidence above threshold (0.6)")
            else:
                print(f"❌ Confidence below threshold: {confidence} < 0.6")
            
            action = signal_data.get('action', 0)
            if action == 1:
                print(f"✅ LONG signal - should trigger buy")
            elif action == 2:
                print(f"✅ SHORT signal - should trigger sell")
            else:
                print(f"⚠️ HOLD signal - no trade expected")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
        except Exception as e:
            print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    test_ea_signal_reading()