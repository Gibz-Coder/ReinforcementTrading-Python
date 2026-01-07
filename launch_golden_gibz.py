#!/usr/bin/env python3
"""
🤖 Golden Gibz Trading System - Main Launcher
Launch the native desktop application
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Launch Golden Gibz Native Desktop Application"""
    try:
        from golden_gibz_native_app import GoldenGibzNativeApp
        
        print("🤖 Starting Golden Gibz Trading System...")
        print("📱 Native Desktop Application")
        print("=" * 50)
        
        app = GoldenGibzNativeApp()
        app.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()