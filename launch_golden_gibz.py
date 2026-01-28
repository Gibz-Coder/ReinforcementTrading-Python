#!/usr/bin/env python3
"""
Launch script for Golden Gibz Trading System
Redirects to the main native app
"""

import subprocess
import sys

if __name__ == "__main__":
    try:
        # Launch the main native app
        subprocess.run([sys.executable, "golden_gibz_native_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error launching Golden Gibz: {e}")
        print("Falling back to console mode...")
        # Fall back to console mode if GUI fails
        subprocess.run([sys.executable, "scripts/technical_goldengibz_signal.py"])
    except KeyboardInterrupt:
        print("\nShutting down Golden Gibz...")
    except Exception as e:
        print(f"Unexpected error: {e}")