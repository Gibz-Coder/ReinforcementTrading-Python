#!/usr/bin/env python3
"""
🤖 Golden Gibz Trading System - Application Launcher
Quick launcher for the Golden Gibz desktop application
"""

import sys
import os
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'tkinter',
        'threading',
        'json',
        'queue',
        'webbrowser'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install missing packages and try again.")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🤖 Golden Gibz Trading System Launcher")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("golden_gibz_app.py"):
        print("❌ Error: golden_gibz_app.py not found!")
        print("Please run this launcher from the project root directory.")
        return 1
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("✅ All dependencies found!")
    
    # Launch the application
    print("🚀 Starting Golden Gibz Trading System...")
    try:
        # Import and run the application
        from golden_gibz_app import main as app_main
        app_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Trying alternative launch method...")
        
        # Alternative: run as subprocess
        try:
            subprocess.run([sys.executable, "golden_gibz_app.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to launch application: {e}")
            return 1
        except FileNotFoundError:
            print("❌ Python interpreter not found!")
            return 1
            
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    print("👋 Golden Gibz Trading System closed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())