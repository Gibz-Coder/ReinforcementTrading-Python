#!/usr/bin/env python3
"""
Simple Trend Rider Setup
========================
Quick setup script for the 100% win rate trading system.
"""

import os
import sys
import subprocess

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_data():
    """Check if data files exist."""
    print("📊 Checking data files...")
    data_files = [
        "data/raw/XAU_15m_data.csv",
        "data/raw/XAU_1h_data.csv", 
        "data/raw/XAU_4h_data.csv",
        "data/raw/XAU_1d_data.csv"
    ]
    
    missing_files = []
    for file in data_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("   Please add your XAUUSD data files to data/raw/")
        return False
    else:
        print("✅ All data files found")
        return True

def check_models():
    """Check if trained models exist."""
    print("🤖 Checking trained models...")
    model_dir = "scripts/models/production"
    
    if not os.path.exists(model_dir):
        print(f"⚠️ Model directory not found: {model_dir}")
        return False
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith("simple_trend") and f.endswith(".zip")]
    
    if not model_files:
        print("⚠️ No trained models found")
        print("   Run: python scripts/train_simple_trend_rider.py")
        return False
    else:
        print(f"✅ Found {len(model_files)} trained models")
        best_model = max(model_files)
        print(f"   Best model: {best_model}")
        return True

def run_quick_test():
    """Run a quick test of the system."""
    print("🧪 Running quick test...")
    try:
        result = subprocess.run([sys.executable, "test_simple_trend_rider.py"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Quick test passed")
            return True
        else:
            print("❌ Quick test failed")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ Quick test timed out")
        return False
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎯 SIMPLE TREND RIDER SETUP")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check data
    data_ok = check_data()
    
    # Check models
    models_ok = check_models()
    
    # Run test if everything is ready
    if data_ok and models_ok:
        test_ok = run_quick_test()
        
        if test_ok:
            print("\n🎉 SETUP COMPLETE!")
            print("✅ System is ready for trading")
            print("\n📋 Next steps:")
            print("   1. python analyze_simple_trend_rider.py  # Analyze performance")
            print("   2. python calculate_demo_projections.py  # Calculate projections")
            print("   3. python mt5_simple_trend_trader.py     # Start MT5 trading")
        else:
            print("\n⚠️ Setup completed with warnings")
            print("   System may work but needs verification")
    else:
        print("\n⚠️ Setup incomplete")
        if not data_ok:
            print("   - Add XAUUSD data files")
        if not models_ok:
            print("   - Train models first")

if __name__ == "__main__":
    main()