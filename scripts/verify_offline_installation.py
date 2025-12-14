#!/usr/bin/env python3
"""
Offline Installation Verification Script
=======================================

Verifies that all offline dependencies are available and complete.
"""

import os
import glob
from pathlib import Path

def check_offline_wheels():
    """Check offline wheel packages."""
    print("🔍 Checking offline wheel packages...")
    
    wheels_dir = Path("dependencies/wheels")
    if not wheels_dir.exists():
        print("❌ Wheels directory not found")
        return False
    
    wheel_files = list(wheels_dir.glob("*.whl"))
    print(f"✅ Found {len(wheel_files)} wheel packages")
    
    # Check for critical packages
    critical_packages = [
        "numpy", "pandas", "torch", "stable_baselines3", 
        "gymnasium", "matplotlib", "scikit_learn", "optuna",
        "tensorboard", "pandas_ta", "numba", "pyyaml"
    ]
    
    found_packages = []
    missing_packages = []
    
    for package in critical_packages:
        package_files = [f for f in wheel_files if package.replace("_", "-") in f.name.lower() or package.replace("-", "_") in f.name.lower()]
        if package_files:
            found_packages.append(package)
            print(f"  ✅ {package}: {len(package_files)} files")
        else:
            missing_packages.append(package)
            print(f"  ❌ {package}: Not found")
    
    if missing_packages:
        print(f"\n⚠️  Missing critical packages: {missing_packages}")
        return False
    
    return True

def check_pytorch_packages():
    """Check PyTorch packages."""
    print("\n🔥 Checking PyTorch packages...")
    
    torch_dir = Path("dependencies/torch")
    if not torch_dir.exists():
        print("❌ PyTorch directory not found")
        return False
    
    torch_files = list(torch_dir.glob("*.whl")) + list(torch_dir.glob("*.tar.gz"))
    print(f"✅ Found {len(torch_files)} PyTorch-related files")
    
    # Check for CPU and GPU versions
    cpu_torch = [f for f in torch_files if "cpu" in f.name.lower()]
    gpu_torch = [f for f in torch_files if "cu118" in f.name.lower() or "cu121" in f.name.lower()]
    
    print(f"  ✅ CPU PyTorch: {len(cpu_torch)} files")
    print(f"  ✅ GPU PyTorch: {len(gpu_torch)} files")
    
    # Check for torch, torchvision, torchaudio
    torch_core = [f for f in torch_files if f.name.startswith("torch-")]
    torchvision = [f for f in torch_files if f.name.startswith("torchvision-")]
    torchaudio = [f for f in torch_files if f.name.startswith("torchaudio-")]
    
    print(f"  ✅ torch: {len(torch_core)} versions")
    print(f"  ✅ torchvision: {len(torchvision)} versions")
    print(f"  ✅ torchaudio: {len(torchaudio)} versions")
    
    return len(torch_core) > 0 and len(torchvision) > 0 and len(torchaudio) > 0

def calculate_total_size():
    """Calculate total size of offline packages."""
    print("\n📊 Calculating package sizes...")
    
    total_size = 0
    
    # Wheels directory
    wheels_dir = Path("dependencies/wheels")
    if wheels_dir.exists():
        wheels_size = sum(f.stat().st_size for f in wheels_dir.glob("*.whl"))
        total_size += wheels_size
        print(f"  Wheels: {wheels_size / (1024*1024):.1f} MB")
    
    # PyTorch directory
    torch_dir = Path("dependencies/torch")
    if torch_dir.exists():
        torch_size = sum(f.stat().st_size for f in torch_dir.glob("*"))
        total_size += torch_size
        print(f"  PyTorch: {torch_size / (1024*1024):.1f} MB")
    
    print(f"  Total: {total_size / (1024*1024):.1f} MB ({total_size / (1024*1024*1024):.2f} GB)")
    
    return total_size

def check_requirements_files():
    """Check requirements files."""
    print("\n📋 Checking requirements files...")
    
    req_files = [
        "requirements.txt",
        "dependencies/requirements_complete.txt",
        "dependencies/requirements_offline.txt"
    ]
    
    for req_file in req_files:
        if os.path.exists(req_file):
            with open(req_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"  ✅ {req_file}: {len(lines)} packages")
        else:
            print(f"  ⚠️  {req_file}: Not found")

def check_installation_scripts():
    """Check installation scripts."""
    print("\n🛠️ Checking installation scripts...")
    
    scripts = [
        "setup.bat",
        "dependencies/install_offline.bat",
        "dependencies/install_complete_offline.bat",
        "scripts/download_dependencies.bat",
        "scripts/download_complete_dependencies.bat"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}: Missing")

def generate_package_inventory():
    """Generate detailed package inventory."""
    print("\n📦 Generating package inventory...")
    
    inventory = {
        'wheels': [],
        'torch_packages': [],
        'total_files': 0,
        'total_size_mb': 0
    }
    
    # Wheels inventory
    wheels_dir = Path("dependencies/wheels")
    if wheels_dir.exists():
        for wheel in wheels_dir.glob("*.whl"):
            inventory['wheels'].append({
                'name': wheel.name,
                'size_mb': wheel.stat().st_size / (1024*1024)
            })
    
    # PyTorch inventory
    torch_dir = Path("dependencies/torch")
    if torch_dir.exists():
        for torch_file in torch_dir.glob("*"):
            inventory['torch_packages'].append({
                'name': torch_file.name,
                'size_mb': torch_file.stat().st_size / (1024*1024)
            })
    
    inventory['total_files'] = len(inventory['wheels']) + len(inventory['torch_packages'])
    inventory['total_size_mb'] = (
        sum(p['size_mb'] for p in inventory['wheels']) +
        sum(p['size_mb'] for p in inventory['torch_packages'])
    )
    
    # Save inventory
    import json
    with open("dependencies/package_inventory.json", 'w') as f:
        json.dump(inventory, f, indent=2)
    
    print(f"  ✅ Inventory saved: dependencies/package_inventory.json")
    print(f"  📊 Total files: {inventory['total_files']}")
    print(f"  📊 Total size: {inventory['total_size_mb']:.1f} MB")
    
    return inventory

def main():
    """Run offline installation verification."""
    
    print("🔍 OFFLINE INSTALLATION VERIFICATION")
    print("=" * 50)
    
    checks = []
    
    # Check offline wheels
    checks.append(check_offline_wheels())
    
    # Check PyTorch packages
    checks.append(check_pytorch_packages())
    
    # Calculate sizes
    total_size = calculate_total_size()
    
    # Check requirements files
    check_requirements_files()
    
    # Check installation scripts
    check_installation_scripts()
    
    # Generate inventory
    inventory = generate_package_inventory()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print("🎉 OFFLINE INSTALLATION READY!")
        print("\n✅ All critical packages available")
        print(f"✅ {inventory['total_files']} packages downloaded")
        print(f"✅ {inventory['total_size_mb']:.1f} MB total size")
        print("\n🚀 You can now install offline with:")
        print("   dependencies\\install_complete_offline.bat")
        
    else:
        print(f"⚠️  INCOMPLETE ({passed}/{total} checks passed)")
        print("\n🔧 Run the complete download:")
        print("   scripts\\download_complete_dependencies.bat")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if inventory['total_size_mb'] > 5000:  # > 5GB
        print("   - Large download size, ensure sufficient storage")
    if inventory['total_files'] > 100:
        print("   - Many packages, installation may take time")
    
    print("   - Test offline installation in clean environment")
    print("   - Keep backup of dependencies folder")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)