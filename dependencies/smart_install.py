#!/usr/bin/env python3
"""
Smart Dependency Installation System
===================================
Checks dependencies folder first, downloads if missing, then installs offline.
"""

import os
import sys
import json
import subprocess
import urllib.request
import shutil
from pathlib import Path
import tempfile

class SmartInstaller:
    def __init__(self):
        self.deps_dir = Path("dependencies")
        self.wheels_dir = self.deps_dir / "wheels"
        self.torch_dir = self.deps_dir / "torch"
        self.inventory_file = self.deps_dir / "package_inventory.json"
        
        # Core requirements mapping
        self.requirements = {
            # Core scientific computing
            "numpy": "numpy>=1.21.0,<2.4.0",
            "pandas": "pandas>=1.5.0,<2.4.0",
            "matplotlib": "matplotlib>=3.5.0,<3.11.0",
            "scikit-learn": "scikit-learn>=1.0.0,<1.9.0",
            "scipy": "scipy>=1.14.0,<1.17.0",
            
            # Technical analysis
            "ta": "ta>=0.10.0",
            "pandas-ta": "pandas-ta>=0.3.14b",
            
            # Machine learning
            "torch": "torch>=1.13.0,<2.10.0",
            "stable-baselines3": "stable-baselines3>=2.0.0,<2.8.0",
            "gymnasium": "gymnasium>=0.28.0,<1.3.0",
            
            # Trading platform
            "MetaTrader5": "MetaTrader5>=5.0.37",
            
            # Configuration & utilities
            "PyYAML": "PyYAML>=6.0,<7.0",
            "tqdm": "tqdm>=4.62.0,<5.0.0",
            "colorama": "colorama>=0.4.4,<0.5.0",
            "psutil": "psutil>=5.8.0,<6.0.0",
            "requests": "requests>=2.25.0,<3.0.0",
            
            # Performance
            "numba": "numba>=0.56.0,<0.62.0",
            "joblib": "joblib>=1.1.0,<1.6.0",
            
            # Training tools
            "tensorboard": "tensorboard>=2.10.0,<2.21.0",
            "rich": "rich>=12.0.0,<14.0.0",
            
            # System utilities
            "setuptools": "setuptools>=60.0.0",
            "wheel": "wheel>=0.37.0",
            "pip": "pip>=21.0.0"
        }
        
        # Critical packages that must be installed first
        self.critical_packages = [
            "setuptools", "wheel", "pip", "numpy", "pandas"
        ]
        
        # Packages that can be installed from PyPI if offline fails
        self.fallback_packages = [
            "MetaTrader5", "ta", "pandas-ta"
        ]
    
    def check_virtual_env(self):
        """Check if we're in a virtual environment"""
        if not os.environ.get('VIRTUAL_ENV'):
            print("❌ ERROR: Please activate virtual environment first")
            print("Run: forex_env\\Scripts\\activate.bat")
            return False
        return True
    
    def load_inventory(self):
        """Load package inventory from JSON file"""
        try:
            if self.inventory_file.exists():
                with open(self.inventory_file, 'r') as f:
                    return json.load(f)
            return {"wheels": [], "torch_packages": []}
        except Exception as e:
            print(f"⚠️ Warning: Could not load inventory: {e}")
            return {"wheels": [], "torch_packages": []}
    
    def check_package_available(self, package_name, inventory):
        """Check if package is available in dependencies folder"""
        # Check in wheels
        for wheel in inventory.get("wheels", []):
            if package_name.lower().replace("-", "_") in wheel["name"].lower():
                wheel_path = self.wheels_dir / wheel["name"]
                if wheel_path.exists():
                    return str(wheel_path)
        
        # Check in torch packages
        for wheel in inventory.get("torch_packages", []):
            if package_name.lower().replace("-", "_") in wheel["name"].lower():
                wheel_path = self.torch_dir / wheel["name"]
                if wheel_path.exists():
                    return str(wheel_path)
        
        return None
    
    def download_package(self, package_name, requirement_spec):
        """Download package if not available locally"""
        print(f"📥 Downloading {package_name}...")
        
        try:
            # Create temporary directory for download
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download wheel
                result = subprocess.run([
                    sys.executable, "-m", "pip", "download",
                    "--only-binary=:all:",
                    "--dest", temp_dir,
                    requirement_spec
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Move downloaded files to appropriate directory
                    temp_path = Path(temp_dir)
                    downloaded_files = list(temp_path.glob("*.whl"))
                    
                    if downloaded_files:
                        # Determine target directory
                        if package_name.lower() in ["torch", "torchvision", "torchaudio"]:
                            target_dir = self.torch_dir
                        else:
                            target_dir = self.wheels_dir
                        
                        target_dir.mkdir(parents=True, exist_ok=True)
                        
                        for file in downloaded_files:
                            target_file = target_dir / file.name
                            shutil.move(str(file), str(target_file))
                            print(f"✅ Downloaded: {file.name}")
                        
                        return True
                    else:
                        print(f"⚠️ No wheel files found for {package_name}")
                        return False
                else:
                    print(f"❌ Download failed for {package_name}: {result.stderr}")
                    return False
        
        except Exception as e:
            print(f"❌ Error downloading {package_name}: {e}")
            return False
    
    def install_package_offline(self, package_path):
        """Install package from local wheel file"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                "--no-index", "--find-links", str(package_path.parent),
                "--force-reinstall", str(package_path)
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def install_package_online(self, requirement_spec):
        """Install package from PyPI as fallback"""
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install",
                requirement_spec
            ], capture_output=True, text=True)
            
            return result.returncode == 0
        except Exception:
            return False
    
    def install_all_packages(self):
        """Main installation method"""
        print("🚀 Starting Smart Dependency Installation")
        print("=" * 50)
        
        if not self.check_virtual_env():
            return False
        
        # Load inventory
        inventory = self.load_inventory()
        print(f"📦 Loaded inventory: {len(inventory.get('wheels', []))} wheels, {len(inventory.get('torch_packages', []))} torch packages")
        
        # Create directories
        self.wheels_dir.mkdir(parents=True, exist_ok=True)
        self.torch_dir.mkdir(parents=True, exist_ok=True)
        
        # Install critical packages first
        print("\n🔧 Installing critical packages...")
        for package in self.critical_packages:
            if package in self.requirements:
                self.install_single_package(package, self.requirements[package], inventory)
        
        # Install remaining packages
        print("\n📚 Installing remaining packages...")
        for package, requirement in self.requirements.items():
            if package not in self.critical_packages:
                self.install_single_package(package, requirement, inventory)
        
        # Verify installation
        print("\n✅ Verifying installation...")
        self.verify_installation()
        
        print("\n🎉 Smart installation completed!")
        return True
    
    def install_single_package(self, package_name, requirement_spec, inventory):
        """Install a single package with smart fallback"""
        print(f"\n📦 Processing {package_name}...")
        
        # Check if already installed and up to date
        try:
            result = subprocess.run([
                sys.executable, "-c", f"import {package_name.replace('-', '_')}"
            ], capture_output=True)
            if result.returncode == 0:
                print(f"✅ {package_name} already installed")
                return True
        except:
            pass
        
        # Check if available locally
        local_path = self.check_package_available(package_name, inventory)
        
        if local_path:
            print(f"📁 Found locally: {Path(local_path).name}")
            if self.install_package_offline(Path(local_path)):
                print(f"✅ Installed {package_name} from local cache")
                return True
            else:
                print(f"⚠️ Local installation failed for {package_name}")
        
        # Try to download and install
        print(f"🌐 Attempting to download {package_name}...")
        if self.download_package(package_name, requirement_spec):
            # Try to install from newly downloaded package
            local_path = self.check_package_available(package_name, self.load_inventory())
            if local_path and self.install_package_offline(Path(local_path)):
                print(f"✅ Downloaded and installed {package_name}")
                return True
        
        # Fallback to online installation
        if package_name in self.fallback_packages:
            print(f"🔄 Falling back to online installation for {package_name}...")
            if self.install_package_online(requirement_spec):
                print(f"✅ Installed {package_name} from PyPI")
                return True
        
        print(f"❌ Failed to install {package_name}")
        return False
    
    def verify_installation(self):
        """Verify that critical packages are working"""
        test_imports = [
            ("numpy", "import numpy; print(f'NumPy {numpy.__version__}')"),
            ("pandas", "import pandas; print(f'Pandas {pandas.__version__}')"),
            ("torch", "import torch; print(f'PyTorch {torch.__version__}')"),
            ("stable_baselines3", "import stable_baselines3; print(f'SB3 {stable_baselines3.__version__}')"),
            ("gymnasium", "import gymnasium; print(f'Gymnasium {gymnasium.__version__}')"),
            ("matplotlib", "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"),
        ]
        
        print("🔍 Testing core packages...")
        for package, test_code in test_imports:
            try:
                result = subprocess.run([
                    sys.executable, "-c", test_code
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"✅ {result.stdout.strip()}")
                else:
                    print(f"❌ {package} test failed")
            except Exception as e:
                print(f"❌ {package} test error: {e}")


def main():
    """Main entry point"""
    installer = SmartInstaller()
    
    try:
        success = installer.install_all_packages()
        if success:
            print("\n🎉 All dependencies installed successfully!")
            print("\nYou can now run:")
            print("  python golden_gibz_native_app.py")
            print("  python train_golden_gibz_model.py")
        else:
            print("\n❌ Installation completed with some errors")
            print("Check the output above for details")
        
        input("\nPress Enter to continue...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        input("Press Enter to continue...")


if __name__ == "__main__":
    main()