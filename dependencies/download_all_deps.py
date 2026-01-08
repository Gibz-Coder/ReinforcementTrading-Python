#!/usr/bin/env python3
"""
Download All Dependencies for Offline Installation
=================================================
Downloads all required packages to the dependencies folder for offline installation.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import shutil

class DependencyDownloader:
    def __init__(self):
        self.deps_dir = Path("dependencies")
        self.wheels_dir = self.deps_dir / "wheels"
        self.torch_dir = self.deps_dir / "torch"
        self.temp_dir = self.deps_dir / "temp"
        self.inventory_file = self.deps_dir / "package_inventory.json"
        
        # Create directories
        self.wheels_dir.mkdir(parents=True, exist_ok=True)
        self.torch_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def download_requirements(self, requirements_file):
        """Download all packages from requirements file"""
        print(f"📥 Downloading packages from {requirements_file}...")
        
        try:
            # Download to temp directory first
            result = subprocess.run([
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--dest", str(self.temp_dir),
                "-r", requirements_file
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Download failed: {result.stderr}")
                return False
            
            print(f"✅ Downloaded packages to temp directory")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            return False
    
    def download_torch_variants(self):
        """Download PyTorch CPU and GPU variants"""
        print("🔥 Downloading PyTorch variants...")
        
        torch_packages = [
            "torch>=1.13.0,<2.10.0",
            "torchvision>=0.14.0",
            "torchaudio>=0.13.0"
        ]
        
        # Download CPU version
        print("📥 Downloading PyTorch CPU version...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--dest", str(self.temp_dir),
                "--index-url", "https://download.pytorch.org/whl/cpu"
            ] + torch_packages, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ PyTorch CPU downloaded")
            else:
                print(f"⚠️ PyTorch CPU download warning: {result.stderr}")
        except Exception as e:
            print(f"⚠️ PyTorch CPU download error: {e}")
        
        return True
    
    def organize_downloads(self):
        """Organize downloaded files into appropriate directories"""
        print("📁 Organizing downloaded files...")
        
        torch_keywords = ["torch", "torchvision", "torchaudio"]
        moved_files = {"wheels": [], "torch": []}
        
        for file_path in self.temp_dir.glob("*.whl"):
            file_name = file_path.name.lower()
            
            # Determine target directory
            is_torch = any(keyword in file_name for keyword in torch_keywords)
            target_dir = self.torch_dir if is_torch else self.wheels_dir
            category = "torch" if is_torch else "wheels"
            
            # Move file
            target_path = target_dir / file_path.name
            
            # Don't overwrite newer files
            if target_path.exists():
                print(f"⚠️ Skipping existing file: {file_path.name}")
                continue
            
            shutil.move(str(file_path), str(target_path))
            
            # Record in inventory
            file_size_mb = target_path.stat().st_size / (1024 * 1024)
            moved_files[category].append({
                "name": file_path.name,
                "size_mb": file_size_mb
            })
            
            print(f"✅ Moved {file_path.name} to {category} ({file_size_mb:.2f} MB)")
        
        return moved_files
    
    def update_inventory(self, new_files):
        """Update package inventory"""
        print("📋 Updating package inventory...")
        
        # Load existing inventory
        inventory = {"wheels": [], "torch_packages": []}
        if self.inventory_file.exists():
            try:
                with open(self.inventory_file, 'r') as f:
                    inventory = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load existing inventory: {e}")
        
        # Add new files
        inventory["wheels"].extend(new_files["wheels"])
        inventory["torch_packages"].extend(new_files["torch"])
        
        # Remove duplicates
        inventory["wheels"] = list({f["name"]: f for f in inventory["wheels"]}.values())
        inventory["torch_packages"] = list({f["name"]: f for f in inventory["torch_packages"]}.values())
        
        # Calculate totals
        inventory["total_files"] = len(inventory["wheels"]) + len(inventory["torch_packages"])
        inventory["total_size_mb"] = (
            sum(f["size_mb"] for f in inventory["wheels"]) +
            sum(f["size_mb"] for f in inventory["torch_packages"])
        )
        
        # Save inventory
        with open(self.inventory_file, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"✅ Inventory updated: {inventory['total_files']} files, {inventory['total_size_mb']:.2f} MB total")
        return inventory
    
    def cleanup_temp(self):
        """Clean up temporary directory"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print("🧹 Cleaned up temporary files")
    
    def download_all(self):
        """Main download method"""
        print("🚀 Starting dependency download for offline installation")
        print("=" * 60)
        
        success = True
        
        # Download from requirements file
        requirements_file = self.deps_dir / "requirements_offline.txt"
        if requirements_file.exists():
            if not self.download_requirements(str(requirements_file)):
                success = False
        else:
            print(f"⚠️ Requirements file not found: {requirements_file}")
            success = False
        
        # Download PyTorch variants
        if not self.download_torch_variants():
            success = False
        
        if success:
            # Organize files
            moved_files = self.organize_downloads()
            
            # Update inventory
            inventory = self.update_inventory(moved_files)
            
            # Cleanup
            self.cleanup_temp()
            
            print("\n🎉 Download completed successfully!")
            print(f"📦 Total packages: {inventory['total_files']}")
            print(f"💾 Total size: {inventory['total_size_mb']:.2f} MB")
            print(f"📁 Wheels: {len(inventory['wheels'])}")
            print(f"🔥 PyTorch packages: {len(inventory['torch_packages'])}")
            
            return True
        else:
            print("\n❌ Download completed with errors")
            self.cleanup_temp()
            return False


def main():
    """Main entry point"""
    downloader = DependencyDownloader()
    
    try:
        print("Golden-Gibz Dependency Downloader")
        print("This will download all required packages for offline installation.")
        print("Make sure you have a good internet connection.\n")
        
        input("Press Enter to start download...")
        
        success = downloader.download_all()
        
        if success:
            print("\n✅ All dependencies downloaded successfully!")
            print("\nYou can now use offline installation:")
            print("  dependencies/smart_install.bat")
        else:
            print("\n❌ Download failed. Check your internet connection and try again.")
        
        input("\nPress Enter to exit...")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Download cancelled by user")
        downloader.cleanup_temp()
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        downloader.cleanup_temp()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()