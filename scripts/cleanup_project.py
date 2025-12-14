#!/usr/bin/env python3
"""
Project Cleanup Script
=====================

Organizes files, removes duplicates, and ensures proper structure.
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_project():
    """Clean up and organize the project structure."""
    
    print("🧹 Starting project cleanup...")
    
    # Remove Python cache files
    print("Removing Python cache files...")
    for cache_dir in glob.glob("**/__pycache__", recursive=True):
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"  Removed: {cache_dir}")
    
    # Remove .pyc files
    for pyc_file in glob.glob("**/*.pyc", recursive=True):
        os.remove(pyc_file)
        print(f"  Removed: {pyc_file}")
    
    # Clean up temporary files
    temp_files = [
        "enhanced_data_sample.csv",
        "project_structure.md",
        "*.tmp",
        "*.log"
    ]
    
    for pattern in temp_files:
        for file in glob.glob(pattern):
            if os.path.exists(file):
                os.remove(file)
                print(f"  Removed: {file}")
    
    # Ensure all directories exist
    directories = [
        "src/environments",
        "src/indicators", 
        "src/training",
        "src/rewards",
        "src/risk",
        "src/testing",
        "src/utils",
        "dependencies/wheels",
        "dependencies/torch",
        "models/production",
        "models/experimental",
        "results/reports",
        "results/analysis",
        "scripts",
        "config",
        "docs",
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
forex_env/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project Specific
logs/
models/experimental/
results/
data/processed/
dependencies/wheels/
dependencies/torch/

# Temporary files
*.tmp
*.log
enhanced_data_sample.csv
"""
    
    if not os.path.exists(".gitignore"):
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")
    
    # Create README files for empty directories
    readme_dirs = [
        ("data/raw", "Place your CSV market data files here"),
        ("data/processed", "Preprocessed data files are stored here"),
        ("models/production", "Production-ready trained models"),
        ("models/experimental", "Experimental and development models"),
        ("results/reports", "Generated performance reports"),
        ("results/analysis", "Analysis files and charts"),
        ("logs", "Training logs and TensorBoard files")
    ]
    
    for dir_path, description in readme_dirs:
        readme_path = os.path.join(dir_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# {os.path.basename(dir_path).title()}\n\n{description}\n")
    
    print("✅ Project cleanup completed!")
    print("\n📁 Current project structure:")
    
    # Display project structure
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        items = sorted(os.listdir(directory))
        dirs = [item for item in items if os.path.isdir(os.path.join(directory, item)) and not item.startswith('.')]
        files = [item for item in items if os.path.isfile(os.path.join(directory, item)) and not item.startswith('.')]
        
        # Print directories first
        for i, item in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last_dir else '├── '}{item}/")
            
            new_prefix = prefix + ("    " if is_last_dir else "│   ")
            print_tree(os.path.join(directory, item), new_prefix, max_depth, current_depth + 1)
        
        # Print files
        for i, item in enumerate(files):
            is_last = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
    
    print("forex-rl-trading/")
    print_tree(".", max_depth=2)

if __name__ == "__main__":
    cleanup_project()