# Installation Guide

## System Requirements

### Hardware Requirements
- **CPU**: 4+ cores recommended (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB free space for full installation
- **GPU**: Optional but recommended (NVIDIA with CUDA support)

### Software Requirements
- **Operating System**: Windows 10/11, Linux (Ubuntu 18.04+), macOS 10.15+
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **Git**: For cloning repository (optional)

## Installation Methods

### Method 1: Automated Setup (Recommended)

```bash
# Windows - Run automated installer
setup.bat

# This will:
# 1. Create virtual environment
# 2. Install all dependencies
# 3. Download offline packages (3+ GB)
# 4. Verify installation
```

### Method 2: Manual Installation

1. **Create Virtual Environment**
```bash
python -m venv forex_env

# Activate environment
# Windows:
forex_env\Scripts\activate.bat
# Linux/macOS:
source forex_env/bin/activate
```

2. **Install Dependencies**
```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
python -c "import torch, stable_baselines3, pandas, numpy; print('✓ Installation successful')"
```

### Method 3: Offline Installation

For air-gapped systems without internet access:

1. **Download Complete Dependencies** (on internet-connected machine)
```bash
# Download all 95 packages (3+ GB)
scripts/download_complete_dependencies.bat

# Verify offline packages
python scripts/verify_offline_installation.py
```

2. **Transfer Project** to offline machine (entire folder)

3. **Install Offline**
```bash
# Complete installation (all packages)
dependencies/install_complete_offline.bat

# Or core packages only
dependencies/install_offline.bat

# Verify installation
python scripts/verify_installation.py
```

**Offline Package Details:**
- **Total Size**: 3.1 GB (95 packages)
- **PyTorch**: CPU + GPU versions included
- **Complete**: All development and production tools
- **Self-contained**: No internet required after setup

## GPU Support (Optional)

### NVIDIA GPU with CUDA

1. **Install CUDA Toolkit** (11.8 or 12.1)
   - Download from: https://developer.nvidia.com/cuda-downloads

2. **Install GPU-enabled PyTorch**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. **Verify GPU Support**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### Common Issues

**1. Python Version Error**
```
ERROR: Python 3.8+ required
```
**Solution**: Install Python 3.8 or higher from https://python.org

**2. Permission Denied (Linux/macOS)**
```
Permission denied: './setup.sh'
```
**Solution**: Make script executable: `chmod +x setup.sh`

**3. Package Installation Fails**
```
ERROR: Failed to install requirements
```
**Solutions**:
- Update pip: `python -m pip install --upgrade pip`
- Clear pip cache: `pip cache purge`
- Install with no cache: `pip install --no-cache-dir -r requirements.txt`

**4. CUDA/GPU Issues**
```
CUDA out of memory
```
**Solutions**:
- Reduce batch size in config
- Use CPU-only version: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

**5. Import Errors**
```
ModuleNotFoundError: No module named 'stable_baselines3'
```
**Solution**: Ensure virtual environment is activated and dependencies are installed

### Environment Variables

Set these if needed:

**Windows:**
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\forex-rl-trading
```

**Linux/macOS:**
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/forex-rl-trading
```

### Performance Optimization

**1. CPU Optimization**
```bash
# Set number of threads (adjust based on your CPU)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

**2. Memory Optimization**
- Reduce batch size if running out of memory
- Use gradient checkpointing for large models
- Monitor memory usage with `htop` (Linux) or Task Manager (Windows)

## Verification

After installation, verify everything works:

```bash
# Test basic functionality
python scripts/test_installation.py

# Run quick training test (5 minutes)
python scripts/train_enhanced.py --timesteps 1000 --model-name test_model

# Test model loading
python scripts/test_model.py --model models/experimental/test_model.zip
```

## Next Steps

1. **Prepare Data**: Place your CSV files in `data/raw/`
2. **Configure Training**: Edit `config/training_config.yaml`
3. **Start Training**: Run `python scripts/train_ultra_aggressive.py`
4. **Monitor Progress**: Use TensorBoard: `tensorboard --logdir logs/`

## Support

If you encounter issues:

1. Check [Troubleshooting Guide](troubleshooting.md)
2. Review system requirements
3. Ensure all dependencies are correctly installed
4. Check Python and package versions
5. Open an issue with detailed error messages