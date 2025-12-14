# Troubleshooting Guide

## Common Installation Issues

### Python Version Issues

**Problem**: `ERROR: Python 3.8+ required`
**Solution**:
1. Install Python 3.8+ from https://python.org
2. Ensure Python is in your PATH
3. Use `python --version` to verify

**Problem**: `python: command not found`
**Solution**:
- **Windows**: Add Python to PATH during installation
- **Linux**: Install with `sudo apt install python3 python3-pip`
- **macOS**: Install with Homebrew: `brew install python`

### Virtual Environment Issues

**Problem**: `Failed to create virtual environment`
**Solution**:
```bash
# Install venv module
python -m pip install --user virtualenv

# Alternative: use virtualenv
pip install virtualenv
virtualenv forex_env
```

**Problem**: Virtual environment not activating
**Solution**:
```bash
# Windows
forex_env\Scripts\activate.bat

# Linux/macOS
source forex_env/bin/activate

# Verify activation
which python  # Should show path to forex_env
```

### Package Installation Issues

**Problem**: `pip install` fails with permission errors
**Solution**:
```bash
# Use --user flag
pip install --user -r requirements.txt

# Or use virtual environment (recommended)
python -m venv forex_env
source forex_env/bin/activate  # Linux/macOS
# or forex_env\Scripts\activate.bat  # Windows
pip install -r requirements.txt
```

**Problem**: `Failed to install torch`
**Solution**:
```bash
# Install CPU version first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: `No module named 'stable_baselines3'`
**Solution**:
```bash
# Ensure virtual environment is activated
pip install stable-baselines3[extra]

# If still fails, try development version
pip install git+https://github.com/DLR-RM/stable-baselines3
```

## Training Issues

### Memory Issues

**Problem**: `CUDA out of memory` or `RuntimeError: out of memory`
**Solutions**:
1. **Reduce batch size**:
   ```yaml
   # In config/training_config.yaml
   model:
     batch_size: 64  # Reduce from 128
     n_steps: 2048   # Reduce from 4096
   ```

2. **Use CPU instead of GPU**:
   ```bash
   export CUDA_VISIBLE_DEVICES=""  # Linux/macOS
   set CUDA_VISIBLE_DEVICES=       # Windows
   ```

3. **Reduce model size**:
   ```yaml
   model:
     policy_kwargs:
       net_arch:
         pi: [256, 128]  # Smaller network
         vf: [256, 128]
   ```

**Problem**: `System runs out of RAM`
**Solutions**:
- Close other applications
- Reduce `n_envs` to 1
- Use smaller datasets for initial testing
- Add swap space (Linux)

### Data Issues

**Problem**: `FileNotFoundError: data file not found`
**Solution**:
```bash
# Check file exists
ls data/raw/your_file.csv

# Use absolute path
python scripts/train_model.py --data /full/path/to/data.csv

# Check file permissions
chmod 644 data/raw/*.csv  # Linux/macOS
```

**Problem**: `Invalid CSV format` or `Date parsing errors`
**Solution**:
1. **Check CSV format**:
   ```csv
   Gmt time,Open,High,Low,Close,Volume
   01.07.2020 00:00:00.000,1.12336,1.12336,1.12275,1.12306,4148.0298
   ```

2. **Fix date format**:
   ```python
   import pandas as pd
   df = pd.read_csv('your_file.csv')
   df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')
   df.to_csv('fixed_file.csv', index=False)
   ```

**Problem**: `Data contains NaN values`
**Solution**:
```python
# Check for missing data
import pandas as pd
df = pd.read_csv('data/raw/your_file.csv')
print(df.isnull().sum())

# Remove or fill missing values
df = df.dropna()  # Remove rows with NaN
# or
df = df.fillna(method='ffill')  # Forward fill
```

### Training Performance Issues

**Problem**: Training is very slow
**Solutions**:
1. **Use GPU acceleration**:
   ```bash
   # Check GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   
   # Install GPU version of PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize hyperparameters**:
   ```yaml
   model:
     n_steps: 2048      # Smaller steps
     batch_size: 64     # Smaller batches
     n_epochs: 5        # Fewer epochs
   ```

3. **Use fewer environments**:
   ```yaml
   training:
     n_envs: 1  # Single environment
   ```

**Problem**: Model not learning (reward stays flat)
**Solutions**:
1. **Check reward function**:
   - Ensure rewards are not too sparse
   - Verify reward scaling is appropriate
   - Add intermediate rewards

2. **Adjust learning rate**:
   ```yaml
   model:
     learning_rate: 0.0001  # Lower learning rate
   ```

3. **Increase exploration**:
   ```yaml
   model:
     ent_coef: 0.1  # Higher entropy coefficient
   ```

**Problem**: Model overfitting (good training, poor testing)
**Solutions**:
1. **Use more training data**
2. **Enable walk-forward validation**:
   ```bash
   python scripts/train_enhanced.py --use-walk-forward
   ```
3. **Reduce model complexity**
4. **Add regularization**

## Testing Issues

### Model Loading Issues

**Problem**: `FileNotFoundError: model not found`
**Solution**:
```bash
# Check model exists
ls models/production/*.zip

# Use full path
python scripts/test_model.py --model /full/path/to/model.zip
```

**Problem**: `Model loading fails with version mismatch`
**Solution**:
```bash
# Check stable-baselines3 version
pip show stable-baselines3

# Reinstall compatible version
pip install stable-baselines3==2.0.0
```

### Performance Issues

**Problem**: Poor backtest results
**Diagnosis**:
1. **Check data quality**:
   ```python
   import pandas as pd
   df = pd.read_csv('test_data.csv')
   print(f"Data range: {df['Gmt time'].min()} to {df['Gmt time'].max()}")
   print(f"Missing values: {df.isnull().sum().sum()}")
   ```

2. **Verify model performance**:
   ```bash
   python scripts/test_model.py --model your_model.zip --data different_data.csv
   ```

3. **Check for overfitting**:
   - Test on multiple time periods
   - Use walk-forward validation
   - Compare in-sample vs out-of-sample results

## Environment Issues

### Import Errors

**Problem**: `ModuleNotFoundError` for project modules
**Solution**:
```bash
# Add project to Python path
export PYTHONPATH=$PYTHONPATH:/path/to/forex-rl-trading  # Linux/macOS
set PYTHONPATH=%PYTHONPATH%;C:\path\to\forex-rl-trading  # Windows

# Or install in development mode
pip install -e .
```

**Problem**: `ImportError: cannot import name 'xxx'`
**Solution**:
1. Check file exists in correct location
2. Verify `__init__.py` files exist
3. Check for circular imports
4. Restart Python interpreter

### Configuration Issues

**Problem**: `YAML configuration not loading`
**Solution**:
```bash
# Install PyYAML
pip install pyyaml

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('config/training_config.yaml'))"
```

**Problem**: Configuration values not taking effect
**Solution**:
1. Check file path is correct
2. Verify YAML syntax (indentation matters)
3. Ensure no typos in parameter names
4. Check parameter precedence (command line > config file)

## System-Specific Issues

### Windows Issues

**Problem**: `'python' is not recognized`
**Solution**:
1. Add Python to PATH during installation
2. Use `py` instead of `python`
3. Reinstall Python with "Add to PATH" checked

**Problem**: Long path issues
**Solution**:
1. Enable long paths in Windows
2. Use shorter directory names
3. Move project closer to root (e.g., `C:\forex-rl\`)

### Linux/macOS Issues

**Problem**: Permission denied errors
**Solution**:
```bash
# Make scripts executable
chmod +x setup.sh
chmod +x scripts/*.py

# Fix ownership
sudo chown -R $USER:$USER /path/to/project
```

**Problem**: Missing system dependencies
**Solution**:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3-dev python3-pip build-essential

# CentOS/RHEL
sudo yum install python3-devel python3-pip gcc

# macOS
xcode-select --install
brew install python
```

## Performance Optimization

### Speed Up Training

1. **Use GPU**: Install CUDA and GPU-enabled PyTorch
2. **Optimize batch size**: Find optimal balance between speed and memory
3. **Reduce model complexity**: Smaller networks train faster
4. **Use fewer environments**: Start with `n_envs=1`
5. **Profile code**: Use `cProfile` to find bottlenecks

### Reduce Memory Usage

1. **Smaller batch sizes**: Reduce `batch_size` and `n_steps`
2. **Gradient checkpointing**: Enable in model configuration
3. **Data streaming**: Process data in chunks
4. **Close unused applications**: Free up system memory

## Getting Help

### Diagnostic Information

When reporting issues, include:

```bash
# System information
python --version
pip list | grep -E "(torch|stable-baselines3|gymnasium|pandas)"

# Error messages (full traceback)
python scripts/train_model.py 2>&1 | tee error.log

# System resources
# Linux/macOS:
free -h
df -h
# Windows:
systeminfo | findstr "Total Physical Memory"
```

### Log Analysis

Check log files for clues:
```bash
# Training logs
tail -f logs/latest_training/progress.csv

# TensorBoard logs
tensorboard --logdir logs/

# System logs
# Linux: /var/log/syslog
# macOS: Console.app
# Windows: Event Viewer
```

### Community Support

1. **Check existing issues**: Search for similar problems
2. **Provide details**: Include error messages, system info, and steps to reproduce
3. **Minimal example**: Create a simple test case that reproduces the issue
4. **Be patient**: Complex issues may take time to diagnose

### Professional Support

For production deployments or complex issues:
1. Consider professional consulting
2. Review code with experienced developers
3. Implement proper testing and monitoring
4. Use version control and backup systems