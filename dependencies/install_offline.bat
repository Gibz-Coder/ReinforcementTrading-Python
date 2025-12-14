@echo off
echo ========================================
echo Offline Installation System
echo ========================================
echo.

:: Check if we're in a virtual environment
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Please activate virtual environment first
    echo Run: forex_env\Scripts\activate.bat
    pause
    exit /b 1
)

echo Installing from local wheels...
echo.

:: Install core dependencies first
echo Installing core packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    numpy ^
    pandas ^
    matplotlib ^
    scikit-learn ^
    tqdm

if errorlevel 1 (
    echo ERROR: Failed to install core packages
    pause
    exit /b 1
)

echo ✓ Core packages installed

:: Install PyTorch (check for GPU support)
echo.
echo Installing PyTorch...
if exist "torch\torch-*-win_amd64.whl" (
    echo Installing PyTorch CPU version...
    pip install --no-index --find-links torch\ --force-reinstall torch torchvision torchaudio
) else (
    echo PyTorch wheels not found, installing from PyPI...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

echo ✓ PyTorch installed

:: Install RL and trading packages
echo.
echo Installing RL and trading packages...
pip install --no-index --find-links wheels\ --force-reinstall ^
    stable-baselines3 ^
    gymnasium ^
    pandas-ta ^
    optuna ^
    tensorboard

if errorlevel 1 (
    echo ERROR: Failed to install RL packages
    pause
    exit /b 1
)

echo ✓ RL packages installed

:: Install remaining dependencies
echo.
echo Installing remaining dependencies...
pip install --no-index --find-links wheels\ --force-reinstall ^
    pyyaml ^
    joblib ^
    cloudpickle ^
    psutil

echo.
echo ========================================
echo OFFLINE INSTALLATION COMPLETE!
echo ========================================
echo.
echo Installed packages:
pip list | findstr -i "torch stable-baselines3 gymnasium pandas numpy"
echo.