@echo off
echo ========================================
echo Downloading Dependencies for Offline Use
echo ========================================
echo.

:: Create directories
if not exist "dependencies\wheels" mkdir dependencies\wheels
if not exist "dependencies\torch" mkdir dependencies\torch

echo Downloading Python packages...

:: Download core packages
pip download -d dependencies\wheels ^
    numpy ^
    pandas ^
    matplotlib ^
    scikit-learn ^
    tqdm ^
    pyyaml ^
    joblib ^
    cloudpickle ^
    psutil

echo ✓ Core packages downloaded

:: Download RL packages
pip download -d dependencies\wheels ^
    stable-baselines3 ^
    gymnasium ^
    pandas-ta ^
    optuna ^
    tensorboard

echo ✓ RL packages downloaded

:: Download PyTorch CPU version
echo.
echo Downloading PyTorch (CPU version)...
pip download -d dependencies\torch ^
    torch ^
    torchvision ^
    torchaudio ^
    --index-url https://download.pytorch.org/whl/cpu

echo ✓ PyTorch downloaded

:: Create requirements file for offline installation
echo.
echo Creating offline requirements file...
pip freeze > dependencies\requirements_offline.txt

echo.
echo ========================================
echo DOWNLOAD COMPLETE!
echo ========================================
echo.
echo Dependencies saved to:
echo   dependencies\wheels\     - Python packages
echo   dependencies\torch\      - PyTorch packages
echo.
echo To install offline: dependencies\install_offline.bat
echo.