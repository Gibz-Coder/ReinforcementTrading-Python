@echo off
echo ========================================
echo Forex RL Trading System - Windows Setup
echo ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.8+ required
    pause
    exit /b 1
)

echo ✓ Python version OK

:: Create virtual environment
echo.
echo Creating virtual environment...
python -m venv forex_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo ✓ Virtual environment created

:: Activate virtual environment
echo.
echo Activating virtual environment...
call forex_env\Scripts\activate.bat

:: Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

:: Check if offline installation is available
if exist "dependencies\wheels" (
    echo.
    echo ========================================
    echo OFFLINE INSTALLATION DETECTED
    echo ========================================
    echo Installing from local wheels...
    call dependencies\install_offline.bat
) else (
    echo.
    echo ========================================
    echo ONLINE INSTALLATION
    echo ========================================
    echo Installing from PyPI...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install requirements
        pause
        exit /b 1
    )
)

echo ✓ Dependencies installed

:: Download offline dependencies for future use
echo.
echo Downloading dependencies for offline use...
call scripts\download_dependencies.bat

:: Verify installation
echo.
echo Verifying installation...
python -c "import torch, stable_baselines3, pandas, numpy; print('✓ All core packages imported successfully')"
if errorlevel 1 (
    echo ERROR: Installation verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo To activate the environment in future sessions:
echo   forex_env\Scripts\activate.bat
echo.
echo To start training:
echo   python scripts\train_ultra_aggressive.py
echo.
echo To test a model:
echo   python scripts\test_model.py --model models\production\best_model.zip
echo.
echo For help: python scripts\help.py
echo.
pause